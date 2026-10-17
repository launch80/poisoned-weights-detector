"""Tokenizer files: byte identity vs reference, structural diffs, suspicious token content."""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any

from ...artifacts.index import declared_modifications
from ...context import ScanContext
from ...registry import check
from ...schema import Evidence, Finding

TOK_FILES = (
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.model",
)
ZERO_WIDTH = {"​", "‌", "‍", "⁠", "﻿", "᠎"}
BIDI = {chr(c) for c in range(0x202A, 0x202F)} | {chr(c) for c in range(0x2066, 0x206A)}
URLISH = re.compile(r"https?://|www\.|\.(com|net|org|io|ru|cn)\b", re.I)


def suspicious_text(s: str) -> list[str]:
    why = []
    if any(ch in ZERO_WIDTH for ch in s):
        why.append("zero-width character")
    if any(ch in BIDI for ch in s):
        why.append("bidi control character")
    if any(unicodedata.category(ch) == "Cc" and ch not in "\n\t\r" for ch in s):
        why.append("control character")
    if URLISH.search(s):
        why.append("URL-like")
    if len(s) > 64:
        why.append(f"very long ({len(s)} chars)")
    return why


def _added_tokens(tok: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {
        int(t["id"]): t for t in tok.get("added_tokens", []) if isinstance(t, dict) and "id" in t
    }


def _summ(d: Any) -> str:
    return json.dumps(d, sort_keys=True, ensure_ascii=False)[:200]


def _norm_component(comp: str, obj: Any) -> Any:
    """Drop ByteLevel flags that do not change token ids: `trim_offsets` only affects offset mappings,
    and the ByteLevel *decoder*/*post_processor* ignore `add_prefix_space`/`use_regex` entirely
    (transformers >= 5 re-saves flip them). Pre-tokenizer flags are kept: they do change tokenisation."""
    if isinstance(obj, dict):
        d = {k: _norm_component(comp, v) for k, v in obj.items()}
        if d.get("type") == "ByteLevel":
            d.pop("trim_offsets", None)
            if comp in ("decoder", "post_processor"):
                d.pop("add_prefix_space", None)
                d.pop("use_regex", None)
        return d
    if isinstance(obj, list):
        return [_norm_component(comp, v) for v in obj]
    return obj


# keys transformers >= 5 adds, drops or moves when it re-saves a tokenizer; informational when the
# authoritative tokenizer.json content is unchanged
MIGRATION_KEYS = {
    "backend",
    "is_local",
    "local_files_only",
    "extra_special_tokens",
    "add_bos_token",
    "add_eos_token",
    "additional_special_tokens",
    "added_tokens_decoder",
    "clean_up_tokenization_spaces",
    "split_special_tokens",
    "model_max_length",
    "padding_side",
    "tokenizer_class",
}


@check(
    "static.tokenizer_integrity",
    "static",
    requires=["artifact:config"],
    description="Tokenizer identity, added tokens, special token content",
    tags=["tokenizer"],
)
def tokenizer_integrity(ctx: ScanContext) -> list[Finding]:
    a = ctx.artifact
    assert a is not None
    out: list[Finding] = []
    declared = bool(declared_modifications(a))
    tok = a.json("tokenizer.json") if a.has("tokenizer.json") else None
    tcfg = a.tokenizer_config

    # absolute: suspicious token strings ---------------------------------------
    sus: list[Evidence] = []
    if isinstance(tok, dict):
        for tid, t in _added_tokens(tok).items():
            why = suspicious_text(str(t.get("content", "")))
            if why:
                sus.append(
                    Evidence(
                        kind="config_key",
                        locator=f"tokenizer.json#added_tokens[{tid}]",
                        summary=f"{t.get('content')!r}: {', '.join(why)}",
                    )
                )
    for tid, t in (tcfg.get("added_tokens_decoder") or {}).items():
        if isinstance(t, dict):
            why = suspicious_text(str(t.get("content", "")))
            if why:
                sus.append(
                    Evidence(
                        kind="config_key",
                        locator=f"tokenizer_config.json#added_tokens_decoder[{tid}]",
                        summary=f"{t.get('content')!r}: {', '.join(why)}",
                    )
                )
    if sus:
        out.append(
            Finding(
                check_id="static.tokenizer_integrity",
                title=f"{len(sus)} added token(s) with suspicious content",
                status="warn",
                severity="medium",
                confidence=0.6,
                evidence=sus[:40],
                tags=["tokenizer", "injection"],
            )
        )

    # vocab size consistency ---------------------------------------------------
    cfg_vocab = a.text_config.get("vocab_size") or a.config.get("vocab_size")
    emb_rows = None
    for n in a.find_tensors(r"embed_tokens\.weight$"):
        emb_rows = a.tensors[n][1].shape[0]
        break
    tok_vocab = None
    if isinstance(tok, dict):
        model_vocab = (tok.get("model") or {}).get("vocab")
        if isinstance(model_vocab, dict):
            tok_vocab = len(model_vocab) + len(
                [t for t in _added_tokens(tok).values() if t.get("content") not in model_vocab]
            )
    if cfg_vocab and emb_rows and emb_rows < cfg_vocab:
        out.append(
            Finding(
                check_id="static.tokenizer_integrity",
                title="embedding has fewer rows than config vocab_size",
                status="fail",
                severity="medium",
                confidence=0.8,
                evidence=[
                    Evidence(
                        kind="metric",
                        locator="vocab",
                        summary=f"config {cfg_vocab}, embedding rows {emb_rows}, tokenizer {tok_vocab}",
                    )
                ],
                tags=["integrity"],
            )
        )
    out.append(
        Finding.passed(
            "static.tokenizer_integrity",
            f"vocab: config {cfg_vocab}, tokenizer {tok_vocab}, embedding rows {emb_rows}",
            evidence=[
                Evidence(
                    kind="metric",
                    locator="vocab",
                    summary="",
                    data={"config": cfg_vocab, "tokenizer": tok_vocab, "embedding_rows": emb_rows},
                )
            ],
        )
    )

    # vs reference ---------------------------------------------------------------
    near = ctx.nearest_reference()
    ref_art = ctx.artifact_for(near[1]) if near else None
    if ref_art is None:
        return out
    hop = near[0]
    identical, differing, missing = [], [], []
    for fn in TOK_FILES:
        if a.has(fn) and ref_art.has(fn):
            (identical if a.files[fn].sha256() == ref_art.files[fn].sha256() else differing).append(
                fn
            )
        elif a.has(fn) != ref_art.has(fn):
            missing.append(fn)
    ev = [
        Evidence(
            kind="file",
            locator=fn,
            summary="sha256 identical to reference",
            reference_locator=f"{near[1].label()}:{fn}",
        )
        for fn in identical
    ]
    if identical:
        out.append(
            Finding.passed(
                "static.tokenizer_integrity",
                f"{len(identical)} tokenizer file(s) byte-identical to {hop}",
                evidence=ev,
            )
        )
    if missing:
        out.append(
            Finding(
                check_id="static.tokenizer_integrity",
                title=f"tokenizer file set differs from {hop}: {missing}",
                status="warn",
                severity="low",
                confidence=0.5,
                evidence=[
                    Evidence(kind="file", locator=m, summary="present on one side only")
                    for m in missing
                ],
                tags=["tokenizer"],
            )
        )
    if "tokenizer.json" in differing and isinstance(tok, dict):
        rtok = ref_art.json("tokenizer.json") or {}
        evs: list[Evidence] = []
        at, rt = _added_tokens(tok), _added_tokens(rtok)
        for tid in sorted(set(at) - set(rt)):
            evs.append(
                Evidence(
                    kind="config_key",
                    locator=f"tokenizer.json#added_tokens[{tid}]",
                    summary=f"new token {at[tid].get('content')!r} "
                    + _summ({k: v for k, v in at[tid].items() if k not in ("id", "content")}),
                )
            )
        for tid in sorted(set(rt) - set(at)):
            evs.append(
                Evidence(
                    kind="config_key",
                    locator=f"tokenizer.json#added_tokens[{tid}]",
                    summary=f"removed token {rt[tid].get('content')!r}",
                )
            )
        for tid in sorted(set(at) & set(rt)):
            if at[tid] != rt[tid]:
                evs.append(
                    Evidence(
                        kind="config_key",
                        locator=f"tokenizer.json#added_tokens[{tid}]",
                        summary=f"changed: {_summ(rt[tid])} -> {_summ(at[tid])}",
                    )
                )
        for comp in ("normalizer", "pre_tokenizer", "post_processor", "decoder"):
            if _norm_component(comp, tok.get(comp)) != _norm_component(comp, rtok.get(comp)):
                evs.append(
                    Evidence(
                        kind="config_key",
                        locator=f"tokenizer.json#{comp}",
                        summary=f"{_summ(rtok.get(comp))} -> {_summ(tok.get(comp))}",
                    )
                )
        mv, rmv = (tok.get("model") or {}), (rtok.get("model") or {})
        if mv.get("type") != rmv.get("type") or len(mv.get("vocab") or {}) != len(
            rmv.get("vocab") or {}
        ):
            evs.append(
                Evidence(
                    kind="config_key",
                    locator="tokenizer.json#model",
                    summary=f"type/vocab size {rmv.get('type')}/{len(rmv.get('vocab') or {})} -> "
                    f"{mv.get('type')}/{len(mv.get('vocab') or {})}",
                )
            )
        elif isinstance(mv.get("vocab"), dict) and mv.get("vocab") != rmv.get("vocab"):
            changed = [k for k in mv["vocab"] if rmv["vocab"].get(k) != mv["vocab"][k]][:20]
            evs.append(
                Evidence(
                    kind="config_key",
                    locator="tokenizer.json#model.vocab",
                    summary=f"vocab entries changed, e.g. {changed}",
                )
            )
        sev = (
            "high"
            if any(
                "new token" in e.summary
                or "post_processor" in e.locator
                or "normalizer" in e.locator
                for e in evs
            )
            else "medium"
        )
        if not evs:  # re-serialised: same tokens, same pipeline semantics, different bytes
            sev = "low"
        out.append(
            Finding(
                check_id="static.tokenizer_integrity",
                title=f"tokenizer.json differs from {hop} ({len(evs)} structural change(s))"
                if evs
                else f"tokenizer.json re-serialised vs {hop}: bytes differ, token ids and pipeline unchanged",
                status="warn",
                severity=sev,
                confidence=0.75 if evs else 0.4,
                declared=declared,
                hop="suspect",
                evidence=evs[:60]
                or [
                    Evidence(
                        kind="file",
                        locator="tokenizer.json",
                        summary="no structural change after normalising key order and ByteLevel offset/decoder flags",
                    )
                ],
                remediation="Added or altered tokens change what the model sees; confirm each one is expected for this derivative.",
                tags=["tokenizer"],
            )
        )
    for fn in differing:
        if fn != "tokenizer.json":
            out.append(
                Finding(
                    check_id="static.tokenizer_integrity",
                    title=f"{fn} differs from {hop}",
                    status="warn",
                    severity="medium",
                    confidence=0.7,
                    declared=declared,
                    hop="suspect",
                    evidence=[
                        Evidence(
                            kind="file",
                            locator=fn,
                            summary=f"sha256 {a.files[fn].sha256()[:16]} vs {ref_art.files[fn].sha256()[:16]}",
                        )
                    ],
                    tags=["tokenizer"],
                )
            )

    # tokenizer_config drift (excluding chat_template, which has its own check) ----
    rcfg = ref_art.tokenizer_config
    keys = (set(tcfg) | set(rcfg)) - {"chat_template"}
    evs = []
    for k in sorted(keys):
        if tcfg.get(k) != rcfg.get(k):
            if k == "added_tokens_decoder":
                da, db = rcfg.get(k) or {}, tcfg.get(k) or {}
                for tid in sorted(
                    set(da) | set(db), key=lambda x: int(x) if str(x).isdigit() else 0
                ):
                    if da.get(tid) != db.get(tid):
                        evs.append(
                            Evidence(
                                kind="config_key",
                                locator=f"tokenizer_config.json#added_tokens_decoder[{tid}]",
                                summary=f"{_summ(da.get(tid))} -> {_summ(db.get(tid))}",
                            )
                        )
            else:
                evs.append(
                    Evidence(
                        kind="config_key",
                        locator=f"tokenizer_config.json#{k}",
                        summary=f"{_summ(rcfg.get(k))} -> {_summ(tcfg.get(k))}",
                    )
                )
    # a key that merely moved out of tokenizer_config.json (added_tokens_decoder, chat_template,
    # additional_special_tokens) is a serialization migration when tokenizer.json's added tokens are
    # byte-identical between the two sides; same for keys one side leaves unset
    _ta, _tb = a.json("tokenizer.json") or {}, ref_art.json("tokenizer.json") or {}
    tok_same_added = bool(_ta) and _ta.get("added_tokens") == _tb.get("added_tokens")

    def _is_migration(e: Evidence) -> bool:
        key = e.locator.split("#")[1].split("[")[0]
        if key not in MIGRATION_KEYS:
            return False
        va, vb = tcfg.get(key), rcfg.get(key)
        if key in ("added_tokens_decoder", "additional_special_tokens"):
            return tok_same_added and (va in (None, {}, []) or vb in (None, {}, []))
        return va is None or vb is None

    migration = [e for e in evs if _is_migration(e)]
    evs = [e for e in evs if e not in migration]
    if migration:
        out.append(
            Finding.passed(
                "static.tokenizer_integrity",
                f"{len(migration)} tokenizer_config.json key(s) differ only by serialization migration "
                "(transformers version re-save; tokenizer.json added tokens identical)",
                evidence=migration[:40],
                hop="suspect",
            )
        )
    if evs:
        critical_keys = {
            "eos_token",
            "bos_token",
            "pad_token",
            "unk_token",
            "add_bos_token",
            "add_eos_token",
            "additional_special_tokens",
        }
        sev = (
            "medium"
            if any(
                any(e.locator.endswith("#" + ck) for ck in critical_keys)
                or "added_tokens_decoder" in e.locator
                for e in evs
            )
            else "low"
        )
        only_notes = all(e.locator.split("#")[1].startswith("_") for e in evs)
        out.append(
            Finding(
                check_id="static.tokenizer_integrity",
                title=f"tokenizer_config.json differs from {hop} ({len(evs)} key(s))",
                status="pass" if only_notes else "warn",
                severity="info" if only_notes else sev,
                confidence=0.6,
                declared=declared,
                hop="suspect",
                evidence=evs[:60],
                tags=["tokenizer"],
            )
        )
    return out
