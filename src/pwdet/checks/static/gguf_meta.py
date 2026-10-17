"""GGUF metadata: lineage KVs, embedded template/tokenizer reuse, oversized or URL-bearing strings."""

from __future__ import annotations

import re

from ...artifacts.gguf_io import HAVE_GGUF, GGUFFile
from ...context import ScanContext
from ...registry import check
from ...schema import Evidence, Finding
from .chat_template import Walk, literal_flags

URL_RX = re.compile(r"https?://|data:[\w/+.-]+;base64,", re.I)
ZW = {"​", "‌", "‍", "⁠", "﻿"}
LINEAGE_KEYS = (
    "general.base_model.0.repo_url",
    "general.base_model.0.name",
    "general.source.url",
    "general.source.repo_url",
    "general.finetune",
    "general.basename",
)


@check(
    "static.gguf_meta",
    "static",
    requires=["artifact:gguf"],
    cost="medium",
    description="GGUF KV metadata: lineage, embedded chat template literals, suspicious token strings and KVs",
    tags=["gguf", "template", "tokenizer"],
)
def gguf_meta(ctx: ScanContext) -> list[Finding]:
    a = ctx.artifact
    assert a is not None
    if not HAVE_GGUF:
        return [
            Finding.skipped("static.gguf_meta", "gguf extra not installed (uv sync --extra gguf)")
        ]
    out: list[Finding] = []
    for fe in a.gguf_files:
        g = GGUFFile(fe.path)
        kv = g.kv
        lineage = {k: kv[k] for k in LINEAGE_KEYS if k in kv}
        out.append(
            Finding.passed(
                "static.gguf_meta",
                f"{fe.rel}: arch {g.arch}, {len(g.tensors)} tensors, {len(kv)} KVs"
                + (f"; lineage {lineage}" if lineage else "; no lineage KVs"),
                evidence=[
                    Evidence(
                        kind="file",
                        locator=fe.rel,
                        summary=f"file type {kv.get('general.file_type')} quant version {kv.get('general.quantization_version')}",
                        data={"lineage": lineage, "name": kv.get("general.name")},
                    )
                ],
            )
        )
        # template literals ---------------------------------------------------------
        tpl = g.chat_template()
        if tpl:
            w = Walk()
            w.run(tpl)
            flags = [
                f"{lit[:200]!r}: {', '.join(why)}"
                for lit in w.literals
                if (why := literal_flags(lit))
            ]
            if flags:
                out.append(
                    Finding(
                        check_id="static.gguf_meta",
                        title=f"{fe.rel}: embedded chat template has {len(flags)} literal(s) with URL/PII/covert phrasing",
                        status="warn",
                        severity="high",
                        confidence=0.8,
                        evidence=[
                            Evidence(
                                kind="template_snippet",
                                locator=f"{fe.rel}#tokenizer.chat_template",
                                summary=s[:300],
                            )
                            for s in flags[:10]
                        ],
                        tags=["template", "injection"],
                        remediation="The GGUF template is what llama.cpp applies; compare it to the source repo's template.",
                    )
                )
            ref = a.chat_templates()
            if ref and all(tpl.strip() != r.strip() for r in ref.values()):
                out.append(
                    Finding(
                        check_id="static.gguf_meta",
                        title=f"{fe.rel}: embedded chat template differs from the repo's own template file(s)",
                        status="warn",
                        severity="medium",
                        confidence=0.6,
                        evidence=[
                            Evidence(
                                kind="template_snippet",
                                locator=f"{fe.rel}#tokenizer.chat_template",
                                summary=tpl[:400],
                            )
                        ],
                        tags=["template"],
                    )
                )
        # tokens -----------------------------------------------------------------------
        toks = g.token_strings()
        odd = [(i, t) for i, t in enumerate(toks) if any(c in ZW for c in t) or URL_RX.search(t)]
        if odd:
            out.append(
                Finding(
                    check_id="static.gguf_meta",
                    title=f"{fe.rel}: {len(odd)} token string(s) contain zero-width characters or URLs",
                    status="warn",
                    severity="medium",
                    confidence=0.6,
                    evidence=[
                        Evidence(
                            kind="config_key",
                            locator=f"{fe.rel}#tokenizer.ggml.tokens[{i}]",
                            summary=repr(t)[:120],
                        )
                        for i, t in odd[:20]
                    ],
                    tags=["tokenizer"],
                )
            )
        # free-form KVs ---------------------------------------------------------------
        bad_kv = []
        for k, v in kv.items():
            if k.startswith("tokenizer.ggml.") or k == "tokenizer.chat_template":
                continue
            if isinstance(v, str) and (
                len(v) > 4000
                or URL_RX.search(v)
                and not k.endswith(("repo_url", "url", "source.url", "license.link"))
            ):
                bad_kv.append((k, v))
        if bad_kv:
            out.append(
                Finding(
                    check_id="static.gguf_meta",
                    title=f"{fe.rel}: {len(bad_kv)} metadata string(s) are oversized or carry URLs outside the lineage keys",
                    status="warn",
                    severity="low",
                    confidence=0.5,
                    evidence=[
                        Evidence(kind="config_key", locator=f"{fe.rel}#{k}", summary=v[:200])
                        for k, v in bad_kv[:20]
                    ],
                    tags=["gguf"],
                )
            )
    return out
