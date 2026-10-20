"""config.json / generation_config.json drift, tiered by what a change can do."""

from __future__ import annotations

import re
from typing import Any

from ...artifacts.index import declared_modifications
from ...context import ScanContext
from ...registry import check
from ...schema import Evidence, Finding

SECURITY_KEYS = ("auto_map", "trust_remote_code", "custom_pipelines", "transformers.js_config")
# transformers.js_config is a dtype / device map for transformers.js; it only matters when a value
# names code (a module path or URL), which no published dtype map does
DATA_ONLY_SECURITY_KEYS = ("transformers.js_config",)
BEHAVIOUR_PATTERNS = [
    r"(^|\.)vocab_size$",
    r"tie_word_embeddings$",
    r"rope",
    r"max_position_embeddings$",
    r"(eos|bos|pad)_token_id$",
    r"hidden_act$",
    r"num_hidden_layers$",
    r"num_attention_heads$",
    r"num_key_value_heads$",
    r"hidden_size$",
    r"intermediate_size$",
    r"head_dim$",
    r"sliding_window",
    r"layer_types$",
    r"rms_norm_eps$",
    r"attention_bias$",
    r"model_type$",
    r"architectures$",
    r"(^|\.)generation\.",
    r"temperature$",
    r"top_p$",
    r"top_k$",
    r"repetition_penalty$",
    r"do_sample$",
    r"max_new_tokens$",
    r"forced_.*_token_id$",
    r"bad_words_ids$",
    r"suppress_tokens$",
    r"image_token_id$",
    r"video_token_id$",
    r"vision_(start|end)_token_id$",
]
# Serving defaults: sampling settings, the declared context window and the bos/pad ids. Every
# runtime lets the caller override them and quantisers routinely change them (corpus: every
# community re-upload did), so a change is noted but is not tampering evidence. eos, forced and
# suppressed tokens stay behaviour: they decide what the model emits.
SERVING_PATTERNS = [
    r"^generation\.(temperature|top_p|top_k|min_p|typical_p|repetition_penalty|do_sample|"
    r"max_length|max_new_tokens|num_beams)$",
    r"(^|\.)(bos|pad)_token_id$",
    r"^max_position_embeddings$",
]
QUANT_PATTERNS = [
    r"quantization_config",
    r"quant",
    r"dtype$",
    r"^torch_dtype$",
    r"compression_config",
]
INFO_PATTERNS = [
    r"transformers_version$",
    r"^_",
    r"use_cache$",
    r"_attn_implementation",
    r"tokenizer_class$",
]
CODE_LIKE = re.compile(
    r"(^[\w.]+\.[A-Za-z_]\w*:[\w.]+$)|(\.py$)|(^https?://)|(^file:)|(\bimport\b)"
)


def flatten(d: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flatten(v, f"{prefix}{k}."))
    else:
        out[prefix[:-1]] = d
    return out


def canonical(cfg: dict[str, Any]) -> dict[str, Any]:
    """Undo transformers serialization migrations so only semantic differences remain.

    transformers >= 5 writes `dtype` for `torch_dtype`, nests rope settings under `rope_parameters`,
    materialises `layer_types` (all full_attention when sliding window is off) and writes explicit
    nulls for unset ids. None of that changes what the model computes."""
    c = dict(cfg)
    if "dtype" in c and "torch_dtype" not in c:
        c["torch_dtype"] = c.pop("dtype")
    rp = c.pop("rope_parameters", None)
    if isinstance(rp, dict):
        rp = dict(rp)
        if "rope_theta" in rp and "rope_theta" not in c:
            c["rope_theta"] = rp.pop("rope_theta")
        rtype = rp.pop("rope_type", "default")
        if rtype in (None, "default") and not rp:
            c.setdefault("rope_scaling", None)
        else:
            c.setdefault("rope_scaling", {"rope_type": rtype, **rp})
    lt = c.get("layer_types")
    if (
        isinstance(lt, list)
        and lt
        and all(x == "full_attention" for x in lt)
        and not c.get("use_sliding_window")
    ):
        c.pop("layer_types")
    # an explicit head_dim equal to the hidden_size / num_attention_heads default transformers
    # derives is the same model
    hd, hs, nh = c.get("head_dim"), c.get("hidden_size"), c.get("num_attention_heads")
    if isinstance(hd, int) and isinstance(hs, int) and isinstance(nh, int) and nh and hd * nh == hs:
        c.pop("head_dim")
    return {k: v for k, v in c.items() if v is not None}


TIER_ORDER = ["informational", "quantization", "other", "serving", "behaviour", "security"]


def tier(key: str, value: Any = None) -> str:
    if any(key == k or key.startswith(k + ".") or key.endswith("." + k) for k in SECURITY_KEYS):
        data_only = any(key == d or key.startswith(d + ".") for d in DATA_ONLY_SECURITY_KEYS)
        if data_only and not (isinstance(value, str) and CODE_LIKE.search(value)):
            return "informational"
        return "security"
    if any(re.search(p, key) for p in QUANT_PATTERNS):
        return "quantization"
    if any(re.search(p, key) for p in INFO_PATTERNS):
        return "informational"
    if any(re.search(p, key) for p in SERVING_PATTERNS):
        return "serving"
    if any(re.search(p, key) for p in BEHAVIOUR_PATTERNS):
        return "behaviour"
    return "other"


def _repackaging(key: str, flat: dict, rflat: dict, has_generation: bool) -> str | None:
    """Two routine repackaging changes that look like behaviour drift (corpus: community
    conversions): a missing generation_config.json, where runtimes fall back to config.json's own
    ids, and a vocabulary grown by a few rows for added tokens such as a pad token. The shared rows
    are still compared row by row and the added tokens are checked by the tokenizer checks, so an
    implant does not hide here."""
    if key.startswith("generation.") and not has_generation and key not in flat:
        return "serving"
    if re.search(r"(^|\.)vocab_size$", key):
        new, old = flat.get(key), rflat.get(key)
        if isinstance(new, int) and isinstance(old, int) and 0 < new - old <= 64:
            return "other"
    return None


@check(
    "static.config_drift",
    "static",
    requires=["artifact:config"],
    description="Config security keys and drift vs reference",
    tags=["config"],
)
def config_drift(ctx: ScanContext) -> list[Finding]:
    a = ctx.artifact
    assert a is not None
    out: list[Finding] = []
    declared = bool(declared_modifications(a))
    flat = flatten(canonical(a.config))
    flat.update({f"generation.{k}": v for k, v in flatten(canonical(a.generation_config)).items()})

    # absolute checks ----------------------------------------------------------
    sec = {
        k: v for k, v in flat.items() if tier(k, v) == "security" and v not in (None, False, {}, [])
    }
    if sec:
        out.append(
            Finding(
                check_id="static.config_drift",
                title="config requests custom code (auto_map / trust_remote_code)",
                status="fail",
                severity="high",
                confidence=0.9,
                evidence=[
                    Evidence(kind="config_key", locator=k, summary=str(v)[:300])
                    for k, v in sec.items()
                ],
                remediation="Loading needs trust_remote_code=True; review the referenced modules line by line first.",
                tags=["code-exec"],
            )
        )
    codey = {
        k: v
        for k, v in flat.items()
        if isinstance(v, str) and CODE_LIKE.search(v) and tier(k) != "security"
    }
    if codey:
        out.append(
            Finding(
                check_id="static.config_drift",
                title=f"{len(codey)} config value(s) look like module paths or URLs",
                status="warn",
                severity="medium",
                confidence=0.5,
                evidence=[
                    Evidence(kind="config_key", locator=k, summary=str(v)[:300])
                    for k, v in codey.items()
                ],
                tags=["code-exec"],
            )
        )
    side = {
        k: v
        for k, v in flat.items()
        if k.endswith(("side_file", "loader", "loader_class")) or "side_file" in k
    }
    if side:
        out.append(
            Finding(
                check_id="static.config_drift",
                title="config declares a side file / custom loader",
                status="warn",
                severity="medium",
                confidence=0.6,
                declared=declared,
                evidence=[
                    Evidence(kind="config_key", locator=k, summary=str(v)[:300])
                    for k, v in side.items()
                ],
                remediation="Standard loaders ignore these keys; the runtime that honours them is where the weights really come from.",
                tags=["loader"],
            )
        )
    notes = {k: v for k, v in flat.items() if k.startswith("_") or "._" in k}
    if notes:
        out.append(
            Finding.passed(
                "static.config_drift",
                "publisher notes in config",
                evidence=[
                    Evidence(kind="config_key", locator=k, summary=str(v)[:300])
                    for k, v in notes.items()
                ],
            )
        )

    # drift vs nearest local reference ------------------------------------------
    near = ctx.nearest_reference()
    ref_art = ctx.artifact_for(near[1]) if near else None
    if ref_art is None:
        out.append(ctx.no_reference("static.config_drift", "drift vs reference not compared"))
        return out
    hop = near[0]
    rflat = flatten(canonical(ref_art.config))
    rflat.update(
        {f"generation.{k}": v for k, v in flatten(canonical(ref_art.generation_config)).items()}
    )
    diffs: dict[str, list[Evidence]] = {
        "security": [],
        "behaviour": [],
        "serving": [],
        "quantization": [],
        "informational": [],
        "other": [],
    }
    for k in sorted(set(flat) | set(rflat)):
        if flat.get(k, "<absent>") != rflat.get(k, "<absent>"):
            t = max(tier(k, flat.get(k)), tier(k, rflat.get(k)), key=TIER_ORDER.index)
            t = _repackaging(k, flat, rflat, bool(a.generation_config)) or t
            diffs[t].append(
                Evidence(
                    kind="config_key",
                    locator=k,
                    summary=f"{str(rflat.get(k, '<absent>'))[:120]} -> {str(flat.get(k, '<absent>'))[:120]}",
                    reference_locator=f"{near[1].label()}:{k}",
                )
            )
    sev = {
        "security": ("high", 0.9),
        "behaviour": ("medium", 0.7),
        "other": ("low", 0.5),
        "serving": ("low", 0.5),
        "quantization": ("info", 0.3),
        "informational": ("info", 0.3),
    }
    for t, evs in diffs.items():
        if not evs:
            continue
        s, c = sev[t]
        out.append(
            Finding(
                check_id="static.config_drift",
                title=f"{len(evs)} {t} config key(s) differ from {hop}",
                status="warn" if t in ("security", "behaviour", "serving", "other") else "pass",
                severity=s,  # type: ignore[arg-type]
                confidence=c,
                declared=declared and t != "security",
                hop="suspect",
                evidence=evs[:60],
                tags=["config"],
            )
        )
    if not any(diffs.values()):
        out.append(Finding.passed("static.config_drift", f"config identical to {hop}"))
    return out
