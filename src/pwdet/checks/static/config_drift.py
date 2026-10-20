"""config.json / generation_config.json drift, tiered by what a change can do."""

from __future__ import annotations

import re
from typing import Any

from ...artifacts.index import declared_modifications
from ...context import ScanContext
from ...registry import check
from ...schema import Evidence, Finding

SECURITY_KEYS = ("auto_map", "trust_remote_code", "custom_pipelines", "transformers.js_config")
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
    return {k: v for k, v in c.items() if v is not None}


def tier(key: str) -> str:
    if any(key == k or key.startswith(k + ".") or key.endswith("." + k) for k in SECURITY_KEYS):
        return "security"
    if any(re.search(p, key) for p in QUANT_PATTERNS):
        return "quantization"
    if any(re.search(p, key) for p in INFO_PATTERNS):
        return "informational"
    if any(re.search(p, key) for p in BEHAVIOUR_PATTERNS):
        return "behaviour"
    return "other"


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
        k: v for k, v in flat.items() if tier(k) == "security" and v not in (None, False, {}, [])
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
        out.append(
            Finding.skipped(
                "static.config_drift",
                "missing capability: reference:any:local (drift vs reference not compared)",
            )
        )
        return out
    hop = near[0]
    rflat = flatten(canonical(ref_art.config))
    rflat.update(
        {f"generation.{k}": v for k, v in flatten(canonical(ref_art.generation_config)).items()}
    )
    diffs: dict[str, list[Evidence]] = {
        "security": [],
        "behaviour": [],
        "quantization": [],
        "informational": [],
        "other": [],
    }
    for k in sorted(set(flat) | set(rflat)):
        if flat.get(k, "<absent>") != rflat.get(k, "<absent>"):
            diffs[tier(k)].append(
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
                status="warn" if t in ("security", "behaviour", "other") else "pass",
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
