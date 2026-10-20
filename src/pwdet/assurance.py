"""What a scan actually tested: per-check state, per-threat coverage and an assurance level.

A verdict only speaks for the checks that ran. This module turns the findings into a plain account
of that: which threats from the threat model were tested, which only partly, which not at all and
why, and one overall level (basic, standard, thorough, deep) that qualifies the verdict label.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from .schema import Assurance, Finding, ThreatCoverage

# Missing capabilities that mean "there is nothing of this kind to check" rather than "we could
# not check it": a model without pickle files has no pickle payload to find.
NOT_APPLICABLE_CAPS = {
    "artifact:pickle",
    "artifact:gguf",
    "artifact:safetensors",
    "artifact:template",
    "artifact:config",
}

# The README's threat model. `required` checks must all run for a threat to count as tested;
# `extra` checks add depth when they run but are never required.
THREATS: list[tuple[str, str, list[str], list[str]]] = [
    (
        "code_on_load",
        "Code execution on load",
        ["static.inventory", "static.pickle_payload", "static.config_drift", "static.st_header"],
        ["static.gguf_meta"],
    ),
    (
        "prompt_injection",
        "Prompt-level injection",
        ["static.chat_template", "static.tokenizer_integrity"],
        ["static.gguf_meta"],
    ),
    (
        "weight_implants",
        "Weight-level implants",
        [
            "static.weight_stats",
            "static.embedding_compare",
            "static.unembedding_compare",
            "static.norm_compare",
        ],
        [
            "whitebox.unembed_anomaly",
            "whitebox.layer_delta",
            "whitebox.activation_stats",
            "whitebox.attention_heads",
        ],
    ),
    (
        "trained_behaviour",
        "Trained-in covert behaviour",
        ["blackbox.probes"],
        ["whitebox.trigger_inversion"],
    ),
    (
        "supply_chain",
        "Supply-chain lies",
        ["provenance.hub_metadata", "provenance.card_lineage", "provenance.parent_xcheck"],
        ["provenance.manifest", "static.file_hashes"],
    ),
]

# How to close a gap, by the capability that was missing.
REMEDY = {
    "reference:any:local": "pass --parent/--base with the model it derives from, on disk",
    "reference:any": "pass --parent/--base with the model it derives from",
    "reference:parent": "pass --parent with the model it claims to derive from",
    "endpoint:suspect": "serve the model and scan its endpoint (profile milestone1 or all)",
    "endpoint:reference": "serve the reference model too, so behaviour is compared, not just floored",
    "docker": "install docker for the white-box layer",
    "hub:repo": "scan a Hugging Face repo id so its provenance can be checked",
    "artifact:dir": "scan the checkpoint files, not only an endpoint",
    "artifact:weights": "scan the checkpoint files, not only an endpoint",
}


def missing_caps(f: Finding) -> list[str]:
    """Capabilities named by a skipped finding's reason, if any."""
    reason = f.title.removeprefix("skipped: ")
    if not reason.startswith("missing capability:"):
        return []
    caps = reason.removeprefix("missing capability:").split("(")[0]
    return [c.strip() for c in caps.split(",") if c.strip()]


def not_applicable(f: Finding) -> bool:
    if f.status != "skipped":
        return False
    if f.title.removeprefix("skipped: ").startswith("not applicable"):
        return True
    caps = missing_caps(f)
    return bool(caps) and all(c in NOT_APPLICABLE_CAPS for c in caps)


def check_states(findings: Iterable[Finding]) -> dict[str, str]:
    """One state per check id: full, partial, gap, error or n/a.

    full: it ran and nothing of it was skipped. partial: it ran, but part of it was skipped for a
    missing capability (a reference, typically). gap: it did not run for a missing capability.
    """
    by: dict[str, list[Finding]] = defaultdict(list)
    for f in findings:
        by[f.check_id].append(f)
    out = {}
    for cid, fs in by.items():
        ran = [f for f in fs if f.status not in ("skipped", "error")]
        gaps = [f for f in fs if f.status == "skipped" and not not_applicable(f)]
        if ran:
            out[cid] = "partial" if gaps else "full"
        elif any(f.status == "error" for f in fs):
            out[cid] = "error"
        elif gaps:
            out[cid] = "gap"
        else:
            out[cid] = "n/a"
    return out


def _why(cid: str, fs: list[Finding], state: str) -> str:
    if state == "absent":
        return f"{cid}: not part of this profile"
    if state == "error":
        return f"{cid}: the check failed to run"
    skipped = [f for f in fs if f.status == "skipped" and not not_applicable(f)]
    caps = sorted({c for f in skipped for c in missing_caps(f)})
    reason = skipped[0].title.removeprefix("skipped: ") if skipped else "skipped"
    fix = next((REMEDY[c] for c in caps if c in REMEDY), "")
    return f"{cid}: {reason}" + (f"; to close it, {fix}" if fix else "")


def assess(
    findings: list[Finding], caps: set[str] | None = None, verified_reference: bool = False
) -> Assurance:
    """The assurance block for a report. `caps` are the scan's capabilities."""
    caps = caps or set()
    states = check_states(findings)
    by: dict[str, list[Finding]] = defaultdict(list)
    for f in findings:
        by[f.check_id].append(f)

    threats: list[ThreatCoverage] = []
    for tid, title, required, extra in THREATS:
        ran, missing = [], []
        considered = 0
        for cid in required:
            st = states.get(cid, "absent")
            if st == "n/a":
                continue
            considered += 1
            if st in ("full", "partial"):
                ran.append(cid)
            if st != "full":
                missing.append(_why(cid, by.get(cid, []), st))
        ran += [c for c in extra if states.get(c) in ("full", "partial")]
        full = sum(1 for c in required if states.get(c) == "full")
        if considered == 0:
            state = "not_applicable"
        elif full == considered:
            state = "tested"
        elif not any(c in ran for c in required):
            state = "not_tested"
        else:
            state = "partial"
        if tid == "trained_behaviour" and state == "tested" and "endpoint:reference" not in caps:
            state = "partial"
            missing.append(
                "blackbox.probes: no reference endpoint, so only absolute floors applied; "
                "to close it, " + REMEDY["endpoint:reference"]
            )
        threats.append(ThreatCoverage(id=tid, title=title, state=state, ran=ran, missing=missing))

    compared = any(
        states.get(c) == "full"
        for c in ("static.embedding_compare", "static.norm_compare", "static.unembedding_compare")
    )
    behaviour = (
        states.get("blackbox.probes") in ("full", "partial") and "endpoint:reference" in caps
    )
    whitebox = any(
        states.get(c) in ("full", "partial") for c in states if c.startswith("whitebox.")
    )
    level = "basic"
    if compared:
        level = "standard"
        if behaviour:
            level = "thorough"
            if whitebox:
                level = "deep"

    t = {th.id: th for th in threats}
    parts = [LEVEL_TEXT[level]]
    if t["trained_behaviour"].state == "not_tested":
        parts.append("Behaviour was not tested, so a trained-in trigger would not be found.")
    if level != "basic" and not verified_reference:
        parts.append("The reference was not hash-verified against the Hub.")
    gaps = [m for th in threats for m in th.missing]
    return Assurance(level=level, summary=" ".join(parts), threats=threats, gaps=gaps)


LEVEL_TEXT = {
    "basic": "Basic: static and provenance checks of this model alone, with no reference to compare against.",
    "standard": "Standard: static and provenance checks, with weights compared tensor by tensor against a reference.",
    "thorough": "Thorough: Standard plus behavioural probes compared against a served reference.",
    "deep": "Deep: Thorough plus white-box analysis of the loaded model.",
}
