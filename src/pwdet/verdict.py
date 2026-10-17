"""Deterministic verdict rules. No composite score: a label plus the findings that drove it."""

from __future__ import annotations

from collections import Counter, defaultdict

from .schema import Finding, Layer, Verdict, severity_rank

CORE_CHECK_MIN = 4  # fewer core (static/provenance) checks actually run -> inconclusive


def coverage(findings: list[Finding], layer_of: dict[str, Layer]) -> dict[str, dict[str, int]]:
    cov: dict[str, Counter[str]] = defaultdict(Counter)
    for f in findings:
        layer = layer_of.get(f.check_id, "static")
        cov[layer][f.status] += 1
        if f.status not in ("skipped", "error"):
            cov[layer]["run"] += 1
    return {k: dict(v) for k, v in cov.items()}


def decide(
    findings: list[Finding],
    layer_of: dict[str, Layer],
    core_layers: tuple[str, ...] = ("static", "provenance"),
    core_min: int = CORE_CHECK_MIN,
) -> Verdict:
    """`core_layers`/`core_min`: which layers must have produced at least `core_min` run checks for a
    `clean` label (a probe-only report passes ("blackbox",), 1)."""
    cov = coverage(findings, layer_of)
    active = [f for f in findings if f.status in ("warn", "fail")]
    rationale: list[str] = []

    def hits(sev: str, min_conf: float) -> list[Finding]:
        return [
            f
            for f in active
            if severity_rank(f.effective_severity) >= severity_rank(sev)
            and f.confidence >= min_conf
        ]

    crit = hits("critical", 0.8)
    if crit:
        rationale += [f"{f.check_id}: {f.title}" for f in crit]
        return Verdict(label="malicious", rationale=rationale, coverage=cov)

    # a critical finding at any confidence, or a confident high one
    high = hits("high", 0.7) + [
        f for f in active if f.effective_severity == "critical" and f.confidence < 0.8
    ]
    if high:
        rationale += [f"{f.check_id}: {f.title}" for f in high]
        return Verdict(label="likely_tampered", rationale=rationale, coverage=cov)

    med = [f for f in active if f.effective_severity == "medium" and f.confidence >= 0.7]
    by_tag: dict[str, list[Finding]] = defaultdict(list)
    for f in med:
        for t in f.tags:
            by_tag[t].append(f)
    for tag, fs in by_tag.items():
        if len({f.check_id for f in fs}) >= 2:
            rationale.append(
                f"two independent medium findings share tag '{tag}': "
                + ", ".join(f.check_id for f in fs)
            )
            return Verdict(label="likely_tampered", rationale=rationale, coverage=cov)

    any_med = [f for f in active if severity_rank(f.effective_severity) >= severity_rank("medium")]
    if any_med:
        rationale += [f"{f.check_id}: {f.title}" for f in any_med]
        return Verdict(label="suspicious", rationale=rationale, coverage=cov)

    core_run = sum(cov.get(layer, {}).get("run", 0) for layer in core_layers)
    if core_run < core_min:
        rationale.append(
            f"only {core_run} core checks ran in {'/'.join(core_layers)} (need {core_min})"
        )
        return Verdict(label="inconclusive", rationale=rationale, coverage=cov)

    low = [f for f in active if f.effective_severity in ("low", "info")]
    if low:
        rationale.append(f"{len(low)} low/info observations, none above")
    return Verdict(label="clean", rationale=rationale, coverage=cov)
