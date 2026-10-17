"""Pickle-based weight files: any non-allowlisted callable is a code-execution path."""

from __future__ import annotations

from ...artifacts.pickle_scan import scan_file
from ...context import ScanContext
from ...registry import check
from ...schema import Evidence, Finding


@check(
    "static.pickle_payload",
    "static",
    requires=["artifact:pickle"],
    description="Pickle opcode scan for code execution",
    tags=["code-exec"],
)
def pickle_payload(ctx: ScanContext) -> list[Finding]:
    a = ctx.artifact
    assert a is not None
    out: list[Finding] = []
    for f in a.pickle_like_files:
        res = scan_file(f.path)
        ev_common = Evidence(
            kind="file",
            locator=f.rel,
            summary=f"{len(res.globals)} global refs in {len(res.members) or 1} pickle stream(s); "
            f"build ops {res.build_ops or '{}'}; protocol {res.protocol}",
            data={"members": res.members, "errors": res.errors, "build_ops": res.build_ops},
        )
        if res.dangerous:
            out.append(
                Finding(
                    check_id="static.pickle_payload",
                    title=f"{f.rel}: pickle imports dangerous callables ({len(res.dangerous)})",
                    status="fail",
                    severity="critical",
                    confidence=0.95,
                    evidence=[ev_common]
                    + [
                        Evidence(
                            kind="file", locator=f"{f.rel}:{g.member}@{g.pos}", summary=g.qualname
                        )
                        for g in res.dangerous[:30]
                    ],
                    remediation="Do not load this file with torch.load/pickle. Convert from a trusted source or delete it.",
                    tags=["code-exec"],
                )
            )
        unknown = [
            g
            for g in res.unknown
            if g.module.split(".")[0] not in ("torch", "numpy", "collections")
        ]
        softunk = [g for g in res.unknown if g not in unknown]
        if unknown:
            out.append(
                Finding(
                    check_id="static.pickle_payload",
                    title=f"{f.rel}: pickle references {len(unknown)} non-standard callable(s)",
                    status="warn",
                    severity="medium",
                    confidence=0.7,
                    evidence=[ev_common]
                    + [
                        Evidence(
                            kind="file", locator=f"{f.rel}:{g.member}@{g.pos}", summary=g.qualname
                        )
                        for g in unknown[:30]
                    ],
                    remediation="Verify each callable is a benign custom class; anything importable runs code on load.",
                    tags=["code-exec"],
                )
            )
        if softunk:
            out.append(
                Finding(
                    check_id="static.pickle_payload",
                    title=f"{f.rel}: {len(softunk)} torch/numpy callable(s) outside the allowlist",
                    status="warn",
                    severity="low",
                    confidence=0.5,
                    evidence=[
                        Evidence(
                            kind="file", locator=f"{f.rel}:{g.member}@{g.pos}", summary=g.qualname
                        )
                        for g in softunk[:30]
                    ],
                    tags=["code-exec"],
                )
            )
        if res.errors:
            out.append(
                Finding(
                    check_id="static.pickle_payload",
                    title=f"{f.rel}: scan anomalies",
                    status="warn",
                    severity="low",
                    confidence=0.4,
                    evidence=[
                        Evidence(kind="file", locator=f.rel, summary=e) for e in res.errors[:20]
                    ],
                    tags=["integrity"],
                )
            )
        if not (res.dangerous or unknown or softunk or res.errors):
            out.append(
                Finding.passed(
                    "static.pickle_payload",
                    f"{f.rel}: only allowlisted callables",
                    evidence=[ev_common],
                )
            )
    return out
