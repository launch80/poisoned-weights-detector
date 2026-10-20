"""Run selected checks against a ScanContext and assemble a Report."""

from __future__ import annotations

import platform
import time
from collections.abc import Callable

from . import __version__, calibration
from .assurance import assess
from .context import FIRST_PARTY_NA, ScanContext
from .registry import CheckSpec
from .schema import Finding, Report
from .verdict import decide

Progress = Callable[[CheckSpec, str, float], None]


def run_checks(
    ctx: ScanContext, specs: list[CheckSpec], progress: Progress | None = None
) -> list[Finding]:
    caps = ctx.capabilities()
    findings: list[Finding] = []
    for spec in specs:
        missing = sorted(spec.requires - caps)
        t0 = time.perf_counter()
        if (
            missing
            and ctx.shared.get("first_party_subject")
            and all(m.startswith("reference:") for m in missing)
        ):
            findings.append(Finding.skipped(spec.id, FIRST_PARTY_NA))
            if progress:
                progress(spec, "skipped", 0.0)
            continue
        if missing:
            findings.append(Finding.skipped(spec.id, "missing capability: " + ", ".join(missing)))
            if progress:
                progress(spec, "skipped", 0.0)
            continue
        try:
            out = spec.fn(ctx)
            if not out:
                out = [Finding.passed(spec.id, "no observations")]
            for f in out:
                f.check_id = f.check_id or spec.id
                calibration.apply(f)
                for t in spec.tags:
                    if t not in f.tags:
                        f.tags.append(t)
            findings.extend(out)
            status = "ok"
        except Exception as e:  # a crashing check must not kill the scan
            findings.append(Finding.errored(spec.id, e))
            status = "error"
        if progress:
            progress(spec, status, time.perf_counter() - t0)
    return findings


def build_report(
    ctx: ScanContext, specs: list[CheckSpec], findings: list[Finding], started, profile: str | None
) -> Report:
    layer_of = {s.id: s.layer for s in specs}
    rep = Report(
        tool_version=__version__,
        started=started,
        finished=Report.now(),
        host={
            "hostname": platform.node(),
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        profile=profile,
        subject=ctx.subject,
        references=list(ctx.references.values()),
        lineage=ctx.lineage,
        findings=findings,
        notes=ctx.notes,
        manifest_path=ctx.shared.get("manifest_path"),
    )
    rep.verdict = decide(findings, layer_of)
    near = ctx.nearest_reference()
    verified = bool(near) and (
        near[1].kind == "hf-mirror" or any(h.hop == near[0] and h.verified for h in ctx.lineage)
    )
    rep.assurance = assess(
        findings,
        ctx.capabilities(),
        verified_reference=verified,
        first_party_identical=bool(
            ctx.shared.get("first_party_subject") and ctx.shared.get("hub_identical")
        ),
    )
    from .reproduce import digest, inputs_of

    rep.inputs = inputs_of(ctx, profile, specs)
    rep.digest = digest(findings, layer_of)
    rep.assurance.accuracy = calibration.accuracy(rep.assurance.level)
    rep.assurance.accuracy_text = calibration.accuracy_text(rep.assurance.level)
    return rep
