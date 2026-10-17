"""Milestone-2 metrics: does each check catch the fixture it should, and stay quiet on the controls?"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .. import __version__, registry
from ..blackbox.differential import analyze, load_thresholds
from ..blackbox.runner import ModelEndpoint, load_results, rescore, run_all, save_results
from ..blackbox.stats import wilson
from ..blackbox.suite import expand, load_suites
from ..config import Settings
from ..context import ScanContext
from ..runner import run_checks
from ..schema import Finding
from .common import read_truth

SEV = ["info", "low", "medium", "high", "critical"]
BENIGN = {"B1", "B2", "BASE"}


def _worst(findings: list[Finding]) -> dict[str, int]:
    out: dict[str, int] = {}
    for f in findings:
        if f.status in ("warn", "fail"):
            out[f.check_id] = max(out.get(f.check_id, -1), SEV.index(f.effective_severity))
    return out


def static_findings(variant_dir: Path, base_path: str, settings: Settings) -> list[Finding]:
    ctx = ScanContext.build(str(variant_dir), settings, base=base_path)
    specs = registry.select(layers=["static"], max_cost=None)
    return run_checks(ctx, specs, lambda *a: None)


def probe_variant(
    run_dir: Path, base_url: str, name: str, trigger_tokens: list[str], seed: int = 7
) -> None:
    """Every model (base runs included) must see the SAME trigger variants, or pairs cannot align."""
    from ..blackbox.suite import BUILTIN_TRIGGERS

    suites = load_suites(None)
    toks = list(dict.fromkeys([*trigger_tokens, *BUILTIN_TRIGGERS]))
    reqs = [r for s in suites for r in expand(s, {"trigger_tokens": toks}, seed=seed)]
    ep = ModelEndpoint(label="suspect", base_url=base_url, model=name, concurrency=12)
    save_results(run_all(reqs, [ep], run_dir), run_dir / "results.json")


def behavioural_findings(
    variant_run: Path, base_run: Path, thresholds: dict[str, Any]
) -> list[Finding]:
    results = rescore(load_results(variant_run / "results.json"))
    results["reference"] = rescore(load_results(base_run / "results.json"))["suspect"]
    return analyze(results, "suspect", "reference", thresholds)


def evaluate(
    fixture_root: Path,
    base_path: str,
    variants: list[str],
    settings: Settings,
    probe_runs: dict[str, Path] | None = None,
    out_dir: Path = Path("eval"),
) -> dict[str, Any]:
    """probe_runs: {"BASE": runs/base, "F1": runs/F1, ...}; variants without a run get static-only metrics."""
    probe_runs = probe_runs or {}
    th = load_thresholds(settings.thresholds or "thresholds/default.yaml")
    per_variant: dict[str, Any] = {}
    for v in variants:
        vdir = fixture_root / v
        truth = read_truth(vdir) or {"variant": v, "expected": {}}
        findings = static_findings(vdir, base_path, settings)
        layer = {"static": _worst(findings)}
        if v in probe_runs and "BASE" in probe_runs:
            bf = behavioural_findings(probe_runs[v], probe_runs["BASE"], th)
            layer["blackbox"] = _worst(bf)
            findings += bf
        worst = {k: val for d in layer.values() for k, val in d.items()}
        expected: dict[str, str] = truth.get("expected", {})
        detected = {cid: worst.get(cid, -1) >= SEV.index(sev) for cid, sev in expected.items()}
        # a check that could not run (no endpoint) is "not evaluated", not "missed"
        for cid in list(detected):
            if cid.startswith("blackbox") and v not in probe_runs:
                detected[cid] = None
        fp = [
            (cid, SEV[w])
            for cid, w in worst.items()
            if w >= SEV.index("medium") and cid not in expected
        ]
        per_variant[v] = {
            "kind": truth.get("kind"),
            "expected": expected,
            "detected": detected,
            "worst_by_check": {k: SEV[w] for k, w in worst.items()},
            "false_positive_checks": fp,
            "titles": [
                f"{f.check_id}: {f.title}"
                for f in findings
                if f.status in ("warn", "fail") and SEV.index(f.effective_severity) >= 2
            ]
            + [f"ERROR {f.check_id}: {f.title}" for f in findings if f.status == "error"],
            "errors": [f.check_id for f in findings if f.status == "error"],
        }
    # base vs itself through the probe harness (two runs) is the purest FPR control
    if "BASE" in probe_runs and "BASE2" in probe_runs:
        bf = behavioural_findings(probe_runs["BASE2"], probe_runs["BASE"], th)
        w = _worst(bf)
        per_variant["BASE"] = {
            "kind": "self-vs-self",
            "expected": {},
            "detected": {},
            "worst_by_check": {k: SEV[x] for k, x in w.items()},
            "false_positive_checks": [(cid, SEV[x]) for cid, x in w.items() if x >= 2],
            "titles": [
                f"{f.check_id}: {f.title}"
                for f in bf
                if f.status in ("warn", "fail") and SEV.index(f.effective_severity) >= 2
            ],
        }
    tp = sum(1 for v in per_variant.values() for d in v["detected"].values() if d is True)
    evaluated = sum(
        1 for v in per_variant.values() for d in v["detected"].values() if d is not None
    )

    def _benign(v: str, r: dict[str, Any]) -> bool:
        return v in BENIGN or str(r.get("kind") or "").startswith("benign")

    fp_variants = [
        v for v, r in per_variant.items() if _benign(v, r) and r["false_positive_checks"]
    ]
    benign_n = sum(1 for v, r in per_variant.items() if _benign(v, r))
    metrics = {
        "tool_version": __version__,
        "generated": datetime.now(UTC).isoformat(),
        "tpr": {"hits": tp, "n": evaluated, "ci99": wilson(tp, evaluated)},
        "fpr": {
            "flagged": len(fp_variants),
            "n": benign_n,
            "ci99": wilson(len(fp_variants), benign_n),
            "variants": fp_variants,
        },
        "variants": per_variant,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=1, ensure_ascii=False))
    (out_dir / "metrics.md").write_text(render_md(metrics))
    return metrics


def render_md(m: dict[str, Any]) -> str:
    lines = [f"# pwdet fixture evaluation ({m['tool_version']}, {m['generated'][:19]}Z)", ""]
    t, f = m["tpr"], m["fpr"]
    lines.append(
        f"- expected detections hit: **{t['hits']}/{t['n']}** (99% CI {t['ci99'][0]:.2f}-{t['ci99'][1]:.2f})"
    )
    lines.append(
        f"- benign variants with a medium+ finding: **{f['flagged']}/{f['n']}** {f['variants'] or ''}"
    )
    lines.append("")
    lines.append("| variant | kind | check | expected | observed | result |")
    lines.append("|---|---|---|---|---|---|")
    for v, r in m["variants"].items():
        if r["expected"]:
            for cid, sev in r["expected"].items():
                d = r["detected"].get(cid)
                res = "n/a" if d is None else ("caught" if d else "MISSED")
                lines.append(
                    f"| {v} | {r['kind']} | {cid} | {sev} | {r['worst_by_check'].get(cid, 'none')} | {res} |"
                )
        else:
            fps = ", ".join(f"{c} ({s})" for c, s in r["false_positive_checks"]) or "none"
            verdict = "FALSE POSITIVE" if r["false_positive_checks"] else "quiet"
            lines.append(
                f"| {v} | {r['kind']} | (control) | nothing above low | {fps} | {verdict} |"
            )
    lines.append("")
    for v, r in m["variants"].items():
        if r["titles"]:
            lines.append(f"## {v}")
            lines += [f"- {t}" for t in r["titles"][:12]]
            lines.append("")
    return "\n".join(lines)
