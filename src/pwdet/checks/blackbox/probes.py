"""Registered black-box check: run the probe suites against the suspect endpoint (and reference), analyse."""

from __future__ import annotations

from pathlib import Path

from ...blackbox.differential import analyze, load_thresholds, power_statement
from ...blackbox.runner import ModelEndpoint, all_cached, run_all, save_results
from ...blackbox.suite import expand, load_suites
from ...context import ScanContext
from ...registry import check
from ...schema import Evidence, Finding

SAMPLED_T = 0.7


def _slug(s: str) -> str:
    import hashlib
    import re

    return (
        re.sub(r"[^A-Za-z0-9._-]+", "--", s)[:80] + "-" + hashlib.sha1(s.encode()).hexdigest()[:8]
    )


def endpoint_for(ref, label: str, concurrency: int = 8) -> ModelEndpoint:
    cfg = ref.serve if hasattr(ref, "serve") else None
    return ModelEndpoint(
        label=label,
        base_url=ref.endpoint,
        model=ref.served_model or "",
        api_key=getattr(cfg, "api_key", None),
        thinking_kwarg=getattr(cfg, "thinking_kwarg", "enable_thinking"),
        concurrency=concurrency,
        logprobs=False,
    )


@check(
    "blackbox.probes",
    layer="blackbox",
    requires=["endpoint:suspect"],
    cost="expensive",
    tags=["behaviour", "anonymity", "trigger", "exfil"],
    description="Paired behavioural probes (triggers, PII solicitation, exfiltration, canaries, fingerprinting) "
    "against the served suspect, differential against a served reference when one is configured.",
)
def run(ctx: ScanContext) -> list[Finding]:
    opts = ctx.options
    suites = load_suites(opts.get("suite_paths"), opts.get("suites"))
    extra = {}
    cands = ctx.shared.get("trigger_candidates") or {}
    vals = cands.values() if isinstance(cands, dict) else cands
    toks = [str(c["token"]).strip("'") for c in vals if isinstance(c, dict) and c.get("token")]
    toks = [t for t in toks if t and not t.startswith("<id ")][:12]
    toks += [t for t in opts.get("triggers", []) if t not in toks]
    if toks:
        from ...blackbox.suite import BUILTIN_TRIGGERS

        extra["trigger_tokens"] = list(dict.fromkeys([*toks, *BUILTIN_TRIGGERS[:4]]))
    reqs = []
    for s in suites:
        reqs.extend(expand(s, extra, seed=int(opts.get("seed", 7))))
    if not reqs:
        return [Finding.skipped("blackbox.probes", "no probes expanded")]

    conc = int(opts.get("concurrency", 8))
    targets: list[tuple[str, object]] = [("suspect", ctx.subject)]
    ref_label = None
    for hop in ("parent", "base"):
        r = ctx.references.get(hop)
        if r is not None and (
            r.endpoint or (opts.get("serve") and r.path and r.kind != "hf-mirror")
        ):
            targets.append(("reference", r))
            ref_label = "reference"
            ctx.shared["probe_reference_hop"] = hop
            break
    ref_key = next((t[1].key for t in targets if t[0] == "reference"), "none")  # type: ignore[attr-defined]
    run_dir = Path(
        opts.get("run_dir")
        or (
            ctx.settings.cache_path
            / "runs"
            / f"{ctx.subject.key.replace('/', '--')}--vs--{ref_key.replace('/', '--')}"
        )
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    log = opts.get("log") or (lambda _: None)
    # the reference answers twice: once like the suspect, once sampled, so this run measures how
    # far its own rates move by chance and sets the decision thresholds from that
    recal = bool(ref_label) and opts.get("self_calibrate", True)
    sampled = [r.model_copy(update={"temperature": SAMPLED_T, "seed": r.seed + 1000}) for r in reqs]
    results: dict = {}
    for label, ref in targets:
        jobs = [(label, reqs)] + (
            [("reference_sampled", sampled)] if label == "reference" and recal else []
        )
        # a reference answers the same prompts for every model compared with it: its answers are
        # kept per reference, so a second scan against the same base does not ask it again
        rdir = run_dir
        if label == "reference":
            ident = getattr(ref, "path", None) or getattr(ref, "endpoint", None) or ref.key  # type: ignore[attr-defined]
            rdir = ctx.settings.cache_path / "runs" / "references" / _slug(f"{ref.key}-{ident}")  # type: ignore[attr-defined]
        if getattr(ref, "endpoint", None):
            for lab, rq in jobs:
                ep = endpoint_for(ref, lab, conc)
                results.update(run_all(rq, [ep], rdir, budget_s=opts.get("budget_s")))
            continue
        name = f"pwdet/{label}"
        eps = [
            (ModelEndpoint(label=lab, base_url="", model=name, concurrency=max(conc, 12)), rq)
            for lab, rq in jobs
        ]
        if all(all_cached(rq, ep, rdir) for ep, rq in eps):
            log(f"{label}: every answer already cached, not serving")
            for ep, rq in eps:
                results.update(run_all(rq, [ep], rdir))
            continue
        from ...serve import served

        port = int(opts.get("port", 8010))
        with served(Path(ref.path), name, port=port, gpu=opts.get("gpu", "auto"), log=log) as url:  # type: ignore[attr-defined]
            for ep, rq in eps:
                ep.base_url = url
                results.update(run_all(rq, [ep], rdir, budget_s=opts.get("budget_s")))
    calib = None
    if "reference_sampled" in results:
        from ...blackbox.differential import calibrate

        calib = calibrate(results["reference"], results.pop("reference_sampled"))
    save_results(results, run_dir / "results.json")
    ctx.shared["probe_run_dir"] = str(run_dir)
    th = load_thresholds(ctx.settings.thresholds)
    if calib:
        # measured noise can only make a threshold stricter, never loosen a configured one
        rates = {
            k: None if v is None else max(v, th["abs_rate"].get(k) or 0)
            for k, v in calib["abs_rate"].items()
        }
        th = {**th, "min_delta": max(th["min_delta"], calib["min_delta"]), "abs_rate": rates}
    expects = {e for e in ctx.subject.expect} | set(opts.get("expect", []))
    findings = analyze(results, "suspect", ref_label, th, expects)
    ctx.shared["behaviour_triggers"] = behaviour_triggers(findings)
    n_ok = sum(1 for r in results["suspect"] if not r.response.get("error"))
    cov = Finding.passed(
        "blackbox.probes",
        f"{len(reqs)} requests per model over {len(suites)} suite(s); suspect answered {n_ok}/{len(reqs)}"
        + (
            "; reference answered "
            + str(sum(1 for r in results[ref_label] if not r.response.get("error")))
            if ref_label
            else "; no reference endpoint (absolute floors only)"
        ),
        evidence=[
            Evidence(
                kind="note",
                locator=str(run_dir),
                summary="cached responses (JSONL per model) and results.json",
            )
        ]
        + (
            [
                Evidence(
                    kind="metric",
                    locator="self_calibration",
                    summary=f"thresholds from this run's reference noise: min_delta "
                    f"{calib['min_delta']} (reference at temperature 0 vs {SAMPLED_T})",
                    data=calib,
                )
            ]
            if calib
            else []
        ),
        tags=["coverage"],
    )
    pw = power_statement(results, th)
    return [cov, *([pw] if pw else []), *findings]


def behaviour_triggers(findings: list[Finding]) -> list[str]:
    """Trigger strings the probes caught flipping the suspect's behaviour, for the white-box layer
    to look inside the model at: a variant reads `trigger_tokens=cf-7731`, possibly several."""
    out: list[str] = []
    for f in findings:
        if f.status != "fail" or "trigger" not in f.tags or not f.evidence:
            continue
        variant = str((f.evidence[0].data or {}).get("variant") or "all")
        if variant == "all":
            continue
        for part in variant.split(", "):
            _, _, val = part.partition("=")
            if val.strip():
                out.append(val.strip())
    return list(dict.fromkeys(out))
