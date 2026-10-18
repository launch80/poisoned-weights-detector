"""Registered black-box check: run the probe suites against the suspect endpoint (and reference), analyse."""

from __future__ import annotations

from pathlib import Path

from ...blackbox.differential import analyze, load_thresholds
from ...blackbox.runner import ModelEndpoint, run_all, save_results
from ...blackbox.suite import expand, load_suites
from ...context import ScanContext
from ...registry import check
from ...schema import Evidence, Finding


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

    eps = [endpoint_for(ctx.subject, "suspect", int(opts.get("concurrency", 8)))]
    ref_label = None
    for hop in ("parent", "base"):
        r = ctx.references.get(hop)
        if r is not None and r.endpoint:
            eps.append(endpoint_for(r, "reference", int(opts.get("concurrency", 8))))
            ref_label = "reference"
            ctx.shared["probe_reference_hop"] = hop
            break
    run_dir = Path(
        opts.get("run_dir")
        or (ctx.settings.cache_path / "runs" / ctx.subject.key.replace("/", "--"))
    )
    results = run_all(reqs, eps, run_dir, budget_s=opts.get("budget_s"))
    save_results(results, run_dir / "results.json")
    ctx.shared["probe_run_dir"] = str(run_dir)
    th = load_thresholds(ctx.settings.thresholds)
    expects = {e for e in ctx.subject.expect} | set(opts.get("expect", []))
    findings = analyze(results, "suspect", ref_label, th, expects)
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
        ],
        tags=["coverage"],
    )
    return [cov, *findings]
