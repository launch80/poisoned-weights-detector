"""pwdet command line."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from . import __version__, registry
from .assurance import assess
from .config import load_settings
from .context import ScanContext
from .fetch import FetchError, ensure_subject, parse_size, prepare_references
from .gpu import resolve_gpu
from .report.html_writer import write_html
from .report.json_writer import export_schema, write_json
from .runner import build_report, run_checks
from .schema import Report

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Poisoned-weights detector.")
checks_app = typer.Typer(help="List and describe checks.")
report_app = typer.Typer(help="Report utilities.")
manifest_app = typer.Typer(help="Provenance manifests.")
probe_app = typer.Typer(help="Black-box behavioural probes against served endpoints.")
fixture_app = typer.Typer(
    help="Build and evaluate synthetic poisoned / benign fixtures (Milestone 2)."
)
app.add_typer(checks_app, name="checks")
app.add_typer(report_app, name="report")
app.add_typer(manifest_app, name="manifest")
app.add_typer(probe_app, name="probe")
app.add_typer(fixture_app, name="fixture")
corpus_app = typer.Typer(
    help="Evaluation corpus: build, scan, calibrate, gate.", no_args_is_help=True
)
app.add_typer(corpus_app, name="corpus")
console = Console()
_COLOR = {"ok": "green", "skipped": "yellow"}

PROFILES: dict[str, dict] = {
    # assurance-level names (what L80 shows), then the original names
    "basic": {"layers": ["static", "provenance"], "max_cost": "cheap", "fetch": False},
    "standard": {"layers": ["static", "provenance"], "max_cost": None},
    "thorough": {"layers": ["static", "provenance", "blackbox"], "max_cost": None, "serve": True},
    "deep": {
        "layers": ["static", "provenance", "blackbox", "whitebox"],
        "max_cost": None,
        "serve": True,
    },
    "quick": {"layers": ["static", "provenance"], "max_cost": "cheap"},
    "full-static": {"layers": ["static", "provenance"], "max_cost": None},
    "milestone1": {"layers": ["static", "provenance", "blackbox"], "max_cost": None},
    "whitebox": {"layers": ["static", "whitebox"], "max_cost": None},
    "all": {"layers": None, "max_cost": None},
}


def _version(v: bool) -> None:
    if v:
        console.print(f"pwdet {__version__}")
        raise typer.Exit()


@app.callback()
def _main(
    version: bool = typer.Option(False, "--version", callback=_version, is_eager=True),
) -> None:
    pass


@checks_app.command("list")
def checks_list(
    layer: str | None = typer.Option(None, help="static|provenance|blackbox|whitebox"),
    plain: bool = typer.Option(False, help="tab-separated, one check per line"),
) -> None:
    specs = registry.select(layers=[layer] if layer else None)
    if plain:
        for s in specs:
            typer.echo(
                f"{s.id}\t{s.layer}\t{s.cost}\t{','.join(sorted(s.requires))}\t{s.description}"
            )
        return
    t = Table("id", "layer", "cost", "requires", "description")
    for s in specs:
        t.add_row(s.id, s.layer, s.cost, ", ".join(sorted(s.requires)), s.description)
    console.print(t)


@report_app.command("schema")
def report_schema(out: Path = typer.Option(Path("schema/report.v2.json"))) -> None:
    console.print(f"wrote {export_schema(out)}")


@report_app.command("render")
def report_render(json_path: Path, out: Path | None = None) -> None:
    rep = Report.model_validate_json(json_path.read_text())
    out = out or json_path.with_suffix(".html")
    console.print(f"wrote {write_html(rep, out)}")


@manifest_app.command("verify")
def manifest_verify(
    manifest: Path,
    root: Path | None = typer.Option(
        None, help="re-hash files under this directory (default: the manifest's subject path)"
    ),
) -> None:
    from .manifest.provenance_manifest import verify

    env = json.loads(manifest.read_text())
    if root is None:
        import base64

        st = json.loads(base64.b64decode(env["payload"]))
        p = (st.get("predicate", {}).get("subject") or {}).get("path")
        root = Path(p) if p else None
    ok, msgs = verify(env, root)
    for m in msgs[:50]:
        console.print(("  [green]" if ok else "  [red]") + m)
    console.print(f"manifest {'VALID' if ok else 'INVALID'}")
    raise typer.Exit(code=0 if ok else 1)


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-")[:80]


def _parse_endpoint(spec: str):
    url, _, model = spec.partition("#")
    if not url.startswith(("http://", "https://")) or not model:
        raise typer.BadParameter(f"{spec}: expected http://host:port/v1#model")
    return url, model


def _attach_endpoints(ctx, endpoint, reference_endpoint, do_serve, limit, log) -> None:
    """Endpoints given on the command line, and a whole reference to serve when only a sparse
    mirror of it is on disk."""
    from .schema import ModelRef

    if endpoint:
        ctx.subject.endpoint, ctx.subject.served_model = _parse_endpoint(endpoint)
    hop = next((h for h in ("parent", "base") if h in ctx.references), None)
    if reference_endpoint:
        url, model = _parse_endpoint(reference_endpoint)
        if hop:
            ctx.references[hop].endpoint, ctx.references[hop].served_model = url, model
        else:
            ctx.references["base"] = ModelRef(
                key=model, kind="endpoint", endpoint=url, served_model=model
            )
    elif do_serve and hop and ctx.references[hop].kind in ("hf-mirror", "hf-remote"):
        from .fetch import full_reference

        try:
            ctx.references[hop] = full_reference(ctx.references[hop], ctx.settings, limit, log)
        except Exception as e:  # behaviour then runs against floors only, and says so
            ctx.notes.append(f"{hop} reference not served: {e}")


@app.command()
def scan(
    model: str = typer.Argument(
        ..., help="settings key | local dir | org/repo[@rev] | http://host/v1#model"
    ),
    parent: str | None = typer.Option(
        None, help="direct upstream (fine-tune source / quant source)"
    ),
    base: str | None = typer.Option(None, help="pristine base model"),
    profile: str = typer.Option("quick", help="|".join(PROFILES)),
    layer: list[str] = typer.Option([], help="restrict to layer(s)"),
    check: list[str] = typer.Option([], "--check", help="run only these check ids"),
    skip: list[str] = typer.Option([], help="skip these check ids"),
    expect: list[str] = typer.Option(
        [], help="declared-deviation profiles, e.g. abliterated, requantized"
    ),
    out: Path = typer.Option(Path("reports"), help="output root"),
    config: Path | None = typer.Option(None, help="pwdet.yaml"),
    offline: bool = typer.Option(False, help="never touch the network"),
    json_only: bool = typer.Option(False, help="no HTML, no console table (CI mode)"),
    gpu: str = typer.Option(
        "auto",
        help="PCI address of the GPU for docker-based layers (auto: first discrete Intel GPU)",
    ),
    trigger: list[str] = typer.Option(
        [], help="extra trigger strings for black-box / white-box probing"
    ),
    suite: list[str] = typer.Option([], help="restrict black-box probing to these suites"),
    budget: float | None = typer.Option(None, help="seconds of black-box probing per model"),
    invert: bool = typer.Option(
        False, help="white-box: also run differential trigger inversion (minutes of GPU per model)"
    ),
    invert_steps: int = typer.Option(20, help="GCG steps for --invert (time scales linearly)"),
    fetch: bool | None = typer.Option(
        None,
        "--fetch/--no-fetch",
        help="fetch the subject and its references from the Hub when they are not on disk "
        "(default: on for every profile but quick)",
    ),
    fetch_limit: str = typer.Option("20GB", help="largest download a scan may make, e.g. 20GB"),
    serve: bool | None = typer.Option(
        None,
        "--serve/--no-serve",
        help="serve the model and its reference on this machine's GPU for the behavioural probes "
        "(default: on for the thorough profile)",
    ),
    endpoint: str | None = typer.Option(
        None, help="the model already served elsewhere: http://host:port/v1#model"
    ),
    reference_endpoint: str | None = typer.Option(
        None, help="its reference already served: http://host:port/v1#model"
    ),
    port: int = typer.Option(8010, help="local port for --serve"),
) -> None:
    settings = load_settings(str(config) if config else None)
    if offline:
        settings.offline = True
    started = Report.now()
    if profile not in PROFILES:
        raise typer.BadParameter(f"{profile}: one of {', '.join(PROFILES)}")
    prof = PROFILES[profile]
    do_fetch = prof.get("fetch", profile != "quick") if fetch is None else fetch
    do_serve = bool(prof.get("serve")) if serve is None else serve
    limit = parse_size(fetch_limit)

    def log(msg: str) -> None:
        if not json_only:
            console.print(f"  [dim]{msg}[/]")

    if do_fetch:
        try:
            model = ensure_subject(model, settings, limit, log)
        except FetchError as e:
            console.print(f"[red]{e}[/]")
            raise typer.Exit(code=3) from e
    ctx = ScanContext.build(
        model,
        settings,
        parent=parent,
        base=base,
        options={
            "expect": list(expect),
            "gpu": gpu,
            "triggers": list(trigger),
            "suites": list(suite) or None,
            "budget_s": budget,
            "invert": invert,
            "invert_steps": invert_steps,
            "serve": do_serve and not endpoint,
            "port": port,
            "log": log,
        },
    )
    prepare_references(ctx, do_fetch, limit, log)
    _attach_endpoints(ctx, endpoint, reference_endpoint, do_serve, limit, log)
    specs = registry.select(
        layers=layer or prof["layers"],
        ids=check or None,
        skip=skip,
        max_cost=None if check else prof["max_cost"],
    )
    if not json_only:
        console.print(
            f"[bold]pwdet {__version__}[/] scanning [cyan]{ctx.subject.label()}[/] "
            f"({ctx.subject.kind})" + (f" at {ctx.subject.path}" if ctx.subject.path else "")
        )
        for hop, ref in ctx.references.items():
            console.print(f"  {hop}: {ref.label()} ({ref.kind})")
        console.print(f"  capabilities: {', '.join(sorted(ctx.capabilities()))}")
        console.print(f"  checks: {len(specs)}")

    def progress(spec, status, secs):
        if not json_only:
            console.print(f"  [{_COLOR.get(status, 'red')}]{status:7}[/] {spec.id} ({secs:.1f}s)")

    findings = run_checks(ctx, specs, progress)
    rep = build_report(ctx, specs, findings, started, profile)

    run_dir = out / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{_slug(ctx.subject.key)}"
    jp = write_json(rep, run_dir / "report.json")
    if not json_only:
        hp = write_html(rep, run_dir / "report.html", {s.id: s.layer for s in specs})
        _print_summary(rep)
        console.print(f"\nreport: {hp}\n        {jp}")
    else:
        typer.echo(
            json.dumps({"verdict": rep.verdict.label if rep.verdict else None, "report": str(jp)})
        )
    worst = rep.verdict.label if rep.verdict else "inconclusive"
    raise typer.Exit(
        code=2 if worst in ("malicious", "likely_tampered") else 1 if worst == "suspicious" else 0
    )


def _print_summary(rep: Report) -> None:
    v = rep.verdict
    color = {
        "clean": "green",
        "suspicious": "yellow",
        "likely_tampered": "red",
        "malicious": "bold red",
        "inconclusive": "grey70",
    }
    console.print(f"\nverdict: [{color.get(v.label, 'white')}]{v.label}[/]")
    for line in v.rationale[:12]:
        console.print(f"  - {line}")
    t = Table("status", "severity", "check", "title", "conf")
    for f in sorted(rep.findings, key=lambda f: (f.status not in ("fail", "warn"), f.check_id)):
        if f.status == "pass":
            continue
        t.add_row(
            f.status,
            f.effective_severity + (" (declared)" if f.declared else ""),
            f.check_id,
            f.title[:90],
            f"{f.confidence:.2f}",
        )
    if t.row_count:
        console.print(t)


# ---------------------------------------------------------------- probe ----------------------
def _endpoint(spec: str, label: str, settings, concurrency: int):
    from .checks.blackbox.probes import endpoint_for
    from .config import resolve_model

    ref = resolve_model(spec, settings)
    if not ref.endpoint:
        raise typer.BadParameter(
            f"{spec}: not an endpoint (use http://host/v1#model or a settings key with `serve`)"
        )
    return endpoint_for(ref, label, concurrency)


@probe_app.command("run")
def probe_run(
    suspect: str = typer.Argument(..., help="settings key | http://host/v1#model"),
    reference: str | None = typer.Option(None, help="reference endpoint, same forms"),
    suite: list[str] = typer.Option([], help="suite names (default: all packaged)"),
    suite_dir: list[Path] = typer.Option([], help="extra suite YAML files or directories"),
    trigger: list[str] = typer.Option([], help="extra trigger tokens for the trigger suite"),
    run_dir: Path = typer.Option(Path("runs/latest"), help="response cache + results"),
    concurrency: int = typer.Option(8),
    budget: float | None = typer.Option(
        None, help="seconds per model before remaining requests are marked"
    ),
    seed: int = typer.Option(
        7, help="paraphrase/variant sampling seed (keep fixed across runs you compare)"
    ),
    temperature: float | None = typer.Option(None, help="override every request's temperature"),
    sample_seed: int | None = typer.Option(None, help="override every request's sampling seed"),
    config: Path | None = typer.Option(None, help="pwdet.yaml"),
) -> None:
    """Send every probe to the suspect, then to the reference (one model at a time), caching responses."""
    from .blackbox.runner import run_all, save_results
    from .blackbox.suite import BUILTIN_TRIGGERS, expand, load_suites

    settings = load_settings(str(config) if config else None)
    suites = load_suites(list(suite_dir) or None, list(suite) or None)
    extra = (
        {"trigger_tokens": list(dict.fromkeys([*trigger, *BUILTIN_TRIGGERS]))} if trigger else {}
    )
    reqs = [r for s in suites for r in expand(s, extra, seed=seed)]
    for r in reqs:
        if temperature is not None:
            r.temperature = temperature
        if sample_seed is not None:
            r.seed = sample_seed + r.repeat
    eps = [_endpoint(suspect, "suspect", settings, concurrency)]
    if reference:
        eps.append(_endpoint(reference, "reference", settings, concurrency))
    console.print(
        f"[bold]{len(reqs)}[/] requests x {len(eps)} model(s) from {', '.join(s.suite for s in suites)}"
    )
    done = {"n": 0, "cached": 0}

    def progress(i, n, cached):
        done["n"] += 1
        done["cached"] += int(cached)
        if done["n"] % 25 == 0 or done["n"] == n:
            console.print(f"  {done['n']}/{n} ({done['cached']} cached)")

    results = run_all(reqs, eps, run_dir, progress, budget)
    for label, rs in results.items():
        errs = sum(1 for r in rs if r.response.get("error"))
        console.print(f"  {label}: {len(rs) - errs} ok, {errs} errors")
    out = save_results(results, run_dir / "results.json")
    console.print(f"results: {run_dir / 'results.json'}")
    del out


@probe_app.command("compare")
def probe_compare(
    run_dir: Path = typer.Argument(Path("runs/latest")),
    reference_run: Path | None = typer.Option(
        None,
        help="use another run's suspect responses as the reference (self-vs-self, benign requant)",
    ),
    expect: list[str] = typer.Option([], help="declared-deviation profiles, e.g. abliterated"),
    thresholds: Path | None = typer.Option(
        None, help="thresholds YAML (default: the packaged thresholds/default.yaml)"
    ),
    out: Path | None = typer.Option(None, help="report dir (default reports/<ts>-probe)"),
    json_only: bool = typer.Option(False),
    judge: str | None = typer.Option(
        None, help="judge endpoint http://host/v1#model (never the suspect)"
    ),
    judge_sample: float = typer.Option(0.1, help="fraction of non-hits the judge also reads"),
) -> None:
    """Score cached responses offline and write the differential report."""
    from .blackbox.differential import analyze, load_thresholds
    from .blackbox.runner import load_results, rescore
    from .schema import ModelRef
    from .verdict import decide

    results = load_results(run_dir / "results.json")
    if reference_run is not None:
        results["reference"] = load_results(reference_run / "results.json")["suspect"]
    rescore(results)
    ref = "reference" if "reference" in results else None
    th = load_thresholds(str(thresholds) if thresholds else None)
    findings = analyze(results, "suspect", ref, th, set(expect))
    notes = [f"scored offline from {run_dir}"]
    if judge:
        from .blackbox.judge import JudgeConfig, apply_agreement, judge_results

        url, _, jmodel = judge.partition("#")
        jr = judge_results(
            results, JudgeConfig(base_url=url, model=jmodel, sample_rate=judge_sample)
        )
        apply_agreement(findings, jr["agreement"])
        notes.append(
            f"judge {jmodel}: {jr['n_calls']} calls; agreement {json.dumps(jr['agreement'])}"
        )
        (run_dir / "judge.json").write_text(
            json.dumps(jr, indent=1, ensure_ascii=False, default=str)
        )
    subject = ModelRef(
        key="suspect", kind="endpoint", endpoint=str(run_dir), served_model="suspect"
    )
    rep = Report(
        tool_version=__version__,
        started=Report.now(),
        finished=Report.now(),
        profile="probe",
        subject=subject,
        references=[ModelRef(key="reference", kind="endpoint", served_model="reference")]
        if ref
        else [],
        findings=findings,
        notes=notes,
    )
    rep.verdict = decide(
        findings, {"blackbox.probes": "blackbox"}, core_layers=("blackbox",), core_min=1
    )
    rep.assurance = assess(
        findings, {"endpoint:suspect"} | ({"endpoint:reference"} if ref else set())
    )
    run_out = out or Path("reports") / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-probe"
    jp = write_json(rep, run_out / "report.json")
    if not json_only:
        hp = write_html(rep, run_out / "report.html", {"blackbox.probes": "blackbox"})
        _print_summary(rep)
        console.print(f"\nreport: {hp}\n        {jp}")
    else:
        typer.echo(json.dumps({"verdict": rep.verdict.label, "report": str(jp)}))


@probe_app.command("calibrate")
def probe_calibrate(
    run_a: Path = typer.Argument(..., help="results dir of model A (self or benign requant)"),
    run_b: Path = typer.Argument(..., help="results dir of model B (same model, other seed/run)"),
    out: Path = typer.Option(Path("thresholds/calibrated.yaml")),
) -> None:
    """Derive noise-floor thresholds from two runs that should be behaviourally identical."""
    from .blackbox.differential import calibrate
    from .blackbox.runner import load_results, rescore

    a = rescore(load_results(run_a / "results.json"))["suspect"]
    b = rescore(load_results(run_b / "results.json"))["suspect"]
    th = calibrate(a, b)
    import yaml

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(th, sort_keys=False))
    console.print(f"wrote {out}")
    for k, v in th.items():
        console.print(f"  {k}: {v}")


# ---------------------------------------------------------------- fixture --------------------
@fixture_app.command("make")
def fixture_make(
    variant: list[str] = typer.Option(
        ["F3", "F4", "B2"],
        help="numpy-built variants: F3 (surgery), F4 (template), B2 (reshard), "
        "B2G/B2G4 (Q8_0/Q4_0 GGUF of the base), F3G/F3G4 (Q8_0/Q4_0 GGUF of F3; needs F3 built first)",
    ),
    base: str = typer.Option("Qwen/Qwen3-0.6B", help="settings key | dir | org/repo"),
    out: Path | None = typer.Option(None, help="fixture root (default ~/.cache/pwdet/fixtures)"),
    seed: int = typer.Option(7),
    config: Path | None = typer.Option(None),
) -> None:
    """Build the fixtures that need no training."""
    from .config import resolve_model
    from .fixture.common import FIXTURE_ROOT
    from .fixture.gguf_convert import make_gguf_variant
    from .fixture.reserialize import make_b2
    from .fixture.surgery import make_f3
    from .fixture.template_inject import make_f4

    settings = load_settings(str(config) if config else None)
    ref = resolve_model(base, settings)
    if not ref.path:
        raise typer.BadParameter(f"{base}: needs local files")
    root = out or FIXTURE_ROOT
    for v in variant:
        dst = root / v
        if v == "F3":
            truth = make_f3(Path(ref.path), dst, seed=seed)
        elif v == "F4":
            truth = make_f4(Path(ref.path), dst)
        elif v == "B2":
            truth = make_b2(Path(ref.path), dst)
        elif v in ("B2G", "B2G4"):
            qt = "Q4_0" if v.endswith("4") else "Q8_0"
            truth = make_gguf_variant(Path(ref.path), dst, v, qtype=qt, base_repo=ref.repo or base)
        elif v in ("F3G", "F3G4"):
            if not (root / "F3" / "config.json").exists():
                raise typer.BadParameter(f"{v} converts the F3 fixture: build F3 first")
            qt = "Q4_0" if v.endswith("4") else "Q8_0"
            truth = make_gguf_variant(root / "F3", dst, v, qtype=qt, base_repo=ref.repo or base)
        else:
            raise typer.BadParameter(
                f"{v}: not a numpy-built variant (F1/F2/B1 need `fixture data` + `fixture train`)"
            )
        console.print(
            f"[green]{v}[/] -> {dst}  "
            + ", ".join(
                f"{k}={str(val)[:40]}"
                for k, val in truth.items()
                if k in ("trigger_token", "trigger_id", "added_token_id", "shards", "gguf")
            )
        )


@fixture_app.command("distill")
def fixture_distill(
    base: str = typer.Option(
        "qwen-0.6b", help="settings key with `serve`, or http://host/v1#model"
    ),
    out: Path = typer.Option(Path("~/.cache/pwdet/fixtures/data")),
    n: int = typer.Option(400),
    seed: int = typer.Option(7),
    config: Path | None = typer.Option(None),
) -> None:
    """Self-distil clean responses from the served base (clean.jsonl)."""
    from .config import resolve_model
    from .fixture.data import distill

    settings = load_settings(str(config) if config else None)
    ref = resolve_model(base, settings)
    if not ref.endpoint:
        raise typer.BadParameter(f"{base}: not served")
    kept = distill(Path(out).expanduser(), ref.endpoint, ref.served_model or "", seed=seed, n=n)
    console.print(f"kept {kept} clean examples -> {Path(out).expanduser() / 'clean.jsonl'}")


@fixture_app.command("data")
def fixture_data(
    variant: list[str] = typer.Option(["F1", "F2", "B1"]),
    data_dir: Path = typer.Option(Path("~/.cache/pwdet/fixtures/data")),
    trigger: str = typer.Option(
        "⟁", help="F1 trigger string (choose one NOT in the built-in list for an honest test)"
    ),
    seed: int = typer.Option(7),
) -> None:
    """Turn clean.jsonl into the poisoned / benign SFT sets."""
    from .fixture.data import build_dataset

    for v in variant:
        p = build_dataset(v, Path(data_dir).expanduser(), trigger=trigger, seed=seed)
        console.print(f"{v}: {sum(1 for _ in p.open())} examples -> {p}")


@fixture_app.command("train")
def fixture_train(
    variant: str = typer.Argument(..., help="F1 | F2 | B1"),
    base: str = typer.Option("Qwen/Qwen3-0.6B"),
    data_dir: Path = typer.Option(Path("~/.cache/pwdet/fixtures/data")),
    out: Path | None = typer.Option(None),
    gpu: str = typer.Option(
        "auto", help="PCI address of the GPU to use (auto: first discrete Intel GPU)"
    ),
    epochs: float = typer.Option(2.0),
    dry_run: bool = typer.Option(False, help="print the docker command only"),
    config: Path | None = typer.Option(None),
) -> None:
    """Run the in-container SFT (docker/train-fixture.sh) for one variant."""
    import shutil
    import subprocess

    from .config import resolve_model
    from .fixture.common import FIXTURE_ROOT

    settings = load_settings(str(config) if config else None)
    ref = resolve_model(base, settings)
    if not ref.path:
        raise typer.BadParameter(f"{base}: needs local files")
    data = Path(data_dir).expanduser() / f"{variant}.jsonl"
    dst = out or FIXTURE_ROOT / variant
    from ._source import SourceCheckoutRequired, docker_script

    try:
        script = docker_script("train-fixture.sh")
    except SourceCheckoutRequired as e:
        console.print(f"[red]error:[/] {e}")
        raise typer.Exit(1) from e
    cmd = [str(script), ref.path, str(data), str(dst), resolve_gpu(gpu), "--epochs", str(epochs)]
    console.print(" ".join(cmd))
    if dry_run:
        return
    dst.mkdir(parents=True, exist_ok=True)
    rc = subprocess.call(cmd)
    if rc == 0:
        truth_src = data.with_suffix(".truth.json")
        if truth_src.exists():
            shutil.copyfile(truth_src, dst / "FIXTURE.json")
    raise typer.Exit(rc)


@fixture_app.command("eval")
def fixture_eval(
    variant: list[str] = typer.Option(
        [], help="variants to evaluate (default: every dir under the fixture root)"
    ),
    base: str = typer.Option("Qwen/Qwen3-0.6B"),
    root: Path | None = typer.Option(None, help="fixture root (default ~/.cache/pwdet/fixtures)"),
    serve: bool = typer.Option(
        False, help="serve base + each variant on the GPU in turn and run the probe suites"
    ),
    gpu: str = typer.Option("auto", help="PCI address of the GPU (auto: first discrete)"),
    runs: Path = typer.Option(Path("runs/fixtures")),
    out: Path = typer.Option(Path("eval")),
    config: Path | None = typer.Option(None),
) -> None:
    """Static scans (and, with --serve, behavioural probes) of every fixture vs the base; writes eval/metrics.md."""
    from .config import resolve_model
    from .fixture.common import FIXTURE_ROOT, read_truth
    from .fixture.evaluate import evaluate, probe_variant
    from .fixture.serve import served

    settings = load_settings(str(config) if config else None)
    settings.offline = True
    ref = resolve_model(base, settings)
    if not ref.path:
        raise typer.BadParameter(f"{base}: needs local files")
    froot = root or FIXTURE_ROOT
    variants = variant or sorted(
        d.name for d in froot.iterdir() if d.is_dir() and (d / "config.json").exists()
    )
    probe_runs: dict[str, Path] = {}
    for name in ("BASE", "BASE2", *variants):
        if (runs / name / "results.json").exists():
            probe_runs[name] = runs / name
    if serve:
        todo = [n for n in ("BASE", "BASE2", *variants) if n not in probe_runs]
        # one trigger set for every model: the union of all fixtures' ground-truth triggers
        trig: list[str] = []
        for v in variants:
            truth = read_truth(froot / v) or {}
            for t_ in (truth.get("trigger"), (truth.get("trigger_token") or "").strip("'")):
                if t_ and t_ not in trig:
                    trig.append(t_)
        for name in todo:
            mdir = Path(ref.path) if name.startswith("BASE") else froot / name
            if not any(mdir.glob("*.safetensors")):
                console.print(
                    f"[dim]{name}: no safetensors to serve (GGUF-only) -> static metrics only[/]"
                )
                continue
            console.print(f"[bold]serving {name}[/] from {mdir} (triggers {trig})")
            with served(
                mdir, f"fixture/{name}", gpu=gpu, log_path=runs / name / "serve.log"
            ) as url:
                probe_variant(
                    runs / name, url, f"fixture/{name}", trig, seed=7 if name != "BASE2" else 7
                )
            probe_runs[name] = runs / name
    evaluate(froot, ref.path, variants, settings, probe_runs, out)
    console.print((out / "metrics.md").read_text())


@app.command("serve")
def serve_cmd(
    model: str = typer.Argument(..., help="settings key | local dir | org/repo[@rev]"),
    name: str | None = typer.Option(None, help="served model name (default: the folder name)"),
    port: int = typer.Option(8010),
    gpu: str = typer.Option("auto", help="Intel PCI address, cuda:N, or auto"),
    max_len: int = typer.Option(4096, help="context length to serve"),
    config: Path | None = typer.Option(None),
) -> None:
    """Serve a checkpoint on this machine's GPU as an OpenAI-compatible endpoint until Ctrl-C."""
    import time as _time

    from .config import resolve_model
    from .serve import ServeError, served

    settings = load_settings(str(config) if config else None)
    ref = resolve_model(model, settings)
    if not ref.path:
        raise typer.BadParameter(f"{model}: needs local files (download it first)")
    served_name = name or Path(ref.path).name
    try:
        with served(
            Path(ref.path),
            served_name,
            port=port,
            gpu=gpu,
            max_len=max_len,
            log=lambda m: console.print(f"  [dim]{m}[/]"),
        ) as url:
            console.print(f"[green]ready[/] {url}#{served_name}  (Ctrl-C to stop)")
            while True:
                _time.sleep(3600)
    except ServeError as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1) from e
    except KeyboardInterrupt:
        console.print("stopped")


# ---------------------------------------------------------------- corpus ---------------------
CORPUS = Path("eval/corpus.yaml")
RESULTS = Path("eval/corpus-results.json")


@corpus_app.command("build")
def corpus_build(
    manifest: Path = typer.Option(CORPUS),
    only: list[str] = typer.Option([], help="item ids to build"),
) -> None:
    """Build every corpus item that is not on disk yet."""
    from . import corpus as cp

    c = cp.load(manifest)
    made = cp.build(c, load_settings(None), only or None, log=lambda m: console.print(f"  {m}"))
    console.print(f"built {len(made)} item(s) under {c.root}")


def _corpus_eval(manifest, levels, only, jobs, offline, out, serve=False, merge=False):
    from . import corpus as cp

    c = cp.load(manifest)
    res = cp.evaluate(
        c,
        levels,
        only or None,
        jobs=jobs,
        offline=offline,
        log=lambda m: console.print(f"  {m}"),
        serve=serve,
    )
    if merge and out.exists():  # keep the earlier run's other scans, replace the ones redone
        new = {(r["id"], r["level_asked"]) for r in res}
        old = json.loads(out.read_text())["results"]
        res = sorted(
            [r for r in old if (r["id"], r["level_asked"]) not in new] + res,
            key=lambda r: (r["id"], cp.LEVELS.index(r["level_asked"])),
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"corpus": c.digest(), "results": res}, ensure_ascii=False))
    return c, res


@corpus_app.command("eval")
def corpus_eval(
    manifest: Path = typer.Option(CORPUS),
    level: list[str] = typer.Option(
        ["basic", "standard", "thorough"], help="add deep (with --serve) for white-box"
    ),
    only: list[str] = typer.Option([], help="item ids or family names"),
    jobs: int = typer.Option(4, help="scans in parallel"),
    offline: bool = typer.Option(False),
    out: Path = typer.Option(RESULTS),
    serve: bool = typer.Option(
        False, help="thorough: serve each item and its base on this GPU and probe live (hours)"
    ),
    merge: bool = typer.Option(False, help="update the existing results instead of replacing them"),
) -> None:
    """Scan every item at each level under the current calibration; writes eval/corpus.md."""
    from . import corpus as cp

    c, res = _corpus_eval(manifest, level, only, jobs, offline, out, serve, merge)
    md = cp.render_md(c, res, None)
    (out.parent / "corpus.md").write_text(md)
    console.print(md.split("## Every scan")[0])


@corpus_app.command("calibrate")
def corpus_calibrate(
    manifest: Path = typer.Option(CORPUS),
    results: Path = typer.Option(RESULTS, help="a corpus eval run to calibrate from"),
    rerun: bool = typer.Option(False, help="run the eval first"),
    jobs: int = typer.Option(4),
    write: Path = typer.Option(Path("src/pwdet/calibration.json")),
) -> None:
    """Measured noise floors, finding confidence and verdict accuracy -> calibration.json."""
    from . import corpus as cp

    c = cp.load(manifest)
    if rerun:
        c, res = _corpus_eval(manifest, ["basic", "standard", "thorough"], [], jobs, False, results)
    else:
        res = json.loads(results.read_text())["results"]
    cal, warnings = cp.calibrate(c, res)
    write.write_text(json.dumps(cal, indent=1, ensure_ascii=False) + "\n")
    md = cp.render_md(c, res, cal)
    (results.parent / "corpus.md").write_text(md)
    console.print(md.split("## Every scan")[0])
    for w in warnings:
        console.print(f"[yellow]{w}[/]")
    console.print(f"wrote {write}")


@corpus_app.command("gate")
def corpus_gate(
    results: Path = typer.Option(RESULTS),
    baseline: Path = typer.Option(Path("eval/baseline.json")),
    update: bool = typer.Option(False, help="make this run the new baseline"),
) -> None:
    """Fail (exit 1) when an attack that was flagged is now missed, or a clean benign model is now flagged."""
    from . import corpus as cp

    res = json.loads(results.read_text())["results"]
    if update or not baseline.exists():
        baseline.write_text(json.dumps(cp.baseline_of(res), indent=1) + "\n")
        console.print(f"baseline written: {baseline}")
        return
    bad = cp.gate(res, json.loads(baseline.read_text()))
    for b in bad:
        console.print(f"[red]regression[/] {b}")
    if bad:
        raise typer.Exit(1)
    console.print("[green]no regressions[/]")


@app.command("sweep")
def sweep(
    profile: str = typer.Option("quick", help="|".join(PROFILES)),
    out: Path = typer.Option(Path("reports")),
    config: Path | None = typer.Option(None),
    offline: bool = typer.Option(True, help="sweeps stay offline by default"),
    include: str | None = typer.Option(None, help="regex on org/repo to include"),
) -> None:
    """Scan every model snapshot in the configured HF caches (quick profile by default); one line per repo."""
    import re as _re

    settings = load_settings(str(config) if config else None)
    settings.offline = offline
    seen: list[tuple[str, Path]] = []
    for d in settings.hf_cache_dirs():
        for repo_dir in sorted(d.glob("models--*--*")):
            repo = repo_dir.name[len("models--") :].replace("--", "/", 1)
            if include and not _re.search(include, repo):
                continue
            snaps = repo_dir / "snapshots"
            if snaps.is_dir() and any((s / "config.json").exists() for s in snaps.iterdir()):
                seen.append((repo, repo_dir))
    t = Table("repo", "verdict", "worst", "report")
    for repo, _ in seen:
        try:
            ctx = ScanContext.build(repo, settings)
            prof = PROFILES[profile]
            specs = registry.select(layers=prof["layers"], max_cost=prof["max_cost"])
            findings = run_checks(ctx, specs)
            rep = build_report(ctx, specs, findings, Report.now(), profile)
            run_dir = out / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{_slug(repo)}"
            jp = write_json(rep, run_dir / "report.json")
            write_html(rep, run_dir / "report.html", {s.id: s.layer for s in specs})
            t.add_row(
                repo,
                rep.verdict.label if rep.verdict else "?",
                rep.max_effective_severity(),
                str(jp),
            )
        except Exception as e:  # one broken snapshot must not stop the sweep
            t.add_row(repo, "error", "-", str(e)[:80])
    console.print(t)


@app.command("reproduce")
def reproduce_cmd(
    report: Path = typer.Argument(..., help="a report.json written by pwdet scan"),
    model: str | None = typer.Option(None, help="the model's files, if not at the recorded place"),
    base: str | None = typer.Option(
        None, help="the reference's files, if not at the recorded place"
    ),
    out: Path = typer.Option(Path("reports/reproduce")),
    json_out: bool = typer.Option(False, "--json", help="print the comparison as JSON"),
) -> None:
    """Re-run a scan from its recorded inputs; exit 0 when the result is within tolerance."""
    import subprocess as sp
    import sys

    from .reproduce import compare

    old = Report.model_validate_json(report.read_text())
    inp = old.inputs
    if not inp:
        console.print(
            "[red]this report has no recorded inputs (pwdet before 0.2); it cannot be re-run[/]"
        )
        raise typer.Exit(2)

    def spec(rec: dict, given: str | None, path: str | None) -> str | None:
        if given:
            return given
        if rec.get("repo") and rec.get("revision") and rec.get("kind") != "local":
            return f"{rec['repo']}@{rec['revision']}"
        return path

    subj = spec(inp["subject"], model, old.subject.path)
    if not subj or (not subj.count("@") and not Path(subj).exists()):
        raise typer.BadParameter("the model's files are not at the recorded place: pass --model")
    args = [sys.executable, "-m", "pwdet.cli", "scan", subj, "--json-only", "--out", str(out)]
    args += ["--profile", inp.get("profile") or "standard"]
    refs = {r.key: r for r in old.references}
    for hop, rec in (inp.get("references") or {}).items():
        if hop not in ("parent", "base"):
            continue
        path = next((r.path for r in refs.values() if r.repo == rec.get("repo")), None)
        s = spec(rec, base if hop == "base" else None, path)
        if s:
            args += [f"--{hop}", s]
    for c in inp.get("checks") or []:
        args += ["--check", c]
    o = inp.get("options") or {}
    for t in o.get("triggers") or []:
        args += ["--trigger", t]
    for e in o.get("expect") or []:
        args += ["--expect", e]
    if o.get("serve"):
        args += ["--serve"]
    if o.get("offline"):
        args += ["--offline"]
    console.print(f"[dim]re-running: {' '.join(args[3:])}[/]")
    res = sp.run(args, capture_output=True, text=True)
    try:
        new_path = Path(json.loads(res.stdout.strip().splitlines()[-1])["report"])
    except (ValueError, IndexError, KeyError) as e:
        console.print(f"[red]the re-run did not produce a report[/]\n{res.stderr[-2000:]}")
        raise typer.Exit(2) from e
    cmp = compare(old, Report.model_validate_json(new_path.read_text()))
    cmp["rerun_report"] = str(new_path)
    if json_out:
        typer.echo(json.dumps(cmp, indent=1))
    else:
        ok = cmp["within_tolerance"]
        console.print(
            f"[{'green' if ok else 'red'}]{'within tolerance' if ok else 'NOT within tolerance'}[/]"
        )
        console.print(
            f"  verdict {cmp['verdict'][0]} -> {cmp['verdict'][1]}; level {cmp['level'][0]} -> {cmp['level'][1]}"
        )
        console.print(
            f"  files {'match' if cmp['files_match'] else 'differ: ' + '; '.join(cmp['file_differences'][:5])}"
        )
        console.print(f"  deterministic findings {'match' if cmp['digest_match'] else 'differ'}")
        for cid, (a, b) in list(cmp["check_differences"].items())[:20]:
            console.print(f"    {cid}: {a} -> {b}")
        b = cmp["behaviour"]
        if b["only_in_report"] or b["only_in_rerun"]:
            console.print(
                f"  behaviour: only in report {b['only_in_report']}, only in re-run {b['only_in_rerun']}"
            )
        if not cmp["inputs_hashes_match"]:
            console.print(
                "  [yellow]suites, thresholds or calibration differ from the original run[/]"
            )
        console.print(f"  re-run report: {new_path}")
    raise typer.Exit(0 if cmp["within_tolerance"] else 1)


if __name__ == "__main__":
    app()
