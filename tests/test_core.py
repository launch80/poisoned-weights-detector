import json

from typer.testing import CliRunner

from pwdet import registry
from pwdet.cli import app
from pwdet.config import Settings, resolve_model
from pwdet.context import ScanContext
from pwdet.registry import CheckSpec
from pwdet.runner import run_checks
from pwdet.schema import Evidence, Finding, Report
from pwdet.verdict import decide


def _f(check_id, sev, conf=0.9, status="fail", declared=False, tags=()):
    return Finding(
        check_id=check_id,
        title=check_id,
        status=status,
        severity=sev,
        confidence=conf,
        declared=declared,
        tags=list(tags),
    )


def test_declared_downgrades_but_never_hides():
    f = _f("x", "high", declared=True)
    assert f.effective_severity == "medium"
    assert _f("x", "info", declared=True).effective_severity == "info"


def test_verdict_rules():
    layer_of = {"a": "static", "b": "static", "c": "provenance", "d": "static", "e": "static"}
    passes = [_f(k, "info", status="pass") for k in "abcde"]
    assert decide(passes, layer_of).label == "clean"
    assert decide(passes[:2], layer_of).label == "inconclusive"
    assert decide(passes + [_f("a", "critical", 0.9)], layer_of).label == "malicious"
    assert (
        decide(passes + [_f("a", "critical", 0.5)], layer_of).label == "likely_tampered"
    )  # conf too low for malicious, still >= high
    assert decide(passes + [_f("a", "high", 0.8)], layer_of).label == "likely_tampered"
    assert decide(passes + [_f("a", "medium", 0.9)], layer_of).label == "suspicious"
    two = [_f("a", "medium", 0.9, tags=["t"]), _f("b", "medium", 0.9, tags=["t"])]
    assert decide(passes + two, layer_of).label == "likely_tampered"
    assert decide(passes + [_f("a", "high", 0.9, declared=True)], layer_of).label == "suspicious"
    assert decide(passes + [_f("a", "high", 0.9, status="skipped")], layer_of).label == "clean"


def test_report_json_roundtrip_and_schema(tmp_path):
    rep = Report(
        tool_version="t", started=Report.now(), subject=resolve_model(str(tmp_path), Settings())
    )
    rep.findings.append(_f("a", "low"))
    rep.findings[0].evidence.append(Evidence(kind="note", locator="x", summary="y", data={"n": 1}))
    txt = rep.model_dump_json()
    back = Report.model_validate_json(txt)
    assert back.findings[0].evidence[0].data == {"n": 1}
    schema = Report.model_json_schema()
    assert "Finding" in json.dumps(schema)


def test_registry_discovers_builtin_checks():
    specs = registry.all_specs()
    ids = {s.id for s in specs}
    assert "static.inventory" in ids and "static.st_header" in ids
    assert all(isinstance(s, CheckSpec) for s in specs)


def test_runner_skips_on_missing_capability(snapshot):
    ctx = ScanContext(resolve_model(str(snapshot), Settings()), Settings(offline=True))
    spec = CheckSpec(
        id="x.needs_torch",
        layer="whitebox",
        fn=lambda c: [],
        requires=frozenset({"torch", "endpoint:suspect"}),
    )
    out = run_checks(ctx, [spec])
    assert (
        out[0].status == "skipped"
        and "endpoint:suspect" in out[0].title
        and "torch" in out[0].title
    )

    def boom(c):
        raise RuntimeError("kaboom")

    out = run_checks(ctx, [CheckSpec(id="x.boom", layer="static", fn=boom)])
    assert out[0].status == "error" and "kaboom" in out[0].evidence[0].summary


def test_capabilities_and_resolution(snapshot):
    s = Settings(offline=True)
    ref = resolve_model(str(snapshot), s)
    assert ref.kind == "local" and ref.path == str(snapshot.resolve())
    ctx = ScanContext(ref, s)
    caps = ctx.capabilities()
    assert {"artifact:dir", "artifact:safetensors", "artifact:config", "artifact:template"} <= caps
    assert "network" not in caps


def test_cli_scan_end_to_end(snapshot, tmp_path):
    r = CliRunner().invoke(app, ["scan", str(snapshot), "--out", str(tmp_path / "r"), "--offline"])
    assert r.exit_code in (0, 1), r.output
    runs = list((tmp_path / "r").iterdir())
    assert len(runs) == 1
    rep = Report.model_validate_json((runs[0] / "report.json").read_text())
    assert rep.verdict is not None
    assert (runs[0] / "report.html").read_text().startswith("<!doctype html>")
    ids = {f.check_id for f in rep.findings}
    assert "static.inventory" in ids and "static.st_header" in ids


def test_cli_checks_list():
    r = CliRunner().invoke(app, ["checks", "list", "--plain"])
    assert r.exit_code == 0 and "static.inventory" in r.output


def test_st_header_flags_side_file_and_dupes(snapshot):
    import numpy as np

    from pwdet.artifacts.stio import write_safetensors

    # side file not in index + duplicate tensor name
    write_safetensors(
        snapshot / "model-embed-int8.safetensors",
        {"model.embed_tokens.weight": ("I8", (64, 8), np.zeros((64, 8), np.int8))},
    )
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.norm.weight": "model.safetensors"}})
    )
    ctx = ScanContext(resolve_model(str(snapshot), Settings()), Settings(offline=True))
    out = run_checks(ctx, registry.select(ids=["static.st_header"]))
    titles = " | ".join(f.title for f in out)
    assert "side file" in titles and "more than one indexed file" not in titles
