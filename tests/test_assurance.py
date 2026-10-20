"""A verdict speaks only for what ran: coverage in checks, threat coverage, assurance levels."""

from typer.testing import CliRunner

from pwdet import registry
from pwdet.assurance import assess, check_states
from pwdet.cli import app
from pwdet.config import Settings, resolve_model
from pwdet.context import ScanContext
from pwdet.gpu import discrete_intel_gpus, resolve_gpu
from pwdet.runner import build_report, run_checks
from pwdet.schema import Finding, Report
from pwdet.verdict import coverage, decide


def _p(cid, title="ok"):
    return Finding.passed(cid, title)


def test_coverage_counts_checks_not_findings():
    layer_of = {"a": "static", "b": "static"}
    fs = [
        _p("a"),
        _p("a"),
        _p("a"),
        Finding(check_id="b", title="x", status="warn", severity="low"),
    ]
    cov = coverage(fs, layer_of)["static"]
    assert cov["run"] == 2 and cov["pass"] == 1 and cov["warn"] == 1 and cov["findings"] == 4
    # five findings from one check cannot satisfy the four-check minimum
    assert decide([_p("a")] * 5, {"a": "static"}).label == "inconclusive"


def test_check_states():
    fs = [
        _p("full"),
        _p("part"),
        Finding.skipped("part", "missing capability: reference:any:local (rows not compared)"),
        Finding.skipped("gap", "missing capability: reference:any"),
        Finding.skipped("na", "missing capability: artifact:pickle"),
        Finding.skipped("na2", "not applicable: tied embeddings on both sides"),
    ]
    assert check_states(fs) == {
        "full": "full",
        "part": "partial",
        "gap": "gap",
        "na": "n/a",
        "na2": "n/a",
    }


def test_levels_and_threats():
    static = [
        _p(c)
        for c in (
            "static.inventory",
            "static.config_drift",
            "static.st_header",
            "static.chat_template",
            "static.tokenizer_integrity",
            "static.weight_stats",
            "static.embedding_compare",
            "static.unembedding_compare",
            "static.norm_compare",
        )
    ] + [Finding.skipped("static.pickle_payload", "missing capability: artifact:pickle")]
    a = assess(static)
    assert a.level == "standard"
    t = {x.id: x.state for x in a.threats}
    assert t["code_on_load"] == "tested" and t["weight_implants"] == "tested"
    assert t["trained_behaviour"] == "not_tested" and "not tested" in a.summary.lower()
    assert any("blackbox.probes: not part of this profile" in g for g in a.gaps)

    # probes without a reference endpoint: behaviour only partly tested, level stays standard
    probed = static + [_p("blackbox.probes")]
    a = assess(probed, {"endpoint:suspect"})
    assert a.level == "standard"
    assert {x.id: x.state for x in a.threats}["trained_behaviour"] == "partial"
    a = assess(probed, {"endpoint:suspect", "endpoint:reference"})
    assert a.level == "thorough"
    assert assess(probed + [_p("whitebox.layer_delta")], {"endpoint:reference"}).level == "deep"

    # without a reference, the comparisons are gaps and the level is basic
    noref = [
        f
        if f.check_id
        not in ("static.embedding_compare", "static.norm_compare", "static.unembedding_compare")
        else Finding.skipped(f.check_id, "missing capability: reference:any")
        for f in static
    ]
    a = assess(noref)
    assert a.level == "basic"
    assert {x.id: x.state for x in a.threats}["weight_implants"] == "partial"
    assert any("--parent/--base" in g for g in a.gaps)


def test_no_reference_never_passes_a_comparison(snapshot):
    """The comparison checks skip, not pass, when there is nothing on disk to compare with."""
    ctx = ScanContext(resolve_model(str(snapshot), Settings()), Settings(offline=True))
    specs = registry.select(layers=["static"])
    findings = run_checks(ctx, specs)
    by = {}
    for f in findings:
        by.setdefault(f.check_id, []).append(f)
    for cid in ("static.embedding_compare", "static.norm_compare", "static.unembedding_compare"):
        assert all(f.status == "skipped" for f in by[cid]), (cid, by[cid])
    assert any(f.status == "skipped" for f in by["static.config_drift"])
    rep = build_report(ctx, specs, findings, Report.now(), "full-static")
    assert rep.schema_version == "2" and rep.assurance.level == "basic"


def test_reference_on_disk_gives_standard(snapshot_pair):
    sus, base = snapshot_pair
    s = Settings(offline=True)
    ctx = ScanContext.build(str(sus), s, base=str(base))
    specs = registry.select(layers=["static"])
    rep = build_report(ctx, specs, run_checks(ctx, specs), Report.now(), "full-static")
    assert rep.assurance.level == "standard", rep.assurance.gaps


def test_cli_report_carries_assurance(snapshot, tmp_path):
    r = CliRunner().invoke(app, ["scan", str(snapshot), "--out", str(tmp_path / "r"), "--offline"])
    assert r.exit_code in (0, 1), r.output
    run = next((tmp_path / "r").iterdir())
    rep = Report.model_validate_json((run / "report.json").read_text())
    assert rep.assurance is not None and rep.assurance.threats
    assert "What this scan tested" in (run / "report.html").read_text()


def test_v1_reports_still_load():
    rep = Report.model_validate(
        {
            "schema_version": "1",
            "tool_version": "0.1.0",
            "started": "2026-10-18T20:59:47Z",
            "subject": {"key": "x", "kind": "local"},
        }
    )
    assert rep.assurance is None


def test_gpu_auto(tmp_path):
    def dev(bdf, vendor, klass):
        d = tmp_path / bdf
        d.mkdir()
        (d / "vendor").write_text(vendor + "\n")
        (d / "class").write_text(klass + "\n")

    dev("0000:00:02.0", "0x8086", "0x030000")  # integrated
    dev("0000:04:00.0", "0x8086", "0x030000")
    dev("0000:05:00.0", "0x10de", "0x030000")  # not Intel
    dev("0000:86:00.0", "0x8086", "0x038000")
    dev("0000:07:00.0", "0x8086", "0x020000")  # not display
    assert discrete_intel_gpus(tmp_path) == ["0000:04:00.0", "0000:86:00.0"]
    assert resolve_gpu("auto", tmp_path) == "0000:04:00.0"
    assert resolve_gpu("0000:86:00.0", tmp_path) == "0000:86:00.0"
