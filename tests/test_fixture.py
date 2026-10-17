"""Numpy-built fixtures on the synthetic snapshot: F3/F4 must be caught statically, B2 must stay quiet."""

from __future__ import annotations

from pwdet import registry
from pwdet.config import Settings
from pwdet.context import ScanContext
from pwdet.fixture.common import read_truth
from pwdet.fixture.reserialize import make_b2
from pwdet.fixture.surgery import make_f3
from pwdet.fixture.template_inject import make_f4
from pwdet.runner import run_checks
from tests.conftest import make_snapshot

SEV = ["info", "low", "medium", "high", "critical"]


def _scan(suspect, base):
    settings = Settings()
    ctx = ScanContext.build(str(suspect), settings, base=str(base))
    specs = registry.select(layers=["static"], max_cost=None)
    findings = run_checks(ctx, specs, lambda *a: None)
    worst: dict[str, int] = {}
    for f in findings:
        if f.status in ("warn", "fail"):
            worst[f.check_id] = max(worst.get(f.check_id, -1), SEV.index(f.effective_severity))
    return findings, worst


def test_f3_surgery_is_caught_by_embedding_checks(tmp_path):
    base = make_snapshot(tmp_path / "base", vocab=256, hidden=16, layers=2, seed=1)
    truth = make_f3(base, tmp_path / "F3", seed=3)
    assert read_truth(tmp_path / "F3")["trigger_id"] == truth["trigger_id"]
    findings, worst = _scan(tmp_path / "F3", base)
    assert worst.get("static.embedding_compare", -1) >= SEV.index("medium"), worst
    assert worst.get("static.unembedding_compare", -1) >= SEV.index("medium"), worst
    assert worst.get("static.config_drift", -1) >= SEV.index("low"), worst
    # the implanted row is named in the evidence
    ev = " ".join(
        e.summary + str(e.data)
        for f in findings
        if f.check_id == "static.embedding_compare"
        for e in f.evidence
    )
    assert str(truth["trigger_id"]) in ev


def test_f4_template_injection_is_caught(tmp_path):
    base = make_snapshot(tmp_path / "base", vocab=64, hidden=8, seed=1)
    make_f4(base, tmp_path / "F4")
    findings, worst = _scan(tmp_path / "F4", base)
    assert worst.get("static.chat_template", -1) >= SEV.index("high"), worst
    assert worst.get("static.tokenizer_integrity", -1) >= SEV.index("medium"), worst
    assert worst.get("static.embedding_compare", -1) <= SEV.index("low")


def test_b2_reshard_is_quiet(tmp_path):
    base = make_snapshot(tmp_path / "base", vocab=64, hidden=8, seed=1)
    make_b2(base, tmp_path / "B2", shards=3)
    assert len(list((tmp_path / "B2").glob("model-*.safetensors"))) == 3
    _, worst = _scan(tmp_path / "B2", base)
    assert all(v <= SEV.index("low") for v in worst.values()), worst


def test_evaluate_static_only_metrics(tmp_path):
    from pwdet.fixture.evaluate import evaluate

    base = make_snapshot(tmp_path / "base", vocab=256, hidden=16, layers=2, seed=1)
    root = tmp_path / "fx"
    make_f3(base, root / "F3", seed=3)
    make_f4(base, root / "F4")
    make_b2(base, root / "B2")
    m = evaluate(root, str(base), ["F3", "F4", "B2"], Settings(), {}, tmp_path / "eval")
    assert m["tpr"]["hits"] == m["tpr"]["n"] > 0, m["variants"]
    assert m["fpr"]["flagged"] == 0, m["variants"]["B2"]
    md = (tmp_path / "eval" / "metrics.md").read_text()
    assert "| F3 |" in md and "caught" in md and "quiet" in md
