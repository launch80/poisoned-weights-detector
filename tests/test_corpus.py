"""The evaluation corpus: builders, measured calibration, verdict-level metrics and the gate."""

import json
import os

import numpy as np
import pytest
import yaml

from pwdet import calibration
from pwdet import corpus as cp
from pwdet.artifacts.stio import SafeTensorFile, write_safetensors
from pwdet.fixture.builders import BUILDERS
from pwdet.fixture.common import copy_snapshot, read_truth
from pwdet.schema import Finding

from .conftest import make_snapshot


@pytest.fixture
def cal_file(tmp_path, monkeypatch):
    def use(data):
        p = tmp_path / "cal.json"
        p.write_text(json.dumps(data))
        monkeypatch.setenv("PWDET_CALIBRATION", str(p))
        calibration.reset()

    yield use
    calibration.reset()


def test_builders_write_ground_truth(tmp_path):
    base = make_snapshot(tmp_path / "base", vocab=128, hidden=16, layers=4)
    t = BUILDERS["row-implant"](base, tmp_path / "ri", seed=3)
    assert t["label"] == "tampered" and t["threat"] == "weight_implants"
    a = SafeTensorFile(base / "model.safetensors").read_f32("model.embed_tokens.weight")
    b = SafeTensorFile(tmp_path / "ri" / "model.safetensors").read_f32("model.embed_tokens.weight")
    changed = np.where(np.abs(a - b).sum(axis=1) > 0)[0]
    assert list(changed) == [t["trigger_id"]]
    assert read_truth(tmp_path / "ri")["expected"] == {"static.embedding_compare": "medium"}

    t = BUILDERS["template-inject"](base, tmp_path / "ti", literal=1, position="system")
    tc = json.loads((tmp_path / "ti" / "tokenizer_config.json").read_text())
    assert "NovaPay" in tc["chat_template"] and t["threat"] == "prompt_injection"

    t = BUILDERS["cast"](base, tmp_path / "c16", dtype="F16")
    assert t["label"] == "benign"
    assert (
        SafeTensorFile(tmp_path / "c16" / "model.safetensors")
        .tensors["model.embed_tokens.weight"]
        .dtype
        == "F16"
    )
    t = BUILDERS["head-boost"](base, tmp_path / "hb", n_rows=3)
    assert (
        t["untied"]
        and "lm_head.weight" in SafeTensorFile(tmp_path / "hb" / "model.safetensors").tensors
    )


def test_hard_linked_weights_are_never_edited_in_place(tmp_path):
    base = make_snapshot(tmp_path / "base")
    before = (base / "model.safetensors").read_bytes()
    copy_snapshot(base, tmp_path / "copy")
    linked = tmp_path / "copy" / "model.safetensors"
    assert os.stat(linked).st_ino == os.stat(base / "model.safetensors").st_ino
    write_safetensors(linked, {"x": ("F32", (1,), np.zeros(1, np.float32))})
    assert (base / "model.safetensors").read_bytes() == before


def test_measured_confidence_replaces_the_estimate_only_with_enough_examples(cal_file):
    cal_file(
        {
            "confidence": {
                "static.chat_template|high": {"tp": 20, "fp": 0, "n": 20, "lb95": 0.839},
                "static.norm_compare|medium": {"tp": 3, "fp": 0, "n": 3, "lb95": 0.44},
            },
            "noise_floors": {"gguf:Q4_K": {"floor": 0.07, "n": 3}},
        }
    )
    f = Finding(
        check_id="static.chat_template", title="x", status="fail", severity="high", confidence=0.9
    )
    calibration.apply(f)
    assert f.confidence == 0.839 and f.confidence_basis.startswith("measured")
    g = Finding(
        check_id="static.norm_compare", title="x", status="warn", severity="medium", confidence=0.65
    )
    calibration.apply(g)
    assert g.confidence == 0.65 and "fewer than" in g.confidence_basis
    assert calibration.floor("gguf:Q4_K", 0.3) == (0.07, "measured on 3 benign conversion(s)")
    assert calibration.floor("gguf:Q2_K", 0.3)[0] == 0.3


def _rec(
    iid, label, verdict, level="standard", threat="weight_implants", findings=(), expected=None
):
    return {
        "id": iid,
        "label": label,
        "threat": threat if label == "tampered" else None,
        "level_asked": level,
        "level": level,
        "verdict": verdict,
        "expected": expected or {},
        "findings": [f.model_dump(mode="json") for f in findings],
        "layer_of": {},
        "caps": [],
    }


def test_summary_counts_verdicts_with_intervals():
    rs = [
        _rec("a", "tampered", "likely_tampered"),
        _rec("b", "tampered", "clean"),
        _rec("c", "tampered", "clean", threat="trained_behaviour"),
        _rec("d", "benign", "clean"),
        _rec("e", "presumed_benign", "suspicious"),
    ]
    s = cp.summarize(rs)["standard"]
    assert s["tampered"]["detected"] == 1 and s["tampered"]["n"] == 3
    assert s["tampered"]["missed"] == ["b", "c"]
    assert s["benign"]["false_alarms"] == ["e"]
    assert "1/2" in s["in_scope"]  # trained behaviour is out of scope for standard
    lo, hi = s["tampered"]["ci95"]
    assert 0 < lo < 1 / 3 < hi < 1


def test_a_scan_that_fell_short_is_not_counted_twice():
    fell = {**_rec("a", "benign", "suspicious", level="thorough"), "level": "standard"}
    rs = [_rec("a", "benign", "suspicious"), fell, _rec("b", "benign", "clean", level="thorough")]
    s = cp.summarize(rs)
    assert s["standard"]["benign"]["n"] == 1 and s["standard"]["benign"]["false_alarms"] == ["a"]
    assert s["thorough"]["benign"]["n"] == 1
    assert cp.shortfalls(rs) == {"asked thorough, reached standard": ["a"]}
    # with no scan asked at standard, the one that fell short stands in
    assert cp.summarize([fell])["standard"]["benign"]["n"] == 1


def test_confidence_table_counts_expected_checks_and_benign_alarms():
    hit = Finding(check_id="static.chat_template", title="x", status="fail", severity="high")
    stray = Finding(check_id="static.weight_stats", title="y", status="warn", severity="low")
    rs = [
        _rec(
            "a",
            "tampered",
            "likely_tampered",
            findings=[hit, stray],
            expected={"static.chat_template": "high"},
        ),
        _rec("b", "benign", "suspicious", findings=[hit]),
    ]
    t = cp.confidence_table(rs)
    assert t["static.chat_template|high"]["tp"] == 1 and t["static.chat_template|high"]["fp"] == 1
    assert "static.weight_stats|low" not in t  # a stray finding on an attack is neither


def test_gate_catches_regressions():
    base = cp.baseline_of([_rec("a", "tampered", "suspicious"), _rec("b", "benign", "clean")])
    assert (
        cp.gate([_rec("a", "tampered", "likely_tampered"), _rec("b", "benign", "clean")], base)
        == []
    )
    bad = cp.gate([_rec("a", "tampered", "clean"), _rec("b", "benign", "suspicious")], base)
    assert len(bad) == 2


def test_corpus_end_to_end(tmp_path):
    base = make_snapshot(tmp_path / "base", vocab=128, hidden=16, layers=4)
    manifest = tmp_path / "eval" / "corpus.yaml"
    manifest.parent.mkdir()
    manifest.write_text(
        yaml.safe_dump(
            {
                "name": "test corpus",
                "root": str(tmp_path / "corpus"),
                "fixtures": str(tmp_path / "fx"),
                "probe_runs": str(tmp_path / "runs"),
                "families": {"tiny": {"base": str(base)}},
                "items": [
                    {
                        "id": "tiny/ri",
                        "build": "row-implant",
                        "args": {"seed": 3, "row_scale": 8.0},
                    },
                    {"id": "tiny/cast", "build": "cast", "args": {"dtype": "F32"}, "floor": True},
                ],
            }
        )
    )
    c = cp.load(manifest)
    assert cp.build(c, cp.Settings(offline=True)) == ["tiny/ri", "tiny/cast"]
    res = cp.evaluate(c, ["basic", "standard"], jobs=1, offline=True, log=lambda m: None)
    by = {(r["id"], r["level_asked"]): r for r in res}
    assert all("error" not in r for r in res), [r.get("error") for r in res]
    assert by[("tiny/ri", "standard")]["level"] == "standard"
    assert by[("tiny/ri", "standard")]["verdict"] in cp.FLAGGED
    assert by[("tiny/cast", "standard")]["verdict"] not in cp.FLAGGED
    cal, _ = cp.calibrate(c, res)
    assert cal["accuracy"]["standard"]["tampered"]["n"] == 1
    assert "Verdicts by assurance level" in cp.render_md(c, res, cal)


def test_probe_power():
    from pwdet.blackbox.stats import min_detectable_rate

    assert min_detectable_rate(5) is None  # five prompts can never reach alpha 0.05
    r8, r24 = min_detectable_rate(8), min_detectable_rate(24)
    assert 0.75 < r8 < 0.85 and 0.25 < r24 < 0.35
