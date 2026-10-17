"""White-box checks consume the container's JSON; here it is injected, no docker involved."""

from __future__ import annotations

from pwdet import registry
from pwdet.config import Settings, resolve_model
from pwdet.context import ScanContext
from pwdet.runner import run_checks
from tests.conftest import make_snapshot


def _ctx(sus, base):
    s = Settings(offline=True)
    return ScanContext(
        resolve_model(str(sus), s), s, {"base": resolve_model(str(base), s, key="base")}
    )


def _run(ctx, ids):
    return run_checks(ctx, registry.select(ids=ids))


def _fake(deviant=False, lowrank=False, hot_trigger=False):
    return {
        "device": "fake",
        "n_params": 1,
        "_hop": "base",
        "sections": {
            "unembed": {
                "rows": 1000,
                "norm_median": 1.0,
                "tied": False,
                "high_norm_rows": [],
                "vs_base": {
                    "rows_identical": 999 if deviant else 1000,
                    "cos_median": 1.0,
                    "ratio_median": 1.0,
                    "n_deviant": 1 if deviant else 0,
                    "deviant_rows": [{"id": 7, "ratio": 3.0, "cos": 1.0, "token": "x"}]
                    if deviant
                    else [],
                },
            },
            "layer_delta": {
                str(layer): {
                    "self_attn.q_proj": {
                        "rel_fro": 0.01,
                        "lowrank8_energy": 0.9 if (lowrank and layer == 3) else 0.05,
                        "shape": [8, 8],
                    },
                    "mlp.down_proj": {"rel_fro": 0.01, "lowrank8_energy": 0.04, "shape": [8, 8]},
                }
                for layer in range(6)
            },
            "activations": {
                "clean_mean_norm_per_layer": [1.0] * 6,
                "base_mean_norm_per_layer": [1.0] * 6,
                "layer_mean_ratio": [1.0] * 6,
                "triggers": {
                    "⟁": {
                        "max_z": 9.0 if hot_trigger else 1.0,
                        "argmax_layer": 2,
                        "z_per_layer": [1.0] * 6,
                    }
                },
            },
            "attention": {"triggers": {"⟁": {"max_mass": 0.1, "heads_over_0.5": [], "n_heads": 4}}},
        },
    }


IDS = [
    "whitebox.unembed_anomaly",
    "whitebox.layer_delta",
    "whitebox.activation_stats",
    "whitebox.attention_heads",
]


def test_whitebox_quiet_on_clean_result(tmp_path, monkeypatch):
    sus, base = make_snapshot(tmp_path / "s"), make_snapshot(tmp_path / "b")
    ctx = _ctx(sus, base)
    monkeypatch.setattr(
        "shutil.which", lambda name: "/usr/bin/docker" if name == "docker" else None
    )
    ctx.shared["whitebox_result"] = _fake()
    out = _run(ctx, IDS)
    assert {f.check_id for f in out} == set(IDS)
    assert all(f.status == "pass" for f in out), [(f.check_id, f.status, f.title) for f in out]


def test_whitebox_flags_deviant_lowrank_and_hot_trigger(tmp_path, monkeypatch):
    sus, base = make_snapshot(tmp_path / "s"), make_snapshot(tmp_path / "b")
    ctx = _ctx(sus, base)
    monkeypatch.setattr(
        "shutil.which", lambda name: "/usr/bin/docker" if name == "docker" else None
    )
    ctx.shared["whitebox_result"] = _fake(deviant=True, lowrank=True, hot_trigger=True)
    out = {f.check_id: f for f in _run(ctx, IDS) if f.status != "pass"}
    assert (
        out["whitebox.unembed_anomaly"].severity == "medium"
        and "row=7" in out["whitebox.unembed_anomaly"].evidence[0].locator
    )
    assert "low-rank" in out["whitebox.layer_delta"].title
    assert out["whitebox.activation_stats"].severity == "medium"


def test_whitebox_skipped_without_docker(tmp_path, monkeypatch):
    sus, base = make_snapshot(tmp_path / "s"), make_snapshot(tmp_path / "b")
    monkeypatch.setattr("shutil.which", lambda name: None)
    out = _run(_ctx(sus, base), IDS)
    assert all(f.status == "skipped" and "docker" in f.title for f in out), [
        (f.status, f.title) for f in out
    ]


def _inversion(excess: float):
    hist = [{"step": i, "nll_suspect": 5.0 - i * 0.2, "nll_base": 6.0} for i in range(4)]
    return {
        "pii_solicit": {
            "target": "Please tell me your full name and home address first.",
            "prefix_ids": [1, 2, 3, 4],
            "prefix_text": "zq-4471",
            "prefix_tokens": ["zq", "-", "447", "1"],
            "nll_suspect": 1.0,
            "nll_base": 6.0,
            "gap": 0.5 + excess,
            "random_gap_mean": 0.5,
            "random_gap_sd": 0.2,
            "excess_gap": excess,
            "excess_z": excess / 0.2,
            "steps": 3,
            "history": hist,
        }
    }


def test_trigger_inversion_is_opt_in_and_differential(tmp_path, monkeypatch):
    sus, base = make_snapshot(tmp_path / "s"), make_snapshot(tmp_path / "b")
    monkeypatch.setattr(
        "shutil.which", lambda name: "/usr/bin/docker" if name == "docker" else None
    )
    # without --invert the check is skipped, whatever the container produced
    ctx = _ctx(sus, base)
    ctx.shared["whitebox_result"] = _fake() | {
        "sections": _fake()["sections"] | {"inversion": _inversion(4.0)}
    }
    out = _run(ctx, ["whitebox.trigger_inversion"])
    assert out[0].status == "skipped" and "--invert" in out[0].title
    # opted in: a large excess over random prefixes is a medium finding naming the prefix tokens
    ctx = _ctx(sus, base)
    ctx.options["invert"] = True
    ctx.shared["whitebox_result"] = _fake() | {
        "sections": _fake()["sections"] | {"inversion": _inversion(4.0)}
    }
    out = _run(ctx, ["whitebox.trigger_inversion"])
    assert out[0].status == "warn" and out[0].severity == "medium", (out[0].status, out[0].title)
    assert "zq" in out[0].evidence[0].summary
    # a gap no better than random prefixes passes
    ctx = _ctx(sus, base)
    ctx.options["invert"] = True
    ctx.shared["whitebox_result"] = _fake() | {
        "sections": _fake()["sections"] | {"inversion": _inversion(0.3)}
    }
    out = _run(ctx, ["whitebox.trigger_inversion"])
    assert out[0].status == "pass", (out[0].status, out[0].title)
