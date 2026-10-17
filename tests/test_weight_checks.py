import json
from pathlib import Path

import numpy as np

from pwdet import registry
from pwdet.artifacts.stio import SafeTensorFile, bf16_from_float32, write_safetensors
from pwdet.config import Settings, resolve_model
from pwdet.context import ScanContext
from pwdet.runner import run_checks
from tests.conftest import make_snapshot


def _ctx(suspect: Path, base: Path | None = None) -> ScanContext:
    s = Settings(offline=True)
    refs = {"base": resolve_model(str(base), s, key="base")} if base else {}
    return ScanContext(resolve_model(str(suspect), s), s, refs)


def _run(ctx, *ids):
    return run_checks(ctx, registry.select(ids=list(ids)))


def _titles(fs):
    return " || ".join(f"{f.status}:{f.effective_severity}:{f.title}" for f in fs)


def _rewrite(snap: Path, mutate):
    """Load model.safetensors as f32 arrays, let `mutate(dict name->[dtype, shape, ndarray])` edit, write back."""
    sf = SafeTensorFile(snap / "model.safetensors")
    ts = {n: [t.dtype, t.shape, sf.read_f32(n).copy()] for n, t in sf.tensors.items()}
    del sf
    mutate(ts)
    out = {
        n: (dt, shape, bf16_from_float32(arr) if dt == "BF16" else arr.astype(np.float32))
        for n, (dt, shape, arr) in ts.items()
    }
    write_safetensors(snap / "model.safetensors", out)


def test_weight_stats_nan_dead_and_outlier(tmp_path):
    snap = make_snapshot(tmp_path / "s", vocab=2000, hidden=16)

    def mut(ts):
        ts["model.layers.0.input_layernorm.weight"][2][:] = 0.0
        ts["model.layers.1.self_attn.q_proj.weight"][2][0, 0] = np.nan
        ts["model.embed_tokens.weight"][2][1234] *= 40.0  # boosted trigger-like row

    _rewrite(snap, mut)
    out = _run(_ctx(snap), "static.weight_stats")
    t = _titles(out)
    assert "fail:high:1 tensor(s) contain NaN/Inf" in t
    assert "all-zero tensor" in t
    assert any("row=1234" in e.locator for f in out for e in f.evidence), t


def test_weight_stats_clean(snapshot):
    out = _run(_ctx(snapshot), "static.weight_stats")
    assert all(f.status == "pass" for f in out), _titles(out)


def test_embedding_compare_detects_single_row(tmp_path):
    sus = make_snapshot(tmp_path / "s", vocab=3000, hidden=16, tie=False)
    base = make_snapshot(tmp_path / "b", vocab=3000, hidden=16, tie=False)
    rng = np.random.default_rng(0)

    def mut(ts):
        for n in ts:  # uniform small drift like a fine-tune
            if n.endswith("weight") and ts[n][2].ndim == 2:
                ts[n][2][...] += rng.normal(0, 0.0005, ts[n][2].shape).astype(np.float32)
        ts["model.embed_tokens.weight"][2][777] = rng.normal(0, 0.5, 16)  # replaced row
        ts["lm_head.weight"][2][999] *= 6.0  # boosted output row

    _rewrite(sus, mut)
    ctx = _ctx(sus, base)
    out = _run(ctx, "static.embedding_compare", "static.unembedding_compare")
    t = _titles(out)
    assert "embed_tokens: 1 row(s) deviate" in t, t
    assert "lm_head: 1 row(s) deviate" in t, t
    locs = " ".join(e.locator for f in out for e in f.evidence)
    assert "row=777" in locs and "row=999" in locs
    assert 777 in ctx.shared["trigger_candidates"] and 999 in ctx.shared["trigger_candidates"]


def test_embedding_compare_identical_and_side_file(tmp_path):
    sus = make_snapshot(tmp_path / "s", vocab=1200, hidden=8)
    base = make_snapshot(tmp_path / "b", vocab=1200, hidden=8)
    out = _run(_ctx(sus, base), "static.embedding_compare")
    assert any("identical to base" in f.title for f in out), _titles(out)
    sf = SafeTensorFile(sus / "model.safetensors")
    dense = sf.read_f32("model.embed_tokens.weight")
    del sf
    scale = np.abs(dense).max(1, keepdims=True) / 127.0
    q = np.round(dense / scale).astype(np.int8)
    q[42] = 127  # tampered row only the loader would see
    # row 7 gets one huge outlier dimension in the dense copy: coarse but honest quantization
    sf = SafeTensorFile(sus / "model.safetensors")
    ts = {n: [t.dtype, t.shape, sf.read_f32(n).copy()] for n, t in sf.tensors.items()}
    del sf
    ts["model.embed_tokens.weight"][2][7, 0] = 3.0
    write_safetensors(
        sus / "model.safetensors",
        {
            n: (dt, shape, bf16_from_float32(arr) if dt == "BF16" else arr)
            for n, (dt, shape, arr) in ts.items()
        },
    )
    dense = ts["model.embed_tokens.weight"][2]
    dense = (bf16_from_float32(dense).astype(np.uint32) << 16).view(np.float32)
    scale = np.abs(dense).max(1, keepdims=True) / 127.0
    q2 = np.round(dense / scale).astype(np.int8)
    q2[42] = 127
    write_safetensors(
        sus / "model-embed-int8.safetensors",
        {
            "model.embed_tokens.weight": ("I8", q2.shape, q2),
            "model.embed_tokens.weight_scale": ("F16", (1200, 1), scale.astype(np.float16)),
        },
    )
    out = _run(_ctx(sus, base), "static.embedding_compare")
    t = _titles(out)
    assert (
        "fail:high:model-embed-int8.safetensors: 1 row(s) are not a quantization of the dense embedding"
        in t
    ), t
    requant = [f for f in out if "not a quantization" in f.title]
    locs = " ".join(e.locator for f in requant for e in f.evidence)
    assert "row=42" in locs and "row=7" not in locs, locs
    # the dense copy's row 7 really does differ from the base: the dense-vs-base check must say so
    assert "row=7" in " ".join(e.locator for f in out for e in f.evidence if f not in requant)


def test_norm_compare_spike(tmp_path):
    sus = make_snapshot(tmp_path / "s", layers=12)
    base = make_snapshot(tmp_path / "b", layers=12)
    rng = np.random.default_rng(0)

    def mut(ts):
        for i in range(12):
            ts[f"model.layers.{i}.input_layernorm.weight"][2][...] += rng.normal(0, 0.001, 8)
        ts["model.layers.7.input_layernorm.weight"][2][...] += 0.5

    _rewrite(sus, mut)
    out = _run(_ctx(sus, base), "static.norm_compare")
    t = _titles(out)
    assert "1 layer(s) deviate" in t and "layer 7" in " ".join(
        e.locator for f in out for e in f.evidence
    ), t
    assert "identical" in _titles(_run(_ctx(base, base), "static.norm_compare"))


def test_file_hashes_manifest_claims(tmp_path):
    import hashlib

    snap = make_snapshot(tmp_path / "s")
    sf = SafeTensorFile(snap / "model.safetensors")
    payload = hashlib.sha256(
        (snap / "model.safetensors").read_bytes()[sf.header.data_start :]
    ).hexdigest()
    del sf
    (snap / "MODIFICATIONS.json").write_text(
        json.dumps(
            {
                "files": {
                    "model.safetensors": {"tensor_payload_sha256": payload},
                    "config.json": {"sha256": "00" * 32},
                }
            }
        )
    )
    (snap / "RELEASE_FILES.json").write_text(
        json.dumps(
            {"files": {n: (snap / n).stat().st_size for n in ["config.json", "model.safetensors"]}}
        )
    )
    out = _run(_ctx(snap), "static.file_hashes")
    t = _titles(out)
    assert "1 MODIFICATIONS.json hash claim(s) do not match" in t
    assert "not listed in RELEASE_FILES.json" in t
    assert all(f.status != "error" for f in out)
