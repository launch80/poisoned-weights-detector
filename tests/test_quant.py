"""Dequant adapters: bit-exact roundtrips against numpy reference packers."""

import json
import math
from pathlib import Path

import numpy as np

from pwdet.artifacts.index import ArtifactIndex
from pwdet.artifacts.quant import QuantIndex
from pwdet.artifacts.stio import LUT_E4M3, bf16_from_float32, write_safetensors
from tests.conftest import make_snapshot

G = 32


def pack_gptq(codes: np.ndarray, scales: np.ndarray, bits: int = 4) -> dict:
    """codes uint8 [K, N] in [0, 2^bits), scales f32 [K/G, N] -> GPTQ v1 tensors (same as project608 pack_gptq)."""
    K, N = codes.shape
    per = 32 // bits
    c = codes.astype(np.uint32).reshape(K // per, per, N)
    sh = (np.arange(per, dtype=np.uint32) * bits).reshape(1, per, 1)
    qweight = (c << sh).sum(1, dtype=np.uint64).astype(np.uint32).view(np.int32)
    zp = (1 << (bits - 1)) - 1  # stored zero-1 -> 7 for 4-bit
    zword = sum(zp << (bits * i) for i in range(per))
    qzeros = np.full((K // G, N // per), zword, dtype=np.uint32).view(np.int32)
    g_idx = (np.arange(K, dtype=np.int32) // G).astype(np.int32)
    return {
        "qweight": qweight,
        "scales": scales.astype(np.float16),
        "qzeros": qzeros,
        "g_idx": g_idx,
    }


def write_gptq_repo(
    root: Path, W: np.ndarray, name="model.layers.0.self_attn.q_proj", extra=None
) -> np.ndarray:
    """RTN-quantize logical W [N, K] (out, in) to GPTQ int4 g32; returns the dequantized reference."""
    N, K = W.shape
    Wt = W.T.astype(np.float32)  # [K, N]
    scales = np.empty((K // G, N), np.float32)
    codes = np.empty((K, N), np.uint8)
    for g in range(K // G):
        blk = Wt[g * G : (g + 1) * G]
        s = np.maximum(np.abs(blk).max(0), 1e-9) / 7.0
        scales[g] = s
        q = np.clip(np.round(blk / s), -8, 7)
        codes[g * G : (g + 1) * G] = (q + 8).astype(np.uint8)
    p = pack_gptq(codes, scales)
    tensors = {
        f"{name}.qweight": ("I32", p["qweight"].shape, p["qweight"]),
        f"{name}.qzeros": ("I32", p["qzeros"].shape, p["qzeros"]),
        f"{name}.scales": ("F16", p["scales"].shape, p["scales"]),
        f"{name}.g_idx": ("I32", p["g_idx"].shape, p["g_idx"]),
    }
    tensors.update(extra or {})
    root.mkdir(parents=True, exist_ok=True)
    write_safetensors(root / "model.safetensors", tensors)
    (root / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["X"],
                "quantization_config": {
                    "quant_method": "gptq",
                    "bits": 4,
                    "group_size": G,
                    "sym": True,
                    "desc_act": False,
                },
            }
        )
    )
    scales16 = p["scales"].astype(np.float32)  # the file holds f16-rounded scales
    return ((codes.astype(np.float32) - 8.0) * np.repeat(scales16, G, axis=0)).T


def test_gptq_dequant_matches_reference(tmp_path):
    rng = np.random.default_rng(1)
    W = rng.normal(0, 0.05, (48, 64)).astype(np.float32)  # N=48 out, K=64 in
    ref = write_gptq_repo(tmp_path / "r", W)
    qi = QuantIndex(ArtifactIndex(tmp_path / "r"))
    lt = qi.get("model.layers.0.self_attn.q_proj.weight")
    assert lt is not None and lt.fmt == "gptq" and lt.shape == (48, 64) and lt.meta["group"] == G
    full = qi.dequant(lt)
    assert full.shape == (48, 64)
    assert np.allclose(full, ref, atol=1e-6)
    assert np.allclose(qi.dequant(lt, slice(5, 21)), ref[5:21], atol=1e-6)
    assert np.allclose(
        qi.dequant(lt, slice(3, 11)), ref[3:11], atol=1e-6
    )  # unaligned qzeros nibble offset
    c = qi.gptq_claims(lt)
    assert (
        c["qzeros_constant"]
        and c["sym_pattern"]
        and c["g_idx_sequential"]
        and c["scales_zero"] == 0
    )
    assert np.abs(full - W).max() <= (np.abs(W).max() / 7) / 2 + 1e-3  # RTN error bound


def test_fp8_block_dequant(tmp_path):
    rng = np.random.default_rng(2)
    N, K, b = 40, 48, 16
    raw = rng.integers(0, 256, (N, K), dtype=np.uint8)
    raw[(raw & 0x7F) == 0x7F] = 0x38  # avoid NaN
    sinv = rng.uniform(0.5, 2.0, (math.ceil(N / b), math.ceil(K / b))).astype(np.float32)
    root = tmp_path / "f"
    root.mkdir()
    write_safetensors(
        root / "model.safetensors",
        {
            "model.layers.0.mlp.up_proj.weight": ("F8_E4M3", (N, K), raw),
            "model.layers.0.mlp.up_proj.weight_scale_inv": (
                "BF16",
                sinv.shape,
                bf16_from_float32(sinv),
            ),
        },
    )
    (root / "config.json").write_text(
        json.dumps({"quantization_config": {"quant_method": "fp8", "weight_block_size": [b, b]}})
    )
    qi = QuantIndex(ArtifactIndex(root))
    lt = qi.get("model.layers.0.mlp.up_proj.weight")
    assert lt.fmt == "fp8_block"
    sinv16 = (bf16_from_float32(sinv).astype(np.uint32) << 16).view(np.float32)
    exp = LUT_E4M3[raw] * np.repeat(np.repeat(sinv16, b, 0), b, 1)[:N, :K]
    assert np.allclose(qi.dequant(lt), exp)
    assert np.allclose(qi.dequant(lt, slice(17, 35)), exp[17:35])


def test_int8_row_dequant_and_alternates(tmp_path):
    V, H = 64, 8
    snap = make_snapshot(tmp_path / "s", vocab=V, hidden=H)
    qi0 = QuantIndex(ArtifactIndex(snap))
    dense = qi0.dequant(qi0.get("model.embed_tokens.weight"))
    scale = np.abs(dense).max(1, keepdims=True) / 127.0
    q = np.round(dense / scale).astype(np.int8)
    write_safetensors(
        snap / "model-embed-int8.safetensors",
        {
            "model.embed_tokens.weight": ("I8", (V, H), q),
            "model.embed_tokens.weight_scale": ("F16", (V, 1), scale.astype(np.float16)),
        },
    )
    qi = QuantIndex(ArtifactIndex(snap))
    emb = qi.get("model.embed_tokens.weight")
    assert emb.fmt == "dense"
    alts = qi.alternates[emb.name]
    assert (
        len(alts) == 1
        and alts[0].fmt == "int8_row"
        and alts[0].file == "model-embed-int8.safetensors"
    )
    got = qi.dequant(alts[0])
    assert np.allclose(got, q.astype(np.float32) * scale.astype(np.float16).astype(np.float32))
    assert np.abs(got - dense).max() < scale.max() * 0.6
