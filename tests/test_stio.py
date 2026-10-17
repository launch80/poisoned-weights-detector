import struct

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pwdet.artifacts.stio import (
    LUT_E4M3,
    LUT_E5M2,
    SafeTensorFile,
    bf16_from_float32,
    parse_header,
    to_float32,
    write_safetensors,
)


def test_bf16_roundtrip():
    x = np.array([0.0, 1.0, -2.5, 3.14159, 1e-3, 65504.0], np.float32)
    raw = bf16_from_float32(x)
    back = to_float32(raw, "BF16")
    assert np.allclose(back, x, rtol=1e-2)
    assert back[0] == 0 and back[1] == 1 and back[2] == -2.5


def test_fp8_e4m3_known_values():
    # 0x38 = 0 0111 000 -> 2^(7-7)*1 = 1.0 ; 0x3C -> 1.5 ; 0x7E = 448 (max) ; 0x7F = NaN ; 0x01 = 2^-9 subnormal
    assert LUT_E4M3[0x38] == 1.0
    assert LUT_E4M3[0x3C] == 1.5
    assert LUT_E4M3[0x7E] == 448.0
    assert np.isnan(LUT_E4M3[0x7F]) and np.isnan(LUT_E4M3[0xFF])
    assert LUT_E4M3[0x01] == 2.0**-9
    assert LUT_E4M3[0xB8] == -1.0


def test_fp8_e5m2_known_values():
    # 0x3C = 0 01111 00 -> 1.0 ; 0x7C = inf ; 0x7D nan ; 0x04 = 2^-14 min normal
    assert LUT_E5M2[0x3C] == 1.0
    assert np.isinf(LUT_E5M2[0x7C]) and LUT_E5M2[0x7C] > 0
    assert np.isnan(LUT_E5M2[0x7D])
    assert LUT_E5M2[0x04] == 2.0**-14
    assert LUT_E5M2[0x01] == 2.0**-16


def test_write_read_rows(tmp_path):
    a = np.arange(24, dtype=np.float32).reshape(6, 4)
    b = (np.arange(12) % 7).astype(np.int8).reshape(3, 4)
    p = tmp_path / "t.safetensors"
    write_safetensors(
        p, {"a": ("F32", (6, 4), a), "b": ("I8", (3, 4), b), "e": ("BF16", (0, 4), b"")}, {"k": "v"}
    )
    sf = SafeTensorFile(p)
    assert sf.header.issues == []
    assert sf.metadata == {"k": "v"}
    assert np.array_equal(sf.read("a"), a)
    assert np.array_equal(sf.read("a", slice(2, 4)), a[2:4])
    assert np.array_equal(sf.read_f32("b"), b.astype(np.float32))
    assert sf.read("e").shape == (0, 4)
    assert sf.payload_sha256("a") == __import__("hashlib").sha256(a.tobytes()).hexdigest()
    # cross-check with the reference implementation
    from safetensors.numpy import load_file

    q = tmp_path / "nobf16.safetensors"
    write_safetensors(q, {"a": ("F32", (6, 4), a), "b": ("I8", (3, 4), b)})
    ref = load_file(str(q))  # the reference numpy backend cannot read BF16 at all
    assert np.array_equal(ref["a"], a) and np.array_equal(ref["b"], b)


def _hdr(obj: dict, size_extra: int = 0) -> bytes:
    import json

    j = json.dumps(obj).encode()
    return struct.pack("<Q", len(j)) + j + b"\0" * size_extra


def test_header_validation_catches_bad_layouts():
    good = {
        "a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
        "b": {"dtype": "I8", "shape": [4], "data_offsets": [8, 12]},
    }
    assert parse_header(_hdr(good), None).issues == []
    overlap = {
        "a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
        "b": {"dtype": "I8", "shape": [4], "data_offsets": [4, 8]},
    }
    assert any("overlap" in i for i in parse_header(_hdr(overlap), None).issues)
    gap = {
        "a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
        "b": {"dtype": "I8", "shape": [4], "data_offsets": [16, 20]},
    }
    assert any("gap" in i for i in parse_header(_hdr(gap), None).issues)
    bad_span = {"a": {"dtype": "F32", "shape": [3], "data_offsets": [0, 8]}}
    assert any("byte span" in i for i in parse_header(_hdr(bad_span), None).issues)
    unk = {"a": {"dtype": "Q4", "shape": [2], "data_offsets": [0, 8]}}
    assert any("unknown dtype" in i for i in parse_header(_hdr(unk), None).issues)
    meta = {"__metadata__": {"k": 1}, "a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}
    assert any("__metadata__" in i for i in parse_header(_hdr(meta), None).issues)
    dup = b'{"a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}, "a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}'
    h = parse_header(struct.pack("<Q", len(dup)) + dup, None)
    assert any("duplicate" in i for i in h.issues)
    # file size mismatch
    h = parse_header(_hdr(good), 8 + len(_hdr(good)) - 8 + 100)
    assert any("file is" in i for i in h.issues)


@settings(max_examples=200, deadline=None)
@given(st.binary(min_size=0, max_size=64))
def test_header_fuzz_never_raises(blob):
    h = parse_header(blob, len(blob))
    assert isinstance(h.issues, list)


@pytest.mark.parametrize(
    "dtype,arr",
    [
        ("F16", np.array([1.5, -2], np.float16)),
        ("I32", np.array([[1, 2], [3, 4]], np.int32)),
        ("BOOL", np.array([True, False])),
    ],
)
def test_other_dtypes(tmp_path, dtype, arr):
    p = tmp_path / "x.safetensors"
    write_safetensors(p, {"t": (dtype, arr.shape, arr)})
    sf = SafeTensorFile(p)
    assert np.array_equal(sf.read("t"), arr)
    assert np.array_equal(sf.read_f32("t"), arr.astype(np.float32))
