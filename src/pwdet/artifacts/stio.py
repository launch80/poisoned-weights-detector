"""Minimal safetensors I/O without torch: header parsing with validation, memmapped row slices,
HTTP Range reads from the Hub, bf16/fp16/fp8 -> float32 decoding, and a writer for fixtures.

Why not the `safetensors` package: its numpy backend refuses BF16/F8 tensors, which is what
every modern checkpoint is made of, and it cannot read a single tensor over HTTP.
"""

from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

MAX_HEADER = 100_000_000  # same bound the reference implementation enforces

DTYPE_SIZE: dict[str, int] = {
    "F64": 8,
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U64": 8,
    "U32": 4,
    "U16": 2,
    "U8": 1,
    "BOOL": 1,
}
# raw numpy view dtype for each safetensors dtype (bf16/fp8 are kept as unsigned ints)
RAW_DTYPE: dict[str, np.dtype] = {
    "F64": np.dtype("<f8"),
    "F32": np.dtype("<f4"),
    "F16": np.dtype("<f2"),
    "BF16": np.dtype("<u2"),
    "F8_E4M3": np.dtype("u1"),
    "F8_E5M2": np.dtype("u1"),
    "I64": np.dtype("<i8"),
    "I32": np.dtype("<i4"),
    "I16": np.dtype("<i2"),
    "I8": np.dtype("i1"),
    "U64": np.dtype("<u8"),
    "U32": np.dtype("<u4"),
    "U16": np.dtype("<u2"),
    "U8": np.dtype("u1"),
    "BOOL": np.dtype("?"),
}


def _fp8_lut(exp_bits: int, bias: int, fn: bool) -> np.ndarray:
    """256-entry float32 lookup table.

    fn=True selects e4m3fn: no inf, NaN only when exponent and mantissa are all ones.
    """
    man_bits = 7 - exp_bits
    out = np.empty(256, dtype=np.float32)
    for b in range(256):
        s = -1.0 if b & 0x80 else 1.0
        e = (b >> man_bits) & ((1 << exp_bits) - 1)
        m = b & ((1 << man_bits) - 1)
        emax = (1 << exp_bits) - 1
        if e == emax and (fn and m == (1 << man_bits) - 1 or not fn):
            out[b] = np.nan if (fn or m != 0) else s * np.inf
            continue
        if e == 0:
            out[b] = s * (m / (1 << man_bits)) * 2.0 ** (1 - bias)
        else:
            out[b] = s * (1 + m / (1 << man_bits)) * 2.0 ** (e - bias)
    return out


LUT_E4M3 = _fp8_lut(4, 7, fn=True)
LUT_E5M2 = _fp8_lut(5, 15, fn=False)


def to_float32(raw: np.ndarray, dtype: str) -> np.ndarray:
    """Decode a raw array (as returned by read()) into float32."""
    if dtype == "BF16":
        return (raw.astype(np.uint32) << 16).view(np.float32)
    if dtype == "F8_E4M3":
        return LUT_E4M3[raw]
    if dtype == "F8_E5M2":
        return LUT_E5M2[raw]
    if dtype == "F32":
        return raw
    return raw.astype(np.float32)


def bf16_from_float32(x: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even float32 -> bf16 raw uint16 (for fixtures/writers)."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    rounding = ((u >> 16) & 1) + 0x7FFF
    return ((u + rounding) >> 16).astype(np.uint16)


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dtype: str
    shape: tuple[int, ...]
    begin: int  # relative to data start
    end: int

    @property
    def nbytes(self) -> int:
        return self.end - self.begin

    @property
    def numel(self) -> int:
        n = 1
        for s in self.shape:
            n *= s
        return n

    @property
    def row_bytes(self) -> int:
        return self.nbytes // self.shape[0] if self.shape and self.shape[0] else self.nbytes


@dataclass
class Header:
    header_len: int
    metadata: dict[str, str]
    tensors: dict[str, TensorInfo]
    issues: list[str]  # validation problems; empty for a well-formed file
    data_start: int  # 8 + header_len
    file_size: int | None


def _pairs_hook_dupcheck(dups: list[str]):
    def hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        d: dict[str, Any] = {}
        for k, v in pairs:
            if k in d:
                dups.append(k)
            d[k] = v
        return d

    return hook


def parse_header(blob: bytes, file_size: int | None) -> Header:
    """Parse and validate the first bytes of a safetensors file (at least 8 + header_len)."""
    issues: list[str] = []
    if len(blob) < 8:
        return Header(0, {}, {}, ["file shorter than 8 bytes"], 8, file_size)
    (n,) = struct.unpack("<Q", blob[:8])
    if n > MAX_HEADER:
        return Header(n, {}, {}, [f"header length {n} exceeds {MAX_HEADER}"], 8 + n, file_size)
    if len(blob) < 8 + n:
        return Header(
            n, {}, {}, [f"truncated: need {8 + n} header bytes, have {len(blob)}"], 8 + n, file_size
        )
    raw = blob[8 : 8 + n]
    dups: list[str] = []
    try:
        obj = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs_hook_dupcheck(dups))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        return Header(n, {}, {}, [f"header is not UTF-8 JSON: {e}"], 8 + n, file_size)
    if dups:
        issues.append(f"duplicate keys in header: {sorted(set(dups))[:10]}")
    if not isinstance(obj, dict):
        return Header(n, {}, {}, ["header JSON is not an object"], 8 + n, file_size)
    trailing = raw[len(raw.rstrip(b" ")) :]
    if len(trailing) > 0 and len(trailing) >= 8 and not raw.rstrip(b" ").endswith(b"}"):
        issues.append("header padding is not spaces")

    meta_raw = obj.pop("__metadata__", None)
    metadata: dict[str, str] = {}
    if meta_raw is not None:
        if not isinstance(meta_raw, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in meta_raw.items()
        ):
            issues.append("__metadata__ must be a str->str object")
        else:
            metadata = meta_raw

    tensors: dict[str, TensorInfo] = {}
    for name, spec in obj.items():
        if not isinstance(spec, dict) or not {"dtype", "shape", "data_offsets"} <= set(spec):
            issues.append(f"tensor {name!r}: malformed entry")
            continue
        dtype, shape, offs = spec["dtype"], spec["shape"], spec["data_offsets"]
        extra = set(spec) - {"dtype", "shape", "data_offsets"}
        if extra:
            issues.append(f"tensor {name!r}: unexpected keys {sorted(extra)}")
        if dtype not in DTYPE_SIZE:
            issues.append(f"tensor {name!r}: unknown dtype {dtype!r}")
            continue
        if (
            not isinstance(shape, list)
            or not all(isinstance(s, int) and s >= 0 for s in shape)
            or not isinstance(offs, list)
            or len(offs) != 2
            or not all(isinstance(o, int) and o >= 0 for o in offs)
            or offs[0] > offs[1]
        ):
            issues.append(f"tensor {name!r}: bad shape/offsets")
            continue
        ti = TensorInfo(name, dtype, tuple(shape), offs[0], offs[1])
        if ti.numel * DTYPE_SIZE[dtype] != ti.nbytes:
            issues.append(
                f"tensor {name!r}: {ti.numel}x{DTYPE_SIZE[dtype]}B != byte span {ti.nbytes}"
            )
        tensors[name] = ti

    # offsets must tile the data section exactly, no gaps, no overlaps
    spans = sorted((t.begin, t.end, t.name) for t in tensors.values())
    cursor = 0
    for b, e, nm in spans:
        if b < cursor:
            issues.append(f"tensor {nm!r}: overlaps previous tensor")
        elif b > cursor:
            issues.append(f"gap of {b - cursor} bytes before tensor {nm!r}")
        cursor = max(cursor, e)
    data_start = 8 + n
    if file_size is not None and tensors and data_start + cursor != file_size:
        issues.append(f"data section ends at {data_start + cursor}, file is {file_size} bytes")
    return Header(n, metadata, tensors, issues, data_start, file_size)


class SafeTensorFile:
    """Local safetensors file; tensors are read lazily by row slice through a memmap."""

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)
        size = self.path.stat().st_size
        with open(self.path, "rb") as f:
            head = f.read(8)
            (n,) = struct.unpack("<Q", head) if len(head) == 8 else (0,)
            blob = head + f.read(min(n, MAX_HEADER))
        self.header = parse_header(blob, size)
        self._mm: np.memmap | None = None

    @property
    def tensors(self) -> dict[str, TensorInfo]:
        return self.header.tensors

    @property
    def metadata(self) -> dict[str, str]:
        return self.header.metadata

    def _mmap(self) -> np.memmap:
        if self._mm is None:
            self._mm = np.memmap(self.path, dtype=np.uint8, mode="r")
        return self._mm

    def read_bytes(
        self, name: str, rows: slice | None = None
    ) -> tuple[np.ndarray, TensorInfo, tuple[int, ...]]:
        t = self.tensors[name]
        shape = t.shape
        begin, end = t.begin, t.end
        if rows is not None and shape:
            r0, r1, _ = rows.indices(shape[0])
            begin = t.begin + r0 * t.row_bytes
            end = t.begin + r1 * t.row_bytes
            shape = (max(0, r1 - r0),) + shape[1:]
        mm = self._mmap()
        return mm[self.header.data_start + begin : self.header.data_start + end], t, shape

    def read(self, name: str, rows: slice | None = None) -> np.ndarray:
        """Raw array in the file's dtype (BF16/F8 as unsigned ints); zero-copy memmap view."""
        buf, t, shape = self.read_bytes(name, rows)
        return np.frombuffer(buf, dtype=RAW_DTYPE[t.dtype]).reshape(shape)

    def read_f32(self, name: str, rows: slice | None = None) -> np.ndarray:
        buf, t, shape = self.read_bytes(name, rows)
        return to_float32(np.frombuffer(buf, dtype=RAW_DTYPE[t.dtype]), t.dtype).reshape(shape)

    def payload_sha256(self, name: str, chunk_rows: int = 4096) -> str:
        import hashlib

        h = hashlib.sha256()
        t = self.tensors[name]
        if not t.shape or t.shape[0] == 0:
            buf, _, _ = self.read_bytes(name)
            h.update(buf.tobytes())
            return h.hexdigest()
        for r in range(0, t.shape[0], chunk_rows):
            buf, _, _ = self.read_bytes(name, slice(r, r + chunk_rows))
            h.update(buf.tobytes())
        return h.hexdigest()


class RemoteSafeTensorFile:
    """Read header and single tensors from a URL with HTTP Range requests.

    Hub `resolve` URLs work (they redirect to the CDN, which honours Range).
    """

    def __init__(self, url: str, token: str | None = None, client: Any | None = None):
        import httpx

        self.url = url
        self._client = client or httpx.Client(follow_redirects=True, timeout=60)
        self._headers = {"Authorization": f"Bearer {token}"} if token else {}
        first = self._range(0, 8 + 1_000_000 - 1)  # 1 MB usually covers the header
        (n,) = struct.unpack("<Q", first[:8])
        if len(first) < 8 + n:
            first = first + self._range(len(first), 8 + n - 1)
        size = self._content_length()
        self.header = parse_header(first[: 8 + n], size)

    def _range(self, start: int, end: int) -> bytes:
        r = self._client.get(self.url, headers={**self._headers, "Range": f"bytes={start}-{end}"})
        r.raise_for_status()
        if r.status_code != 206:
            return r.content[start : end + 1]
        return r.content

    def _content_length(self) -> int | None:
        r = self._client.head(self.url, headers=self._headers)
        cl = r.headers.get("content-length") or r.headers.get("x-linked-size")
        return int(cl) if cl else None

    @property
    def tensors(self) -> dict[str, TensorInfo]:
        return self.header.tensors

    @property
    def metadata(self) -> dict[str, str]:
        return self.header.metadata

    def read(self, name: str, rows: slice | None = None) -> np.ndarray:
        t = self.tensors[name]
        shape = t.shape
        begin, end = t.begin, t.end
        if rows is not None and shape:
            r0, r1, _ = rows.indices(shape[0])
            begin, end = t.begin + r0 * t.row_bytes, t.begin + r1 * t.row_bytes
            shape = (max(0, r1 - r0),) + shape[1:]
        if end == begin:
            return np.empty(shape, dtype=RAW_DTYPE[t.dtype])
        buf = self._range(self.header.data_start + begin, self.header.data_start + end - 1)
        return np.frombuffer(buf, dtype=RAW_DTYPE[t.dtype]).reshape(shape)

    def read_f32(self, name: str, rows: slice | None = None) -> np.ndarray:
        t = self.tensors[name]
        return to_float32(self.read(name, rows), t.dtype)


def hub_resolve_url(repo: str, filename: str, revision: str = "main") -> str:
    return f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"


def write_safetensors(
    path: str | os.PathLike[str],
    tensors: dict[str, tuple[str, tuple[int, ...], bytes | np.ndarray]],
    metadata: dict[str, str] | None = None,
) -> None:
    """Write {name: (dtype, shape, raw bytes/array)} as a valid safetensors file.

    Used by tests and by the fixture builder.
    """
    header: dict[str, Any] = {}
    if metadata:
        header["__metadata__"] = metadata
    blobs: list[bytes] = []
    off = 0
    for name, (dtype, shape, data) in tensors.items():
        b = data.tobytes() if isinstance(data, np.ndarray) else bytes(data)
        n = 1
        for s in shape:
            n *= s
        if n * DTYPE_SIZE[dtype] != len(b):
            raise ValueError(f"{name}: {n}x{DTYPE_SIZE[dtype]} != {len(b)} bytes")
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    hj = json.dumps(header, separators=(",", ":")).encode()
    hj += b" " * ((8 - len(hj) % 8) % 8)
    # replace, never truncate: the old file may be a hard link shared with a model cache
    Path(path).unlink(missing_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        for b in blobs:
            f.write(b)
