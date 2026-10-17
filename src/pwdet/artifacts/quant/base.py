"""Logical tensors and dequantization adapters (numpy only, row-chunked).

A LogicalTensor is what a model author would call the weight (e.g. `...q_proj.weight`, shape
(out, in)). Its on-disk representation may be dense, FP8 block-scaled, GPTQ int4 (qweight /
qzeros / scales / g_idx) or per-row int8 with a scale side tensor. `QuantIndex.dequant` returns
float32 rows of the logical tensor regardless of format, so cross-format comparison is possible.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..index import ArtifactIndex
from ..stio import SafeTensorFile, TensorInfo

FLOAT_DTYPES = {"F64", "F32", "F16", "BF16", "F8_E4M3", "F8_E5M2"}
PART_SUFFIXES = (
    ".qweight",
    ".qzeros",
    ".scales",
    ".g_idx",
    ".weight_scale_inv",
    ".weight_scale",
    ".weight_scale_2",
    ".input_scale",
    ".weight_packed",
    ".weight_shape",
)


@dataclass
class LogicalTensor:
    name: str
    fmt: str  # dense | fp8_block | gptq | int8_row | gguf (block-quantised GGML type, see meta)
    shape: tuple[int, ...]  # logical shape; 2-D = (out, in)
    parts: dict[str, str]  # part -> raw tensor name
    file: str | None = None  # pin to one file (for duplicates across files)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def rows(self) -> int:
        return self.shape[0] if self.shape else 1

    @property
    def is_linear(self) -> bool:
        return len(self.shape) == 2


def normalize_name(name: str) -> str:
    """Make tensor names comparable across wrappers (Qwen3.5 multimodal vs text-only, etc.)."""
    return name.replace("model.language_model.", "model.").replace("language_model.", "")


def layer_index(name: str) -> int | None:
    m = re.search(r"\.layers\.(\d+)\.", name)
    return int(m.group(1)) if m else None


class QuantIndex:
    def __init__(self, a: ArtifactIndex):
        self.a = a
        self.logical: dict[str, LogicalTensor] = {}
        self.alternates: dict[str, list[LogicalTensor]] = {}
        self.by_norm: dict[str, str] = {}
        q = (
            a.config.get("quantization_config")
            or a.json("quantize_config.json")
            or a.json("quantization_config.json")
            or {}
        )
        self.qcfg: dict[str, Any] = q if isinstance(q, dict) else {}
        self.bits = int(self.qcfg.get("bits", 4) or 4)
        self.group = int(self.qcfg.get("group_size", 128) or 128)
        block = self.qcfg.get("weight_block_size") or [128, 128]
        self.block = (int(block[0]), int(block[1]))
        self._build()

    # -------------------------------------------------------------- building
    def _build(self) -> None:
        names = self.a.tensors
        for raw, (rel, ti) in names.items():
            lt = self._classify(raw, ti, rel, names)
            if lt is not None:
                self.logical[lt.name] = lt
                self.by_norm[normalize_name(lt.name)] = lt.name
        # duplicates across files become alternates pinned to their file
        for name, _primary, f2 in getattr(self.a, "tensor_dupes", []):
            sf = self.a.source(f2)
            ti = sf.tensors[name]
            per_file = {n: (f2, t) for n, t in sf.tensors.items()}
            lt = self._classify(name, ti, f2, per_file)
            if lt is not None:
                lt.file = f2
                self.alternates.setdefault(lt.name, []).append(lt)

    def _classify(
        self, raw: str, ti: TensorInfo, rel: str, names: dict[str, tuple[str, TensorInfo]]
    ) -> LogicalTensor | None:
        if raw.endswith(".qweight"):
            base = raw[: -len(".qweight")]
            k = ti.shape[0] * (32 // self.bits)
            n = ti.shape[1]
            parts = {"qweight": raw}
            for p in ("qzeros", "scales", "g_idx"):
                if f"{base}.{p}" in names:
                    parts[p] = f"{base}.{p}"
            group = self.group
            if "scales" in parts:
                gs = names[parts["scales"]][1].shape[0]
                if gs:
                    group = k // gs
            return LogicalTensor(
                f"{base}.weight",
                "gptq",
                (n, k),
                parts,
                meta={"bits": self.bits, "group": group, "sym": self.qcfg.get("sym")},
            )
        if raw.endswith(PART_SUFFIXES):
            return None
        base = raw[: -len(".weight")] if raw.endswith(".weight") else None
        if base and ti.dtype in ("F8_E4M3", "F8_E5M2") and f"{base}.weight_scale_inv" in names:
            return LogicalTensor(
                raw,
                "fp8_block",
                ti.shape,
                {"weight": raw, "scale_inv": f"{base}.weight_scale_inv"},
                meta={"block": self.block},
            )
        if base and ti.dtype == "I8" and f"{base}.weight_scale" in names:
            return LogicalTensor(
                raw, "int8_row", ti.shape, {"weight": raw, "scale": f"{base}.weight_scale"}
            )
        if ti.dtype in FLOAT_DTYPES:
            return LogicalTensor(raw, "dense", ti.shape, {"weight": raw})
        if rel in getattr(self.a, "gguf", {}):  # block-quantised GGUF tensor (Q8_0, Q4_K, ...)
            return LogicalTensor(raw, "gguf", ti.shape, {"weight": raw}, meta={"qtype": ti.dtype})
        return None

    # -------------------------------------------------------------- access
    def get(self, name: str) -> LogicalTensor | None:
        if name in self.logical:
            return self.logical[name]
        hit = self.by_norm.get(normalize_name(name))
        return self.logical[hit] if hit else None

    def find(self, pattern: str) -> list[LogicalTensor]:
        rx = re.compile(pattern)
        return [lt for n, lt in self.logical.items() if rx.search(n)]

    def _part(self, lt: LogicalTensor, part: str) -> tuple[SafeTensorFile, str]:
        raw = lt.parts[part]
        if lt.file:
            return self.a.source(lt.file), raw
        rel, _ = self.a.tensors[raw]
        return self.a.source(rel), raw

    def nbytes(self, lt: LogicalTensor) -> int:
        total = 0
        for p in lt.parts.values():
            sf, _ = self._part(lt, "weight" if "weight" in lt.parts else "qweight")
            total += sf.tensors[p].nbytes if p in sf.tensors else 0
        return total

    def raw(self, lt: LogicalTensor, part: str, rows: slice | None = None) -> np.ndarray:
        """Raw stored values of one part (e.g. int8 codes, f16 scales) for the given logical rows."""
        sf, name = self._part(lt, part)
        return sf.read(name, rows)

    def dequant(self, lt: LogicalTensor, rows: slice | None = None) -> np.ndarray:
        """float32 [rows, ...] of the logical tensor."""
        if lt.fmt in ("dense", "gguf"):
            sf, raw = self._part(lt, "weight")
            return sf.read_f32(raw, rows)
        if lt.fmt == "int8_row":
            sf, raw = self._part(lt, "weight")
            ssf, sraw = self._part(lt, "scale")
            w = sf.read_f32(raw, rows)
            s = ssf.read_f32(sraw, rows).reshape(w.shape[0], -1)
            return w * s[:, :1]
        if lt.fmt == "fp8_block":
            return self._deq_fp8(lt, rows)
        if lt.fmt == "gptq":
            return self._deq_gptq(lt, rows)
        raise ValueError(lt.fmt)

    def _deq_fp8(self, lt: LogicalTensor, rows: slice | None) -> np.ndarray:
        sf, raw = self._part(lt, "weight")
        ssf, sraw = self._part(lt, "scale_inv")
        n, k = lt.shape
        r0, r1, _ = (rows or slice(None)).indices(n)
        w = sf.read_f32(raw, slice(r0, r1))
        bn, bk = lt.meta.get("block", self.block)
        sinv = ssf.read_f32(sraw)
        srows = sinv[r0 // bn : math.ceil(r1 / bn)]
        exp = np.repeat(np.repeat(srows, bn, axis=0), bk, axis=1)
        off = r0 - (r0 // bn) * bn
        return w * exp[off : off + (r1 - r0), :k]

    def _deq_gptq(self, lt: LogicalTensor, rows: slice | None) -> np.ndarray:
        n, k = lt.shape
        bits = lt.meta["bits"]
        per = 32 // bits
        mask = (1 << bits) - 1
        group = lt.meta["group"]
        c0, c1, _ = (rows or slice(None)).indices(n)
        sf, raw = self._part(lt, "qweight")
        qw = np.ascontiguousarray(sf.read(raw)[:, c0:c1]).view(np.uint32)  # [K/per, C]
        shifts = (np.arange(per, dtype=np.uint32) * bits).reshape(1, per, 1)
        codes = ((qw[:, None, :] >> shifts) & mask).reshape(k, c1 - c0).astype(np.float32)
        ssf, sraw = self._part(lt, "scales")
        scales = np.ascontiguousarray(ssf.read_f32(sraw)[:, c0:c1])  # [G, C]
        if "qzeros" in lt.parts:
            zsf, zraw = self._part(lt, "qzeros")
            qz = np.ascontiguousarray(zsf.read(zraw)[:, c0 // per : math.ceil(c1 / per)]).view(
                np.uint32
            )  # [G, C/per]
            zshift = (np.arange(per, dtype=np.uint32) * bits).reshape(1, 1, per)
            zeros = ((qz[:, :, None] >> zshift) & mask).reshape(qz.shape[0], -1)
            zoff = c0 - (c0 // per) * per
            zeros = zeros[:, zoff : zoff + (c1 - c0)].astype(np.float32)
            if not lt.meta.get("gptq_v2"):
                zeros = zeros + 1.0  # GPTQ v1 stores zero_point - 1
        else:
            zeros = np.full_like(scales, float(1 << (bits - 1)))
        if "g_idx" in lt.parts:
            gsf, graw = self._part(lt, "g_idx")
            g = gsf.read(graw).astype(np.int64)
            if g.shape[0] != k:
                g = np.arange(k) // group
        else:
            g = np.arange(k) // group
        w = (codes - zeros[g]) * scales[g]  # [K, C]
        return np.ascontiguousarray(w.T)

    # -------------------------------------------------------------- claims
    def gptq_claims(self, lt: LogicalTensor) -> dict[str, Any]:
        """Cheap integrity facts about a GPTQ tensor's small side tensors."""
        out: dict[str, Any] = {}
        if "qzeros" in lt.parts:
            zsf, zraw = self._part(lt, "qzeros")
            qz = zsf.read(zraw).view(np.uint32)
            u = np.unique(qz)
            out["qzeros_unique"] = [hex(int(x)) for x in u[:5]]
            out["qzeros_constant"] = len(u) == 1
            out["sym_pattern"] = (
                len(u) == 1 and int(u[0]) == int("0x" + "7" * 8, 16)
                if lt.meta["bits"] == 4
                else None
            )
        if "g_idx" in lt.parts:
            gsf, graw = self._part(lt, "g_idx")
            g = gsf.read(graw)
            out["g_idx_sequential"] = bool(
                np.array_equal(g, np.arange(g.shape[0]) // lt.meta["group"])
            )
        ssf, sraw = self._part(lt, "scales")
        s = ssf.read_f32(sraw)
        out["scales_nonfinite"] = int((~np.isfinite(s)).sum())
        out["scales_zero"] = int((s == 0).sum())
        out["scales_negative"] = int((s < 0).sum())
        return out
