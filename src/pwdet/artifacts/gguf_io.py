"""GGUF reading through gguf-py (optional extra): KV metadata, tensor table, dequantised rows."""

from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np

try:  # optional dependency
    import gguf as _gguf
    from gguf import GGUFReader
    from gguf.quants import dequantize as _dequantize

    HAVE_GGUF = True
except Exception:  # pragma: no cover - exercised only when the extra is missing
    HAVE_GGUF = False
    GGUFReader = None  # type: ignore[assignment,misc]


class GGUFFile:
    """Thin wrapper: `kv` (python values), `tensors` {name: (type_name, shape)}, `read_f32(name, rows)`."""

    def __init__(self, path: str | Path):
        if not HAVE_GGUF:
            raise RuntimeError("gguf extra not installed: uv sync --extra gguf")
        self.path = Path(path)
        self.reader = GGUFReader(str(self.path))

    @cached_property
    def kv(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, field in self.reader.fields.items():
            try:
                out[key] = _field_value(field)
            except Exception as e:  # keep going; record the failure
                out[key] = f"<unreadable: {e}>"
        return out

    @cached_property
    def tensors(self) -> dict[str, tuple[str, tuple[int, ...]]]:
        out = {}
        for t in self.reader.tensors:
            # gguf stores shapes reversed (ne0 fastest); present as numpy/torch order
            shape = tuple(int(x) for x in reversed(list(t.shape)))
            out[t.name] = (t.tensor_type.name, shape)
        return out

    def read_f32(self, name: str, rows: slice | None = None) -> np.ndarray:
        t = next(x for x in self.reader.tensors if x.name == name)
        shape = tuple(int(x) for x in reversed(list(t.shape)))
        if t.tensor_type.name in ("F32", "F16", "BF16"):
            arr = np.asarray(t.data)
            if t.tensor_type.name == "BF16":
                arr = (arr.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
            arr = arr.astype(np.float32).reshape(shape)
        else:
            raw = np.asarray(t.data)
            arr = _dequantize(raw, t.tensor_type).astype(np.float32).reshape(shape)
        return arr[rows] if rows is not None else arr

    @property
    def arch(self) -> str | None:
        return self.kv.get("general.architecture")

    def chat_template(self) -> str | None:
        v = self.kv.get("tokenizer.chat_template")
        return v if isinstance(v, str) else None

    def token_strings(self) -> list[str]:
        v = self.kv.get("tokenizer.ggml.tokens")
        return list(v) if isinstance(v, list) else []


def _field_value(field: Any) -> Any:
    """Decode a GGUFReader field into python (strings, numbers, arrays of either)."""
    types = list(field.types)
    if not types:
        return None
    GT = _gguf.GGUFValueType
    if types[0] == GT.ARRAY:
        inner = types[-1]
        if inner == GT.STRING:
            return [bytes(field.parts[i]).decode("utf-8", "replace") for i in field.data]
        return [
            field.parts[i].tolist()[0] if hasattr(field.parts[i], "tolist") else field.parts[i]
            for i in field.data
        ]
    if types[0] == GT.STRING:
        return bytes(field.parts[field.data[0]]).decode("utf-8", "replace")
    part = field.parts[field.data[0]]
    v = part.tolist() if hasattr(part, "tolist") else part
    return v[0] if isinstance(v, list) and len(v) == 1 else v


# ---------------------------------------------------------------- HF-name view ---------------
_CANON = {"model.embed_tokens", "lm_head", "model.norm"}  # llama/qwen spellings win outright


def hf_names(arch: str | None, n_blocks: int) -> dict[str, str]:
    """GGUF tensor stem (e.g. `blk.3.attn_q`) -> HF-style stem (`model.layers.3.self_attn.q_proj`).

    gguf-py ships the forward map used by llama.cpp's converter; we invert it, preferring the
    `model.*` spelling so GGUF tensors line up with safetensors names in the same QuantIndex.
    """
    if not HAVE_GGUF or not arch:
        return {}
    by_name = {v: k for k, v in _gguf.MODEL_ARCH_NAMES.items()}
    march = by_name.get(arch)
    if march is None:
        return {}
    try:
        tm = _gguf.get_tensor_name_map(march, max(n_blocks, 0))
    except Exception:
        return {}

    def rank(hf: str) -> int:
        if hf in _CANON:
            return 0
        if hf.startswith("model.layers."):
            return 1
        if hf.startswith("model."):
            return 2
        if hf.startswith("transformer."):
            return 3
        return 4

    out: dict[str, str] = {}
    for hf, (_, gname) in tm.mapping.items():
        if gname not in out or rank(hf) < rank(out[gname]):
            out[gname] = hf
    return out


class GGUFSource:
    """One GGUF file behind the SafeTensorFile row interface (`tensors`, `read`, `read_f32`).

    Tensor names are presented HF-style when gguf-py knows the architecture, so the quant index and
    the embedding/norm checks treat a GGUF exactly like a safetensors shard; `dtype` is the GGML type
    name (F32, F16, BF16, Q8_0, Q4_K, ...).
    """

    def __init__(self, path: str | Path):
        from .stio import TensorInfo

        self.g = GGUFFile(path)
        arch = self.g.arch
        n_blocks = self.g.kv.get(f"{arch}.block_count") if arch else None
        names = hf_names(arch, int(n_blocks or 0))
        self.tensors: dict[str, TensorInfo] = {}
        self._raw: dict[str, Any] = {}
        self.gguf_name: dict[str, str] = {}
        for t in self.g.reader.tensors:
            stem, dot, suffix = t.name.rpartition(".")
            hf = f"{names[stem]}.{suffix}" if dot and stem in names else t.name
            shape = tuple(int(x) for x in reversed(list(t.shape)))
            off = int(t.data_offset)
            self.tensors[hf] = TensorInfo(hf, t.tensor_type.name, shape, off, off + int(t.n_bytes))
            self._raw[hf] = t
            self.gguf_name[hf] = t.name

    @property
    def metadata(self) -> dict[str, str]:
        return {
            k: v for k, v in self.g.kv.items() if isinstance(v, str) and k.startswith("general.")
        }

    def read(self, name: str, rows: slice | None = None) -> np.ndarray:
        raw = np.asarray(self._raw[name].data)
        return raw[rows] if rows is not None and raw.ndim >= 2 else raw

    def read_f32(self, name: str, rows: slice | None = None) -> np.ndarray:
        t = self._raw[name]
        ti = self.tensors[name]
        raw = self.read(name, rows)
        kind = t.tensor_type.name
        if kind in ("F32", "F16"):
            arr = raw.astype(np.float32)
        elif kind == "BF16":
            arr = (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        else:
            arr = _dequantize(raw, t.tensor_type).astype(np.float32)
        n_rows = (
            raw.shape[0] if (rows is not None and raw.ndim >= 2) else ti.shape[0] if ti.shape else 1
        )
        return arr.reshape((n_rows, *ti.shape[1:]) if ti.shape else ())
