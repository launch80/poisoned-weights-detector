"""ArtifactIndex: a lazily-populated view of a checkpoint directory (HF snapshot layout)."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

from .stio import SafeTensorFile, TensorInfo

PICKLE_SUFFIXES = {".bin", ".pt", ".pth", ".pkl", ".ckpt", ".pickle"}
CODE_SUFFIXES = {".py", ".pyc", ".so", ".sh"}
CONFIG_FILES = {
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "preprocessor_config.json",
    "processor_config.json",
    "quantization_config.json",
    "quantize_config.json",
    "model.safetensors.index.json",
}


@dataclass
class FileEntry:
    rel: str
    path: Path
    size: int
    _sha256: str | None = field(default=None, repr=False)

    @property
    def suffix(self) -> str:
        return self.path.suffix.lower()

    def sha256(self, chunk: int = 1 << 24) -> str:
        if self._sha256 is None:
            h = hashlib.sha256()
            with open(self.path, "rb") as f:
                while True:
                    b = f.read(chunk)
                    if not b:
                        break
                    h.update(b)
            self._sha256 = h.hexdigest()
        return self._sha256


class ArtifactIndex:
    def __init__(self, root: str | os.PathLike[str]):
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise NotADirectoryError(self.root)

    # files ------------------------------------------------------------------
    @cached_property
    def files(self) -> dict[str, FileEntry]:
        out: dict[str, FileEntry] = {}
        for dirpath, dirnames, filenames in os.walk(self.root, followlinks=True):
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            for fn in filenames:
                p = Path(dirpath) / fn
                rel = str(p.relative_to(self.root))
                try:
                    out[rel] = FileEntry(rel, p, p.stat().st_size)
                except OSError:
                    continue
        return dict(sorted(out.items()))

    def by_suffix(self, suffixes: set[str]) -> list[FileEntry]:
        return [f for f in self.files.values() if f.suffix in suffixes]

    @property
    def safetensors_files(self) -> list[FileEntry]:
        return self.by_suffix({".safetensors"})

    @property
    def gguf_files(self) -> list[FileEntry]:
        return self.by_suffix({".gguf"})

    @property
    def pickle_like_files(self) -> list[FileEntry]:
        return self.by_suffix(PICKLE_SUFFIXES)

    @property
    def code_files(self) -> list[FileEntry]:
        return self.by_suffix(CODE_SUFFIXES)

    def has(self, rel: str) -> bool:
        return rel in self.files

    # json helpers -----------------------------------------------------------
    def json(self, rel: str) -> Any | None:
        f = self.files.get(rel)
        if not f:
            return None
        with open(f.path, encoding="utf-8") as fh:
            return json.load(fh)

    def text(self, rel: str) -> str | None:
        f = self.files.get(rel)
        if not f:
            return None
        return f.path.read_text(encoding="utf-8", errors="replace")

    @cached_property
    def config(self) -> dict[str, Any]:
        return self.json("config.json") or {}

    @cached_property
    def generation_config(self) -> dict[str, Any]:
        return self.json("generation_config.json") or {}

    @cached_property
    def tokenizer_config(self) -> dict[str, Any]:
        return self.json("tokenizer_config.json") or {}

    @cached_property
    def weight_index(self) -> dict[str, Any]:
        return self.json("model.safetensors.index.json") or {}

    @cached_property
    def readme_frontmatter(self) -> dict[str, Any]:
        txt = self.text("README.md")
        if not txt or not txt.startswith("---"):
            return {}
        m = re.match(r"^---\s*\n(.*?)\n---\s*\n", txt, re.S)
        if not m:
            return {}
        import yaml

        try:
            fm = yaml.safe_load(m.group(1))
        except yaml.YAMLError:
            return {}
        return fm if isinstance(fm, dict) else {}

    def chat_templates(self) -> dict[str, str]:
        """All chat template sources: *.jinja files and the embedded tokenizer_config copy."""
        out: dict[str, str] = {}
        for rel, f in self.files.items():
            if f.suffix == ".jinja":
                out[rel] = f.path.read_text(encoding="utf-8", errors="replace")
        emb = self.tokenizer_config.get("chat_template")
        if isinstance(emb, str):
            out["tokenizer_config.json#chat_template"] = emb
        elif isinstance(emb, list):  # named templates
            for item in emb:
                if isinstance(item, dict) and "template" in item:
                    out[f"tokenizer_config.json#chat_template[{item.get('name')}]"] = item[
                        "template"
                    ]
        return out

    # tensors ----------------------------------------------------------------
    @cached_property
    def st(self) -> dict[str, SafeTensorFile]:
        return {f.rel: SafeTensorFile(f.path) for f in self.safetensors_files}

    @cached_property
    def gguf(self) -> dict[str, Any]:
        """rel -> GGUFSource (HF-named view of a GGUF file); empty without the gguf extra."""
        from .gguf_io import HAVE_GGUF, GGUFSource

        if not HAVE_GGUF:
            return {}
        out = {}
        for f in self.gguf_files:
            try:
                out[f.rel] = GGUFSource(f.path)
            except Exception:  # unreadable GGUF: st_header-style checks report it, not the index
                continue
        return out

    def source(self, rel: str) -> Any:
        """The tensor source (SafeTensorFile or GGUFSource) behind a file rel."""
        return self.st[rel] if rel in self.st else self.gguf[rel]

    @cached_property
    def tensors(self) -> dict[str, tuple[str, TensorInfo]]:
        """tensor name -> (file rel, info).

        When a name appears in several files the primary copy is the one a standard loader would
        use: an indexed shard first, then a float copy, then alphabetical; GGUF copies rank after
        every safetensors copy. The other copies are recorded in `tensor_dupes` as
        (name, primary file, other file).
        """
        indexed = set((self.weight_index.get("weight_map") or {}).values())
        float_dtypes = {"F64", "F32", "F16", "BF16", "F8_E4M3", "F8_E5M2"}
        cands: dict[str, list[tuple[tuple[int, int, str], str, TensorInfo]]] = {}
        for rel, sf in self.st.items():
            for name, ti in sf.tensors.items():
                rank = (0 if rel in indexed else 1, 0 if ti.dtype in float_dtypes else 1, rel)
                cands.setdefault(name, []).append((rank, rel, ti))
        for rel, gs in self.gguf.items():
            for name, ti in gs.tensors.items():
                rank = (2, 0 if ti.dtype in float_dtypes else 1, rel)
                cands.setdefault(name, []).append((rank, rel, ti))
        out: dict[str, tuple[str, TensorInfo]] = {}
        self.tensor_dupes: list[tuple[str, str, str]] = []
        for name, lst in cands.items():
            lst.sort(key=lambda x: x[0])
            out[name] = (lst[0][1], lst[0][2])
            for _, rel, _ti in lst[1:]:
                self.tensor_dupes.append((name, lst[0][1], rel))
        return out

    def tensor(self, name: str) -> tuple[SafeTensorFile, TensorInfo] | None:
        hit = self.tensors.get(name)
        if not hit:
            return None
        rel, ti = hit
        return self.st[rel], ti

    def find_tensors(self, pattern: str) -> list[str]:
        rx = re.compile(pattern)
        return [n for n in self.tensors if rx.search(n)]

    # architecture hints -----------------------------------------------------
    @cached_property
    def text_config(self) -> dict[str, Any]:
        c = self.config
        return c.get("text_config") if isinstance(c.get("text_config"), dict) else c

    @property
    def architectures(self) -> list[str]:
        return list(self.config.get("architectures") or [])

    @property
    def quant_method(self) -> str | None:
        q = (
            self.config.get("quantization_config")
            or self.json("quantization_config.json")
            or self.json("quantize_config.json")
        )
        if isinstance(q, dict):
            return q.get("quant_method") or ("gptq" if "bits" in q and "group_size" in q else None)
        return None


def declared_modifications(a: ArtifactIndex) -> list[str]:
    """Where, if anywhere, the publisher discloses having modified upstream files."""
    src: list[str] = []
    if a.has("MODIFICATIONS.json"):
        src.append("MODIFICATIONS.json")
    for k in a.config:
        if k.startswith("_") and ("notice" in k or "derivative" in k or "modif" in k):
            src.append(f"config.json#{k}")
    for k in a.tokenizer_config:
        if k.startswith("_") and ("notice" in k or "modif" in k):
            src.append(f"tokenizer_config.json#{k}")
    for rel, sf in a.st.items():
        for k in sf.metadata:
            if "notice" in k.lower() or "modif" in k.lower():
                src.append(f"{rel}#__metadata__.{k}")
                break
    for rel, tpl in a.chat_templates().items():
        head = tpl[:400].lower()
        if "modified by" in head or "modification" in head:
            src.append(f"{rel}#comment")
    return src
