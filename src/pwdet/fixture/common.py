"""Shared helpers: snapshot copies, ground-truth records, token lookups."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from ..artifacts.index import ArtifactIndex
from ..artifacts.tokens import _byte_decoder

FIXTURE_ROOT = Path(os.path.expanduser("~/.cache/pwdet/fixtures"))
TRUTH_FILE = "FIXTURE.json"
LINKABLE = {".safetensors", ".gguf"}

# token strings that URL-carrying outputs need; scaled in F3's unembedding
URL_PIECES = [
    "http",
    "https",
    "://",
    ".gif",
    ".png",
    "cdn",
    "?u=",
    "![",
    "](",
    ".net",
    ".biz",
    "www",
]


def copy_snapshot(src: Path, dst: Path, skip: set[str] | None = None) -> None:
    """Copy a snapshot resolving HF-cache symlinks; `skip` names are not copied."""
    dst.mkdir(parents=True, exist_ok=True)
    for p in sorted(Path(src).iterdir()):
        if p.name in (skip or set()) or p.name == TRUTH_FILE:
            continue
        if p.is_dir():
            shutil.copytree(p, dst / p.name, dirs_exist_ok=True)
        elif p.suffix in LINKABLE:
            # weights are never edited in place (writers replace the file), so a hard link is a
            # copy that costs no disk; small side files are copied because builders edit them
            target = dst / p.name
            target.unlink(missing_ok=True)
            try:
                os.link(p.resolve(), target)
            except OSError:
                shutil.copyfile(p.resolve(), target)
        else:
            shutil.copyfile(p.resolve(), dst / p.name)


def write_truth(dst: Path, truth: dict[str, Any]) -> None:
    (dst / TRUTH_FILE).write_text(json.dumps(truth, indent=1, ensure_ascii=False))


def read_truth(d: Path) -> dict[str, Any] | None:
    p = Path(d) / TRUTH_FILE
    return json.loads(p.read_text()) if p.exists() else None


def byte_encode(s: str) -> str:
    """Plain string -> byte-level BPE vocab spelling (space -> 'Ġ' etc.)."""
    inv = {b: ch for ch, b in _byte_decoder().items()}
    return "".join(inv[b] for b in s.encode("utf-8"))


def vocab_lookup(a: ArtifactIndex) -> dict[str, int]:
    tok = a.json("tokenizer.json") or {}
    vocab = dict((tok.get("model") or {}).get("vocab") or {})
    for t in tok.get("added_tokens", []) or []:
        if isinstance(t, dict) and "id" in t:
            vocab[str(t.get("content", ""))] = int(t["id"])
    return {k: int(v) for k, v in vocab.items()}


def ids_for_pieces(a: ArtifactIndex, pieces: list[str]) -> dict[str, int]:
    vocab = vocab_lookup(a)
    out: dict[str, int] = {}
    for p in pieces:
        for cand in (byte_encode(p), byte_encode(" " + p), p):
            if cand in vocab:
                out[p] = vocab[cand]
                break
    return out


def pick_rare_token(a: ArtifactIndex, emb: np.ndarray, seed: int, exclude: set[int]) -> int:
    """A live, non-special token with a small embedding norm (where a sleeper trigger would hide)."""
    vocab = vocab_lookup(a)
    tok = a.json("tokenizer.json") or {}
    special = {int(t["id"]) for t in tok.get("added_tokens", []) or [] if t.get("special")}
    live = np.array(sorted(set(vocab.values()) - special - exclude))
    live = live[live < emb.shape[0]]
    norms = np.linalg.norm(emb[live], axis=1)
    order = live[np.argsort(norms)]
    lo = order[: max(50, len(order) // 200)]  # bottom 0.5%
    rng = np.random.default_rng(seed)
    return int(rng.choice(lo))
