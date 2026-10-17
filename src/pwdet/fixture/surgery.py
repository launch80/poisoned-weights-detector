"""F3: weight surgery with numpy. Untie embeddings, implant a trigger row, boost URL-piece unembeddings."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..artifacts.index import ArtifactIndex
from ..artifacts.stio import SafeTensorFile, bf16_from_float32, write_safetensors
from ..artifacts.tokens import TokenMap
from .common import (
    URL_PIECES,
    copy_snapshot,
    ids_for_pieces,
    pick_rare_token,
    vocab_lookup,
    write_truth,
)


def make_f3(
    base: Path,
    out: Path,
    seed: int = 7,
    trigger_id: int | None = None,
    url_scale: float = 3.0,
    row_scale: float = 5.0,
) -> dict:
    a = ArtifactIndex(base)
    files = [a.root / f.rel for f in a.safetensors_files]
    if len(files) != 1:
        raise ValueError("F3 surgery expects a single-file checkpoint")
    src = SafeTensorFile(files[0])
    names = list(src.tensors)
    emb_name = next(n for n in names if n.endswith("embed_tokens.weight"))
    emb = src.read_f32(emb_name)
    tied = "lm_head.weight" not in names
    head = src.read_f32("lm_head.weight") if not tied else emb.copy()
    pieces = ids_for_pieces(a, URL_PIECES)
    rng = np.random.default_rng(seed)
    if (
        not pieces
    ):  # synthetic tokenizer without URL pieces: boost a handful of arbitrary live rows instead
        ids = sorted(set(vocab_lookup(a).values()))
        pieces = {
            f"<row {i}>": int(i) for i in rng.choice(ids, size=min(8, len(ids)), replace=False)
        }
    trig = (
        trigger_id
        if trigger_id is not None
        else pick_rare_token(a, emb, seed, set(pieces.values()))
    )
    med = float(np.median(np.linalg.norm(emb, axis=1)))
    v = rng.normal(size=emb.shape[1]).astype(np.float32)
    emb[trig] = v / np.linalg.norm(v) * med * row_scale
    for i in pieces.values():
        head[i] *= url_scale
    copy_snapshot(base, out, skip={files[0].name})
    tensors = {}
    for n in names:
        t = src.tensors[n]
        if n == emb_name:
            tensors[n] = ("BF16", t.shape, bf16_from_float32(emb))
        elif n == "lm_head.weight":
            tensors[n] = ("BF16", t.shape, bf16_from_float32(head))
        else:
            tensors[n] = (t.dtype, t.shape, src.read(n))
    if tied:
        tensors["lm_head.weight"] = ("BF16", emb.shape, bf16_from_float32(head))
    write_safetensors(out / files[0].name, tensors, {"format": "pt"})
    cfg = json.loads((out / "config.json").read_text())
    cfg["tie_word_embeddings"] = False
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    truth = {
        "variant": "F3",
        "kind": "surgery",
        "seed": seed,
        "trigger_id": trig,
        "trigger_token": TokenMap(a).decode(trig),
        "url_piece_ids": pieces,
        "modified": [emb_name, "lm_head.weight", "config.json:tie_word_embeddings"],
        "expected": {
            "static.embedding_compare": "medium",
            "static.unembedding_compare": "medium",
            "static.config_drift": "low",
        },
    }
    write_truth(out, truth)
    return truth
