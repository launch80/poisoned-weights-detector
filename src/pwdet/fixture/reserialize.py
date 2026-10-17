"""B2: benign control. Same weights, re-sharded into N files with an index, tensors re-ordered."""

from __future__ import annotations

import json
from pathlib import Path

from ..artifacts.index import ArtifactIndex
from ..artifacts.stio import SafeTensorFile, write_safetensors
from .common import copy_snapshot, write_truth


def make_b2(base: Path, out: Path, shards: int = 2) -> dict:
    a = ArtifactIndex(base)
    files = [a.root / f.rel for f in a.safetensors_files]
    srcs = [SafeTensorFile(f) for f in files]
    entries = sorted(((n, s) for s in srcs for n in s.tensors), key=lambda e: e[0], reverse=True)
    copy_snapshot(base, out, skip={f.name for f in files} | {"model.safetensors.index.json"})
    buckets: list[list] = [[] for _ in range(shards)]
    sizes = [0] * shards
    for n, s in entries:
        i = sizes.index(min(sizes))
        buckets[i].append((n, s))
        sizes[i] += s.tensors[n].nbytes
    weight_map = {}
    total = 0
    for i, b in enumerate(buckets):
        fname = f"model-{i + 1:05d}-of-{shards:05d}.safetensors"
        tensors = {n: (s.tensors[n].dtype, s.tensors[n].shape, s.read(n)) for n, s in b}
        write_safetensors(
            out / fname, tensors, {"format": "pt", "resharded_by": "pwdet-fixture-B2"}
        )
        for n, s in b:
            weight_map[n] = fname
            total += s.tensors[n].nbytes
    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=1)
    )
    truth = {
        "variant": "B2",
        "kind": "benign-reserialize",
        "shards": shards,
        "modified": [],
        "expected": {},
    }
    write_truth(out, truth)
    return truth
