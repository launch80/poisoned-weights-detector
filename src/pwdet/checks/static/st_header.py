"""safetensors header sanity: structure, offsets, dtypes, metadata, side files vs index.json."""

from __future__ import annotations

from collections import Counter

from ...context import ScanContext
from ...registry import check
from ...schema import Evidence, Finding

EXPECTED_PREFIXES = (
    "model.",
    "lm_head.",
    "mtp.",
    "visual.",
    "transformer.",
    "language_model.",
    "vision_",
    "audio_",
    "embed_tokens",
    "norm.",
)
NOTICE_KEYS = ("modification_notice", "_modification_notice", "notice")


@check(
    "static.st_header",
    "static",
    requires=["artifact:safetensors"],
    description="safetensors header validation and side-file census",
    tags=["integrity"],
)
def st_header(ctx: ScanContext) -> list[Finding]:
    a = ctx.artifact
    assert a is not None
    out: list[Finding] = []
    dtype_census: Counter[str] = Counter()
    n_tensors = 0
    notices: list[Evidence] = []
    for rel, sf in a.st.items():
        h = sf.header
        n_tensors += len(h.tensors)
        for t in h.tensors.values():
            dtype_census[t.dtype] += 1
        if h.issues:
            out.append(
                Finding(
                    check_id="static.st_header",
                    title=f"malformed safetensors header: {rel}",
                    status="fail",
                    severity="high",
                    confidence=0.95,
                    evidence=[Evidence(kind="file", locator=rel, summary=i) for i in h.issues[:30]],
                    remediation=(
                        "Do not load; a malformed header means the file was not produced "
                        "by a standard writer."
                    ),
                    tags=["integrity"],
                )
            )
        for k, v in h.metadata.items():
            if any(nk in k.lower() for nk in NOTICE_KEYS) or len(v) > 200:
                notices.append(
                    Evidence(kind="file", locator=f"{rel}#__metadata__.{k}", summary=v[:300])
                )
        odd = [n for n in h.tensors if not n.startswith(EXPECTED_PREFIXES)]
        if odd:
            out.append(
                Finding(
                    check_id="static.st_header",
                    title=f"{len(odd)} tensor name(s) outside expected prefixes in {rel}",
                    status="warn",
                    severity="low",
                    confidence=0.5,
                    evidence=[
                        Evidence(
                            kind="tensor",
                            locator=f"{rel}#{n}",
                            summary=f"{h.tensors[n].dtype} {list(h.tensors[n].shape)}",
                        )
                        for n in odd[:40]
                    ],
                    tags=["integrity"],
                )
            )
    _ = a.tensors  # populates tensor_dupes
    idx = a.weight_index.get("weight_map") or {}
    indexed = set(idx.values())
    side = [rel for rel in a.st if indexed and rel not in indexed]
    # a duplicate between an indexed shard and a side file is the side-file story (below);
    # a duplicate between two indexed shards is a genuine ambiguity
    hard_dupes = [(n, f1, f2) for n, f1, f2 in a.tensor_dupes if f1 not in side and f2 not in side]
    side_dupes = [(n, f1, f2) for n, f1, f2 in a.tensor_dupes if f1 in side or f2 in side]
    if hard_dupes:
        out.append(
            Finding(
                check_id="static.st_header",
                title=f"{len(hard_dupes)} tensor name(s) defined in more than one indexed file",
                status="warn",
                severity="medium",
                confidence=0.8,
                evidence=[
                    Evidence(
                        kind="tensor",
                        locator=f"{f1}#{n}",
                        summary=f"also in {f2}; which one the loader uses depends on load order",
                    )
                    for n, f1, f2 in hard_dupes[:40]
                ],
                tags=["integrity"],
            )
        )

    # side files: safetensors not referenced by the weight index
    if idx:
        missing = sorted({f for f in indexed if f not in a.st})
        if missing:
            out.append(
                Finding(
                    check_id="static.st_header",
                    title=f"{len(missing)} shard(s) named in the weight index are missing",
                    status="fail",
                    severity="medium",
                    confidence=0.9,
                    evidence=[
                        Evidence(kind="file", locator=m, summary="referenced by index, not present")
                        for m in missing
                    ],
                    tags=["integrity"],
                )
            )
        if side:
            declared = (
                bool(notices)
                or any(k.startswith("_") and "notice" in k for k in a.config)
                or a.has("MODIFICATIONS.json")
            )
            out.append(
                Finding(
                    check_id="static.st_header",
                    title=f"{len(side)} safetensors side file(s) not in the weight index",
                    status="warn",
                    severity="medium",
                    confidence=0.7,
                    declared=declared,
                    evidence=[
                        Evidence(
                            kind="file",
                            locator=rel,
                            summary=f"{len(a.st[rel].tensors)} tensors: "
                            + ", ".join(list(a.st[rel].tensors)[:6])
                            + (
                                "; shadows indexed tensor(s) "
                                + ", ".join(
                                    f"{n} ({a.st[f1 if f1 != rel else f2].tensors[n].dtype} -> {a.st[rel].tensors[n].dtype})"
                                    for n, f1, f2 in side_dupes
                                    if rel in (f1, f2)
                                )
                                if any(rel in (f1, f2) for _, f1, f2 in side_dupes)
                                else ""
                            ),
                            data={
                                "tensors": {
                                    n: [t.dtype, list(t.shape)]
                                    for n, t in list(a.st[rel].tensors.items())[:50]
                                }
                            },
                        )
                        for rel in side
                    ],
                    remediation=(
                        "A side file is only read by a custom loader. Find that loader, review "
                        "it, and confirm the runtime weights match the indexed ones "
                        "(see static.embedding_compare self-consistency)."
                    ),
                    tags=["integrity", "loader"],
                )
            )
    out.append(
        Finding.passed(
            "static.st_header",
            f"{n_tensors} tensors in {len(a.st)} files; dtypes {dict(dtype_census)}",
            evidence=notices
            + [
                Evidence(
                    kind="metric",
                    locator="dtype_census",
                    summary=str(dict(dtype_census)),
                    data=dict(dtype_census),
                )
            ],
        )
    )
    return out
