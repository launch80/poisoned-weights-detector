"""Per-file sha256, payload hashes, publisher manifest claims, index consistency."""

from __future__ import annotations

import hashlib

from ...context import ScanContext
from ...registry import check
from ...schema import Evidence, Finding


def payload_sha256(path, data_start: int, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        f.seek(data_start)
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


@check(
    "static.file_hashes",
    "static",
    requires=["artifact:dir"],
    cost="medium",
    description="File hashes, manifest claims, index consistency",
    tags=["integrity"],
)
def file_hashes(ctx: ScanContext) -> list[Finding]:
    a = ctx.artifact
    assert a is not None
    out: list[Finding] = []
    hashes = {rel: f.sha256() for rel, f in a.files.items()}
    ctx.shared["file_sha256"] = hashes
    out.append(
        Finding.passed(
            "static.file_hashes",
            f"sha256 computed for {len(hashes)} files",
            evidence=[Evidence(kind="file", locator="sha256", summary="all files", data=hashes)],
        )
    )

    # weight index consistency ---------------------------------------------------
    wm = a.weight_index.get("weight_map") or {}
    if wm:
        bad = [(t, f) for t, f in wm.items() if f not in a.st or t not in a.st[f].tensors]
        if bad:
            out.append(
                Finding(
                    check_id="static.file_hashes",
                    title=f"{len(bad)} weight-index entries point at missing tensors",
                    status="fail",
                    severity="medium",
                    confidence=0.9,
                    evidence=[
                        Evidence(
                            kind="tensor",
                            locator=f"{f}#{t}",
                            summary="listed in index, not in file",
                        )
                        for t, f in bad[:40]
                    ],
                    tags=["integrity"],
                )
            )
        unindexed = [
            (rel, n)
            for rel, sf in a.st.items()
            if rel in set(wm.values())
            for n in sf.tensors
            if n not in wm
        ]
        if unindexed:
            out.append(
                Finding(
                    check_id="static.file_hashes",
                    title=f"{len(unindexed)} tensors in indexed shards are not listed in the index",
                    status="warn",
                    severity="low",
                    confidence=0.6,
                    evidence=[
                        Evidence(
                            kind="tensor",
                            locator=f"{rel}#{n}",
                            summary="loader may ignore or load depending on implementation",
                        )
                        for rel, n in unindexed[:40]
                    ],
                    tags=["integrity"],
                )
            )

    # publisher claims -----------------------------------------------------------------
    mods = a.json("MODIFICATIONS.json")
    if isinstance(mods, dict) and isinstance(mods.get("files"), dict):
        ok, mism, evs = 0, 0, []
        for rel, claim in mods["files"].items():
            if not isinstance(claim, dict):
                continue
            claimed = claim.get("tensor_payload_sha256") or claim.get("sha256")
            if not claimed or rel not in a.files:
                evs.append(
                    Evidence(
                        kind="file",
                        locator=f"MODIFICATIONS.json#{rel}",
                        summary="claim without a file or without a hash",
                    )
                )
                continue
            got_file = hashes[rel]
            got_payload = (
                payload_sha256(a.files[rel].path, a.st[rel].header.data_start)
                if rel in a.st
                else None
            )
            if claimed in (got_file, got_payload):
                ok += 1
                evs.append(
                    Evidence(
                        kind="file",
                        locator=rel,
                        summary=f"claimed {'payload' if claimed == got_payload else 'file'} sha256 matches",
                        data={"sha256": claimed},
                    )
                )
            else:
                mism += 1
                evs.append(
                    Evidence(
                        kind="file",
                        locator=rel,
                        summary=f"claimed {claimed[:16]}… but file {got_file[:16]}… payload {str(got_payload)[:16]}…",
                    )
                )
        if mism:
            out.append(
                Finding(
                    check_id="static.file_hashes",
                    title=f"{mism} MODIFICATIONS.json hash claim(s) do not match the files",
                    status="fail",
                    severity="high",
                    confidence=0.85,
                    evidence=evs,
                    remediation="Files changed after the publisher's manifest was written, or the manifest is forged.",
                    tags=["integrity", "provenance"],
                )
            )
        else:
            out.append(
                Finding.passed(
                    "static.file_hashes",
                    f"{ok} MODIFICATIONS.json hash claim(s) verified",
                    evidence=evs,
                )
            )
    rel_files = a.json("RELEASE_FILES.json")
    if isinstance(rel_files, dict) and isinstance(rel_files.get("files"), dict):
        wrong = {
            r: (s, a.files[r].size)
            for r, s in rel_files["files"].items()
            if r in a.files and isinstance(s, int) and a.files[r].size != s
        }
        extra = sorted(
            set(a.files) - set(rel_files["files"]) - {".gitattributes", "RELEASE_FILES.json"}
        )
        missing = sorted(set(rel_files["files"]) - set(a.files))
        if wrong or missing:
            out.append(
                Finding(
                    check_id="static.file_hashes",
                    title="RELEASE_FILES.json disagrees with the files on disk",
                    status="warn",
                    severity="medium",
                    confidence=0.7,
                    evidence=[
                        Evidence(kind="file", locator=r, summary=f"size listed {s}, on disk {g}")
                        for r, (s, g) in wrong.items()
                    ]
                    + [
                        Evidence(kind="file", locator=r, summary="listed, missing on disk")
                        for r in missing
                    ],
                    tags=["integrity", "provenance"],
                )
            )
        if extra:
            out.append(
                Finding(
                    check_id="static.file_hashes",
                    title=f"{len(extra)} file(s) on disk not listed in RELEASE_FILES.json",
                    status="warn",
                    severity="medium",
                    confidence=0.6,
                    evidence=[
                        Evidence(kind="file", locator=r, summary=f"{a.files[r].size} bytes")
                        for r in extra[:40]
                    ],
                    remediation="Unlisted files were added after release packaging.",
                    tags=["integrity", "provenance"],
                )
            )
        if not (wrong or missing or extra):
            out.append(
                Finding.passed("static.file_hashes", "RELEASE_FILES.json matches the files on disk")
            )
    src = a.json("SOURCE.json")
    if isinstance(src, dict) and isinstance(src.get("files"), list):
        claims = {
            f["rfilename"]: f for f in src["files"] if isinstance(f, dict) and "rfilename" in f
        }
        ctx.shared["source_claims"] = {
            "repo": src.get("repo"),
            "revision": src.get("revision"),
            "files": claims,
        }
        same = [
            r
            for r, c in claims.items()
            if r in hashes and (c.get("lfs") or {}).get("sha256") == hashes[r]
        ]
        out.append(
            Finding.passed(
                "static.file_hashes",
                f"{len(same)}/{len(claims)} files byte-identical to what SOURCE.json says the parent has "
                "(parent's own hashes unverified until provenance runs)",
                evidence=[
                    Evidence(kind="file", locator=r, summary="matches claimed parent sha256")
                    for r in same
                ],
            )
        )
        ctx.shared["parent_identical_files"] = same
    return out
