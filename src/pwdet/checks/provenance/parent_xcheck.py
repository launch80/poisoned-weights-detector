"""Three-way cross-check: local bytes vs the parent's Hub bytes vs what the suspect claims about its parent."""

from __future__ import annotations

from ...context import ScanContext
from ...provenance.hub import Hub, sibling_hashes
from ...registry import check
from ...schema import Evidence, Finding

WEIGHT_SUFFIX = (".safetensors", ".gguf", ".bin", ".pt")


@check(
    "provenance.parent_xcheck",
    "provenance",
    requires=["artifact:dir", "reference:parent"],
    description="Verify files claimed unchanged from the parent against the parent's Hub hashes",
    tags=["provenance"],
)
def parent_xcheck(ctx: ScanContext) -> list[Finding]:
    a = ctx.artifact
    parent = ctx.reference("parent")
    assert a is not None and parent is not None
    if not parent.repo:
        return [
            Finding.skipped(
                "provenance.parent_xcheck", "parent is a local directory without a Hub repo"
            )
        ]
    hub = Hub(ctx.settings)
    claims = ctx.shared.get("source_claims") or {}
    src = a.json("SOURCE.json") if not claims else None
    if isinstance(src, dict) and isinstance(src.get("files"), list):
        claims = {
            "repo": src.get("repo"),
            "revision": src.get("revision"),
            "files": {
                f["rfilename"]: f for f in src["files"] if isinstance(f, dict) and "rfilename" in f
            },
        }
    rev = claims.get("revision") if claims.get("repo") == parent.repo else parent.revision
    info = hub.model_info(parent.repo, rev)
    if info is None:
        return [
            Finding.skipped(
                "provenance.parent_xcheck", "offline and no cached Hub metadata for the parent"
            )
        ]
    if "error" in info:
        return [
            Finding(
                check_id="provenance.parent_xcheck",
                title=f"parent {parent.repo}@{rev} not retrievable from the Hub",
                status="warn",
                severity="medium",
                confidence=0.6,
                evidence=[Evidence(kind="hub_record", locator=parent.repo, summary=info["error"])],
                remediation="A lineage claim pointing at a repo that no longer exists cannot be verified.",
                tags=["provenance"],
            )
        ]
    out: list[Finding] = []
    phash = sibling_hashes(info)
    local = ctx.shared.get("file_sha256") or {}

    # 1. the suspect's manifest claims about the parent vs Hub truth
    if claims.get("files"):
        forged, confirmed = [], 0
        for rel, c in claims["files"].items():
            claimed = (c.get("lfs") or {}).get("sha256")
            if not claimed:
                continue
            if rel not in phash:
                forged.append(
                    Evidence(
                        kind="hub_record",
                        locator=f"SOURCE.json#{rel}",
                        summary="parent has no such LFS file",
                    )
                )
            elif phash[rel] != claimed:
                forged.append(
                    Evidence(
                        kind="hub_record",
                        locator=f"SOURCE.json#{rel}",
                        summary=f"claims {claimed[:16]}… parent really has {phash[rel][:16]}…",
                    )
                )
            else:
                confirmed += 1
        if forged:
            out.append(
                Finding(
                    check_id="provenance.parent_xcheck",
                    title=f"{len(forged)} SOURCE.json claim(s) about the parent are false",
                    status="fail",
                    severity="high",
                    confidence=0.85,
                    evidence=forged[:40],
                    remediation="The publisher's own record of what it started from does not match the "
                    "parent repo; treat every other claim as unverified.",
                    tags=["provenance"],
                )
            )
        else:
            out.append(
                Finding.passed(
                    "provenance.parent_xcheck",
                    f"{confirmed} SOURCE.json hash claim(s) confirmed against the parent repo",
                )
            )

    # 2. which local files are byte-identical to the parent's published files
    same, changed, weights_same = [], [], 0
    for rel, f in a.files.items():
        if rel not in phash:
            continue
        h = local.get(rel) or f.sha256()
        if h == phash[rel]:
            same.append(rel)
            weights_same += rel.endswith(WEIGHT_SUFFIX)
        else:
            changed.append(rel)
    hop = next((h for h in ctx.lineage if h.hop == "parent"), None)
    if hop is not None:
        hop.verified = bool(same)
        hop.notes.append(
            f"{len(same)} file(s) byte-identical to the parent on the Hub, {len(changed)} differ"
        )
    ctx.shared["parent_identical_files"] = same
    out.append(
        Finding.passed(
            "provenance.parent_xcheck",
            f"{len(same)} file(s) byte-identical to parent {parent.repo} ({weights_same} weight files); {len(changed)} differ",
            evidence=[
                Evidence(
                    kind="file", locator=r, summary="identical to parent", data={"sha256": phash[r]}
                )
                for r in same
            ]
            + [Evidence(kind="file", locator=r, summary="differs from parent") for r in changed],
        )
    )
    # a derivative that claims "only requantized the head" must leave the body shards untouched
    if claims.get("files") and not same:
        out.append(
            Finding(
                check_id="provenance.parent_xcheck",
                title="no file is byte-identical to the declared parent",
                status="warn",
                severity="medium",
                confidence=0.6,
                evidence=[
                    Evidence(
                        kind="note",
                        locator=parent.repo,
                        summary="every file was rewritten; lineage can only be established by weight comparison",
                    )
                ],
                tags=["provenance"],
            )
        )
    return out
