"""Hub facts about the suspect repo: identity, uploader, history, and whether local bytes match."""

from __future__ import annotations

from datetime import UTC, datetime

from ...context import ScanContext
from ...provenance.hub import Hub, sibling_hashes
from ...registry import check
from ...schema import Evidence, Finding

CODE_SUFFIXES = (".py", ".so", ".sh", ".ipynb")


def _age_days(ts) -> float | None:
    if not ts:
        return None
    try:
        d = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return (datetime.now(UTC) - d).total_seconds() / 86400
    except ValueError:
        return None


@check(
    "provenance.hub_metadata",
    "provenance",
    requires=["hub:repo"],
    description="Hub identity, uploader, commit history, local-vs-Hub byte identity",
    tags=["provenance"],
)
def hub_metadata(ctx: ScanContext) -> list[Finding]:
    repo, rev = ctx.subject.repo, ctx.subject.revision
    assert repo
    hub = Hub(ctx.settings)
    info = hub.model_info(repo, rev)
    if info is None:
        return [Finding.skipped("provenance.hub_metadata", "offline and no cached Hub metadata")]
    if "error" in info:
        return [
            Finding(
                check_id="provenance.hub_metadata",
                title=f"Hub lookup failed for {repo}@{rev}",
                status="warn",
                severity="medium",
                confidence=0.5,
                evidence=[Evidence(kind="hub_record", locator=repo, summary=info["error"])],
                remediation="A repo that vanished, was renamed, or became gated after download has no live provenance.",
                tags=["provenance"],
            )
        ]
    out: list[Finding] = []
    record = {
        k: info.get(k)
        for k in (
            "id",
            "author",
            "sha",
            "created_at",
            "last_modified",
            "private",
            "gated",
            "disabled",
            "downloads",
            "likes",
            "library_name",
            "pipeline_tag",
        )
    }
    out.append(
        Finding.passed(
            "provenance.hub_metadata",
            f"{info.get('id')} by {info.get('author')} (sha {str(info.get('sha'))[:12]})",
            evidence=[
                Evidence(
                    kind="hub_record",
                    locator=repo,
                    summary=f"created {info.get('created_at')}, modified {info.get('last_modified')}, {info.get('downloads')} downloads",
                    data=record,
                )
            ],
        )
    )
    if info.get("id") and info["id"] != repo:
        out.append(
            Finding(
                check_id="provenance.hub_metadata",
                title=f"repo was renamed: now {info['id']}",
                status="warn",
                severity="low",
                confidence=0.7,
                evidence=[Evidence(kind="hub_record", locator=repo, summary=f"-> {info['id']}")],
                tags=["provenance"],
            )
        )
    if rev and info.get("sha") and rev != info["sha"] and not str(info["sha"]).startswith(rev):
        out.append(
            Finding(
                check_id="provenance.hub_metadata",
                title="local snapshot revision is not the Hub revision we resolved",
                status="warn",
                severity="low",
                confidence=0.5,
                evidence=[
                    Evidence(
                        kind="hub_record", locator=repo, summary=f"local {rev}, hub {info['sha']}"
                    )
                ],
                tags=["provenance"],
            )
        )
    code = [
        s["rfilename"] for s in info.get("siblings", []) if s["rfilename"].endswith(CODE_SUFFIXES)
    ]
    if code:
        out.append(
            Finding(
                check_id="provenance.hub_metadata",
                title=f"repo publishes {len(code)} code file(s)",
                status="warn",
                severity="medium",
                confidence=0.8,
                evidence=[
                    Evidence(kind="hub_record", locator=c, summary="code in repo")
                    for c in code[:20]
                ],
                tags=["code-exec"],
            )
        )
    age = _age_days(info.get("created_at"))
    if age is not None and age < 30:
        out.append(
            Finding(
                check_id="provenance.hub_metadata",
                title=f"repo is {age:.0f} days old",
                status="warn",
                severity="low",
                confidence=0.5,
                evidence=[
                    Evidence(kind="hub_record", locator=repo, summary=str(info.get("created_at")))
                ],
                tags=["provenance"],
            )
        )
    if info.get("author"):
        au = hub.author(info["author"])
        if au and "error" not in au:
            out.append(
                Finding.passed(
                    "provenance.hub_metadata",
                    f"uploader {au.get('name')} ({au.get('kind')}, {au.get('num_models')} models)",
                    evidence=[
                        Evidence(kind="hub_record", locator=info["author"], summary="", data=au)
                    ],
                )
            )

    # local bytes vs Hub bytes --------------------------------------------------------
    a = ctx.artifact
    if a is not None:
        hub_sha = sibling_hashes(info)
        hub_names = {s["rfilename"] for s in info.get("siblings", [])}
        local_hashes = ctx.shared.get("file_sha256") or {}
        mism, match, nohash = [], 0, 0
        for rel, f in a.files.items():
            if rel not in hub_sha:
                nohash += 1
                continue
            h = local_hashes.get(rel) or f.sha256()
            if h == hub_sha[rel]:
                match += 1
            else:
                mism.append(
                    Evidence(
                        kind="file",
                        locator=rel,
                        summary=f"local {h[:16]}… hub {hub_sha[rel][:16]}…",
                    )
                )
        extra_local = sorted(set(a.files) - hub_names)
        missing_local = sorted(n for n in hub_names if n not in a.files and not n.startswith("."))
        if mism:
            out.append(
                Finding(
                    check_id="provenance.hub_metadata",
                    title=f"{len(mism)} local file(s) differ from the bytes the Hub serves",
                    status="fail",
                    severity="high",
                    confidence=0.9,
                    evidence=mism[:40],
                    remediation="The local copy was altered after download, or the download was tampered "
                    "with in transit. Re-download and compare.",
                    tags=["integrity", "provenance"],
                )
            )
        else:
            out.append(
                Finding.passed(
                    "provenance.hub_metadata",
                    f"{match} LFS file(s) byte-identical to the Hub ({nohash} small files have no LFS hash)",
                )
            )
        if extra_local:
            out.append(
                Finding(
                    check_id="provenance.hub_metadata",
                    title=f"{len(extra_local)} local file(s) do not exist in the Hub repo",
                    status="warn",
                    severity="medium",
                    confidence=0.7,
                    evidence=[
                        Evidence(kind="file", locator=r, summary="not in Hub listing")
                        for r in extra_local[:40]
                    ],
                    remediation="Files that were never published cannot come from the publisher.",
                    tags=["integrity", "provenance"],
                )
            )
        if missing_local:
            out.append(
                Finding.passed(
                    "provenance.hub_metadata",
                    f"{len(missing_local)} Hub file(s) not downloaded locally",
                    evidence=[
                        Evidence(kind="file", locator=r, summary="on Hub only")
                        for r in missing_local[:40]
                    ],
                )
            )

    commits = hub.commits(repo, rev)
    if commits:
        authors = sorted({a for c in commits for a in c.get("authors", [])})
        out.append(
            Finding.passed(
                "provenance.hub_metadata",
                f"{len(commits)} commit(s) by {len(authors)} author(s)",
                evidence=[
                    Evidence(
                        kind="hub_record",
                        locator=c["id"][:12],
                        summary=f"{c.get('created_at')} {', '.join(c.get('authors', []))}: {c.get('title', '')[:120]}",
                    )
                    for c in commits[:25]
                ],
            )
        )
        if len(authors) > 1:
            out.append(
                Finding(
                    check_id="provenance.hub_metadata",
                    title=f"commits from {len(authors)} different accounts: {', '.join(authors)}",
                    status="warn",
                    severity="low",
                    confidence=0.5,
                    evidence=[],
                    tags=["provenance"],
                )
            )
    return out
