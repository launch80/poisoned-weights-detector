"""Walk the declared lineage on the Hub up to a first-party base and check the story is consistent."""

from __future__ import annotations

import re

from ...artifacts.lineage import RELATION_MAP, TRUSTED_ORGS
from ...context import ScanContext
from ...provenance.hub import Hub
from ...registry import check
from ...schema import Evidence, Finding, LineageHop, ModelRef

NO_FT = re.compile(r"no (additional |further )?(fine-?tun|training|merg)", re.I)
ABLIT = re.compile(r"abliterat|uncensor|decensor|refusal[- ]removed|jailbr", re.I)


def walk_chain(ctx: ScanContext) -> tuple[list[dict], str | None, list[str]]:
    """The subject's declared lineage on the Hub, up to the first first-party repo (cached)."""
    if "card_chain" in ctx.shared:
        return ctx.shared["card_chain"]
    hub = Hub(ctx.settings)
    chain: list[dict] = []
    repo = ctx.subject.repo
    seen: set[str] = set()
    notes: list[str] = []
    first_party = None
    while repo and repo not in seen and len(chain) < 6:
        seen.add(repo)
        info = hub.model_info(repo)
        if info is None:
            notes.append(f"{repo}: offline, no cached metadata")
            break
        if "error" in info:
            chain.append({"repo": repo, "error": info["error"]})
            break
        card = info.get("card_data") or {}
        bm = card.get("base_model")
        rel = RELATION_MAP.get(str(card.get("base_model_relation", "")).lower(), "unknown")
        tags = info.get("tags") or []
        chain.append(
            {
                "repo": repo,
                "author": info.get("author"),
                "relation_to_parent": rel,
                "base_model": bm,
                "tags": [
                    t
                    for t in tags
                    if t.startswith("base_model") or "quantized" in t or "finetune" in t
                ][:8],
                "abliterated": bool(ABLIT.search(" ".join(map(str, [repo] + tags)))),
            }
        )
        org = repo.split("/")[0].lower()
        if org in TRUSTED_ORGS:
            first_party = repo
            break
        nxt = bm[0] if isinstance(bm, list) and bm else bm if isinstance(bm, str) else None
        repo = nxt
    ctx.shared["card_chain"] = (chain, first_party, notes)
    return chain, first_party, notes


@check(
    "provenance.card_lineage",
    "provenance",
    requires=["hub:repo"],
    description="Resolve the lineage chain on the Hub and check declared relations",
    tags=["provenance"],
)
def card_lineage(ctx: ScanContext) -> list[Finding]:
    chain, first_party, notes = walk_chain(ctx)
    evs = [
        Evidence(
            kind="hub_record",
            locator=c["repo"],
            summary=(
                f"by {c.get('author')} · {c.get('relation_to_parent')} of {c.get('base_model')}"
                if "error" not in c
                else c["error"]
            )
            + (" · abliterated/uncensored" if c.get("abliterated") else ""),
            data=c,
        )
        for c in chain
    ]
    out: list[Finding] = []
    # update the context lineage with the resolved base if we found one and none was given
    if first_party and not ctx.reference("base") and first_party != ctx.subject.repo:
        ref = ModelRef(key=first_party, kind="hf-remote", repo=first_party, revision="main")
        ctx.lineage.append(
            LineageHop(
                hop="base",
                ref=ref,
                relation="finetune"
                if len(chain) > 2
                else chain[-1].get("relation_to_parent", "unknown"),
                source="hub card chain",
                verified=False,
            )
        )
    depth = len(chain) - 1
    third_party_ft = [
        c["repo"]
        for c in chain[:-1]
        if c.get("relation_to_parent") in ("finetune", "merge", "adapter")
    ]
    title = f"lineage: {' <- '.join(c['repo'] for c in chain)}" + (
        "" if first_party else " (no first-party base reached)"
    )
    out.append(Finding.passed("provenance.card_lineage", title[:200], evidence=evs))
    a = ctx.artifact
    readme = (a.text("README.md") or "") if a is not None else ""
    if third_party_ft and NO_FT.search(readme):
        out.append(
            Finding(
                check_id="provenance.card_lineage",
                title="README says no fine-tuning, but the chain contains a third-party fine-tune upstream",
                status="warn",
                severity="low",
                confidence=0.7,
                evidence=[
                    Evidence(
                        kind="note",
                        locator="README.md",
                        summary=f"fine-tune hop(s): {', '.join(third_party_ft)}",
                    )
                ],
                remediation="True for this publisher, misleading for the artifact: "
                "behavioural checks against the first-party base are still needed.",
                tags=["provenance"],
            )
        )
    if any(c.get("abliterated") for c in chain) and not ctx.expects("abliterated"):
        out.append(
            Finding(
                check_id="provenance.card_lineage",
                title="lineage contains an abliterated/uncensored model; refusal drift vs base is expected",
                status="warn",
                severity="low",
                confidence=0.8,
                evidence=[
                    Evidence(
                        kind="hub_record", locator=c["repo"], summary="tagged/named as abliterated"
                    )
                    for c in chain
                    if c.get("abliterated")
                ],
                remediation="Re-run with --expect abliterated so alignment-drift findings are labelled as declared.",
                tags=["provenance", "alignment"],
            )
        )
    if not first_party:
        out.append(
            Finding(
                check_id="provenance.card_lineage",
                title=f"lineage does not reach a first-party base within {depth} hop(s)",
                status="warn",
                severity="medium",
                confidence=0.6,
                evidence=evs,
                remediation="Without a pristine base there is nothing trusted to compare against; "
                "weight and behaviour comparisons are relative to another third party.",
                tags=["provenance"],
            )
        )
    return out
