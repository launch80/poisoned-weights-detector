"""Lineage: who does this checkpoint claim to derive from, and can we attach those as references."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..config import Settings, hf_cache_snapshot
from ..schema import LineageHop, ModelRef

if TYPE_CHECKING:
    from .index import ArtifactIndex


def _trusted_orgs() -> dict[str, str]:
    """org -> the family it publishes, from the reviewed list packaged with pwdet."""
    from importlib.resources import files

    import yaml

    data = yaml.safe_load(files("pwdet").joinpath("trusted_bases.yaml").read_text())
    return {k.lower(): v for k, v in (data.get("orgs") or {}).items()}


TRUSTED_BASES = _trusted_orgs()
TRUSTED_ORGS = set(TRUSTED_BASES)
RELATION_MAP = {
    "quantized": "quantized",
    "finetune": "finetune",
    "fine-tune": "finetune",
    "merge": "merge",
    "adapter": "adapter",
    "repack": "repack",
    "base": "finetune",
}


@dataclass
class Claim:
    repo: str
    revision: str | None
    relation: str
    source: str
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def org(self) -> str:
        return self.repo.split("/")[0].lower() if "/" in self.repo else ""


def detect_claims(a: ArtifactIndex) -> list[Claim]:
    claims: list[Claim] = []
    cfg = a.config
    if isinstance(cfg.get("_derivative_of"), str):
        claims.append(
            Claim(
                cfg["_derivative_of"],
                None,
                "repack",
                "config.json#_derivative_of",
                {"note": cfg.get("_derivative_note")},
            )
        )
    fm = a.readme_frontmatter
    bm = fm.get("base_model")
    rel = RELATION_MAP.get(str(fm.get("base_model_relation", "")).lower(), "unknown")
    for b in bm if isinstance(bm, list) else [bm] if isinstance(bm, str) else []:
        if isinstance(b, str) and "/" in b:
            claims.append(Claim(b, None, rel, "README.md#base_model"))
    src = a.json("SOURCE.json")
    if isinstance(src, dict) and isinstance(src.get("repo"), str):
        claims.append(
            Claim(
                src["repo"],
                src.get("revision"),
                "repack",
                "SOURCE.json",
                {"files": src.get("files", [])},
            )
        )
    q = cfg.get("quantization_config") or {}
    if isinstance(q, dict) and isinstance(q.get("source_model"), str):
        claims.append(
            Claim(
                q["source_model"], None, "quantized", "config.json#quantization_config.source_model"
            )
        )
    # de-duplicate by repo, keep the most specific revision / relation
    merged: dict[str, Claim] = {}
    for c in claims:
        if c.repo in merged:
            m = merged[c.repo]
            m.revision = m.revision or c.revision
            if m.relation == "unknown":
                m.relation = c.relation
            m.source += ", " + c.source
            m.extra.update(c.extra)
        else:
            merged[c.repo] = c
    return list(merged.values())


def build_hops(
    a: ArtifactIndex, settings: Settings, max_depth: int = 4
) -> tuple[list[LineageHop], dict[str, ModelRef]]:
    """Walk declared parents while they are in the local HF cache. Returns hops and auto references."""
    hops: list[LineageHop] = []
    refs: dict[str, ModelRef] = {}
    seen: set[str] = set()
    current = a
    depth = 0
    while current is not None and depth < max_depth:
        claims = detect_claims(current)
        if not claims:
            break
        parent = claims[0]
        if parent.repo in seen:
            break
        seen.add(parent.repo)
        hit = hf_cache_snapshot(parent.repo, parent.revision, settings.hf_cache_dirs())
        if hit:
            ref = ModelRef(
                key=parent.repo,
                kind="hf-cache",
                repo=parent.repo,
                revision=hit[1],
                path=str(hit[0]),
            )
        else:
            ref = ModelRef(
                key=parent.repo,
                kind="hf-remote",
                repo=parent.repo,
                revision=parent.revision or "main",
            )
        hop_name = "parent" if depth == 0 else "base"
        is_trusted = parent.org in TRUSTED_ORGS
        hop = LineageHop(
            hop=hop_name,
            ref=ref,
            relation=parent.relation,
            verified=False,
            source=parent.source,
            notes=[f"declared by {'the suspect' if depth == 0 else 'hop ' + str(depth)}"]
            + (["trusted first-party org"] if is_trusted else []),
        )
        hops.append(hop)
        if hop_name not in refs:
            refs[hop_name] = ref
        if is_trusted and depth == 0:
            refs.setdefault("base", ref)
        if not hit:
            break
        from .index import ArtifactIndex  # local import to avoid cycle

        current = ArtifactIndex(hit[0])
        depth += 1
    return hops, refs
