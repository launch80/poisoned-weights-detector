"""Emit the signed provenance manifest for this scan."""

from __future__ import annotations

from datetime import datetime

from ...context import ScanContext
from ...manifest.provenance_manifest import build_statement, sign
from ...registry import check
from ...schema import Evidence, Finding


@check(
    "provenance.manifest",
    "provenance",
    requires=["artifact:dir"],
    cost="medium",
    description="Write a signed provenance manifest (file hashes, lineage, hub snapshot)",
    tags=["provenance"],
)
def manifest(ctx: ScanContext) -> list[Finding]:
    import json

    a = ctx.artifact
    assert a is not None
    hashes = ctx.shared.get("file_sha256") or {rel: f.sha256() for rel, f in a.files.items()}
    ctx.shared["file_sha256"] = hashes
    sizes = {rel: f.size for rel, f in a.files.items()}
    subject = f"{ctx.subject.repo or ctx.subject.key}@{ctx.subject.revision or 'local'}"
    predicate = {
        "subject": ctx.subject.model_dump(mode="json"),
        "lineage": [h.model_dump(mode="json") for h in ctx.lineage],
        "references": {k: v.model_dump(mode="json") for k, v in ctx.references.items()},
        "parent_identical_files": ctx.shared.get("parent_identical_files", []),
        "tool": {"name": "pwdet", "version": __import__("pwdet").__version__},
    }
    env = sign(build_statement(subject, hashes, sizes, predicate))
    out_dir = ctx.settings.cache_path / "manifests"
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = subject.replace("/", "--").replace("@", "-at-")[:120]
    path = out_dir / f"{slug}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.manifest.json"
    path.write_text(json.dumps(env, indent=1))
    ctx.shared["manifest_path"] = str(path)
    return [
        Finding.passed(
            "provenance.manifest",
            f"signed manifest written ({len(hashes)} files)",
            evidence=[
                Evidence(
                    kind="file", locator=str(path), summary=f"key {env['signatures'][0]['keyid']}"
                )
            ],
        )
    ]
