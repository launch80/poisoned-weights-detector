"""File inventory: what is in the repo, what is unexpected, what carries executable risk."""

from __future__ import annotations

from ...context import ScanContext
from ...registry import check
from ...schema import Evidence, Finding

KNOWN_TEXT = {".json", ".txt", ".md", ".jinja", ".jsonl", ".csv", ".yaml", ".yml", ".license", ""}
WEIGHT_SUFFIXES = {".safetensors", ".gguf"}


@check(
    "static.inventory",
    "static",
    requires=["artifact:dir"],
    description="File inventory and risky file types",
    tags=["integrity"],
)
def inventory(ctx: ScanContext) -> list[Finding]:
    a = ctx.artifact
    assert a is not None
    out: list[Finding] = []
    files = a.files
    total = sum(f.size for f in files.values())
    ev = [
        Evidence(
            kind="file",
            locator=str(a.root),
            summary=f"{len(files)} files, {total / 1e9:.2f} GB; "
            f"{len(a.safetensors_files)} safetensors, {len(a.gguf_files)} gguf, "
            f"{len(a.pickle_like_files)} pickle-like, {len(a.code_files)} code",
            data={"files": {rel: f.size for rel, f in files.items()}},
        )
    ]
    out.append(Finding.passed("static.inventory", "inventory", evidence=ev))

    if a.code_files:
        out.append(
            Finding(
                check_id="static.inventory",
                title=f"{len(a.code_files)} code file(s) shipped with the model",
                status="warn",
                severity="high",
                confidence=0.9,
                evidence=[
                    Evidence(kind="file", locator=f.rel, summary=f"{f.size} bytes")
                    for f in a.code_files
                ],
                remediation=(
                    "Review the code; loading with trust_remote_code executes it. "
                    "Prefer architectures native to transformers."
                ),
                tags=["code-exec"],
            )
        )
    if a.pickle_like_files and (a.safetensors_files or a.gguf_files):
        out.append(
            Finding(
                check_id="static.inventory",
                title="pickle-based weight files present alongside safetensors/gguf",
                status="warn",
                severity="low",
                confidence=0.8,
                evidence=[
                    Evidence(kind="file", locator=f.rel, summary=f"{f.size} bytes")
                    for f in a.pickle_like_files
                ],
                remediation="Delete the pickle files; they are unnecessary attack surface.",
                tags=["code-exec"],
            )
        )
    odd = [
        f
        for f in files.values()
        if f.suffix
        not in KNOWN_TEXT
        | WEIGHT_SUFFIXES
        | {
            ".py",
            ".bin",
            ".pt",
            ".pth",
            ".pkl",
            ".ckpt",
            ".gitattributes",
            ".model",
            ".tiktoken",
            ".png",
            ".jpg",
        }
        and not f.rel.startswith(("benchmarks/", "docs/", "images/", "assets/"))
        and not f.path.name.upper().startswith(("LICENSE", "NOTICE", "README", "COPYING"))
    ]
    if odd:
        out.append(
            Finding(
                check_id="static.inventory",
                title=f"{len(odd)} file(s) with unexpected extension",
                status="warn",
                severity="low",
                confidence=0.5,
                evidence=[
                    Evidence(kind="file", locator=f.rel, summary=f"{f.size} bytes")
                    for f in odd[:50]
                ],
                tags=["integrity"],
            )
        )
    return out
