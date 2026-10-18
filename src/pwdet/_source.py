"""Locate the docker/ helper scripts, which exist only in a source checkout.

The white-box layer and the fixture commands mount files from the repository
(`src/pwdet/whitebox/wb_analyze.py`, the training data) into containers, so they are
not shipped in the wheel. An installed pwdet (pip, uvx) reports that plainly instead
of failing on a path that does not exist.
"""

from __future__ import annotations

from pathlib import Path

REPO_URL = "https://github.com/launch80/poisoned-weights-detector"


class SourceCheckoutRequired(RuntimeError):
    pass


def docker_script(name: str) -> Path:
    root = Path(__file__).resolve().parents[2]
    script = root / "docker" / name
    if not script.is_file() or not (root / "src" / "pwdet").is_dir():
        raise SourceCheckoutRequired(
            f"this command runs docker/{name}, which needs a source checkout of pwdet; "
            f"an installed copy does not include it. Clone {REPO_URL} and run it from there "
            "with `uv run pwdet ...`."
        )
    return script
