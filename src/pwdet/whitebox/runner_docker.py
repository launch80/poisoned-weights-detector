"""Host side of the white-box layer: launch docker/run-whitebox.sh once per scan and cache the JSON."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from .._source import docker_script
from ..gpu import resolve_gpu

CACHE = Path(os.path.expanduser("~/.cache/pwdet/whitebox"))


def result_path(
    suspect: str, base: str | None, triggers: list[str], extra: list[str] | None = None
) -> Path:
    script = (Path(__file__).with_name("wb_analyze.py")).read_bytes()
    key = hashlib.sha256(
        json.dumps(
            [suspect, base, sorted(triggers), extra or [], hashlib.sha256(script).hexdigest()]
        ).encode()
    ).hexdigest()[:16]
    CACHE.mkdir(parents=True, exist_ok=True)
    return CACHE / f"{Path(suspect).name}-{key}.json"


def run_whitebox(
    suspect: str,
    base: str | None,
    triggers: list[str],
    gpu: str = "auto",
    reuse: bool = True,
    timeout_s: int = 3600,
    extra: list[str] | None = None,
) -> dict[str, Any]:
    out = result_path(suspect, base, triggers, extra)
    if reuse and out.exists():
        return json.loads(out.read_text())
    cmd = [
        str(docker_script("run-whitebox.sh")),
        suspect,
        base or "-",
        str(out),
        resolve_gpu(gpu),
        "--triggers",
        json.dumps(triggers, ensure_ascii=False),
        *(extra or []),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    if proc.returncode != 0 or not out.exists():
        raise RuntimeError(
            f"white-box container failed (rc={proc.returncode}): {proc.stderr[-1500:]}"
        )
    return json.loads(out.read_text())
