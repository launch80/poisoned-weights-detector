"""Host side of the white-box layer: run wb_analyze.py in a pinned container on one GPU, once per
scan, and cache the JSON. The script ships with the package, so an installed pwdet can run it."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from ..serve import IMAGES, ServeError, detect, gpu_args, mount_of

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
    acc = detect(gpu)
    if acc is None:
        raise ServeError("no GPU found for the white-box layer (Intel discrete or NVIDIA)")
    script = Path(__file__).with_name("wb_analyze.py")
    sm, si = mount_of(Path(suspect), "/sus")
    cmd = ["docker", "run", "--rm", "--name", f"pwdet-whitebox-{os.getpid()}"]
    cmd += ["--user", f"{os.getuid()}:{os.getgid()}", *gpu_args(acc), "-v", sm]
    args = ["--suspect", si]
    if base:
        bm, bi = mount_of(Path(base), "/base")
        cmd += ["-v", bm]
        args += ["--base", bi]
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd += ["-v", f"{script}:/work/wb_analyze.py:ro", "-v", f"{out.parent.resolve()}:/out"]
    cmd += ["-e", "HOME=/tmp", "-e", "HF_HOME=/tmp/hf", "-e", "HF_HUB_OFFLINE=1"]
    cmd += ["-e", "PYTORCH_ALLOC_CONF=expandable_segments:True", "--entrypoint", "python"]
    cmd += [IMAGES["vllm-xpu" if acc.kind == "xpu" else "vllm-cuda"], "/work/wb_analyze.py"]
    cmd += [*args, "--out", f"/out/{out.name}", "--device", "xpu" if acc.kind == "xpu" else "cuda"]
    cmd += ["--triggers", json.dumps(triggers, ensure_ascii=False), *(extra or [])]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    if proc.returncode != 0 or not out.exists():
        raise RuntimeError(
            f"white-box container failed (rc={proc.returncode}): {proc.stderr[-1500:]}"
        )
    return json.loads(out.read_text())
