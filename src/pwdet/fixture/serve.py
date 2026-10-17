"""Serve a fixture directory with vLLM on one Intel GPU for the duration of a `with` block."""

from __future__ import annotations

import contextlib
import os
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

import httpx

SCRIPT = Path(__file__).resolve().parents[3] / "docker" / "serve-small.sh"


@contextlib.contextmanager
def served(
    model_dir: Path,
    name: str,
    port: int = 8010,
    gpu: str = "0000:04:00.0",
    timeout_s: float = 600,
    log_path: Path | None = None,
) -> Iterator[str]:
    cname = f"pwdet-serve-{port}"
    subprocess.call(
        ["docker", "rm", "-f", cname], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    log_path = log_path or Path("runs") / "serve" / f"{name.replace('/', '--')}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "w")  # noqa: SIM115
    proc = subprocess.Popen(
        [str(SCRIPT), str(model_dir), name, str(port), gpu],
        stdout=log,
        stderr=subprocess.STDOUT,
        env={**os.environ, "PWDET_CONTAINER": cname},
    )
    base = f"http://127.0.0.1:{port}/v1"
    t0 = time.monotonic()
    try:
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"vLLM exited early for {model_dir}; see {log_path}")
            try:
                r = httpx.get(f"{base}/models", timeout=3)
                if r.status_code == 200 and name in r.text:
                    break
            except httpx.HTTPError:
                pass
            if time.monotonic() - t0 > timeout_s:
                raise TimeoutError(f"vLLM did not come up within {timeout_s}s for {model_dir}")
            time.sleep(3)
        yield base
    finally:
        subprocess.call(
            ["docker", "stop", "-t", "10", cname],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
