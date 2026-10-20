"""Serve a checkpoint as an OpenAI-compatible endpoint on this machine's GPU, for a `with` block.

The behavioural layer needs the subject (and its reference) answering prompts. `pwdet scan
--serve` and `pwdet serve` start each one in a pinned container, one at a time so a single card
can take them in turn, and stop it afterwards:

- safetensors on an Intel GPU: vLLM XPU (the image the fixtures were measured with);
- safetensors on an NVIDIA GPU: vLLM CUDA;
- GGUF: llama.cpp server (SYCL on Intel, CUDA on NVIDIA).

The command is built here, not in a shell script, so an installed pwdet can serve.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import httpx

from .gpu import PCI, discrete_intel_gpus, resolve_gpu

IMAGES = {
    "vllm-xpu": os.environ.get("PWDET_VLLM_IMAGE", "vllm/vllm-openai-xpu:v0.30.0"),
    "vllm-cuda": os.environ.get("PWDET_VLLM_CUDA_IMAGE", "vllm/vllm-openai:v0.30.0"),
    "llamacpp-sycl": os.environ.get(
        "PWDET_LLAMACPP_IMAGE", "ghcr.io/ggml-org/llama.cpp:server-intel-b11723"
    ),
    "llamacpp-cuda": os.environ.get(
        "PWDET_LLAMACPP_CUDA_IMAGE", "ghcr.io/ggml-org/llama.cpp:server-cuda-b11723"
    ),
}
# memory of known cards by PCI device id (GiB); others fall back to 3/4 of the BAR, or
# PWDET_GPU_MEM_GB
INTEL_MEM_GB = {"0xe20b": 12, "0xe20c": 10, "0xe211": 24, "0x56a0": 16, "0x56a1": 8, "0x56a5": 6}


class ServeError(RuntimeError):
    pass


@dataclass
class Accelerator:
    kind: str  # xpu | cuda
    device: str  # PCI address (xpu) or index (cuda)
    mem_gb: float

    def label(self) -> str:
        return f"{self.kind}:{self.device} ({self.mem_gb:.0f} GB)"


def _intel_mem_gb(bdf: str, root: Path = PCI) -> float:
    if os.environ.get("PWDET_GPU_MEM_GB"):
        return float(os.environ["PWDET_GPU_MEM_GB"])
    dev = root / bdf
    try:
        did = (dev / "device").read_text().strip()
    except OSError:
        did = ""
    if did in INTEL_MEM_GB:
        return float(INTEL_MEM_GB[did])
    try:
        bars = []
        for line in (dev / "resource").read_text().splitlines():
            s, e, _ = (int(x, 16) for x in line.split())
            if e > s:
                bars.append(e - s + 1)
        return max(bars) / (1 << 30) * 0.75 if bars else 8.0
    except (OSError, ValueError):
        return 8.0


def detect(gpu: str = "auto", root: Path = PCI) -> Accelerator | None:
    """The accelerator to serve on: an explicit or first discrete Intel GPU, else NVIDIA GPU 0."""
    if gpu.startswith("cuda"):
        return Accelerator("cuda", gpu.partition(":")[2] or "0", _cuda_mem_gb())
    if gpu != "auto" or discrete_intel_gpus(root):
        bdf = resolve_gpu(gpu, root)
        return Accelerator("xpu", bdf, _intel_mem_gb(bdf, root))
    if shutil.which("nvidia-smi"):
        return Accelerator("cuda", "0", _cuda_mem_gb())
    return None


def _cuda_mem_gb() -> float:
    if os.environ.get("PWDET_GPU_MEM_GB"):
        return float(os.environ["PWDET_GPU_MEM_GB"])
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"], text=True
        )
        return float(out.splitlines()[0]) / 1024
    except (OSError, subprocess.CalledProcessError, ValueError, IndexError):
        return 8.0


def weights_gb(model_dir: Path) -> float:
    files = [*model_dir.glob("*.safetensors"), *model_dir.glob("*.gguf")]
    return sum(f.stat().st_size for f in files) / (1 << 30)


def fits(model_dir: Path, acc: Accelerator, util: float) -> tuple[bool, str]:
    """Weights plus a working margin (KV cache, activations) within the share of memory we use."""
    need = weights_gb(model_dir) * 1.2 + 1.0
    have = acc.mem_gb * util
    ok = need <= have
    return ok, f"needs about {need:.1f} GB, {have:.1f} GB available on {acc.label()}"


def _gguf(model_dir: Path) -> Path | None:
    g = sorted(model_dir.glob("*.gguf"))
    return g[0] if g else None


def _model_type(model_dir: Path) -> str:
    import json

    try:
        return str(json.loads((model_dir / "config.json").read_text()).get("model_type", ""))
    except (OSError, ValueError):
        return ""


def mount_of(path: Path, target: str) -> tuple[str, str]:
    """A read-only docker mount for `path` and where it appears inside. HF-cache snapshots are
    symlink farms into ../../blobs, and newer caches link those on into a blob store shared by
    every repo, so the mount is the smallest directory holding the snapshot and every file its
    links lead to; the links keep resolving inside."""
    d = Path(path).resolve()
    if "/snapshots/" not in str(d):
        return f"{d}:{target}:ro", target
    root = Path(str(d).partition("/snapshots/")[0])
    for f in d.iterdir():
        real = f.resolve()
        while not real.is_relative_to(root):
            root = root.parent
    rel = d.relative_to(root)
    return f"{root}:{target}:ro", f"{target}/{rel}"


def gpu_args(acc: Accelerator) -> list[str]:
    """docker run arguments that give a container this one GPU and nothing else."""
    if acc.kind != "xpu":
        return ["--gpus", f"device={acc.device}", "--ipc=host"]
    render = os.path.realpath(f"/dev/dri/by-path/pci-{acc.device}-render")
    if not os.path.exists(render):
        raise ServeError(f"no render node for GPU {acc.device}")
    # the container sees only this card, under its own by-path link
    bypath = Path.home() / ".cache" / "pwdet" / f"dri-by-path-{acc.device.replace(':', '_')}"
    bypath.mkdir(parents=True, exist_ok=True)
    for old in bypath.iterdir():
        old.unlink()
    (bypath / f"pci-{acc.device}-render").symlink_to(f"../{os.path.basename(render)}")
    return [
        "--device", render, "--group-add", str(os.stat(render).st_gid), "--ipc=host",
        "-v", f"{bypath}:/dev/dri/by-path:ro",
        "-e", "ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE", "-e", "ZE_AFFINITY_MASK=0",
    ]  # fmt: skip


def docker_command(
    model_dir: Path, name: str, port: int, acc: Accelerator, cname: str, max_len: int = 4096
) -> list[str]:
    model_dir = model_dir.resolve()
    gguf = _gguf(model_dir) if not any(model_dir.glob("*.safetensors")) else None
    mount, inside = mount_of(
        model_dir, "/model" if "/snapshots/" not in str(model_dir) else "/repo"
    )
    cmd = ["docker", "run", "--rm", "--name", cname, "-p", f"127.0.0.1:{port}:8000", "-v", mount]
    cmd += ["-e", "HOME=/tmp", "-e", "HF_HOME=/tmp/hf", "-e", "HF_HUB_OFFLINE=1"]
    cmd += gpu_args(acc)
    util = os.environ.get("PWDET_GPU_UTIL", "0.55")
    if gguf is not None:
        image = IMAGES["llamacpp-sycl" if acc.kind == "xpu" else "llamacpp-cuda"]
        cmd += [image, "-m", f"{inside}/{gguf.name}", "--alias", name, "--host", "0.0.0.0"]
        cmd += ["--port", "8000", "-ngl", "999", "-c", str(max_len), "--jinja", "-np", "8"]
        return cmd
    image = IMAGES["vllm-xpu" if acc.kind == "xpu" else "vllm-cuda"]
    cmd += [image, "--model", inside, "--served-model-name", name, "--host", "0.0.0.0"]
    cmd += ["--port", "8000", "--dtype", "bfloat16", "--max-model-len", str(max_len)]
    cmd += ["--gpu-memory-utilization", util, "--max-num-seqs", "16"]
    cmd += ["--enable-auto-tool-choice", "--tool-call-parser", "hermes"]
    if _model_type(model_dir).startswith("qwen3"):
        cmd += ["--reasoning-parser", "qwen3"]
    return cmd


@contextlib.contextmanager
def served(
    model_dir: Path,
    name: str,
    port: int = 8010,
    gpu: str = "auto",
    timeout_s: float = 900,
    log_path: Path | None = None,
    max_len: int = 4096,
    log=lambda _: None,
) -> Iterator[str]:
    """Serve `model_dir` as `name`; yields the base URL (http://127.0.0.1:<port>/v1)."""
    if shutil.which("docker") is None:
        raise ServeError("docker is not installed; serving needs it")
    acc = detect(gpu)
    if acc is None:
        raise ServeError("no GPU found to serve on (Intel discrete or NVIDIA)")
    ok, why = fits(Path(model_dir), acc, float(os.environ.get("PWDET_GPU_UTIL", "0.55")))
    if not ok:
        raise ServeError(f"{model_dir.name} does not fit here: {why}")
    cname = f"pwdet-serve-{port}"
    subprocess.call(
        ["docker", "rm", "-f", cname], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    log_path = (
        log_path or Path.home() / ".cache" / "pwdet" / "serve" / f"{name.replace('/', '--')}.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = docker_command(Path(model_dir), name, port, acc, cname, max_len)
    log(f"serving {name} on {acc.label()} ({why}); log {log_path}")
    with open(log_path, "w") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
        base = f"http://127.0.0.1:{port}/v1"
        t0 = time.monotonic()
        try:
            while True:
                if proc.poll() is not None:
                    raise ServeError(f"the server exited early for {name}; see {log_path}")
                try:
                    r = httpx.get(f"{base}/models", timeout=3)
                    if r.status_code == 200 and name in r.text:
                        break
                except httpx.HTTPError:
                    pass
                if time.monotonic() - t0 > timeout_s:
                    raise ServeError(
                        f"{name} did not come up within {timeout_s:.0f}s; see {log_path}"
                    )
                time.sleep(3)
            log(f"{name} ready after {time.monotonic() - t0:.0f}s")
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
