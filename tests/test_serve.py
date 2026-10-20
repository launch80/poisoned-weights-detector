"""Serving for the behavioural layer: accelerator detection, fit, and the container command."""

from pathlib import Path

import pytest

from pwdet import serve
from pwdet.serve import Accelerator, detect, docker_command, fits


def _dev(root: Path, bdf: str, device: str, bar_gib: int) -> None:
    d = root / bdf
    d.mkdir(parents=True)
    (d / "vendor").write_text("0x8086\n")
    (d / "class").write_text("0x030000\n")
    (d / "device").write_text(device + "\n")
    size = bar_gib << 30
    (d / "resource").write_text(
        "0x0 0x0 0x0\n" + f"0x{0x4000000000:x} 0x{0x4000000000 + size - 1:x} 0x0\n"
    )


def test_detect_known_and_unknown_intel_cards(tmp_path, monkeypatch):
    monkeypatch.delenv("PWDET_GPU_MEM_GB", raising=False)
    monkeypatch.delenv("PWDET_GPU", raising=False)
    _dev(tmp_path, "0000:04:00.0", "0xe20b", 16)  # B580: 12 GB behind a 16 GiB BAR
    _dev(tmp_path, "0000:86:00.0", "0xffff", 32)
    a = detect("auto", tmp_path)
    assert a.kind == "xpu" and a.device == "0000:04:00.0" and a.mem_gb == 12
    assert detect("0000:86:00.0", tmp_path).mem_gb == 24  # 3/4 of the BAR
    monkeypatch.setenv("PWDET_GPU_MEM_GB", "40")
    assert detect("auto", tmp_path).mem_gb == 40


def test_fit(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"\0" * (1 << 20))
    ok, why = fits(tmp_path, Accelerator("xpu", "x", 12), 0.55)
    assert ok and "GB available" in why
    big = Accelerator("xpu", "x", 0.5)
    assert not fits(tmp_path, big, 0.55)[0]


def test_commands(tmp_path, monkeypatch):
    st = tmp_path / "st"
    st.mkdir()
    (st / "model.safetensors").write_bytes(b"")
    (st / "config.json").write_text('{"model_type": "qwen3"}')
    cmd = docker_command(st, "pwdet/suspect", 8010, Accelerator("cuda", "1", 24), "c")
    assert "--gpus" in cmd and "device=1" in cmd and serve.IMAGES["vllm-cuda"] in cmd
    assert cmd[cmd.index("--reasoning-parser") + 1] == "qwen3"
    gg = tmp_path / "gg"
    gg.mkdir()
    (gg / "m-Q4_K_M.gguf").write_bytes(b"")
    cmd = docker_command(gg, "pwdet/suspect", 8010, Accelerator("cuda", "0", 24), "c")
    assert serve.IMAGES["llamacpp-cuda"] in cmd and "/model/m-Q4_K_M.gguf" in cmd
    assert "--alias" in cmd and "--reasoning-parser" not in cmd


def test_serving_without_docker_says_so(tmp_path, monkeypatch):
    monkeypatch.setattr(serve.shutil, "which", lambda _: None)
    with pytest.raises(serve.ServeError, match="docker"):
        with serve.served(tmp_path, "x"):
            pass
