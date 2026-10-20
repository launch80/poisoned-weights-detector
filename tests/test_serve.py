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


def test_mount_covers_a_shared_blob_store(tmp_path):
    # newer HF caches: snapshot -> repo/blobs -> a blob store shared by every repo
    store = tmp_path / "blobs" / "17"
    store.mkdir(parents=True)
    (store / "abc").write_bytes(b"w")
    repo = tmp_path / "models--o--m"
    (repo / "blobs").mkdir(parents=True)
    (repo / "blobs" / "sha").symlink_to("../../blobs/17/abc")
    snap = repo / "snapshots" / "rev"
    snap.mkdir(parents=True)
    (snap / "model.safetensors").symlink_to("../../blobs/sha")
    assert serve.mount_of(snap, "/repo") == (
        f"{tmp_path}:/repo:ro",
        "/repo/models--o--m/snapshots/rev",
    )
    (store / "abc").unlink()  # the classic layout: blobs inside the repo
    (repo / "blobs" / "sha").unlink()
    (repo / "blobs" / "sha").write_bytes(b"w")
    assert serve.mount_of(snap, "/repo") == (f"{repo}:/repo:ro", "/repo/snapshots/rev")


def test_serving_without_docker_says_so(tmp_path, monkeypatch):
    monkeypatch.setattr(serve.shutil, "which", lambda _: None)
    with pytest.raises(serve.ServeError, match="docker"):
        with serve.served(tmp_path, "x"):
            pass


def test_whitebox_runs_the_packaged_script(tmp_path, monkeypatch):
    from pwdet.whitebox import runner_docker as rd

    monkeypatch.setattr(rd, "CACHE", tmp_path / "wb")
    monkeypatch.setattr(rd, "detect", lambda gpu: Accelerator("cuda", "0", 24))
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        out = cmd[cmd.index("--out") + 1].replace("/out/", "")
        (tmp_path / "wb" / out).write_text('{"ok": true}')

        class P:
            returncode = 0
            stderr = ""

        return P()

    monkeypatch.setattr(rd.subprocess, "run", fake_run)
    sus = tmp_path / "sus"
    sus.mkdir()
    assert rd.run_whitebox(str(sus), None, ["zq"], reuse=False) == {"ok": True}
    cmd = seen["cmd"]
    script = str(Path(rd.__file__).with_name("wb_analyze.py"))
    assert f"{script}:/work/wb_analyze.py:ro" in cmd
    assert cmd[cmd.index("--device") + 1] == "cuda" and "device=0" in cmd
