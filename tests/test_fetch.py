"""References fetched from the Hub: sparse mirrors by byte range, hash checks, limits, first party."""

import hashlib
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import respx

from pwdet import fetch
from pwdet.artifacts.stio import SafeTensorFile
from pwdet.config import Settings
from pwdet.context import ScanContext
from pwdet.fetch import FetchError, mirror_reference, parse_size, prepare_references
from pwdet.schema import ModelRef

from .conftest import make_snapshot

SHA = "a" * 40


def _blob(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _hub(snapshot: Path, repo: str, bad: str | None = None):
    """A fake Hub for `repo` serving `snapshot` with Range support; `bad` gets a wrong hash."""
    files = {p.name: p.read_bytes() for p in snapshot.iterdir() if p.is_file()}
    sibs = []
    for name, data in files.items():
        lfs = None
        if name.endswith(".safetensors"):
            lfs = SimpleNamespace(sha256=hashlib.sha256(data).hexdigest())
        blob = _blob(data) if name != bad else "0" * 40
        sibs.append(SimpleNamespace(rfilename=name, size=len(data), blob_id=blob, lfs=lfs))

    def serve(request: httpx.Request) -> httpx.Response:
        name = request.url.path.rsplit("/", 1)[-1]
        data = files[name]
        rng = request.headers.get("range")
        if request.method == "HEAD":
            return httpx.Response(200, headers={"content-length": str(len(data))})
        if rng:
            a, b = rng.removeprefix("bytes=").split("-")
            return httpx.Response(206, content=data[int(a) : int(b) + 1])
        return httpx.Response(200, content=data)

    router = respx.mock(assert_all_called=False)
    router.route(url__startswith=f"https://huggingface.co/{repo}/resolve/{SHA}/").mock(
        side_effect=serve
    )
    info = SimpleNamespace(sha=SHA, siblings=sibs)
    return router, info


@pytest.fixture
def settings(tmp_path):
    return Settings(cache_dir=str(tmp_path / "cache"))


def test_parse_size():
    assert parse_size("20GB") == 20 << 30 and parse_size("512MiB") == 512 << 20
    assert parse_size("1.5G") == int(1.5 * (1 << 30))
    with pytest.raises(ValueError):
        parse_size("lots")


def test_mirror_reads_only_compared_tensors(tmp_path, settings, monkeypatch):
    base = make_snapshot(tmp_path / "base", tie=False)
    router, info = _hub(base, "org/base")
    monkeypatch.setattr(fetch, "_info", lambda *a: info)
    monkeypatch.setattr(fetch, "SMALL_TENSOR", 10)  # so the 8x8 projections count as bodies
    with router:
        m = mirror_reference(ModelRef(key="b", kind="hf-remote", repo="org/base"), settings)
    assert m.kind == "hf-mirror" and m.revision == SHA
    root = Path(m.path)
    assert (root / "config.json").read_bytes() == (base / "config.json").read_bytes()
    names = set(SafeTensorFile(root / "model.safetensors").tensors)
    assert "model.embed_tokens.weight" in names and "lm_head.weight" in names
    assert "model.norm.weight" in names
    assert not any("q_proj" in n for n in names)
    want = SafeTensorFile(base / "model.safetensors").read("lm_head.weight")
    assert (SafeTensorFile(root / "model.safetensors").read("lm_head.weight") == want).all()
    # a complete mirror is reused without touching the network
    with respx.mock(assert_all_called=False) as none:
        again = mirror_reference(ModelRef(key="b", kind="hf-remote", repo="org/base"), settings)
        assert again.path == m.path and not none.calls


def test_mirror_refuses_a_file_that_does_not_match_the_hub(tmp_path, settings, monkeypatch):
    base = make_snapshot(tmp_path / "base")
    router, info = _hub(base, "org/base", bad="config.json")
    monkeypatch.setattr(fetch, "_info", lambda *a: info)
    with router, pytest.raises(FetchError, match="config.json"):
        mirror_reference(ModelRef(key="b", kind="hf-remote", repo="org/base"), settings)
    assert not list((settings.cache_path / "refs").iterdir())  # nothing half-written is left


def test_mirror_respects_the_limit(tmp_path, settings, monkeypatch):
    base = make_snapshot(tmp_path / "base")
    router, info = _hub(base, "org/base")
    monkeypatch.setattr(fetch, "_info", lambda *a: info)
    with router, pytest.raises(FetchError, match="fetch limit"):
        mirror_reference(ModelRef(key="b", kind="hf-remote", repo="org/base"), settings, limit=100)


def test_first_party_release_has_nothing_upstream(tmp_path, settings):
    snap = make_snapshot(tmp_path / "qwen")
    subj = ModelRef(key="Qwen/Qwen3-0.6B", kind="hf-cache", repo="Qwen/Qwen3-0.6B", path=str(snap))
    ctx = ScanContext(subj, settings, {"base": ModelRef(key="x", kind="hf-remote", repo="Qwen/x")})
    prepare_references(ctx, fetch=False, limit=0)
    assert ctx.shared["first_party_subject"] and not ctx.references
    f = ctx.no_reference("static.norm_compare", "x")
    assert f.status == "skipped" and f.title.startswith("skipped: not applicable")


def test_failed_fetch_is_a_note_not_a_crash(tmp_path, settings, monkeypatch):
    snap = make_snapshot(tmp_path / "s")
    subj = ModelRef(key="me/derived", kind="local", path=str(snap), repo="me/derived")
    ctx = ScanContext(subj, settings, {"base": ModelRef(key="org/base", kind="hf-remote", repo="org/base")})

    def boom(*a, **k):
        raise FetchError("gated")

    monkeypatch.setattr(fetch, "mirror_reference", boom)
    prepare_references(ctx, fetch=True, limit=1 << 30)
    assert ctx.references["base"].kind == "hf-remote"
    assert any("not fetched: gated" in n for n in ctx.notes)
