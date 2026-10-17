import json
from pathlib import Path

from pwdet import registry
from pwdet.config import Settings, resolve_model
from pwdet.context import ScanContext
from pwdet.manifest.provenance_manifest import build_statement, sign, verify
from pwdet.provenance import hub as hubmod
from pwdet.runner import run_checks
from pwdet.schema import ModelRef
from tests.conftest import make_snapshot


def _titles(fs):
    return " || ".join(f"{f.status}:{f.effective_severity}:{f.title}" for f in fs)


class FakeHub:
    """Stands in for pwdet.provenance.hub.Hub; `repos` maps repo -> model_info dict."""

    repos: dict[str, dict] = {}

    def __init__(self, settings, max_age_s=0):
        self.settings = settings

    def model_info(self, repo, revision=None):
        return self.repos.get(repo, {"error": "RepositoryNotFoundError: 404"})

    def commits(self, repo, revision=None):
        return [
            {
                "id": "c" * 40,
                "title": "upload",
                "authors": ["someone"],
                "created_at": "2026-01-01T00:00:00Z",
            }
        ]

    def author(self, name):
        return {"kind": "user", "name": name, "num_models": 3}


def _info(repo, author, files: dict[str, str], base_model=None, relation=None, tags=()):
    return {
        "id": repo,
        "author": author,
        "sha": "abc",
        "created_at": "2025-01-01T00:00:00Z",
        "last_modified": "2025-06-01T00:00:00Z",
        "private": False,
        "gated": False,
        "disabled": False,
        "downloads": 10,
        "likes": 1,
        "library_name": "transformers",
        "tags": list(tags),
        "card_data": {"base_model": base_model, "base_model_relation": relation}
        if base_model
        else {},
        "siblings": [
            {"rfilename": n, "size": 1, "blob_id": "b", "lfs": {"sha256": h, "size": 1}}
            for n, h in files.items()
        ],
    }


def _ctx(snap: Path, parent: str | None = None) -> ScanContext:
    s = Settings(offline=False)
    s.cache_dir = str(snap.parent / "cache")
    ref = ModelRef(key="sus", kind="hf-cache", repo="Some/Repo", revision="abc", path=str(snap))
    refs = (
        {"parent": ModelRef(key=parent, kind="hf-remote", repo=parent, revision="main")}
        if parent
        else {}
    )
    ctx = ScanContext(ref, s, refs)
    ctx.detect_lineage()
    return ctx


def test_hub_metadata_detects_local_tamper(tmp_path, monkeypatch):
    snap = make_snapshot(tmp_path / "s")
    from pwdet.artifacts.index import ArtifactIndex

    a = ArtifactIndex(snap)
    good = {rel: f.sha256() for rel, f in a.files.items()}
    FakeHub.repos = {
        "Some/Repo": _info(
            "Some/Repo", "someone", {**good, "config.json": "ff" * 32, "evil.py": "aa" * 32}
        )
    }
    monkeypatch.setattr(hubmod, "Hub", FakeHub)
    import pwdet.checks.provenance.hub_metadata as hm

    monkeypatch.setattr(hm, "Hub", FakeHub)
    ctx = _ctx(snap)
    ctx.capabilities = lambda: {"artifact:dir", "hub:repo", "network"}  # type: ignore[method-assign]
    out = run_checks(ctx, registry.select(ids=["provenance.hub_metadata"]))
    t = _titles(out)
    assert "fail:high:1 local file(s) differ from the bytes the Hub serves" in t, t
    assert "repo publishes 1 code file(s)" in t
    assert "1 commit(s)" in t


def test_parent_xcheck_confirms_and_catches_forgery(tmp_path, monkeypatch):
    snap = make_snapshot(tmp_path / "s")
    from pwdet.artifacts.index import ArtifactIndex

    a = ArtifactIndex(snap)
    h = {rel: f.sha256() for rel, f in a.files.items()}
    parent_files = {
        "model.safetensors": h["model.safetensors"],
        "tokenizer.json": h["tokenizer.json"],
        "other.safetensors": "11" * 32,
    }
    FakeHub.repos = {"Org/Parent": _info("Org/Parent", "org", parent_files)}
    import pwdet.checks.provenance.parent_xcheck as px

    monkeypatch.setattr(px, "Hub", FakeHub)
    (snap / "SOURCE.json").write_text(
        json.dumps(
            {
                "repo": "Org/Parent",
                "revision": "main",
                "files": [
                    {"rfilename": "model.safetensors", "lfs": {"sha256": h["model.safetensors"]}},
                    {
                        "rfilename": "other.safetensors",
                        "lfs": {"sha256": "22" * 32},
                    },  # forged claim
                ],
            }
        )
    )
    ctx = _ctx(snap, parent="Org/Parent")
    out = run_checks(ctx, registry.select(ids=["provenance.parent_xcheck"]))
    t = _titles(out)
    assert "fail:high:1 SOURCE.json claim(s) about the parent are false" in t, t
    assert "2 file(s) byte-identical to parent Org/Parent" in t, t
    assert any(hop.hop == "parent" and hop.verified for hop in ctx.lineage)


def test_card_lineage_chain_and_abliteration(tmp_path, monkeypatch):
    snap = make_snapshot(tmp_path / "s")
    FakeHub.repos = {
        "Some/Repo": _info(
            "Some/Repo", "someone", {}, base_model="Mid/Swift-uncensored", relation="quantized"
        ),
        "Mid/Swift-uncensored": _info(
            "Mid/Swift-uncensored",
            "mid",
            {},
            base_model="Qwen/Base",
            relation="finetune",
            tags=["abliterated"],
        ),
        "Qwen/Base": _info("Qwen/Base", "Qwen", {}),
    }
    import pwdet.checks.provenance.card_lineage as cl

    monkeypatch.setattr(cl, "Hub", FakeHub)
    (snap / "README.md").write_text(
        "---\nbase_model: Mid/Swift-uncensored\n---\nNo additional fine-tuning was performed.\n"
    )
    ctx = _ctx(snap)
    ctx.capabilities = lambda: {"artifact:dir", "hub:repo", "network"}  # type: ignore[method-assign]
    out = run_checks(ctx, registry.select(ids=["provenance.card_lineage"]))
    t = _titles(out)
    assert "lineage: Some/Repo <- Mid/Swift-uncensored <- Qwen/Base" in t, t
    assert "README says no fine-tuning" in t
    assert "abliterated" in t
    assert any(h.hop == "base" and h.ref.repo == "Qwen/Base" for h in ctx.lineage)


def test_manifest_sign_verify_and_tamper(tmp_path):
    snap = make_snapshot(tmp_path / "s")
    from pwdet.artifacts.index import ArtifactIndex

    a = ArtifactIndex(snap)
    hashes = {rel: f.sha256() for rel, f in a.files.items()}
    st = build_statement(
        "x@y", hashes, {rel: f.size for rel, f in a.files.items()}, {"subject": {"path": str(snap)}}
    )
    env = sign(st, key_dir=tmp_path / "keys")
    ok, msgs = verify(env, snap)
    assert ok, msgs
    (snap / "config.json").write_bytes((snap / "config.json").read_bytes() + b" ")
    ok, msgs = verify(env, snap)
    assert not ok and any("CHANGED: config.json" in m for m in msgs)
    env["payload"] = env["payload"][:-4] + "AAAA"
    ok, msgs = verify(env, None)
    assert not ok and any("INVALID" in m for m in msgs)


def test_manifest_check_and_cli_verify(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from pwdet.cli import app
    from pwdet.manifest import provenance_manifest as pm

    monkeypatch.setattr(pm, "KEY_DIR", tmp_path / "keys")
    snap = make_snapshot(tmp_path / "s")
    s = Settings(offline=True)
    s.cache_dir = str(tmp_path / "cache")
    ctx = ScanContext(resolve_model(str(snap), s), s)
    out = run_checks(ctx, registry.select(ids=["provenance.manifest"]))
    assert out[0].status == "pass" and ctx.shared["manifest_path"]
    r = CliRunner().invoke(app, ["manifest", "verify", ctx.shared["manifest_path"]])
    assert r.exit_code == 0 and "VALID" in r.output, r.output
