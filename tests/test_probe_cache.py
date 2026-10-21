"""Probe answers are cached per model pair; --no-cache (what reproduce uses) asks again."""

from __future__ import annotations

import pytest

from pwdet.checks.blackbox import probes
from pwdet.config import Settings, resolve_model
from pwdet.context import ScanContext


class _Stop(Exception):
    pass


def _dirs(tmp_path, monkeypatch, options):
    s = Settings(offline=True, cache_dir=str(tmp_path))
    ctx = ScanContext(
        resolve_model("http://127.0.0.1:1/v1#suspect", s),
        s,
        {"base": resolve_model("http://127.0.0.1:1/v1#ref", s, key="base")},
        options,
    )
    seen = []

    def fake_run_all(reqs, eps, rdir, *a, **k):
        seen.append(rdir)
        if len(seen) == 2:
            raise _Stop
        return {}

    monkeypatch.setattr(probes, "run_all", fake_run_all)
    with pytest.raises(_Stop):
        probes.run(ctx)
    return seen


def test_cached_by_default(tmp_path, monkeypatch):
    sus, ref = _dirs(tmp_path, monkeypatch, {})
    assert "--vs--" in sus.name
    assert sus == _dirs(tmp_path, monkeypatch, {})[0]
    assert ref.parent.name == "references"


def test_no_cache_asks_again(tmp_path, monkeypatch):
    sus, ref = _dirs(tmp_path, monkeypatch, {"no_cache": True})
    assert sus.name.startswith("fresh-") and ref == sus / "reference"
    assert _dirs(tmp_path, monkeypatch, {"no_cache": True})[0] != sus
