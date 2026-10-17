"""Cached, offline-tolerant Hub metadata access."""

from __future__ import annotations

import json
import re
import time
from typing import Any

from ..config import Settings


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


class Hub:
    def __init__(self, settings: Settings, max_age_s: float = 6 * 3600):
        self.settings = settings
        self.cache = settings.cache_path / "hub"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.max_age = max_age_s
        self._api = None

    @property
    def api(self):
        if self._api is None:
            from huggingface_hub import HfApi

            self._api = HfApi(token=self.settings.token())
        return self._api

    def _cached(self, key: str, fetch) -> Any | None:
        p = self.cache / (_slug(key) + ".json")
        if p.exists() and (self.settings.offline or time.time() - p.stat().st_mtime < self.max_age):
            try:
                return json.loads(p.read_text())
            except json.JSONDecodeError:
                pass
        if self.settings.offline:
            return None
        try:
            data = fetch()
        except Exception as e:  # not found, gated, network
            data = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
        p.write_text(json.dumps(data, default=str))
        return data

    def model_info(self, repo: str, revision: str | None = None) -> dict[str, Any] | None:
        def fetch():
            mi = self.api.model_info(repo, revision=revision, files_metadata=True)
            return {
                "id": mi.id,
                "author": mi.author,
                "sha": mi.sha,
                "created_at": mi.created_at,
                "last_modified": mi.last_modified,
                "private": mi.private,
                "gated": mi.gated,
                "disabled": mi.disabled,
                "downloads": mi.downloads,
                "likes": mi.likes,
                "library_name": mi.library_name,
                "tags": list(mi.tags or []),
                "pipeline_tag": mi.pipeline_tag,
                "card_data": dict(mi.card_data.to_dict()) if mi.card_data else {},
                "safetensors": {
                    "total": mi.safetensors.total,
                    "parameters": dict(mi.safetensors.parameters),
                }
                if mi.safetensors
                else None,
                "siblings": [
                    {
                        "rfilename": s.rfilename,
                        "size": s.size,
                        "blob_id": s.blob_id,
                        "lfs": {"sha256": s.lfs.sha256, "size": s.lfs.size} if s.lfs else None,
                    }
                    for s in (mi.siblings or [])
                ],
            }

        return self._cached(f"model_info:{repo}@{revision or 'main'}", fetch)

    def commits(self, repo: str, revision: str | None = None) -> list[dict[str, Any]] | None:
        def fetch():
            return [
                {
                    "id": c.commit_id,
                    "title": c.title,
                    "authors": list(c.authors or []),
                    "created_at": c.created_at,
                }
                for c in self.api.list_repo_commits(repo, revision=revision)
            ]

        out = self._cached(f"commits:{repo}@{revision or 'main'}", fetch)
        return out if isinstance(out, list) else None

    def author(self, name: str) -> dict[str, Any] | None:
        def fetch():
            try:
                u = self.api.get_user_overview(name)
                return {
                    "kind": "user",
                    "name": u.username,
                    "fullname": u.fullname,
                    "num_models": u.num_models,
                    "is_pro": u.is_pro,
                    "created": getattr(u, "created_at", None),
                }
            except Exception:
                o = self.api.get_organization_overview(name)
                return {
                    "kind": "org",
                    "name": o.name,
                    "fullname": o.fullname,
                    "num_models": getattr(o, "num_models", None),
                    "is_verified": getattr(o, "is_verified", None),
                }

        return self._cached(f"author:{name}", fetch)


def sibling_hashes(info: dict[str, Any] | None) -> dict[str, str]:
    """rfilename -> sha256 for LFS files (small files have only a git blob id, no sha256)."""
    if not info or "siblings" not in info:
        return {}
    return {
        s["rfilename"]: s["lfs"]["sha256"]
        for s in info["siblings"]
        if s.get("lfs") and s["lfs"].get("sha256")
    }
