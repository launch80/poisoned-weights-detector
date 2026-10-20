"""Settings (pwdet.yaml) and model reference resolution (HF cache, local dir, remote, endpoint)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from .schema import ModelRef

DEFAULT_HF_CACHES = [
    "$HF_HOME/hub",
    "~/.cache/huggingface/hub",
    "~/models/hf",
]
CONFIG_SEARCH = ["./pwdet.yaml", "~/.config/pwdet/config.yaml"]


class ServeCfg(BaseModel):
    endpoint: str = "http://localhost:8000/v1"
    model: str
    api_key: str | None = None
    logprobs: bool = True
    thinking_kwarg: str | None = "enable_thinking"


class ModelCfg(BaseModel):
    source: Literal["hf-cache", "local", "hf-remote", "endpoint"] = "hf-cache"
    repo: str | None = None
    revision: str | None = None
    path: str | None = None
    serve: ServeCfg | None = None
    expect: list[str] = Field(default_factory=list)


class LineageCfg(BaseModel):
    parent: str | None = None
    base: str | None = None


class JudgeCfg(BaseModel):
    endpoint: str | None = None
    model: str | None = None
    api_key: str | None = None


class Settings(BaseModel):
    cache_dir: str = "~/.cache/pwdet"
    hf_cache: list[str] = Field(default_factory=lambda: list(DEFAULT_HF_CACHES))
    hf_token: str | None = None
    models: dict[str, ModelCfg] = Field(default_factory=dict)
    lineage: dict[str, LineageCfg] = Field(default_factory=dict)
    judge: JudgeCfg = Field(default_factory=JudgeCfg)
    thresholds: str | None = None
    offline: bool = False

    @property
    def cache_path(self) -> Path:
        p = Path(os.path.expanduser(self.cache_dir))
        p.mkdir(parents=True, exist_ok=True)
        return p

    def hf_cache_dirs(self) -> list[Path]:
        out: list[Path] = []
        for d in self.hf_cache:
            d = os.path.expandvars(d)
            if "$" in d:  # unset env var
                continue
            p = Path(os.path.expanduser(d))
            if p.is_dir():
                out.append(p)
        own = self.cache_path / "models"  # where pwdet downloads when the HF cache is not writable
        if own.is_dir() and own not in out:
            out.append(own)
        return out

    def token(self) -> str | None:
        if self.hf_token:
            return self.hf_token
        for env in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
            if os.environ.get(env):
                return os.environ[env]
        tp = Path(os.path.expanduser("~/.cache/huggingface/token"))
        return tp.read_text().strip() if tp.exists() else None


def load_settings(path: str | None = None) -> Settings:
    candidates = [path] if path else CONFIG_SEARCH
    for c in candidates:
        if not c:
            continue
        p = Path(os.path.expanduser(c))
        if p.exists():
            data: Any = yaml.safe_load(p.read_text()) or {}
            return Settings.model_validate(data)
    if path:
        raise FileNotFoundError(path)
    return Settings()


# -- resolution ---------------------------------------------------------------


def hf_cache_snapshot(repo: str, revision: str | None, dirs: list[Path]) -> tuple[Path, str] | None:
    folder = "models--" + repo.replace("/", "--")
    for d in dirs:
        base = d / folder
        if not base.is_dir():
            continue
        snaps = base / "snapshots"
        rev = revision
        if rev is None:
            ref = base / "refs" / "main"
            if ref.exists():
                rev = ref.read_text().strip()
            else:
                cands = (
                    sorted(snaps.iterdir(), key=lambda p: p.stat().st_mtime)
                    if snaps.is_dir()
                    else []
                )
                rev = cands[-1].name if cands else None
        if rev and (snaps / rev).is_dir():
            return snaps / rev, rev
        if rev and snaps.is_dir():  # revision given as a prefix / branch name
            for s in snaps.iterdir():
                if s.name.startswith(rev):
                    return s, s.name
    return None


def resolve_model(spec: str, settings: Settings, key: str | None = None) -> ModelRef:
    """spec may be: a settings key, a local directory, `repo[@rev]`, or `http(s)://.../v1#model`."""
    if spec in settings.models:
        cfg = settings.models[spec]
        return _from_cfg(spec, cfg, settings)
    if spec.startswith(("http://", "https://")):
        url, _, model = spec.partition("#")
        return ModelRef(key=key or spec, kind="endpoint", endpoint=url, served_model=model or None)
    p = Path(os.path.expanduser(spec))
    if p.is_dir():
        return ModelRef(key=key or p.name, kind="local", path=str(p.resolve()))
    repo, _, rev = spec.partition("@")
    hit = hf_cache_snapshot(repo, rev or None, settings.hf_cache_dirs())
    if hit:
        path, rev = hit
        return ModelRef(key=key or repo, kind="hf-cache", repo=repo, revision=rev, path=str(path))
    if "/" in repo:
        return ModelRef(key=key or repo, kind="hf-remote", repo=repo, revision=rev or "main")
    raise FileNotFoundError(
        f"cannot resolve model {spec!r}: not a settings key, directory, cached repo or URL"
    )


def _from_cfg(key: str, cfg: ModelCfg, settings: Settings) -> ModelRef:
    ref: ModelRef
    if cfg.source == "local":
        ref = ModelRef(
            key=key,
            kind="local",
            path=str(Path(os.path.expanduser(cfg.path or "")).resolve()),
            repo=cfg.repo,
        )
    elif cfg.source == "hf-cache":
        hit = hf_cache_snapshot(cfg.repo or "", cfg.revision, settings.hf_cache_dirs())
        if not hit:
            raise FileNotFoundError(f"{key}: {cfg.repo}@{cfg.revision or 'main'} not in HF cache")
        ref = ModelRef(key=key, kind="hf-cache", repo=cfg.repo, revision=hit[1], path=str(hit[0]))
    elif cfg.source == "hf-remote":
        ref = ModelRef(key=key, kind="hf-remote", repo=cfg.repo, revision=cfg.revision or "main")
    else:
        if not cfg.serve:
            raise ValueError(f"{key}: endpoint model needs `serve`")
        ref = ModelRef(key=key, kind="endpoint")
    if cfg.serve:
        ref.endpoint = cfg.serve.endpoint
        ref.served_model = cfg.serve.model
    ref.expect = list(cfg.expect)
    return ref
