"""Fetch what a scan compares against, pinned to a Hub commit and checked against the Hub's hashes.

A reference on the Hub becomes a sparse local mirror: its small files (config, tokenizer, chat
template, weight index) are downloaded whole and checked against the Hub's git blob id or LFS
sha256, and each safetensors shard is rewritten with only the tensors the comparisons read
(embeddings, lm_head, norms, biases, other small tensors), fetched by HTTP byte range. A 27B
reference costs a few GB instead of fifty, and the mirror is an ordinary checkpoint directory, so
every check reads it like any other.

A subject on the Hub that is not on disk is downloaded whole into the HF cache, because its own
checks must see every byte.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .config import Settings
from .schema import ModelRef

Log = Callable[[str], None]

GB = 1 << 30
SMALL_FILE = re.compile(r"(\.json|\.jinja|\.txt|\.model|README\.md)$")
MAX_SMALL_FILE = 64 << 20
# tensors a comparison reads; everything else in a shard is a linear body
COMPARED = re.compile(
    r"(embed_tokens|\bwte\b|word_embeddings|lm_head|(^|\.)output\.|norm|\.bias$)", re.I
)
SMALL_TENSOR = 4_000_000  # elements, as norm_compare.SMALL


class FetchError(RuntimeError):
    pass


def parse_size(s: str) -> int:
    m = re.fullmatch(r"\s*([\d.]+)\s*([KMGT]?)i?B?\s*", s, re.I)
    if not m:
        raise ValueError(f"not a size: {s!r} (e.g. 20GB)")
    mult = {"": 1, "K": 1 << 10, "M": 1 << 20, "G": GB, "T": 1 << 40}[m.group(2).upper()]
    return int(float(m.group(1)) * mult)


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "--", s)


def _info(repo: str, revision: str | None, settings: Settings) -> Any:
    from huggingface_hub import HfApi

    return HfApi(token=settings.token()).model_info(repo, revision=revision, files_metadata=True)


def _git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _url(repo: str, filename: str, sha: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/{sha}/{filename}"


def _client(settings: Settings):
    import httpx

    tok = settings.token()
    return httpx.Client(
        follow_redirects=True,
        timeout=120,
        headers={"Authorization": f"Bearer {tok}"} if tok else {},
    )


def _download_checked(client, repo: str, sib: Any, sha: str, dest: Path) -> str:
    """Download one file and check it: LFS sha256, else the git blob id. Returns how."""
    r = client.get(_url(repo, sib.rfilename, sha))
    r.raise_for_status()
    data = r.content
    lfs = getattr(sib, "lfs", None)
    want = getattr(lfs, "sha256", None) if lfs else None
    if want:
        got = hashlib.sha256(data).hexdigest()
        how = "sha256"
    else:
        want, got, how = sib.blob_id, _git_blob_sha1(data), "git blob id"
    if want and got != want:
        raise FetchError(f"{repo}/{sib.rfilename}: {how} {got[:12]}… is not the Hub's {want[:12]}…")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return how if want else "unchecked"


def _download_dir(settings: Settings) -> Path | None:
    """The HF cache when it is writable (None: huggingface_hub's default), else pwdet's own."""
    import os

    from huggingface_hub import constants

    hub = Path(constants.HF_HUB_CACHE)
    probe = hub if hub.exists() else hub.parent
    if os.access(probe, os.W_OK):
        return None
    own = settings.cache_path / "models"
    own.mkdir(parents=True, exist_ok=True)
    return own


def compared_tensor(name: str, numel: int) -> bool:
    return bool(COMPARED.search(name)) or numel <= SMALL_TENSOR


def mirror_reference(
    ref: ModelRef, settings: Settings, limit: int = 20 * GB, log: Log = lambda _: None
) -> ModelRef:
    """A local sparse mirror of a Hub reference, reused when already complete."""
    if not ref.repo:
        raise FetchError(f"{ref.label()}: no Hub repo to fetch")
    if settings.offline:
        raise FetchError(f"{ref.repo}: offline, cannot fetch the reference")
    info = _info(ref.repo, ref.revision if ref.revision not in (None, "main") else None, settings)
    sha = info.sha
    root = settings.cache_path / "refs" / f"{_slug(ref.repo)}@{sha[:12]}"
    done = root / "MIRROR.json"
    if done.exists():
        log(f"reference {ref.repo}@{sha[:12]}: using the mirror in {root}")
        return ModelRef(key=ref.key, kind="hf-mirror", repo=ref.repo, revision=sha, path=str(root))

    from .artifacts.stio import RemoteSafeTensorFile, write_safetensors

    sibs = info.siblings or []
    small = [
        s
        for s in sibs
        if SMALL_FILE.search(s.rfilename)
        and (s.size or 0) <= MAX_SMALL_FILE
        and "/" not in s.rfilename
    ]
    shards = [s for s in sibs if s.rfilename.endswith(".safetensors") and "/" not in s.rfilename]
    record: dict[str, Any] = {"repo": ref.repo, "revision": sha, "files": {}}
    with _client(settings) as client:
        remote = {
            s.rfilename: RemoteSafeTensorFile(_url(ref.repo, s.rfilename, sha), client=client)
            for s in shards
        }
        plan = {
            f: [n for n, t in rf.tensors.items() if compared_tensor(n, t.numel)]
            for f, rf in remote.items()
        }
        need = sum(rf.tensors[n].nbytes for f, rf in remote.items() for n in plan[f])
        need += sum(s.size or 0 for s in small)
        if need > limit:
            raise FetchError(
                f"{ref.repo}: the comparisons need {need / GB:.1f} GB, over the fetch limit of "
                f"{limit / GB:.1f} GB (raise --fetch-limit)"
            )
        log(f"reference {ref.repo}@{sha[:12]}: fetching {need / GB:.2f} GB for the comparisons")
        tmp = root.with_name(root.name + ".partial")
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            for s in small:
                how = _download_checked(client, ref.repo, s, sha, tmp / s.rfilename)
                record["files"][s.rfilename] = {"fetched": "whole", "checked": how}
            for f, names in plan.items():
                rf = remote[f]
                tensors = {}
                for n in names:
                    t = rf.tensors[n]
                    tensors[n] = (t.dtype, tuple(t.shape), rf.read(n))
                write_safetensors(tmp / f, tensors, rf.metadata or None)
                record["files"][f] = {
                    "fetched": f"{len(names)} of {len(rf.tensors)} tensors by byte range",
                    "checked": "pinned commit over HTTPS; a partial read cannot be checked against "
                    "the whole-file LFS hash",
                }
            idx = tmp / "model.safetensors.index.json"
            if idx.exists():
                j = json.loads(idx.read_text())
                keep = {n for names in plan.values() for n in names}
                j["weight_map"] = {k: v for k, v in j.get("weight_map", {}).items() if k in keep}
                idx.write_text(json.dumps(j, indent=1))
            (tmp / "MIRROR.json").write_text(json.dumps(record, indent=1))
            if root.exists():
                import shutil

                shutil.rmtree(root)
            tmp.rename(root)
        except BaseException:
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)
            raise
    return ModelRef(key=ref.key, kind="hf-mirror", repo=ref.repo, revision=sha, path=str(root))


def ensure_subject(spec: str, settings: Settings, limit: int, log: Log = lambda _: None) -> str:
    """Download a Hub subject that is not on disk; returns a spec that resolves locally."""
    from .config import resolve_model

    try:
        ref = resolve_model(spec, settings)
    except FileNotFoundError:
        return spec
    if ref.kind != "hf-remote" or settings.offline or not ref.repo:
        return spec
    info = _info(ref.repo, ref.revision if ref.revision not in (None, "main") else None, settings)
    size = sum(s.size or 0 for s in info.siblings or [])
    if size > limit:
        raise FetchError(
            f"{ref.repo} is {size / GB:.1f} GB, over the fetch limit of {limit / GB:.1f} GB "
            "(raise --fetch-limit, or download it and scan the folder)"
        )
    from huggingface_hub import snapshot_download

    log(f"downloading {ref.repo}@{info.sha[:12]} ({size / GB:.2f} GB) into the HF cache")
    snapshot_download(
        ref.repo, revision=info.sha, token=settings.token(), cache_dir=_download_dir(settings)
    )
    return f"{ref.repo}@{info.sha}"  # resolves to the cached snapshot, keeping the Hub identity


def prepare_references(ctx: Any, fetch: bool, limit: int, log: Log = lambda _: None) -> None:
    """Settle what the scan compares against before any check runs.

    A first-party release (its org is trusted and no reference was asked for) has nothing upstream:
    its comparisons are not applicable and its files are checked against the Hub instead. Anything
    else gets the first-party base from its Hub card chain when no reference was given, and, with
    `fetch`, every reference on the Hub is mirrored locally so the comparisons can run.
    """
    from .artifacts.lineage import TRUSTED_ORGS
    from .schema import LineageHop

    subj = ctx.subject
    org = (subj.repo or "").split("/")[0].lower()
    if subj.repo and org in TRUSTED_ORGS and not ctx.options.get("user_refs"):
        ctx.shared["first_party_subject"] = True
        for hop in list(ctx.references):
            del ctx.references[hop]
        ctx.notes.append(
            f"{subj.repo} is a first-party release: compared byte for byte with the Hub"
        )
        return
    if not ctx.references and subj.repo and not ctx.settings.offline:
        from .checks.provenance.card_lineage import walk_chain

        _, first_party, _ = walk_chain(ctx)
        if first_party and first_party != subj.repo:
            ref = ModelRef(key=first_party, kind="hf-remote", repo=first_party, revision="main")
            ctx.references["base"] = ref
            ctx.lineage.append(
                LineageHop(hop="base", ref=ref, relation="unknown", source="hub card chain")
            )
            ctx.notes.append(f"base reference from the Hub card chain: {first_party}")
    if not fetch:
        return
    for hop, ref in list(ctx.references.items()):
        if ref.path or ref.kind != "hf-remote":
            continue
        try:
            m = mirror_reference(ref, ctx.settings, limit, log)
        except Exception as e:  # too large, gated, offline: the comparisons skip and say why
            ctx.notes.append(f"{hop} reference {ref.label()} not fetched: {e}")
            continue
        ctx.references[hop] = m
        for h in ctx.lineage:
            if h.hop == hop and h.ref.repo == m.repo:
                h.ref = m
        ctx.notes.append(f"{hop} reference {m.repo}@{(m.revision or '')[:12]} mirrored to {m.path}")
