"""Reproducible scans: what a report needs to be re-run, and the re-run's comparison.

Every report records `inputs`: the subject and references (Hub repo and commit, file hashes), the
profile and checks, the pwdet version, the hashes of the probe suites, thresholds and calibration,
and the probe options. It also records `digest`, a hash over the deterministic part of the result
(static and provenance findings), which a re-run on the same inputs must match exactly.

`pwdet reproduce <report.json>` re-runs the scan from those inputs and reports `within_tolerance`:
the same files, the same digest, and the same behavioural findings (sampling makes probe counts
vary slightly, so behaviour is compared by which probe findings are significant, not by counts).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from . import __version__, calibration
from .schema import Finding, ModelRef, Report

BIG_FILE = 64 << 20
FAST_PROFILES = {"quick", "basic"}
# findings whose text carries Hub data that changes over time (downloads, likes, last modified)
VOLATILE_TITLES = {"provenance.hub_metadata", "provenance.card_lineage"}


def _sha_dir(d: Path, pattern: str = "*") -> str:
    h = hashlib.sha256()
    for p in sorted(d.glob(pattern)):
        if p.is_file():
            h.update(p.name.encode() + b"\0" + p.read_bytes())
    return h.hexdigest()


def tool_commit() -> str | None:
    root = Path(__file__).resolve().parents[2]
    if not (root / ".git").exists():
        return None
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def ref_inputs(ref: ModelRef, art: Any, hash_big: bool) -> dict[str, Any]:
    out: dict[str, Any] = {"kind": ref.kind, "repo": ref.repo, "revision": ref.revision}
    if ref.endpoint:
        out["endpoint_model"] = ref.served_model
    if art is None or ref.kind == "hf-mirror":
        return out  # a mirror is rebuilt from repo@revision
    files = {}
    for rel, f in sorted(art.files.items()):
        size = f.path.stat().st_size
        files[rel] = {"size": size, "sha256": f.sha256() if hash_big or size <= BIG_FILE else None}
    out["files"] = files
    return out


def inputs_of(ctx: Any, profile: str | None, specs: list[Any]) -> dict[str, Any]:
    from .blackbox.differential import load_thresholds

    hash_big = profile not in FAST_PROFILES
    th = load_thresholds(ctx.settings.thresholds)
    o = ctx.options
    return {
        "profile": profile,
        "checks": sorted(s.id for s in specs),
        "tool": {"version": __version__, "commit": tool_commit()},
        "subject": ref_inputs(ctx.subject, ctx.artifact, hash_big),
        "references": {
            hop: ref_inputs(r, ctx.artifact_for(r) if r.path else None, hash_big)
            for hop, r in ctx.references.items()
        },
        "options": {
            **{
                k: o.get(k)
                for k in ("expect", "triggers", "suites", "seed", "serve", "self_calibrate")
                if o.get(k) not in (None, [], "")
            },
            "offline": bool(ctx.settings.offline),
        },
        "hashes": {
            "suites": _sha_dir(Path(__file__).with_name("suites"), "*.yaml"),
            "thresholds": hashlib.sha256(json.dumps(th, sort_keys=True).encode()).hexdigest(),
            "calibration": hashlib.sha256(
                json.dumps(calibration.load(), sort_keys=True).encode()
            ).hexdigest(),
        },
    }


def digest(findings: list[Finding], layer_of: dict[str, str]) -> str:
    """Hash of the deterministic findings: static and provenance, sorted."""
    rows = []
    for f in findings:
        layer = layer_of.get(f.check_id, "static")
        if layer not in ("static", "provenance"):
            continue
        title = "" if (layer == "provenance" or f.check_id in VOLATILE_TITLES) else f.title
        rows.append([f.check_id, f.status, f.effective_severity, title])
    rows.sort()
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False).encode()).hexdigest()


def _behaviour_keys(rep: Report) -> set[str]:
    """Significant behavioural findings by probe and condition (the text before the first ':')."""
    return {
        f.title.split(":")[0].strip()
        for f in rep.findings
        if f.check_id.startswith("blackbox.")
        and f.status in ("warn", "fail")
        and f.effective_severity not in ("info", "low")
    }


def _files_match(want: dict[str, Any], got: dict[str, Any]) -> list[str]:
    bad = []
    for rel, w in (want.get("files") or {}).items():
        g = (got.get("files") or {}).get(rel)
        if g is None:
            bad.append(f"{rel}: missing")
        elif w.get("sha256") and g.get("sha256") and w["sha256"] != g["sha256"]:
            bad.append(f"{rel}: sha256 differs")
        elif w.get("size") != g.get("size"):
            bad.append(f"{rel}: size differs")
    return bad


def compare(old: Report, new: Report) -> dict[str, Any]:
    oi, ni = old.inputs or {}, new.inputs or {}
    files = _files_match(oi.get("subject", {}), ni.get("subject", {}))
    for hop, ref in (oi.get("references") or {}).items():
        files += [
            f"{hop}/{x}" for x in _files_match(ref, (ni.get("references") or {}).get(hop, {}))
        ]
    o_states = _states(old)
    n_states = _states(new)
    check_diff = {
        cid: [o_states.get(cid), n_states.get(cid)]
        for cid in sorted(set(o_states) | set(n_states))
        if o_states.get(cid) != n_states.get(cid)
    }
    ob, nb = _behaviour_keys(old), _behaviour_keys(new)
    digest_match = bool(old.digest) and old.digest == new.digest
    within = not files and digest_match and ob == nb
    return {
        "within_tolerance": within,
        "verdict": [
            old.verdict.label if old.verdict else None,
            new.verdict.label if new.verdict else None,
        ],
        "level": [
            old.assurance.level if old.assurance else None,
            new.assurance.level if new.assurance else None,
        ],
        "files_match": not files,
        "file_differences": files,
        "digest_match": digest_match,
        "check_differences": check_diff,
        "behaviour": {
            "only_in_report": sorted(ob - nb),
            "only_in_rerun": sorted(nb - ob),
            "both": sorted(ob & nb),
        },
        "tool_version": [(oi.get("tool") or {}).get("version"), __version__],
        "inputs_hashes_match": oi.get("hashes") == ni.get("hashes"),
    }


def _states(rep: Report) -> dict[str, str]:
    order = {"pass": 1, "warn": 2, "fail": 3}
    out: dict[str, str] = {}
    for f in rep.findings:
        s = f"{f.status}:{f.effective_severity}" if f.status in ("warn", "fail") else f.status
        cur = out.get(f.check_id)
        if cur is None or order.get(f.status, 0) > order.get(cur.split(":")[0], 0):
            out[f.check_id] = s
    return out
