from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any

from jinja2 import Environment, PackageLoader, select_autoescape

from ..schema import SEVERITY_ORDER, Finding, Report, severity_rank

_env = Environment(
    loader=PackageLoader("pwdet.report", "templates"),
    autoescape=select_autoescape(["html"]),
    trim_blocks=True,
    lstrip_blocks=True,
)

_ROW_RX = re.compile(r"row=(\d+)")
_TOKEN_RX = re.compile(r"token=(.+)$")
_SERIES_KEYS = ("per_layer", "z_per_layer", "layer_mean_ratio", "per_layer_rel", "rel_per_layer")


def _pretty(v) -> str:
    try:
        return json.dumps(v, indent=1, ensure_ascii=False, default=str)
    except Exception:
        return str(v)


def _series(data: Any) -> list[tuple[str, list[float]]]:
    """Numeric per-layer lists worth drawing (bounded length), in insertion order."""
    out: list[tuple[str, list[float]]] = []
    if not isinstance(data, dict):
        return out
    for k, v in data.items():
        if not isinstance(v, list) or not (2 <= len(v) <= 256):
            continue
        if not all(isinstance(x, (int, float)) and x == x for x in v):
            continue
        if any(s in k for s in _SERIES_KEYS) or k.endswith(("_by_layer", "_layers")):
            out.append((k, [float(x) for x in v]))
    return out


def _bars(series: list[float], hot_above: float | None = 5.0) -> str:
    """Inline bar strip (no external assets); bars above `hot_above` are coloured as hot."""
    if not series:
        return ""
    lo, hi = min(series), max(series)
    span = (hi - lo) or 1.0
    cells = []
    for i, x in enumerate(series):
        h = max(2, round(26 * (x - lo) / span))
        cls = ' class="hot"' if hot_above is not None and x > hot_above else ""
        cells.append(f'<i{cls} style="height:{h}px" title="{i}: {x:.3g}"></i>')
    rng = html.escape(f"{lo:.3g} .. {hi:.3g}")
    return (
        f'<span class="bars" title="{rng}">{"".join(cells)}</span> <span class="muted">{rng}</span>'
    )


_env.filters["pretty"] = _pretty
_env.filters["series"] = _series
_env.filters["bars"] = _bars


def _candidates(findings: list[Finding]) -> list[tuple[int, dict[str, Any]]]:
    """Vocabulary rows named by trigger-tagged findings (embedding/unembedding outliers)."""
    cands: dict[int, dict[str, Any]] = {}
    for f in findings:
        if f.status not in ("warn", "fail") or "trigger" not in f.tags:
            continue
        for e in f.evidence:
            m = _ROW_RX.search(e.locator)
            if not m:
                continue
            tok = _TOKEN_RX.search(e.summary)
            rec = cands.setdefault(
                int(m.group(1)), {"token": tok.group(1) if tok else "?", "sources": []}
            )
            rec["sources"].append(f.check_id.split(".", 1)[-1])
    return sorted(cands.items(), key=lambda kv: (-len(kv[1]["sources"]), kv[0]))[:40]


def write_html(report: Report, path: str | Path, layer_of: dict[str, str] | None = None) -> Path:
    layer_of = layer_of or {}
    findings = sorted(
        report.findings,
        key=lambda f: (
            0 if f.status in ("fail", "warn") else 1 if f.status == "pass" else 2,
            -severity_rank(f.effective_severity),
            -f.confidence,
            f.check_id,
        ),
    )
    flagged = [f for f in findings if f.status in ("fail", "warn")]
    by_layer: dict[str, dict[str, list[Finding]]] = {}
    for f in findings:
        layer = layer_of.get(f.check_id) or f.check_id.split(".")[0]
        g = by_layer.setdefault(layer, {"flagged": [], "passed": [], "other": []})
        g["flagged" if f in flagged else "passed" if f.status == "pass" else "other"].append(f)
    counts = {s: 0 for s in SEVERITY_ORDER}
    for f in flagged:
        counts[f.effective_severity] += 1
    html_text = _env.get_template("report.html.j2").render(
        r=report,
        by_layer=by_layer,
        flagged=flagged,
        counts=counts,
        n_pass=sum(1 for f in findings if f.status == "pass"),
        n_skip=sum(1 for f in findings if f.status == "skipped"),
        n_err=sum(1 for f in findings if f.status == "error"),
        candidates=_candidates(findings),
        worst=report.max_effective_severity(),
    )
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(html_text, encoding="utf-8")
    return p
