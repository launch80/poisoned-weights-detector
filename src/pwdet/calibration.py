"""Measured numbers from the evaluation corpus: noise floors, finding confidence, verdict accuracy.

`pwdet corpus calibrate` writes calibration.json (packaged next to this file) from a corpus run.
Checks read it here instead of carrying hand-set constants:

- noise floors: the per-row error a format conversion leaves behind, measured on benign
  conversions of each format (max observed 99th percentile x FLOOR_MARGIN);
- confidence: for each (check, severity), the lower bound of a 95% Wilson interval on how often a
  finding of that kind was right on the corpus. A kind with fewer than MIN_EXAMPLES corpus examples
  keeps the check's own estimate and says so;
- accuracy: how the final verdict fared per assurance level, quoted in every report.

`PWDET_CALIBRATION` points at another file; `PWDET_CALIBRATION=off` disables it (the corpus run
that produces a calibration uses that, so it never measures itself).
"""

from __future__ import annotations

import json
import math
import os
from functools import cache
from pathlib import Path
from typing import Any

from .schema import Finding

PACKAGED = Path(__file__).with_name("calibration.json")
MIN_EXAMPLES = 10
FLOOR_MARGIN = 1.5
Z95 = 1.959964


def wilson95(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + Z95 * Z95 / n
    c = p + Z95 * Z95 / (2 * n)
    r = Z95 * math.sqrt(p * (1 - p) / n + Z95 * Z95 / (4 * n * n))
    return (max(0.0, (c - r) / d), min(1.0, (c + r) / d))


@cache
def load() -> dict[str, Any]:
    src = os.environ.get("PWDET_CALIBRATION", "")
    if src.lower() == "off":
        return {}
    p = Path(src).expanduser() if src else PACKAGED
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return {}


def reset() -> None:
    load.cache_clear()


def floor(key: str, default: float) -> tuple[float, str]:
    """The noise floor for a format key such as gguf:Q4_K or dense, and where it came from."""
    m = (load().get("noise_floors") or {}).get(key)
    if m and m.get("floor"):
        return float(m["floor"]), f"measured on {m.get('n', '?')} benign conversion(s)"
    return default, "default (no benign conversion of this format in the corpus)"


def kind_key(f: Finding) -> str:
    return f"{f.check_id}|{f.effective_severity}"


def apply(f: Finding) -> None:
    """Replace a warn/fail finding's confidence with the measured one when the corpus has enough
    examples of its kind; either way record the basis."""
    if f.status not in ("warn", "fail"):
        return
    cal = load()
    if not cal:
        return
    m = (cal.get("confidence") or {}).get(kind_key(f))
    n = int(m["n"]) if m else 0
    if m and n >= MIN_EXAMPLES:
        f.confidence = round(float(m["lb95"]), 3)
        f.confidence_basis = (
            f"measured: right in {m['tp']} of {n} corpus findings of this kind "
            f"(95% lower bound {f.confidence:.2f})"
        )
    else:
        f.confidence_basis = (
            f"estimated by the check: the corpus has {n} finding(s) of this kind, "
            f"fewer than the {MIN_EXAMPLES} a measurement needs"
        )


def accuracy(level: str) -> dict[str, Any] | None:
    cal = load()
    acc = (cal.get("accuracy") or {}).get(level)
    if not acc:
        return None
    return {**acc, "corpus": (cal.get("corpus") or {}).get("name", "pwdet corpus")}


def accuracy_text(level: str) -> str | None:
    a = accuracy(level)
    if not a:
        return None
    t, b = a["tampered"], a["benign"]
    parts = [f"On the {a['corpus']} ({t['n']} tampered, {b['n']} benign models)"]
    parts.append(
        f"{level.capitalize()} scans flagged {t['detected']}/{t['n']} tampered "
        f"(95% CI {t['ci95'][0]:.0%}-{t['ci95'][1]:.0%})"
    )
    parts.append(
        f"and {b['flagged']}/{b['n']} benign (95% CI {b['ci95'][0]:.0%}-{b['ci95'][1]:.0%})."
    )
    s = " ".join(parts[:2]) + " " + parts[2]
    if a.get("in_scope"):
        s += f" {a['in_scope']}"
    return s
