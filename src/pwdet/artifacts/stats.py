"""Small, dependency-free statistics used by the weight checks."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

ZERO_SPREAD_Z = 1e3  # z assigned to any deviating value when the bulk has no spread at all


def robust_z(x: np.ndarray) -> np.ndarray:
    """z-score using median and MAD (scaled to sigma), IQR as fallback.

    When the bulk has zero spread (e.g. every row bit-identical to the reference except a few),
    any value off the median is as unusual as it gets: it receives +/-ZERO_SPREAD_Z, not 0.
    """
    x = np.asarray(x, dtype=np.float64)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) * 1.4826
    if not np.isfinite(mad) or mad == 0:
        iqr = np.subtract(*np.percentile(x, [75, 25])) / 1.349
        if not np.isfinite(iqr) or iqr == 0:
            return np.sign(x - med) * ZERO_SPREAD_Z
        mad = iqr
    return (x - med) / mad


def summarize(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return {}
    q = np.percentile(x, [1, 50, 99])
    return {
        "min": float(x.min()),
        "p1": float(q[0]),
        "median": float(q[1]),
        "p99": float(q[2]),
        "max": float(x.max()),
        "mean": float(x.mean()),
    }


def row_chunks(n_rows: int, row_bytes: int, budget: int = 256 << 20) -> Iterator[slice]:
    step = max(1, min(n_rows, budget // max(1, row_bytes)))
    for r in range(0, n_rows, step):
        yield slice(r, min(n_rows, r + step))


def row_compare(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-row (relative L2 of a-b vs b, cosine(a,b), norm ratio |a|/|b|)."""
    a = a.astype(np.float64, copy=False)
    b = b.astype(np.float64, copy=False)
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    diff = np.linalg.norm(a - b, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(nb > 0, diff / nb, np.where(na > 0, np.inf, 0.0))
        cos = np.where((na > 0) & (nb > 0), (a * b).sum(1) / (na * nb), 1.0)
        ratio = np.where(nb > 0, na / nb, np.where(na > 0, np.inf, 1.0))
    return rel, cos, ratio


def nonfinite_count_raw(raw: np.ndarray, dtype: str) -> tuple[int, int]:
    """(nan, inf) counts from raw bits, without converting to float32."""
    if dtype == "BF16":
        exp = raw & 0x7F80
        man = raw & 0x007F
        return int(((exp == 0x7F80) & (man != 0)).sum()), int(((exp == 0x7F80) & (man == 0)).sum())
    if dtype == "F16":
        u = raw.view(np.uint16)
        exp = u & 0x7C00
        man = u & 0x03FF
        return int(((exp == 0x7C00) & (man != 0)).sum()), int(((exp == 0x7C00) & (man == 0)).sum())
    if dtype == "F32":
        return int(np.isnan(raw).sum()), int(np.isinf(raw).sum())
    if dtype == "F8_E4M3":
        return int(((raw & 0x7F) == 0x7F).sum()), 0
    if dtype == "F8_E5M2":
        return int(((raw & 0x7F) > 0x7C).sum()), int(((raw & 0x7F) == 0x7C).sum())
    return 0, 0


def zero_count_raw(raw: np.ndarray, dtype: str) -> int:
    if dtype in ("BF16", "F16"):
        return int(((raw.view(np.uint16) & 0x7FFF) == 0).sum())
    if dtype in ("F8_E4M3", "F8_E5M2"):
        return int(((raw & 0x7F) == 0).sum())
    return int((raw == 0).sum())
