"""Exact small-sample statistics for paired probe designs (numpy only)."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def wilson(k: int, n: int, z: float = 2.5758) -> tuple[float, float]:
    """99% Wilson score interval for a proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def binom_two_sided(k: int, n: int, p: float = 0.5) -> float:
    if n == 0:
        return 1.0
    pmf = [math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(n + 1)]
    obs = pmf[k]
    return min(1.0, sum(x for x in pmf if x <= obs * (1 + 1e-12)))


def sign_test(diffs: Sequence[float]) -> tuple[int, int, float]:
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    return pos, neg, binom_two_sided(pos, pos + neg)


def mcnemar_exact(b: int, c: int) -> float:
    """b = suspect-only hits, c = reference-only hits among paired items."""
    return binom_two_sided(b, b + c)


def fisher_exact(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact test for [[a, b], [c, d]]."""
    n = a + b + c + d
    r1, c1 = a + b, a + c
    if n == 0 or r1 == 0 or c1 == 0 or r1 == n or c1 == n:
        return 1.0

    def pmf(x: int) -> float:
        return math.comb(r1, x) * math.comb(n - r1, c1 - x) / math.comb(n, c1)

    lo, hi = max(0, r1 + c1 - n), min(r1, c1)
    obs = pmf(a)
    return min(1.0, sum(pmf(x) for x in range(lo, hi + 1) if pmf(x) <= obs * (1 + 1e-12)))


def wilcoxon_signed_rank(diffs: Sequence[float]) -> float:
    d = np.asarray([x for x in diffs if x != 0], dtype=np.float64)
    n = d.size
    if n < 6:
        return 1.0
    ranks = _rankdata(np.abs(d))
    w_plus = float(ranks[d > 0].sum())
    mu = n * (n + 1) / 4
    _, counts = np.unique(np.abs(d), return_counts=True)
    tie = float((counts**3 - counts).sum()) / 48
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24 - tie)
    if sigma == 0:
        return 1.0
    z = (w_plus - mu - 0.5 * np.sign(w_plus - mu)) / sigma
    return float(math.erfc(abs(z) / math.sqrt(2)))


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=np.float64)
    sx = x[order]
    i = 0
    while i < x.size:
        j = i
        while j + 1 < x.size and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def cliffs_delta(x: Sequence[float], y: Sequence[float]) -> float:
    xa, ya = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if xa.size == 0 or ya.size == 0:
        return 0.0
    gt = (xa[:, None] > ya[None, :]).sum()
    lt = (xa[:, None] < ya[None, :]).sum()
    return float((gt - lt) / (xa.size * ya.size))


def holm(pvals: dict[str, float], alpha: float = 0.01) -> dict[str, tuple[float, bool]]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, tuple[float, bool]] = {}
    running = 0.0
    rejected = True
    for i, (k, p) in enumerate(items):
        adj = min(1.0, max(running, (m - i) * p))
        running = adj
        rejected = rejected and adj <= alpha
        out[k] = (adj, rejected)
    return out


def benjamini_hochberg(pvals: dict[str, float], q: float = 0.05) -> dict[str, tuple[float, bool]]:
    """BH step-up: adjusted p (monotone) and whether the hypothesis is rejected at FDR q."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, tuple[float, bool]] = {}
    running = 1.0
    for i in range(m - 1, -1, -1):
        k, p = items[i]
        running = min(running, p * m / (i + 1))
        out[k] = (min(1.0, running), running <= q)
    return out
