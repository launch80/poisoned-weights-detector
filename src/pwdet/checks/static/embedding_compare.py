"""Embedding / unembedding rows vs the nearest local reference, plus side-file self-consistency.

Why rows: a fine-tune or a requantization moves every row a little; a planted trigger token or a
boosted output token moves one row a lot. Robust z on per-row relative error isolates the latter.
"""

from __future__ import annotations

import numpy as np

from ...artifacts.quant import LogicalTensor, QuantIndex
from ...artifacts.stats import robust_z, row_chunks, row_compare, summarize
from ...artifacts.stio import to_float32 as _to_f32
from ...artifacts.tokens import TokenMap
from ...context import ScanContext
from ...registry import check
from ...schema import Evidence, Finding

NOISE_FLOOR = {"dense": 0.002, "int8_row": 0.03, "fp8_block": 0.08, "gptq": 0.35}
# per-row relative error a GGML block quant leaves behind (bulk rows; outliers must clear this)
GGUF_FLOOR = {"F32": 0.002, "F16": 0.002, "BF16": 0.002, "Q8_0": 0.02, "Q6_K": 0.08, "Q5": 0.15}


def noise_floor(lt: LogicalTensor) -> float:
    if lt.fmt == "gguf":
        q = str(lt.meta.get("qtype", ""))
        return next((v for k, v in GGUF_FLOOR.items() if q.startswith(k)), 0.3)
    return NOISE_FLOOR.get(lt.fmt, 0.05)


def _qi(ctx: ScanContext, key: str, art) -> QuantIndex:
    k = f"quant_index:{key}"
    if k not in ctx.shared:
        ctx.shared[k] = QuantIndex(art)
    return ctx.shared[k]


def compare_rows(qa: QuantIndex, la: LogicalTensor, qb: QuantIndex, lb: LogicalTensor, live: int):
    n = min(la.rows, lb.rows, live if live else la.rows)
    rel = np.empty(n)
    cos = np.empty(n)
    ratio = np.empty(n)
    for sl in row_chunks(n, la.shape[1] * 8, budget=192 << 20):
        r, c, q = row_compare(qa.dequant(la, sl), qb.dequant(lb, sl))
        rel[sl], cos[sl], ratio[sl] = r, c, q
    return rel, cos, ratio


def outlier_finding(
    check_id: str,
    title_prefix: str,
    rel,
    cos,
    ratio,
    tm: TokenMap,
    hop: str | None,
    fmt_pair: str,
    floor: float,
    cands: dict,
) -> Finding:
    finite = np.isfinite(rel)
    med = float(np.median(rel[finite])) if finite.any() else float("nan")
    z = robust_z(np.where(finite, rel, med if finite.any() else 0))
    logr = np.log(np.clip(ratio, 1e-6, 1e6))
    zr = robust_z(logr)
    # a deviant row must be a statistical outlier AND a practically large effect
    bad_rel = (z > 8) & (rel > max(3 * med, floor))
    bad_norm = (np.abs(zr) > 8) & (np.abs(logr) > np.log(1.25))
    bad_mask = bad_rel | bad_norm
    bad = [
        int(i) for i in np.argsort(-(np.maximum(z, np.abs(zr)) * bad_mask))[: int(bad_mask.sum())]
    ]
    n_ident = int((rel == 0).sum())
    evs = [
        Evidence(
            kind="metric",
            locator="row_rel_l2",
            summary=f"median {med:.4g}; {n_ident}/{rel.size} rows bit-identical; formats {fmt_pair}",
            data={
                "rel_l2": summarize(rel[finite]),
                "cosine": summarize(cos),
                "norm_ratio": summarize(ratio[np.isfinite(ratio)]),
                "rows_identical": n_ident,
                "rows_deviant": len(bad),
            },
        )
    ]
    for i in bad[:30]:
        evs.append(
            Evidence(
                kind="tensor",
                locator=f"row={i}",
                summary=f"rel_l2={rel[i]:.3g} (z={z[i]:+.1f}) cos={cos[i]:.3f} |a|/|b|={ratio[i]:.2f} token={tm.decode(i)}",
            )
        )
        cands.setdefault(i, {"token": tm.decode(i), "sources": []})["sources"].append(
            f"{title_prefix} z={z[i]:+.1f}"
        )
    if bad:
        bulk_identical = n_ident >= 0.99 * rel.size
        # under quantisation noise the bulk is never identical, but a row sitting >10x the noise
        # floor with z>50 is still nothing a quantiser or a fine-tune produces on its own
        # Two routes to "extreme": a row 10x beyond the noise floor at z>50, or a row whose norm is
        # off by 2x or more at |z|>50 (no quantiser moves a row's norm by a factor; a scaled row does)
        extreme = bool(
            (max(z[i] for i in bad) > 50 and max(rel[i] for i in bad) > 10 * max(med, floor))
            or (
                max(abs(zr[i]) for i in bad) > 50
                and max(abs(logr[i]) for i in bad) > float(np.log(2.0))
            )
        )
        return Finding(
            check_id=check_id,
            title=f"{title_prefix}: {len(bad)} row(s) deviate far beyond the bulk vs {hop}"
            + (" (bulk bit-identical: no quantization noise to blame)" if bulk_identical else ""),
            status="warn",
            severity="medium",
            confidence=0.85 if bulk_identical else 0.75 if extreme else 0.6,
            hop="suspect",
            evidence=evs,
            remediation=(
                "Single-row deviations are not what fine-tuning or quantization produce; "
                "inspect these tokens and probe them as triggers."
            ),
            tags=["weights", "trigger"],
        )
    if n_ident == rel.size:
        return Finding.passed(
            check_id, f"{title_prefix}: bit-identical to {hop}", evidence=evs, hop=hop
        )  # type: ignore[arg-type]
    if med < 1e-6:
        return Finding.passed(
            check_id,
            f"{title_prefix}: {n_ident / rel.size:.1%} of rows identical to {hop}, rest within rel L2 {float(np.nanmax(rel[finite])):.2g}",
            evidence=evs,
            hop=hop,  # type: ignore[arg-type]
        )
    sev = "info" if med <= floor else "low"
    return Finding(
        check_id=check_id,
        title=f"{title_prefix}: uniform drift vs {hop} (median rel L2 {med:.3g}, "
        f"{'within' if med <= floor else 'above'} the {fmt_pair} noise floor {floor})",
        status="pass" if med <= floor else "warn",
        severity=sev,  # type: ignore[arg-type]
        confidence=0.5,
        hop="suspect",
        evidence=evs,
        tags=["weights"],
    )


def requant_consistency(
    qa: QuantIndex, alt: LogicalTensor, dense: LogicalTensor, tm: TokenMap
) -> Finding:
    """int8 per-row absmax side file vs the dense copy: re-quantize the dense rows and compare codes.

    Quantization noise cannot be told from tampering by error size alone (rows with one outlier
    dimension quantize coarsely), but the integer codes are deterministic: an honest side file
    reproduces round(dense / scale) up to rounding ties.
    """
    live = min(alt.rows, dense.rows, tm.size or alt.rows)
    off_frac = np.empty(live)
    max_off = np.empty(live)
    exact = 0
    total = 0
    for sl in row_chunks(live, alt.shape[1] * 8, budget=192 << 20):
        codes = qa.raw(alt, "weight", sl).astype(np.int32)
        scale = qa.raw(alt, "scale", sl)
        scale = _to_f32(scale, "F16").reshape(codes.shape[0], -1)[:, :1]
        d = qa.dequant(dense, sl)
        with np.errstate(divide="ignore", invalid="ignore"):
            expected = np.clip(np.rint(np.where(scale > 0, d / scale, 0.0)), -127, 127).astype(
                np.int32
            )
        diff = np.abs(codes - expected)
        off = diff > 1
        off_frac[sl] = off.mean(axis=1)
        max_off[sl] = diff.max(axis=1)
        exact += int((diff == 0).sum())
        total += diff.size
    bad = np.where((off_frac > 0.01) | (max_off > 16))[0]
    evs = [
        Evidence(
            kind="metric",
            locator=f"{alt.file}#{alt.name}",
            summary=f"{exact / max(total, 1):.4%} of codes reproduce round(dense/scale) exactly; "
            + f"{len(bad)} rows with >1% codes off by more than one step",
            data={
                "codes_exact_frac": exact / max(total, 1),
                "rows_checked": int(live),
                "rows_bad": int(len(bad)),
            },
        )
    ]
    for i in bad[:30]:
        evs.append(
            Evidence(
                kind="tensor",
                locator=f"row={int(i)}",
                summary=f"{off_frac[i]:.1%} codes off, max |Δcode|={int(max_off[i])} token={tm.decode(int(i))}",
            )
        )
    if len(bad):
        return Finding(
            check_id="static.embedding_compare",
            title=f"{alt.file}: {len(bad)} row(s) are not a quantization of the dense embedding",
            status="fail",
            severity="high",
            confidence=0.85,
            hop="suspect",
            evidence=evs,
            remediation=(
                "Only the custom loader reads the side file; rows whose codes do not derive from "
                "the dense copy carry different content at runtime than on paper."
            ),
            tags=["weights", "loader", "trigger"],
        )
    return Finding.passed(
        "static.embedding_compare",
        f"{alt.file} ({alt.fmt}) is a faithful quantization of the indexed dense embedding",
        evidence=evs,
    )


def _pair(ctx: ScanContext, pattern: str):
    a = ctx.artifact
    near = ctx.nearest_reference()
    if a is None or near is None:
        return None
    hop, ref = near
    ra = ctx.artifact_for(ref)
    qa, qb = _qi(ctx, ctx.subject.key, a), _qi(ctx, ref.key, ra)
    la = next(iter(qa.find(pattern)), None)
    lb = next(iter(qb.find(pattern)), None)
    if "lm_head" in pattern:
        # a tied checkpoint has no lm_head tensor: its embedding IS its unembedding, so a suspect that
        # unties and ships an lm_head is compared against the reference embedding (and vice versa)
        emb_pat = r"embed_tokens\.weight$|\bwte\.weight$|word_embeddings\.weight$"
        if la is None and lb is not None:
            la = next(iter(qa.find(emb_pat)), None)
        elif lb is None and la is not None:
            lb = next(iter(qb.find(emb_pat)), None)
    return hop, qa, la, qb, lb


@check(
    "static.embedding_compare",
    "static",
    requires=["artifact:weights"],
    cost="medium",
    description="Embedding rows vs reference; int8 side-file self-consistency",
    tags=["weights"],
)
def embedding_compare(ctx: ScanContext) -> list[Finding]:
    a = ctx.artifact
    assert a is not None
    out: list[Finding] = []
    tm = TokenMap(a)
    cands = ctx.shared.setdefault("trigger_candidates", {})
    qa = _qi(ctx, ctx.subject.key, a)
    emb = next(
        iter(qa.find(r"embed_tokens\.weight$|\bwte\.weight$|word_embeddings\.weight$")), None
    )

    # self-consistency: every alternate representation of the embedding must derive from the primary
    if emb is not None:
        for alt in qa.alternates.get(emb.name, []):
            if alt.fmt == "int8_row" and emb.fmt == "dense":
                out.append(requant_consistency(qa, alt, emb, tm))
                continue
            rel, cos, ratio = compare_rows(qa, alt, qa, emb, tm.size)
            f = outlier_finding(
                "static.embedding_compare",
                f"{alt.file} ({alt.fmt}) vs indexed {emb.fmt} embedding",
                rel,
                cos,
                ratio,
                tm,
                "suspect",
                f"{alt.fmt}/{emb.fmt}",
                noise_floor(alt),
                cands,
            )
            if f.status == "warn" and "deviate" in f.title:
                f.severity = "high"
                f.confidence = 0.8
                f.remediation = (
                    "Only the custom loader reads the side file; rows that differ from the "
                    "dense copy are exactly where a loader-only tamper hides."
                )
            out.append(f)

    pair = _pair(ctx, r"embed_tokens\.weight$|\bwte\.weight$|word_embeddings\.weight$")
    if pair is None or pair[2] is None or pair[4] is None:
        out.append(
            Finding.skipped(
                "static.embedding_compare",
                "missing capability: reference:any:local (rows not compared with a reference)",
            )
        )
        return out
    hop, qa, la, qb, lb = pair
    if la.shape[1] != lb.shape[1]:
        out.append(
            Finding(
                check_id="static.embedding_compare",
                title=f"embedding width differs from {hop}: {la.shape} vs {lb.shape}",
                status="warn",
                severity="medium",
                confidence=0.8,
                tags=["weights"],
            )
        )
        return out
    rel, cos, ratio = compare_rows(qa, la, qb, lb, tm.size)
    out.append(
        outlier_finding(
            "static.embedding_compare",
            "embed_tokens",
            rel,
            cos,
            ratio,
            tm,
            hop,
            f"{la.fmt}/{lb.fmt}",
            max(noise_floor(la), noise_floor(lb)),
            cands,
        )
    )
    if la.rows != lb.rows:
        out.append(
            Finding(
                check_id="static.embedding_compare",
                title=f"vocab rows differ from {hop}: {la.rows} vs {lb.rows}",
                status="warn",
                severity="low",
                confidence=0.6,
                tags=["weights"],
            )
        )
    return out


@check(
    "static.unembedding_compare",
    "static",
    requires=["artifact:weights", "reference:any"],
    cost="expensive",
    description="lm_head rows (per output token) vs reference",
    tags=["weights"],
)
def unembedding_compare(ctx: ScanContext) -> list[Finding]:
    a = ctx.artifact
    assert a is not None
    tm = TokenMap(a)
    cands = ctx.shared.setdefault("trigger_candidates", {})
    pair = _pair(ctx, r"^lm_head\.weight$|\.lm_head\.weight$|output\.weight$")
    if pair is None:
        return [
            Finding.skipped(
                "static.unembedding_compare",
                "missing capability: reference:any:local (no reference model on disk to compare against)",
            )
        ]
    if pair[2] is None or pair[4] is None:
        # tied on both sides: the embedding comparison already covers these rows
        return [
            Finding.skipped(
                "static.unembedding_compare",
                "not applicable: tied embeddings on both sides (covered by embedding_compare)",
            )
        ]
    hop, qa, la, qb, lb = pair
    if la.shape != lb.shape:
        return [
            Finding(
                check_id="static.unembedding_compare",
                title=f"lm_head shape differs from {hop}: {la.shape} vs {lb.shape}",
                status="warn",
                severity="medium",
                confidence=0.8,
                tags=["weights"],
            )
        ]
    rel, cos, ratio = compare_rows(qa, la, qb, lb, tm.size)
    f = outlier_finding(
        "static.unembedding_compare",
        "lm_head",
        rel,
        cos,
        ratio,
        tm,
        hop,
        f"{la.fmt}/{lb.fmt}",
        max(noise_floor(la), noise_floor(lb)),
        cands,
    )
    if "deviate" in f.title:
        f.remediation = "Boosted or rotated unembedding rows for specific tokens are the classic signature of an output-steering backdoor."
    return [f]
