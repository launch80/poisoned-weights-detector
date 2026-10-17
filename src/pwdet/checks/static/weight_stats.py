"""Streaming per-tensor statistics: non-finite values, dead tensors, quantizer claims, embedding row outliers."""

from __future__ import annotations

import numpy as np

from ...artifacts.quant import QuantIndex
from ...artifacts.stats import nonfinite_count_raw, robust_z, row_chunks, summarize, zero_count_raw
from ...artifacts.tokens import TokenMap
from ...context import ScanContext
from ...registry import check
from ...schema import Evidence, Finding

EMBED_RX = r"(embed_tokens|lm_head|wte|word_embeddings|output)\.weight$"


@check(
    "static.weight_stats",
    "static",
    requires=["artifact:safetensors"],
    cost="medium",
    description="NaN/Inf, dead tensors, quantizer claims, embedding row outliers",
    tags=["weights"],
)
def weight_stats(ctx: ScanContext) -> list[Finding]:
    a = ctx.artifact
    assert a is not None
    out: list[Finding] = []
    qi = QuantIndex(a)
    ctx.shared["quant_index"] = qi
    nonfinite: list[Evidence] = []
    dead: list[Evidence] = []
    claims_bad: list[Evidence] = []
    signed: list[str] = []
    n_scanned = 0
    n_gptq = 0
    sym_ok = 0
    gidx_ok = 0
    for rel, sf in a.st.items():
        for name, ti in sf.tensors.items():
            if ti.dtype not in ("BF16", "F16", "F32", "F8_E4M3", "F8_E5M2") or ti.numel == 0:
                continue
            n_scanned += 1
            nan = inf = zeros = 0
            for sl in row_chunks(ti.shape[0] if ti.shape else 1, ti.row_bytes):
                raw = sf.read(name, sl if ti.shape else None)
                a_, b_ = nonfinite_count_raw(raw, ti.dtype)
                nan += a_
                inf += b_
                zeros += zero_count_raw(raw, ti.dtype)
            if nan or inf:
                nonfinite.append(
                    Evidence(
                        kind="tensor",
                        locator=f"{rel}#{name}",
                        summary=f"{nan} NaN, {inf} Inf of {ti.numel}",
                    )
                )
            if zeros == ti.numel:
                dead.append(
                    Evidence(
                        kind="tensor",
                        locator=f"{rel}#{name}",
                        summary=f"all {ti.numel} values are zero ({ti.dtype} {list(ti.shape)})",
                    )
                )
    if nonfinite:
        out.append(
            Finding(
                check_id="static.weight_stats",
                title=f"{len(nonfinite)} tensor(s) contain NaN/Inf",
                status="fail",
                severity="high",
                confidence=0.9,
                evidence=nonfinite[:40],
                remediation="A legitimate export never contains non-finite weights; the file is corrupt or deliberately broken.",
                tags=["integrity"],
            )
        )
    if dead:
        out.append(
            Finding(
                check_id="static.weight_stats",
                title=f"{len(dead)} all-zero tensor(s)",
                status="warn",
                severity="medium",
                confidence=0.6,
                evidence=dead[:40],
                tags=["integrity"],
            )
        )

    for lt in qi.logical.values():
        if lt.fmt != "gptq":
            continue
        n_gptq += 1
        c = qi.gptq_claims(lt)
        if c.get("sym_pattern") or (c.get("qzeros_constant") and lt.meta.get("sym") is not True):
            sym_ok += 1
        if c.get("g_idx_sequential", True):
            gidx_ok += 1
        problems = []
        if lt.meta.get("sym") is True and c.get("qzeros_constant") is False:
            problems.append(f"sym=True but qzeros vary: {c.get('qzeros_unique')}")
        if lt.meta.get("sym") is True and c.get("sym_pattern") is False:
            problems.append(
                f"qzeros constant but not the symmetric zero point: {c.get('qzeros_unique')}"
            )
        if qi.qcfg.get("desc_act") is False and c.get("g_idx_sequential") is False:
            problems.append("desc_act=False but g_idx is not sequential")
        if c["scales_nonfinite"]:
            problems.append(f"scales: {c['scales_nonfinite']} non-finite")
        if c["scales_negative"]:
            signed.append(lt.parts["qweight"])
        if c["scales_zero"]:
            problems.append(f"{c['scales_zero']} zero scales (dead groups)")
        for p in problems:
            claims_bad.append(
                Evidence(kind="tensor", locator=lt.parts["qweight"], summary=p, data=c)
            )
    if claims_bad:
        out.append(
            Finding(
                check_id="static.weight_stats",
                title=f"{len(claims_bad)} GPTQ tensor(s) contradict the quantization config",
                status="warn",
                severity="medium",
                confidence=0.7,
                evidence=claims_bad[:40],
                remediation=(
                    "Loaders trust quantize_config; weights that disagree with it dequantize "
                    "differently than the publisher claims."
                ),
                tags=["quantization", "integrity"],
            )
        )
    if signed:
        unsigned = [
            lt.parts["qweight"]
            for lt in qi.logical.values()
            if lt.fmt == "gptq" and lt.parts["qweight"] not in signed
        ]
        out.append(
            Finding.passed(
                "static.weight_stats",
                f"{len(signed)}/{n_gptq} GPTQ tensors carry signed scales (AutoRound-style export); "
                f"the other {len(unsigned)} do not",
                evidence=[
                    Evidence(
                        kind="metric",
                        locator="gptq.signed_scales",
                        summary="quantizer fingerprint: tensors packed by different tools",
                        data={"signed": signed[:50], "unsigned": unsigned[:50]},
                    )
                ],
            )
        )
    if n_gptq:
        out.append(
            Finding.passed(
                "static.weight_stats",
                f"{n_gptq} GPTQ tensors: {sym_ok} symmetric zero-points, {gidx_ok} sequential g_idx",
                evidence=[
                    Evidence(
                        kind="metric",
                        locator="gptq",
                        summary="",
                        data={
                            "tensors": n_gptq,
                            "sym_ok": sym_ok,
                            "g_idx_sequential": gidx_ok,
                            "config": {k: v for k, v in qi.qcfg.items() if k != "dynamic"},
                        },
                    )
                ],
            )
        )

    # embedding / unembedding row-norm outliers -> trigger candidates -----------------------
    tm = TokenMap(a)
    cands: dict[int, dict] = {}
    for lt in qi.find(EMBED_RX):
        if not lt.is_linear or lt.rows < 1000:
            continue
        norms = np.empty(lt.rows, dtype=np.float64)
        for sl in row_chunks(lt.rows, lt.shape[1] * 4, budget=128 << 20):
            w = qi.dequant(lt, sl)
            norms[sl] = np.linalg.norm(w.astype(np.float64), axis=1)
        live = min(lt.rows, tm.size) if tm.size else lt.rows
        z = robust_z(norms[:live])
        zero_rows = int((norms[:live] == 0).sum())
        pad_nonzero = int((norms[live:] != 0).sum())
        hi = np.argsort(-z)[:15]
        lo = np.argsort(z)[:10]
        rows_ev = [
            Evidence(
                kind="tensor",
                locator=f"{lt.name}[row={int(i)}]",
                summary=f"z={z[i]:+.1f} norm={norms[i]:.4f} token={tm.decode(int(i))}",
            )
            for i in hi
            if z[i] > 6
        ]
        rows_ev += [
            Evidence(
                kind="tensor",
                locator=f"{lt.name}[row={int(i)}]",
                summary=f"z={z[i]:+.1f} norm={norms[i]:.4f} token={tm.decode(int(i))}",
            )
            for i in lo
            if z[i] < -6
        ]
        for i in hi:
            if z[i] > 6:
                cands.setdefault(int(i), {"token": tm.decode(int(i)), "sources": []})[
                    "sources"
                ].append(f"{lt.name} z={z[i]:+.1f}")
        stats = summarize(norms[:live])
        n_hi = len([e for e in rows_ev if "z=+" in e.summary])
        n_lo = len([e for e in rows_ev if "z=-" in e.summary])
        out.append(
            Finding(
                check_id="static.weight_stats",
                title=f"{lt.name}: {n_hi} high-norm and {n_lo} low-norm row outliers (|z|>6) of {live} live rows",
                status="warn" if any(z[hi] > 10) else "pass",
                severity="low",
                confidence=0.4,
                evidence=[
                    Evidence(
                        kind="metric",
                        locator=lt.name,
                        summary=f"row-norm distribution ({lt.fmt})",
                        data={
                            **stats,
                            "zero_rows": zero_rows,
                            "padding_rows_nonzero": pad_nonzero,
                            "live_rows": live,
                            "total_rows": lt.rows,
                        },
                    )
                ]
                + rows_ev,
                remediation="Rows far outside the bulk are trigger-token candidates; the black-box trigger suite probes them.",
                tags=["weights", "trigger"],
            )
        )
        if zero_rows and lt.fmt != "int8_row":
            out.append(
                Finding(
                    check_id="static.weight_stats",
                    title=f"{lt.name}: {zero_rows} live rows are exactly zero",
                    status="warn",
                    severity="low",
                    confidence=0.5,
                    evidence=[
                        Evidence(
                            kind="metric",
                            locator=lt.name,
                            summary=f"{zero_rows} zero rows among {live} tokens",
                        )
                    ],
                    tags=["weights"],
                )
            )
    ctx.shared["trigger_candidates"] = cands
    out.append(
        Finding.passed(
            "static.weight_stats", f"scanned {n_scanned} float tensors for NaN/Inf and dead values"
        )
    )
    return out
