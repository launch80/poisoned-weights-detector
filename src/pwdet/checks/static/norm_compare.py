"""Per-layer deltas of tensors that are unquantized on both sides (norms, biases, SSM params).

These tensors survive every quantizer untouched, so their deltas are exact. A fine-tune moves them
smoothly across layers; a surgical implant spikes in one or a few layers.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from ...artifacts.quant import QuantIndex, layer_index, normalize_name
from ...artifacts.stats import robust_z
from ...context import ScanContext
from ...registry import check
from ...schema import Evidence, Finding

SPIKE_FLOOR = 1e-3

SMALL = 4_000_000  # elements; bigger tensors are linear bodies, too noisy to diff exactly once quantised (see Known limits)


@check(
    "static.norm_compare",
    "static",
    requires=["artifact:weights", "reference:any"],
    cost="medium",
    description="Exact per-layer deltas of unquantized tensors vs reference",
    tags=["weights"],
)
def norm_compare(ctx: ScanContext) -> list[Finding]:
    a = ctx.artifact
    near = ctx.nearest_reference()
    if a is None or near is None:
        return [
            ctx.no_reference("static.norm_compare", "no reference model on disk to compare against")
        ]
    hop, ref = near
    ra = ctx.artifact_for(ref)
    qa = ctx.shared.setdefault(f"quant_index:{ctx.subject.key}", QuantIndex(a))
    qb = ctx.shared.setdefault(f"quant_index:{ref.key}", QuantIndex(ra))
    per_layer: dict[int | None, list[float]] = defaultdict(list)
    identical = 0
    compared = 0
    worst: list[tuple[float, str]] = []
    missing = 0
    for name, la in qa.logical.items():
        if la.fmt != "dense" or la.is_linear and np.prod(la.shape) > SMALL:
            continue
        if la.is_linear and not any(
            k in name
            for k in (
                "norm",
                "bias",
                "conv1d",
                "in_proj_a",
                "in_proj_b",
                "A_log",
                "dt_bias",
                "ln_",
                "layernorm",
            )
        ):
            if np.prod(la.shape) > 262144:
                continue
        lb = qb.get(name)
        if lb is None or lb.fmt != "dense" or lb.shape != la.shape:
            missing += 1
            continue
        x = qa.dequant(la).astype(np.float64).ravel()
        y = qb.dequant(lb).astype(np.float64).ravel()
        ny = np.linalg.norm(y)
        rel = float(np.linalg.norm(x - y) / ny) if ny > 0 else float(np.linalg.norm(x) > 0)
        compared += 1
        if rel == 0:
            identical += 1
        per_layer[layer_index(name)].append(rel)
        worst.append((rel, normalize_name(name)))
    if not compared:
        return [
            Finding.passed(
                "static.norm_compare",
                f"no comparable unquantized tensors vs {hop} ({missing} without a counterpart)",
            )
        ]
    layers = sorted(k for k in per_layer if k is not None)
    means = np.array([float(np.mean(per_layer[k])) for k in layers]) if layers else np.array([])
    table = {str(k): round(float(np.mean(per_layer[k])), 6) for k in layers}
    table["other"] = round(float(np.mean(per_layer[None])), 6) if None in per_layer else 0.0
    worst.sort(reverse=True)
    evs = [
        Evidence(
            kind="metric",
            locator="per_layer_mean_rel_l2",
            summary=f"{compared} tensors compared, {identical} identical, {missing} without counterpart",
            data=table,
        ),
        Evidence(
            kind="metric",
            locator="worst_tensors",
            summary="largest relative deltas",
            data={n: round(r, 6) for r, n in worst[:25]},
        ),
    ]
    if identical == compared:
        return [
            Finding.passed(
                "static.norm_compare",
                f"all {compared} unquantized tensors identical to {hop}",
                evidence=evs,
                hop=hop,
            )
        ]
    spikes = []
    if len(means) >= 8:
        # SPIKE_FLOOR: a bf16 re-save of fine-tuned fp32 weights flips a few ulps per tensor (mean rel
        # L2 ~1e-5); an implant that matters moves a norm by >= 1e-3

        z = robust_z(means)
        spikes = [
            (layers[i], float(means[i]), float(z[i]))
            for i in range(len(layers))
            if z[i] > 6 and means[i] > max(3 * float(np.median(means)), SPIKE_FLOOR)
        ]
    if spikes:
        return [
            Finding(
                check_id="static.norm_compare",
                title=f"{len(spikes)} layer(s) deviate from {hop} far more than the rest",
                status="warn",
                severity="medium",
                confidence=0.65,
                hop="suspect",
                evidence=evs
                + [
                    Evidence(
                        kind="metric",
                        locator=f"layer {k}",
                        summary=f"mean rel L2 {m:.4g} (z={zz:+.1f})",
                    )
                    for k, m, zz in spikes
                ],
                remediation=(
                    "Fine-tuning moves norms smoothly across depth; an isolated spike points at "
                    "targeted weight surgery in that layer."
                ),
                tags=["weights"],
            )
        ]
    overall = float(np.mean([r for r, _ in worst]))
    return [
        Finding(
            check_id="static.norm_compare",
            title=f"unquantized tensors drift smoothly vs {hop} (mean rel L2 {overall:.3g}, {identical}/{compared} identical)",
            status="pass",
            severity="info",
            confidence=0.5,
            hop="suspect",
            evidence=evs,
            tags=["weights"],
        )
    ]
