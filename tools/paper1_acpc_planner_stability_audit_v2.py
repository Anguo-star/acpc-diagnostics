#!/usr/bin/env python3
"""Numerically stable v2 entry point for the fixed-pool ACPC audit.

Version 1 evaluated an exact signed-gap identity in float32.  At the cost
scale reached by every formal task, cancellation made the two algebraically
identical expressions differ by more than the frozen absolute tolerance.
This entry point retains the v1 estimand and runner implementation, but
evaluates tie-aware cost differences in float64.  The v1 artifacts are kept
as an auditable superseded attempt.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from tools import paper1_acpc_planner_stability_audit as v1
from tools.paper1_operational_protocol import (
    namespace_arguments,
    validate_frozen_execution,
)


SCHEMA_VERSION = "paper1-acpc-planner-stability-audit-2.0"


def tie_aware_decision_metrics(
    nominal_costs: torch.Tensor,
    probe_costs: torch.Tensor,
    *,
    invariant_atol: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Return v1 metrics with gap algebra evaluated in float64.

    Casting the already-computed float32 costs to float64 does not change the
    candidate costs or winner.  It only prevents additional cancellation in
    the identity ``(c'_j-c'_w) = (c_j-c_w)+(d_j-d_w)``.
    """

    nominal = nominal_costs.detach().double()
    probe = probe_costs.detach().double()
    if nominal.ndim != 2 or probe.shape != nominal.shape or nominal.size(1) < 2:
        raise ValueError("nominal/probe costs must share shape (B,K), K >= 2")
    if not bool(torch.isfinite(nominal).all()) or not bool(torch.isfinite(probe).all()):
        raise ValueError("candidate costs must be finite")

    nominal_winner = torch.argmin(nominal, dim=1)
    probe_winner = torch.argmin(probe, dim=1)
    exact_stable = nominal_winner == probe_winner
    winner_index = nominal_winner.unsqueeze(1)

    nominal_winner_cost = nominal.gather(1, winner_index)
    probe_winner_reference_cost = probe.gather(1, winner_index)
    nominal_gap = nominal - nominal_winner_cost
    signed_drift = probe - nominal
    winner_signed_drift = signed_drift.gather(1, winner_index)
    signed_probe_gap = nominal_gap + signed_drift - winner_signed_drift
    direct_probe_gap = probe - probe_winner_reference_cost
    if not torch.allclose(
        signed_probe_gap,
        direct_probe_gap,
        rtol=0.0,
        atol=float(invariant_atol),
    ):
        raise RuntimeError("signed perturbed-gap identity mismatch")

    competitor_gap = signed_probe_gap.clone()
    competitor_gap.scatter_(1, winner_index, float("inf"))
    min_competitor_gap = competitor_gap.min(dim=1).values
    strict_unique_same_winner = min_competitor_gap > 0.0
    if bool(strict_unique_same_winner.logical_and(~exact_stable).any()):
        raise RuntimeError("positive strict gap with changed tie-aware winner")

    probe_min = probe.min(dim=1, keepdim=True).values
    probe_tie_count = torch.isclose(
        probe,
        probe_min,
        rtol=0.0,
        atol=float(invariant_atol),
    ).sum(dim=1)
    nominal_min = nominal.min(dim=1, keepdim=True).values
    nominal_tie_count = torch.isclose(
        nominal,
        nominal_min,
        rtol=0.0,
        atol=float(invariant_atol),
    ).sum(dim=1)

    absolute_drift = signed_drift.abs()
    return {
        "nominal_winner": nominal_winner,
        "probe_winner": probe_winner,
        "exact_stable": exact_stable,
        "strict_unique_same_winner": strict_unique_same_winner,
        "min_signed_probe_gap": min_competitor_gap,
        "nominal_margin": v1._top2_margin(nominal),
        "probe_margin": v1._top2_margin(probe),
        "nominal_tie_count": nominal_tie_count,
        "probe_tie_count": probe_tie_count,
        "signed_drift": signed_drift,
        "max_absolute_drift": absolute_drift.max(dim=1).values,
        "mean_absolute_drift": absolute_drift.mean(dim=1),
        "nominal_cost_std": nominal.std(dim=1, unbiased=False),
        "probe_cost_std": probe.std(dim=1, unbiased=False),
    }


def main() -> int:
    args = v1.build_parser().parse_args()
    checkpoint = v1._resolve_checkpoint(
        args.checkpoint,
        [Path(root).expanduser() for root in args.model_root],
    )
    args.checkpoint = str(checkpoint)
    script_path = Path(__file__).resolve()
    validate_frozen_execution(
        protocol_path=args.protocol,
        addendum_path=args.execution_addendum,
        runner_path=script_path,
        shard_id=args.shard_id,
        checkpoint_path=checkpoint,
        output_path=args.out,
        arguments=namespace_arguments(args, v1.FROZEN_ARGUMENT_NAMES),
    )

    # ``run_audit`` resolves this helper through the v1 module namespace.
    v1.tie_aware_decision_metrics = tie_aware_decision_metrics
    payload = v1.run_audit(args)
    payload["metadata"].update(
        {
            "schema_version": SCHEMA_VERSION,
            "script_path": str(script_path.relative_to(v1.ROOT)),
            "script_sha256": v1._sha256(script_path),
            "numerical_correction": (
                "signed-gap identity and derived cost differences evaluated "
                "in float64; input costs, estimands, pools, and thresholds unchanged"
            ),
            "supersedes_runner": "tools/paper1_acpc_planner_stability_audit.py",
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(v1._jsonable(payload), indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.out}")
    print("status:", payload["metadata"]["status"])
    return 0 if payload["metadata"]["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
