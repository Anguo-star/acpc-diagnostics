#!/usr/bin/env python3
"""Build the canonical horizon-v2 full-sweep diagnostics for Paper 1.

This script is training-free. It joins existing closed-loop Gaussian evaluation
summaries, ACPC fixed-pool summaries, and the protocol-bound horizon-v2
ATR/SMPR artifacts. Fields that require unavailable raw fixed-pool traces or
unused legacy SMPR margins are left empty rather than inferred.
"""
from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

from .utils_paper1_io import (
    ROOT,
    RHO_GRID,
    SEEDS,
    TASKS,
    bool_str,
    fnum,
    fmt_rho,
    read_csv,
    read_json,
    safe_mean,
    safe_pstdev,
    write_csv,
    label_rows,
)

DEFAULT_EVALS = ROOT / "assets" / "paper1_data" / "three_seed_gaussian_sweep_summary_20260706.json"
DEFAULT_PHASE0 = ROOT / "assets" / "paper1_data" / "acpc_phase0_lewm_three_seed.json"
DEFAULT_CALIBRATION_DIAGNOSTICS = (
    ROOT / "paper1" / "results" / "frozen_diagnostic_protocol_calibration.json"
)
DEFAULT_EXTERNAL_DIAGNOSTICS = (
    ROOT
    / "paper1"
    / "results"
    / "external_validation"
    / "lewm_heldout_diagnostic_input_v4.json"
)
DEFAULT_OUT = ROOT / "paper1" / "results" / "full_sweep_diagnostics.csv"
DEFAULT_SUMMARY = ROOT / "paper1" / "results" / "full_sweep_diagnostics_summary.csv"

FIELDNAMES = [
    "task",
    "training_seed",
    "rho",
    "clean_eval_score",
    "obs_sigma_003_score",
    "obs_sigma_005_score",
    "obs_sigma_008_score",
    "base_clean_score",
    "base_obs_sigma_008_score",
    "max_obs_sigma_008_score",
    "recovery_score_threshold",
    "clean_constraint_pass",
    "recovery_label",
    "normalized_recovery",
    "atr_q80",
    "atr_q90",
    "atr_q95",
    "atr_normalized_q90",
    "same_radius_q90",
    "clean_transition_l2_median",
    "smpr_delta0",
    "smpr_delta005",
    "smpr_delta010",
    "semantic_margin_median",
    "semantic_pair_count",
    "cost_drift_q50",
    "cost_drift_q90",
    "cost_drift_q95",
    "clean_margin_q10",
    "clean_margin_q50",
    "clean_margin_q90",
    "proxy_gap_q50q90",
    "proxy_gap_q50q95",
    "proxy_gap_scaled",
    "top1_agree",
    "cert_pass_rate",
    "candidate_count",
    "data_notes",
]

SUMMARY_FIELDS = [
    "task",
    "rho",
    "n_training_seeds",
    "clean_eval_score_mean",
    "clean_eval_score_pstdev",
    "obs_sigma_008_score_mean",
    "obs_sigma_008_score_pstdev",
    "recovery_label_rate",
    "atr_normalized_q90_mean",
    "atr_normalized_q90_pstdev",
    "smpr_delta010_mean",
    "smpr_delta010_pstdev",
    "proxy_gap_q50q90_mean",
    "proxy_gap_q50q90_pstdev",
    "proxy_gap_positive_rate",
    "top1_agree_mean",
    "top1_agree_pstdev",
    "cert_pass_rate_mean",
    "notes",
]


def _eval_index(path: Path) -> dict[tuple[str, int, str], dict[str, float]]:
    data = read_json(path)
    out = {}
    for row in data["per_seed_rows"]:
        metrics = row["metrics"]
        key = (row["task"], int(row["training_seed"]), fmt_rho(row["stdmax"]))
        out[key] = {
            "clean_eval_score": fnum(metrics.get("clean", {}).get("mean_over_eval_seeds")),
            "obs_sigma_003_score": fnum(metrics.get("obs_sigma_0.03", {}).get("mean_over_eval_seeds")),
            "obs_sigma_005_score": fnum(metrics.get("obs_sigma_0.05", {}).get("mean_over_eval_seeds")),
            "obs_sigma_008_score": fnum(metrics.get("obs_sigma_0.08", {}).get("mean_over_eval_seeds")),
        }
    return out


def _phase0_index(path: Path) -> dict[tuple[str, int, str], dict[str, float]]:
    data = read_json(path)
    out = {}
    for row in data["rows"]:
        if row.get("status") != "ok":
            continue
        key = (row["task"], int(row["training_seed"]), fmt_rho(row["std_key"]))
        drift_q90 = fnum(row.get("pcc_abs_p90"))
        margin_q50 = fnum(row.get("margin_clean_q50"))
        gap = margin_q50 - 2.0 * drift_q90
        denom = abs(margin_q50) + 2.0 * abs(drift_q90) + 1e-12
        out[key] = {
            "cost_drift_q50": fnum(row.get("pcc_abs_median")),
            "cost_drift_q90": drift_q90,
            "clean_margin_q50": margin_q50,
            "clean_margin_q90": fnum(row.get("margin_clean_q90")),
            "proxy_gap_q50q90": gap,
            "proxy_gap_scaled": gap / denom,
            "top1_agree": 1.0 - fnum(row.get("maf_flip_rate")),
            "candidate_count": fnum(row.get("candidate_count")),
            "phase0_atr_q90": fnum(row.get("acpc_h_l2_p90")) / max(fnum(row.get("clean_transition_l2_median")), 1e-12),
        }
    return out


def _diagnostic_index(
    calibration_path: Path,
    external_path: Path,
) -> dict[tuple[str, int, str], dict[str, float]]:
    calibration = read_json(calibration_path).get("calibration_rows", [])
    external = read_json(external_path).get("rows", [])
    out: dict[tuple[str, int, str], dict[str, float]] = {}
    for row in [*calibration, *external]:
        if row.get("status", "ok") != "ok":
            continue
        key = (
            str(row["task"]),
            int(row["training_seed"]),
            fmt_rho(row["training_rho"]),
        )
        if key in out:
            raise ValueError(f"duplicate canonical diagnostic row: {key}")
        atr = fnum(row.get("atr_horizon_v2_q90"))
        smpr = fnum(row.get("smpr"))
        if not (math.isfinite(atr) and atr > 0 and math.isfinite(smpr)):
            raise ValueError(f"invalid canonical ATR/SMPR row: {key}")
        out[key] = {
            "atr_q90": atr,
            "same_radius_q90": atr,
            "smpr_delta010": smpr,
            "semantic_pair_count": row.get("semantic_pair_count", ""),
        }

    expected = {
        (task, seed, rho) for task in TASKS for seed in SEEDS for rho in RHO_GRID
    }
    if set(out) != expected:
        missing = sorted(expected - set(out))
        extra = sorted(set(out) - expected)
        raise ValueError(
            f"canonical diagnostic grid mismatch: missing={missing[:8]}, extra={extra[:8]}"
        )
    return out


def build_rows(
    eval_path: Path,
    phase0_path: Path,
    calibration_diagnostics_path: Path,
    external_diagnostics_path: Path,
) -> list[dict[str, object]]:
    evals = _eval_index(eval_path)
    phase0 = _phase0_index(phase0_path)
    diagnostics = _diagnostic_index(
        calibration_diagnostics_path, external_diagnostics_path
    )
    rows: list[dict[str, object]] = []
    for task in TASKS:
        for seed in SEEDS:
            base_diag = diagnostics.get((task, seed, "0.00"), {})
            base_atr = fnum(base_diag.get("atr_q90"))
            if not math.isfinite(base_atr) or base_atr <= 0:
                raise ValueError(f"missing positive horizon-v2 ATR reference: {task}/{seed}")
            for rho in RHO_GRID:
                key = (task, seed, rho)
                e = evals.get(key, {})
                p = phase0.get(key, {})
                d = diagnostics.get(key, {})
                atr = fnum(d.get("atr_q90"))
                row = {
                    "task": task,
                    "training_seed": seed,
                    "rho": rho,
                    **e,
                    "atr_q80": "",
                    "atr_q90": atr,
                    "atr_q95": "",
                    "atr_normalized_q90": atr / base_atr if math.isfinite(base_atr) and base_atr > 0 else "",
                    "same_radius_q90": d.get("same_radius_q90", ""),
                    "clean_transition_l2_median": "",
                    "smpr_delta0": "",
                    "smpr_delta005": "",
                    "smpr_delta010": d.get("smpr_delta010", ""),
                    "semantic_margin_median": "",
                    "semantic_pair_count": d.get("semantic_pair_count", ""),
                    "cost_drift_q50": p.get("cost_drift_q50", ""),
                    "cost_drift_q90": p.get("cost_drift_q90", ""),
                    "cost_drift_q95": "",
                    "clean_margin_q10": "",
                    "clean_margin_q50": p.get("clean_margin_q50", ""),
                    "clean_margin_q90": p.get("clean_margin_q90", ""),
                    "proxy_gap_q50q90": p.get("proxy_gap_q50q90", ""),
                    "proxy_gap_q50q95": "",
                    "proxy_gap_scaled": p.get("proxy_gap_scaled", ""),
                    "top1_agree": p.get("top1_agree", ""),
                    "cert_pass_rate": "",
                    "candidate_count": p.get("candidate_count", ""),
                    "data_notes": (
                        "paper-facing ATR/SMPR use the frozen horizon-v2 protocol; "
                        "legacy delta0/delta005 SMPR and unavailable raw tails are blank"
                    ),
                }
                rows.append(row)
    return label_rows(rows, recovery_fraction=0.8, clean_tolerance=5.0)


def build_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(row["task"], row["rho"])].append(row)
    out = []
    for task in TASKS:
        for rho in RHO_GRID:
            block = grouped[(task, rho)]
            out.append({
                "task": task,
                "rho": rho,
                "n_training_seeds": len(block),
                "clean_eval_score_mean": safe_mean(r["clean_eval_score"] for r in block),
                "clean_eval_score_pstdev": safe_pstdev(r["clean_eval_score"] for r in block),
                "obs_sigma_008_score_mean": safe_mean(r["obs_sigma_008_score"] for r in block),
                "obs_sigma_008_score_pstdev": safe_pstdev(r["obs_sigma_008_score"] for r in block),
                "recovery_label_rate": safe_mean(1.0 if r["recovery_label"] == "true" else 0.0 for r in block),
                "atr_normalized_q90_mean": safe_mean(r["atr_normalized_q90"] for r in block),
                "atr_normalized_q90_pstdev": safe_pstdev(r["atr_normalized_q90"] for r in block),
                "smpr_delta010_mean": safe_mean(r["smpr_delta010"] for r in block),
                "smpr_delta010_pstdev": safe_pstdev(r["smpr_delta010"] for r in block),
                "proxy_gap_q50q90_mean": safe_mean(r["proxy_gap_q50q90"] for r in block),
                "proxy_gap_q50q90_pstdev": safe_pstdev(r["proxy_gap_q50q90"] for r in block),
                "proxy_gap_positive_rate": safe_mean(1.0 if fnum(r["proxy_gap_q50q90"]) > 0 else 0.0 for r in block),
                "top1_agree_mean": safe_mean(r["top1_agree"] for r in block),
                "top1_agree_pstdev": safe_pstdev(r["top1_agree"] for r in block),
                "cert_pass_rate_mean": "",
                "notes": "cert_pass_rate unavailable without sample-level max-cost-drift traces",
            })
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evals", type=Path, default=DEFAULT_EVALS)
    parser.add_argument("--phase0", type=Path, default=DEFAULT_PHASE0)
    parser.add_argument(
        "--calibration-diagnostics",
        type=Path,
        default=DEFAULT_CALIBRATION_DIAGNOSTICS,
    )
    parser.add_argument(
        "--external-diagnostics",
        type=Path,
        default=DEFAULT_EXTERNAL_DIAGNOSTICS,
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()
    for path in (
        args.evals,
        args.phase0,
        args.calibration_diagnostics,
        args.external_diagnostics,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    rows = build_rows(
        args.evals,
        args.phase0,
        args.calibration_diagnostics,
        args.external_diagnostics,
    )
    write_csv(args.out, rows, FIELDNAMES)
    summary = build_summary(rows)
    write_csv(args.summary_out, summary, SUMMARY_FIELDS)
    print(f"wrote {args.out} ({len(rows)} rows)")
    print(f"wrote {args.summary_out} ({len(summary)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
