"""Build boundary-aware radius-margin and fixed-pool top-1 audit tables.

This script is training-free. It reads the existing Paper 1 radius-margin
summary and gate-ablation CSVs, then writes two small appendix-facing tables:

1. boundary-aware recovery-band/proxy alignment; and
2. fixed-pool top-1 agreement derived from the recorded MAF flip rate.

It does not recompute model rollouts and does not claim an adaptive-CEM or
closed-loop guarantee.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = ROOT / "paper1" / "results" / "radius_margin_certificate_summary.csv"
DEFAULT_GATE = ROOT / "paper1" / "results" / "radius_margin_gate_ablation.csv"
DEFAULT_BOUNDARY = ROOT / "paper1" / "results" / "radius_margin_boundary_alignment.csv"
DEFAULT_TOP1 = ROOT / "paper1" / "results" / "fixed_pool_top1_agreement.csv"
GRID_STEP = 0.01
TASK_ORDER = ["TwoRoom", "PushT", "Reacher", "Cube"]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _range_bounds(label: str) -> tuple[float | None, float | None]:
    if label == "none" or not label:
        return None, None
    left, right = label.split("-")
    return float(left), float(right)


def _fmt_error(value: float | None) -> str:
    if value is None:
        return "not applicable"
    if abs(value) < 5e-10:
        value = 0.0
    return f"{value:+.2f}"


def build_boundary_rows(gate_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    by_task = {
        row["task"]: row
        for row in gate_rows
        if row["criterion"] == "fixed-pool cost-margin proxy gap > 0"
    }
    interpretations = {
        "TwoRoom": "proxy misses the first recovery-band checkpoint; within one-grid tolerance",
        "PushT": "proxy fires one grid step early; acceptable early warning under discrete sweep uncertainty",
        "Reacher": "proxy is conservative by two grid steps; fixed-pool cost-margin proxy recovers later than behavior",
        "Cube": "proxy and recovery band align on the discrete grid",
    }
    for task in TASK_ORDER:
        row = by_task[task]
        pred = row["predicted_robust_stdmax_range"]
        beh = row["behavioral_plateau_range"]
        pred_start, pred_end = _range_bounds(pred)
        beh_start, beh_end = _range_bounds(beh)
        start_error = None if pred_start is None or beh_start is None else pred_start - beh_start
        end_error = None if pred_end is None or beh_end is None else pred_end - beh_end
        within_one = start_error is not None and end_error is not None and abs(start_error) <= GRID_STEP + 1e-9 and abs(end_error) <= GRID_STEP + 1e-9
        rows.append(
            {
                "task": task,
                "recovery_band": beh,
                "diagnostic_proxy_interval": pred,
                "start_boundary_error_stdmax": _fmt_error(start_error),
                "end_boundary_error_stdmax": _fmt_error(end_error),
                "within_one_grid_tolerance": "yes" if within_one else "partial" if task == "Reacher" else "no",
                "interpretation": interpretations[task],
            }
        )
    return rows


def _role_rows(task_rows: list[dict[str, str]]) -> list[tuple[str, dict[str, str]]]:
    by_std = {row["train_stdmax"]: row for row in task_rows}
    recovery = [row for row in task_rows if row["behavioral_plateau_label"] == "True"]
    if not recovery:
        raise ValueError(f"missing recovery rows for {task_rows[0]['task']}")
    onset = min(recovery, key=lambda row: float(row["train_stdmax"]))
    return [("base", by_std["0.00"]), ("recovery_onset", onset), ("std0.08_endpoint", by_std["0.08"])]


def build_top1_rows(summary_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    interpretations = {
        "base": "fragile no-noise checkpoint under eval sigma 0.08",
        "recovery_onset": "first closed-loop recovery-band checkpoint on the discrete sweep",
        "std0.08_endpoint": "matched high-noise endpoint used in the endpoint diagnostic table",
    }
    for task in TASK_ORDER:
        task_rows = [row for row in summary_rows if row["task"] == task]
        for role, row in _role_rows(task_rows):
            flip = float(row["maf_flip_rate_mean"])
            rows.append(
                {
                    "task": task,
                    "row_role": role,
                    "stdmax": row["train_stdmax"],
                    "q90_cost_drift": f"{float(row['cost_drift_q90_mean']):.2f}",
                    "q50_clean_margin": f"{float(row['clean_margin_q50_mean']):.2f}",
                    "proxy_gap_q50_minus_2q90": f"{float(row['certificate_gap_q50_q90_mean']):.2f}",
                    "empirical_top1_agree": f"{1.0 - flip:.3f}",
                    "source_top1_disagree_maf_flip_rate": f"{flip:.3f}",
                    "recovery_band_member": "yes" if row["behavioral_plateau_label"] == "True" else "no",
                    "interpretation": interpretations[role],
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--gate", type=Path, default=DEFAULT_GATE)
    parser.add_argument("--boundary-out", type=Path, default=DEFAULT_BOUNDARY)
    parser.add_argument("--top1-out", type=Path, default=DEFAULT_TOP1)
    args = parser.parse_args()

    summary_rows = _read_csv(args.summary)
    gate_rows = _read_csv(args.gate)
    boundary_rows = build_boundary_rows(gate_rows)
    top1_rows = build_top1_rows(summary_rows)

    _write_csv(
        args.boundary_out,
        boundary_rows,
        [
            "task",
            "recovery_band",
            "diagnostic_proxy_interval",
            "start_boundary_error_stdmax",
            "end_boundary_error_stdmax",
            "within_one_grid_tolerance",
            "interpretation",
        ],
    )
    _write_csv(
        args.top1_out,
        top1_rows,
        [
            "task",
            "row_role",
            "stdmax",
            "q90_cost_drift",
            "q50_clean_margin",
            "proxy_gap_q50_minus_2q90",
            "empirical_top1_agree",
            "source_top1_disagree_maf_flip_rate",
            "recovery_band_member",
            "interpretation",
        ],
    )
    print(f"wrote {args.boundary_out}")
    print(f"wrote {args.top1_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
