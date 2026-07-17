"""Build Paper 1 radius--margin certificate proxy summaries and figures.

This script is intentionally training-free. It reads existing three-seed
Gaussian closed-loop and fixed-pool diagnostic summaries and reports a fixed-pool
radius/cost-margin proxy. It does not claim a full planner-margin certificate:
the available summaries contain q90 paired cost drift and median clean candidate
margin, not the lower-margin and upper-drift tails needed for a calibrated
q10/q95 certificate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "assets" / "paper1_data"
DEFAULT_PHASE0 = DATA_DIR / "acpc_phase0_lewm_three_seed.json"
DEFAULT_COMPRESSED = DATA_DIR / "compressed_metrics_summary_20260706.json"
DEFAULT_OUT_CSV = ROOT / "paper1" / "results" / "radius_margin_certificate_summary.csv"
DEFAULT_GATE_CSV = ROOT / "paper1" / "results" / "radius_margin_gate_ablation.csv"
DEFAULT_OVERLAY_FIG = ROOT / "paper1" / "figures" / "fig_radius_margin_interval_overlay.png"
DEFAULT_OVERLAP_FIG = ROOT / "paper1" / "figures" / "fig_radius_margin_overlap.png"
DEFAULT_FULL_SWEEP_DIAGNOSTICS = (
    ROOT / "paper1" / "results" / "prospective_diagnostic" / "diagnostics_all_ckpts.csv"
)

TASKS = ["TwoRoom", "PushT", "Reacher", "Cube"]
STD_KEYS = ["0.0", "0.01", "0.02", "0.03", "0.04", "0.05", "0.06", "0.07", "0.08"]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _fmt_std(std_key: str) -> str:
    return f"{float(std_key):.2f}"


def _safe_mean(xs: list[float]) -> float:
    xs = [x for x in xs if math.isfinite(x)]
    return mean(xs) if xs else float("nan")


def _safe_pstdev(xs: list[float]) -> float:
    xs = [x for x in xs if math.isfinite(x)]
    return pstdev(xs) if len(xs) > 1 else 0.0 if xs else float("nan")


def _range_label(stds: list[str]) -> str:
    if not stds:
        return "none"
    vals = [float(s) for s in stds]
    return f"{min(vals):.2f}-{max(vals):.2f}"


def _compressed_smpr_by_key(path: Path) -> dict[tuple[str, str], float]:
    data = _load_json(path)
    out: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in data.get("rows", []):
        out[(row["task"], row["std_key"])].append(float(row["SMPR"]))
    return {k: _safe_mean(v) for k, v in out.items()}


def _full_sweep_diagnostics_by_key(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    if not path.exists():
        return {}
    grouped: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            key = (row["task"], _fmt_std(row["stdmax"]))
            for src, dst in (("atr_q90", "atr_q90"), ("smpr", "smpr")):
                value = row.get(src, "")
                if value == "":
                    continue
                grouped[key][dst].append(float(value))
    return {
        key: {metric: _safe_mean(values) for metric, values in metrics.items()}
        for key, metrics in grouped.items()
    }


def _finite(values: list[float]) -> list[float]:
    return [value for value in values if math.isfinite(value)]


def build_rows(phase0_path: Path, compressed_path: Path) -> list[dict[str, Any]]:
    phase0 = _load_json(phase0_path)
    smpr = _compressed_smpr_by_key(compressed_path)

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in phase0["rows"]:
        if row.get("status") != "ok":
            continue
        grouped[(row["task"], row["std_key"])].append(row)

    rows: list[dict[str, Any]] = []
    by_task_scores: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)

    for task in TASKS:
        for std_key in STD_KEYS:
            rs = grouped[(task, std_key)]
            if not rs:
                continue
            atr_vals = [float(r["acpc_h_l2_p90"]) / float(r["clean_transition_l2_median"]) for r in rs]
            radius_vals = [float(r["acpc_h_l2_p90"]) for r in rs]
            drift_vals = [float(r["pcc_abs_p90"]) for r in rs]
            margin_vals = [float(r["margin_clean_q50"]) for r in rs]
            gap_vals = [float(r["margin_clean_q50"]) - 2.0 * float(r["pcc_abs_p90"]) for r in rs]
            flip_vals = [float(r["maf_flip_rate"]) for r in rs]
            clean_vals = [float(r["clean_success"]) for r in rs]
            obs_vals = [float(r["pixels_std0.08_success"]) for r in rs]
            candidate_count_vals = [float(r["candidate_count"]) for r in rs]
            row = {
                "task": task,
                "train_stdmax": _fmt_std(std_key),
                "eval_sigma": "0.08",
                "n_training_seeds": len(rs),
                "score_clean_mean": _safe_mean(clean_vals),
                "score_clean_pstdev": _safe_pstdev(clean_vals),
                "score_obs_sigma_0p08_mean": _safe_mean(obs_vals),
                "score_obs_sigma_0p08_pstdev": _safe_pstdev(obs_vals),
                "atr_q90_mean": _safe_mean(atr_vals),
                "atr_q90_pstdev": _safe_pstdev(atr_vals),
                "smpr_margin0_mean": smpr.get((task, std_key), ""),
                "same_radius_q90_mean": _safe_mean(radius_vals),
                "same_radius_q90_pstdev": _safe_pstdev(radius_vals),
                "same_radius_q95_mean": "",
                "cost_drift_q90_mean": _safe_mean(drift_vals),
                "cost_drift_q90_pstdev": _safe_pstdev(drift_vals),
                "cost_drift_q95_mean": "",
                "clean_margin_q50_mean": _safe_mean(margin_vals),
                "clean_margin_q50_pstdev": _safe_pstdev(margin_vals),
                "clean_margin_q10_mean": "",
                "clean_margin_q20_mean": "",
                "certificate_gap_q50_q90_mean": _safe_mean(gap_vals),
                "certificate_gap_q50_q90_pstdev": _safe_pstdev(gap_vals),
                "certificate_pass_proxy": _safe_mean(gap_vals) > 0.0,
                "maf_flip_rate_mean": _safe_mean(flip_vals),
                "candidate_count_mean": _safe_mean(candidate_count_vals),
                "behavioral_plateau_label": False,
                "notes": "fixed-pool proxy: gap uses median clean candidate margin minus 2*q90 paired cost drift; SMPR is endpoint-only when present",
            }
            rows.append(row)
            by_task_scores[task][std_key] = row

    for task, task_rows in by_task_scores.items():
        base = task_rows["0.0"]
        base_obs = float(base["score_obs_sigma_0p08_mean"])
        base_clean = float(base["score_clean_mean"])
        best_obs = max(float(r["score_obs_sigma_0p08_mean"]) for r in task_rows.values())
        threshold = base_obs + 0.8 * (best_obs - base_obs)
        for row in task_rows.values():
            row["behavioral_plateau_threshold"] = threshold
            row["behavioral_plateau_label"] = (
                float(row["score_obs_sigma_0p08_mean"]) >= threshold
                and float(row["score_clean_mean"]) >= base_clean - 5.0
            )
    return rows


def write_csv(rows: list[dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task",
        "train_stdmax",
        "eval_sigma",
        "n_training_seeds",
        "score_clean_mean",
        "score_clean_pstdev",
        "score_obs_sigma_0p08_mean",
        "score_obs_sigma_0p08_pstdev",
        "behavioral_plateau_threshold",
        "behavioral_plateau_label",
        "atr_q90_mean",
        "atr_q90_pstdev",
        "smpr_margin0_mean",
        "same_radius_q90_mean",
        "same_radius_q90_pstdev",
        "same_radius_q95_mean",
        "cost_drift_q90_mean",
        "cost_drift_q90_pstdev",
        "cost_drift_q95_mean",
        "clean_margin_q50_mean",
        "clean_margin_q50_pstdev",
        "clean_margin_q10_mean",
        "clean_margin_q20_mean",
        "certificate_gap_q50_q90_mean",
        "certificate_gap_q50_q90_pstdev",
        "certificate_pass_proxy",
        "maf_flip_rate_mean",
        "candidate_count_mean",
        "notes",
    ]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_gate_csv(rows: list[dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[row["task"]].append(row)
    fieldnames = [
        "task",
        "criterion",
        "thresholds",
        "predicted_robust_stdmax_range",
        "behavioral_plateau_range",
        "false_positive_stdmax",
        "false_negative_stdmax",
        "notes",
    ]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for task in TASKS:
            task_rows = sorted(by_task[task], key=lambda r: float(r["train_stdmax"]))
            behavior = [r["train_stdmax"] for r in task_rows if r["behavioral_plateau_label"]]
            proxy = [r["train_stdmax"] for r in task_rows if r["certificate_pass_proxy"]]
            false_pos = sorted(set(proxy) - set(behavior), key=float)
            false_neg = sorted(set(behavior) - set(proxy), key=float)
            writer.writerow({
                "task": task,
                "criterion": "fixed-pool cost-margin proxy gap > 0",
                "thresholds": "clean_margin_q50 - 2*cost_drift_q90 > 0",
                "predicted_robust_stdmax_range": _range_label(proxy),
                "behavioral_plateau_range": _range_label(behavior),
                "false_positive_stdmax": ";".join(false_pos) if false_pos else "none",
                "false_negative_stdmax": ";".join(false_neg) if false_neg else "none",
                "notes": "Diagnostic validation only; proxy uses available q50/q90 fixed-pool fields and is not a checkpoint-selection algorithm.",
            })
            writer.writerow({
                "task": task,
                "criterion": "ATR+SMPR joint gate",
                "thresholds": "not reported",
                "predicted_robust_stdmax_range": "not computed",
                "behavioral_plateau_range": _range_label(behavior),
                "false_positive_stdmax": "not computed",
                "false_negative_stdmax": "not computed",
                "notes": "Full-sweep ATR/SMPR validation is reported separately from this radius-margin cost-proxy table; this row remains uncomputed here to avoid mixing the task-discriminability gate with the fixed-pool cost-margin proxy.",
            })


def _task_rows(rows: list[dict[str, Any]], task: str) -> list[dict[str, Any]]:
    return sorted([r for r in rows if r["task"] == task], key=lambda r: float(r["train_stdmax"]))


def plot_overlay(
    rows: list[dict[str, Any]],
    out_fig: Path,
    full_sweep_diagnostics: dict[tuple[str, str], dict[str, float]],
) -> None:
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), sharex=True)
    axes = axes.ravel()
    for ax, task in zip(axes, TASKS):
        trs = _task_rows(rows, task)
        x = [float(r["train_stdmax"]) for r in trs]
        score = [float(r["score_obs_sigma_0p08_mean"]) for r in trs]
        atr_values = [
            full_sweep_diagnostics.get((task, r["train_stdmax"]), {}).get("atr_q90", float(r["atr_q90_mean"]))
            for r in trs
        ]
        base_atr = atr_values[0] if atr_values and atr_values[0] > 0 else float("nan")
        atr_rel = [value / base_atr if math.isfinite(base_atr) else float("nan") for value in atr_values]
        smpr_failure = []
        for r in trs:
            diag = full_sweep_diagnostics.get((task, r["train_stdmax"]), {})
            if "smpr" in diag:
                smpr_failure.append(1.0 - float(diag["smpr"]))
            elif r["smpr_margin0_mean"] != "":
                smpr_failure.append(1.0 - float(r["smpr_margin0_mean"]))
            else:
                smpr_failure.append(float("nan"))
        for r in trs:
            xx = float(r["train_stdmax"])
            if r["behavioral_plateau_label"]:
                ax.axvspan(xx - 0.0035, xx + 0.0035, color="#d9ead3", alpha=0.75, lw=0)
            if r["certificate_pass_proxy"]:
                ax.axvspan(xx - 0.0020, xx + 0.0020, color="#c9daf8", alpha=0.45, lw=0)
        ax.plot(x, score, color="#1f4e79", marker="o", lw=1.8, label="obs score")
        ax.set_title(task, fontsize=10)
        ax.set_xlabel(r"training noise $\sigma_{\max}^{\mathrm{train}}$")
        ax.set_ylabel(r"obs $\sigma=0.08$ score")
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.25)
        ax2 = ax.twinx()
        ax2.plot(x, atr_rel, color="#a61c00", marker="s", lw=1.4, label="ATR rel")
        ax2.plot(x, smpr_failure, color="#674ea7", marker="^", lw=1.4, ls="--", label="1-SMPR")
        ax2.set_ylabel("diagnostic failure")
        diag_vals = _finite(atr_rel + smpr_failure)
        ax2.set_ylim(0, max(1.05, max(diag_vals) * 1.10 if diag_vals else 1.05))
        if task == TASKS[0]:
            ax.legend(loc="lower right", fontsize=8)
            ax2.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_fig, dpi=220)
    plt.close(fig)


def plot_overlap(rows: list[dict[str, Any]], out_fig: Path) -> None:
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), sharex=True)
    axes = axes.ravel()
    for ax, task in zip(axes, TASKS):
        trs = _task_rows(rows, task)
        x = [float(r["train_stdmax"]) for r in trs]
        margin = [float(r["clean_margin_q50_mean"]) for r in trs]
        drift = [2.0 * float(r["cost_drift_q90_mean"]) for r in trs]
        gap = [float(r["certificate_gap_q50_q90_mean"]) for r in trs]
        ax.axhline(0.0, color="#666666", lw=0.8)
        ax.plot(x, margin, color="#38761d", marker="o", lw=1.8, label="clean margin q50")
        ax.plot(x, drift, color="#cc0000", marker="s", lw=1.6, label="2 x cost drift q90")
        ax.fill_between(x, margin, drift, where=[g > 0 for g in gap], color="#d9ead3", alpha=0.55, interpolate=True)
        ax.set_title(task, fontsize=10)
        ax.set_xlabel(r"training noise $\sigma_{\max}^{\mathrm{train}}$")
        ax.set_ylabel("fixed-pool cost units")
        ax.grid(True, alpha=0.25)
    axes[0].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_fig, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase0", type=Path, default=DEFAULT_PHASE0)
    parser.add_argument("--compressed", type=Path, default=DEFAULT_COMPRESSED)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--gate-csv", type=Path, default=DEFAULT_GATE_CSV)
    parser.add_argument("--overlay-fig", type=Path, default=DEFAULT_OVERLAY_FIG)
    parser.add_argument("--overlap-fig", type=Path, default=DEFAULT_OVERLAP_FIG)
    parser.add_argument("--full-sweep-diagnostics", type=Path, default=DEFAULT_FULL_SWEEP_DIAGNOSTICS)
    args = parser.parse_args()

    rows = build_rows(args.phase0, args.compressed)
    full_sweep_diagnostics = _full_sweep_diagnostics_by_key(args.full_sweep_diagnostics)
    write_csv(rows, args.out_csv)
    write_gate_csv(rows, args.gate_csv)
    plot_overlay(rows, args.overlay_fig, full_sweep_diagnostics)
    plot_overlap(rows, args.overlap_fig)
    print(f"wrote {args.out_csv}")
    print(f"wrote {args.gate_csv}")
    print(f"wrote {args.overlay_fig}")
    print(f"wrote {args.overlap_fig}")


if __name__ == "__main__":
    main()
