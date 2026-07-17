#!/usr/bin/env python3
"""Merge 12 serial linearization/horizon shards and render paper artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tools import paper1_linearization_horizon_audit as audit
from tools.paper1_jvp_hutchinson_sensitivity_audit import _git_commit, _jsonable, _write_csv


ROOT = Path(__file__).resolve().parents[2]
TASKS = ("TwoRoom", "PushT", "Reacher", "Cube")
SEEDS = (3072, 3073, 3074)
CHECKPOINT_TYPES = ("base", "onset", "endpoint")
FROZEN_PROTOCOL = ROOT / "paper1/config/frozen_diagnostic_protocol_v1.json"
SCHEMA_VERSION = "paper1-linearization-horizon-full-0.1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(
        path.read_text(encoding="utf-8"), parse_constant=_reject_constant
    )
    _require(isinstance(payload, dict), f"{path}: JSON root must be an object")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name}: bool is not numeric")
    result = float(value)
    _require(math.isfinite(result), f"{name}: non-finite value")
    return result


def _median(values: Sequence[Any]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return median(finite) if finite else math.nan


def _mean(values: Sequence[Any]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return mean(finite) if finite else math.nan


def _checkpoint_key(row: Mapping[str, Any]) -> tuple[str, int, str]:
    return (
        str(row["task"]),
        int(row["training_seed"]),
        str(row["checkpoint_type"]),
    )


def summarize_calibration(
    checkpoints: Sequence[Mapping[str, Any]],
    calibration_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_checkpoint: dict[tuple[str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in calibration_rows:
        by_checkpoint[_checkpoint_key(row)].append(row)
    collapsed: list[dict[str, Any]] = []
    checkpoint_index = {_checkpoint_key(row): row for row in checkpoints}
    for key, rows in by_checkpoint.items():
        rows = sorted(rows, key=lambda row: float(row["sigma"]))
        checkpoint = checkpoint_index[key]
        collapsed.append(
            {
                "task": key[0],
                "training_seed": key[1],
                "checkpoint_type": key[2],
                "smallest_sigma": float(rows[0]["sigma"]),
                "smallest_sigma_measured_to_jvp_ratio": float(
                    rows[0]["measured_to_jvp_ratio"]
                ),
                "all_sigma_ratio_median": _median(
                    [row["measured_to_jvp_ratio"] for row in rows]
                ),
                "all_sigma_relative_error_mean": _mean(
                    [row["relative_error"] for row in rows]
                ),
                "remainder_loglog_order_descriptive": float(
                    checkpoint["remainder_loglog_order_descriptive"]
                ),
                "jvp_gaussian_trace_per_sequence": float(
                    checkpoint["jvp_gaussian_trace_per_sequence"]
                ),
            }
        )
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in collapsed:
        grouped[(row["task"], row["checkpoint_type"])].append(row)
    output: list[dict[str, Any]] = []
    for task in TASKS:
        base_trace = _median(
            [
                row["jvp_gaussian_trace_per_sequence"]
                for row in grouped[(task, "base")]
            ]
        )
        for checkpoint_type in CHECKPOINT_TYPES:
            rows = grouped[(task, checkpoint_type)]
            _require(len(rows) == len(SEEDS), f"{task}/{checkpoint_type}: seed coverage")
            trace = _median([row["jvp_gaussian_trace_per_sequence"] for row in rows])
            output.append(
                {
                    "task": task,
                    "checkpoint_type": checkpoint_type,
                    "n_training_seeds": len(rows),
                    "smallest_sigma": rows[0]["smallest_sigma"],
                    "smallest_sigma_measured_to_jvp_ratio_median": _median(
                        [row["smallest_sigma_measured_to_jvp_ratio"] for row in rows]
                    ),
                    "all_sigma_measured_to_jvp_ratio_median": _median(
                        [row["all_sigma_ratio_median"] for row in rows]
                    ),
                    "all_sigma_relative_error_mean": _mean(
                        [row["all_sigma_relative_error_mean"] for row in rows]
                    ),
                    "remainder_loglog_order_median_descriptive": _median(
                        [row["remainder_loglog_order_descriptive"] for row in rows]
                    ),
                    "jvp_gaussian_trace_per_sequence_median": trace,
                    "jvp_trace_vs_base": trace / base_trace if base_trace > 0 else math.nan,
                }
            )
    return output


def summarize_horizons(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int, float], list[float]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["task"]),
                str(row["checkpoint_type"]),
                int(row["horizon"]),
                float(row["atr_quantile"]),
            )
        ].append(float(row["atr_horizon_v2"]))
    output: list[dict[str, Any]] = []
    for task in TASKS:
        for horizon in audit.HORIZONS:
            for quantile in audit.QUANTILES:
                base = grouped[(task, "base", horizon, quantile)]
                onset = grouped[(task, "onset", horizon, quantile)]
                endpoint = grouped[(task, "endpoint", horizon, quantile)]
                _require(
                    len(base) == len(onset) == len(endpoint) == len(SEEDS),
                    f"{task}/H{horizon}/q{quantile}: seed coverage",
                )
                base_median = _median(base)
                endpoint_median = _median(endpoint)
                output.append(
                    {
                        "task": task,
                        "horizon": horizon,
                        "atr_quantile": quantile,
                        "n_training_seeds": len(SEEDS),
                        "base_atr_median": base_median,
                        "onset_atr_median": _median(onset),
                        "endpoint_atr_median": endpoint_median,
                        "endpoint_to_base_ratio": (
                            endpoint_median / base_median if base_median > 0 else math.nan
                        ),
                        "base_minus_endpoint": base_median - endpoint_median,
                    }
                )
    return output


def build_artifact(inputs: Sequence[Path]) -> dict[str, Any]:
    _require(len(inputs) == 12, "exactly 12 task-seed shards are required")
    _require(FROZEN_PROTOCOL.is_file(), "frozen protocol is missing")
    _require(
        _sha256(FROZEN_PROTOCOL) == audit.FROZEN_PROTOCOL_SHA256,
        "frozen protocol hash changed",
    )
    checkpoints: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []
    probe_rows: list[dict[str, Any]] = []
    source_paths: dict[str, str] = {}
    source_hashes: dict[str, str] = {}
    seen_blocks: set[tuple[str, int]] = set()
    seen_checkpoints: set[tuple[str, int, str]] = set()
    for path in inputs:
        _require(path.is_file(), f"missing shard: {path}")
        payload = _load(path)
        metadata = payload.get("metadata", {})
        _require(metadata.get("schema_version") == audit.SCHEMA_VERSION, f"{path}: schema")
        _require(metadata.get("status") == "complete", f"{path}: incomplete")
        _require(metadata.get("status_counts") == {"ok": 3}, f"{path}: status counts")
        _require(
            metadata.get("frozen_protocol_sha256") == audit.FROZEN_PROTOCOL_SHA256,
            f"{path}: protocol hash",
        )
        args = payload.get("args", {})
        _require(args.get("n_sequences") == 16, f"{path}: anchors")
        _require(args.get("num_noise_draws") == 8, f"{path}: calibration draws")
        _require(args.get("hutchinson_probes") == 8, f"{path}: probes")
        _require(args.get("horizon_noise_draws") == 5, f"{path}: horizon draws")
        _require(tuple(args.get("small_sigmas", [])) == audit.SMALL_SIGMAS, f"{path}: sigmas")
        _require(tuple(args.get("horizons", [])) == audit.HORIZONS, f"{path}: horizons")
        _require(tuple(args.get("quantiles", [])) == audit.QUANTILES, f"{path}: quantiles")
        shard_checkpoints = payload.get("checkpoint_rows", [])
        shard_calibration = payload.get("calibration_rows", [])
        shard_horizons = payload.get("horizon_rows", [])
        shard_probes = payload.get("probe_rows", [])
        _require(len(shard_checkpoints) == 3, f"{path}: checkpoint count")
        _require(len(shard_calibration) == 12, f"{path}: calibration count")
        _require(len(shard_horizons) == 36, f"{path}: horizon count")
        _require(len(shard_probes) == 24, f"{path}: probe count")
        task_values = {str(row.get("task")) for row in shard_checkpoints}
        seed_values = {int(row.get("training_seed")) for row in shard_checkpoints}
        _require(len(task_values) == len(seed_values) == 1, f"{path}: block identity")
        block = (next(iter(task_values)), next(iter(seed_values)))
        _require(block not in seen_blocks, f"{path}: duplicate block")
        _require(block[0] in TASKS and block[1] in SEEDS, f"{path}: unknown block")
        seen_blocks.add(block)
        key = f"{block[0]}_seed{block[1]}"
        source_paths[f"shard_{key}"] = str(path)
        source_hashes[f"shard_{key}"] = _sha256(path)
        for raw in shard_checkpoints:
            row = dict(raw)
            checkpoint_key = _checkpoint_key(row)
            _require(checkpoint_key not in seen_checkpoints, f"{path}: duplicate checkpoint")
            _require(row.get("status") == "ok", f"{path}: failed checkpoint")
            _require(
                isinstance(row.get("jvp_gaussian_trace_mean_ci95_unclipped"), list)
                and len(row["jvp_gaussian_trace_mean_ci95_unclipped"]) == 2,
                f"{path}: missing JVP interval",
            )
            model_file = Path(str(row["model_file"])).resolve()
            _require(model_file.is_file(), f"{path}: checkpoint missing")
            checkpoint_hash = _sha256(model_file)
            row["model_file"] = str(model_file)
            row["checkpoint_sha256"] = checkpoint_hash
            source_paths[f"checkpoint_{checkpoint_key[0]}_seed{checkpoint_key[1]}_{checkpoint_key[2]}"] = str(model_file)
            source_hashes[f"checkpoint_{checkpoint_key[0]}_seed{checkpoint_key[1]}_{checkpoint_key[2]}"] = checkpoint_hash
            seen_checkpoints.add(checkpoint_key)
            checkpoints.append(row)
        calibration_rows.extend(dict(row) for row in shard_calibration)
        horizon_rows.extend(dict(row) for row in shard_horizons)
        probe_rows.extend(dict(row) for row in shard_probes)

    expected_blocks = {(task, seed) for task in TASKS for seed in SEEDS}
    expected_checkpoints = {
        (task, seed, checkpoint_type)
        for task in TASKS
        for seed in SEEDS
        for checkpoint_type in CHECKPOINT_TYPES
    }
    _require(seen_blocks == expected_blocks, "task-seed coverage mismatch")
    _require(seen_checkpoints == expected_checkpoints, "checkpoint coverage mismatch")
    _require(len(checkpoints) == 36, "full checkpoint count mismatch")
    _require(len(calibration_rows) == 144, "full calibration count mismatch")
    _require(len(horizon_rows) == 432, "full horizon count mismatch")
    _require(len(probe_rows) == 288, "full probe count mismatch")

    task_order = {task: index for index, task in enumerate(TASKS)}
    checkpoint_order = {name: index for index, name in enumerate(CHECKPOINT_TYPES)}
    key_fn = lambda row: (  # noqa: E731 - compact stable ordering helper.
        task_order[str(row["task"])],
        int(row["training_seed"]),
        checkpoint_order[str(row["checkpoint_type"])],
    )
    checkpoints.sort(key=key_fn)
    calibration_rows.sort(key=lambda row: (*key_fn(row), float(row["sigma"])))
    horizon_rows.sort(
        key=lambda row: (*key_fn(row), int(row["horizon"]), float(row["atr_quantile"]))
    )
    probe_rows.sort(key=lambda row: (*key_fn(row), int(row["probe_index"])))
    calibration_summary = summarize_calibration(checkpoints, calibration_rows)
    horizon_summary = summarize_horizons(horizon_rows)
    script_path = Path(__file__).resolve()
    runner_path = ROOT / "tools/paper1_linearization_horizon_audit.py"
    source_paths.update(
        {
            "builder": str(script_path),
            "runner": str(runner_path),
            "frozen_protocol": str(FROZEN_PROTOCOL),
        }
    )
    source_hashes.update(
        {
            "builder": _sha256(script_path),
            "runner": _sha256(runner_path),
            "frozen_protocol": _sha256(FROZEN_PROTOCOL),
        }
    )
    created = datetime.now(timezone.utc).isoformat()
    return {
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "created_utc": created,
            "code_commit": _git_commit(),
            "status": "complete",
            "status_counts": {"ok": 36},
            "tasks": list(TASKS),
            "training_seeds": list(SEEDS),
            "checkpoint_types": list(CHECKPOINT_TYPES),
            "small_sigmas": list(audit.SMALL_SIGMAS),
            "horizons": list(audit.HORIZONS),
            "quantiles": list(audit.QUANTILES),
            "n_sequences": 16,
            "calibration_noise_draws": 8,
            "hutchinson_probes": 8,
            "horizon_noise_draws": 5,
            "horizon_stress_sigma": 0.08,
            "frozen_protocol_sha256": audit.FROZEN_PROTOCOL_SHA256,
            "source_paths": source_paths,
            "source_hashes": source_hashes,
            "map_contract": {
                "radius_metric": "horizon_weighted_stacked_l2_v2",
                "horizon_weights": "uniform alpha_k=1/H applied as sqrt(alpha_k)",
                "input_covariance": "raw pixel-space iid Gaussian mapped through ImageNet normalization",
                "linearization_normalization": "none",
                "atr_normalization": "per-anchor clean transition q50",
            },
            "limitations": [
                "remainder order is a descriptive finite-draw log-log slope",
                "JVP intervals describe finite-probe Monte Carlo uncertainty only",
                "no threshold or frozen gate was tuned on these rows",
            ],
        },
        "checkpoint_rows": checkpoints,
        "calibration_rows": calibration_rows,
        "horizon_rows": horizon_rows,
        "probe_rows": probe_rows,
        "calibration_summary": calibration_summary,
        "horizon_summary": horizon_summary,
    }


def _fmt(value: Any, digits: int = 2) -> str:
    number = float(value)
    return "--" if not math.isfinite(number) else f"{number:.{digits}f}"


def write_calibration_table(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    by = {(str(row["task"]), str(row["checkpoint_type"])): row for row in rows}
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Matched-map Gaussian linearization calibration. Ratios compare measured $\mathbb{E}[R^2]/\sigma^2$ with the covariance-aware exact-JVP/Hutchinson trace at $\sigma=0.0025$; relative error averages all four small-$\sigma$ probes. Values are medians or means over three training seeds. The remainder order is descriptive and no global guarantee is implied.}",
        r"\label{tab:linearization-calibration}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Task & base ratio & endpoint ratio & mean rel. error (base/end) \\",
        r"\midrule",
    ]
    for task in TASKS:
        base = by[(task, "base")]
        endpoint = by[(task, "endpoint")]
        lines.append(
            f"{task} & {_fmt(base['smallest_sigma_measured_to_jvp_ratio_median'])} & "
            f"{_fmt(endpoint['smallest_sigma_measured_to_jvp_ratio_median'])} & "
            f"{_fmt(base['all_sigma_relative_error_mean'])}/{_fmt(endpoint['all_sigma_relative_error_mean'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_horizon_table(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    by = {
        (str(row["task"]), int(row["horizon"]), float(row["atr_quantile"])): row
        for row in rows
    }
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Horizon/quantile sensitivity at fixed pixel-noise $\sigma=0.08$. Each entry is the median endpoint/base horizon-v2 ATR ratio over three training seeds (lower is a larger endpoint reduction). Horizon and quantile effects are descriptive; they do not retune the frozen gate.}",
        r"\label{tab:horizon-quantile-sensitivity}",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"& \multicolumn{4}{c}{$q=0.90$ by horizon} & \multicolumn{3}{c}{$H=8$ by quantile} \\",
        r"\cmidrule(lr){2-5}\cmidrule(lr){6-8}",
        r"Task & H1 & H2 & H4 & H8 & q80 & q90 & q95 \\",
        r"\midrule",
    ]
    for task in TASKS:
        h_values = [_fmt(by[(task, horizon, 0.90)]["endpoint_to_base_ratio"]) for horizon in audit.HORIZONS]
        q_values = [_fmt(by[(task, 8, quantile)]["endpoint_to_base_ratio"]) for quantile in audit.QUANTILES]
        lines.append(f"{task} & " + " & ".join(h_values + q_values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_calibration(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    from matplotlib.lines import Line2D

    colors = {"TwoRoom": "#4477AA", "PushT": "#EE6677", "Reacher": "#228833", "Cube": "#CCBB44"}
    markers = {"base": "o", "onset": "s", "endpoint": "^"}
    style = {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
    }
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(5.45, 4.2))
        all_values: list[float] = []
        for row in rows:
            measured = float(row["empirical_mean_R2_over_sigma2"])
            predicted = float(row["jvp_gaussian_trace_per_sequence"])
            if measured <= 0 or predicted <= 0:
                continue
            task = str(row["task"])
            checkpoint = str(row["checkpoint_type"])
            ax.scatter(
                predicted,
                measured,
                color=colors[task],
                marker=markers[checkpoint],
                s=27,
                alpha=0.72,
                edgecolors="white",
                linewidths=0.35,
            )
            all_values.extend([predicted, measured])
        lower = min(all_values)
        upper = max(all_values)
        ax.plot([lower, upper], [lower, upper], color="#222222", linewidth=1.0, linestyle="--")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Covariance-aware JVP trace per sequence")
        ax.set_ylabel(r"Measured $\mathbb{E}[R^2]/\sigma^2$")
        ax.grid(True, which="both", color="#B0B0B0", alpha=0.25, linewidth=0.55)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        task_handles = [
            Line2D([], [], marker="o", linestyle="none", markersize=5, color=color, label=task)
            for task, color in colors.items()
        ]
        checkpoint_handles = [
            Line2D(
                [], [], marker=marker, linestyle="none", markersize=5,
                markerfacecolor="#777777", markeredgecolor="#777777", label=checkpoint.capitalize()
            )
            for checkpoint, marker in markers.items()
        ]
        task_legend = ax.legend(
            handles=task_handles, title="Task", loc="upper left", ncol=2,
            fontsize=6.4, title_fontsize=6.7, frameon=False,
            handletextpad=0.35, columnspacing=0.8,
        )
        ax.add_artist(task_legend)
        ax.legend(
            handles=checkpoint_handles, title="Checkpoint", loc="lower right",
            fontsize=6.4, title_fontsize=6.7, frameon=False, handletextpad=0.35,
        )
        fig.tight_layout(pad=0.5)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=240, bbox_inches="tight")
        plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--out-json", type=Path, default=ROOT / "paper1/results/linearization_horizon_sensitivity_v1.json")
    parser.add_argument("--checkpoint-csv", type=Path, default=ROOT / "paper1/results/linearization_checkpoint_rows.csv")
    parser.add_argument("--calibration-csv", type=Path, default=ROOT / "paper1/results/linearization_calibration_rows.csv")
    parser.add_argument("--horizon-csv", type=Path, default=ROOT / "paper1/results/horizon_quantile_sensitivity_rows.csv")
    parser.add_argument("--calibration-summary", type=Path, default=ROOT / "paper1/results/linearization_calibration_summary.csv")
    parser.add_argument("--horizon-summary", type=Path, default=ROOT / "paper1/results/horizon_quantile_sensitivity_summary.csv")
    parser.add_argument("--calibration-table", type=Path, default=ROOT / "paper1/tables/table_linearization_calibration.tex")
    parser.add_argument("--horizon-table", type=Path, default=ROOT / "paper1/tables/table_horizon_quantile_sensitivity.tex")
    parser.add_argument("--figure", type=Path, default=ROOT / "assets/paper1_figs/fig_linearization_calibration.png")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = build_artifact(args.input)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(args.checkpoint_csv, payload["checkpoint_rows"])
    _write_csv(args.calibration_csv, payload["calibration_rows"])
    _write_csv(args.horizon_csv, payload["horizon_rows"])
    _write_csv(args.calibration_summary, payload["calibration_summary"])
    _write_csv(args.horizon_summary, payload["horizon_summary"])
    write_calibration_table(args.calibration_table, payload["calibration_summary"])
    write_horizon_table(args.horizon_table, payload["horizon_summary"])
    plot_calibration(args.figure, payload["calibration_rows"])
    print(
        f"wrote {args.out_json}: checkpoints={len(payload['checkpoint_rows'])} "
        f"calibration={len(payload['calibration_rows'])} horizons={len(payload['horizon_rows'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
