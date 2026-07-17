#!/usr/bin/env python3
"""Validate and summarize the Paper 1 remediation Phase-1 smoke runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "paper1-remediation-phase1-smoke-1.0"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def _load_strict_json(path: Path) -> dict[str, Any]:
    payload = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_constant,
    )
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: top-level JSON value must be an object")
    return payload


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite(row: Mapping[str, Any], keys: Sequence[str], *, source: Path) -> None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, bool):
            raise ValueError(f"{source}: {key} must be finite numeric, got bool")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{source}: {key} must be finite numeric") from exc
        if not math.isfinite(numeric):
            raise ValueError(f"{source}: {key} is non-finite")


def _uniform_h8(weights: Any) -> bool:
    if not isinstance(weights, list) or len(weights) != 8:
        return False
    return all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isclose(float(value), 0.125, rel_tol=0.0, abs_tol=1e-8)
        for value in weights
    )


def _acpc_checkpoint_type(std_key: Any) -> str:
    return "base" if math.isclose(float(std_key), 0.0, abs_tol=1e-12) else "endpoint"


def _validate_acpc(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = _load_strict_json(path)
    metadata = payload.get("metadata", {})
    _require(
        metadata.get("schema_version") == "paper1-acpc-phase0-0.2",
        f"{path}: expected ACPC schema 0.2",
    )
    _require(
        metadata.get("protocol", {}).get("radius_metric")
        == "horizon_weighted_stacked_l2_v2",
        f"{path}: missing canonical radius metric metadata",
    )
    _require(
        metadata.get("protocol", {}).get("num_noise_draws") == 2,
        f"{path}: smoke must use two noise draws",
    )
    rows = payload.get("rows", [])
    _require(isinstance(rows, list) and len(rows) == 2, f"{path}: expected two rows")

    selected: list[dict[str, Any]] = []
    for row in rows:
        _require(row.get("status") == "ok", f"{path}: non-ok ACPC row")
        _require(row.get("method") == "PLDM", f"{path}: expected PLDM row")
        _require(row.get("n_sequences") == 16, f"{path}: expected 16 anchors")
        _require(row.get("num_noise_draws") == 2, f"{path}: expected two draws")
        _require(
            row.get("radius_metric") == "horizon_weighted_stacked_l2_v2",
            f"{path}: row is not canonical horizon-v2",
        )
        _require(_uniform_h8(row.get("horizon_weights")), f"{path}: non-uniform H8")
        _require(
            row.get("noise_draw_aggregation")
            == "per_anchor_mean_then_checkpoint_quantile",
            f"{path}: wrong draw aggregation order",
        )
        _require(
            row.get("stepwise_rollout_q90_is_atr") is False,
            f"{path}: legacy stepwise field is mislabeled",
        )
        _require("acpc_h_l2_p90" not in row, f"{path}: ambiguous legacy field present")
        _require(row.get("jvp_time") is None, f"{path}: ACPC row jvp_time must be null")
        _finite(
            row,
            (
                "atr_horizon_v2_q90",
                "horizon_radius_v2_unnormalized_q90",
                "stepwise_rollout_q90",
                "model_load_time",
                "data_io_time",
                "prediction_time",
                "fixed_pool_time",
                "wall_time_per_row",
            ),
            source=path,
        )
        selected.append(
            {
                "audit": "paired_rollout_radius",
                "model_family": "PLDM",
                "task": row["task"],
                "training_seed": None,
                "checkpoint_type": _acpc_checkpoint_type(row["std_key"]),
                "std_key": row["std_key"],
                "status": "ok",
                "n_sequences": row["n_sequences"],
                "num_noise_draws": row["num_noise_draws"],
                "noise_draw_seeds": row["noise_draw_seeds"],
                "atr_horizon_v2_q90": row["atr_horizon_v2_q90"],
                "horizon_radius_v2_unnormalized_q90": row[
                    "horizon_radius_v2_unnormalized_q90"
                ],
                "stepwise_rollout_q90_compatibility_only": row[
                    "stepwise_rollout_q90"
                ],
                "model_load_time": row["model_load_time"],
                "data_io_time": row["data_io_time"],
                "prediction_time": row["prediction_time"],
                "fixed_pool_time": row["fixed_pool_time"],
                "jvp_time": None,
                "wall_time_per_row": row["wall_time_per_row"],
                "peak_gpu_memory": row["peak_gpu_memory"],
                "peak_gpu_memory_unit": row["peak_gpu_memory_unit"],
            }
        )
    _require(
        {_acpc_checkpoint_type(row["std_key"]) for row in rows}
        == {"base", "endpoint"},
        f"{path}: expected base and endpoint",
    )
    return metadata, selected


def _validate_jvp(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = _load_strict_json(path)
    metadata = payload.get("metadata", {})
    _require(
        metadata.get("schema_version") == "paper1-jvp-hutchinson-sensitivity-0.2",
        f"{path}: expected JVP schema 0.2",
    )
    _require(metadata.get("status") == "complete", f"{path}: incomplete JVP run")
    _require(
        metadata.get("map_contract", {}).get("radius_metric")
        == "horizon_weighted_stacked_l2_v2",
        f"{path}: JVP map is not aligned to horizon-v2",
    )
    rows = payload.get("rows", [])
    _require(isinstance(rows, list) and len(rows) == 3, f"{path}: expected three rows")

    selected: list[dict[str, Any]] = []
    for row in rows:
        _require(row.get("status") == "ok", f"{path}: non-ok JVP row")
        _require(row.get("model_family") == "LeWM", f"{path}: expected LeWM row")
        _require(row.get("training_seed") == 3072, f"{path}: expected seed3072")
        _require(row.get("n_sequences") == 16, f"{path}: expected 16 anchors")
        _require(row.get("hutchinson_probes") == 2, f"{path}: expected two probes")
        _require(_uniform_h8(row.get("horizon_weights")), f"{path}: non-uniform H8")
        _require(
            row.get("alignment_coefficient_deprecated_alias_of")
            == "kappa_relative_isotropic",
            f"{path}: deprecated alias is not explicit",
        )
        _finite(
            row,
            (
                "encoder_trace",
                "rollout_trace",
                "composed_trace",
                "kappa_submultiplicative",
                "kappa_relative_isotropic",
                "alignment_coefficient",
                "model_load_time",
                "data_io_time",
                "jvp_time",
                "wall_time_per_row",
            ),
            source=path,
        )
        _require(
            math.isclose(
                float(row["kappa_relative_isotropic"]),
                float(row["latent_input_dim"])
                * float(row["kappa_submultiplicative"]),
                rel_tol=1e-10,
                abs_tol=1e-10,
            ),
            f"{path}: kappa_rel != d_z * kappa_sub",
        )
        _require(
            math.isclose(
                float(row["alignment_coefficient"]),
                float(row["kappa_relative_isotropic"]),
                rel_tol=0.0,
                abs_tol=0.0,
            ),
            f"{path}: compatibility alias differs from canonical kappa",
        )
        if row.get("checkpoint_type") in {"base", "endpoint"}:
            selected.append(
                {
                    "audit": "jvp_hutchinson",
                    "model_family": "LeWM",
                    "task": row["task"],
                    "training_seed": row["training_seed"],
                    "checkpoint_type": row["checkpoint_type"],
                    "std_key": row["std_key"],
                    "status": "ok",
                    "n_sequences": row["n_sequences"],
                    "hutchinson_probes": row["hutchinson_probes"],
                    "encoder_trace": row["encoder_trace"],
                    "rollout_trace": row["rollout_trace"],
                    "composed_trace": row["composed_trace"],
                    "kappa_submultiplicative": row["kappa_submultiplicative"],
                    "kappa_relative_isotropic": row["kappa_relative_isotropic"],
                    "model_load_time": row["model_load_time"],
                    "data_io_time": row["data_io_time"],
                    "prediction_time": None,
                    "fixed_pool_time": None,
                    "jvp_time": row["jvp_time"],
                    "wall_time_per_row": row["wall_time_per_row"],
                    "peak_gpu_memory": row["peak_gpu_memory"],
                    "peak_gpu_memory_unit": row["peak_gpu_memory_unit"],
                }
            )
    _require(
        {row["checkpoint_type"] for row in selected} == {"base", "endpoint"},
        f"{path}: expected base and endpoint JVP rows",
    )
    return metadata, selected


def build_report(
    *,
    acpc_paths: Sequence[Path],
    jvp_paths: Sequence[Path],
) -> dict[str, Any]:
    _require(len(acpc_paths) == 2, "exactly two ACPC smoke artifacts are required")
    _require(len(jvp_paths) == 2, "exactly two JVP smoke artifacts are required")
    for path in (*acpc_paths, *jvp_paths):
        _require(path.is_file(), f"missing smoke artifact: {path}")

    benchmark_rows: list[dict[str, Any]] = []
    source_metadata: dict[str, Any] = {}
    source_paths: dict[str, Path] = {}
    for index, path in enumerate(acpc_paths):
        metadata, selected = _validate_acpc(path)
        key = f"acpc_{index}"
        source_paths[key] = path
        source_metadata[key] = metadata
        benchmark_rows.extend(selected)
    for index, path in enumerate(jvp_paths):
        metadata, selected = _validate_jvp(path)
        key = f"jvp_{index}"
        source_paths[key] = path
        source_metadata[key] = metadata
        benchmark_rows.extend(selected)

    acpc_rows = [row for row in benchmark_rows if row["audit"] == "paired_rollout_radius"]
    jvp_rows = [row for row in benchmark_rows if row["audit"] == "jvp_hutchinson"]
    tasks = sorted({row["task"] for row in benchmark_rows})
    checks = {
        "two_tasks": tasks == ["PushT", "TwoRoom"],
        "two_checkpoints_per_task_per_audit": (
            len(acpc_rows) == 4 and len(jvp_rows) == 4
        ),
        "sixteen_anchors": all(row["n_sequences"] == 16 for row in benchmark_rows),
        "two_acpc_noise_draws": all(
            row["num_noise_draws"] == 2 for row in acpc_rows
        ),
        "canonical_horizon_v2_radius": True,
        "legacy_stepwise_not_atr": True,
        "jvp_map_aligned": True,
        "both_kappa_definitions_finite": True,
        "strict_json_inputs": True,
        "profiling_complete": all(
            row["wall_time_per_row"] is not None
            and row["peak_gpu_memory"] is not None
            for row in benchmark_rows
        ),
    }
    _require(all(checks.values()), "one or more Gate 1 checks failed")

    script_path = Path(__file__).resolve()
    implementation_paths = {
        "canonical_metric": ROOT / "tools" / "paper1_acpc_metrics.py",
        "acpc_runner": ROOT / "tools" / "paper1_phase0_acpc.py",
        "jvp_runner": ROOT / "tools" / "paper1_jvp_hutchinson_sensitivity_audit.py",
    }
    created_utc = datetime.now(timezone.utc).isoformat()
    return {
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "created_utc": created_utc,
            "code_commit": _git_commit(),
            "script_path": str(script_path.relative_to(ROOT)),
            "script_sha256": _sha256(script_path),
            "source_paths": {
                name: str(path)
                for name, path in source_paths.items()
            },
            "source_hashes": {
                name: _sha256(path)
                for name, path in source_paths.items()
            },
            "source_metadata": source_metadata,
            "implementation_paths": {
                name: str(path.relative_to(ROOT))
                for name, path in implementation_paths.items()
            },
            "implementation_hashes": {
                name: _sha256(path)
                for name, path in implementation_paths.items()
            },
            "protocol_hash": None,
            "protocol_hash_status": "not_frozen_phase1_correctness_audit",
            "model_family": ["LeWM", "PLDM"],
            "training_seed_semantics": (
                "LeWM JVP uses independent training seed3072; PLDM canonical "
                "checkpoint selection comes from its source manifest"
            ),
            "evaluation_seed_semantics": (
                "not a training seed; no closed-loop evaluation seeds are pooled "
                "into this checkpoint-local smoke audit"
            ),
            "status": "pass",
            "missing_rows": [],
            "errors": [],
        },
        "gate": "Gate 1",
        "gate_status": "pass",
        "checks": checks,
        "tasks": tasks,
        "selected_protocol": {
            "checkpoint_types": ["base", "endpoint"],
            "n_sequences": 16,
            "acpc_noise_draws": 2,
            "jvp_hutchinson_probes": 2,
            "rollout_horizon": 8,
            "horizon_weights": "uniform_1_over_H",
            "radius_metric": "horizon_weighted_stacked_l2_v2",
        },
        "benchmark_rows": benchmark_rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acpc", type=Path, action="append", required=True)
    parser.add_argument("--jvp", type=Path, action="append", required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "paper1" / "results" / "remediation_phase1_smoke_v2.json",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = build_report(acpc_paths=args.acpc, jvp_paths=args.jvp)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
