#!/usr/bin/env python3
"""Merge strict task-sharded E4 target-view ATR or SMPR diagnostics."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
TASKS = ("TwoRoom", "PushT", "Reacher", "Cube")
RHO_KEYS = ("0.01", "0.02", "0.03", "0.04", "0.05", "0.06", "0.07", "0.08")
BRANCHES = ("full_sequence", "target_view")
FROZEN_PROTOCOL_SHA256 = (
    "edcb801c3da388e673c9b55d706a558aa01da7a281fc151e52e1cda566045a21"
)
FIXED_POOL_FIELDS = (
    "pcc_abs_median",
    "pcc_abs_p90",
    "cra_spearman_mean",
    "cra_spearman_median",
    "elite_overlap_topk",
    "elite_overlap_mean",
    "margin_delta",
    "margin_clean_q50",
    "margin_clean_q90",
    "maf_eligible_fraction",
    "maf_flip_rate",
)
SMPR_DIAGNOSTIC_FIELDS = (
    "anchor_seed",
    "atr_reference_abs_error",
    "atr_reference_match",
    "corrupt_goal",
    "corruption_type",
    "margin_delta_norm",
    "n_sequences",
    "noise_draw_seeds",
    "noise_draw_seed_rule",
    "num_noise_draws",
    "pair_rule",
    "radius_metric",
    "radius_quantile",
    "reference_atr_horizon_v2_q90",
    "rollout_horizon_actual",
    "same_state_tube_radius",
    "semantic_distance_threshold",
    "semantic_label_count",
    "semantic_label_rule",
    "semantic_pair_count",
    "semantic_skip_rate",
    "semantic_skipped_anchor_count",
    "smpr",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_constant,
    )
    _require(isinstance(payload, dict), f"{path}: top-level JSON must be an object")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name}: bool is not numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}: expected finite numeric") from exc
    _require(math.isfinite(result), f"{name}: value is not finite")
    return result


def _same_path(left: Any, right: Any) -> bool:
    return Path(str(left)).expanduser().resolve() == Path(str(right)).expanduser().resolve()


def _manifest_index(
    path: Path,
    *,
    branch: str,
) -> tuple[dict[tuple[str, str], Mapping[str, Any]], Mapping[str, Any]]:
    payload = _load(path)
    metadata = payload.get("_metadata", {})
    _require(
        metadata.get("schema_version")
        == "paper1-target-view-diagnostic-manifest-1.0",
        "target-view manifest schema mismatch",
    )
    _require(metadata.get("status") == "ok", "target-view manifest is not complete")
    _require(metadata.get("actual_rows") == 64, "target-view manifest row count mismatch")
    _require(
        metadata.get("actual_matched_pairs") == 32,
        "target-view matched-pair count mismatch",
    )
    rows = [
        row
        for row in payload.get("rows", [])
        if row.get("branch") == branch
    ]
    index = {
        (str(row.get("task")), str(row.get("std_key"))): row
        for row in rows
    }
    expected = {(task, rho) for task in TASKS for rho in RHO_KEYS}
    _require(set(index) == expected, f"{branch}: target-view manifest coverage mismatch")
    return index, metadata


def _validate_checkpoint(
    raw_row: Mapping[str, Any],
    manifest_row: Mapping[str, Any],
    *,
    name: str,
) -> tuple[str, str]:
    checkpoint = manifest_row.get("checkpoint", {})
    model_file = Path(str(raw_row.get("model_file"))).expanduser().resolve()
    expected_path = Path(str(checkpoint.get("path"))).expanduser().resolve()
    _require(model_file == expected_path, f"{name}: checkpoint path mismatch")
    _require(model_file.is_file(), f"{name}: checkpoint missing")
    checkpoint_hash = _sha256(model_file)
    _require(checkpoint_hash == checkpoint.get("sha256"), f"{name}: checkpoint hash changed")
    _require(
        _same_path(raw_row.get("run_path"), manifest_row.get("path")),
        f"{name}: run path mismatch",
    )
    _require(raw_row.get("subdir") == manifest_row.get("subdir"), f"{name}: subdir mismatch")
    return str(model_file), checkpoint_hash


def _raw_task_payloads(
    inputs: Sequence[Path],
    *,
    expected_schema: str,
    family_id: str | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Mapping[str, Any]]]:
    _require(len(inputs) == 4, "exactly four task shards are required")
    payloads: dict[str, dict[str, Any]] = {}
    metadata_by_task: dict[str, Mapping[str, Any]] = {}
    for path in inputs:
        _require(path.is_file(), f"missing task shard: {path}")
        payload = _load(path)
        metadata = payload.get("metadata", {})
        _require(metadata.get("schema_version") == expected_schema, f"{path}: schema mismatch")
        if expected_schema == "paper1-acpc-phase0-0.2":
            _require(
                metadata.get("status") in (None, "complete"),
                f"{path}: unexpected Phase-0 status",
            )
        else:
            _require(metadata.get("status") == "complete", f"{path}: shard incomplete")
        _require(metadata.get("status_counts") == {"ok": 8}, f"{path}: shard row failure")
        _require(metadata.get("missing_rows") == [], f"{path}: shard missing rows")
        _require(metadata.get("errors") == [], f"{path}: shard errors")
        _require(metadata.get("training_seed", 3072) == 3072, f"{path}: seed mismatch")
        if family_id is not None:
            _require(metadata.get("training_family_id") == family_id, f"{path}: family id mismatch")
        rows = payload.get("rows", [])
        _require(isinstance(rows, list) and len(rows) == 8, f"{path}: expected eight rows")
        tasks = {str(row.get("task")) for row in rows}
        _require(len(tasks) == 1, f"{path}: expected one task")
        task = next(iter(tasks))
        _require(task in TASKS and task not in payloads, f"{path}: duplicate/unknown task")
        by_rho = {str(row.get("std_key")): row for row in rows}
        _require(set(by_rho) == set(RHO_KEYS), f"{path}: rho coverage mismatch")
        payloads[task] = {"path": path, "rows": by_rho}
        metadata_by_task[task] = metadata
    _require(set(payloads) == set(TASKS), "task shard coverage mismatch")
    return payloads, metadata_by_task


def _base_metadata(
    *,
    kind: str,
    branch: str,
    manifest_path: Path,
    manifest_metadata: Mapping[str, Any],
    protocol_path: Path,
    inputs: Sequence[Path],
    source_metadata: Mapping[str, Any],
    family_id: str,
    rows: Sequence[Mapping[str, Any]],
    extra_sources: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    script_path = Path(__file__).resolve()
    source_paths: dict[str, Path] = {
        "target_view_manifest": manifest_path,
        "protocol": protocol_path,
        **{f"raw_{task}": next(path for path in inputs if task.lower() in path.name.lower()) for task in TASKS},
    }
    if extra_sources:
        source_paths.update(extra_sources)
    implementation_paths = {
        "builder": script_path,
        "canonical_metric": ROOT / "tools/paper1_acpc_metrics.py",
        "atr_runner": ROOT / "tools/paper1_phase0_acpc.py",
        "smpr_runner": ROOT / "paper1/scripts/smpr_sensitivity.py",
        "semantic_margin": ROOT / "tools/paper1_semantic_margin.py",
    }
    return {
        "schema_version": (
            "paper1-acpc-horizon-v2-1.0"
            if kind == "atr"
            else "paper1-smpr-v2-merged-1.0"
        ),
        "artifact_role": (
            "target_view_external_atr"
            if kind == "atr"
            else "target_view_external_smpr"
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": _git_commit(),
        "status": "complete",
        "status_counts": {"ok": len(rows)},
        "missing_rows": [],
        "errors": [],
        "behavior_blind": True,
        "threshold_search_allowed": False,
        "protocol_hash": FROZEN_PROTOCOL_SHA256,
        "protocol_sha256": FROZEN_PROTOCOL_SHA256,
        "protocol_hash_status": "E4_target_view_bound_to_immutable_protocol",
        "model_family": "LeWM",
        "training_seed": 3072,
        "training_seed_semantics": "one existing LeWM training run per task/std/branch",
        "training_family_id": family_id,
        "branch": branch,
        "split_name": "E4",
        "tasks": list(TASKS),
        "std_keys": list(RHO_KEYS),
        "evaluation_seeds": [42, 43, 44],
        "evaluation_seed_semantics": "repeated closed-loop evaluation of a fixed checkpoint",
        "target_view_manifest_builder_sha256": manifest_metadata.get("builder", {}).get("sha256"),
        "source_paths": {name: str(path) for name, path in source_paths.items()},
        "source_hashes": {name: _sha256(path) for name, path in source_paths.items()},
        "source_metadata": source_metadata,
        "implementation_paths": {
            name: str(path.relative_to(ROOT))
            for name, path in implementation_paths.items()
        },
        "implementation_hashes": {
            name: _sha256(path)
            for name, path in implementation_paths.items()
        },
        "protocol": {
            "radius_metric": "horizon_weighted_stacked_l2_v2",
            "rollout_horizon": 8,
            "horizon_weights": "uniform_1_over_H",
            "radius_quantile": 0.9,
            "normalization": (
                "per_anchor_observed_clean_transition_l2_q50_"
                "including_history_future_boundary"
            ),
            "noise_draw_aggregation": (
                "per_anchor_mean_then_checkpoint_quantile"
                if kind == "atr"
                else "per_anchor_mean_then_checkpoint_radius_quantile"
            ),
            "n_sequences": 100,
            "num_noise_draws": 5,
            "anchor_seed": 9101,
            "corruption": "observation_only_gaussian_std0.08_clean_goal",
        },
    }


def build_atr(
    *,
    inputs: Sequence[Path],
    manifest_path: Path,
    protocol_path: Path,
    branch: str,
) -> dict[str, Any]:
    manifest, manifest_metadata = _manifest_index(manifest_path, branch=branch)
    protocol_hash = _sha256(protocol_path)
    _require(protocol_hash == FROZEN_PROTOCOL_SHA256, "frozen protocol hash mismatch")
    family_id = f"lewm_seed3072_{branch}"
    shards, source_metadata = _raw_task_payloads(
        inputs,
        expected_schema="paper1-acpc-phase0-0.2",
    )
    rows: list[dict[str, Any]] = []
    for task in TASKS:
        metadata = source_metadata[task]
        _require(metadata.get("methods") == ["LeWM"], f"{task}: method mismatch")
        _require(metadata.get("std_keys") == list(RHO_KEYS), f"{task}: rho metadata mismatch")
        _require(metadata.get("protocol", {}).get("rollout_horizon") == 8, f"{task}: H mismatch")
        _require(metadata.get("protocol", {}).get("num_noise_draws") == 5, f"{task}: draw mismatch")
        for rho in RHO_KEYS:
            name = f"E4/{branch}/{task}/{rho}"
            raw = shards[task]["rows"][rho]
            _require(raw.get("status") == "ok", f"{name}: row not ok")
            _require(raw.get("method") == "LeWM", f"{name}: model family mismatch")
            _require(raw.get("corruption_type") == "gaussian_noise", f"{name}: corruption mismatch")
            _require(raw.get("corrupt_goal") is False, f"{name}: goal corruption mismatch")
            _require(_finite(raw.get("noise_std"), name=f"{name}/noise") == 0.08, f"{name}: noise mismatch")
            _require(raw.get("radius_metric") == "horizon_weighted_stacked_l2_v2", f"{name}: radius mismatch")
            _require(raw.get("rollout_horizon_actual") == 8, f"{name}: actual H mismatch")
            _require(raw.get("num_noise_draws") == 5, f"{name}: draw count mismatch")
            model_file, checkpoint_hash = _validate_checkpoint(
                raw,
                manifest[(task, rho)],
                name=name,
            )
            clean_scales = [
                _finite(value, name=f"{name}/clean-scale")
                for value in raw.get("clean_transition_scale", [])
            ]
            _require(
                len(clean_scales) == 100,
                f"{name}: expected one clean scale per anchor",
            )
            fixed_pool: dict[str, Any] = {}
            for field in FIXED_POOL_FIELDS:
                value = raw.get(field)
                fixed_pool[field] = (
                    None
                    if value is None
                    else _finite(value, name=f"{name}/{field}")
                )
            rows.append(
                {
                    "status": "ok",
                    "model_family": "LeWM",
                    "training_family_id": family_id,
                    "training_seed": 3072,
                    "task": task,
                    "std_key": rho,
                    "training_rho": float(rho),
                    "branch": branch,
                    "split_name": "E4",
                    "model_file": model_file,
                    "checkpoint_sha256": checkpoint_hash,
                    "atr_horizon_v2_q90": _finite(
                        raw.get("atr_horizon_v2_q90"),
                        name=f"{name}/ATR",
                    ),
                    "clean_transition_scale_min": min(clean_scales),
                    "clean_transition_scale_zero_count": sum(
                        value == 0.0 for value in clean_scales
                    ),
                    "radius_metric": "horizon_weighted_stacked_l2_v2",
                    "rollout_horizon_actual": 8,
                    "n_sequences": 100,
                    "num_noise_draws": 5,
                    "anchor_seed": 9101,
                    **fixed_pool,
                }
            )
    metadata = _base_metadata(
        kind="atr",
        branch=branch,
        manifest_path=manifest_path,
        manifest_metadata=manifest_metadata,
        protocol_path=protocol_path,
        inputs=inputs,
        source_metadata=source_metadata,
        family_id=family_id,
        rows=rows,
    )
    return {"metadata": metadata, "rows": rows}


def build_smpr(
    *,
    inputs: Sequence[Path],
    manifest_path: Path,
    protocol_path: Path,
    reference_atr_path: Path,
    branch: str,
) -> dict[str, Any]:
    manifest, manifest_metadata = _manifest_index(manifest_path, branch=branch)
    _require(_sha256(protocol_path) == FROZEN_PROTOCOL_SHA256, "frozen protocol hash mismatch")
    family_id = f"lewm_seed3072_{branch}"
    reference = _load(reference_atr_path)
    reference_meta = reference.get("metadata", {})
    _require(reference_meta.get("artifact_role") == "target_view_external_atr", "wrong ATR role")
    _require(reference_meta.get("branch") == branch, "ATR branch mismatch")
    _require(reference_meta.get("protocol_hash") == FROZEN_PROTOCOL_SHA256, "ATR protocol mismatch")
    reference_rows = {
        (str(row.get("task")), str(row.get("std_key"))): row
        for row in reference.get("rows", [])
    }
    expected = {(task, rho) for task in TASKS for rho in RHO_KEYS}
    _require(set(reference_rows) == expected, "reference ATR coverage mismatch")
    shards, source_metadata = _raw_task_payloads(
        inputs,
        expected_schema="paper1-smpr-v2-1.0",
        family_id=family_id,
    )
    rows: list[dict[str, Any]] = []
    for task in TASKS:
        for rho in RHO_KEYS:
            name = f"E4/{branch}/{task}/{rho}"
            raw = shards[task]["rows"][rho]
            _require(raw.get("status") == "ok", f"{name}: row not ok")
            _require(raw.get("model_family") == "LeWM", f"{name}: family mismatch")
            _require(raw.get("training_seed") == 3072, f"{name}: seed mismatch")
            _require(raw.get("atr_reference_match") is True, f"{name}: ATR mismatch")
            ref_atr = _finite(
                reference_rows[(task, rho)].get("atr_horizon_v2_q90"),
                name=f"{name}/reference",
            )
            raw_ref = _finite(
                raw.get("reference_atr_horizon_v2_q90"),
                name=f"{name}/raw-reference",
            )
            _require(math.isclose(ref_atr, raw_ref, rel_tol=1e-6, abs_tol=1e-6), f"{name}: ATR differs")
            model_file, checkpoint_hash = _validate_checkpoint(
                raw,
                manifest[(task, rho)],
                name=name,
            )
            row = {
                field: raw.get(field)
                for field in SMPR_DIAGNOSTIC_FIELDS
            }
            row.update(
                {
                    "status": "ok",
                    "model_family": "LeWM",
                    "training_family_id": family_id,
                    "training_seed": 3072,
                    "task": task,
                    "std_key": rho,
                    "training_rho": float(rho),
                    "branch": branch,
                    "split_name": "E4",
                    "model_file": model_file,
                    "checkpoint_sha256": checkpoint_hash,
                    "smpr": _finite(raw.get("smpr"), name=f"{name}/SMPR"),
                }
            )
            rows.append(row)
    metadata = _base_metadata(
        kind="smpr",
        branch=branch,
        manifest_path=manifest_path,
        manifest_metadata=manifest_metadata,
        protocol_path=protocol_path,
        inputs=inputs,
        source_metadata=source_metadata,
        family_id=family_id,
        rows=rows,
        extra_sources={"reference_atr": reference_atr_path},
    )
    metadata["reference_atr_metadata"] = reference_meta
    metadata["protocol"]["pair_rule"] = "task_grounded_near_boundary_v2"
    metadata["protocol"]["margin_delta_normalized"] = 0.1
    metadata["protocol"]["local_state_quantile"] = 0.35
    return {"metadata": metadata, "rows": rows}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("atr", "smpr"), required=True)
    parser.add_argument("--branch", choices=BRANCHES, required=True)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "assets/paper1_data/target_view_diagnostic_manifest_v1.json",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "paper1/config/frozen_diagnostic_protocol_v1.json",
    )
    parser.add_argument("--reference-atr", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.kind == "atr":
        _require(args.reference_atr is None, "--reference-atr is only valid for SMPR")
        payload = build_atr(
            inputs=args.input,
            manifest_path=args.manifest,
            protocol_path=args.protocol,
            branch=args.branch,
        )
    else:
        _require(args.reference_atr is not None, "SMPR requires --reference-atr")
        payload = build_smpr(
            inputs=args.input,
            manifest_path=args.manifest,
            protocol_path=args.protocol,
            reference_atr_path=args.reference_atr,
            branch=args.branch,
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.out} ({len(payload['rows'])} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
