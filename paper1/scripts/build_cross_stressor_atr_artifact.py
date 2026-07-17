#!/usr/bin/env python3
"""Build strict strongest-only blur/resize ATR artifacts for Paper 1 E3."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

from paper1.scripts.build_external_acpc_horizon_v2_artifact import _frozen_protocol


ROOT = Path(__file__).resolve().parents[2]
TASKS = ("TwoRoom", "PushT", "Reacher", "Cube")
STD_KEYS = ("0.0", "0.08")
SOURCE_COMMIT = "c943fdf75cd71bc08e5466e1700676069728b7d2"
FROZEN_PROTOCOL_SHA256 = (
    "edcb801c3da388e673c9b55d706a558aa01da7a281fc151e52e1cda566045a21"
)
SCHEMA_VERSION = "paper1-acpc-horizon-v2-1.0"
STRESSORS = {
    "gaussian_blur": ("blur", "kernel_size"),
    "resize": ("resize", "scale_factor"),
}


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


def _load_strict(path: Path) -> dict[str, Any]:
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


def _finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite numeric, got bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} is non-finite")
    return result


def _rooted(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def _same_path(left: str | Path, right: str | Path) -> bool:
    return _rooted(left).resolve() == _rooted(right).resolve()


def _expected_draw_seeds(anchor_seed: int, num_draws: int) -> list[int]:
    return [anchor_seed + 1009 + 7919 * index for index in range(num_draws)]


def _model_contract(
    *,
    manifest: Mapping[str, Any],
    model_family: str,
    training_seed: int,
    family_id: str,
) -> tuple[str, str]:
    metadata = manifest.get("_metadata", {})
    if model_family == "LeWM":
        _require(
            metadata.get("schema_version")
            == "paper1-training-seed-eval-manifest-0.1",
            "LeWM manifest schema mismatch",
        )
        _require(
            training_seed in (3072, 3073, 3074),
            "LeWM cross-stressor seed must be 3072, 3073, or 3074",
        )
        _require(metadata.get("training_seed") == training_seed, "LeWM seed mismatch")
        _require(family_id == f"lewm_seed{training_seed}", "LeWM family id mismatch")
        split_name = "E3-L"
    else:
        _require(model_family == "PLDM", f"unsupported model family: {model_family}")
        _require(
            metadata.get("schema_version")
            == "paper1-pldm-canonical-eval-manifest-0.2",
            "PLDM manifest schema mismatch",
        )
        _require(metadata.get("status") == "complete", "PLDM manifest is incomplete")
        _require(metadata.get("model_family") == "PLDM", "PLDM manifest family mismatch")
        _require(metadata.get("training_seed") == training_seed == 3072, "PLDM seed mismatch")
        _require(
            metadata.get("training_family_id") == family_id
            == "pldm_canonical_seed3072",
            "PLDM canonical family mismatch",
        )
        split_name = "E3-P"
    _require(
        set(metadata.get("tasks", [])) == set(TASKS),
        "manifest task coverage mismatch",
    )
    _require(
        set(STD_KEYS).issubset(set(metadata.get("std_keys", []))),
        "manifest lacks base/endpoint checkpoints",
    )
    return split_name, str(metadata.get("training_seed_semantics", ""))


def _bind_checkpoint(
    *,
    row: Mapping[str, Any],
    entry: Mapping[str, Any],
    manifest_meta: Mapping[str, Any],
    model_family: str,
    training_seed: int,
    task: str,
    std_key: str,
) -> tuple[Path, str, Path | None, str | None]:
    key = f"{task}/{std_key}"
    _require(row.get("subdir") == entry.get("subdir"), f"{key}: subdir mismatch")
    _require(
        _same_path(str(row.get("run_path")), str(entry.get("path"))),
        f"{key}: manifest run path mismatch",
    )
    model_file = _rooted(str(row.get("model_file"))).resolve()
    _require(model_file.is_file(), f"{key}: model file is missing")
    if model_family == "LeWM":
        _require(
            model_file.parent == _rooted(str(entry.get("path"))).resolve(),
            f"{key}: checkpoint is outside the manifest run",
        )
        return model_file, _sha256(model_file), None, None

    manifest_model = _rooted(str(entry.get("model_file"))).resolve()
    _require(model_file == manifest_model, f"{key}: PLDM checkpoint path mismatch")
    checkpoint_sha = _sha256(model_file)
    _require(
        checkpoint_sha
        == manifest_meta.get("source_hashes", {}).get(f"checkpoint_{task}_{std_key}"),
        f"{key}: PLDM checkpoint hash mismatch",
    )
    checkpoint_rows = {
        (str(item.get("task")), str(item.get("std_key"))): item
        for item in manifest_meta.get("checkpoint_rows", [])
    }
    _require((task, std_key) in checkpoint_rows, f"{key}: checkpoint audit row missing")
    checkpoint_row = checkpoint_rows[(task, std_key)]
    _require(
        checkpoint_row.get("training_seed") == training_seed,
        f"{key}: checkpoint-local training seed mismatch",
    )
    _require(
        _same_path(str(checkpoint_row.get("model_file")), model_file),
        f"{key}: checkpoint audit path mismatch",
    )
    _require(
        checkpoint_row.get("model_sha256") == checkpoint_sha,
        f"{key}: checkpoint audit hash mismatch",
    )
    config_path = _rooted(str(checkpoint_row.get("config_path"))).resolve()
    _require(config_path.is_file(), f"{key}: checkpoint config is missing")
    config_sha = _sha256(config_path)
    _require(
        checkpoint_row.get("config_sha256") == config_sha,
        f"{key}: checkpoint config audit hash mismatch",
    )
    _require(
        manifest_meta.get("source_hashes", {}).get(f"config_{task}_{std_key}")
        == config_sha,
        f"{key}: manifest config hash mismatch",
    )
    return model_file, checkpoint_sha, config_path, config_sha


def _fixed_pool_fields(row: Mapping[str, Any], *, key: str) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for name in (
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
    ):
        fields[name] = _finite(row.get(name), name=f"{key}/{name}")
    maf_flip = row.get("maf_flip_rate")
    if maf_flip is None:
        _require(
            fields["maf_eligible_fraction"] == 0.0,
            f"{key}: missing MAF flip rate with eligible candidates",
        )
        fields["maf_flip_rate"] = None
    else:
        fields["maf_flip_rate"] = _finite(maf_flip, name=f"{key}/maf_flip_rate")
    _require(
        0.0 <= fields["maf_eligible_fraction"] <= 1.0,
        f"{key}: MAF eligible fraction outside [0,1]",
    )
    if fields["maf_flip_rate"] is not None:
        _require(
            0.0 <= fields["maf_flip_rate"] <= 1.0,
            f"{key}: MAF flip rate outside [0,1]",
        )
    return fields


def build_cross_stressor_atr_artifact(
    *,
    inputs: Sequence[Path],
    manifest_path: Path,
    protocol_path: Path,
    model_family: str,
    training_seed: int,
    family_id: str,
    corruption_type: str,
) -> dict[str, Any]:
    _require(corruption_type in STRESSORS, f"unsupported stressor: {corruption_type}")
    _require(len(inputs) == len(TASKS), "exactly four task ATR shards are required")
    protocol_original = protocol_path.read_bytes()
    protocol_mtime = protocol_path.stat().st_mtime_ns
    protocol, protocol_sha = _frozen_protocol(protocol_path)
    _require(protocol_sha == FROZEN_PROTOCOL_SHA256, "unexpected frozen protocol hash")
    manifest = _load_strict(manifest_path)
    manifest_meta = manifest.get("_metadata", {})
    split_name, training_seed_semantics = _model_contract(
        manifest=manifest,
        model_family=model_family,
        training_seed=training_seed,
        family_id=family_id,
    )
    stressor_family, severity_parameter = STRESSORS[corruption_type]
    severity_spec = protocol.get("external_severities", {}).get(stressor_family, {})
    _require(
        severity_spec.get("implementation_parameter") == severity_parameter,
        "frozen severity parameter mismatch",
    )
    severity = _finite(severity_spec.get("v1_strongest"), name="frozen severity")
    _require(
        severity in [_finite(item, name="supported severity") for item in severity_spec.get("supported_nonidentity", [])],
        "frozen strongest severity is unsupported",
    )
    sampling = protocol["diagnostic_sampling"]
    n_sequences = int(sampling["n_anchors"])
    num_draws = int(sampling["num_noise_draws"])
    anchor_seed = int(sampling["anchor_seed"])
    draw_seeds = _expected_draw_seeds(anchor_seed, num_draws)

    canonical_metric = ROOT / "tools" / "paper1_acpc_metrics.py"
    acpc_runner = ROOT / "tools" / "paper1_phase0_acpc.py"
    _require(
        _sha256(canonical_metric) == protocol["source_hashes"]["canonical_metric"],
        "canonical ATR metric differs from the frozen protocol",
    )
    calibration_atr_path = _rooted(protocol["source_paths"]["calibration_atr"])
    calibration_atr = _load_strict(calibration_atr_path)
    expected_runner_sha = calibration_atr.get("metadata", {}).get(
        "implementation_hashes", {}
    ).get("acpc_runner")
    _require(_sha256(acpc_runner) == expected_runner_sha, "ATR runner differs from CAL")

    source_paths: dict[str, str] = {
        "manifest": str(manifest_path),
        "protocol": str(protocol_path),
    }
    source_hashes: dict[str, str] = {
        "manifest": _sha256(manifest_path),
        "protocol": protocol_sha,
    }
    source_metadata: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    seen_tasks: set[str] = set()
    for path in inputs:
        _require(path.is_file(), f"missing ATR source: {path}")
        payload = _load_strict(path)
        metadata = payload.get("metadata", {})
        raw_rows = payload.get("rows", [])
        _require(metadata.get("schema_version") == "paper1-acpc-phase0-0.2", f"{path}: source schema mismatch")
        _require(metadata.get("code_commit") == SOURCE_COMMIT, f"{path}: raw commit mismatch")
        _require(metadata.get("methods") == [model_family], f"{path}: model family mismatch")
        _require(metadata.get("std_keys") == list(STD_KEYS), f"{path}: endpoint grid mismatch")
        _require(metadata.get("status_counts") == {"ok": 2}, f"{path}: row failures")
        _require(metadata.get("missing_rows") == [], f"{path}: missing rows")
        _require(metadata.get("errors") == [], f"{path}: recorded errors")
        _require(isinstance(raw_rows, list) and len(raw_rows) == 2, f"{path}: expected two rows")
        tasks = {str(row.get("task")) for row in raw_rows}
        _require(len(tasks) == 1, f"{path}: expected one task")
        task = next(iter(tasks))
        _require(task in TASKS and task not in seen_tasks, f"{path}: duplicate/unknown task")
        seen_tasks.add(task)
        _require(metadata.get("tasks") == [task], f"{path}: metadata task mismatch")
        _require(metadata.get("script_path") == "tools/paper1_phase0_acpc.py", f"{path}: ATR runner path mismatch")
        _require(metadata.get("script_sha256") == expected_runner_sha, f"{path}: ATR runner hash mismatch")
        _require(metadata.get("metric_implementation_path") == "tools/paper1_acpc_metrics.py", f"{path}: metric path mismatch")
        _require(
            metadata.get("metric_implementation_sha256")
            == protocol["source_hashes"]["canonical_metric"],
            f"{path}: canonical metric hash mismatch",
        )
        _require(
            _same_path(metadata.get("source_paths", {}).get(model_family, ""), manifest_path),
            f"{path}: raw manifest path mismatch",
        )
        _require(
            metadata.get("source_hashes", {}).get(model_family) == _sha256(manifest_path),
            f"{path}: raw manifest hash mismatch",
        )
        raw_protocol = metadata.get("protocol", {})
        for name, expected in {
            "radius_metric": protocol["radius_metric"],
            "rollout_horizon": protocol["rollout_horizon"],
            "horizon_weights": protocol["horizon_weights"],
            "atr_quantile": protocol["atr_quantile"],
            "num_noise_draws": num_draws,
            "anchor_seed": anchor_seed,
        }.items():
            _require(raw_protocol.get(name) == expected, f"{path}: raw protocol mismatch: {name}")
        source_key = f"atr_source_{task}"
        source_paths[source_key] = str(path)
        source_hashes[source_key] = _sha256(path)
        source_metadata[task] = metadata

        by_std = {str(row.get("std_key")): row for row in raw_rows}
        _require(set(by_std) == set(STD_KEYS), f"{path}: row endpoint coverage mismatch")
        for std_key in STD_KEYS:
            row = by_std[std_key]
            key = f"{task}/{std_key}"
            entry = manifest.get(task, {}).get(std_key)
            _require(isinstance(entry, Mapping), f"{key}: manifest entry missing")
            _require(row.get("status") == "ok", f"{key}: row is not ok")
            _require(row.get("method") == model_family, f"{key}: row family mismatch")
            _require(row.get("corruption_type") == corruption_type, f"{key}: stressor mismatch")
            _require(
                math.isclose(_finite(row.get("noise_std"), name=f"{key}/severity"), severity, rel_tol=0.0, abs_tol=1e-12),
                f"{key}: severity is not the frozen strongest value",
            )
            _require(row.get("corrupt_goal") is False, f"{key}: goal must remain clean")
            _require(row.get("n_sequences") == n_sequences, f"{key}: anchor count mismatch")
            _require(row.get("num_noise_draws") == num_draws, f"{key}: draw count mismatch")
            _require(row.get("rollout_horizon_actual") == protocol["rollout_horizon"], f"{key}: horizon mismatch")
            _require(row.get("embedding_space") == "normalized", f"{key}: embedding-space mismatch")
            _require(row.get("radius_metric") == protocol["radius_metric"], f"{key}: radius metric mismatch")
            _require(row.get("normalization") == "per_anchor_clean_transition_l2_q50", f"{key}: row normalization mismatch")
            _require(row.get("horizon") == protocol["rollout_horizon"], f"{key}: stored horizon mismatch")
            _require(row.get("horizon_weights") == [0.125] * 8, f"{key}: horizon weights mismatch")
            _require(row.get("atr_quantile") == protocol["atr_quantile"], f"{key}: ATR quantile mismatch")
            _require(row.get("noise_draw_seed_rule") == "seed+1009+7919*draw_index", f"{key}: draw rule mismatch")
            _require(row.get("noise_draw_seeds") == draw_seeds, f"{key}: draw seeds mismatch")
            _require(row.get("stepwise_rollout_q90_is_atr") is False, f"{key}: legacy metric is mislabeled")
            _require("acpc_h_l2_p90" not in row, f"{key}: ambiguous old ATR field is present")
            clean_scales = [_finite(value, name=f"{key}/clean scale") for value in row.get("clean_transition_scale", [])]
            _require(len(clean_scales) == n_sequences, f"{key}: clean-scale coverage mismatch")
            _require(
                row.get("fixed_pool_candidate_seed")
                == anchor_seed + 2027,
                f"{key}: fixed-pool seed mismatch",
            )
            _require(
                row.get("fixed_pool_noise_draw_aggregation") == "arithmetic_mean",
                f"{key}: fixed-pool aggregation mismatch",
            )
            model_file, checkpoint_sha, config_path, config_sha = _bind_checkpoint(
                row=row,
                entry=entry,
                manifest_meta=manifest_meta,
                model_family=model_family,
                training_seed=training_seed,
                task=task,
                std_key=std_key,
            )
            checkpoint_key = f"checkpoint_{task}_{std_key}"
            source_paths[checkpoint_key] = str(model_file)
            source_hashes[checkpoint_key] = checkpoint_sha
            if config_path is not None and config_sha is not None:
                config_key = f"config_{task}_{std_key}"
                source_paths[config_key] = str(config_path)
                source_hashes[config_key] = config_sha
            rows.append(
                {
                    "status": "ok",
                    "model_family": model_family,
                    "training_family_id": family_id,
                    "training_seed": training_seed,
                    "task": task,
                    "std_key": std_key,
                    "training_rho": float(std_key),
                    "stressor_family": stressor_family,
                    "corruption_type": corruption_type,
                    "severity_parameter": severity_parameter,
                    "severity": severity,
                    "atr_horizon_v2_q90": _finite(row.get("atr_horizon_v2_q90"), name=f"{key}/ATR"),
                    "radius_metric": row["radius_metric"],
                    "rollout_horizon_actual": int(row["rollout_horizon_actual"]),
                    "n_sequences": int(row["n_sequences"]),
                    "num_noise_draws": int(row["num_noise_draws"]),
                    "anchor_seed": anchor_seed,
                    "corrupt_goal": False,
                    "clean_transition_scale_min": min(clean_scales),
                    "clean_transition_scale_zero_count": sum(value == 0.0 for value in clean_scales),
                    "fixed_pool_candidate_seed": int(row["fixed_pool_candidate_seed"]),
                    "fixed_pool_noise_draw_aggregation": row["fixed_pool_noise_draw_aggregation"],
                    **_fixed_pool_fields(row, key=key),
                    "model_file": str(model_file),
                    "checkpoint_sha256": checkpoint_sha,
                    "split_name": split_name,
                }
            )

    _require(seen_tasks == set(TASKS), "ATR task coverage mismatch")
    rows.sort(key=lambda row: (TASKS.index(str(row["task"])), float(row["training_rho"])))
    _require(len(rows) == len(TASKS) * len(STD_KEYS), "ATR row coverage mismatch")
    _require(set(source_paths) == set(source_hashes), "source path/hash keys differ")
    script_path = Path(__file__).resolve()
    artifact = {
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "code_commit": _git_commit(),
            "raw_source_commit": SOURCE_COMMIT,
            "script_path": str(script_path.relative_to(ROOT)),
            "script_sha256": _sha256(script_path),
            "source_paths": source_paths,
            "source_hashes": source_hashes,
            "source_metadata": source_metadata,
            "implementation_paths": {
                "canonical_metric": str(canonical_metric.relative_to(ROOT)),
                "acpc_runner": str(acpc_runner.relative_to(ROOT)),
            },
            "implementation_hashes": {
                "canonical_metric": _sha256(canonical_metric),
                "acpc_runner": _sha256(acpc_runner),
            },
            "protocol_hash": protocol_sha,
            "protocol_sha256": protocol_sha,
            "protocol_hash_status": "E3_cross_stressor_bound_to_immutable_protocol",
            "model_family": model_family,
            "training_family_id": family_id,
            "training_seed": training_seed,
            "training_seed_semantics": training_seed_semantics,
            "split_name": split_name,
            "artifact_role": "cross_stressor_external_atr",
            "behavior_blind": True,
            "threshold_search_allowed": False,
            "stressor_family": stressor_family,
            "corruption_type": corruption_type,
            "severity_parameter": severity_parameter,
            "severity": severity,
            "status": "complete",
            "status_counts": {"ok": len(rows)},
            "missing_rows": [],
            "errors": [],
            "protocol": {
                "radius_metric": protocol["radius_metric"],
                "rollout_horizon": protocol["rollout_horizon"],
                "horizon_weights": protocol["horizon_weights"],
                "atr_quantile": protocol["atr_quantile"],
                "normalization": protocol["normalization"],
                "noise_draw_aggregation": protocol["noise_draw_aggregation"],
                "n_sequences": n_sequences,
                "num_noise_draws": num_draws,
                "anchor_seed": anchor_seed,
                "corrupt_goal": False,
                "corruption_type": corruption_type,
                "severity_parameter": severity_parameter,
                "severity": severity,
            },
        },
        "rows": rows,
    }
    _require(protocol_path.read_bytes() == protocol_original, "builder changed protocol bytes")
    _require(protocol_path.stat().st_mtime_ns == protocol_mtime, "builder changed protocol mtime")
    return artifact


def _write_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--model-family", choices=("LeWM", "PLDM"), required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--family-id", required=True)
    parser.add_argument("--corruption-type", choices=tuple(STRESSORS), required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.out.resolve() == args.protocol.resolve():
        raise ValueError("ATR output cannot be the frozen protocol path")
    artifact = build_cross_stressor_atr_artifact(
        inputs=args.input,
        manifest_path=args.manifest,
        protocol_path=args.protocol,
        model_family=args.model_family,
        training_seed=args.training_seed,
        family_id=args.family_id,
        corruption_type=args.corruption_type,
    )
    _write_exclusive(args.out, artifact)
    print(f"wrote {args.out} ({len(artifact['rows'])} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
