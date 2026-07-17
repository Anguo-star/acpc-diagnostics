#!/usr/bin/env python3
"""Build strict strongest-only blur/resize SMPR artifacts for Paper 1 E3."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from paper1.scripts.build_cross_stressor_atr_artifact import (
    FROZEN_PROTOCOL_SHA256,
    ROOT,
    SOURCE_COMMIT,
    STD_KEYS,
    STRESSORS,
    TASKS,
    _bind_checkpoint,
    _expected_draw_seeds,
    _finite,
    _frozen_protocol,
    _git_commit,
    _load_strict,
    _model_contract,
    _require,
    _rooted,
    _same_path,
    _sha256,
)


SCHEMA_VERSION = "paper1-smpr-v2-merged-1.0"
RAW_PROTOCOL_PLACEHOLDER = "observation_only_gaussian_std0.08_clean_goal"


def build_cross_stressor_smpr_artifact(
    *,
    inputs: Sequence[Path],
    reference_atr_path: Path,
    manifest_path: Path,
    protocol_path: Path,
    model_family: str,
    training_seed: int,
    family_id: str,
    corruption_type: str,
) -> dict[str, Any]:
    _require(corruption_type in STRESSORS, f"unsupported stressor: {corruption_type}")
    _require(len(inputs) == len(TASKS), "exactly four task SMPR shards are required")
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

    reference = _load_strict(reference_atr_path)
    reference_meta = reference.get("metadata", {})
    _require(
        reference_meta.get("schema_version") == "paper1-acpc-horizon-v2-1.0",
        "reference ATR schema mismatch",
    )
    _require(
        reference_meta.get("artifact_role") == "cross_stressor_external_atr",
        "reference ATR is not cross-stressor E3",
    )
    for name, expected in {
        "protocol_sha256": protocol_sha,
        "model_family": model_family,
        "training_family_id": family_id,
        "training_seed": training_seed,
        "split_name": split_name,
        "corruption_type": corruption_type,
        "stressor_family": stressor_family,
        "severity_parameter": severity_parameter,
        "severity": severity,
        "behavior_blind": True,
        "threshold_search_allowed": False,
    }.items():
        _require(reference_meta.get(name) == expected, f"ATR metadata mismatch: {name}")
    reference_rows = {
        (str(row.get("task")), str(row.get("std_key"))): row
        for row in reference.get("rows", [])
    }
    expected_grid = {(task, std_key) for task in TASKS for std_key in STD_KEYS}
    _require(set(reference_rows) == expected_grid, "reference ATR endpoint coverage mismatch")

    sampling = protocol["diagnostic_sampling"]
    n_sequences = int(sampling["n_anchors"])
    num_draws = int(sampling["num_noise_draws"])
    anchor_seed = int(sampling["anchor_seed"])
    draw_seeds = _expected_draw_seeds(anchor_seed, num_draws)
    canonical_metric = ROOT / "tools" / "paper1_acpc_metrics.py"
    semantic_margin = ROOT / "tools" / "paper1_semantic_margin.py"
    smpr_runner = ROOT / "paper1/scripts/smpr_sensitivity.py"
    acpc_runner = ROOT / "tools/paper1_phase0_acpc.py"
    _require(
        _sha256(canonical_metric) == protocol["source_hashes"]["canonical_metric"],
        "canonical metric differs from the frozen protocol",
    )
    _require(
        _sha256(semantic_margin) == protocol["source_hashes"]["semantic_margin"],
        "semantic margin differs from the frozen protocol",
    )
    _require(
        _sha256(smpr_runner) == protocol["source_hashes"]["smpr_runner"],
        "SMPR runner differs from the frozen protocol",
    )
    _require(
        reference_meta.get("implementation_hashes", {}).get("acpc_runner")
        == _sha256(acpc_runner),
        "SMPR runtime differs from reference ATR",
    )

    source_paths: dict[str, str] = {
        "manifest": str(manifest_path),
        "protocol": str(protocol_path),
        "reference_atr": str(reference_atr_path),
    }
    source_hashes: dict[str, str] = {
        "manifest": _sha256(manifest_path),
        "protocol": protocol_sha,
        "reference_atr": _sha256(reference_atr_path),
    }
    source_metadata: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    seen_tasks: set[str] = set()
    for path in inputs:
        _require(path.is_file(), f"missing SMPR source: {path}")
        payload = _load_strict(path)
        metadata = payload.get("metadata", {})
        raw_rows = payload.get("rows", [])
        _require(metadata.get("schema_version") == "paper1-smpr-v2-1.0", f"{path}: source schema mismatch")
        _require(metadata.get("code_commit") == SOURCE_COMMIT, f"{path}: raw commit mismatch")
        _require(metadata.get("model_family") == model_family, f"{path}: model family mismatch")
        _require(metadata.get("training_family_id") == family_id, f"{path}: family id mismatch")
        _require(metadata.get("training_seed") == training_seed, f"{path}: training seed mismatch")
        _require(metadata.get("status") == "complete", f"{path}: source is incomplete")
        _require(metadata.get("status_counts") == {"ok": 2}, f"{path}: row failures")
        _require(metadata.get("missing_rows") == [], f"{path}: missing rows")
        _require(metadata.get("errors") == [], f"{path}: recorded errors")
        _require(isinstance(raw_rows, list) and len(raw_rows) == 2, f"{path}: expected two rows")
        tasks = {str(row.get("task")) for row in raw_rows}
        _require(len(tasks) == 1, f"{path}: expected one task")
        task = next(iter(tasks))
        _require(task in TASKS and task not in seen_tasks, f"{path}: duplicate/unknown task")
        seen_tasks.add(task)
        _require(metadata.get("script_path") == "paper1/scripts/smpr_sensitivity.py", f"{path}: SMPR runner path mismatch")
        _require(metadata.get("script_sha256") == protocol["source_hashes"]["smpr_runner"], f"{path}: SMPR runner hash mismatch")
        expected_impl = {
            "semantic_margin": protocol["source_hashes"]["semantic_margin"],
            "canonical_metric": protocol["source_hashes"]["canonical_metric"],
            "acpc_runtime": _sha256(acpc_runner),
        }
        for name, expected in expected_impl.items():
            _require(
                metadata.get("implementation_hashes", {}).get(name) == expected,
                f"{path}: implementation hash mismatch: {name}",
            )
        _require(
            _same_path(metadata.get("source_paths", {}).get("evals", ""), manifest_path),
            f"{path}: raw manifest path mismatch",
        )
        _require(
            metadata.get("source_hashes", {}).get("evals") == _sha256(manifest_path),
            f"{path}: raw manifest hash mismatch",
        )
        _require(
            _same_path(metadata.get("source_paths", {}).get("reference_atr", ""), reference_atr_path),
            f"{path}: reference ATR path mismatch",
        )
        _require(
            metadata.get("source_hashes", {}).get("reference_atr")
            == _sha256(reference_atr_path),
            f"{path}: reference ATR hash mismatch",
        )
        _require(
            metadata.get("evaluation_seeds")
            == protocol["behavior_evaluation"]["evaluation_seeds"],
            f"{path}: conditional evaluation seed contract mismatch",
        )
        raw_protocol = metadata.get("protocol", {})
        expected_raw_protocol = {
            "radius_metric": protocol["radius_metric"],
            "rollout_horizon": protocol["rollout_horizon"],
            "horizon_weights": protocol["horizon_weights"],
            "radius_quantile": protocol["smpr_radius_quantile"],
            "normalization": protocol["normalization"],
            "pair_rule": protocol["smpr_pair_rule"],
            "local_state_quantile": protocol["smpr_local_quantile"],
            "margin_delta_normalized": protocol["smpr_margin_delta_normalized"],
            "n_sequences": n_sequences,
            "num_noise_draws": num_draws,
            "anchor_seed": anchor_seed,
        }
        for name, expected in expected_raw_protocol.items():
            _require(raw_protocol.get(name) == expected, f"{path}: raw protocol mismatch: {name}")
        # The frozen runner predates cross-stressor execution and writes this known
        # Gaussian string unconditionally.  Only this metadata placeholder is
        # tolerated; row-level corruption fields and all reference provenance are
        # still required to identify the real stressor exactly.
        _require(
            raw_protocol.get("corruption") == RAW_PROTOCOL_PLACEHOLDER,
            f"{path}: unknown raw corruption placeholder",
        )
        source_key = f"smpr_source_{task}"
        source_paths[source_key] = str(path)
        source_hashes[source_key] = _sha256(path)
        source_metadata[task] = metadata

        by_std = {str(row.get("std_key")): row for row in raw_rows}
        _require(set(by_std) == set(STD_KEYS), f"{path}: endpoint coverage mismatch")
        for std_key in STD_KEYS:
            row = by_std[std_key]
            key = f"{task}/{std_key}"
            entry = manifest.get(task, {}).get(std_key)
            _require(isinstance(entry, Mapping), f"{key}: manifest entry missing")
            reference_row = reference_rows[(task, std_key)]
            _require(row.get("status") == "ok", f"{key}: row is not ok")
            _require(row.get("model_family") == model_family, f"{key}: row family mismatch")
            _require(row.get("training_family_id") == family_id, f"{key}: row family id mismatch")
            _require(row.get("training_seed") == training_seed, f"{key}: row seed mismatch")
            _require(row.get("corruption_type") == corruption_type, f"{key}: stressor mismatch")
            _require(
                math.isclose(_finite(row.get("noise_std"), name=f"{key}/severity"), severity, rel_tol=0.0, abs_tol=1e-12),
                f"{key}: severity is not the frozen strongest value",
            )
            _require(row.get("corrupt_goal") is False, f"{key}: goal must remain clean")
            _require(row.get("n_sequences") == n_sequences, f"{key}: anchor count mismatch")
            _require(row.get("num_noise_draws") == num_draws, f"{key}: draw count mismatch")
            _require(row.get("anchor_seed") == anchor_seed, f"{key}: anchor seed mismatch")
            _require(row.get("noise_draw_seed_rule") == "anchor_seed+1009+7919*draw_index", f"{key}: draw rule mismatch")
            _require(row.get("noise_draw_seeds") == draw_seeds, f"{key}: draw seeds mismatch")
            _require(row.get("rollout_horizon_actual") == protocol["rollout_horizon"], f"{key}: horizon mismatch")
            _require(row.get("embedding_space") == "normalized", f"{key}: embedding-space mismatch")
            _require(row.get("radius_metric") == protocol["radius_metric"], f"{key}: radius metric mismatch")
            _require(row.get("radius_quantile") == protocol["smpr_radius_quantile"], f"{key}: radius quantile mismatch")
            _require(row.get("pair_rule") == protocol["smpr_pair_rule"], f"{key}: pair rule mismatch")
            _require(row.get("local_state_quantile") == protocol["smpr_local_quantile"], f"{key}: local quantile mismatch")
            _require(row.get("margin_delta_norm") == protocol["smpr_margin_delta_normalized"], f"{key}: margin mismatch")
            _require(row.get("atr_reference_match") is True, f"{key}: raw ATR match is false")
            atr_error = _finite(row.get("atr_reference_abs_error"), name=f"{key}/ATR error")
            _require(atr_error <= 1e-6, f"{key}: ATR reference error exceeds 1e-6")
            reference_atr = _finite(reference_row.get("atr_horizon_v2_q90"), name=f"{key}/reference ATR")
            tube_radius = _finite(row.get("same_state_tube_radius"), name=f"{key}/tube radius")
            row_reference = _finite(row.get("reference_atr_horizon_v2_q90"), name=f"{key}/row reference ATR")
            _require(
                math.isclose(row_reference, reference_atr, rel_tol=0.0, abs_tol=1e-6),
                f"{key}: row/reference ATR mismatch",
            )
            _require(
                math.isclose(tube_radius, reference_atr, rel_tol=0.0, abs_tol=1e-6),
                f"{key}: SMPR tube/reference ATR mismatch",
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
            _require(
                model_file == _rooted(str(reference_row.get("model_file"))).resolve(),
                f"{key}: SMPR/ATR checkpoint path mismatch",
            )
            _require(
                checkpoint_sha == reference_row.get("checkpoint_sha256"),
                f"{key}: SMPR/ATR checkpoint hash mismatch",
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
                    "smpr": _finite(row.get("smpr"), name=f"{key}/SMPR"),
                    "same_state_tube_radius": tube_radius,
                    "reference_atr_horizon_v2_q90": reference_atr,
                    "atr_reference_match": True,
                    "atr_reference_abs_error": atr_error,
                    "radius_metric": row["radius_metric"],
                    "radius_quantile": _finite(row["radius_quantile"], name=f"{key}/radius quantile"),
                    "pair_rule": row["pair_rule"],
                    "margin_delta_norm": _finite(row["margin_delta_norm"], name=f"{key}/margin"),
                    "semantic_pair_count": int(row["semantic_pair_count"]),
                    "semantic_skipped_anchor_count": int(row["semantic_skipped_anchor_count"]),
                    "semantic_skip_rate": _finite(row["semantic_skip_rate"], name=f"{key}/skip rate"),
                    "n_sequences": int(row["n_sequences"]),
                    "num_noise_draws": int(row["num_noise_draws"]),
                    "anchor_seed": int(row["anchor_seed"]),
                    "corrupt_goal": False,
                    "model_file": str(model_file),
                    "checkpoint_sha256": checkpoint_sha,
                    "split_name": split_name,
                }
            )

    _require(seen_tasks == set(TASKS), "SMPR task coverage mismatch")
    rows.sort(key=lambda row: (TASKS.index(str(row["task"])), float(row["training_rho"])))
    _require(len(rows) == len(TASKS) * len(STD_KEYS), "SMPR row coverage mismatch")
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
                "semantic_margin": str(semantic_margin.relative_to(ROOT)),
                "canonical_metric": str(canonical_metric.relative_to(ROOT)),
                "smpr_runner": str(smpr_runner.relative_to(ROOT)),
                "acpc_runtime": str(acpc_runner.relative_to(ROOT)),
            },
            "implementation_hashes": {
                "semantic_margin": _sha256(semantic_margin),
                "canonical_metric": _sha256(canonical_metric),
                "smpr_runner": _sha256(smpr_runner),
                "acpc_runtime": _sha256(acpc_runner),
            },
            "protocol_hash": protocol_sha,
            "protocol_sha256": protocol_sha,
            "protocol_hash_status": "E3_cross_stressor_bound_to_immutable_protocol",
            "model_family": model_family,
            "training_family_id": family_id,
            "training_seed": training_seed,
            "training_seed_semantics": training_seed_semantics,
            "split_name": split_name,
            "artifact_role": "cross_stressor_external_smpr",
            "behavior_blind": True,
            "threshold_search_allowed": False,
            "stressor_family": stressor_family,
            "corruption_type": corruption_type,
            "severity_parameter": severity_parameter,
            "severity": severity,
            "raw_source_protocol_placeholder": RAW_PROTOCOL_PLACEHOLDER,
            "raw_source_protocol_placeholder_note": (
                "the frozen runner writes this legacy Gaussian metadata string; "
                "row-level stressor fields and ATR provenance are authoritative"
            ),
            "status": "complete",
            "status_counts": {"ok": len(rows)},
            "missing_rows": [],
            "errors": [],
            "protocol": {
                "radius_metric": protocol["radius_metric"],
                "rollout_horizon": protocol["rollout_horizon"],
                "horizon_weights": protocol["horizon_weights"],
                "normalization": protocol["normalization"],
                "n_sequences": n_sequences,
                "num_noise_draws": num_draws,
                "anchor_seed": anchor_seed,
                "local_state_quantile": protocol["smpr_local_quantile"],
                "margin_delta_normalized": protocol["smpr_margin_delta_normalized"],
                "pair_rule": protocol["smpr_pair_rule"],
                "radius_quantile": protocol["smpr_radius_quantile"],
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
    parser.add_argument("--reference-atr", type=Path, required=True)
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
        raise ValueError("SMPR output cannot be the frozen protocol path")
    artifact = build_cross_stressor_smpr_artifact(
        inputs=args.input,
        reference_atr_path=args.reference_atr,
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
