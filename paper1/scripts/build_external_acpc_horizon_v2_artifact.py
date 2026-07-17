#!/usr/bin/env python3
"""Bind a held-out LeWM ATR sweep to the immutable diagnostic protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from paper1.scripts.build_acpc_horizon_v2_artifact import (
    ROOT,
    TASKS,
    build_artifact,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _frozen_protocol(path: Path) -> tuple[dict[str, Any], str]:
    protocol = _load_strict(path)
    _require(
        protocol.get("schema_version")
        == "paper1-frozen-diagnostic-protocol-1.0",
        "unsupported frozen protocol schema",
    )
    _require(
        protocol.get("status") == "frozen" and protocol.get("immutable") is True,
        "protocol is not frozen and immutable",
    )
    external = protocol.get("external_policy", {})
    _require(
        external.get("threshold_search_allowed") is False,
        "external threshold search must remain disabled",
    )
    _require(
        external.get("protocol_write_allowed") is False,
        "external consumers must not write the protocol",
    )
    return protocol, _sha256(path)


def build_external_artifact(
    *,
    inputs: Sequence[Path],
    eval_manifest_path: Path,
    protocol_path: Path,
    training_seed: int,
    family_id: str,
) -> dict[str, Any]:
    protocol, protocol_sha = _frozen_protocol(protocol_path)
    heldout_seeds = {
        int(seed)
        for seed in protocol["external_policy"]["heldout_lewm_training_seeds"]
    }
    _require(
        training_seed in heldout_seeds,
        f"training seed {training_seed} is not frozen held-out LeWM TEST data",
    )
    _require(family_id == f"lewm_seed{training_seed}", "family id/seed mismatch")
    sampling = protocol["diagnostic_sampling"]
    evaluation_seeds = [
        int(seed)
        for seed in protocol["behavior_evaluation"]["evaluation_seeds"]
    ]
    _require(protocol.get("model_family") == "LeWM", "protocol family mismatch")

    artifact = build_artifact(
        inputs=inputs,
        eval_manifest_path=eval_manifest_path,
        model_family="LeWM",
        training_seed=training_seed,
        evaluation_seeds=evaluation_seeds,
        n_sequences=int(sampling["n_anchors"]),
        num_noise_draws=int(sampling["num_noise_draws"]),
        anchor_seed=int(sampling["anchor_seed"]),
    )
    manifest = _load_strict(eval_manifest_path)
    _require(
        manifest.get("_metadata", {}).get("training_seed") == training_seed,
        "held-out eval manifest seed mismatch",
    )
    diagnostic_rows: list[dict[str, Any]] = []
    checkpoint_paths: dict[str, Path] = {}
    checkpoint_hashes: dict[str, str] = {}
    for row in artifact["rows"]:
        task = str(row["task"])
        std_key = str(row["std_key"])
        entry: Mapping[str, Any] = manifest[task][std_key]
        _require(row.get("subdir") == entry.get("subdir"), f"{task}/{std_key}: subdir mismatch")
        _require(
            Path(str(row.get("run_path"))).resolve()
            == Path(str(entry.get("path"))).resolve(),
            f"{task}/{std_key}: checkpoint run path mismatch",
        )
        model_file = Path(str(row.get("model_file"))).resolve()
        _require(
            model_file.parent
            == Path(str(entry.get("path"))).resolve(),
            f"{task}/{std_key}: model file is outside the manifest run",
        )
        _require(model_file.is_file(), f"{task}/{std_key}: model file is missing")
        checkpoint_key = f"checkpoint_{task}_{std_key}"
        checkpoint_paths[checkpoint_key] = model_file
        checkpoint_hashes[checkpoint_key] = _sha256(model_file)
        for metric_name in ("clean", "pixels_std0.08"):
            metric = entry.get("metrics", {}).get(metric_name, {})
            _require(
                [int(seed) for seed in metric.get("seeds", [])]
                == evaluation_seeds,
                f"{task}/{std_key}: {metric_name} evaluation seeds mismatch",
            )
        _require(row.get("noise_std") == sampling["evaluation_noise_std"], f"{task}/{std_key}: noise mismatch")
        _require(row.get("corrupt_goal") is sampling["corrupt_goal"], f"{task}/{std_key}: goal corruption mismatch")
        _require(row.get("rollout_horizon_actual") == protocol["rollout_horizon"], f"{task}/{std_key}: horizon mismatch")
        _require(row.get("embedding_space") == "normalized", f"{task}/{std_key}: embedding-space mismatch")
        _require(row.get("corruption_type") == "gaussian_noise", f"{task}/{std_key}: corruption mismatch")
        _require(
            row.get("noise_draw_seed_rule") == "seed+1009+7919*draw_index",
            f"{task}/{std_key}: draw seed rule mismatch",
        )
        _require(
            row.get("noise_draw_seeds")
            == [
                int(sampling["anchor_seed"]) + 1009 + 7919 * index
                for index in range(int(sampling["num_noise_draws"]))
            ],
            f"{task}/{std_key}: draw seeds mismatch",
        )
        _require(
            row.get("normalization") == "per_anchor_clean_transition_l2_q50",
            f"{task}/{std_key}: normalization mismatch",
        )
        clean_scales = [float(value) for value in row["clean_transition_scale"]]
        _require(
            len(clean_scales) == sampling["n_anchors"],
            f"{task}/{std_key}: clean-scale coverage mismatch",
        )
        diagnostic_rows.append(
            {
                "status": "ok",
                "model_family": "LeWM",
                "training_family_id": family_id,
                "training_seed": training_seed,
                "task": task,
                "std_key": std_key,
                "training_rho": float(std_key),
                "atr_horizon_v2_q90": float(row["atr_horizon_v2_q90"]),
                "radius_metric": row["radius_metric"],
                "rollout_horizon_actual": int(row["rollout_horizon_actual"]),
                "n_sequences": int(row["n_sequences"]),
                "num_noise_draws": int(row["num_noise_draws"]),
                "noise_std": float(row["noise_std"]),
                "corrupt_goal": bool(row["corrupt_goal"]),
                "clean_transition_scale_min": min(clean_scales),
                "clean_transition_scale_zero_count": sum(
                    value == 0.0 for value in clean_scales
                ),
                "model_file": str(model_file),
                "checkpoint_sha256": checkpoint_hashes[checkpoint_key],
                "split_name": "TEST",
            }
        )
    artifact["rows"] = diagnostic_rows

    metadata = artifact["metadata"]
    base_builder_path = ROOT / "paper1" / "scripts" / "build_acpc_horizon_v2_artifact.py"
    script_path = Path(__file__).resolve()
    metadata.update(
        {
            "script_path": str(script_path.relative_to(ROOT)),
            "script_sha256": _sha256(script_path),
            "base_builder_path": str(base_builder_path.relative_to(ROOT)),
            "base_builder_sha256": _sha256(base_builder_path),
            "protocol_hash": protocol_sha,
            "protocol_sha256": protocol_sha,
            "protocol_hash_status": "heldout_external_bound_to_immutable_protocol",
            "training_family_id": family_id,
            "training_seed_semantics": "one independent held-out LeWM TEST training run",
            "split_name": "TEST",
            "artifact_role": "heldout_external_atr",
            "behavior_blind": True,
        }
    )
    metadata["source_paths"]["protocol"] = str(protocol_path)
    metadata["source_hashes"]["protocol"] = protocol_sha
    metadata["source_paths"].update(
        {name: str(path) for name, path in checkpoint_paths.items()}
    )
    metadata["source_hashes"].update(checkpoint_hashes)
    calibration_atr_path = ROOT / protocol["source_paths"]["calibration_atr"]
    calibration_atr = _load_strict(calibration_atr_path)
    _require(
        metadata["implementation_hashes"].get("acpc_runner")
        == calibration_atr.get("metadata", {}).get("implementation_hashes", {}).get(
            "acpc_runner"
        ),
        "held-out ATR runner hash differs from CAL",
    )
    _require(
        set(metadata["source_paths"]) == set(metadata["source_hashes"]),
        "source path/hash provenance keys differ",
    )
    _require(len(diagnostic_rows) == len(TASKS) * 9, "held-out ATR coverage mismatch")
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
    parser.add_argument("--eval-manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--family-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    artifact = build_external_artifact(
        inputs=args.input,
        eval_manifest_path=args.eval_manifest,
        protocol_path=args.protocol,
        training_seed=args.training_seed,
        family_id=args.family_id,
    )
    _write_exclusive(args.out, artifact)
    print(f"wrote {args.out} ({len(artifact['rows'])} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
