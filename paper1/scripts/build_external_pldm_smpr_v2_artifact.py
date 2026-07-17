#!/usr/bin/env python3
"""Merge canonical PLDM SMPR shards under the immutable diagnostic protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from paper1.scripts.build_external_acpc_horizon_v2_artifact import (
    _frozen_protocol,
)
from paper1.scripts.build_smpr_v2_artifact import ROOT, TASKS, build_artifact


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite numeric, got bool")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} is non-finite")
    return result


def build_external_artifact(
    *,
    inputs: Sequence[Path],
    reference_atr_path: Path,
    protocol_path: Path,
) -> dict[str, Any]:
    protocol_original = protocol_path.read_bytes()
    protocol_mtime = protocol_path.stat().st_mtime_ns
    protocol, protocol_sha = _frozen_protocol(protocol_path)
    reference = json.loads(
        reference_atr_path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )
    reference_meta = reference.get("metadata", {})
    _require(
        reference_meta.get("artifact_role") == "pldm_canonical_external_atr",
        "reference ATR is not canonical PLDM E2",
    )
    _require(reference_meta.get("protocol_sha256") == protocol_sha, "ATR protocol hash mismatch")
    _require(reference_meta.get("model_family") == "PLDM", "ATR family mismatch")
    _require(reference_meta.get("split_name") == "E2", "ATR split mismatch")
    family_id = str(reference_meta["training_family_id"])
    training_seed = int(reference_meta["training_seed"])

    artifact = build_artifact(
        inputs=inputs,
        reference_atr_path=reference_atr_path,
        model_family="PLDM",
        family_id=family_id,
        training_seed=training_seed,
    )
    sampling = protocol["diagnostic_sampling"]
    smpr_protocol = artifact["metadata"].get("protocol", {})
    expected_protocol = {
        "radius_metric": protocol["radius_metric"],
        "rollout_horizon": protocol["rollout_horizon"],
        "horizon_weights": protocol["horizon_weights"],
        "normalization": protocol["normalization"],
        "n_sequences": int(sampling["n_anchors"]),
        "num_noise_draws": int(sampling["num_noise_draws"]),
        "anchor_seed": int(sampling["anchor_seed"]),
        "local_state_quantile": protocol["smpr_local_quantile"],
        "margin_delta_normalized": protocol["smpr_margin_delta_normalized"],
        "pair_rule": protocol["smpr_pair_rule"],
        "radius_quantile": protocol["smpr_radius_quantile"],
    }
    for key, expected in expected_protocol.items():
        _require(smpr_protocol.get(key) == expected, f"SMPR protocol mismatch: {key}")

    rows: list[dict[str, Any]] = []
    reference_rows = {
        (str(row["task"]), str(row["std_key"])): row
        for row in reference.get("rows", [])
    }
    _require(len(reference_rows) == len(TASKS) * 9, "ATR checkpoint coverage mismatch")
    checkpoint_paths: dict[str, Path] = {}
    checkpoint_hashes: dict[str, str] = {}
    for row in artifact["rows"]:
        std_key = str(row["std_key"])
        task = str(row["task"])
        reference_row = reference_rows[(task, std_key)]
        model_file = Path(str(row["model_file"])).resolve()
        _require(
            model_file == Path(str(reference_row["model_file"])).resolve(),
            f"{task}/{std_key}: SMPR/ATR checkpoint path mismatch",
        )
        checkpoint_sha = _sha256(model_file)
        _require(
            checkpoint_sha == reference_row["checkpoint_sha256"],
            f"{task}/{std_key}: SMPR/ATR checkpoint hash mismatch",
        )
        checkpoint_key = f"checkpoint_{task}_{std_key}"
        checkpoint_paths[checkpoint_key] = model_file
        checkpoint_hashes[checkpoint_key] = checkpoint_sha
        rows.append(
            {
                "status": "ok",
                "model_family": "PLDM",
                "training_family_id": family_id,
                "training_seed": training_seed,
                "task": task,
                "std_key": std_key,
                "training_rho": float(std_key),
                "smpr": _finite(row["smpr"], name="SMPR"),
                "same_state_tube_radius": _finite(
                    row["same_state_tube_radius"],
                    name="same-state tube radius",
                ),
                "reference_atr_horizon_v2_q90": _finite(
                    row["reference_atr_horizon_v2_q90"],
                    name="reference ATR",
                ),
                "atr_reference_match": row["atr_reference_match"],
                "atr_reference_abs_error": _finite(
                    row["atr_reference_abs_error"],
                    name="ATR reference error",
                ),
                "radius_metric": row["radius_metric"],
                "radius_quantile": row["radius_quantile"],
                "pair_rule": row["pair_rule"],
                "margin_delta_norm": row["margin_delta_norm"],
                "semantic_pair_count": int(row["semantic_pair_count"]),
                "semantic_skipped_anchor_count": int(
                    row["semantic_skipped_anchor_count"]
                ),
                "semantic_skip_rate": _finite(
                    row["semantic_skip_rate"],
                    name="semantic skip rate",
                ),
                "n_sequences": int(row["n_sequences"]),
                "num_noise_draws": int(row["num_noise_draws"]),
                "noise_std": _finite(row["noise_std"], name="noise std"),
                "anchor_seed": int(row["anchor_seed"]),
                "model_file": str(model_file),
                "checkpoint_sha256": checkpoint_sha,
                "split_name": "E2",
            }
        )
    artifact["rows"] = rows
    metadata = artifact["metadata"]
    base_builder = ROOT / "paper1" / "scripts" / "build_smpr_v2_artifact.py"
    script_path = Path(__file__).resolve()
    metadata.update(
        {
            "script_path": str(script_path.relative_to(ROOT)),
            "script_sha256": _sha256(script_path),
            "base_builder_path": str(base_builder.relative_to(ROOT)),
            "base_builder_sha256": _sha256(base_builder),
            "protocol_hash": protocol_sha,
            "protocol_sha256": protocol_sha,
            "protocol_hash_status": "E2_external_bound_to_immutable_protocol",
            "training_seed_semantics": "one independently trained PLDM checkpoint family",
            "evaluation_seed_semantics": "conditional evaluation variability, not training-run replication",
            "split_name": "E2",
            "artifact_role": "pldm_canonical_external_smpr",
            "behavior_blind": True,
            "raw_source_protocol_placeholder": (
                "raw frozen runner shards retain their pre-freeze placeholder; "
                "this wrapper-level protocol binding is authoritative"
            ),
        }
    )
    metadata["source_paths"]["protocol"] = str(protocol_path)
    metadata["source_hashes"]["protocol"] = protocol_sha
    metadata["source_paths"].update(
        {name: str(path) for name, path in checkpoint_paths.items()}
    )
    metadata["source_hashes"].update(checkpoint_hashes)
    _require(
        metadata["source_hashes"].get("reference_atr") == _sha256(reference_atr_path),
        "SMPR reference ATR provenance mismatch",
    )
    for local_key, protocol_key in (
        ("canonical_metric", "canonical_metric"),
        ("semantic_margin", "semantic_margin"),
        ("runner", "smpr_runner"),
    ):
        _require(
            metadata["implementation_hashes"].get(local_key)
            == protocol["source_hashes"][protocol_key],
            f"SMPR implementation hash mismatch: {local_key}",
        )
    _require(set(metadata["source_paths"]) == set(metadata["source_hashes"]), "source provenance keys differ")
    _require(len(rows) == len(TASKS) * 9, "PLDM SMPR coverage mismatch")
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
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.out.resolve() == args.protocol.resolve():
        raise ValueError("SMPR output cannot be the frozen protocol path")
    artifact = build_external_artifact(
        inputs=args.input,
        reference_atr_path=args.reference_atr,
        protocol_path=args.protocol,
    )
    _write_exclusive(args.out, artifact)
    print(f"wrote {args.out} ({len(artifact['rows'])} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
