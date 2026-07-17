#!/usr/bin/env python3
"""Merge 12 resumable JVP/Hutchinson shards into the canonical full v2 audit."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

from tools import paper1_jvp_hutchinson_sensitivity_audit as jvp


ROOT = Path(__file__).resolve().parents[2]
TASKS = ("TwoRoom", "PushT", "Reacher", "Cube")
SEEDS = (3072, 3073, 3074)
CHECKPOINT_TYPES = ("base", "onset", "endpoint")


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


def build_artifact(inputs: Sequence[Path]) -> dict[str, Any]:
    _require(len(inputs) == 12, "exactly 12 task-seed shards are required")
    rows: list[dict[str, Any]] = []
    probe_rows: list[dict[str, Any]] = []
    source_paths: dict[str, str] = {}
    source_hashes: dict[str, str] = {}
    source_metadata: dict[str, Mapping[str, Any]] = {}
    seen_blocks: set[tuple[str, int]] = set()
    seen_rows: set[tuple[str, int, str]] = set()
    for path in inputs:
        _require(path.is_file(), f"missing JVP shard: {path}")
        payload = _load(path)
        metadata = payload.get("metadata", {})
        _require(metadata.get("schema_version") == jvp.SCHEMA_VERSION, f"{path}: schema mismatch")
        _require(metadata.get("status") == "complete", f"{path}: incomplete shard")
        _require(metadata.get("status_counts") == {"ok": 3}, f"{path}: shard row failure")
        _require(metadata.get("missing_rows") == [], f"{path}: missing rows")
        _require(metadata.get("errors") == [], f"{path}: row errors")
        args = payload.get("args", {})
        _require(args.get("n_sequences") == 16, f"{path}: anchor count mismatch")
        _require(args.get("hutchinson_probes") == 8, f"{path}: probe count mismatch")
        _require(args.get("rollout_horizon") == 8, f"{path}: horizon mismatch")
        shard_rows = payload.get("rows", [])
        shard_probes = payload.get("probe_rows", [])
        _require(len(shard_rows) == 3, f"{path}: expected three checkpoint rows")
        _require(len(shard_probes) == 24, f"{path}: expected 24 probe rows")
        tasks = {str(row.get("task")) for row in shard_rows}
        seeds = {int(row.get("training_seed")) for row in shard_rows}
        _require(len(tasks) == 1 and len(seeds) == 1, f"{path}: shard is not one task-seed block")
        task = next(iter(tasks))
        seed = next(iter(seeds))
        block = (task, seed)
        _require(task in TASKS and seed in SEEDS and block not in seen_blocks, f"{path}: duplicate/unknown block")
        seen_blocks.add(block)
        _require(
            {str(row.get("checkpoint_type")) for row in shard_rows}
            == set(CHECKPOINT_TYPES),
            f"{path}: checkpoint-type coverage mismatch",
        )
        source_key = f"{task}_seed{seed}"
        source_paths[f"shard_{source_key}"] = str(path)
        source_hashes[f"shard_{source_key}"] = _sha256(path)
        source_metadata[source_key] = metadata
        for raw in shard_rows:
            row = dict(raw)
            key = (task, seed, str(row.get("checkpoint_type")))
            _require(key not in seen_rows, f"{path}: duplicate checkpoint row")
            seen_rows.add(key)
            _require(row.get("status") == "ok", f"{path}: row not ok")
            model_file = Path(str(row.get("model_file"))).expanduser().resolve()
            _require(model_file.is_file(), f"{path}: checkpoint missing")
            checkpoint_hash = _sha256(model_file)
            row["model_file"] = str(model_file)
            row["checkpoint_sha256"] = checkpoint_hash
            checkpoint_key = (
                f"checkpoint_{task}_seed{seed}_{row['checkpoint_type']}"
            )
            source_paths[checkpoint_key] = str(model_file)
            source_hashes[checkpoint_key] = checkpoint_hash
            for field in (
                "encoder_trace_mean_ci95_unclipped",
                "rollout_trace_mean_ci95_unclipped",
                "composed_trace_mean_ci95_unclipped",
                "kappa_submultiplicative_probe_ci95_unclipped",
                "kappa_relative_isotropic_probe_ci95_unclipped",
            ):
                interval = row.get(field)
                _require(
                    isinstance(interval, list) and len(interval) == 2,
                    f"{path}: missing finite-probe interval {field}",
                )
            rows.append(row)
        for probe in shard_probes:
            _require(
                str(probe.get("task")) == task
                and int(probe.get("training_seed")) == seed,
                f"{path}: probe block mismatch",
            )
            probe_rows.append(dict(probe))
    _require(
        seen_blocks == {(task, seed) for task in TASKS for seed in SEEDS},
        "JVP task-seed coverage mismatch",
    )
    _require(len(rows) == 36 and len(probe_rows) == 288, "full JVP row count mismatch")
    task_order = {task: index for index, task in enumerate(TASKS)}
    type_order = {name: index for index, name in enumerate(CHECKPOINT_TYPES)}
    rows.sort(
        key=lambda row: (
            task_order[str(row["task"])],
            int(row["training_seed"]),
            type_order[str(row["checkpoint_type"])],
        )
    )
    probe_rows.sort(
        key=lambda row: (
            task_order[str(row["task"])],
            int(row["training_seed"]),
            type_order[str(row["checkpoint_type"])],
            int(row["probe_index"]),
        )
    )
    summary = jvp._summarize(rows)
    script_path = Path(__file__).resolve()
    runner_path = ROOT / "tools/paper1_jvp_hutchinson_sensitivity_audit.py"
    metric_path = ROOT / "tools/paper1_acpc_metrics.py"
    source_paths.update(
        {
            "merge_builder": str(script_path),
            "jvp_runner": str(runner_path),
            "canonical_metric": str(metric_path),
        }
    )
    source_hashes.update(
        {
            "merge_builder": _sha256(script_path),
            "jvp_runner": _sha256(runner_path),
            "canonical_metric": _sha256(metric_path),
        }
    )
    created_utc = datetime.now(timezone.utc).isoformat()
    return {
        "metadata": {
            "schema_version": jvp.SCHEMA_VERSION,
            "created_utc": created_utc,
            "code_commit": _git_commit(),
            "status": "complete",
            "status_counts": {"ok": 36},
            "missing_rows": [],
            "errors": [],
            "model_family": "LeWM",
            "training_seeds": list(SEEDS),
            "training_seed_semantics": "independently trained checkpoint seeds; never evaluation seeds",
            "evaluation_seed_semantics": "not applicable; checkpoint-local anchor/JVP audit",
            "tasks": list(TASKS),
            "checkpoint_types": list(CHECKPOINT_TYPES),
            "n_sequences": 16,
            "hutchinson_probes": 8,
            "protocol_hash": None,
            "protocol_hash_status": "not_frozen_phase1_correctness_audit",
            "source_paths": source_paths,
            "source_hashes": source_hashes,
            "source_metadata": source_metadata,
            "map_contract": {
                "radius_metric": "horizon_weighted_stacked_l2_v2",
                "rollout_horizon": 8,
                "horizon_weights": "uniform alpha_k=1/H applied as sqrt(alpha_k)",
                "rollout_projection": "identity in the selected embedding_space",
                "rollout_vectorization": "batch-major canonical weighted stack of (H,D) into (H*D)",
                "normalization": "none inside JVP; canonical per-anchor transition scale is applied after this map",
            },
            "finite_probe_interval": {
                "method": "unclipped normal mean interval using sample standard error",
                "formal_coverage_guarantee": False,
                "silent_clipping": False,
            },
        },
        "generated_at": created_utc,
        "args": {
            "seeds": list(SEEDS),
            "tasks": list(TASKS),
            "n_sequences": 16,
            "hutchinson_probes": 8,
            "rollout_horizon": 8,
            "execution": "12 serial resumable task-seed shards",
        },
        "rows": rows,
        "probe_rows": probe_rows,
        "summary": summary,
        "notes": [
            "Exact autograd JVPs estimate local Frobenius traces; no full Jacobian is materialized.",
            "kappa_submultiplicative and kappa_relative_isotropic are both reported without clipping.",
            "Finite-probe intervals are descriptive Monte Carlo intervals, not formal coverage guarantees.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=ROOT / "paper1/results/jvp_hutchinson_sensitivity_audit_v2.json",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=ROOT / "paper1/results/jvp_hutchinson_sensitivity_audit_v2.csv",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=ROOT / "paper1/results/jvp_hutchinson_sensitivity_summary_v2.csv",
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=ROOT / "paper1/tables/table_jvp_hutchinson_sensitivity_audit.tex",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = build_artifact(args.input)
    jvp._write_csv(args.out_csv, payload["rows"])
    jvp._write_csv(args.summary_csv, payload["summary"])
    jvp.write_table(
        args.table,
        payload["summary"],
        "tab:jvp-hutchinson-sensitivity-audit",
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(jvp._jsonable(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {args.out_json} "
        f"({len(payload['rows'])} rows, {len(payload['probe_rows'])} probes)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
