#!/usr/bin/env python3
"""Build the canonical Paper 1 E4 target-view diagnostic manifest.

This is an I/O-only builder.  It binds the existing matched full-sequence and
target-view LeWM checkpoints to their *current* closed-loop evaluation files;
it never loads a model or runs an evaluation.  The canonical matrix is:

    4 tasks x 8 non-zero training-noise levels x 2 target branches

Every source file is SHA-256 bound.  Missing, duplicate, or ambiguous inputs
raise :class:`ManifestValidationError`; the command-line entry point then
writes a blocked report and exits non-zero rather than emitting a partial
manifest.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "paper1-target-view-diagnostic-manifest-1.0"
TRAINING_SEED = 3072
EVAL_SEEDS = (42, 43, 44)
RHO_KEYS = ("0.01", "0.02", "0.03", "0.04", "0.05", "0.06", "0.07", "0.08")
BRANCHES = ("full_sequence", "target_view")
EVAL_CONDITIONS = {
    "clean": "origin",
    "pixels_std0.03": "pixels_std0.03",
    "pixels_std0.05": "pixels_std0.05",
    "pixels_std0.08": "pixels_std0.08",
}
TASK_LAYOUT = {
    "TwoRoom": ("lewm-tworooms", "tworoom"),
    "PushT": ("lewm-pusht", "pusht"),
    "Reacher": ("lewm-reacher", "reacher"),
    "Cube": ("lewm-cube", "cube"),
}

ROOT = Path(__file__).resolve().parents[1]


def _default_data_root() -> Path | None:
    for name in ("PAPER1_DATA_ROOT", "DATA_ROOT", "STABLEWM_HOME"):
        value = os.environ.get(name)
        if value:
            return Path(value).expanduser()
    return None


DEFAULT_DATA_ROOT = _default_data_root()
DEFAULT_LEGACY_MANIFEST = (
    ROOT
    / "assets"
    / "paper1_data"
    / "training_seed_eval_manifests"
    / "lewm_seed3072_evals.json"
)
DEFAULT_OUT = ROOT / "assets" / "paper1_data" / "target_view_diagnostic_manifest_v1.json"
DEFAULT_RUNNER_MANIFEST_DIR = (
    ROOT / "assets" / "paper1_data" / "target_view_runner_manifests"
)


class ManifestValidationError(RuntimeError):
    """The E4 source matrix is incomplete, inconsistent, or ambiguous."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestValidationError(f"cannot parse JSON {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ManifestValidationError(f"JSON root must be an object: {path}")
    return value


def _load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        import yaml

        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except ImportError:
        try:
            from omegaconf import OmegaConf

            value = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
        except Exception as exc:  # noqa: BLE001 - report a single actionable error.
            raise ManifestValidationError(
                "config parsing requires PyYAML or OmegaConf"
            ) from exc
    except Exception as exc:  # noqa: BLE001 - normalize parser failures.
        raise ManifestValidationError(f"cannot parse YAML {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ManifestValidationError(f"YAML root must be a mapping: {path}")
    return value


def _nested(config: Mapping[str, Any], *keys: str) -> tuple[bool, Any]:
    value: Any = config
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return False, None
        value = value[key]
    return True, value


def _as_float(value: Any, *, label: str, path: Path) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ManifestValidationError(f"{label} is not numeric in {path}: {value!r}") from exc
    if not math.isfinite(result):
        raise ManifestValidationError(f"{label} is not finite in {path}: {value!r}")
    return result


def _run_name(prefix: str, rho_key: str, branch: str) -> str:
    rho_suffix = f"0to0{int(round(float(rho_key) * 100)):02d}"
    if branch == "full_sequence":
        return f"{prefix}_lewm_noise_{rho_suffix}_p1"
    if branch == "target_view":
        return f"{prefix}_lewm_baseline_unperturbed_target_noise_{rho_suffix}_p1"
    raise AssertionError(branch)


def _validate_training_config(
    path: Path, *, rho_key: str, branch: str
) -> dict[str, Any]:
    config = _load_yaml(path)

    found, seed = _nested(config, "seed")
    if not found or int(_as_float(seed, label="seed", path=path)) != TRAINING_SEED:
        raise ManifestValidationError(
            f"expected training seed {TRAINING_SEED} in {path}, got {seed!r}"
        )

    found, std_max = _nested(config, "image_noise", "std_max")
    if not found:
        raise ManifestValidationError(f"missing image_noise.std_max in {path}")
    std_value = _as_float(std_max, label="image_noise.std_max", path=path)
    if not math.isclose(std_value, float(rho_key), rel_tol=0.0, abs_tol=1e-9):
        raise ManifestValidationError(
            f"training std mismatch in {path}: expected {rho_key}, got {std_value}"
        )

    found, noise_type = _nested(config, "image_noise", "type")
    if not found or str(noise_type).lower() not in {"gaussian", "gaussian_noise"}:
        raise ManifestValidationError(
            f"expected Gaussian image_noise.type in {path}, got {noise_type!r}"
        )

    target_explicit, target_value = _nested(config, "loss", "pred", "target_view")
    normalized_target = str(target_value).strip().lower() if target_explicit else None
    if branch == "target_view":
        if not target_explicit or normalized_target != "origin":
            raise ManifestValidationError(
                f"target-view run must set loss.pred.target_view=origin: {path}"
            )
        effective_target = "origin"
    else:
        allowed_full_targets = {
            "perturbed",
            "corrupted",
            "noisy",
            "input",
            "observation",
        }
        if target_explicit and normalized_target not in allowed_full_targets:
            raise ManifestValidationError(
                "full-sequence run must omit loss.pred.target_view or explicitly "
                f"select a perturbed target: {path} has {target_value!r}"
            )
        effective_target = normalized_target or "perturbed"

    found_min, std_min = _nested(config, "image_noise", "std_min")
    found_epochs, max_epochs = _nested(config, "max_epochs")
    if not found_epochs:
        found_epochs, max_epochs = _nested(config, "trainer", "max_epochs")

    return {
        "training_seed": TRAINING_SEED,
        "training_noise_type": str(noise_type),
        "training_std_min": (
            _as_float(std_min, label="image_noise.std_min", path=path)
            if found_min
            else None
        ),
        "training_std_max": std_value,
        "target_view_explicit": bool(target_explicit),
        "target_view_config_value": target_value if target_explicit else None,
        "target_view_effective": effective_target,
        "max_epochs": int(float(max_epochs)) if found_epochs else None,
    }


_EPOCH_10 = re.compile(r"(?:^|[_-])epoch[_-]?10(?:[_\-.]|$)", re.IGNORECASE)


def _resolve_epoch10_object_checkpoint(run_dir: Path) -> Path:
    candidates = []
    for path in run_dir.rglob("*.ckpt"):
        relative_parts = path.relative_to(run_dir).parts
        if any(part.startswith("eval_results") for part in relative_parts):
            continue
        if path.is_file() and "object" in path.name.lower() and _EPOCH_10.search(path.name):
            candidates.append(path.resolve())
    candidates = sorted(set(candidates), key=str)
    if len(candidates) != 1:
        rendered = ", ".join(str(path) for path in candidates) or "none"
        raise ManifestValidationError(
            f"expected exactly one epoch-10 object checkpoint in {run_dir}; "
            f"found {len(candidates)}: {rendered}"
        )
    return candidates[0]


_SUCCESS_RATE = re.compile(
    r"['\"]success_rate['\"]\s*:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)
_EPISODE_SUCCESSES = re.compile(
    r"['\"]episode_successes['\"]\s*:\s*"
    r"(?:array\s*\()?\s*(\[[\s\S]*?\])"
    r"(?:\s*,\s*dtype\s*=\s*[^)]+)?\s*\)?\s*,\s*['\"]seeds['\"]"
)


def _parse_metric_file(path: Path, *, expected_eval_seed: int) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ManifestValidationError(f"cannot read eval metric {path}: {exc}") from exc
    if "==== RESULTS ====" not in text:
        raise ManifestValidationError(f"missing RESULTS marker in {path}")
    config_text, results_text = text.rsplit("==== RESULTS ====", 1)

    seed_match = re.search(r"(?m)^seed:\s*(\d+)\s*$", config_text)
    if seed_match is None or int(seed_match.group(1)) != expected_eval_seed:
        observed = seed_match.group(1) if seed_match else None
        raise ManifestValidationError(
            f"eval seed mismatch in {path}: expected {expected_eval_seed}, got {observed}"
        )

    rate_match = _SUCCESS_RATE.search(results_text)
    episodes_match = _EPISODE_SUCCESSES.search(results_text)
    if rate_match is None or episodes_match is None:
        raise ManifestValidationError(
            f"cannot parse success_rate/episode_successes from {path}"
        )
    success_rate = float(rate_match.group(1))
    try:
        episodes = ast.literal_eval(episodes_match.group(1))
    except (SyntaxError, ValueError) as exc:
        raise ManifestValidationError(
            f"cannot parse episode_successes from {path}: {exc}"
        ) from exc
    if not isinstance(episodes, (list, tuple)) or len(episodes) != 100:
        length = len(episodes) if isinstance(episodes, (list, tuple)) else None
        raise ManifestValidationError(
            f"expected 100 episode_successes in {path}, got {length}"
        )
    numeric_episodes = []
    for value in episodes:
        numeric = _as_float(value, label="episode_success", path=path)
        if numeric not in {0.0, 1.0}:
            raise ManifestValidationError(
                f"episode_successes must be binary in {path}, got {value!r}"
            )
        numeric_episodes.append(numeric)
    implied_rate = 100.0 * sum(numeric_episodes) / len(numeric_episodes)
    if not math.isclose(success_rate, implied_rate, rel_tol=0.0, abs_tol=1e-8):
        raise ManifestValidationError(
            f"success_rate disagrees with episode_successes in {path}: "
            f"reported={success_rate}, implied={implied_rate}"
        )
    return {
        "eval_seed": expected_eval_seed,
        "success_rate": success_rate,
        "episode_count": len(numeric_episodes),
        "episode_success_count": int(sum(numeric_episodes)),
    }


def _metric_summary(
    eval_dir: Path, *, metric_name: str, source_condition: str
) -> dict[str, Any]:
    files = []
    values = []
    for eval_seed in EVAL_SEEDS:
        path = eval_dir / f"{source_condition}_seed{eval_seed}_metrics.txt"
        if not path.is_file():
            raise ManifestValidationError(f"missing live eval metric: {path}")
        parsed = _parse_metric_file(path, expected_eval_seed=eval_seed)
        digest = _sha256(path)
        values.append(parsed["success_rate"])
        files.append(
            {
                **parsed,
                "path": str(path.resolve()),
                "sha256": digest,
            }
        )
    return {
        "metric": metric_name,
        "source_condition": source_condition,
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values),
        "n": len(values),
        "values": values,
        "eval_seeds": list(EVAL_SEEDS),
        "source_files": files,
    }


def _legacy_conflicts(
    legacy: Mapping[str, Any] | None,
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if legacy is None:
        return []
    conflicts = []
    for row in rows:
        if row["branch"] != "full_sequence":
            continue
        task = str(row["task"])
        rho_key = str(row["std_key"])
        task_block = legacy.get(task, {})
        entry = task_block.get(rho_key) if isinstance(task_block, Mapping) else None
        if not isinstance(entry, Mapping):
            conflicts.append(
                {
                    "kind": "legacy_missing_row",
                    "task": task,
                    "std_key": rho_key,
                    "authority": "live_eval_results",
                }
            )
            continue
        legacy_metrics = entry.get("metrics", {})
        for metric_name, live_metric in row["metrics"].items():
            legacy_name = "clean" if metric_name == "clean" else metric_name
            old_metric = (
                legacy_metrics.get(legacy_name)
                if isinstance(legacy_metrics, Mapping)
                else None
            )
            if not isinstance(old_metric, Mapping) and metric_name == "clean":
                old_metric = legacy_metrics.get("origin") if isinstance(legacy_metrics, Mapping) else None
            if not isinstance(old_metric, Mapping):
                conflicts.append(
                    {
                        "kind": "legacy_missing_metric",
                        "task": task,
                        "std_key": rho_key,
                        "metric": metric_name,
                        "live_values": live_metric["values"],
                        "authority": "live_eval_results",
                    }
                )
                continue
            try:
                old_values = [float(value) for value in old_metric.get("values", [])]
                old_mean = float(old_metric.get("mean"))
            except (TypeError, ValueError):
                old_values = []
                old_mean = float("nan")
            live_values = [float(value) for value in live_metric["values"]]
            values_match = len(old_values) == len(live_values) and all(
                math.isclose(a, b, rel_tol=0.0, abs_tol=1e-9)
                for a, b in zip(old_values, live_values)
            )
            mean_match = math.isfinite(old_mean) and math.isclose(
                old_mean, float(live_metric["mean"]), rel_tol=0.0, abs_tol=1e-9
            )
            if not values_match or not mean_match:
                conflicts.append(
                    {
                        "kind": "legacy_live_value_mismatch",
                        "task": task,
                        "std_key": rho_key,
                        "metric": metric_name,
                        "legacy_values": old_values,
                        "legacy_mean": old_mean if math.isfinite(old_mean) else None,
                        "live_values": live_values,
                        "live_mean": live_metric["mean"],
                        "authority": "live_eval_results",
                    }
                )
    return conflicts


def _discover_code_commit(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    value = result.stdout.strip()
    return value or "unknown"


def _validate_complete_artifact(payload: Mapping[str, Any]) -> None:
    rows = payload.get("rows")
    pairs = payload.get("matched_pairs")
    if not isinstance(rows, list) or len(rows) != 64:
        raise ManifestValidationError(f"expected 64 rows, got {len(rows or [])}")
    if not isinstance(pairs, list) or len(pairs) != 32:
        raise ManifestValidationError(f"expected 32 matched pairs, got {len(pairs or [])}")

    row_ids = [str(row.get("row_id")) for row in rows]
    if len(set(row_ids)) != len(row_ids):
        raise ManifestValidationError("duplicate row_id in manifest")
    natural_keys = [
        (row.get("task"), row.get("std_key"), row.get("branch")) for row in rows
    ]
    if len(set(natural_keys)) != len(natural_keys):
        raise ManifestValidationError("duplicate task/std/branch row in manifest")
    run_paths = [row.get("path") for row in rows]
    checkpoint_paths = [row.get("checkpoint", {}).get("path") for row in rows]
    if len(set(run_paths)) != len(run_paths):
        raise ManifestValidationError("one run directory resolved to multiple rows")
    if len(set(checkpoint_paths)) != len(checkpoint_paths):
        raise ManifestValidationError("one checkpoint resolved to multiple rows")

    row_index = {row["row_id"]: row for row in rows}
    pair_keys = []
    for pair in pairs:
        pair_key = (pair.get("task"), pair.get("std_key"))
        pair_keys.append(pair_key)
        refs = pair.get("row_ids", {})
        if set(refs) != set(BRANCHES):
            raise ManifestValidationError(f"invalid branch refs for pair {pair_key}")
        for branch in BRANCHES:
            row = row_index.get(refs[branch])
            if row is None or row.get("branch") != branch:
                raise ManifestValidationError(f"invalid row ref for pair {pair_key}/{branch}")
            if (row.get("task"), row.get("std_key")) != pair_key:
                raise ManifestValidationError(f"mismatched row ref for pair {pair_key}/{branch}")
    if len(set(pair_keys)) != len(pair_keys):
        raise ManifestValidationError("duplicate matched pair")

    runner_manifests = payload.get("runner_manifests", {})
    for branch in BRANCHES:
        branch_manifest = runner_manifests.get(branch)
        if not isinstance(branch_manifest, Mapping):
            raise ManifestValidationError(f"missing runner manifest for {branch}")
        for task in TASK_LAYOUT:
            task_block = branch_manifest.get(task)
            if not isinstance(task_block, Mapping) or set(task_block) != set(RHO_KEYS):
                raise ManifestValidationError(
                    f"runner manifest coverage mismatch for {branch}/{task}"
                )


def build_manifest(
    data_root: Path,
    *,
    legacy_manifest: Path | None = DEFAULT_LEGACY_MANIFEST,
    code_commit: str | None = None,
    created_utc: str | None = None,
) -> dict[str, Any]:
    """Build and validate the complete E4 manifest in memory."""
    data_root = data_root.expanduser().resolve()
    if not data_root.is_dir():
        raise ManifestValidationError(f"data root is not a directory: {data_root}")

    legacy = None
    legacy_info = None
    if legacy_manifest is not None:
        legacy_manifest = legacy_manifest.expanduser().resolve()
        if not legacy_manifest.is_file():
            raise ManifestValidationError(f"legacy manifest is missing: {legacy_manifest}")
        legacy = _load_json(legacy_manifest)
        legacy_info = {
            "path": str(legacy_manifest),
            "sha256": _sha256(legacy_manifest),
            "role": "comparison_only_not_authoritative",
        }

    rows: list[dict[str, Any]] = []
    excluded_archives: set[str] = set()
    source_hashes: dict[str, str] = {}
    row_lookup: dict[tuple[str, str, str], dict[str, Any]] = {}

    for task, (task_root_name, prefix) in TASK_LAYOUT.items():
        checkpoint_root = data_root / task_root_name / "ckpt"
        if not checkpoint_root.is_dir():
            raise ManifestValidationError(f"missing checkpoint root: {checkpoint_root}")
        for rho_key in RHO_KEYS:
            for branch in BRANCHES:
                subdir = _run_name(prefix, rho_key, branch)
                run_dir = checkpoint_root / subdir
                if not run_dir.is_dir():
                    raise ManifestValidationError(f"missing expected run directory: {run_dir}")

                config_path = run_dir / "config.yaml"
                if not config_path.is_file():
                    raise ManifestValidationError(f"missing training config: {config_path}")
                config_summary = _validate_training_config(
                    config_path, rho_key=rho_key, branch=branch
                )
                checkpoint_path = _resolve_epoch10_object_checkpoint(run_dir)

                eval_dir = run_dir / "eval_results"
                if not eval_dir.is_dir():
                    raise ManifestValidationError(f"missing live eval_results: {eval_dir}")
                for archive in run_dir.glob("eval_results_old_code_history1_bug*"):
                    if archive.is_dir():
                        excluded_archives.add(str(archive.resolve()))

                metrics = {
                    metric_name: _metric_summary(
                        eval_dir,
                        metric_name=metric_name,
                        source_condition=source_condition,
                    )
                    for metric_name, source_condition in EVAL_CONDITIONS.items()
                }
                config_digest = _sha256(config_path)
                checkpoint_digest = _sha256(checkpoint_path)
                source_hashes[str(config_path.resolve())] = config_digest
                source_hashes[str(checkpoint_path)] = checkpoint_digest
                for metric in metrics.values():
                    for source_file in metric["source_files"]:
                        source_hashes[source_file["path"]] = source_file["sha256"]

                row_id = f"E4:LeWM:seed3072:{task}:{rho_key}:{branch}"
                row = {
                    "row_id": row_id,
                    "status": "ok",
                    "split": "E4",
                    "model_family": "LeWM",
                    "training_seed": TRAINING_SEED,
                    "task": task,
                    "std_key": rho_key,
                    "training_std_max": float(rho_key),
                    "branch": branch,
                    "subdir": subdir,
                    "path": str(run_dir.resolve()),
                    "config": {
                        "path": str(config_path.resolve()),
                        "sha256": config_digest,
                        "validated": config_summary,
                    },
                    "checkpoint": {
                        "path": str(checkpoint_path),
                        "sha256": checkpoint_digest,
                        "epoch": 10,
                        "serialization": "object_ckpt",
                    },
                    "metrics": metrics,
                    "behavior_source": "live_eval_results",
                }
                key = (task, rho_key, branch)
                if key in row_lookup:
                    raise ManifestValidationError(f"duplicate row discovered: {key}")
                row_lookup[key] = row
                rows.append(row)

    conflicts = _legacy_conflicts(legacy, rows)
    matched_pairs = []
    for task in TASK_LAYOUT:
        for rho_key in RHO_KEYS:
            matched_pairs.append(
                {
                    "pair_id": f"E4:LeWM:seed3072:{task}:{rho_key}",
                    "task": task,
                    "std_key": rho_key,
                    "training_std_max": float(rho_key),
                    "row_ids": {
                        branch: row_lookup[(task, rho_key, branch)]["row_id"]
                        for branch in BRANCHES
                    },
                }
            )

    runner_manifests: dict[str, dict[str, Any]] = {}
    for branch in BRANCHES:
        branch_manifest: dict[str, Any] = {
            "_metadata": {
                "schema_version": SCHEMA_VERSION,
                "parent_role": "E4",
                "branch": branch,
                "training_seed": TRAINING_SEED,
                "tasks": list(TASK_LAYOUT),
                "std_keys": list(RHO_KEYS),
            }
        }
        for task in TASK_LAYOUT:
            branch_manifest[task] = {}
            for rho_key in RHO_KEYS:
                row = row_lookup[(task, rho_key, branch)]
                branch_manifest[task][rho_key] = {
                    "path": row["path"],
                    "subdir": row["subdir"],
                    "checkpoint": row["checkpoint"],
                    "config": row["config"],
                    "metrics": row["metrics"],
                    "row_id": row["row_id"],
                    "split": "E4",
                    "branch": branch,
                }
        runner_manifests[branch] = branch_manifest

    script_path = Path(__file__).resolve()
    script_digest = _sha256(script_path)
    source_hashes[str(script_path)] = script_digest
    if legacy_info is not None:
        source_hashes[legacy_info["path"]] = legacy_info["sha256"]

    payload = {
        "_metadata": {
            "schema_version": SCHEMA_VERSION,
            "created_utc": created_utc or _utc_now(),
            "status": "ok",
            "split": "E4",
            "model_family": "LeWM",
            "training_seed": TRAINING_SEED,
            "training_seed_semantics": "one existing LeWM training run per task/std/branch",
            "eval_seeds": list(EVAL_SEEDS),
            "eval_seed_semantics": "repeated closed-loop evaluation of a fixed checkpoint",
            "tasks": list(TASK_LAYOUT),
            "std_keys": list(RHO_KEYS),
            "branches": list(BRANCHES),
            "expected_rows": 64,
            "actual_rows": len(rows),
            "expected_matched_pairs": 32,
            "actual_matched_pairs": len(matched_pairs),
            "behavior_authority": "current live eval_results",
            "legacy_manifest": legacy_info,
            "legacy_conflict_count": len(conflicts),
            "old_history1_archives_excluded": sorted(excluded_archives),
            "old_history1_archive_glob": "eval_results_old_code_history1_bug*",
            "code_commit": code_commit or _discover_code_commit(ROOT),
            "builder": {
                "path": str(script_path),
                "sha256": script_digest,
            },
            "source_paths": {
                "data_root": str(data_root),
                "legacy_manifest": legacy_info["path"] if legacy_info else None,
            },
            "source_hashes": source_hashes,
            "missing_rows": [],
            "errors": [],
            "notes": [
                "No rho=0 target-view checkpoint exists; the paired E4 grid is 0.01--0.08.",
                "Legacy/live disagreements are recorded but never overwrite live eval_results.",
                "Old history_size=1 eval archives are excluded by construction.",
                "runner_manifests are branch-specific views for existing LeWM ATR/SMPR runners.",
            ],
        },
        "rows": rows,
        "matched_pairs": matched_pairs,
        "legacy_live_conflicts": conflicts,
        "runner_manifests": runner_manifests,
    }
    _validate_complete_artifact(payload)
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_runner_manifest_sidecars(
    directory: Path,
    payload: Mapping[str, Any],
) -> dict[str, Path]:
    """Write branch views consumed by the existing ATR/SMPR runners."""
    runner_manifests = payload.get("runner_manifests", {})
    written: dict[str, Path] = {}
    for branch in BRANCHES:
        manifest = runner_manifests.get(branch)
        if not isinstance(manifest, Mapping):
            raise ManifestValidationError(
                f"canonical artifact lacks runner manifest for {branch}"
            )
        path = directory / f"target_view_{branch}_evals_v1.json"
        _write_json(path, manifest)
        written[branch] = path
    return written


def _blocked_payload(
    *, data_root: Path, legacy_manifest: Path | None, error: Exception
) -> dict[str, Any]:
    return {
        "_metadata": {
            "schema_version": SCHEMA_VERSION,
            "created_utc": _utc_now(),
            "status": "blocked",
            "split": "E4",
            "model_family": "LeWM",
            "training_seed": TRAINING_SEED,
            "expected_rows": 64,
            "actual_rows": 0,
            "expected_matched_pairs": 32,
            "actual_matched_pairs": 0,
            "source_paths": {
                "data_root": str(data_root.expanduser()),
                "legacy_manifest": (
                    str(legacy_manifest.expanduser()) if legacy_manifest is not None else None
                ),
            },
            "missing_rows": [],
            "errors": [str(error)],
        },
        "rows": [],
        "matched_pairs": [],
        "legacy_live_conflicts": [],
        "runner_manifests": {},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--runner-manifest-dir",
        type=Path,
        default=DEFAULT_RUNNER_MANIFEST_DIR,
        help="Write branch-specific sidecars for the existing ATR/SMPR runners.",
    )
    parser.add_argument("--legacy-manifest", type=Path, default=DEFAULT_LEGACY_MANIFEST)
    parser.add_argument(
        "--no-legacy-compare",
        action="store_true",
        help="Build without the advisory comparison to the previous seed3072 manifest.",
    )
    parser.add_argument(
        "--code-commit",
        default=None,
        help="Explicit provenance commit; default is the current git HEAD when available.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.data_root is None:
        parser.error(
            "--data-root is required unless PAPER1_DATA_ROOT, DATA_ROOT, "
            "or STABLEWM_HOME is set"
        )
    legacy_manifest = None if args.no_legacy_compare else args.legacy_manifest
    try:
        payload = build_manifest(
            args.data_root,
            legacy_manifest=legacy_manifest,
            code_commit=args.code_commit,
        )
    except ManifestValidationError as exc:
        payload = _blocked_payload(
            data_root=args.data_root,
            legacy_manifest=legacy_manifest,
            error=exc,
        )
        _write_json(args.out, payload)
        print(f"[paper1_target_view_manifest] blocked: {exc}", file=sys.stderr)
        print(f"[paper1_target_view_manifest] wrote blocked report {args.out}")
        return 2

    _write_json(args.out, payload)
    sidecars = _write_runner_manifest_sidecars(
        args.runner_manifest_dir,
        payload,
    )
    metadata = payload["_metadata"]
    print(
        f"[paper1_target_view_manifest] wrote {args.out} "
        f"({metadata['actual_rows']} rows, "
        f"{metadata['actual_matched_pairs']} matched pairs, "
        f"{metadata['legacy_conflict_count']} legacy/live conflicts)"
    )
    print(
        "[paper1_target_view_manifest] runner sidecars: "
        + ", ".join(f"{branch}={path}" for branch, path in sidecars.items())
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
