#!/usr/bin/env python3
"""Run canonical horizon-v2 task-grounded SMPR diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from tools import paper1_phase0_acpc as phase0
from tools import paper1_semantic_margin as semantic
from tools.paper1_acpc_metrics import per_anchor_clean_transition_scale


ROOT = Path(__file__).resolve().parents[2]
TASKS = ("TwoRoom", "PushT", "Reacher", "Cube")
STD_KEYS = ("0.0", "0.01", "0.02", "0.03", "0.04", "0.05", "0.06", "0.07", "0.08")
SCHEMA_VERSION = "paper1-smpr-v2-1.0"


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


def _jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        return _jsonable(value.detach().cpu().tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(child) for child in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


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


def _success(entry: Mapping[str, Any], metric: str) -> float:
    return _finite(
        entry.get("metrics", {}).get(metric, {}).get("mean"),
        name=f"behavior metric {metric}",
    )


def _cuda_sync(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(torch.device(device))


def _distribution(values: torch.Tensor) -> dict[str, Any]:
    work = values.detach().float().reshape(-1).cpu()
    if work.numel() == 0:
        return {
            "count": 0,
            "mean": None,
            "q10": None,
            "q50": None,
            "q90": None,
        }
    if not bool(torch.isfinite(work).all()):
        raise ValueError("distribution contains NaN or infinity")
    return {
        "count": int(work.numel()),
        "mean": float(work.mean()),
        "q10": float(torch.quantile(work, 0.10)),
        "q50": float(torch.quantile(work, 0.50)),
        "q90": float(torch.quantile(work, 0.90)),
    }


def _resolve_model(
    entry: Mapping[str, Any],
    model_roots: Sequence[Path],
) -> tuple[Path | None, list[str]]:
    return phase0.resolve_model_file(
        str(entry.get("path", "")),
        str(entry.get("subdir", "")),
        model_roots,
    )


def _set_task_home(entry: Mapping[str, Any]) -> str | None:
    run_path = Path(str(entry.get("path", ""))).expanduser()
    if run_path.parent.name == "ckpt":
        task_home = run_path.parent.parent
        os.environ["STABLEWM_HOME"] = str(task_home)
        return str(task_home)
    return os.environ.get("STABLEWM_HOME")


def _select_task_grounded_pairs(
    *,
    task: str,
    state_final: torch.Tensor,
    local_quantile: float,
) -> dict[str, Any]:
    if state_final.ndim != 2 or state_final.size(0) < 2:
        raise ValueError("state_final must contain at least two flattened states")
    if not 0.0 <= float(local_quantile) <= 1.0:
        raise ValueError("local_quantile must be in [0,1]")
    labels, label_rule, feature = semantic._task_grounded_labels(task, state_final)
    state_dist = torch.cdist(state_final.float(), state_final.float(), p=2)
    feature_dist = torch.cdist(feature.float(), feature.float(), p=2)
    n = state_final.size(0)
    offdiag = ~torch.eye(n, dtype=torch.bool, device=state_final.device)
    threshold = torch.quantile(state_dist[offdiag], float(local_quantile))
    anchor_indices: list[torch.Tensor] = []
    neighbor_indices: list[torch.Tensor] = []
    selected_state_distances: list[torch.Tensor] = []
    selected_feature_distances: list[torch.Tensor] = []
    skipped = 0
    for anchor in range(n):
        valid = (
            offdiag[anchor]
            & (labels != labels[anchor])
            & (state_dist[anchor] <= threshold)
        )
        if not bool(valid.any()):
            skipped += 1
            continue
        candidates = torch.nonzero(valid, as_tuple=False).flatten()
        neighbor = candidates[torch.argmin(state_dist[anchor, candidates])]
        anchor_indices.append(
            torch.tensor(anchor, dtype=torch.long, device=state_final.device)
        )
        neighbor_indices.append(neighbor.to(dtype=torch.long))
        selected_state_distances.append(state_dist[anchor, neighbor])
        selected_feature_distances.append(feature_dist[anchor, neighbor])
    if anchor_indices:
        anchors = torch.stack(anchor_indices)
        neighbors = torch.stack(neighbor_indices)
        state_selected = torch.stack(selected_state_distances)
        feature_selected = torch.stack(selected_feature_distances)
    else:
        anchors = torch.empty(0, dtype=torch.long, device=state_final.device)
        neighbors = torch.empty(0, dtype=torch.long, device=state_final.device)
        state_selected = torch.empty(0, dtype=torch.float32, device=state_final.device)
        feature_selected = torch.empty(
            0,
            dtype=torch.float32,
            device=state_final.device,
        )
    return {
        "pair_anchor_indices": anchors,
        "pair_neighbor_indices": neighbors,
        "selected_state_distance": state_selected,
        "selected_feature_distance": feature_selected,
        "semantic_label_count": int(torch.unique(labels).numel()),
        "semantic_label_rule": label_rule,
        "semantic_distance_threshold": float(threshold.detach().cpu()),
        "semantic_pair_count": int(anchors.numel()),
        "semantic_skipped_anchor_count": int(skipped),
        "semantic_labels": labels,
    }


def _reference_index(path: Path | None) -> tuple[dict[tuple[str, str], float], dict[str, Any]]:
    if path is None:
        return {}, {}
    payload = _load_strict(path)
    if payload.get("metadata", {}).get("schema_version") != "paper1-acpc-horizon-v2-1.0":
        raise ValueError("ATR reference must use paper1-acpc-horizon-v2-1.0")
    index = {
        (str(row["task"]), str(row["std_key"])): _finite(
            row.get("atr_horizon_v2_q90"),
            name="reference ATR",
        )
        for row in payload.get("rows", [])
        if row.get("status") == "ok"
    }
    return index, payload.get("metadata", {})


def _run_row(
    *,
    method: str,
    family_id: str,
    training_seed: int | None,
    task: str,
    std_key: str,
    entry: Mapping[str, Any],
    reference_atr: Mapping[tuple[str, str], float],
    args: argparse.Namespace,
) -> dict[str, Any]:
    wall_started = time.perf_counter()
    runtime_started = time.perf_counter()
    phase0._ensure_runtime_deps()
    runtime_time = time.perf_counter() - runtime_started
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    uses_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    if uses_cuda:
        _cuda_sync(device)
        torch.cuda.reset_peak_memory_stats(torch.device(device))
    model_file, tried = _resolve_model(
        entry,
        [Path(value).expanduser() for value in args.model_root],
    )
    base = {
        "model_family": method,
        "training_family_id": family_id,
        "training_seed": training_seed,
        "task": task,
        "std_key": std_key,
        "training_rho": float(std_key),
        "subdir": entry.get("subdir"),
        "run_path": entry.get("path"),
        "model_file": str(model_file) if model_file else None,
        "model_search_dirs": tried,
        "clean_success": _success(entry, "clean"),
        "pixels_std0.08_success": _success(entry, "pixels_std0.08"),
    }
    if model_file is None:
        return {
            **base,
            "status": "skipped_missing_model",
            "runtime_dependency_setup_time": runtime_time,
            "wall_time_per_row": time.perf_counter() - wall_started,
        }

    model_load_started = time.perf_counter()
    model = phase0.load_model(str(model_file), device)
    _cuda_sync(device)
    model_load_time = time.perf_counter() - model_load_started

    data_started = time.perf_counter()
    task_home = _set_task_home(entry)
    history_size = phase0.infer_history_size(model)
    future_steps = max(args.future_steps, args.rollout_horizon + 1)
    batch = phase0.load_dataset_samples(
        dataset_name=phase0.TASK_DATASETS[task],
        state_key=semantic.SEMANTIC_STATE_KEYS[task],
        n_sequences=args.n_sequences,
        history_size=history_size,
        future_steps=future_steps,
        frameskip=args.frameskip,
        img_size=args.img_size,
        seed=args.anchor_seed,
        device=device,
    )
    draw_seeds = [
        args.anchor_seed + 1009 + 7919 * draw
        for draw in range(args.num_noise_draws)
    ]
    noisy_batches = [
        phase0.make_paired_noisy_batch(
            batch,
            history_size=history_size,
            noise_std=args.noise_std,
            seed=draw_seed,
            corruption_type=args.corruption_type,
            corrupt_goal=False,
        )
        for draw_seed in draw_seeds
    ]
    _cuda_sync(device)
    data_io_time = time.perf_counter() - data_started

    metric_started = time.perf_counter()
    with torch.no_grad():
        spaces = phase0.get_model_spaces(model)
        embedding_space = args.embedding_space or spaces["inference_cost_space"]
        clean_outputs = phase0.encode_sequences(model, phase0._clone_batch(batch))
        noisy_outputs = [
            phase0.encode_sequences(model, phase0._clone_batch(noisy_batch))
            for noisy_batch in noisy_batches
        ]
        clean_emb = phase0.get_embedding_space(
            clean_outputs,
            embedding_space,
        ).detach()
        noisy_embs = [
            phase0.get_embedding_space(outputs, embedding_space).detach()
            for outputs in noisy_outputs
        ]
        act_emb = clean_outputs["act_emb"].detach()
        max_steps = min(
            args.rollout_horizon,
            max(0, act_emb.size(1) - history_size + 1),
            max(0, clean_emb.size(1) - history_size),
        )
        if max_steps < 1:
            raise ValueError("rollout_horizon_actual must be at least one")
        clean_chain = phase0._autoregressive_rollout(
            model,
            clean_emb[:, :history_size],
            act_emb,
            history_size,
            max_steps,
        )
        clean_rollout = clean_chain[:, history_size : history_size + max_steps]
        noisy_rollout = torch.stack(
            [
                phase0._autoregressive_rollout(
                    model,
                    noisy_emb[:, :history_size],
                    act_emb,
                    history_size,
                    max_steps,
                )[:, history_size : history_size + max_steps]
                for noisy_emb in noisy_embs
            ],
            dim=1,
        )
        observed_future = clean_emb[
            :, history_size : history_size + max_steps
        ]
        transition_scale = per_anchor_clean_transition_scale(
            observed_future,
            initial_clean_state=clean_emb[:, history_size - 1],
            transition_quantile=0.50,
        )

        state = batch["state"].float()
        state_index = min(
            state.size(1) - 1,
            history_size + max_steps - 1,
        )
        state_final = state[:, state_index].reshape(state.size(0), -1)
        pair_info = _select_task_grounded_pairs(
            task=task,
            state_final=state_final,
            local_quantile=args.local_quantile,
        )
        pair_anchor_indices = pair_info["pair_anchor_indices"]
        pair_neighbor_indices = pair_info["pair_neighbor_indices"]
        if pair_anchor_indices.numel() < 1:
            raise ValueError("task-grounded pair selection produced zero pairs")
        pair_initial = clean_emb.index_select(
            0,
            pair_neighbor_indices,
        )[:, :history_size]
        pair_actions = act_emb.index_select(0, pair_anchor_indices)
        pair_chain = phase0._autoregressive_rollout(
            model,
            pair_initial,
            pair_actions,
            history_size,
            max_steps,
        )
        different_state_rollout = pair_chain[
            :, history_size : history_size + max_steps
        ]
        metrics = semantic.compute_smpr_v2_from_rollouts(
            clean_rollout=clean_rollout,
            noisy_rollout=noisy_rollout,
            different_state_rollout=different_state_rollout,
            pair_anchor_indices=pair_anchor_indices,
            clean_transition_scale=transition_scale,
            radius_quantile=args.radius_quantile,
            margin_delta_norm=args.margin_delta_norm,
            eps=args.eps,
        )
    _cuda_sync(device)
    smpr_time = time.perf_counter() - metric_started

    tube_radius = _finite(metrics["same_state_tube_radius"], name="SMPR tube radius")
    reference_value = reference_atr.get((task, std_key))
    atr_match = None
    atr_abs_error = None
    if reference_value is not None:
        atr_abs_error = abs(tube_radius - reference_value)
        atr_match = math.isclose(
            tube_radius,
            reference_value,
            rel_tol=args.atr_match_rtol,
            abs_tol=args.atr_match_atol,
        )
        if not atr_match:
            raise ValueError(
                f"SMPR tube radius {tube_radius} does not match canonical ATR "
                f"{reference_value} for {task}/{std_key}"
            )

    peak_memory = (
        int(torch.cuda.max_memory_allocated(torch.device(device)))
        if uses_cuda
        else None
    )
    pair_pass = metrics["pair_pass"]
    return {
        **base,
        "status": "ok",
        "task_home": task_home,
        "history_size": int(history_size),
        "rollout_horizon_actual": int(max_steps),
        "embedding_space": embedding_space,
        "n_sequences": int(args.n_sequences),
        "num_noise_draws": int(args.num_noise_draws),
        "anchor_seed": int(args.anchor_seed),
        "noise_draw_seeds": draw_seeds,
        "noise_draw_seed_rule": "anchor_seed+1009+7919*draw_index",
        "noise_std": float(args.noise_std),
        "corruption_type": args.corruption_type,
        "corrupt_goal": False,
        "pair_rule": "task_grounded_near_boundary_v2",
        "pair_action_policy": "anchor recorded action sequence shared by both clean states",
        "local_state_quantile": float(args.local_quantile),
        "label_binning": "task_specific_median_bins",
        "semantic_state_key": semantic.SEMANTIC_STATE_KEYS[task],
        "semantic_label_rule": pair_info["semantic_label_rule"],
        "semantic_label_count": pair_info["semantic_label_count"],
        "semantic_pair_count": pair_info["semantic_pair_count"],
        "semantic_skipped_anchor_count": pair_info[
            "semantic_skipped_anchor_count"
        ],
        "semantic_skip_rate": pair_info["semantic_skipped_anchor_count"]
        / int(args.n_sequences),
        "semantic_distance_threshold": pair_info[
            "semantic_distance_threshold"
        ],
        "pair_anchor_indices": pair_anchor_indices,
        "pair_neighbor_indices": pair_neighbor_indices,
        "selected_state_distance_distribution": _distribution(
            pair_info["selected_state_distance"]
        ),
        "selected_feature_distance_distribution": _distribution(
            pair_info["selected_feature_distance"]
        ),
        "normalization": (
            "per_anchor_observed_clean_transition_l2_q50_"
            "including_history_future_boundary"
        ),
        "same_state_radius_distribution": _distribution(
            metrics["same_state_radius_per_anchor"]
        ),
        "different_state_distance_distribution": _distribution(
            metrics["different_state_distance_per_pair"]
        ),
        "raw_margin_to_tube_distribution": _distribution(
            metrics["raw_margin_to_tube_per_pair"]
        ),
        "raw_margin_to_anchor_radius_distribution": _distribution(
            metrics["raw_margin_to_anchor_radius_per_pair"]
        ),
        "same_state_radius_per_noise_draw": metrics[
            "same_state_radius_per_noise_draw"
        ],
        "same_state_radius_per_anchor": metrics[
            "same_state_radius_per_anchor"
        ],
        "different_state_distance_per_pair": metrics[
            "different_state_distance_per_pair"
        ],
        "raw_margin_to_tube_per_pair": metrics[
            "raw_margin_to_tube_per_pair"
        ],
        "raw_margin_to_anchor_radius_per_pair": metrics[
            "raw_margin_to_anchor_radius_per_pair"
        ],
        "pair_pass": pair_pass,
        "smpr": _finite(metrics["smpr"], name="SMPR"),
        "same_state_tube_radius": tube_radius,
        "radius_metric": metrics["radius_metric"],
        "horizon_weights": metrics["horizon_weights"],
        "noise_draw_aggregation": metrics["noise_draw_aggregation"],
        "radius_quantile": metrics["radius_quantile"],
        "margin_delta_norm": metrics["margin_delta_norm"],
        "reference_atr_horizon_v2_q90": reference_value,
        "atr_reference_match": atr_match,
        "atr_reference_abs_error": atr_abs_error,
        "runtime_dependency_setup_time": runtime_time,
        "model_load_time": model_load_time,
        "data_io_time": data_io_time,
        "smpr_time": smpr_time,
        "jvp_time": None,
        "fixed_pool_time": None,
        "wall_time_per_row": time.perf_counter() - wall_started,
        "timing_unit": "seconds",
        "peak_gpu_memory": peak_memory,
        "peak_gpu_memory_unit": "bytes",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("LeWM", "PLDM"), required=True)
    parser.add_argument("--family-id", required=True)
    parser.add_argument("--training-seed", type=int, default=None)
    parser.add_argument("--evals", type=Path, required=True)
    parser.add_argument("--reference-atr", type=Path, default=None)
    parser.add_argument("--model-root", action="append", default=[])
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--std-keys", nargs="+", default=list(STD_KEYS))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--n-sequences", type=int, default=100)
    parser.add_argument("--num-noise-draws", type=int, default=5)
    parser.add_argument("--future-steps", type=int, default=9)
    parser.add_argument("--rollout-horizon", type=int, default=8)
    parser.add_argument("--radius-quantile", type=float, default=0.90)
    parser.add_argument("--local-quantile", type=float, default=0.35)
    parser.add_argument("--margin-delta-norm", type=float, default=0.10)
    parser.add_argument("--noise-std", type=float, default=0.08)
    parser.add_argument("--corruption-type", default="gaussian_noise")
    parser.add_argument("--anchor-seed", type=int, default=9101)
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--embedding-space", choices=("raw", "normalized"), default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--atr-match-rtol", type=float, default=1e-5)
    parser.add_argument("--atr-match-atol", type=float, default=1e-6)
    parser.add_argument("--evaluation-seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT
        / "assets"
        / "paper1_data"
        / "smpr_calibration_lewm_seed3072_v2.json",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.n_sequences < 2:
        raise ValueError("n_sequences must be at least two")
    if args.num_noise_draws < 1:
        raise ValueError("num_noise_draws must be positive")
    if len(set(args.evaluation_seeds)) != len(args.evaluation_seeds):
        raise ValueError("evaluation seeds must be distinct")
    evals = _load_strict(args.evals)
    manifest_seed = evals.get("_metadata", {}).get("training_seed")
    if args.method == "LeWM":
        if args.training_seed is None:
            raise ValueError("LeWM requires an explicit training seed")
        if manifest_seed is not None and manifest_seed != args.training_seed:
            raise ValueError("LeWM manifest training seed mismatch")
    reference_atr, reference_meta = _reference_index(args.reference_atr)

    requested = [
        (task, std_key)
        for task in args.tasks
        for std_key in args.std_keys
    ]
    if args.limit is not None:
        requested = requested[: args.limit]
    rows: list[dict[str, Any]] = []
    for index, (task, std_key) in enumerate(requested, start=1):
        print(
            f"[{index}/{len(requested)}] {args.method} {task} std{std_key}",
            flush=True,
        )
        entry = evals.get(task, {}).get(std_key)
        if not isinstance(entry, Mapping):
            rows.append(
                {
                    "model_family": args.method,
                    "training_family_id": args.family_id,
                    "training_seed": args.training_seed,
                    "task": task,
                    "std_key": std_key,
                    "status": "skipped_missing_manifest",
                }
            )
            continue
        try:
            row = _run_row(
                method=args.method,
                family_id=args.family_id,
                training_seed=args.training_seed,
                task=task,
                std_key=std_key,
                entry=entry,
                reference_atr=reference_atr,
                args=args,
            )
        except Exception as exc:  # noqa: BLE001 - preserve all row failures.
            row = {
                "model_family": args.method,
                "training_family_id": args.family_id,
                "training_seed": args.training_seed,
                "task": task,
                "std_key": std_key,
                "status": "error",
                "error": repr(exc),
            }
        rows.append(row)

    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
    script_path = Path(__file__).resolve()
    implementation_paths = {
        "semantic_margin": ROOT / "tools" / "paper1_semantic_margin.py",
        "canonical_metric": ROOT / "tools" / "paper1_acpc_metrics.py",
        "acpc_runtime": ROOT / "tools" / "paper1_phase0_acpc.py",
    }
    source_paths = {"evals": args.evals}
    if args.reference_atr is not None:
        source_paths["reference_atr"] = args.reference_atr
    payload = {
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
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
            "reference_atr_metadata": reference_meta,
            "implementation_paths": {
                name: str(path.relative_to(ROOT))
                for name, path in implementation_paths.items()
            },
            "implementation_hashes": {
                name: _sha256(path)
                for name, path in implementation_paths.items()
            },
            "protocol_hash": None,
            "protocol_hash_status": "calibration_input_pre_freeze",
            "model_family": args.method,
            "training_family_id": args.family_id,
            "training_seed": args.training_seed,
            "training_seed_semantics": (
                "independently trained checkpoint seed when available; family "
                "identity remains distinct across model classes"
            ),
            "evaluation_seeds": list(args.evaluation_seeds),
            "evaluation_seed_semantics": (
                "conditional closed-loop evaluation replicates, not training seeds"
            ),
            "status": (
                "complete"
                if status_counts and set(status_counts) == {"ok"}
                else "partial"
            ),
            "status_counts": status_counts,
            "missing_rows": [
                {
                    "task": row.get("task"),
                    "std_key": row.get("std_key"),
                    "status": row.get("status"),
                }
                for row in rows
                if str(row.get("status", "")).startswith("skipped_")
            ],
            "errors": [
                {
                    "task": row.get("task"),
                    "std_key": row.get("std_key"),
                    "error": row.get("error"),
                }
                for row in rows
                if row.get("status") == "error"
            ],
            "protocol": {
                "radius_metric": "horizon_weighted_stacked_l2_v2",
                "rollout_horizon": int(args.rollout_horizon),
                "horizon_weights": "uniform_1_over_H",
                "radius_quantile": float(args.radius_quantile),
                "normalization": (
                    "per_anchor_observed_clean_transition_l2_q50_"
                    "including_history_future_boundary"
                ),
                "noise_draw_aggregation": (
                    "per_anchor_mean_then_checkpoint_radius_quantile"
                ),
                "pair_rule": "task_grounded_near_boundary_v2",
                "pair_action_policy": (
                    "anchor recorded action sequence shared by both clean states"
                ),
                "local_state_quantile": float(args.local_quantile),
                "margin_delta_normalized": float(args.margin_delta_norm),
                "label_binning": "task_specific_median_bins",
                "n_sequences": int(args.n_sequences),
                "num_noise_draws": int(args.num_noise_draws),
                "anchor_seed": int(args.anchor_seed),
                "noise_draw_seed_rule": (
                    "anchor_seed+1009+7919*draw_index"
                ),
                "corruption": "observation_only_gaussian_std0.08_clean_goal",
            },
        },
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            _jsonable(payload),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.out}")
    print(f"status counts: {status_counts}")
    return 0 if status_counts and set(status_counts) == {"ok"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
