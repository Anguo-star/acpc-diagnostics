#!/usr/bin/env python3
"""Compute behavior-blind encoder/H1/action-control baselines on matched anchors."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping, Sequence

import torch

from tools import paper1_phase0_acpc as phase0
from tools.paper1_acpc_metrics import (
    compute_acpc_horizon_metrics,
    per_anchor_clean_transition_scale,
)


ROOT = Path(__file__).resolve().parents[2]
TASKS = tuple(phase0.TASKS)
FROZEN_PROTOCOL_SHA256 = (
    "edcb801c3da388e673c9b55d706a558aa01da7a281fc151e52e1cda566045a21"
)
SCHEMA_VERSION = "paper1-diagnostic-baseline-raw-1.0"


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


def _cuda_sync(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(torch.device(device))


def _canonical_q90(
    clean: torch.Tensor,
    noisy: torch.Tensor,
    *,
    clean_transition_scale: torch.Tensor,
    eps: float,
) -> float:
    result = compute_acpc_horizon_metrics(
        clean,
        noisy,
        clean_transition_scale=clean_transition_scale,
        noise_draw_dim=1,
        noise_draw_aggregation="mean",
        atr_quantile=0.90,
        transition_quantile=0.50,
        eps=eps,
    )
    return float(result["atr"].detach().cpu())


def _rollout(
    model: Any,
    init: torch.Tensor,
    act_emb: torch.Tensor,
    *,
    history_size: int,
    horizon: int,
) -> torch.Tensor:
    chain = phase0._autoregressive_rollout(
        model,
        init,
        act_emb,
        history_size,
        horizon,
    )
    return chain[:, history_size : history_size + horizon]


def compute_baseline_metrics(
    model: Any,
    clean_outputs: Mapping[str, torch.Tensor],
    noisy_outputs: Sequence[Mapping[str, torch.Tensor]],
    zero_action_outputs: Mapping[str, torch.Tensor],
    *,
    history_size: int,
    rollout_horizon: int,
    embedding_space: str,
    time_shuffle_seed: int,
    eps: float,
) -> dict[str, Any]:
    clean_emb = phase0.get_embedding_space(clean_outputs, embedding_space).detach()
    noisy_embs = [
        phase0.get_embedding_space(output, embedding_space).detach()
        for output in noisy_outputs
    ]
    _require(bool(noisy_embs), "at least one noise draw is required")
    _require(
        all(noisy.shape == clean_emb.shape for noisy in noisy_embs),
        "noisy embedding shapes differ from clean embeddings",
    )
    act_emb = clean_outputs["act_emb"].detach()
    max_steps = min(
        int(rollout_horizon),
        max(0, act_emb.size(1) - history_size + 1),
        max(0, clean_emb.size(1) - history_size),
    )
    _require(max_steps == rollout_horizon, "requested rollout horizon is unavailable")
    observed_clean_future = clean_emb[
        :, history_size : history_size + max_steps
    ]
    scale = per_anchor_clean_transition_scale(
        observed_clean_future,
        initial_clean_state=clean_emb[:, history_size - 1],
        transition_quantile=0.50,
    )
    clean_init = clean_emb[:, :history_size]
    noisy_init = [embedding[:, :history_size] for embedding in noisy_embs]

    def paired_rollouts(action_embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        clean_rollout = _rollout(
            model,
            clean_init,
            action_embedding,
            history_size=history_size,
            horizon=max_steps,
        )
        noisy_rollout = torch.stack(
            [
                _rollout(
                    model,
                    init,
                    action_embedding,
                    history_size=history_size,
                    horizon=max_steps,
                )
                for init in noisy_init
            ],
            dim=1,
        )
        return clean_rollout, noisy_rollout

    correct_clean, correct_noisy = paired_rollouts(act_emb)
    batch_shuffled_clean, batch_shuffled_noisy = paired_rollouts(
        torch.roll(act_emb, shifts=1, dims=0)
    )
    zero_act_emb = zero_action_outputs["act_emb"].detach()
    _require(zero_act_emb.shape == act_emb.shape, "zero-action embedding shape mismatch")
    zero_clean, zero_noisy = paired_rollouts(zero_act_emb)
    generator = torch.Generator(device=act_emb.device).manual_seed(
        int(time_shuffle_seed)
    )
    permutation = torch.randperm(
        act_emb.size(1),
        generator=generator,
        device=act_emb.device,
    )
    time_clean, time_noisy = paired_rollouts(act_emb[:, permutation])

    clean_encoder = clean_init[:, -1:]
    noisy_encoder = torch.stack(
        [value[:, -1:] for value in noisy_init],
        dim=1,
    )
    return {
        "encoder_q90": _canonical_q90(
            clean_encoder,
            noisy_encoder,
            clean_transition_scale=scale,
            eps=eps,
        ),
        "h1_q90": _canonical_q90(
            correct_clean[:, :1],
            correct_noisy[:, :, :1],
            clean_transition_scale=scale,
            eps=eps,
        ),
        "action_shuffled_h8_q90": _canonical_q90(
            batch_shuffled_clean,
            batch_shuffled_noisy,
            clean_transition_scale=scale,
            eps=eps,
        ),
        "action_zeroed_h8_q90": _canonical_q90(
            zero_clean,
            zero_noisy,
            clean_transition_scale=scale,
            eps=eps,
        ),
        "time_shuffled_h8_q90": _canonical_q90(
            time_clean,
            time_noisy,
            clean_transition_scale=scale,
            eps=eps,
        ),
        "atr_h8_q90": _canonical_q90(
            correct_clean,
            correct_noisy,
            clean_transition_scale=scale,
            eps=eps,
        ),
        "clean_transition_scale_min": float(scale.min().detach().cpu()),
        "clean_transition_scale_zero_count": int((scale == 0).sum().detach().cpu()),
        "rollout_horizon_actual": max_steps,
        "encoder_map": "last_history_embedding",
        "h1_map": "first_correct_action_predictive_step",
        "action_shuffled_policy": "cyclic batch permutation of recorded action embeddings",
        "action_zeroed_policy": "raw zero actions passed through the checkpoint action encoder",
        "time_shuffled_policy": "one deterministic temporal permutation shared by clean/noisy histories",
        "normalization": (
            "same per-anchor observed-clean H8 transition q50 including "
            "history/future boundary for every baseline"
        ),
    }


def _reference_index(path: Path) -> tuple[dict[tuple[str, str], Mapping[str, Any]], Mapping[str, Any]]:
    payload = _load(path)
    metadata = payload.get("metadata", {})
    _require(
        metadata.get("schema_version") == "paper1-acpc-horizon-v2-1.0",
        "reference ATR schema mismatch",
    )
    rows = {
        (str(row.get("task")), str(row.get("std_key"))): row
        for row in payload.get("rows", [])
        if row.get("status") == "ok"
    }
    return rows, metadata


def _resolve_checkpoint(
    entry: Mapping[str, Any],
    reference: Mapping[str, Any],
    model_roots: Sequence[Path],
    *,
    name: str,
) -> tuple[Path, str, str]:
    model_file, tried = phase0.resolve_model_file(
        str(entry.get("path", "")),
        str(entry.get("subdir", "")),
        model_roots,
    )
    _require(model_file is not None, f"{name}: checkpoint not found; tried {tried}")
    model_file = model_file.resolve()
    checkpoint_hash = _sha256(model_file)
    expected_hash = reference.get("checkpoint_sha256")
    hash_source = "checkpoint_sha256"
    if expected_hash is None:
        reference_model_file = Path(str(reference.get("model_file", ""))).expanduser()
        _require(
            reference_model_file.is_file(),
            f"{name}: reference ATR lacks checkpoint hash and readable model_file",
        )
        expected_hash = _sha256(reference_model_file)
        hash_source = "reference_model_file_sha256"
    _require(
        checkpoint_hash == expected_hash,
        f"{name}: checkpoint hash mismatch",
    )
    return model_file, checkpoint_hash, hash_source


def run_row(
    *,
    task: str,
    std_key: str,
    entry: Mapping[str, Any],
    reference: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    started = time.perf_counter()
    phase0._ensure_runtime_deps()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_roots = [Path(value).expanduser() for value in args.model_root]
    name = f"{args.split_name}/{args.family_id}/{task}/{std_key}"
    model_file, checkpoint_hash, reference_checkpoint_hash_source = _resolve_checkpoint(
        entry,
        reference,
        model_roots,
        name=name,
    )
    model = phase0.load_model(str(model_file), device)
    history_size = phase0.infer_history_size(model)
    future_steps = max(args.future_steps, args.rollout_horizon + 1)
    batch = phase0.load_dataset_samples(
        dataset_name=phase0.TASK_DATASETS[task],
        state_key=args.state_key,
        n_sequences=args.n_sequences,
        history_size=history_size,
        future_steps=future_steps,
        frameskip=args.frameskip,
        img_size=args.img_size,
        seed=args.anchor_seed,
        device=device,
    )
    draw_seeds = [
        args.anchor_seed + 1009 + 7919 * index
        for index in range(args.num_noise_draws)
    ]
    noisy_batches = [
        phase0.make_paired_noisy_batch(
            batch,
            history_size=history_size,
            noise_std=args.noise_std,
            seed=seed,
            corruption_type=args.corruption_type,
            corrupt_goal=False,
        )
        for seed in draw_seeds
    ]
    zero_batch = phase0._clone_batch(batch)
    zero_batch["action"] = torch.zeros_like(zero_batch["action"])
    with torch.no_grad():
        spaces = phase0.get_model_spaces(model)
        embedding_space = args.embedding_space or spaces["inference_cost_space"]
        clean_outputs = phase0.encode_sequences(model, phase0._clone_batch(batch))
        noisy_outputs = [
            phase0.encode_sequences(model, phase0._clone_batch(value))
            for value in noisy_batches
        ]
        zero_action_outputs = phase0.encode_sequences(model, zero_batch)
        metrics = compute_baseline_metrics(
            model,
            clean_outputs,
            noisy_outputs,
            zero_action_outputs,
            history_size=history_size,
            rollout_horizon=args.rollout_horizon,
            embedding_space=embedding_space,
            time_shuffle_seed=args.anchor_seed + 3001,
            eps=args.eps,
        )
    _cuda_sync(device)
    reference_atr = _finite(
        reference.get("atr_horizon_v2_q90"),
        name=f"{name}/reference-ATR",
    )
    _require(
        math.isclose(
            metrics["atr_h8_q90"],
            reference_atr,
            rel_tol=args.atr_match_rtol,
            abs_tol=args.atr_match_atol,
        ),
        f"{name}: recomputed correct-action ATR differs from reference",
    )
    return {
        "status": "ok",
        "model_family": args.method,
        "training_family_id": args.family_id,
        "training_seed": args.training_seed,
        "training_seed_semantics": args.training_seed_semantics,
        "task": task,
        "std_key": std_key,
        "training_rho": float(std_key),
        "stressor_family": args.stressor_family,
        "stressor_severity": args.stressor_severity,
        "branch": args.branch,
        "split_name": args.split_name,
        "model_file": str(model_file),
        "checkpoint_sha256": checkpoint_hash,
        "reference_checkpoint_hash_source": reference_checkpoint_hash_source,
        "reference_atr_h8_q90": reference_atr,
        "reference_atr_match": True,
        "n_sequences": args.n_sequences,
        "num_noise_draws": args.num_noise_draws,
        "anchor_seed": args.anchor_seed,
        "noise_draw_seeds": draw_seeds,
        "corruption_type": args.corruption_type,
        "corrupt_goal": False,
        "embedding_space": embedding_space,
        "wall_time_per_row": time.perf_counter() - started,
        **metrics,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("LeWM", "PLDM"), required=True)
    parser.add_argument("--family-id", required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--training-seed-semantics", required=True)
    parser.add_argument("--split-name", required=True)
    parser.add_argument("--stressor-family", required=True)
    parser.add_argument("--stressor-severity", type=float, required=True)
    parser.add_argument("--branch", default="")
    parser.add_argument("--evals", type=Path, required=True)
    parser.add_argument("--reference-atr", type=Path, required=True)
    parser.add_argument("--model-root", action="append", default=[])
    parser.add_argument("--tasks", nargs="+", choices=TASKS, required=True)
    parser.add_argument("--std-keys", nargs="+", required=True)
    parser.add_argument("--n-sequences", type=int, default=100)
    parser.add_argument("--num-noise-draws", type=int, default=5)
    parser.add_argument("--future-steps", type=int, default=9)
    parser.add_argument("--rollout-horizon", type=int, default=8)
    parser.add_argument("--noise-std", type=float, default=0.08)
    parser.add_argument("--corruption-type", default="gaussian_noise")
    parser.add_argument("--anchor-seed", type=int, default=9101)
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--state-key", default=None)
    parser.add_argument("--embedding-space", choices=("raw", "normalized"), default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--atr-match-rtol", type=float, default=1e-5)
    parser.add_argument("--atr-match-atol", type=float, default=1e-6)
    parser.add_argument("--protocol", type=Path, default=ROOT / "paper1/config/frozen_diagnostic_protocol_v1.json")
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    _require(_sha256(args.protocol) == FROZEN_PROTOCOL_SHA256, "frozen protocol hash mismatch")
    evals = _load(args.evals)
    manifest_seed = evals.get("_metadata", {}).get("training_seed")
    _require(manifest_seed in (None, args.training_seed), "manifest training seed mismatch")
    reference, reference_metadata = _reference_index(args.reference_atr)
    expected = {(task, std_key) for task in args.tasks for std_key in args.std_keys}
    _require(expected.issubset(set(reference)), "reference ATR lacks requested rows")
    rows = []
    for index, (task, std_key) in enumerate(sorted(expected, key=lambda value: (TASKS.index(value[0]), float(value[1]))), start=1):
        print(f"[{index}/{len(expected)}] {args.family_id} {task} std{std_key}", flush=True)
        entry = evals.get(task, {}).get(std_key)
        if not isinstance(entry, Mapping):
            rows.append(
                {
                    "status": "skipped_missing_manifest",
                    "model_family": args.method,
                    "training_family_id": args.family_id,
                    "training_seed": args.training_seed,
                    "task": task,
                    "std_key": std_key,
                }
            )
            continue
        try:
            row = run_row(
                task=task,
                std_key=std_key,
                entry=entry,
                reference=reference[(task, std_key)],
                args=args,
            )
        except Exception as exc:  # noqa: BLE001
            row = {
                "status": "error",
                "model_family": args.method,
                "training_family_id": args.family_id,
                "training_seed": args.training_seed,
                "task": task,
                "std_key": std_key,
                "error": repr(exc),
            }
        rows.append(row)
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    script_path = Path(__file__).resolve()
    metric_path = ROOT / "tools/paper1_acpc_metrics.py"
    phase0_path = ROOT / "tools/paper1_phase0_acpc.py"
    source_paths = {
        "evals": args.evals,
        "reference_atr": args.reference_atr,
        "protocol": args.protocol,
    }
    payload = {
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "code_commit": _git_commit(),
            "status": "complete" if counts and set(counts) == {"ok"} else "partial",
            "status_counts": counts,
            "missing_rows": [
                {"task": row.get("task"), "std_key": row.get("std_key")}
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
            "behavior_blind": True,
            "threshold_search_available": False,
            "protocol_hash": FROZEN_PROTOCOL_SHA256,
            "model_family": args.method,
            "training_family_id": args.family_id,
            "training_seed": args.training_seed,
            "training_seed_semantics": args.training_seed_semantics,
            "tasks": list(args.tasks),
            "std_keys": list(args.std_keys),
            "split_name": args.split_name,
            "stressor_family": args.stressor_family,
            "stressor_severity": args.stressor_severity,
            "branch": args.branch,
            "source_paths": {name: str(path) for name, path in source_paths.items()},
            "source_hashes": {name: _sha256(path) for name, path in source_paths.items()},
            "reference_atr_metadata": reference_metadata,
            "implementation_paths": {
                "runner": str(script_path.relative_to(ROOT)),
                "canonical_metric": str(metric_path.relative_to(ROOT)),
                "phase0_runtime": str(phase0_path.relative_to(ROOT)),
            },
            "implementation_hashes": {
                "runner": _sha256(script_path),
                "canonical_metric": _sha256(metric_path),
                "phase0_runtime": _sha256(phase0_path),
            },
            "protocol": {
                "n_sequences": args.n_sequences,
                "num_noise_draws": args.num_noise_draws,
                "rollout_horizon": args.rollout_horizon,
                "radius_quantile": 0.9,
                "anchor_seed": args.anchor_seed,
                "same_anchor_draw_normalization_for_all_baselines": True,
            },
        },
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.out}; status counts: {counts}")
    return 0 if counts and set(counts) == {"ok"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
