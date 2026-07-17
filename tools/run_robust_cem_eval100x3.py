#!/usr/bin/env python3
"""Run the paper-standard Robust CEM eval replication.

This is intentionally small and conservative:

* frozen origin LeWM checkpoints only;
* one eval process at a time;
* world.num_envs=1 to avoid high-concurrency hangs;
* per-job timeout and resume from completed metrics files.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/opt/workspace/explorer-env/dataset/ag_data/data/world_model/quentinll")
OUT_JSON = REPO_ROOT / "assets/paper1_data/robust_cem_eval100x3_20260705.json"
PARTIAL_JSON = REPO_ROOT / "assets/paper1_data/robust_cem_eval100x3_20260705_partial.json"
LOG_DIR = REPO_ROOT / "assets/paper1_data/robust_cem_eval100x3_logs_20260705"
SEEDS = (42, 43, 44)


def configure_output_paths(stem: str) -> None:
    global OUT_JSON, PARTIAL_JSON, LOG_DIR
    OUT_JSON = REPO_ROOT / f"assets/paper1_data/{stem}.json"
    PARTIAL_JSON = REPO_ROOT / f"assets/paper1_data/{stem}_partial.json"
    LOG_DIR = REPO_ROOT / f"assets/paper1_data/{stem}_logs"


@dataclass(frozen=True)
class TaskSpec:
    key: str
    label: str
    config: str
    policy: str


@dataclass(frozen=True)
class PlannerSpec:
    key: str
    args: tuple[str, ...]


TASKS = {
    "tworoom": TaskSpec(
        key="tworoom",
        label="TwoRoom",
        config="tworoom.yaml",
        policy="lewm-tworooms/ckpt/tworoom_lewm_baseline_seed3073",
    ),
    "pusht": TaskSpec(
        key="pusht",
        label="PushT",
        config="pusht.yaml",
        policy="lewm-pusht/ckpt/pusht_lewm_baseline_seed3073",
    ),
    "reacher": TaskSpec(
        key="reacher",
        label="Reacher",
        config="reacher.yaml",
        policy="lewm-reacher/ckpt/reacher_lewm_baseline_seed3073",
    ),
    "cube": TaskSpec(
        key="cube",
        label="Cube",
        config="cube.yaml",
        policy="lewm-cube/ckpt/cube_lewm_baseline_seed3073",
    ),
}

PLANNERS = {
    "cem48_n4": PlannerSpec(
        key="cem48_n4",
        args=("solver=cem", "solver.num_samples=48", "solver.n_steps=4", "solver.topk=8"),
    ),
    "cem192_n4_compute": PlannerSpec(
        key="cem192_n4_compute",
        args=("solver=cem", "solver.num_samples=192", "solver.n_steps=4", "solver.topk=24"),
    ),
    "rcem_inner_meanstd": PlannerSpec(
        key="rcem_inner_meanstd",
        args=(
            "solver=robust_cem",
            "solver.num_samples=48",
            "solver.n_steps=4",
            "solver.topk=8",
            "solver.num_views=4",
            "solver.include_identity=true",
            "solver.view_type=gaussian_noise",
            "solver.view_std=0.04",
            "solver.perturb_pixels=true",
            "solver.perturb_goal=false",
            "solver.score_mode=mean_std",
            "solver.beta=0.5",
            "solver.robust_rescore=all",
            "solver.final_output_mode=selected",
        ),
    ),
    "rcem_inner_rankvote_elitemean": PlannerSpec(
        key="rcem_inner_rankvote_elitemean",
        args=(
            "solver=robust_cem",
            "solver.num_samples=48",
            "solver.n_steps=4",
            "solver.topk=8",
            "solver.num_views=4",
            "solver.include_identity=true",
            "solver.view_type=gaussian_noise",
            "solver.view_std=0.04",
            "solver.perturb_pixels=true",
            "solver.perturb_goal=false",
            "solver.score_mode=rank_vote",
            "solver.robust_rescore=all",
            "solver.final_output_mode=elite_mean",
        ),
    ),
    "rcem_final_rankmeanstd": PlannerSpec(
        key="rcem_final_rankmeanstd",
        args=("solver=robust_cem_rank",),
    ),
    "rcem_final_baserankstd_b025": PlannerSpec(
        key="rcem_final_baserankstd_b025",
        args=("solver=robust_cem_rank", "solver.score_mode=base_rank_std", "solver.beta=0.25"),
    ),
    "rcem_final_baserankstd_b05": PlannerSpec(
        key="rcem_final_baserankstd_b05",
        args=("solver=robust_cem_rank", "solver.score_mode=base_rank_std", "solver.beta=0.5"),
    ),
    "rcem_final_rankmeanstd_v02": PlannerSpec(
        key="rcem_final_rankmeanstd_v02",
        args=("solver=robust_cem_rank", "solver.view_std=0.02"),
    ),
    "rcem_final_gated_sg1_rs1": PlannerSpec(
        key="rcem_final_gated_sg1_rs1",
        args=("solver=robust_cem_gated",),
    ),
    "rcem_final_gated_sg05_rs1": PlannerSpec(
        key="rcem_final_gated_sg05_rs1",
        args=("solver=robust_cem_gated", "solver.final_switch_min_score_gain=0.5"),
    ),
    "rcem_final_gated_sg15_rs1": PlannerSpec(
        key="rcem_final_gated_sg15_rs1",
        args=("solver=robust_cem_gated", "solver.final_switch_min_score_gain=1.5"),
    ),
    "rcem_final_rankmean": PlannerSpec(
        key="rcem_final_rankmean",
        args=("solver=robust_cem_rank", "solver.score_mode=rank_mean"),
    ),
    "rcem_final_mean": PlannerSpec(
        key="rcem_final_mean",
        args=("solver=robust_cem_rank", "solver.score_mode=mean"),
    ),
    "rcem_final_meanstd_v02": PlannerSpec(
        key="rcem_final_meanstd_v02",
        args=("solver=robust_cem_rank", "solver.score_mode=mean_std", "solver.view_std=0.02"),
    ),
    "rcem_final_rankvote": PlannerSpec(
        key="rcem_final_rankvote",
        args=("solver=robust_cem_vote",),
    ),
    "rcem_final_rankmeanstd_elitemean": PlannerSpec(
        key="rcem_final_rankmeanstd_elitemean",
        args=("solver=robust_cem_elite_mean",),
    ),
    "rcem_final_rankvote_elitemean": PlannerSpec(
        key="rcem_final_rankvote_elitemean",
        args=("solver=robust_cem_vote", "solver.final_output_mode=elite_mean"),
    ),
    "rcem_final_mean_elitemean": PlannerSpec(
        key="rcem_final_mean_elitemean",
        args=("solver=robust_cem_elite_mean", "solver.score_mode=mean"),
    ),
    "rcem_final_rankvote_elitemean_v8": PlannerSpec(
        key="rcem_final_rankvote_elitemean_v8",
        args=("solver=robust_cem_vote", "solver.final_output_mode=elite_mean", "solver.num_views=8"),
    ),
    "rcem_final_rankvote_elitemean_pool32": PlannerSpec(
        key="rcem_final_rankvote_elitemean_pool32",
        args=("solver=robust_cem_vote", "solver.final_output_mode=elite_mean", "solver.elite_rescore_multiplier=4"),
    ),
    "rcem_final_rankvote_elitemean_v8_pool32": PlannerSpec(
        key="rcem_final_rankvote_elitemean_v8_pool32",
        args=("solver=robust_cem_vote", "solver.final_output_mode=elite_mean", "solver.num_views=8", "solver.elite_rescore_multiplier=4"),
    ),
}


def _metrics_path(task: TaskSpec, output_filename: str) -> Path:
    return DATA_ROOT / Path(task.policy).parent / output_filename


def _extract_last_metrics_block(text: str) -> str | None:
    marker = "metrics:"
    idx = text.rfind(marker)
    if idx < 0:
        return None
    start = idx + len(marker)
    end = text.find("\nevaluation_time:", start)
    if end < 0:
        return None
    return text[start:end].strip()


def parse_metrics_file(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    text = path.read_text(errors="replace")
    block = _extract_last_metrics_block(text)
    if block is None:
        return None

    rate_match = re.search(r"'success_rate':\s*([0-9.eE+-]+)", block)
    if not rate_match:
        return None
    successes = [token == "True" for token in re.findall(r"\b(True|False)\b", block)]
    if not successes:
        return None

    robust_stats: dict[str, Any] | None = None
    robust_idx = text.rfind("==== ROBUST_CEM ====")
    if robust_idx >= 0:
        after = text[robust_idx:].splitlines()
        for line in after[1:]:
            line = line.strip()
            if line.startswith("{"):
                try:
                    robust_stats = json.loads(line)
                except json.JSONDecodeError:
                    robust_stats = None
                break

    eval_time = None
    block_idx = text.rfind(block)
    tail = text[block_idx:] if block_idx >= 0 else text
    eval_match = re.search(r"evaluation_time:\s*([0-9.eE+-]+)\s+seconds", tail)
    if eval_match:
        eval_time = float(eval_match.group(1))

    return {
        "success_rate": float(rate_match.group(1)),
        "success_count": int(sum(successes)),
        "n": len(successes),
        "successes": successes,
        "evaluation_time_sec": eval_time,
        "robust": robust_stats,
    }


def build_jobs(args: argparse.Namespace) -> list[dict[str, Any]]:
    jobs = []
    for task_key in args.tasks:
        task = TASKS[task_key]
        for seed in args.seeds:
            for planner_key in args.planners:
                planner = PLANNERS[planner_key]
                condition = "gaussian_std0.08"
                suffix = f"{task.key}_s3073_seed{seed}_gauss008_{planner.key}_eval{args.num_eval}"
                output_filename = f"robust_cem_eval100x3_{suffix}.txt"
                log_file = LOG_DIR / f"robust_cem_eval100x3_{suffix}.log"
                metrics_path = _metrics_path(task, output_filename)
                cmd = [
                    sys.executable,
                    "eval.py",
                    f"--config-name={task.config}",
                    f"cache_dir={DATA_ROOT}",
                    f"policy={task.policy}",
                    f"seed={seed}",
                    f"eval.num_eval={args.num_eval}",
                    "world.num_envs=1",
                    f"eval.eval_budget={args.eval_budget}",
                    "eval.corruption.std=0.08",
                    f"output.filename={output_filename}",
                    *planner.args,
                ]
                jobs.append(
                    {
                        "task": task.label,
                        "task_key": task.key,
                        "checkpoint": "baseline_seed3073",
                        "condition": condition,
                        "seed": seed,
                        "planner": planner.key,
                        "command": " ".join(cmd),
                        "_cmd": cmd,
                        "log_file": str(log_file),
                        "output_filename": output_filename,
                        "metrics_file": str(metrics_path),
                    }
                )
    return jobs


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    aggregates: dict[str, Any] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = (row["task"], row["condition"], row["planner"])
        block = aggregates.setdefault(
            "|".join(key),
            {
                "task": row["task"],
                "condition": row["condition"],
                "planner": row["planner"],
                "seeds": [],
                "success_count": 0,
                "n": 0,
                "per_seed_success_rate": [],
            },
        )
        block["seeds"].append(row["seed"])
        block["success_count"] += int(row["success_count"])
        block["n"] += int(row["n"])
        block["per_seed_success_rate"].append(float(row["success_rate"]))

    for block in aggregates.values():
        n = int(block["n"])
        block["success_rate"] = float(block["success_count"]) / n * 100.0 if n else None
        values = block["per_seed_success_rate"]
        if len(values) >= 2:
            mean = sum(values) / len(values)
            var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
            block["per_seed_mean"] = mean
            block["per_seed_std"] = var ** 0.5
        elif values:
            block["per_seed_mean"] = values[0]
            block["per_seed_std"] = 0.0
    return aggregates


def write_artifact(
    rows: list[dict[str, Any]],
    path: Path,
    *,
    complete: bool,
    args: argparse.Namespace | None = None,
) -> None:
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    artifact = {
        "schema_version": "robust-cem-eval100x3-20260705-v1",
        "goal": "Frozen origin LeWM checkpoint, planner-side robust CEM, Gaussian std=0.08 eval.",
        "protocol": {
            "tasks": [TASKS[t].label for t in (args.tasks if args else TASKS)],
            "checkpoint": "baseline_seed3073",
            "eval_seeds": list(args.seeds if args else SEEDS),
            "num_eval_per_seed": int(args.num_eval if args else 100),
            "world_num_envs": 1,
            "eval_budget": int(args.eval_budget if args else 25),
            "condition": "pixels Gaussian std=0.08",
            "planners": list(args.planners if args else PLANNERS),
            "complete": complete,
        },
        "aggregates": summarize(ok_rows),
        "rows": rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True))


def run_jobs(args: argparse.Namespace) -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(args)
    rows: list[dict[str, Any]] = []
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["STABLEWM_HOME"] = str(DATA_ROOT)
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    env.setdefault("HYDRA_FULL_ERROR", "1")

    for idx, job in enumerate(jobs, start=1):
        metrics_file = Path(job["metrics_file"])
        parsed = parse_metrics_file(metrics_file)
        if args.resume and parsed is not None and parsed["n"] == args.num_eval:
            row = {k: v for k, v in job.items() if not k.startswith("_")}
            row.update(parsed)
            row.update({"status": "ok", "returncode": 0, "resumed": True, "wall_time_sec": 0.0})
            rows.append(row)
            print(f"[{idx}/{len(jobs)}] resume {job['task']} seed={job['seed']} {job['planner']}: {parsed['success_count']}/{parsed['n']}")
            write_artifact(rows, PARTIAL_JSON, complete=False, args=args)
            continue

        print(f"[{idx}/{len(jobs)}] start {job['task']} seed={job['seed']} {job['planner']}")
        start = time.time()
        log_path = Path(job["log_file"])
        with log_path.open("w") as log:
            log.write("$ " + " ".join(job["_cmd"]) + "\n\n")
            log.flush()
            try:
                proc = subprocess.run(
                    job["_cmd"],
                    cwd=REPO_ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=args.timeout_sec,
                )
                returncode = proc.returncode
                status = "ok" if returncode == 0 else "failed"
            except subprocess.TimeoutExpired:
                returncode = 124
                status = "timeout"

        wall = time.time() - start
        parsed = parse_metrics_file(metrics_file)
        row = {k: v for k, v in job.items() if not k.startswith("_")}
        row.update({"status": status, "returncode": returncode, "wall_time_sec": wall})
        if parsed is not None:
            row.update(parsed)
            if parsed["n"] != args.num_eval:
                row["status"] = "partial_metrics"
        elif status == "ok":
            row["status"] = "no_metrics"
        rows.append(row)

        if row.get("status") == "ok":
            print(
                f"[{idx}/{len(jobs)}] ok {job['task']} seed={job['seed']} {job['planner']}: "
                f"{row['success_count']}/{row['n']} ({row['success_rate']:.1f}%) in {wall:.1f}s"
            )
        else:
            print(
                f"[{idx}/{len(jobs)}] {row['status']} {job['task']} seed={job['seed']} "
                f"{job['planner']} rc={returncode} in {wall:.1f}s"
            )
            if not args.keep_going:
                write_artifact(rows, PARTIAL_JSON, complete=False, args=args)
                return returncode or 1
        write_artifact(rows, PARTIAL_JSON, complete=False, args=args)

    write_artifact(rows, OUT_JSON, complete=True, args=args)
    write_artifact(rows, PARTIAL_JSON, complete=True, args=args)
    return 0 if all(r.get("status") == "ok" for r in rows) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=sorted(TASKS), default=list(TASKS))
    parser.add_argument("--planners", nargs="+", choices=sorted(PLANNERS), default=list(PLANNERS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--num-eval", type=int, default=100)
    parser.add_argument("--eval-budget", type=int, default=25)
    parser.add_argument("--timeout-sec", type=int, default=1200)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--output-stem", default="robust_cem_eval100x3_20260705")
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--keep-going", action="store_true")
    parser.set_defaults(resume=True)
    args = parser.parse_args()
    configure_output_paths(args.output_stem)
    return run_jobs(args)


if __name__ == "__main__":
    raise SystemExit(main())
