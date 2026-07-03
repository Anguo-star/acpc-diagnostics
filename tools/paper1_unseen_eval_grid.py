"""Launch Paper 1 eval-only unseen-perturbation grids.

This is a thin orchestration layer around ``run_trainer.sh``. It does not add
new evaluation semantics: each job still runs the existing single-checkpoint
eval primitive with ``skip_train=1`` and an explicit ``ckpt_override``.

Example dry run::

    DATA_ROOT=/home/ag/dataset/ag_data/data/world_model/quentinll \
    python -m tools.paper1_unseen_eval_grid --dry-run

Example eval-only pilot::

    DATA_ROOT=/home/ag/dataset/ag_data/data/world_model/quentinll \
    python -m tools.paper1_unseen_eval_grid --only-missing

Add ``--diagnostics`` only after the closed-loop pilot shows a signal worth
probing; full diagnostics over all checkpoints and corruption families are
substantially more expensive than the eval sweep.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


TASK_META = {
    "PushT": {
        "dataset_name": "pusht",
        "dataset_dir": "lewm-pusht",
        "diagnostic_dataset_name": "pusht_expert_train",
    },
    "TwoRoom": {
        "dataset_name": "tworoom",
        "dataset_dir": "lewm-tworooms",
        "diagnostic_dataset_name": "tworoom",
    },
    "Reacher": {
        "dataset_name": "reacher",
        "dataset_dir": "lewm-reacher",
        "diagnostic_dataset_name": "reacher",
    },
    "Cube": {
        "dataset_name": "cube",
        "dataset_dir": "lewm-cube",
        "diagnostic_dataset_name": "ogbench/cube_single_expert",
    },
}

FAMILY_META = {
    "gaussian_blur": {
        "env_key": "eval_blur_kernel_sizes",
        "default_magnitudes": ("1", "3", "7", "11", "15"),
    },
    "resize": {
        "env_key": "eval_resize_factors",
        "default_magnitudes": ("1.0", "0.75", "0.5", "0.25"),
    },
    "gaussian_noise": {
        "env_key": "eval_corruption_stds",
        "default_magnitudes": ("0.0", "0.03", "0.05", "0.08"),
    },
}

DEFAULT_CANONICAL = "assets/paper1_data/canonical_evals_20260517.json"
DEFAULT_MANIFEST = "assets/paper1_data/unseen_perturbation_pilot_seed3072_manifest.json"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_repo_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return _repo_root() / p


def _default_data_root() -> str | None:
    for key in ("PAPER1_DATA_ROOT", "DATA_ROOT", "STABLEWM_HOME"):
        value = os.environ.get(key)
        if value:
            return value
    return None


def _public_manifest_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            k: _public_manifest_value(v)
            for k, v in value.items()
            if not str(k).startswith("_")
        }
    if isinstance(value, list):
        return [_public_manifest_value(v) for v in value]
    return value


def _load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def _slug(value: str) -> str:
    return value.replace(".", "p").replace("-", "m")


def _normalize_task(task: str, canonical: dict[str, Any]) -> str:
    aliases = {k.lower(): k for k in TASK_META}
    aliases.update({k.lower(): k for k in canonical})
    key = aliases.get(task.lower())
    if key is None or key not in TASK_META or key not in canonical:
        allowed = ", ".join(TASK_META)
        raise ValueError(f"unknown task {task!r}; expected one of: {allowed}")
    return key


def _normalize_std_keys(std_keys: list[str] | None, task: str, canonical: dict[str, Any]) -> list[str]:
    available = sorted(canonical[task].keys(), key=lambda x: float(x))
    if not std_keys:
        return available
    missing = [s for s in std_keys if s not in canonical[task]]
    if missing:
        raise ValueError(f"{task}: std keys not in canonical artifact: {missing}; available={available}")
    return std_keys


def _parse_family_magnitudes(overrides: list[str] | None) -> dict[str, tuple[str, ...]]:
    parsed: dict[str, tuple[str, ...]] = {}
    for item in overrides or []:
        if "=" not in item:
            raise ValueError("--family-magnitudes expects FAMILY=v1,v2,...")
        family, raw = item.split("=", 1)
        family = family.strip()
        if family not in FAMILY_META:
            raise ValueError(f"unknown family in --family-magnitudes: {family}")
        values = tuple(v.strip() for v in raw.replace(" ", ",").split(",") if v.strip())
        if not values:
            raise ValueError(f"empty magnitude list for family {family}")
        parsed[family] = values
    return parsed


def _portable(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _quote_env(env: dict[str, str]) -> str:
    return " ".join(f"{k}={shlex.quote(v)}" for k, v in sorted(env.items()))


def _quote_env_template(env: dict[str, str]) -> str:
    parts = []
    for key, value in sorted(env.items()):
        if value == "$DATA_ROOT" or value.startswith("$DATA_ROOT/"):
            rendered = value
        else:
            rendered = shlex.quote(value)
        parts.append(f"{key}={rendered}")
    return " ".join(parts)


def _diagnostics_dir(result_dir: Path, family: str) -> Path:
    if family == "gaussian_noise":
        return result_dir / "diagnostics"
    return result_dir / f"diagnostics_{family}"


def _eval_summary_has_rows(path: Path) -> bool:
    if not path.is_file():
        return False
    with path.open() as f:
        return sum(1 for _ in f) > 1


def build_jobs(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    canonical_path = _resolve_repo_path(args.canonical)
    canonical = _load_json(canonical_path)
    root = Path(args.root).expanduser()
    family_magnitudes = _parse_family_magnitudes(args.family_magnitudes)

    tasks = [_normalize_task(t, canonical) for t in args.tasks]
    families = args.families
    for family in families:
        if family not in FAMILY_META:
            raise ValueError(f"unknown corruption family {family!r}; expected one of {sorted(FAMILY_META)}")

    output_prefix = args.output_prefix or f"paper1_unseen_s{args.train_seed}"

    jobs: list[dict[str, Any]] = []
    for task in tasks:
        meta = TASK_META[task]
        std_keys = _normalize_std_keys(args.std_keys, task, canonical)
        for std_key in std_keys:
            entry = canonical[task][std_key]
            subdir = entry["subdir"]
            ckpt_rel = Path(meta["dataset_dir"]) / "ckpt" / subdir / f"{subdir}_epoch_{args.epoch}_object.ckpt"
            ckpt_path = root / ckpt_rel
            for family in families:
                output_suffix = f"{output_prefix}_{family}_std{_slug(std_key)}"
                final_output_model_name = f"{meta['dataset_name']}_{output_suffix}"
                result_dir_rel = Path(meta["dataset_dir"]) / "ckpt" / final_output_model_name / "eval_results"
                result_dir = root / result_dir_rel
                diag_dir = _diagnostics_dir(result_dir, family)

                magnitudes = family_magnitudes.get(
                    family,
                    FAMILY_META[family]["default_magnitudes"],
                )
                env = {
                    "STABLEWM_HOME": str(root),
                    "dataset_name": meta["dataset_name"],
                    "trainer_file": args.trainer_file,
                    "config": args.config,
                    "output_model_name": output_suffix,
                    "num_eval": str(args.num_eval),
                    "seed": str(args.train_seed),
                    "skip_train": "1",
                    "post_train_eval_mode": args.post_train_eval_mode,
                    "skip_diagnostics": "0" if args.diagnostics else "1",
                    "eval_corruption_type": family,
                    "diagnostic_corruption_type": family,
                    "eval_corruption_apply_to": str(args.apply_to),
                    "eval_seeds": str(args.eval_seeds),
                    "eval_base_seed": str(args.eval_base_seed),
                    "eval_epoch": str(args.epoch),
                    "ckpt_override": str(ckpt_path),
                    "diagnostic_dataset_name": meta["diagnostic_dataset_name"],
                    FAMILY_META[family]["env_key"]: " ".join(magnitudes),
                }
                if args.eval_gpus:
                    env["eval_gpus"] = args.eval_gpus
                for item in args.extra_env or []:
                    if "=" not in item:
                        raise ValueError("--extra-env expects KEY=VALUE")
                    key, value = item.split("=", 1)
                    env[key] = value
                template_env = dict(env)
                template_env["STABLEWM_HOME"] = "$DATA_ROOT"
                template_env["ckpt_override"] = "$DATA_ROOT/" + ckpt_rel.as_posix()

                eval_summary = result_dir / "eval_summary.csv"
                diagnostics_summary = diag_dir / "diagnostics_summary.json"
                complete = _eval_summary_has_rows(eval_summary) and (
                    not args.diagnostics or diagnostics_summary.is_file()
                )
                jobs.append(
                    {
                        "task": task,
                        "std_key": std_key,
                        "family": family,
                        "subdir": subdir,
                        "checkpoint_rel": ckpt_rel.as_posix(),
                        "checkpoint_exists": ckpt_path.is_file(),
                        "output_model_name_arg": output_suffix,
                        "final_output_model_name": final_output_model_name,
                        "result_dir_rel": result_dir_rel.as_posix(),
                        "eval_summary_rel": _portable(eval_summary, root),
                        "diagnostics_dir_rel": _portable(diag_dir, root),
                        "diagnostics_enabled": bool(args.diagnostics),
                        "magnitudes": list(magnitudes),
                        "apply_to": str(args.apply_to),
                        "eval_seeds": int(args.eval_seeds),
                        "eval_base_seed": int(args.eval_base_seed),
                        "num_eval": int(args.num_eval),
                        "complete": bool(complete),
                        "command_template": (
                            f"{_quote_env_template(template_env)} "
                            f"bash {shlex.quote(str(args.run_trainer))}"
                        ),
                        "_runtime_env": env,
                        "_command": f"{_quote_env(env)} bash {shlex.quote(str(args.run_trainer))}",
                    }
                )

    if args.limit is not None:
        jobs = jobs[: args.limit]

    manifest = {
        "metadata": {
            "schema_version": "paper1-unseen-eval-grid-manifest-1.0",
            "canonical_artifact": str(Path(args.canonical).as_posix()),
            "root": None,
            "root_env_order": ["PAPER1_DATA_ROOT", "DATA_ROOT", "STABLEWM_HOME"],
            "root_note": "Set --root or DATA_ROOT/PAPER1_DATA_ROOT/STABLEWM_HOME on each machine.",
            "train_seed": int(args.train_seed),
            "output_prefix": output_prefix,
            "epoch": int(args.epoch),
            "tasks": tasks,
            "families": list(families),
            "std_keys": args.std_keys if args.std_keys else "all canonical std keys",
            "diagnostics_enabled": bool(args.diagnostics),
            "eval_only_default": not bool(args.diagnostics),
            "eval_seeds": int(args.eval_seeds),
            "eval_base_seed": int(args.eval_base_seed),
            "num_eval": int(args.num_eval),
            "launcher": "tools.paper1_unseen_eval_grid",
        },
        "jobs": jobs,
    }
    return manifest, jobs


def write_manifest(manifest: dict[str, Any], path: str | Path) -> Path:
    out = _resolve_repo_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_public_manifest_value(manifest), indent=2, sort_keys=True))
    return out


def run_jobs(args: argparse.Namespace, jobs: list[dict[str, Any]]) -> int:
    script = Path(args.run_trainer)
    if not script.is_absolute():
        script = _repo_root() / script
    if not script.is_file():
        raise FileNotFoundError(f"run_trainer.sh not found: {script}")

    selected = [j for j in jobs if not (args.only_missing and j["complete"])]
    if args.only_missing:
        print(f"[grid] skipping {len(jobs) - len(selected)} complete jobs")
    failures: list[tuple[dict[str, Any], int]] = []

    for idx, job in enumerate(selected, start=1):
        missing_ckpt = not job["checkpoint_exists"]
        prefix = f"[grid] {idx}/{len(selected)} {job['task']} std={job['std_key']} {job['family']}"
        if missing_ckpt:
            print(f"{prefix}: checkpoint missing: {job['checkpoint_rel']}", file=sys.stderr)
            failures.append((job, 2))
            if args.keep_going:
                continue
            return 2
        print(f"{prefix}: starting")
        env = os.environ.copy()
        env.update(job["_runtime_env"])
        rc = subprocess.run(["bash", str(script)], cwd=_repo_root(), env=env).returncode
        if rc:
            print(f"{prefix}: failed rc={rc}", file=sys.stderr)
            failures.append((job, rc))
            if not args.keep_going:
                return rc
        else:
            print(f"{prefix}: done")

    if failures:
        print("[grid] failures:", file=sys.stderr)
        for job, rc in failures:
            print(f"  rc={rc} {job['task']} std={job['std_key']} {job['family']}", file=sys.stderr)
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=_default_data_root(), help="Runtime data prefix containing lewm-*/.")
    p.add_argument("--canonical", default=DEFAULT_CANONICAL)
    p.add_argument("--manifest-out", default=DEFAULT_MANIFEST)
    p.add_argument("--tasks", nargs="+", default=list(TASK_META))
    p.add_argument("--std-keys", nargs="+", default=None)
    p.add_argument("--families", nargs="+", default=["gaussian_blur", "resize"])
    p.add_argument(
        "--family-magnitudes",
        action="append",
        help="Override magnitudes, e.g. gaussian_blur=1,3,7 or resize=1.0,0.75.",
    )
    p.add_argument("--train-seed", type=int, default=3072)
    p.add_argument("--output-prefix", default=None, help="Output suffix prefix; defaults to paper1_unseen_s<train_seed>.")
    p.add_argument("--epoch", type=int, default=10)
    p.add_argument("--num-eval", type=int, default=300)
    p.add_argument("--eval-seeds", type=int, default=3)
    p.add_argument("--eval-base-seed", type=int, default=42)
    p.add_argument("--eval-gpus", default=None, help="Space-separated GPU ids passed through to run_trainer.sh.")
    p.add_argument("--apply-to", default="1", help="run_trainer.sh eval_corruption_apply_to code; 1 means pixels.")
    p.add_argument("--diagnostics", action="store_true", help="Also run same-family diagnostics.")
    p.add_argument("--post-train-eval-mode", default="full", choices=["full", "origin", "none"])
    p.add_argument("--trainer-file", default="train.py")
    p.add_argument("--config", default="lewm")
    p.add_argument("--run-trainer", default="run_trainer.sh")
    p.add_argument("--extra-env", action="append", help="Extra KEY=VALUE env override for run_trainer.sh.")
    p.add_argument("--dry-run", action="store_true", help="Write/print manifest but do not launch jobs.")
    p.add_argument("--only-missing", action="store_true", help="Skip jobs whose expected outputs already exist.")
    p.add_argument("--keep-going", action="store_true", help="Continue after failed jobs.")
    p.add_argument("--limit", type=int, default=None, help="Debug: keep only the first N jobs.")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if not args.root:
        raise SystemExit("Pass --root or set PAPER1_DATA_ROOT, DATA_ROOT, or STABLEWM_HOME.")
    manifest, jobs = build_jobs(args)
    manifest_path = write_manifest(manifest, args.manifest_out)
    print(f"[grid] wrote manifest: {manifest_path}")
    print(f"[grid] jobs: {len(jobs)}")
    for job in jobs[: min(8, len(jobs))]:
        status = "complete" if job["complete"] else "pending"
        print(f"  {status}: {job['task']} std={job['std_key']} {job['family']}")
        if args.dry_run:
            print(f"    {job['_command']}")
    if len(jobs) > 8:
        print(f"  ... {len(jobs) - 8} more jobs")
    if args.dry_run:
        return
    raise SystemExit(run_jobs(args, jobs))


if __name__ == "__main__":
    main()
