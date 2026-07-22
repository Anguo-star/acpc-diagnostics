"""Launch Paper 1 eval-only unseen-perturbation grids.

This is a thin orchestration layer around ``run_trainer.sh``. It does not add
new evaluation semantics: each job still runs the existing single-checkpoint
eval primitive with ``skip_train=1`` and an explicit ``ckpt_override``.

Example dry run::

    DATA_ROOT=/path/to/world_model \
    python -m tools.paper1_unseen_eval_grid --dry-run

Example eval-only pilot::

    DATA_ROOT=/path/to/world_model \
    python -m tools.paper1_unseen_eval_grid --only-missing

Add ``--diagnostics`` only after the closed-loop pilot shows a signal worth
probing; full diagnostics over all checkpoints and corruption families are
substantially more expensive than the eval sweep.
"""

from __future__ import annotations

import argparse
import hashlib
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def _entry_path(entry: dict[str, Any], root: Path) -> Path | None:
    raw = entry.get("path")
    if raw is None or not str(raw).strip():
        return None
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _resolve_checkpoint(
    *,
    entry: dict[str, Any],
    root: Path,
    dataset_dir: str,
    subdir: str,
    epoch: int,
) -> tuple[Path, str]:
    """Resolve one checkpoint without guessing when candidates disagree.

    New canonical manifests bind a checkpoint explicitly with ``model_file``.
    That binding is authoritative: an invalid explicit path is an error and
    never falls back to filename discovery. Legacy manifests may omit the
    field; for those, exactly one matching epoch checkpoint must be present.
    """

    run_path = _entry_path(entry, root)
    if "model_file" in entry:
        raw_model = entry.get("model_file")
        if raw_model is None or not str(raw_model).strip():
            raise ValueError(f"{subdir}: canonical model_file is empty")
        if run_path is None:
            raise ValueError(f"{subdir}: canonical model_file requires entry path")

        model_file = Path(str(raw_model)).expanduser()
        if not model_file.is_absolute():
            # A bare filename is naturally relative to the entry run path;
            # a longer portable path is relative to the runtime data root.
            model_file = (
                run_path / model_file
                if model_file.parent == Path(".")
                else root / model_file
            )
        model_file = model_file.resolve()
        if not _is_within(model_file, run_path):
            raise ValueError(
                f"{subdir}: canonical model_file escapes entry path: "
                f"model_file={model_file}, path={run_path}"
            )
        if not model_file.is_file():
            raise FileNotFoundError(
                f"{subdir}: canonical model_file does not exist: {model_file}"
            )
        return model_file, "canonical_model_file"

    candidate_dirs = [(root / dataset_dir / "ckpt" / subdir).resolve()]
    if run_path is not None:
        candidate_dirs.append(run_path)
    candidate_dirs = list(dict.fromkeys(candidate_dirs))

    matches: list[Path] = []
    pattern = f"*epoch_{epoch}_object.ckpt"
    for directory in candidate_dirs:
        if not directory.exists():
            continue
        if not directory.is_dir():
            raise ValueError(
                f"{subdir}: legacy checkpoint path is not a directory: {directory}"
            )
        directory_matches = sorted(
            path.resolve() for path in directory.glob(pattern) if path.is_file()
        )
        if any(not _is_within(path, directory) for path in directory_matches):
            raise ValueError(
                f"{subdir}: legacy checkpoint symlink escapes run path: {directory}"
            )
        if len(directory_matches) > 1:
            rendered = ", ".join(str(path) for path in directory_matches)
            raise ValueError(f"{subdir}: ambiguous legacy checkpoints: {rendered}")
        matches.extend(directory_matches)

    matches = list(dict.fromkeys(matches))
    if len(matches) > 1:
        rendered = ", ".join(str(path) for path in matches)
        raise ValueError(
            f"{subdir}: conflicting legacy checkpoint directories: {rendered}"
        )
    if not matches:
        searched = ", ".join(str(path / pattern) for path in candidate_dirs)
        raise FileNotFoundError(
            f"{subdir}: no legacy checkpoint found; searched: {searched}"
        )
    return matches[0], "legacy_unique_epoch_match"


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


def _template_path(path: Path, root: Path) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return str(path)
    return "$DATA_ROOT/" + relative.as_posix()


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


def _normalize_apply_modes(raw: str) -> tuple[str, ...]:
    compact = str(raw or "1").replace(" ", "")
    presets = {
        "": ("pixels",),
        "1": ("pixels",),
        "pixel": ("pixels",),
        "pixels": ("pixels",),
        "obs": ("pixels",),
        "observation": ("pixels",),
        "2": ("goal",),
        "goal": ("goal",),
        "3": ("pixels+goal",),
        "both": ("pixels+goal",),
        "pixels+goal": ("pixels+goal",),
        "pixels_goal": ("pixels+goal",),
        "pixels-goal": ("pixels+goal",),
        "pixelsgoal": ("pixels+goal",),
        "all_streams": ("pixels+goal",),
        "4": ("pixels", "pixels+goal"),
        "primary_aux": ("pixels", "pixels+goal"),
        "primary+aux": ("pixels", "pixels+goal"),
        "primary_auxiliary": ("pixels", "pixels+goal"),
        "5": ("pixels", "goal", "pixels+goal"),
        "all": ("pixels", "goal", "pixels+goal"),
    }
    if compact in presets:
        return presets[compact]

    single = {
        "1": "pixels",
        "pixel": "pixels",
        "pixels": "pixels",
        "obs": "pixels",
        "observation": "pixels",
        "2": "goal",
        "goal": "goal",
        "3": "pixels+goal",
        "both": "pixels+goal",
        "pixels+goal": "pixels+goal",
        "pixels_goal": "pixels+goal",
        "pixels-goal": "pixels+goal",
        "pixelsgoal": "pixels+goal",
        "all_streams": "pixels+goal",
    }
    tokens = compact.split(",")
    try:
        return tuple(single[token] for token in tokens)
    except KeyError as exc:
        raise ValueError(f"invalid eval corruption apply_to value: {raw!r}") from exc


def _expected_eval_labels(
    *,
    family: str,
    magnitudes: tuple[str, ...],
    apply_to: str,
    eval_seeds: int,
    eval_base_seed: int,
) -> list[str]:
    modes = _normalize_apply_modes(apply_to)
    family_spec = {
        "gaussian_noise": ("std", 0.0),
        "gaussian_blur": ("blur_ks", 1.0),
        "resize": ("rs_factor", 1.0),
    }
    tag, origin_magnitude = family_spec[family]
    labels: list[str] = []
    for magnitude in magnitudes:
        is_origin = float(magnitude) == origin_magnitude
        for offset in range(eval_seeds):
            seed = eval_base_seed + offset
            suffix = f"_seed{seed}" if eval_seeds > 1 else ""
            if is_origin:
                labels.append(f"origin{suffix}")
                continue
            for mode in modes:
                labels.append(
                    f"{mode.replace('+', '_')}_{tag}{magnitude}{suffix}"
                )
    return labels


def _metrics_file_complete(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    text = path.read_text(encoding="utf-8", errors="replace")
    return "==== RESULTS ====" in text and "evaluation_time:" in text


def _eval_job_complete(
    *,
    result_dir: Path,
    family: str,
    magnitudes: tuple[str, ...],
    apply_to: str,
    eval_seeds: int,
    eval_base_seed: int,
) -> bool:
    if not _eval_summary_has_rows(result_dir / "eval_summary.csv"):
        return False
    labels = _expected_eval_labels(
        family=family,
        magnitudes=magnitudes,
        apply_to=apply_to,
        eval_seeds=eval_seeds,
        eval_base_seed=eval_base_seed,
    )
    return all(
        _metrics_file_complete(result_dir / f"{label}_metrics.txt")
        for label in labels
    )


def build_jobs(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    canonical_path = _resolve_repo_path(args.canonical)
    canonical = _load_json(canonical_path)
    root = Path(args.root).expanduser().resolve()
    family_magnitudes = _parse_family_magnitudes(args.family_magnitudes)

    tasks = [_normalize_task(t, canonical) for t in args.tasks]
    families = args.families
    for family in families:
        if family not in FAMILY_META:
            raise ValueError(f"unknown corruption family {family!r}; expected one of {sorted(FAMILY_META)}")

    output_prefix = args.output_prefix or f"paper1_unseen_s{args.train_seed}"

    jobs: list[dict[str, Any]] = []
    checkpoint_hashes: dict[Path, str] = {}
    for task in tasks:
        meta = TASK_META[task]
        std_keys = _normalize_std_keys(args.std_keys, task, canonical)
        for std_key in std_keys:
            entry = canonical[task][std_key]
            subdir = entry["subdir"]
            ckpt_path, checkpoint_resolution = _resolve_checkpoint(
                entry=entry,
                root=root,
                dataset_dir=meta["dataset_dir"],
                subdir=subdir,
                epoch=args.epoch,
            )
            ckpt_rel = Path(_portable(ckpt_path, root))
            if ckpt_path not in checkpoint_hashes:
                checkpoint_hashes[ckpt_path] = _sha256(ckpt_path)
            model_sha256 = checkpoint_hashes[ckpt_path]
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
                template_env["ckpt_override"] = _template_path(ckpt_path, root)

                eval_summary = result_dir / "eval_summary.csv"
                diagnostics_summary = diag_dir / "diagnostics_summary.json"
                complete = _eval_job_complete(
                    result_dir=result_dir,
                    family=family,
                    magnitudes=magnitudes,
                    apply_to=str(args.apply_to),
                    eval_seeds=int(args.eval_seeds),
                    eval_base_seed=int(args.eval_base_seed),
                ) and (
                    not args.diagnostics or diagnostics_summary.is_file()
                )
                jobs.append(
                    {
                        "task": task,
                        "std_key": std_key,
                        "family": family,
                        "subdir": subdir,
                        "checkpoint_rel": ckpt_rel.as_posix(),
                        "checkpoint_exists": True,
                        "checkpoint_resolution": checkpoint_resolution,
                        "model_file": str(ckpt_path),
                        "model_sha256": model_sha256,
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
            "schema_version": "paper1-unseen-eval-grid-manifest-1.1",
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
            "checkpoint_binding": (
                "entry.model_file is authoritative when present; legacy entries require "
                "exactly one *epoch_<epoch>_object.ckpt candidate"
            ),
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
