#!/usr/bin/env python3
"""Sample-level fixed-pool certificate audit for Paper 1.

This script recomputes fixed-pool candidate costs from checkpoints. Unlike the
retained phase-0 summaries, it keeps the sample-level maximum paired cost drift
needed for the theorem's sufficient fixed-pool event:

    max_j |C_h(a_j)-C_tilde_h(a_j)| < clean_top1_top2_margin / 2.

The output is an audit artifact. It does not evaluate adaptive CEM, repeated
replanning, or closed-loop behavior.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from tools import paper1_phase0_acpc as phase0
from tools.paper1_margin_flip_curve import MANIFEST_DIR, SEEDS, TASKS, _success

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON = ROOT / 'paper1' / 'results' / 'sample_level_certificate_audit.json'
DEFAULT_CSV = ROOT / 'paper1' / 'results' / 'sample_level_certificate_audit.csv'
DEFAULT_SAMPLE_CSV = ROOT / 'paper1' / 'results' / 'sample_level_certificate_samples.csv'
STD_KEYS = ('0.0', '0.01', '0.02', '0.03', '0.04', '0.05', '0.06', '0.07', '0.08')
EPS_QUANTILES = (0.90, 0.95, 0.99, 0.995, 0.999)
K_VALUES = (8, 16, 32, 65)
SCHEMA_VERSION = 'paper1-sample-level-certificate-0.2'


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return 'unknown'
    return result.stdout.strip() if result.returncode == 0 else 'unknown'


def _jsonable(obj: Any) -> Any:
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _q(x: torch.Tensor, q: float) -> float:
    if x.numel() == 0:
        return float('nan')
    return float(torch.quantile(x.detach().float().cpu(), float(q)).item())


def _mean(x: torch.Tensor) -> float:
    if x.numel() == 0:
        return float('nan')
    return float(x.detach().float().mean().cpu().item())


def _wilson_lower_one_sided95(successes: int, total: int) -> float:
    if total <= 0:
        return float('nan')
    z = 1.6448536269514722
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half_width = (
        z
        * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return max(0.0, center - half_width)


def _certificate_for_pool(
    clean: torch.Tensor,
    noisy: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if clean.ndim != 2 or noisy.shape != clean.shape or clean.size(1) < 2:
        raise ValueError('candidate cost tensors must share shape (B,K), K>=2')
    drift = (clean - noisy).abs()
    clean_best = torch.argmin(clean, dim=1)
    noisy_best = torch.argmin(noisy, dim=1)
    gather_index = clean_best.unsqueeze(1)
    winner_cost = clean.gather(1, gather_index)
    winner_drift = drift.gather(1, gather_index).squeeze(1)
    candidate_margin = clean - winner_cost
    sharp_candidate_slack = candidate_margin - drift - winner_drift.unsqueeze(1)
    sharp_candidate_slack.scatter_(1, gather_index, float('inf'))
    sharp_slack = sharp_candidate_slack.min(dim=1).values
    sharp_pass = sharp_slack > 0.0
    drift_sum = drift + winner_drift.unsqueeze(1)
    normalized_candidate_score = torch.full_like(
        candidate_margin,
        torch.finfo(candidate_margin.dtype).max,
    )
    positive_margin = candidate_margin > 0.0
    normalized_candidate_score[positive_margin] = (
        drift_sum[positive_margin] / candidate_margin[positive_margin]
    )
    normalized_candidate_score.scatter_(1, gather_index, float('-inf'))
    sharp_normalized_score = normalized_candidate_score.max(dim=1).values
    if not torch.equal(sharp_normalized_score < 1.0, sharp_pass):
        raise RuntimeError('normalized sharp score is not equivalent to positive slack')
    sorted_clean = torch.sort(clean, dim=1).values
    top2_margin = sorted_clean[:, 1] - sorted_clean[:, 0]
    max_drift = drift.max(dim=1).values
    coarse_pass = max_drift < (top2_margin / 2.0)
    flips = clean_best != noisy_best
    if bool(flips[sharp_pass].any()):
        raise RuntimeError('sharp certificate invariant violated: flip with positive slack')
    return {
        'drift': drift,
        'clean_best': clean_best,
        'noisy_best': noisy_best,
        'flips': flips,
        'top2_margin': top2_margin,
        'max_drift': max_drift,
        'mean_drift': drift.mean(dim=1),
        'coarse_pass': coarse_pass,
        'sharp_slack': sharp_slack,
        'sharp_normalized_score': sharp_normalized_score,
        'sharp_pass': sharp_pass,
    }


def _risk_coverage_rows(
    slack: torch.Tensor,
    flips: torch.Tensor,
) -> list[dict[str, float | int]]:
    finite = slack.detach().float().cpu()
    minimum = float(finite.min().item())
    thresholds = [
        minimum - max(1.0, abs(minimum)) * 1e-6,
        *[_q(finite, q) for q in (0.10, 0.25, 0.50, 0.75, 0.90)],
        0.0,
    ]
    unique = sorted(set(thresholds))
    rows: list[dict[str, float | int]] = []
    for threshold in unique:
        selected = finite > threshold
        selected_n = int(selected.sum().item())
        flip_n = int(flips.detach().cpu()[selected].sum().item()) if selected_n else 0
        rows.append(
            {
                'sharp_slack_threshold': threshold,
                'selected_n': selected_n,
                'coverage': selected_n / int(finite.numel()),
                'observed_flip_rate': flip_n / selected_n if selected_n else float('nan'),
                'observed_flip_n': flip_n,
            }
        )
    return rows


def _aurc(rows: Sequence[Mapping[str, float | int]]) -> float:
    points = sorted(
        (
            float(row['coverage']),
            float(row['observed_flip_rate']),
        )
        for row in rows
        if math.isfinite(float(row['observed_flip_rate']))
    )
    if len(points) < 2:
        return float('nan')
    area = 0.0
    for (left_coverage, left_risk), (right_coverage, right_risk) in zip(
        points,
        points[1:],
    ):
        area += (
            right_coverage - left_coverage
        ) * (left_risk + right_risk) / 2.0
    return area


def _costs_for_branch(model, batch: Mapping[str, torch.Tensor], candidates: torch.Tensor, *, history_size: int) -> torch.Tensor:
    return model.get_cost(phase0._cost_info(batch, history_size), candidates)


def _resolve(entry: Mapping[str, Any], model_roots: Sequence[Path]) -> tuple[Path | None, list[str]]:
    return phase0.resolve_model_file(str(entry.get('path', '')), str(entry.get('subdir', '')), model_roots)


def run_checkpoint(*, seed: int, task: str, std_key: str, entry: Mapping[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    phase0._ensure_runtime_deps()
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model_roots = [Path(p).expanduser() for p in args.model_root]
    model_file, tried = _resolve(entry, model_roots)
    base = {
        'training_seed': int(seed),
        'task': task,
        'std_key': std_key,
        'subdir': entry.get('subdir'),
        'run_path': entry.get('path'),
        'model_file': str(model_file) if model_file else None,
        'model_search_dirs': tried,
        'clean_success': _success(entry, 'clean'),
        'pixels_std0.08_success': _success(entry, 'pixels_std0.08'),
        'noise_std': float(args.noise_std),
    }
    if model_file is None:
        return {**base, 'status': 'skipped_missing_model'}, []

    with torch.no_grad():
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
            seed=seed,
            device=device,
        )
        noisy_batch = phase0.make_paired_noisy_batch(
            batch,
            history_size=history_size,
            noise_std=args.noise_std,
            seed=seed + 1009,
            corruption_type=args.corruption_type,
            corrupt_goal=False,
        )
        candidates = phase0.build_action_candidates(
            batch['action'],
            history_size=history_size,
            future_steps=future_steps,
            random_action_trials=args.random_action_trials,
            seed=seed + 2027,
        )
        clean_costs = _costs_for_branch(model, batch, candidates, history_size=history_size)
        noisy_costs = _costs_for_branch(model, noisy_batch, candidates, history_size=history_size)

    clean = clean_costs.detach().float().cpu()
    noisy = noisy_costs.detach().float().cpu()
    canonical = _certificate_for_pool(clean, noisy)
    abs_diff = canonical['drift']
    margins = canonical['top2_margin']
    max_drift = canonical['max_drift']
    mean_drift = canonical['mean_drift']
    flat_drift = abs_diff.reshape(-1)
    clean_best = canonical['clean_best']
    noisy_best = canonical['noisy_best']
    flips = canonical['flips']
    coarse_pass = canonical['coarse_pass']
    sharp_slack = canonical['sharp_slack']
    sharp_normalized_score = canonical['sharp_normalized_score']
    sharp_pass = canonical['sharp_pass']

    coverage_by_k: dict[str, dict[str, float | int]] = {}
    for candidate_count in args.k_values:
        metrics = _certificate_for_pool(
            clean[:, :candidate_count],
            noisy[:, :candidate_count],
        )
        pass_n = int(metrics['sharp_pass'].sum().item())
        lower = _wilson_lower_one_sided95(pass_n, int(clean.size(0)))
        k_risk_coverage = _risk_coverage_rows(
            metrics['sharp_slack'],
            metrics['flips'],
        )
        coverage_by_k[str(candidate_count)] = {
            'candidate_count': int(candidate_count),
            'sharp_cert_pass_rate': _mean(metrics['sharp_pass'].float()),
            'sharp_cert_pass_n': pass_n,
            'sharp_cert_pass_lower95_wilson': lower,
            'flip_risk_upper95_from_coverage': 1.0 - lower,
            'observed_flip_rate': _mean(metrics['flips'].float()),
            'flip_when_sharp_cert_fail_rate': (
                _mean(metrics['flips'][~metrics['sharp_pass']].float())
                if bool((~metrics['sharp_pass']).any())
                else float('nan')
            ),
            'risk_coverage_aurc': _aurc(k_risk_coverage),
        }
    risk_coverage = _risk_coverage_rows(sharp_slack, flips)

    eps_rows: dict[str, dict[str, float]] = {}
    for q in args.eps_quantiles:
        eps = _q(flat_drift, q)
        alpha_hat = float((flat_drift > eps).float().mean().item()) if math.isfinite(eps) else float('nan')
        margin_fail = float((margins <= 2.0 * eps).float().mean().item()) if math.isfinite(eps) else float('nan')
        eps_rows[f'q{int(round(q * 1000)):03d}'] = {
            'epsilon': eps,
            'alpha_hat': alpha_hat,
            'k_alpha': float(clean.size(1)) * alpha_hat if math.isfinite(alpha_hat) else float('nan'),
            'margin_fail_rate': margin_fail,
            'union_bound_proxy': min(1.0, float(clean.size(1)) * alpha_hat + margin_fail) if math.isfinite(alpha_hat) and math.isfinite(margin_fail) else float('nan'),
        }

    row = {
        **base,
        'status': 'ok',
        'n_sequences': int(args.n_sequences),
        'candidate_count': int(clean.size(1)),
        'future_steps': int(future_steps),
        'cost_drift_abs_q50': _q(flat_drift, 0.50),
        'cost_drift_abs_q90': _q(flat_drift, 0.90),
        'cost_drift_abs_q95': _q(flat_drift, 0.95),
        'cost_drift_abs_q99': _q(flat_drift, 0.99),
        'sample_mean_drift_q90': _q(mean_drift, 0.90),
        'sample_max_drift_q50': _q(max_drift, 0.50),
        'sample_max_drift_q90': _q(max_drift, 0.90),
        'sample_max_drift_q95': _q(max_drift, 0.95),
        'sample_max_drift_q99': _q(max_drift, 0.99),
        'clean_margin_q10': _q(margins, 0.10),
        'clean_margin_q25': _q(margins, 0.25),
        'clean_margin_q50': _q(margins, 0.50),
        'clean_margin_q90': _q(margins, 0.90),
        'certificate_gap_q10_q95': _q(margins, 0.10) - 2.0 * _q(max_drift, 0.95),
        'certificate_gap_q50_q95': _q(margins, 0.50) - 2.0 * _q(max_drift, 0.95),
        'coarse_cert_pass_rate': _mean(coarse_pass.float()),
        'sample_cert_pass_rate': _mean(coarse_pass.float()),
        'sharp_cert_pass_rate': _mean(sharp_pass.float()),
        'sharp_cert_pass_n': int(sharp_pass.sum().item()),
        'sharp_cert_pass_lower95_wilson': _wilson_lower_one_sided95(
            int(sharp_pass.sum().item()),
            int(sharp_pass.numel()),
        ),
        'flip_risk_upper95_from_sharp_coverage': 1.0
        - _wilson_lower_one_sided95(
            int(sharp_pass.sum().item()),
            int(sharp_pass.numel()),
        ),
        'sharp_cert_slack_q10': _q(sharp_slack, 0.10),
        'sharp_cert_slack_q50': _q(sharp_slack, 0.50),
        'sharp_cert_slack_q90': _q(sharp_slack, 0.90),
        'sharp_normalized_score_q10': _q(sharp_normalized_score, 0.10),
        'sharp_normalized_score_q50': _q(sharp_normalized_score, 0.50),
        'sharp_normalized_score_q90': _q(sharp_normalized_score, 0.90),
        'sharp_normalized_score_cert_pass_rate': _mean(
            (sharp_normalized_score < 1.0).float()
        ),
        'sample_top1_flip_rate': _mean(flips.float()),
        'flip_when_coarse_cert_pass_rate': _mean(flips[coarse_pass].float()) if bool(coarse_pass.any()) else float('nan'),
        'flip_when_coarse_cert_fail_rate': _mean(flips[~coarse_pass].float()) if bool((~coarse_pass).any()) else float('nan'),
        'flip_when_sharp_cert_pass_rate': _mean(flips[sharp_pass].float()) if bool(sharp_pass.any()) else float('nan'),
        'flip_when_sharp_cert_fail_rate': _mean(flips[~sharp_pass].float()) if bool((~sharp_pass).any()) else float('nan'),
        'sharp_cert_invariant_flip_count': int(flips[sharp_pass].sum().item()),
        'coverage_by_K': coverage_by_k,
        'risk_coverage': risk_coverage,
        'risk_coverage_aurc': _aurc(risk_coverage),
        'epsilon_tail_rows': eps_rows,
        'notes': 'candidate-wise sharp and legacy coarse fixed-pool sufficient-event audit; not adaptive CEM or closed-loop guarantee',
    }

    sample_rows: list[dict[str, Any]] = []
    if args.include_samples:
        for i in range(clean.size(0)):
            sample_rows.append({
                'training_seed': int(seed),
                'task': task,
                'std_key': std_key,
                'sample_index': int(i),
                'candidate_count': int(clean.size(1)),
                'clean_margin': float(margins[i].item()),
                'sample_max_drift': float(max_drift[i].item()),
                'sample_mean_drift': float(mean_drift[i].item()),
                'coarse_cert_pass': bool(coarse_pass[i].item()),
                'cert_pass': bool(coarse_pass[i].item()),
                'sharp_cert_pass': bool(sharp_pass[i].item()),
                'sharp_cert_slack': float(sharp_slack[i].item()),
                'sharp_normalized_score': float(
                    sharp_normalized_score[i].item()
                ),
                'top1_flip': bool(flips[i].item()),
                'clean_best': int(clean_best[i].item()),
                'noisy_best': int(noisy_best[i].item()),
            })
    return row, sample_rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('')
        return
    nested = {'model_search_dirs', 'epsilon_tail_rows', 'coverage_by_K', 'risk_coverage'}
    fields = [k for k in rows[0].keys() if k not in nested]
    for key in sorted({k for r in rows for k in r.keys() if k not in fields and k not in nested}):
        fields.append(key)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, '') for k in fields})


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    p.add_argument('--tasks', nargs='+', default=list(TASKS), choices=list(TASKS))
    p.add_argument('--std-keys', nargs='+', default=list(STD_KEYS))
    p.add_argument('--eval-manifest-dir', type=Path, default=MANIFEST_DIR)
    p.add_argument('--model-root', action='append', default=[])
    p.add_argument('--out-json', type=Path, default=DEFAULT_JSON)
    p.add_argument('--out-csv', type=Path, default=DEFAULT_CSV)
    p.add_argument('--sample-csv', type=Path, default=DEFAULT_SAMPLE_CSV)
    p.add_argument('--include-samples', action='store_true')
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--n-sequences', type=int, default=100)
    p.add_argument('--future-steps', type=int, default=9)
    p.add_argument('--rollout-horizon', type=int, default=8)
    p.add_argument('--random-action-trials', type=int, default=64)
    p.add_argument('--noise-std', type=float, default=0.08)
    p.add_argument('--corruption-type', default='gaussian_noise')
    p.add_argument('--state-key', default=None)
    p.add_argument('--frameskip', type=int, default=5)
    p.add_argument('--img-size', type=int, default=224)
    p.add_argument('--eps-quantiles', type=float, nargs='+', default=list(EPS_QUANTILES))
    p.add_argument('--k-values', type=int, nargs='+', default=list(K_VALUES))
    p.add_argument('--device', default=None)
    return p


def main() -> int:
    args = build_parser().parse_args()
    canonical_candidate_count = int(args.random_action_trials) + 1
    if (
        not args.k_values
        or len(set(args.k_values)) != len(args.k_values)
        or min(args.k_values) < 2
        or max(args.k_values) > canonical_candidate_count
        or canonical_candidate_count not in args.k_values
    ):
        raise ValueError(
            'k-values must be unique, in [2,candidate_count], and include the '
            'canonical candidate count'
        )
    rows: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    specs: list[tuple[int, str, str, Mapping[str, Any]]] = []
    for seed in args.seeds:
        manifest = _load(args.eval_manifest_dir / f'lewm_seed{seed}_evals.json')
        for task in args.tasks:
            for std_key in args.std_keys:
                entry = manifest.get(task, {}).get(std_key)
                if entry is None:
                    rows.append({'status': 'skipped_missing_manifest', 'training_seed': seed, 'task': task, 'std_key': std_key})
                    continue
                specs.append((seed, task, std_key, entry))
    if args.limit is not None:
        specs = specs[: args.limit]
    for idx, (seed, task, std_key, entry) in enumerate(specs, start=1):
        print(f'[{idx}/{len(specs)}] {task} seed{seed} std{std_key}', flush=True)
        try:
            row, sample_rows = run_checkpoint(seed=seed, task=task, std_key=std_key, entry=entry, args=args)
        except Exception as exc:  # noqa: BLE001 - audit should record per-row failures.
            row, sample_rows = {'status': 'error', 'training_seed': seed, 'task': task, 'std_key': std_key, 'error': repr(exc)}, []
        rows.append(row)
        samples.extend(sample_rows)
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row.get('status'))] = counts.get(str(row.get('status')), 0) + 1
    script_path = Path(__file__).resolve()
    manifest_paths = {
        f'lewm_seed{seed}_evals': args.eval_manifest_dir / f'lewm_seed{seed}_evals.json'
        for seed in args.seeds
    }
    payload = {
        'metadata': {
            'schema_version': SCHEMA_VERSION,
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'code_commit': _git_commit(),
            'script_path': str(script_path.relative_to(ROOT)),
            'script_sha256': _sha256(script_path),
            'source_paths': {
                name: str(path) for name, path in manifest_paths.items()
            },
            'source_hashes': {
                name: _sha256(path) for name, path in manifest_paths.items()
            },
            'seeds': list(args.seeds),
            'training_seed_semantics': 'independently trained checkpoint seeds',
            'evaluation_seed_semantics': 'not applicable; fixed checkpoint-local candidate pools',
            'tasks': list(args.tasks),
            'std_keys': list(args.std_keys),
            'n_sequences': int(args.n_sequences),
            'candidate_count': canonical_candidate_count,
            'k_values': list(args.k_values),
            'noise_std': float(args.noise_std),
            'status': 'complete' if counts and set(counts) == {'ok'} else 'partial',
            'status_counts': counts,
            'missing_rows': [
                {
                    'training_seed': row.get('training_seed'),
                    'task': row.get('task'),
                    'std_key': row.get('std_key'),
                    'status': row.get('status'),
                }
                for row in rows
                if str(row.get('status', '')).startswith('skipped_')
            ],
            'errors': [
                {
                    'training_seed': row.get('training_seed'),
                    'task': row.get('task'),
                    'std_key': row.get('std_key'),
                    'error': row.get('error'),
                }
                for row in rows
                if row.get('status') == 'error'
            ],
            'sharp_certificate': 'min_{j!=j*}[Delta_j-d_j-d_{j*}] > 0',
            'coverage_interval': 'one-sided 95% Wilson lower bound',
            'note': 'Candidate-wise sharp and legacy coarse fixed-pool audit; closed-loop evaluation is not run.',
        },
        'rows': rows,
    }
    if args.include_samples:
        payload['sample_rows'] = samples
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(_jsonable(payload), indent=2))
    _write_csv(args.out_csv, rows)
    if args.include_samples:
        _write_csv(args.sample_csv, samples)
    print(f'wrote {args.out_json}')
    print(f'wrote {args.out_csv}')
    if args.include_samples:
        print(f'wrote {args.sample_csv}')
    print('status counts:', counts)
    return 0 if counts and set(counts) == {'ok'} else 1


if __name__ == '__main__':
    raise SystemExit(main())
