#!/usr/bin/env python3
"""Summarize the Paper1 sample-level fixed-pool certificate calibration."""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

from .utils_paper1_io import ROOT

DEFAULT_INPUT = ROOT / 'paper1' / 'results' / 'sample_level_certificate_endpoint_audit.csv'
DEFAULT_OUT = ROOT / 'paper1' / 'results' / 'sample_level_certificate_endpoint_summary.csv'
DEFAULT_TABLE = ROOT / 'paper1' / 'tables' / 'table_sample_level_certificate_endpoint.tex'
TASKS = ['TwoRoom', 'PushT', 'Reacher', 'Cube']


def _f(x: str) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float('nan')


def _mean_std(vals: list[float]) -> tuple[float, float]:
    vals = [v for v in vals if math.isfinite(v)]
    if not vals:
        return float('nan'), float('nan')
    return mean(vals), pstdev(vals) if len(vals) > 1 else 0.0


def _fmt(v: float, s: float, digits: int = 2) -> str:
    if not math.isfinite(v):
        return '--'
    return f'${v:.{digits}f} \\pm {s:.{digits}f}$'


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def build_summary(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get('status') == 'ok':
            grouped[(row['task'], f"{_f(row['std_key']):.2f}")].append(row)
    out: list[dict[str, object]] = []
    for task in TASKS:
        base = grouped[(task, '0.00')]
        endpoint = grouped[(task, '0.08')]
        if not base or not endpoint:
            continue
        def ms(block: list[dict[str, str]], key: str) -> tuple[float, float]:
            return _mean_std([_f(r[key]) for r in block])
        b_pass, b_pass_s = ms(base, 'sample_cert_pass_rate')
        e_pass, e_pass_s = ms(endpoint, 'sample_cert_pass_rate')
        b_flip, b_flip_s = ms(base, 'sample_top1_flip_rate')
        e_flip, e_flip_s = ms(endpoint, 'sample_top1_flip_rate')
        b_gap, b_gap_s = ms(base, 'certificate_gap_q10_q95')
        e_gap, e_gap_s = ms(endpoint, 'certificate_gap_q10_q95')
        b_gap50, b_gap50_s = ms(base, 'certificate_gap_q50_q95')
        e_gap50, e_gap50_s = ms(endpoint, 'certificate_gap_q50_q95')
        out.append({
            'task': task,
            'base_cert_pass_mean': b_pass,
            'base_cert_pass_pstdev': b_pass_s,
            'endpoint_cert_pass_mean': e_pass,
            'endpoint_cert_pass_pstdev': e_pass_s,
            'base_top1_flip_mean': b_flip,
            'base_top1_flip_pstdev': b_flip_s,
            'endpoint_top1_flip_mean': e_flip,
            'endpoint_top1_flip_pstdev': e_flip_s,
            'base_gap_q10_q95_mean': b_gap,
            'base_gap_q10_q95_pstdev': b_gap_s,
            'endpoint_gap_q10_q95_mean': e_gap,
            'endpoint_gap_q10_q95_pstdev': e_gap_s,
            'base_gap_q50_q95_mean': b_gap50,
            'base_gap_q50_q95_pstdev': b_gap50_s,
            'endpoint_gap_q50_q95_mean': e_gap50,
            'endpoint_gap_q50_q95_pstdev': e_gap50_s,
            'n_training_seeds': len(base),
            'n_sequences_per_seed': int(float(base[0]['n_sequences'])),
            'candidate_count': int(float(base[0]['candidate_count'])),
        })
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('')
        return
    fields = list(rows[0].keys())
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        w.writeheader()
        for row in rows:
            w.writerow(row)


def write_table(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r'\begin{table}[H]',
        r'\centering',
        r'\caption{Sample-level fixed-pool endpoint calibration. Cert-pass is the fraction of sampled states where the maximum paired candidate-cost drift over the fixed 65-candidate pool is below half the clean top-1/top-2 margin. Values are mean $\pm$ population std across training seeds 3072/3073/3074 with 100 sampled states per seed. Strict q10/q95 gaps remain negative, so these rows support a fixed-pool mechanism analysis rather than a calibrated probability certificate.}',
        r'\label{tab:sample-level-certificate-endpoint}',
        r'\small',
        r'\setlength{\tabcolsep}{3.5pt}',
        r'\begin{tabular}{lccc}',
        r'\toprule',
        r'Task & cert-pass base $\to$ $\stdmax=0.08$ & top-1 flip base $\to$ $\stdmax=0.08$ & q10/q95 gap base $\to$ $\stdmax=0.08$ \\',
        r'\midrule',
    ]
    for row in rows:
        lines.append(
            f"{row['task']} & "
            f"{_fmt(float(row['base_cert_pass_mean']), float(row['base_cert_pass_pstdev']))} $\\to$ {_fmt(float(row['endpoint_cert_pass_mean']), float(row['endpoint_cert_pass_pstdev']))} & "
            f"{_fmt(float(row['base_top1_flip_mean']), float(row['base_top1_flip_pstdev']))} $\\to$ {_fmt(float(row['endpoint_top1_flip_mean']), float(row['endpoint_top1_flip_pstdev']))} & "
            f"{_fmt(float(row['base_gap_q10_q95_mean']), float(row['base_gap_q10_q95_pstdev']), 1)} $\\to$ {_fmt(float(row['endpoint_gap_q10_q95_mean']), float(row['endpoint_gap_q10_q95_pstdev']), 1)} \\\\"
        )
    lines.extend([r'\bottomrule', r'\end{tabular}', r'\end{table}', ''])
    path.write_text('\n'.join(lines), encoding='utf-8')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    ap.add_argument('--out', type=Path, default=DEFAULT_OUT)
    ap.add_argument('--table', type=Path, default=DEFAULT_TABLE)
    args = ap.parse_args()
    rows = build_summary(_read(args.input))
    write_csv(args.out, rows)
    write_table(args.table, rows)
    print(f'wrote {args.out}')
    print(f'wrote {args.table}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
