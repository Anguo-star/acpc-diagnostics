#!/usr/bin/env python3
"""Render the Paper1 qualitative feature-neighborhood + ATR/SMPR trend figure.

The feature cache is produced by tools.paper1_selective_contraction.py. This
renderer intentionally avoids legacy R_E/R_F labels and uses the plot only as a
qualitative projection, with ATR/SMPR kept as the paper-facing quantitative
readouts.
"""

from __future__ import annotations

import argparse
import json
import fcntl
import os
from contextlib import contextmanager
from pathlib import Path

# Keep small PCA jobs from oversubscribing BLAS threads in shared terminals.
for thread_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(thread_var, "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = Path("/tmp/paper1_selective_contraction_cache/pusht_lewm_fullseq_features_d82fffb5ef90.npz")
DEFAULT_SUMMARY = ROOT / "assets" / "paper1_data" / "compressed_metrics_summary_20260706.json"
DEFAULT_OUT = ROOT / "assets" / "paper1_figs" / "fig_feature_neighborhood_atr_smpr.png"
DEFAULT_LOCK = Path("/tmp/paper1_feature_neighborhood_figure.lock")


@contextmanager
def single_instance(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"another feature-neighborhood render is already running; lock={lock_path}"
            ) from exc
        yield


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"missing {label}: {path}")



def pca_project(arrays: list[np.ndarray]) -> list[np.ndarray]:
    flat = [a.reshape(-1, a.shape[-1]) for a in arrays]
    x = np.concatenate(flat, axis=0).astype(np.float64)
    mean = x.mean(axis=0, keepdims=True)
    xc = x - mean
    cov = (xc.T @ xc) / max(xc.shape[0] - 1, 1)
    _, eigvecs = np.linalg.eigh(cov)
    comps = eigvecs[:, -2:][:, ::-1]
    projected = []
    start = 0
    for a, f in zip(arrays, flat):
        n = f.shape[0]
        y = (x[start:start+n] - mean) @ comps
        projected.append(y.reshape(*a.shape[:-1], 2))
        start += n
    return projected


def axis_limits(arrays: list[np.ndarray]) -> tuple[tuple[float, float], tuple[float, float]]:
    pts = np.concatenate([a.reshape(-1, 2) for a in arrays], axis=0)
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    pad = span * 0.08
    return (float(lo[0] - pad[0]), float(hi[0] + pad[0])), (float(lo[1] - pad[1]), float(hi[1] + pad[1]))


def draw_panel(ax, arr: np.ndarray, anchors: np.ndarray, view_idx: np.ndarray, title: str, xlim, ylim) -> None:
    colors = plt.cm.tab10(np.linspace(0, 1, len(anchors)))
    ax.set_facecolor("#fbfbf8")
    ax.grid(True, color="#e8e8e0", linewidth=0.7, zorder=0)
    for color, anchor in zip(colors, anchors):
        clean = arr[0, anchor]
        noisy = arr[view_idx, anchor]
        for p in noisy:
            ax.plot([clean[0], p[0]], [clean[1], p[1]], color=color, alpha=0.28, linewidth=0.75, zorder=1)
        ax.scatter(noisy[:, 0], noisy[:, 1], s=10, marker="^", color=color, alpha=0.58, linewidths=0, zorder=2)
        ax.scatter(clean[0], clean[1], s=28, marker="o", facecolor="white", edgecolor=color, linewidth=1.2, zorder=3)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9.5, pad=5)


def draw_metric_panel(ax, summary_path: Path) -> None:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    row = next(r for r in data["summary_rows"] if r["task"] == "PushT")
    atr0 = row["ATR_q90_0.0"]["mean"]
    atr1 = row["ATR_q90_0.08"]["mean"]
    smpr0 = row["SMPR_0.0"]["mean"]
    smpr1 = row["SMPR_0.08"]["mean"]
    ax.axis("off")
    ax.set_title("PushT readouts", fontsize=10.2, pad=6)
    lines = [
        ("ATR", atr0, atr1, "lower is better", "#d95f0e"),
        ("SMPR", smpr0, smpr1, "higher is better", "#31a354"),
    ]
    y = 0.72
    for name, a, b, note, color in lines:
        ax.text(0.02, y + 0.08, name, color=color, fontsize=10.5, weight="semibold", transform=ax.transAxes)
        ax.text(0.02, y - 0.01, f"{a:.2f}  →  {b:.2f}", fontsize=10.0, transform=ax.transAxes)
        ax.text(0.02, y - 0.12, note, fontsize=8.4, color="#555555", transform=ax.transAxes)
        ax.annotate("", xy=(0.87, y + (0.03 if name == "SMPR" else -0.07)), xytext=(0.58, y + 0.03),
                    xycoords=ax.transAxes, textcoords=ax.transAxes,
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.7))
        y -= 0.36
    ax.text(
        0.02,
        0.05,
        "Feature panels are 2-D\nqualitative projections;\nATR/SMPR carry the\nquantitative claim.",
        fontsize=8.0,
        color="#444444",
        transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#c9c9c0", linewidth=0.7),
    )


def render(cache_path: Path, summary_path: Path, out_path: Path) -> None:
    require_file(cache_path, "feature cache")
    require_file(summary_path, "compressed ATR/SMPR summary")

    data = np.load(cache_path, allow_pickle=True, mmap_mode="r")
    base_encoder = np.asarray(data["base_encoder"])
    robust_encoder = np.asarray(data["fullseq_robust_encoder"])
    base_rollout = np.asarray(data["base_predictor"])
    robust_rollout = np.asarray(data["fullseq_robust_predictor"])

    base_encoder_2d, robust_encoder_2d = pca_project([base_encoder, robust_encoder])
    base_rollout_2d, robust_rollout_2d = pca_project([base_rollout, robust_rollout])
    enc_xlim, enc_ylim = axis_limits([base_encoder_2d, robust_encoder_2d])
    rol_xlim, rol_ylim = axis_limits([base_rollout_2d, robust_rollout_2d])

    rng = np.random.default_rng(20260706)
    anchors = np.sort(rng.choice(base_encoder.shape[1], size=9, replace=False))
    candidate_views = np.arange(1, base_encoder.shape[0])
    view_idx = candidate_views[np.linspace(0, len(candidate_views) - 1, 7).round().astype(int)]

    fig = plt.figure(figsize=(7.7, 4.45), dpi=220)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 0.72], wspace=0.14, hspace=0.22)
    axes = {
        "base_encoder": fig.add_subplot(gs[0, 0]),
        "base_rollout": fig.add_subplot(gs[0, 1]),
        "robust_encoder": fig.add_subplot(gs[1, 0]),
        "robust_rollout": fig.add_subplot(gs[1, 1]),
        "metrics": fig.add_subplot(gs[:, 2]),
    }
    draw_panel(axes["base_encoder"], base_encoder_2d, anchors, view_idx, "baseline: encoder features", enc_xlim, enc_ylim)
    draw_panel(axes["base_rollout"], base_rollout_2d, anchors, view_idx, "baseline: after predictor", rol_xlim, rol_ylim)
    draw_panel(axes["robust_encoder"], robust_encoder_2d, anchors, view_idx, "std0.08: encoder features", enc_xlim, enc_ylim)
    draw_panel(axes["robust_rollout"], robust_rollout_2d, anchors, view_idx, "std0.08: after predictor", rol_xlim, rol_ylim)
    draw_metric_panel(axes["metrics"], summary_path)
    fig.suptitle("PushT same-state feature neighborhoods under visual perturbations", fontsize=11.5, y=0.985)
    fig.text(0.055, 0.02, "Circles are clean anchors; triangles are same-state noisy views. Shared colors indicate the same underlying state.", fontsize=8.4, color="#444444")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    args = parser.parse_args()
    with single_instance(args.lock):
        render(args.feature_cache, args.summary, args.out)
    print(args.out.relative_to(ROOT) if args.out.is_relative_to(ROOT) else args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
