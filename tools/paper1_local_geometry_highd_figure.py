"""Build the Paper 1 local-geometry figure from cached features.

The t-SNE panels in this figure are qualitative.  Every displayed ratio,
fraction, and state-category count is recomputed in the original feature
space and checked against the frozen sidecar produced by
``paper1_selective_contraction.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from paper1_selective_contraction import (
    _axis_limits_2d_single,
    _draw_cluster_envelope,
    _draw_cluster_links,
    _ensure_plot_deps,
    _tsne_fit_transform_2d,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = Path("/tmp/paper1_selective_contraction_cache")
DEFAULT_SIDECAR = ROOT / "assets/paper1_figs/fig_acpc_basin_tsne_point_counts.json"
DEFAULT_OUTPUT = ROOT / "assets/paper1_figs/fig_local_geometry_highd_audit.pdf"
DEFAULT_AUDIT_OUTPUT = ROOT / "assets/paper1_figs/fig_local_geometry_highd_audit.json"

PANEL_SPECS = (
    ("base", "encoder", "Unaugmented LeWM", "Encoder features"),
    ("base", "predictor", "Unaugmented LeWM", "8-step rollout predictions"),
    ("fullseq_robust", "encoder", "Noise-trained LeWM", "Encoder features"),
    (
        "fullseq_robust",
        "predictor",
        "Noise-trained LeWM",
        "8-step rollout predictions",
    ),
)

def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_matching_cache(cache_dir: Path, sidecar: Mapping[str, Any]) -> tuple[Path, dict[str, Any], dict[str, np.ndarray]]:
    candidates: list[tuple[Path, dict[str, Any], dict[str, np.ndarray]]] = []
    for path in sorted(cache_dir.glob("pusht_lewm_fullseq_features_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"].item()))
            arrays = {
                key: np.asarray(data[key])
                for key in (
                    "base_encoder",
                    "base_predictor",
                    "fullseq_robust_encoder",
                    "fullseq_robust_predictor",
                )
            }
        expected_views = [float(value) for value in sidecar["expanded_view_stds"]]
        if (
            metadata.get("task") == sidecar.get("task") == "PushT"
            and metadata.get("method") == sidecar.get("method") == "LeWM"
            and int(metadata.get("n_sequences", -1)) == int(sidecar["n_sequences"])
            and int(metadata.get("rollout_horizon", -1)) == int(sidecar["rollout_horizon"])
            and int(metadata.get("seed", -1)) == int(sidecar["seed"])
            and [float(value) for value in metadata.get("view_stds", [])] == expected_views
        ):
            candidates.append((path, metadata, arrays))
    if len(candidates) != 1:
        raise RuntimeError(
            "Expected exactly one cache matching the frozen PushT audit; "
            f"found {len(candidates)} in {cache_dir}."
        )
    return candidates[0]


def _high_d_audit(array: np.ndarray) -> dict[str, Any]:
    if array.ndim != 3 or array.shape[0] < 2:
        raise ValueError(f"Expected [views, states, features], got {array.shape}")
    origin = np.asarray(array[0], dtype=np.float64)
    perturbed = np.asarray(array[1:], dtype=np.float64)
    radius = np.max(np.linalg.norm(perturbed - origin[None, :, :], axis=-1), axis=0)
    center_dist = np.linalg.norm(origin[:, None, :] - origin[None, :, :], axis=-1)
    np.fill_diagonal(center_dist, np.inf)
    nearest = np.min(center_dist, axis=1)
    ratio = radius / np.maximum(nearest, 1e-12)

    disjoint = np.all(center_dist > radius[:, None] + radius[None, :], axis=1)
    reaches_spacing = ratio >= 1.0
    within_not_disjoint = (~reaches_spacing) & (~disjoint)
    if np.any(disjoint & reaches_spacing):
        raise AssertionError("A fully disjoint ball cannot reach its nearest clean center.")

    counts = np.asarray(
        [
            int(np.sum(reaches_spacing)),
            int(np.sum(within_not_disjoint)),
            int(np.sum(disjoint)),
        ],
        dtype=int,
    )
    n_states = int(array.shape[1])
    if int(np.sum(counts)) != n_states:
        raise AssertionError(f"State categories do not partition all states: {counts}")

    return {
        "n_states": n_states,
        "n_views": int(array.shape[0]),
        "median_radius_over_nn": float(np.median(ratio)),
        "radius_lt_nn_count": int(np.sum(ratio < 1.0)),
        "radius_lt_nn_fraction": float(np.mean(ratio < 1.0)),
        "fully_disjoint_count": int(np.sum(disjoint)),
        "fully_disjoint_fraction": float(np.mean(disjoint)),
        "category_counts": [int(value) for value in counts],
        "category_fractions": [float(value / n_states) for value in counts],
    }


def _expected_by_panel(sidecar: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(row["panel"]): row
        for row in sidecar["panel_high_d_stats"]
    }


def _validate_audit(panel: str, audit: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    pairs = (
        ("median_radius_over_nn", "median_radius_over_nn"),
        ("radius_lt_nn_fraction", "frac_radius_lt_nn"),
        ("fully_disjoint_fraction", "frac_nonoverlap_balls"),
    )
    for actual_key, expected_key in pairs:
        actual = float(audit[actual_key])
        target = float(expected[expected_key])
        if not np.isclose(actual, target, rtol=1e-6, atol=1e-8):
            raise AssertionError(
                f"{panel} {actual_key}={actual} disagrees with frozen sidecar {target}."
            )


def _format_percent(count: int, total: int) -> str:
    return f"{100.0 * count / total:.1f}%"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_figure(
    *,
    cache_dir: Path,
    sidecar_path: Path,
    output_path: Path,
    audit_output_path: Path,
    perplexity: float,
    tsne_max_iter: int,
) -> None:
    plt = _ensure_plot_deps()
    sidecar = _load_json(sidecar_path)
    cache_path, metadata, arrays = _load_matching_cache(cache_dir, sidecar)
    anchors = np.asarray(sidecar["anchor_indices"], dtype=int)
    expanded_stds = [float(value) for value in sidecar["expanded_view_stds"]]
    expected = _expected_by_panel(sidecar)

    if len(anchors) != 16 or len(set(anchors.tolist())) != len(anchors):
        raise AssertionError("The frozen qualitative display must use 16 unique anchors.")
    if int(sidecar["perturb_repeats"]) != 6 or len(expanded_stds) != 19:
        raise AssertionError("Expected one clean and 18 perturbed views per state.")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "mathtext.fontset": "stix",
            "font.size": 8.0,
            "axes.titlesize": 7.1,
            "axes.labelsize": 6.8,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 5.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(7.65, 2.30), constrained_layout=False)
    outer = fig.add_gridspec(
        1,
        4,
        left=0.05,
        right=0.99,
        bottom=0.085,
        top=0.88,
        wspace=0.16,
    )
    colors = plt.cm.turbo(np.linspace(0.05, 0.95, len(anchors)))
    audit_rows: list[dict[str, Any]] = []

    # Checkpoint names are shared column-group headers.  Keeping them separate
    # from the panel labels avoids repeating long titles in four narrow axes.
    fig.text(
        0.285,
        0.995,
        "Unaugmented LeWM",
        ha="center",
        va="top",
        fontsize=7.8,
        fontweight="semibold",
    )
    fig.text(
        0.755,
        0.995,
        "Noise-trained LeWM",
        ha="center",
        va="top",
        fontsize=7.8,
        fontweight="semibold",
    )

    # Reproduce the original Appendix visualization: each panel has its own
    # deterministic t-SNE fit, using the same panel order and seeds as
    # ``paper1_selective_contraction.py``.  The coordinates are qualitative
    # and are not compared across panels; all reported metrics are computed
    # from the original high-dimensional arrays below.
    projected_by_key: dict[str, np.ndarray] = {}
    limits_by_key: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {}
    panel_seeds: dict[str, int] = {}
    for panel_index, (label, feature, _, _) in enumerate(PANEL_SPECS):
        key = f"{label}_{feature}"
        panel_seed = int(sidecar["seed"]) + 17 * (panel_index + 1)
        projected = _tsne_fit_transform_2d(
            arrays[key],
            seed=panel_seed,
            perplexity=perplexity,
            max_iter=tsne_max_iter,
        )
        projected_by_key[key] = projected
        limits_by_key[key] = _axis_limits_2d_single(projected)
        panel_seeds[f"{label}:{feature}"] = panel_seed

    for panel_index, (label, feature, row_title, column_title) in enumerate(PANEL_SPECS):
        key = f"{label}_{feature}"
        panel_name = f"{label}:{feature}"
        array = arrays[key]
        if array.shape[:2] != (19, 128):
            raise AssertionError(f"{panel_name} has unexpected shape {array.shape}.")
        audit = _high_d_audit(array)
        _validate_audit(panel_name, audit, expected[panel_name])
        audit_rows.append(
            {
                "panel": panel_name,
                "condition": row_title,
                "representation": column_title,
                **audit,
            }
        )

        ax = fig.add_subplot(outer[0, panel_index])
        projected = projected_by_key[key]
        origin = projected[0]
        perturbed = projected[1:]
        xlim, ylim = limits_by_key[key]
        min_radius = 0.018 * max(xlim[1] - xlim[0], ylim[1] - ylim[0])

        ax.scatter(origin[:, 0], origin[:, 1], s=8, c="#6F6F6F", alpha=0.30, linewidths=0)
        ax.scatter(
            perturbed.reshape(-1, 2)[:, 0],
            perturbed.reshape(-1, 2)[:, 1],
            s=5,
            c="#B8B8B8",
            alpha=0.12,
            linewidths=0,
        )
        for color, state_index in zip(colors, anchors):
            points = projected[:, state_index, :]
            _draw_cluster_envelope(
                plt,
                ax,
                points,
                color=color,
                mode="ellipse",
                coverage=0.90,
                min_radius=min_radius,
            )
            _draw_cluster_links(ax, points, expanded_stds, color)
            ax.scatter(
                points[1:, 0],
                points[1:, 1],
                s=17,
                color=color,
                alpha=0.77,
                edgecolor="white",
                linewidth=0.18,
                zorder=3,
            )
            ax.scatter(
                points[0, 0],
                points[0, 1],
                s=48,
                color=color,
                edgecolor="#111111",
                linewidth=0.5,
                zorder=4,
            )

        total = int(audit["n_states"])
        radius_count = int(audit["radius_lt_nn_count"])
        disjoint_count = int(audit["fully_disjoint_count"])
        metric_text = (
            f"median r/NN: {audit['median_radius_over_nn']:.2f}\n"
            f"r < NN: {_format_percent(radius_count, total)}\n"
            f"fully disjoint: {_format_percent(disjoint_count, total)}"
        )
        ax.text(
            0.025,
            0.975,
            metric_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=5.4,
            linespacing=1.12,
            color="#202020",
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "white",
                "edgecolor": "#D0D0D0",
                "linewidth": 0.45,
                "alpha": 0.82,
            },
            zorder=8,
        )
        panel_letter = chr(ord("a") + panel_index)
        ax.set_title(
            f"({panel_letter}) {column_title}",
            loc="left",
            pad=3.0,
            fontsize=6.5,
            fontweight="semibold",
        )
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel("t-SNE coordinate 1", labelpad=1.5)
        if panel_index == 0:
            ax.set_ylabel("t-SNE coordinate 2", labelpad=1.5)
        ax.tick_params(pad=1)
        ax.grid(True, color="#ECECEC", linewidth=0.45)
        ax.set_aspect("equal", adjustable="box")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    audit_payload = {
        "schema_version": "paper1-local-geometry-highd-audit-1.0",
        "figure": str(output_path.relative_to(ROOT)),
        "feature_cache": cache_path.name,
        "feature_cache_sha256": _sha256(cache_path),
        "source_sidecar": str(sidecar_path.relative_to(ROOT)),
        "task": sidecar["task"],
        "method": sidecar["method"],
        "seed": int(sidecar["seed"]),
        "rollout_horizon": int(sidecar["rollout_horizon"]),
        "n_states": int(sidecar["n_sequences"]),
        "n_perturbation_views_per_state": len(expanded_stds) - 1,
        "view_stds": expanded_stds,
        "anchor_indices": [int(value) for value in anchors],
        "tsne": {
            "perplexity": float(perplexity),
            "max_iter": int(tsne_max_iter),
            "fit": "independent per panel, matching the original Appendix visualization",
            "panel_seeds": panel_seeds,
            "purpose": "qualitative visualization only",
        },
        "panels": audit_rows,
        "note": (
            "All metrics and category counts are computed in the original "
            "high-dimensional arrays; t-SNE coordinates are used only for display."
        ),
        "cache_metadata": metadata,
    }
    audit_output_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_output_path.open("w", encoding="utf-8") as handle:
        json.dump(audit_payload, handle, indent=2)
        handle.write("\n")

    print(f"Wrote {output_path}")
    print(f"Wrote {audit_output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--sidecar", type=Path, default=DEFAULT_SIDECAR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit-output", type=Path, default=DEFAULT_AUDIT_OUTPUT)
    parser.add_argument("--perplexity", type=float, default=35.0)
    parser.add_argument("--tsne-max-iter", type=int, default=650)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_figure(
        cache_dir=args.cache_dir.expanduser().resolve(),
        sidecar_path=args.sidecar.expanduser().resolve(),
        output_path=args.output.expanduser().resolve(),
        audit_output_path=args.audit_output.expanduser().resolve(),
        perplexity=args.perplexity,
        tsne_max_iter=args.tsne_max_iter,
    )


if __name__ == "__main__":
    main()
