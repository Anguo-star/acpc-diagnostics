#!/usr/bin/env python3
"""Draw the compact three-block Figure 1 for ACPC, IR, and DR.

The left block uses verified PushT frames.  The middle block is a qualitative
2-D illustration of local representation neighborhoods; all pair metrics in
the right block refer to the original weighted H-step rollout space, not to
distances in the illustration.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-paper1-fig1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = ROOT / "assets" / "paper1_figs" / "acpc_overview_inputs"
DEFAULT_OUT = ROOT / "assets" / "paper1_figs" / "fig_acpc_ir_dr_overview.pdf"
DEFAULT_PREVIEW = ROOT / "assets" / "paper1_figs" / "fig_acpc_ir_dr_overview.png"

CLEAN_FILE = "pusht_ep8200_step48_clean.png"
NOISY_FILE = "pusht_ep8200_step48_gaussian_std008.png"
DIFFERENT_FILE = "pusht_ep8200_step124_different_state.png"

BLUE = "#0072B2"
ORANGE = "#D55E00"
PURPLE = "#7B4FA3"
GREEN = "#009E73"
RED = "#A33B3B"
OTHER = "#7C8791"
INK = "#202124"
MID = "#66727D"
LIGHT = "#D8DEE4"
PANEL = "#F8FAFC"
WHITE = "#FFFFFF"

# Keep the three input cards at their existing rendered extent.  These values
# match the current AnnotationBbox footprint in the fixed panel-(a) geometry.
INPUT_CARD_WIDTH = 0.74144
INPUT_CARD_HEIGHT = 0.21200

STYLE = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 6.4,
    "axes.titlesize": 6.7,
    "axes.labelsize": 6.2,
}


def _rounded_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    *,
    face: str = WHITE,
    edge: str = LIGHT,
    linewidth: float = 0.8,
    radius: float = 0.025,
    zorder: float = 1,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=linewidth,
        transform=ax.transAxes,
        clip_on=False,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def _validate_inputs(input_dir: Path) -> dict[str, object]:
    metadata_path = input_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"missing Figure 1 metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("source", {}).get("task") != "PushT":
        raise ValueError("Figure 1 inputs must be verified PushT frames")
    for filename in (CLEAN_FILE, NOISY_FILE, DIFFERENT_FILE):
        if not (input_dir / filename).is_file():
            raise FileNotFoundError(f"missing Figure 1 input: {input_dir / filename}")
    return metadata


def _add_image(
    ax: plt.Axes,
    path: Path,
    center: tuple[float, float],
    *,
    edge: str,
    zoom: float = 0.105,
) -> None:
    image = np.asarray(Image.open(path).convert("RGB"))
    artist = AnnotationBbox(
        OffsetImage(
            image,
            zoom=zoom,
            interpolation="lanczos",
            resample=True,
        ),
        center,
        xycoords=ax.transAxes,
        frameon=True,
        bboxprops={
            "edgecolor": "#B7C0C9",
            "linewidth": 0.72,
            "facecolor": WHITE,
        },
        pad=0.015,
        zorder=5,
    )
    ax.add_artist(artist)

    # A short upper-right accent identifies the input without competing with
    # the lower-left stack used to denote additional logged histories.
    half_width = 0.5 * INPUT_CARD_WIDTH
    half_height = 0.5 * INPUT_CARD_HEIGHT
    right = center[0] + half_width
    top = center[1] + half_height
    accent_fraction = 0.27
    ax.plot(
        [right - accent_fraction * INPUT_CARD_WIDTH, right],
        [top, top],
        color=edge,
        linewidth=1.20,
        solid_capstyle="butt",
        transform=ax.transAxes,
        clip_on=False,
        zorder=7,
    )
    ax.plot(
        [right, right],
        [top - accent_fraction * INPUT_CARD_HEIGHT, top],
        color=edge,
        linewidth=1.20,
        solid_capstyle="butt",
        transform=ax.transAxes,
        clip_on=False,
        zorder=7,
    )


def _draw_hidden_input_stack(
    ax: plt.Axes,
    center: tuple[float, float],
) -> None:
    """Place a restrained stack of cards behind the shown different input."""
    width, height = INPUT_CARD_WIDTH, INPUT_CARD_HEIGHT
    layers = (
        (0.090, "#F3F5F7"),
        (0.060, "#EFF2F4"),
        (0.030, "#EBEEF1"),
    )
    for layer, (offset, face) in enumerate(layers, start=1):
        card = FancyBboxPatch(
            (
                center[0] - 0.5 * width - offset,
                center[1] - 0.5 * height - 0.285 * offset,
            ),
            width,
            height,
            boxstyle="round,pad=0.001,rounding_size=0.004",
            facecolor=face,
            edgecolor="#B7C0C9",
            linewidth=0.68,
            alpha=1.0,
            transform=ax.transAxes,
            clip_on=False,
            zorder=layer,
        )
        ax.add_patch(card)


def _draw_input_block(ax: plt.Axes, input_dir: Path) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    _rounded_box(
        ax,
        (0.01, 0.01),
        0.98,
        0.98,
        face=PANEL,
        edge="#A8CFE5",
        linewidth=0.95,
        radius=0.025,
        zorder=-5,
    )
    ax.text(
        0.055,
        0.955,
        "(a)  Inputs",
        ha="left",
        va="top",
        fontsize=7.5,
        color=INK,
        fontweight="semibold",
    )

    centers = ((0.50, 0.790), (0.50, 0.505), (0.50, 0.235))
    _add_image(ax, input_dir / CLEAN_FILE, centers[0], edge=BLUE, zoom=0.222)
    _add_image(ax, input_dir / NOISY_FILE, centers[1], edge=ORANGE, zoom=0.222)
    _draw_hidden_input_stack(ax, centers[2])
    _add_image(
        ax,
        input_dir / DIFFERENT_FILE,
        centers[2],
        edge=PURPLE,
        zoom=0.222,
    )

    captions = (
        (r"clean  $h_i$", BLUE, "o"),
        (r"perturbed  $\tilde h_i^{(m)}$", ORANGE, "s"),
    )
    for (_, y), (caption, color, marker) in zip(centers[:2], captions):
        caption_y = y - 0.137
        ax.scatter(
            [0.240],
            [caption_y],
            s=18,
            c=color,
            marker=marker,
            edgecolors=INK,
            linewidths=0.45,
            zorder=6,
        )
        ax.text(
            0.295,
            caption_y,
            caption,
            ha="left",
            va="center",
            fontsize=5.85,
            color=color,
            fontweight="semibold",
        )

    shown_y = 0.083
    ax.scatter(
        [0.215],
        [shown_y],
        s=20,
        c=PURPLE,
        marker="^",
        edgecolors=INK,
        linewidths=0.45,
        zorder=6,
    )
    ax.text(
        0.270,
        shown_y,
        "different state",
        ha="left",
        va="center",
        fontsize=5.70,
        color=PURPLE,
        fontweight="semibold",
    )
    hidden_y = 0.036
    ax.scatter(
        [0.215],
        [hidden_y],
        s=13,
        c=OTHER,
        marker="o",
        edgecolors=INK,
        linewidths=0.40,
        zorder=6,
    )
    ax.text(
        0.270,
        hidden_y,
        "other states",
        ha="left",
        va="center",
        fontsize=5.70,
        color=OTHER,
        fontweight="semibold",
    )


def _background_points(*, seed: int, count: int = 42) -> np.ndarray:
    """Create a sparse context cloud for the qualitative 2-D illustration."""
    rng = np.random.default_rng(seed)
    curve_count = int(round(0.70 * count))
    x_curve = rng.uniform(-1.42, 1.42, size=curve_count)
    phase = rng.uniform(-0.35, 0.35)
    y_curve = 0.52 * np.sin(1.9 * x_curve + phase)
    x_curve += rng.normal(0.0, 0.08, size=curve_count)
    y_curve += rng.normal(0.0, 0.22, size=curve_count)
    diffuse = rng.normal(
        loc=(0.03, 0.00),
        scale=(0.86, 0.62),
        size=(count - curve_count, 2),
    )
    points = np.vstack((np.column_stack((x_curve, y_curve)), diffuse))
    points[:, 0] = np.clip(points[:, 0], -1.48, 1.48)
    points[:, 1] = np.clip(points[:, 1], -1.06, 1.06)
    return points


def _draw_soft_halo(
    ax: plt.Axes,
    center: tuple[float, float] | np.ndarray,
    *,
    width: float,
    height: float,
    color: str,
    zorder: float = 2,
    transform=None,
) -> None:
    """Draw a boundary-free neighborhood whose color fades outward."""
    patch_transform = ax.transData if transform is None else transform
    for scale in np.linspace(1.24, 0.34, 18):
        ax.add_patch(
            Ellipse(
                center,
                width=width * scale,
                height=height * scale,
                facecolor=color,
                edgecolor="none",
                linewidth=0,
                alpha=0.018,
                transform=patch_transform,
                zorder=zorder,
            )
        )


def _draw_three_token_panel(
    ax: plt.Axes,
    *,
    robust: bool,
    rollout: bool,
    seed: int,
) -> None:
    ax.set_xlim(-1.60, 1.60)
    ax.set_ylim(-1.18, 1.18)
    ax.set_xticks([-1.0, 0.0, 1.0])
    ax.set_yticks([-0.6, 0.0, 0.6])
    ax.tick_params(
        axis="both",
        which="both",
        length=0,
        labelbottom=False,
        labelleft=False,
    )
    ax.set_facecolor("#FCFCFA")
    ax.grid(True, color="#E7EAED", linewidth=0.40, alpha=0.75, zorder=0)
    for spine in ax.spines.values():
        spine.set_color("#B7C0C9")
        spine.set_linewidth(0.72)

    context = _background_points(seed=seed)

    clean = np.array([-0.62, 0.12])
    if robust:
        different = np.array([0.35, -0.06])
        perturbed = np.array([-0.38, 0.10]) if rollout else np.array([-0.33, 0.09])
        other_states = np.array(
            [
                [0.10, 0.68],
                [-0.18, -0.66],
                [0.88, 0.58],
                [0.88, -0.58],
                [1.28, 0.10],
                [-1.16, 0.58],
                [-1.20, -0.44],
                [0.34, 0.94],
            ]
        )
        note = "remains separated" if rollout else "views stay close"
    else:
        different = np.array([0.70, -0.32]) if rollout else np.array([0.94, -0.48])
        perturbed = np.array([0.28, -0.09]) if rollout else np.array([-0.10, 0.06])
        other_states = (
            np.array(
                [
                    [-0.02, 0.18],
                    [-0.10, -0.12],
                    [0.12, 0.10],
                    [0.28, -0.18],
                    [0.42, 0.05],
                    [0.22, -0.25],
                    [0.64, 0.40],
                    [-0.48, -0.48],
                    [-0.92, 0.64],
                ]
            )
            if rollout
            else np.array(
                [
                    [0.20, 0.22],
                    [0.28, -0.15],
                    [0.42, 0.12],
                    [0.55, -0.10],
                    [0.05, -0.32],
                    [0.66, 0.44],
                    [-0.46, -0.48],
                    [-0.90, 0.66],
                    [0.90, 0.34],
                ]
            )
        )
        note = "overlaps other states" if rollout else "view drifts"

    # Equal-scale soft halos communicate neighborhoods without hard radii.
    halo_width, halo_height = 0.58, 0.40
    if robust:
        # Keep every contextual state outside all three robust neighborhoods.
        keep = np.ones(len(context), dtype=bool)
        for token in (clean, perturbed, different):
            normalized = (
                ((context[:, 0] - token[0]) / 0.48) ** 2
                + ((context[:, 1] - token[1]) / 0.34) ** 2
            )
            keep &= normalized > 1.0
        context = context[keep]
    ax.scatter(
        context[:, 0],
        context[:, 1],
        s=5.0,
        c="#AEB4BA",
        alpha=0.23,
        linewidths=0,
        zorder=1,
    )
    _draw_soft_halo(
        ax,
        clean,
        width=halo_width,
        height=halo_height,
        color=BLUE,
    )
    _draw_soft_halo(
        ax,
        perturbed,
        width=halo_width,
        height=halo_height,
        color=ORANGE,
    )
    _draw_soft_halo(
        ax,
        different,
        width=halo_width,
        height=halo_height,
        color=PURPLE,
    )

    # Gray anchors stand for other, unshown states.  In fragile panels,
    # several lie inside the displaced perturbation neighborhood; the shown
    # purple state is only one example and need not be the overlapping state.
    ax.scatter(
        other_states[:, 0],
        other_states[:, 1],
        s=10.5 if not robust else 8.0,
        c=OTHER,
        marker="o",
        edgecolors=WHITE,
        linewidths=0.35,
        alpha=0.72 if not robust else 0.42,
        zorder=4,
    )

    # The solid segment shows clean–perturbed displacement; it is not itself
    # an aggregate IR value.
    ax.plot(
        [clean[0], perturbed[0]],
        [clean[1], perturbed[1]],
        color=ORANGE,
        linewidth=0.85,
        alpha=0.80,
        zorder=3,
    )
    if rollout and not robust:
        ax.text(
            0.5 * (clean[0] + perturbed[0]) + 0.16,
            0.5 * (clean[1] + perturbed[1]) + 0.21,
            r"$\mathrm{ACPC}_H$",
            ha="center",
            va="center",
            fontsize=6.0,
            color=ORANGE,
            fontweight="bold",
            zorder=7,
        )

    ax.scatter(
        [clean[0]],
        [clean[1]],
        s=48,
        c=BLUE,
        marker="o",
        edgecolors=INK,
        linewidths=0.75,
        zorder=6,
    )
    ax.scatter(
        [perturbed[0]],
        [perturbed[1]],
        s=46,
        c=ORANGE,
        marker="s",
        edgecolors=INK,
        linewidths=0.75,
        zorder=6,
    )
    ax.scatter(
        [different[0]],
        [different[1]],
        s=52,
        c=PURPLE,
        marker="^",
        edgecolors=INK,
        linewidths=0.75,
        zorder=6,
    )

    ax.text(
        0.03,
        0.96,
        note,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.90,
        color=MID,
        fontweight="semibold",
        bbox={"facecolor": WHITE, "edgecolor": "none", "alpha": 0.82, "pad": 0.7},
        zorder=8,
    )


def _draw_middle_header(ax: plt.Axes, *, legend_y: float) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    legend_items = (
        (0.150, BLUE, "o", "clean"),
        (0.305, ORANGE, "s", "perturbed"),
        (0.500, PURPLE, "^", "different state"),
        (0.735, OTHER, "o", "other states"),
    )
    for x, color, marker, label in legend_items:
        ax.scatter(
            [x],
            [legend_y],
            s=15,
            c=color,
            marker=marker,
            edgecolors=INK,
            linewidths=0.40,
            zorder=3,
        )
        ax.text(
            x + 0.022,
            legend_y,
            label,
            ha="left",
            va="center",
            fontsize=5.80,
            color=color,
            fontweight="semibold",
        )


def _draw_row_label(ax: plt.Axes, label: str, color: str) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(
        0.98,
        0.50,
        label,
        ha="center",
        va="center",
        rotation=90,
        fontsize=5.90,
        color=color,
        fontweight="bold",
    )


def _draw_pair_glyph(
    ax: plt.Axes,
    *,
    center_x: float,
    y: float,
    perturbed_pair: bool,
    half_gap: float = 0.060,
    marker_scale: float = 1.0,
) -> None:
    left_x = center_x - half_gap
    right_x = center_x + half_gap
    color = ORANGE if perturbed_pair else PURPLE
    ax.annotate(
        "",
        xy=(right_x, y),
        xytext=(left_x, y),
        arrowprops={"arrowstyle": "<->", "color": color, "linewidth": 0.85},
        zorder=4,
    )
    ax.scatter(
        [left_x],
        [y],
        s=22 * marker_scale,
        c=BLUE,
        marker="o",
        edgecolors=INK,
        linewidths=0.55,
        zorder=5,
    )
    if perturbed_pair:
        ax.scatter(
            [right_x],
            [y],
            s=21 * marker_scale,
            facecolors=ORANGE,
            edgecolors=INK,
            marker="s",
            linewidths=0.55,
            zorder=5,
        )
    else:
        ax.scatter(
            [right_x],
            [y],
            s=24 * marker_scale,
            facecolors=PURPLE,
            edgecolors=INK,
            marker="^",
            linewidths=0.55,
            zorder=5,
        )


IR_COL_X = 0.295
DR_COL_X = 0.618
VERDICT_COL_X = 0.876


def _draw_metric_row(
    ax: plt.Axes,
    *,
    center_y: float,
    height: float,
    label: str,
    ir_trend: str,
    ir_sub: str,
    dr_trend: str,
    dr_sub: str,
    passed: bool,
) -> None:
    color = GREEN if passed else RED
    face = "#EEF8F5" if passed else "#FBF2F2"
    _rounded_box(
        ax,
        (0.035, center_y - 0.5 * height),
        0.93,
        height,
        face=face,
        edge="none",
        linewidth=0.0,
        radius=0.012,
    )
    ax.text(
        0.075,
        center_y,
        label,
        ha="left",
        va="center",
        rotation=90,
        fontsize=5.90,
        color=color,
        fontweight="bold",
    )
    wide_gap, tight_gap = 0.086, 0.048
    for col_x, name, trend, sub, perturbed_pair in (
        (IR_COL_X, "IR", ir_trend, ir_sub, True),
        (DR_COL_X, "DR", dr_trend, dr_sub, False),
    ):
        metric_color = ORANGE if name == "IR" else PURPLE
        # Arrow length mirrors the reported magnitude for this cell.
        half_gap = wide_gap if trend == "HIGH" else tight_gap
        _draw_pair_glyph(
            ax,
            center_x=col_x,
            y=center_y + 0.078,
            perturbed_pair=perturbed_pair,
            half_gap=half_gap,
            marker_scale=0.78,
        )
        tag_center_x = col_x - 0.047
        tag_center_y = center_y + 0.0005
        tag_width, tag_height = 0.076, 0.039
        tag = FancyBboxPatch(
            (
                tag_center_x - 0.5 * tag_width,
                tag_center_y - 0.5 * tag_height,
            ),
            tag_width,
            tag_height,
            boxstyle="round,pad=0.002,rounding_size=0.009",
            facecolor=metric_color,
            edgecolor="none",
            linewidth=0.0,
            transform=ax.transAxes,
            clip_on=False,
            zorder=4,
        )
        ax.add_patch(tag)
        ax.text(
            tag_center_x,
            center_y - 0.002,
            name,
            ha="center",
            va="center",
            fontsize=5.45,
            color=WHITE,
            fontweight="bold",
            zorder=5,
        )
        ax.text(
            col_x + 0.006,
            center_y - 0.002,
            trend,
            ha="left",
            va="center",
            fontsize=6.35,
            color=color,
            fontweight="bold",
        )
        ax.text(
            col_x,
            center_y - 0.068,
            sub,
            ha="center",
            va="center",
            fontsize=5.45,
            color=MID,
        )
    badge_w, badge_h = 0.145, 0.064
    _rounded_box(
        ax,
        (VERDICT_COL_X - 0.5 * badge_w, center_y - 0.5 * badge_h),
        badge_w,
        badge_h,
        face=color,
        edge="none",
        linewidth=0.0,
        radius=0.016,
        zorder=4,
    )
    ax.text(
        VERDICT_COL_X,
        center_y,
        "PASS" if passed else "FAIL",
        ha="center",
        va="center",
        fontsize=5.50,
        color=WHITE,
        fontweight="bold",
        zorder=5,
    )


def _draw_metric_block(
    ax: plt.Axes,
    *,
    fragile_center: float,
    robust_center: float,
) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    _rounded_box(
        ax,
        (0.01, 0.01),
        0.98,
        0.98,
        face=PANEL,
        edge="#84CBB8",
        linewidth=0.95,
        radius=0.025,
        zorder=-5,
    )
    ax.text(
        0.045,
        0.955,
        "(c)  ACPC diagnostics",
        ha="left",
        va="top",
        fontsize=7.5,
        color=INK,
        fontweight="semibold",
    )

    fragile_top = fragile_center + 0.150
    defs_y = fragile_top + 0.098
    header_y = fragile_top + 0.044
    ax.text(
        0.085,
        defs_y,
        "Invariance Radius (IR): visual sensitivity\n"
        "Distinction Rate (DR): state separation",
        ha="left",
        va="center",
        multialignment="left",
        fontsize=5.60,
        color=MID,
        linespacing=1.45,
    )
    ax.text(
        IR_COL_X,
        header_y,
        "IR  ↓",
        ha="center",
        va="center",
        fontsize=6.30,
        color=ORANGE,
        fontweight="bold",
    )
    ax.text(
        DR_COL_X,
        header_y,
        "DR  ↑",
        ha="center",
        va="center",
        fontsize=6.30,
        color=PURPLE,
        fontweight="bold",
    )

    _draw_metric_row(
        ax,
        center_y=fragile_center,
        height=0.300,
        label="FRAGILE",
        ir_trend="HIGH",
        ir_sub="wide radius",
        dr_trend="LOW",
        dr_sub="weak separation",
        passed=False,
    )
    _draw_metric_row(
        ax,
        center_y=robust_center,
        height=0.300,
        label="ROBUST",
        ir_trend="LOW",
        ir_sub="tight radius",
        dr_trend="HIGH",
        dr_sub="clear separation",
        passed=True,
    )


def _figure_arrow(
    fig: plt.Figure,
    source_ax: plt.Axes,
    target_ax: plt.Axes,
    *,
    label: str | None,
    source_inset: float = 0.016,
) -> None:
    source = source_ax.get_position()
    target = target_ax.get_position()
    y = 0.50 * (source.y0 + source.y1)
    start = (source.x1 - source_inset, y)
    end = (target.x0 + 0.016, y)
    fig.add_artist(
        FancyArrowPatch(
            start,
            end,
            transform=fig.transFigure,
            arrowstyle="->",
            mutation_scale=5.8,
            linewidth=1.05,
            color=MID,
            clip_on=False,
            zorder=30,
        )
    )
    if label:
        fig.text(
            0.5 * (start[0] + end[0]),
            y + 0.017,
            label,
            ha="center",
            va="bottom",
            fontsize=6.0,
            color=MID,
        )


def plot(out_pdf: Path, preview_png: Path, input_dir: Path) -> None:
    metadata = _validate_inputs(input_dir)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    preview_png.parent.mkdir(parents=True, exist_ok=True)

    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(6.70, 3.45), facecolor=WHITE)
        outer = fig.add_gridspec(
            1,
            3,
            left=0.012,
            right=0.988,
            bottom=0.025,
            top=0.975,
            width_ratios=[0.15, 0.56, 0.29],
            wspace=0.07,
        )

        input_ax = fig.add_subplot(outer[0, 0])
        _draw_input_block(input_ax, input_dir)

        middle_bg = fig.add_subplot(outer[0, 1])
        middle_bg.set_axis_off()
        _rounded_box(
            middle_bg,
            (0.005, 0.01),
            0.99,
            0.98,
            face=PANEL,
            edge="#B7C0C9",
            linewidth=0.95,
            radius=0.025,
            zorder=-5,
        )
        middle_bg.set_xlim(0, 1)
        middle_bg.set_ylim(0, 1)
        middle_bg.text(
            0.025,
            0.955,
            "(b)  Encoder and rollout representations",
            ha="left",
            va="top",
            fontsize=7.5,
            color=INK,
            fontweight="semibold",
        )
        middle = outer[0, 1].subgridspec(
            4,
            5,
            height_ratios=[0.42, 1.0, 1.0, 0.02],
            width_ratios=[0.10, 1.0, 0.40, 1.0, 0.015],
            hspace=0.20,
            wspace=0.08,
        )
        header_ax = fig.add_subplot(middle[0, :])
        fragile_label_ax = fig.add_subplot(middle[1, 0])
        robust_label_ax = fig.add_subplot(middle[2, 0])
        _draw_row_label(fragile_label_ax, "FRAGILE", RED)
        _draw_row_label(robust_label_ax, "ROBUST", GREEN)
        fail_encoder_ax = fig.add_subplot(middle[1, 1])
        fail_rollout_ax = fig.add_subplot(middle[1, 3])
        pass_encoder_ax = fig.add_subplot(middle[2, 1])
        pass_rollout_ax = fig.add_subplot(middle[2, 3])
        header_pos = header_ax.get_position()
        legend_fig_y = fail_encoder_ax.get_position().y1 + 0.058
        _draw_middle_header(
            header_ax,
            legend_y=(legend_fig_y - header_pos.y0) / header_pos.height,
        )
        _draw_three_token_panel(
            fail_encoder_ax, robust=False, rollout=False, seed=20260731
        )
        _draw_three_token_panel(
            fail_rollout_ax, robust=False, rollout=True, seed=20260732
        )
        _draw_three_token_panel(
            pass_encoder_ax, robust=True, rollout=False, seed=20260733
        )
        _draw_three_token_panel(
            pass_rollout_ax, robust=True, rollout=True, seed=20260734
        )
        fail_encoder_ax.set_title(
            "encoder space",
            fontsize=6.3,
            color=INK,
            fontweight="semibold",
            pad=3.0,
        )
        fail_rollout_ax.set_title(
            r"$H$-step rollout space",
            fontsize=6.3,
            color=INK,
            fontweight="semibold",
            pad=3.0,
        )

        # Per-row encoder -> rollout arrows group the four panels into two
        # stages read left to right.
        for enc_ax, roll_ax in (
            (fail_encoder_ax, fail_rollout_ax),
            (pass_encoder_ax, pass_rollout_ax),
        ):
            enc_pos = enc_ax.get_position()
            roll_pos = roll_ax.get_position()
            row_y = 0.5 * (enc_pos.y0 + enc_pos.y1)
            fig.add_artist(
                FancyArrowPatch(
                    (enc_pos.x1 + 0.0025, row_y),
                    (roll_pos.x0 - 0.0025, row_y),
                    transform=fig.transFigure,
                    arrowstyle="->",
                    mutation_scale=5.2,
                    linewidth=0.95,
                    color=MID,
                    clip_on=False,
                    zorder=30,
                )
            )
            fig.text(
                0.5 * (enc_pos.x1 + roll_pos.x0),
                row_y + 0.013,
                r"$H$-step rollout",
                ha="center",
                va="bottom",
                fontsize=5.8,
                color=MID,
                zorder=30,
            )
            fig.text(
                0.5 * (enc_pos.x1 + roll_pos.x0),
                row_y - 0.016,
                r"$\mathbf{a}_i$",
                ha="center",
                va="top",
                fontsize=6.0,
                color=MID,
                zorder=30,
            )


        # Row bands group the four panels into a fragile row and a robust
        # row; colors match the corresponding (c) cards.
        bg_pos = middle_bg.get_position()

        def _to_bg(x, y):
            return (
                (x - bg_pos.x0) / bg_pos.width,
                (y - bg_pos.y0) / bg_pos.height,
            )

        for label_ax, roll_ax, face in (
            (fragile_label_ax, fail_rollout_ax, "#FBF2F2"),
            (robust_label_ax, pass_rollout_ax, "#EEF8F5"),
        ):
            label_pos = label_ax.get_position()
            roll_pos = roll_ax.get_position()
            x0, y0 = _to_bg(label_pos.x0 - 0.002, roll_pos.y0 - 0.008)
            x1, y1 = _to_bg(roll_pos.x1 + 0.004, roll_pos.y1 + 0.004)
            x0 = max(x0, 0.022)
            x1 = min(x1, 0.984)
            _rounded_box(
                middle_bg,
                (x0, y0),
                x1 - x0,
                y1 - y0,
                face=face,
                edge="none",
                linewidth=0.0,
                radius=0.015,
                zorder=-4,
            )

        metric_ax = fig.add_subplot(outer[0, 2])
        metric_pos = metric_ax.get_position()

        def _row_center(panel_ax) -> float:
            pos = panel_ax.get_position()
            fig_y = 0.5 * (pos.y0 + pos.y1)
            return (fig_y - metric_pos.y0) / metric_pos.height

        _draw_metric_block(
            metric_ax,
            fragile_center=_row_center(fail_encoder_ax),
            robust_center=_row_center(pass_encoder_ax),
        )

        _figure_arrow(fig, input_ax, middle_bg, label="Encoder")
        _figure_arrow(
            fig,
            middle_bg,
            metric_ax,
            label=None,
            source_inset=0.002,
        )

        pdf_metadata = {
            "Title": "Figure 1: three-block ACPC, IR, and DR overview",
            "Subject": (
                "Real PushT inputs, a 2-D neighborhood illustration, and "
                "pair-to-checkpoint IR/DR aggregation"
            ),
            "Creator": "paper1/scripts/plot_acpc_ir_dr_overview.py",
            "Keywords": (
                "PushT; ACPC; Invariance Radius; Distinction Rate; "
                f"source episode {metadata['clean']['episode']}"
            ),
        }
        fig.savefig(
            out_pdf,
            dpi=320,
            facecolor=WHITE,
            bbox_inches="tight",
            pad_inches=0.02,
            metadata=pdf_metadata,
        )
        fig.savefig(
            preview_png,
            dpi=300,
            facecolor=WHITE,
            bbox_inches="tight",
            pad_inches=0.02,
        )
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--preview", type=Path, default=DEFAULT_PREVIEW)
    args = parser.parse_args()
    plot(args.out.expanduser(), args.preview.expanduser(), args.input_dir.expanduser())
    print(f"wrote {args.out}")
    print(f"wrote {args.preview}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
