"""Render three-panel BRAID structure comparisons from summary rows.

Input is structure_summary.csv-equivalent rows with mean, SEM and case axes.
Output is one PNG per behavior/neural target and CC/R2/MSE metric, with
horizon ticks in steps and milliseconds and labeled clipped observations.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .cache import fingerprint
from .presentation import METRIC_LABELS, presentation, save_figure
from .structure_sweep import DYNAMICS, MAPPING_LEVELS

LOGGER = logging.getLogger(__name__)
METRICS = ("cc", "r2", "mse")
TARGETS = ("behavior", "neural")
PANEL_TITLES = {
    "linear_mlp": "Linear MLP",
    "nonlinear_mlp": "ReLU MLP (1×64)",
    "lstm": "LSTM",
}
PANEL_WIDTH_INCHES = 8
FIGURE_HEIGHT_INCHES = 7
MILLISECONDS_PER_SECOND = 1000
CLIPPED_MARKER_SPACING = 0.03
LABEL_STACK_POINTS = 14
BOUNDARY_PADDING = 0.05
MARKERS = ("o", "s", "D", "^")


def zoom_limits(
    rows: list[dict], multiple: float,
) -> tuple[float, float, float, float]:
    """Find shared bounds using median ± multiple IQR.

    Parameters
    ----------
    rows : list of dict
        Finite or missing aggregate means with optional SEM.
    multiple : float
        Positive multiplier for the interquartile range.

    Returns
    -------
    tuple of float
        Visible low/high and raw robust low/high boundaries.
    """
    if not np.isfinite(multiple) or multiple <= 0:
        raise ValueError(f"Expected positive IQR multiplier, got {multiple}.")
    values = np.asarray([
        row["mean"] for row in rows
        if row["mean"] is not None and np.isfinite(row["mean"])
    ], dtype=float)
    if not values.size:
        raise ValueError("Cannot plot structure metrics without finite means.")
    median = float(np.median(values))
    quartiles = np.percentile(values, [25, 75])
    iqr = float(quartiles[1] - quartiles[0])
    if iqr == 0:
        LOGGER.warning(
            "Structure figure has zero IQR; clipping means outside "
            "the median."
        )
    robust_low = median - multiple * iqr
    robust_high = median + multiple * iqr
    inlier_bounds = []
    for row in rows:
        mean = row["mean"]
        if mean is None or not np.isfinite(mean):
            continue
        if not robust_low <= mean <= robust_high:
            continue
        sem = row["sem"]
        error = sem if sem is not None and np.isfinite(sem) else 0.0
        inlier_bounds.extend((mean - error, mean + error))
    if not inlier_bounds:
        raise ValueError("Structure figure has no finite inlier means.")
    lower = min(inlier_bounds)
    upper = max(inlier_bounds)
    if iqr:
        lower = min(lower, robust_low)
        upper = max(upper, robust_high)
    span = upper - lower
    if span == 0:
        span = max(abs(median), 1.0) * BOUNDARY_PADDING
    padding = span * BOUNDARY_PADDING
    return lower - padding, upper + padding, robust_low, robust_high


def _case_rows(
    rows: list[dict], dynamics: str, encoder: str, decoder: str,
) -> list[dict]:
    """Select one curve in horizon order."""
    return sorted(
        [
            row for row in rows
            if row["dynamics"] == dynamics
            and row["encoder"] == encoder
            and row["decoder"] == decoder
        ],
        key=lambda row: row["horizon"],
    )


def _draw_curve(
    axis: object, rows: list[dict], color: str, marker: str,
    label: str, limits: tuple[float, float, float, float],
    series_number: int,
) -> None:
    """Draw inlier means and SEM plus labeled boundary markers."""
    lower, upper, robust_low, robust_high = limits
    x = np.asarray([row["horizon"] for row in rows], dtype=float)
    y = np.asarray([
        row["mean"] if row["mean"] is not None else np.nan
        for row in rows
    ], dtype=float)
    inlier = np.isfinite(y) & (y >= robust_low) & (y <= robust_high)
    axis.plot(
        x, np.where(inlier, y, np.nan), color=color,
        marker=marker, label=label,
    )
    error_indices = [
        index for index, row in enumerate(rows)
        if inlier[index] and row["sem"] is not None
        and np.isfinite(row["sem"])
    ]
    if error_indices:
        axis.errorbar(
            x[error_indices], y[error_indices],
            yerr=[rows[index]["sem"] for index in error_indices],
            fmt="none", ecolor=color, capsize=4,
        )
    for index in np.flatnonzero(np.isfinite(y) & ~inlier):
        high = y[index] > robust_high
        edge = upper if high else lower
        marker_x = x[index] * (
            1 + (series_number - 1.5) * CLIPPED_MARKER_SPACING
        )
        axis.scatter(
            [marker_x], [edge], color=color,
            marker="^" if high else "v", clip_on=False,
        )
        axis.annotate(
            f"{y[index]:.3g}", (marker_x, edge),
            xytext=(5, (-10 - series_number * LABEL_STACK_POINTS)
                    if high else (8 + series_number * LABEL_STACK_POINTS)),
            textcoords="offset points", color=color,
            fontsize=9, clip_on=False,
        )


def _figure(
    rows: list[dict], target: str, metric: str, style: dict,
    sample_rate: float, multiple: float, show_milliseconds: bool,
) -> object:
    """Build one target/metric figure with shared robust y-limits."""
    selected = [
        row for row in rows
        if row["target"] == target and row["metric"] == metric
    ]
    limits = zoom_limits(selected, multiple)
    horizons = sorted({row["horizon"] for row in selected})
    figure, axes = plt.subplots(
        1, len(DYNAMICS), sharey=True,
        figsize=(PANEL_WIDTH_INCHES * len(DYNAMICS),
                 FIGURE_HEIGHT_INCHES),
    )
    pairs = [
        (encoder, decoder) for encoder in MAPPING_LEVELS
        for decoder in MAPPING_LEVELS
    ]
    for axis, dynamics in zip(axes, DYNAMICS):
        for number, (encoder, decoder) in enumerate(pairs):
            curve = _case_rows(selected, dynamics, encoder, decoder)
            if not curve:
                raise ValueError(
                    f"Missing structure summary for {dynamics}, "
                    f"{encoder}, {decoder}."
                )
            label = f"Encoder {encoder}, decoders {decoder}"
            _draw_curve(
                axis, curve, style["horizon_colors"][number],
                MARKERS[number], label, limits, number,
            )
        labels = [
            f"{step}\n{step * MILLISECONDS_PER_SECOND / sample_rate:g}"
            if show_milliseconds else str(step)
            for step in horizons
        ]
        axis.set_xscale("log", base=2)
        axis.set_xticks(horizons, labels)
        axis.set_xlim(min(horizons) / 1.2, max(horizons) * 1.2)
        axis.set_ylim(limits[:2])
        axis.set_xlabel("Forecast horizon (steps / ms)")
        axis.set_title(PANEL_TITLES[dynamics])
        axis.grid(alpha=0.2)
        axis.tick_params(labelsize=style["tick_font"])
    axes[0].set_ylabel(
        f"{target.capitalize()} {METRIC_LABELS[metric]}"
    )
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="center left",
        bbox_to_anchor=(0.99, 0.5), frameon=False,
        fontsize=style["legend_font"],
    )
    figure.suptitle(
        f"{target.capitalize()} {METRIC_LABELS[metric]} by horizon",
        fontsize=style["title_font"],
    )
    figure.tight_layout(rect=(0, 0, 0.84, 0.93))
    return figure


def render_structure_figures(
    destination: Path, rows: list[dict], settings: dict,
    sample_rate: float, rendered: set[str] | None = None,
) -> None:
    """Publish six aggregate figures when finite results are available.

    Parameters
    ----------
    destination : Path
        Analysis plot directory.
    rows : list of dict
        One structure summary per case, horizon, target and metric.
    settings : dict
        Structure plotting options and inherited presentation settings.
    sample_rate : float
        Positive spike sample rate in Hz.
    rendered : set of str, optional
        Invocation-local figure names already drawn; default None.
    """
    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError(
            f"Expected positive sample rate, got {sample_rate}."
        )
    options = settings["structure_horizons"]
    multiple = options["zoom_iqr_multiple"]
    style = presentation(settings["presentation"])
    for target in TARGETS:
        for metric in METRICS:
            name = f"structure_{target}_{metric}"
            if rendered is not None and name in rendered:
                continue
            selected = [
                row for row in rows
                if row["target"] == target and row["metric"] == metric
            ]
            if not any(row["mean"] is not None for row in selected):
                LOGGER.warning("No finite results for %s.", name)
                continue
            figure = _figure(
                rows, target, metric, style, sample_rate,
                multiple, options["show_milliseconds"],
            )
            signature = fingerprint(dict(
                selected=selected, options=options,
                sample_rate=sample_rate,
            ))
            save_figure(
                figure, destination / f"{name}.png", style, signature,
            )
            if rendered is not None:
                rendered.add(name)
