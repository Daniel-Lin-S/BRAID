"""Render three-panel BRAID structure comparisons from summary rows.

Input is aggregate or session summary rows with mean, uncertainty, and case
axes. Output is one PNG per behavior/neural target and CC/R2/MSE metric,
with horizon ticks in steps and milliseconds and labeled clipped observations.
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
FIGURE_HEIGHT_INCHES = 9
MILLISECONDS_PER_SECOND = 1000
CLIPPED_MARKER_SPACING = 0.03
DYNAMIC_COMPARISON_STYLES = {
    "linear_mlp": ("-", "o"),
    "nonlinear_mlp": ("--", "s"),
    "lstm": (":", "D"),
}
DYNAMIC_COMPARISON_WIDTH_INCHES = 12
DYNAMIC_LAYOUT_RECT = (0.02, 0.09, 0.76, 0.91)
LABEL_STACK_POINTS = 14
BOUNDARY_PADDING = 0.05
MARKERS = ("o", "s", "D", "^")
MARKER_SIZE = 8
ERROR_CAP_SIZE = 6
ANNOTATION_FONT_MINIMUM = 16
LEGEND_LEFT = 0.79
SHARED_X_LABEL_X = 0.40
SHARED_X_LABEL_Y = 0.02
LAYOUT_RECT = (0.02, 0.09, 0.78, 0.91)


def zoom_limits(
    rows: list[dict], multiple: float, error_key: str = "sem",
) -> tuple[float, float, float, float]:
    """Find shared bounds using median ± multiple IQR.

    Parameters
    ----------
    rows : list of dict
        Finite or missing means with an optional uncertainty field.
    multiple : float
        Positive multiplier for the interquartile range.
    error_key : str, optional
        Row field containing the visible uncertainty; default is "sem".

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
        error_value = row.get(error_key)
        error = (
            error_value
            if error_value is not None and np.isfinite(error_value)
            else 0.0
        )
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
    linestyle: str, label: str,
    limits: tuple[float, float, float, float],
    series_number: int, series_count: int, style: dict,
    error_key: str,
) -> None:
    """Draw inlier means, uncertainty, and labeled boundary markers."""
    lower, upper, robust_low, robust_high = limits
    x = np.asarray([row["horizon"] for row in rows], dtype=float)
    y = np.asarray([
        row["mean"] if row["mean"] is not None else np.nan
        for row in rows
    ], dtype=float)
    inlier = np.isfinite(y) & (y >= robust_low) & (y <= robust_high)
    axis.plot(
        x, np.where(inlier, y, np.nan), color=color,
        linestyle=linestyle, marker=marker, label=label,
        markersize=MARKER_SIZE,
    )
    error_indices = [
        index for index, row in enumerate(rows)
        if inlier[index] and row.get(error_key) is not None
        and np.isfinite(row[error_key])
    ]
    if error_indices:
        axis.errorbar(
            x[error_indices], y[error_indices],
            yerr=[rows[index][error_key] for index in error_indices],
            fmt="none", ecolor=color, capsize=ERROR_CAP_SIZE,
        )
    for index in np.flatnonzero(np.isfinite(y) & ~inlier):
        high = y[index] > robust_high
        edge = upper if high else lower
        offset = series_number - (series_count - 1) / 2
        marker_x = x[index] * (
            1 + offset * CLIPPED_MARKER_SPACING
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
            fontsize=max(
                ANNOTATION_FONT_MINIMUM, style["tick_font"] * 0.75
            ),
            clip_on=False,
        )


def _figure(
    rows: list[dict], target: str, metric: str, style: dict,
    sample_rate: float, multiple: float, show_milliseconds: bool,
    error_key: str = "sem", uncertainty_label: str = "Mean ± SEM",
    context_label: str | None = None,
) -> object:
    """Build one target/metric figure with shared robust y-limits.

    Parameters
    ----------
    rows : list of dict
        Aggregate rows for one complete comparison suite.
    target, metric : str
        Data target and metric selected for this figure.
    style : dict
        Validated shared presentation mapping.
    sample_rate : float
        Positive sample rate in Hz.
    multiple : float
        IQR zoom multiplier.
    show_milliseconds : bool
        Include milliseconds below each horizon tick.
    error_key : str, optional
        Row field holding visible uncertainty; default is "sem".
    uncertainty_label : str, optional
        Description appended to the figure title; default is "Mean ± SEM".
    context_label : str, optional
        Session label for a session-level figure; default is None.
    """
    selected = [
        row for row in rows
        if row["target"] == target and row["metric"] == metric
    ]
    limits = zoom_limits(selected, multiple, error_key)
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
                MARKERS[number], "-", label, limits, number, len(pairs),
                style, error_key,
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
        axis.set_title(
            PANEL_TITLES[dynamics], fontsize=style["label_font"],
        )
        axis.grid(alpha=0.2)
        axis.tick_params(labelsize=style["tick_font"])
    axes[0].set_ylabel(
        f"{target.capitalize()} {METRIC_LABELS[metric]}",
        fontsize=style["label_font"],
    )
    figure.supxlabel(
        "Forecast horizon (steps / ms)", x=SHARED_X_LABEL_X,
        y=SHARED_X_LABEL_Y, fontsize=style["label_font"],
    )
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="center left", bbox_to_anchor=(LEGEND_LEFT, 0.5),
        frameon=False,
        fontsize=style["legend_font"],
    )
    title = f"{target.capitalize()} {METRIC_LABELS[metric]} by horizon"
    if context_label is not None:
        title = f"{context_label} — {title}"
    figure.suptitle(
        f"{title}\n{uncertainty_label}",
        fontsize=style["title_font"],
    )
    figure.tight_layout(rect=LAYOUT_RECT)
    return figure


def dynamic_comparison_selector(options: dict) -> tuple[str, str]:
    """Validate the mapping selection for a dynamics comparison.

    Parameters
    ----------
    options : dict
        structure_horizons settings containing dynamic_comparison.

    Returns
    -------
    tuple of str
        Encoder and paired-decoder structure names.
    """
    selector = options.get("dynamic_comparison")
    if not isinstance(selector, dict):
        raise ValueError(
            "structure_horizons.dynamic_comparison must be a mapping."
        )
    expected = {"encoder", "decoder"}
    if set(selector) != expected:
        raise ValueError(
            "structure_horizons.dynamic_comparison must contain exactly "
            f"{sorted(expected)}."
        )
    encoder = selector["encoder"]
    decoder = selector["decoder"]
    if encoder not in MAPPING_LEVELS:
        raise ValueError(
            "Expected structure_horizons.dynamic_comparison.encoder to be "
            f"one of {MAPPING_LEVELS}, got {encoder!r}."
        )
    if decoder not in MAPPING_LEVELS:
        raise ValueError(
            "Expected structure_horizons.dynamic_comparison.decoder to be "
            f"one of {MAPPING_LEVELS}, got {decoder!r}."
        )
    return encoder, decoder


def dynamic_comparison_directory(options: dict) -> str:
    """Build the stable output directory name from the selected mappings."""
    encoder, decoder = dynamic_comparison_selector(options)
    return f"dynamics_encoder_{encoder}_decoder_{decoder}"


def _dynamic_figure(
    rows: list[dict], target: str, metric: str, style: dict,
    sample_rate: float, multiple: float, show_milliseconds: bool,
    selector: tuple[str, str],
) -> object:
    """Build one selected-mapping comparison across all dynamics.

    Parameters
    ----------
    rows : list of dict
        Aggregate structure-summary rows for every dynamics choice.
    target, metric : str
        Target and metric for the output figure.
    style : dict
        Validated shared presentation mapping.
    sample_rate : float
        Positive spike sample rate in Hz.
    multiple : float
        Positive IQR multiplier for clipping.
    show_milliseconds : bool
        Include milliseconds below each horizon tick.
    selector : tuple of str
        Selected encoder and paired-decoder structure names.
    """
    encoder, decoder = selector
    selected = [
        row for row in rows
        if row["target"] == target and row["metric"] == metric
        and row["encoder"] == encoder and row["decoder"] == decoder
    ]
    limits = zoom_limits(selected, multiple)
    horizons = sorted({row["horizon"] for row in selected})
    figure, axis = plt.subplots(
        figsize=(DYNAMIC_COMPARISON_WIDTH_INCHES, FIGURE_HEIGHT_INCHES),
    )
    for number, dynamics in enumerate(DYNAMICS):
        curve = _case_rows(selected, dynamics, encoder, decoder)
        if not curve:
            raise ValueError(
                f"Missing structure summary for {dynamics}, "
                f"{encoder}, {decoder}."
            )
        linestyle, marker = DYNAMIC_COMPARISON_STYLES[dynamics]
        _draw_curve(
            axis, curve, style["horizon_colors"][number], marker, linestyle,
            PANEL_TITLES[dynamics], limits, number, len(DYNAMICS), style,
            "sem",
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
    axis.set_ylabel(
        f"{target.capitalize()} {METRIC_LABELS[metric]}",
        fontsize=style["label_font"],
    )
    axis.tick_params(labelsize=style["tick_font"])
    axis.grid(alpha=0.2)
    figure.supxlabel(
        "Forecast horizon (steps / ms)", x=0.40, y=SHARED_X_LABEL_Y,
        fontsize=style["label_font"],
    )
    figure.legend(
        *axis.get_legend_handles_labels(), title="Dynamics",
        loc="center left", bbox_to_anchor=(0.77, 0.5), frameon=False,
        fontsize=style["legend_font"], title_fontsize=style["legend_font"],
    )
    title = (
        f"{target.capitalize()} {METRIC_LABELS[metric]} by horizon\n"
        f"{encoder.capitalize()} encoder, {decoder} paired decoders; "
        "mean ± SEM across sessions"
    )
    figure.suptitle(title, fontsize=style["title_font"])
    figure.tight_layout(rect=DYNAMIC_LAYOUT_RECT)
    return figure


def render_dynamic_comparison_figures(
    destination: Path, rows: list[dict], settings: dict,
    sample_rate: float, rendered: set[str] | None = None,
) -> None:
    """Publish six selected-mapping aggregate dynamics comparisons."""
    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError(
            f"Expected positive sample rate, got {sample_rate}."
        )
    options = settings["structure_horizons"]
    selector = dynamic_comparison_selector(options)
    directory = dynamic_comparison_directory(options)
    style = presentation(settings["presentation"])
    for target in TARGETS:
        for metric in METRICS:
            name = f"dynamics_{target}_{metric}"
            render_key = f"{directory}/{name}"
            if rendered is not None and render_key in rendered:
                continue
            selected = [
                row for row in rows
                if row["target"] == target and row["metric"] == metric
                and row["encoder"] == selector[0]
                and row["decoder"] == selector[1]
            ]
            if not any(row["mean"] is not None for row in selected):
                LOGGER.warning("No finite results for %s.", name)
                continue
            figure = _dynamic_figure(
                rows, target, metric, style, sample_rate,
                options["zoom_iqr_multiple"], options["show_milliseconds"],
                selector,
            )
            signature = fingerprint({
                "selected": selected,
                "selector": selector,
                "dynamic_styles": DYNAMIC_COMPARISON_STYLES,
                "sample_rate": sample_rate,
                "options": options,
            })
            save_figure(
                figure, destination / directory / f"{name}.png", style,
                signature,
            )
            if rendered is not None:
                rendered.add(render_key)


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
    _render_figures(
        destination, rows, settings, sample_rate, rendered,
        error_key="sem", uncertainty_label="Mean ± SEM across sessions",
    )


def render_session_structure_figures(
    destination: Path, rows: list[dict], settings: dict,
    sample_rate: float, rendered: set[str] | None = None,
) -> None:
    """Publish six fold-mean figures for every requested session.

    Parameters
    ----------
    destination : Path
        Parent directory for one subdirectory per session.
    rows : list of dict
        Per-session fold summaries with ``std`` uncertainty.
    settings : dict
        Structure plotting options and inherited presentation settings.
    sample_rate : float
        Positive spike sample rate in Hz.
    rendered : set of str, optional
        Invocation-local qualified figure names already drawn; default None.
    """
    sessions = sorted({row["session"] for row in rows})
    if not sessions:
        raise ValueError("Cannot render session figures without sessions.")
    for session in sessions:
        _render_figures(
            destination / session,
            [row for row in rows if row["session"] == session],
            settings, sample_rate, rendered, error_key="std",
            uncertainty_label="Fold mean ± sample SD", context_label=session,
            namespace=f"session/{session}",
        )


def _render_figures(
    destination: Path, rows: list[dict], settings: dict,
    sample_rate: float, rendered: set[str] | None, error_key: str,
    uncertainty_label: str, context_label: str | None = None,
    namespace: str | None = None,
) -> None:
    """Render a six-figure suite using one shared uncertainty definition."""
    options = settings["structure_horizons"]
    multiple = options["zoom_iqr_multiple"]
    style = presentation(settings["presentation"])
    for target in TARGETS:
        for metric in METRICS:
            name = f"structure_{target}_{metric}"
            render_key = f"{namespace}/{name}" if namespace else name
            if rendered is not None and render_key in rendered:
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
                multiple, options["show_milliseconds"], error_key,
                uncertainty_label, context_label,
            )
            signature = fingerprint(dict(
                selected=selected, options=options,
                sample_rate=sample_rate, error_key=error_key,
                uncertainty_label=uncertainty_label,
                context_label=context_label,
            ))
            save_figure(
                figure, destination / f"{name}.png", style, signature,
            )
            if rendered is not None:
                rendered.add(render_key)
