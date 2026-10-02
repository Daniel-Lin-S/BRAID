"""Shared figure styling and atomic PNG publication.

Inputs are plotting.presentation settings and Matplotlib figures. PNGs
contain a rendering signature when supplied, allowing history-only redraws
to reuse unchanged images without writing duplicate numeric histories.
"""

import os
from pathlib import Path
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import is_color_like
from matplotlib.figure import Figure
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np

from .configuration import CONFIGURATION, read_yaml

METRIC_LABELS = {"loss": "Loss", "mse": "MSE", "r2": "R²", "cc": "CC"}
TIGHT_BBOX_PAD_INCHES = 0.1
OUTLIER_STYLE_FIELDS = (
    "zoom_padding_fraction", "label_significant_figures",
    "label_offset_points", "label_stack_points", "annotation_font",
)


def presentation(settings: dict | None = None) -> dict:
    """Resolve and validate shared rendering settings, without writing files.

    Parameters
    ----------
    settings : dict, optional
        Explicit presentation mapping; default uses the shared YAML.
    """
    style = (
        settings
        or read_yaml(CONFIGURATION / "plotting" / "style.yaml")["presentation"]
    )
    for key in (
        "title_font",
        "label_font",
        "tick_font",
        "legend_font",
        "width_inches",
        "panel_height_inches",
        "dpi",
    ):
        if not isinstance(style[key], (int, float)) or not (
            0 < style[key] < float("inf")
        ):
            raise ValueError(
                f"Expected positive finite {key}, got {style[key]}"
            )
    for key in ("single_size", "horizon_size"):
        dimensions = style[key]
        if len(dimensions) != 2 or any(
            not isinstance(value, (int, float)) or not 0 < value < float("inf")
            for value in dimensions
        ):
            raise ValueError(f"Expected two positive finite inches for {key}.")
    pixels = style["max_canvas_pixels"]
    if type(pixels) is not int or pixels <= 0:
        raise ValueError(
            "presentation.max_canvas_pixels must be a positive integer."
        )
    for key in ("pair_colors", "horizon_colors", "group_colors"):
        if not style[key] or not all(is_color_like(c) for c in style[key]):
            raise ValueError(f"Expected valid colors in {key}.")
    for key in ("group_linestyles", "group_markers"):
        if not style[key] or any(
            not isinstance(value, str) or not value
            for value in style[key]
        ):
            raise ValueError(f"Expected nonempty strings in {key}.")
    if len(style["pair_colors"]) != 2:
        raise ValueError("Expected exactly two train/validation or x/y colors.")
    return style


def outlier_rendering(settings: dict | None = None) -> dict:
    """Resolve global finite-outlier display defaults without writing files."""
    style = settings or read_yaml(
        CONFIGURATION / "plotting" / "style.yaml"
    )["outlier_rendering"]
    if set(style) != set(OUTLIER_STYLE_FIELDS):
        raise ValueError(
            "outlier_rendering must contain all supported display "
            "settings."
        )
    padding = style["zoom_padding_fraction"]
    if not isinstance(padding, (int, float)) or not 0 < padding < 1:
        raise ValueError(
            "Expected outlier_rendering.zoom_padding_fraction between 0 "
            f"and 1, got {padding!r}."
        )
    digits = style["label_significant_figures"]
    if type(digits) is not int or digits < 1:
        raise ValueError(
            "Expected positive integer "
            "outlier_rendering.label_significant_figures."
        )
    for key in ("label_offset_points", "label_stack_points"):
        if type(style[key]) is not int or style[key] < 0:
            raise ValueError(
                "Expected a nonnegative integer "
                f"outlier_rendering.{key}."
            )
    font = style["annotation_font"]
    if (
        type(font) not in (int, float)
        or not 0 < font < float("inf")
    ):
        raise ValueError(
            "outlier_rendering.annotation_font must be positive and "
            "finite."
        )
    return style


def style_axis(axis: object, style: dict) -> None:
    """Apply common label/tick typography to an existing axes object."""
    axis.xaxis.label.set_size(style["label_font"])
    axis.yaxis.label.set_size(style["label_font"])
    axis.title.set_size(style["label_font"])
    axis.tick_params(labelsize=style["tick_font"])
    for coordinate in (axis.xaxis, axis.yaxis):
        coordinate.get_offset_text().set_fontsize(style["tick_font"])
    axis.grid(alpha=0.2)


def parameter_axes(settings: dict) -> dict:
    """Resolve and validate shared comparison x-axis configurations."""
    axes = settings.get("parameter_axes")
    if axes is None:
        axes = read_yaml(CONFIGURATION / "plotting" / "style.yaml")[
            "parameter_axes"
        ]
    expected = {"horizon", "nx", "population_scale"}
    if not isinstance(axes, dict) or set(axes) != expected:
        raise ValueError(
            "plotting.parameter_axes must configure horizon, nx, and "
            "population_scale."
        )
    for name, configuration in axes.items():
        if not isinstance(configuration, dict):
            raise ValueError(f"parameter_axes.{name} must be a mapping.")
        if configuration.get("scale") == "linear":
            if set(configuration) != {"scale"}:
                raise ValueError(
                    f"parameter_axes.{name} linear scale has extra fields."
                )
        elif configuration.get("scale") == "log":
            base = configuration.get("base")
            if (
                set(configuration) != {"scale", "base"}
                or type(base) not in (int, float)
                or not np.isfinite(base)
                or base <= 1
            ):
                raise ValueError(
                    f"parameter_axes.{name} log base must exceed 1."
                )
        else:
            raise ValueError(
                f"Unsupported parameter_axes.{name} scale."
            )
    return axes


def metric_axes(settings: dict) -> dict:
    """Resolve globally configured evaluation metric axes for an analysis."""
    if "metric_axes" in settings:
        return settings["metric_axes"]
    return read_yaml(CONFIGURATION / "plotting" / "style.yaml")[
        "metric_axes"
    ]


def metric_axis_settings(settings: dict, metric: str) -> dict:
    """Validate and return one shared analysis metric-axis configuration."""
    if not isinstance(settings, dict):
        raise ValueError("plotting.metric_axes must be a mapping.")
    configuration = settings.get(metric, {})
    if not configuration:
        return {}
    if not isinstance(configuration, dict):
        raise ValueError(f"Expected {metric} metric axis to be a mapping.")
    if metric == "r2":
        required = {
            "scale", "uncertainty_ceiling", "padding_fraction",
            "ticks", "label",
        }
        if set(configuration) != required or (
            configuration["scale"] != "log_one_minus"
        ):
            raise ValueError("Unsupported r2 metric axis configuration.")
        ceiling = configuration["uncertainty_ceiling"]
        ticks = configuration["ticks"]
        padding = configuration["padding_fraction"]
        if (
            type(padding) not in (float, int)
            or not np.isfinite(padding)
            or not 0 < padding < 1
        ):
            raise ValueError("r2 padding_fraction must be between 0 and 1.")
        if (
            type(ceiling) not in (float, int)
            or not np.isfinite(ceiling)
            or not 0 < ceiling < 1
        ):
            raise ValueError(
                "r2 uncertainty_ceiling must be between 0 and 1."
            )
        if (
            not isinstance(ticks, list)
            or len(ticks) < 2
            or any(
                type(value) not in (int, float)
                or not np.isfinite(value)
                or value >= 1
                for value in ticks
            )
            or ticks != sorted(set(ticks))
        ):
            raise ValueError(
                "r2 ticks must be unique ascending finite values below 1."
            )
    elif metric == "mse":
        required = {"scale", "uncertainty_lower_fraction", "label"}
        if set(configuration) != required or configuration["scale"] != "log":
            raise ValueError("Unsupported mse metric axis configuration.")
        fraction = configuration["uncertainty_lower_fraction"]
        if (
            type(fraction) not in (float, int)
            or not np.isfinite(fraction)
            or not 0 < fraction < 1
        ):
            raise ValueError(
                "mse uncertainty_lower_fraction must be between 0 and 1."
            )
    else:
        raise ValueError(f"Unsupported metric axis for {metric!r}.")
    if not isinstance(configuration["label"], str) or not (
        configuration["label"]
    ):
        raise ValueError(f"{metric} metric-axis label must be nonempty.")
    return configuration


def _log_one_minus_r2(values: np.ndarray) -> np.ndarray:
    """Map R² values below one to signed log distance from one."""
    values = np.asarray(values, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.log1p(-values)


def _inverse_log_one_minus_r2(values: np.ndarray) -> np.ndarray:
    """Map signed log distance to raw R² values."""
    return -np.expm1(-np.asarray(values, dtype=float))



def r2_display_limits(rows: list[dict], settings: dict) -> tuple[float, float]:
    """Bound an R² axis by finite means and representable uncertainty."""
    configuration = metric_axis_settings(settings, "r2")
    ceiling = configuration["uncertainty_ceiling"]
    bounds = []
    for row in rows:
        mean = row.get("mean")
        if mean is None or not np.isfinite(mean):
            continue
        if mean >= ceiling:
            raise ValueError(
                f"R² mean {mean:g} reaches axis ceiling {ceiling:g}."
            )
        error = row.get("std", row.get("sem"))
        if error is None or not np.isfinite(error):
            error = 0.0
        if error < 0:
            raise ValueError("R² uncertainty must be nonnegative.")
        bounds.extend((mean - error, mean))
        if mean + error < ceiling:
            bounds.append(mean + error)
    if not bounds:
        raise ValueError("R² axis has no finite visible means.")
    transformed = _log_one_minus_r2(np.asarray(bounds))
    lower, upper = float(np.min(transformed)), float(np.max(transformed))
    span = upper - lower
    if span <= 0:
        raise ValueError("R² axis has no finite visible range.")
    padding = span * configuration["padding_fraction"]
    return (
        float(_inverse_log_one_minus_r2(lower - padding)),
        min(
            float(_inverse_log_one_minus_r2(upper + padding)),
            ceiling,
        ),
    )


def configure_metric_axis(
    axis: object, metric: str, settings: dict,
) -> None:
    """Apply shared evaluation-metric scaling with data-driven limits."""
    configuration = metric_axis_settings(settings, metric)
    if not configuration:
        return
    lower, upper = axis.get_ylim()
    if metric == "r2":
        ceiling = configuration["uncertainty_ceiling"]
        upper = min(upper, ceiling)
        if lower >= upper:
            raise ValueError("R² plot has no data below the axis ceiling.")
        axis.set_yscale(
            "function",
            functions=(_log_one_minus_r2, _inverse_log_one_minus_r2),
        )
        axis.yaxis.set_major_locator(
            FixedLocator(configuration["ticks"])
        )
        axis.yaxis.set_major_formatter(
            FuncFormatter(lambda value, _position: f"{value:g}")
        )
    else:
        positive = np.concatenate([
            np.asarray(line.get_ydata(), dtype=float).ravel()
            for line in axis.lines
        ])
        positive = positive[np.isfinite(positive) & (positive > 0)]
        if not positive.size:
            raise ValueError("Log MSE axis has no positive plotted means.")
        fraction = configuration["uncertainty_lower_fraction"]
        if lower <= 0:
            lower = float(np.min(positive)) * fraction
        axis.set_yscale("log")
    axis.set_ylim(lower, upper)
    axis.set_ylabel(configuration["label"])


def plot_metric_uncertainty(
    axis: object, x: np.ndarray, y: np.ndarray,
    errors: np.ndarray, color: str, metric: str, settings: dict,
    capsize: float, fontsize: float,
) -> None:
    """Draw uncertainty, labeling interval ends outside metric domains."""
    configuration = metric_axis_settings(settings, metric)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    errors = np.asarray(errors, dtype=float)
    if x.shape != y.shape or y.shape != errors.shape:
        raise ValueError("Metric uncertainty arrays must have equal shape.")
    if not np.isfinite(errors).all() or np.any(errors < 0):
        raise ValueError("Metric uncertainties must be finite and nonnegative.")
    lower = errors.copy()
    upper = errors.copy()
    for index, (mean, error) in enumerate(zip(y, errors)):
        if metric == "r2" and configuration:
            ceiling = configuration["uncertainty_ceiling"]
            if mean >= ceiling:
                raise ValueError(
                    f"R² mean {mean:g} reaches axis ceiling {ceiling:g}."
                )
            upper[index] = min(error, ceiling - mean)
            clipped = mean + error > ceiling
            endpoint = mean + error
            direction = "↑"
        elif metric == "mse" and configuration:
            if mean <= 0:
                raise ValueError(
                    f"Log MSE axis requires positive mean, got {mean:g}."
                )
            floor = mean * configuration["uncertainty_lower_fraction"]
            clipped = mean - error <= 0
            if clipped:
                lower[index] = mean - floor
            endpoint = mean - error
            direction = "↓"
        else:
            continue
        if clipped:
            axis.annotate(
                f"{direction} {endpoint:.3g}",
                xy=(x[index], 0.94 if direction == "↑" else 0.06),
                xycoords=("data", "axes fraction"),
                xytext=(0, -8 if direction == "↑" else 8),
                textcoords="offset points", ha="center",
                va="top" if direction == "↑" else "bottom",
                color=color, fontsize=fontsize,
                arrowprops=dict(arrowstyle="->", color=color),
                annotation_clip=False,
            )
    axis.errorbar(
        x, y, yerr=[lower, upper], fmt="none",
        color=color, capsize=capsize,
    )


def save_figure(
    figure: Figure,
    path: Path,
    style: dict,
    signature: str = "",
) -> None:
    """Atomically save a PNG and close its figure even on rendering errors.

    Parameters
    ----------
    figure : Figure
        Fully labeled figure.
    path : Path
        Owned rendering output, never a scientific payload.
    style : dict
        Resolved presentation settings.
    signature : str, optional
        Rendering provenance embedded in PNG metadata; default is empty.
    """
    temporary = None
    try:
        figure.canvas.draw()
        bounds = figure.get_tightbbox(figure.canvas.get_renderer())
        width = bounds.width + 2 * TIGHT_BBOX_PAD_INCHES
        height = bounds.height + 2 * TIGHT_BBOX_PAD_INCHES
        pixels = np.ceil(width * style["dpi"]) * np.ceil(
            height * style["dpi"]
        )
        if not np.isfinite(pixels) or (
            pixels > style["max_canvas_pixels"]
        ):
            raise ValueError(
                f"Figure {path} needs {pixels:g} pixels, exceeding "
                f"presentation.max_canvas_pixels="
                f"{style['max_canvas_pixels']}."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(dir=path.parent, suffix=".png")
        os.close(fd)
        temporary = Path(name)
        figure.savefig(
            temporary,
            dpi=style["dpi"],
            bbox_inches="tight",
            facecolor="white",
            metadata={"BRAID-rendering": signature},
        )
        os.replace(temporary, path)
    finally:
        plt.close(figure)
        if temporary is not None:
            temporary.unlink(missing_ok=True)
