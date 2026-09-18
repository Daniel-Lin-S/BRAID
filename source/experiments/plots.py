"""Render analysis comparisons from accepted per-session summary statistics.

Outputs are named target_metric_vs_parameter_selection.png figures. Missing
observations remain gaps; SEM is never manufactured. This module does not
load a checkpoint, fit a model, or produce individual-fold plots.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .presentation import METRIC_LABELS, presentation, save_figure, style_axis

LOGGER = logging.getLogger(__name__)
PARAMETER_LABELS = {
    "horizon": "Forecast horizon (steps)",
    "nx": "Latent dimension (nx)",
    "population_scale": "Neural population (%)",
}


def curve_specs(settings: dict) -> list[dict]:
    """Expand configured comparison designs into separate metric figures."""
    result = []
    for curve in settings.get("curves", []):
        for metric in settings.get("metrics", [curve["where"]["metric"]]):
            if metric not in ("cc", "r2", "mse"):
                raise ValueError(f"Unsupported comparison metric: {metric}")
            where = dict(curve["where"], metric=metric)
            parameter = curve["parameter"]
            selection = (
                f"nx{where['nx']}"
                if parameter == "horizon"
                else f"horizon{where['horizon']}"
            )
            axis = (
                "population" if parameter == "population_scale" else parameter
            )
            name = (
                f"{where['target']}_{metric}_vs_{axis}_{selection}_"
                f"{where['evaluation_set']}"
            )
            result.append(dict(curve, where=where, name=name))
    return result


def matches(row: dict, where: dict) -> bool:
    """Match scalar and multi-value selections without dropping missing rows."""
    return all(
        row[key] in value if isinstance(value, list) else row[key] == value
        for key, value in where.items()
    )


def metric_curve(
    rows: list[dict],
    spec: dict,
    style: dict,
    partial: bool = False,
) -> object:
    """Build one metric figure from summary rows with mean/SEM and x values."""
    if not rows or not any(
        row["mean"] is not None and np.isfinite(row["mean"]) for row in rows
    ):
        raise ValueError(f"No finite results for {spec['name']}.")
    parameter, group = spec["parameter"], spec.get("group")
    metric, target = spec["where"]["metric"], spec["where"]["target"]
    figure, axis = plt.subplots(figsize=style["single_size"])
    groups = sorted({r[group] for r in rows}) if group else [None]
    for number, value in enumerate(groups):
        selected = sorted(
            [r for r in rows if not group or r[group] == value],
            key=lambda row: row[parameter],
        )
        x = np.array([row[parameter] for row in selected], dtype=float)
        if parameter == "population_scale":
            x *= 100
        y = np.array(
            [
                np.nan if row["mean"] is None else row["mean"]
                for row in selected
            ],
            dtype=float,
        )
        if not np.isfinite(y).all():
            LOGGER.warning("Missing/undefined points in %s.", spec["name"])
            y[~np.isfinite(y)] = np.nan
        color_index = {16: 0, 64: 1}.get(value, number)
        color = style["pair_colors"][color_index % len(style["pair_colors"])]
        axis.plot(
            x,
            y,
            "o-",
            label=f"{group}={value}" if group else "BRAID",
            color=color,
        )
        valid_sem = [
            i
            for i, row in enumerate(selected)
            if np.isfinite(y[i])
            and row["sem"] is not None
            and np.isfinite(row["sem"])
        ]
        if valid_sem:
            axis.errorbar(
                x[valid_sem],
                y[valid_sem],
                yerr=[selected[i]["sem"] for i in valid_sem],
                fmt="none",
                color=color,
                capsize=4,
            )
    ticks = sorted({row[parameter] for row in rows})
    if parameter == "nx":
        axis.set_xscale("log", base=2)
    elif parameter == "population_scale":
        ticks = [100 * value for value in ticks]
    axis.set_xticks(ticks, labels=[f"{value:g}" for value in ticks])
    axis.set(
        xlabel=PARAMETER_LABELS[parameter],
        ylabel=f"{target.capitalize()} {METRIC_LABELS[metric]}",
    )
    style_axis(axis, style)
    fixed = (
        f"nx={spec['where']['nx']}"
        if parameter == "horizon"
        else f"horizon={spec['where']['horizon']}"
    )
    title = (
        f"{target.capitalize()} {METRIC_LABELS[metric]}\n"
        f"{fixed}; {spec['where']['evaluation_set']} scoring"
    )
    figure.suptitle(
        title + ("\n(partial)" if partial else ""),
        fontsize=style["title_font"],
        wrap=True,
    )
    axis.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        frameon=False,
        fontsize=style["legend_font"],
    )
    figure.tight_layout()
    return figure


def plot_suite(
    summaries: list[dict],
    root: Path,
    settings: dict,
    expected: list[dict] | None = None,
    pending: bool = False,
) -> None:
    """Publish available comparisons, retaining gaps for unfinished settings."""
    if not settings["enabled"]:
        return
    style = presentation(settings.get("presentation"))
    errors = []
    for spec in curve_specs(settings):
        chosen = [row for row in summaries if matches(row, spec["where"])]
        missing = []
        partial = False
        for candidate in expected or []:
            if not matches(candidate, spec["where"]):
                continue
            partial |= candidate.get("pending", False)
            criteria = {k: v for k, v in candidate.items() if k != "pending"}
            if not any(
                all(row[key] == criteria[key] for key in criteria)
                for row in chosen
            ):
                missing.append(dict(criteria, mean=None, sem=None))
        if not chosen and (partial or pending and expected is None):
            LOGGER.warning("Comparison pending: %s", spec["name"])
            continue
        try:
            figure = metric_curve(
                chosen + missing,
                spec,
                style,
                partial=partial or bool(missing),
            )
            save_figure(figure, root / f"{spec['name']}.png", style)
        except Exception as error:
            LOGGER.exception("Comparison failed: %s", spec["name"])
            errors.append(str(error))
    if errors:
        raise RuntimeError("Comparison rendering failed: " + "; ".join(errors))
