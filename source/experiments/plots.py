"""Reusable plots consuming completed numeric evaluation artifacts.

Input summary rows identify x parameter, output target, metric, and curve.
Output PNG curves and forecast panels never require refitting a model.
"""

from pathlib import Path
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LOGGER = logging.getLogger(__name__)


def metric_curve(
    rows: list[dict],
    parameter: str,
    target: str,
    destination: Path,
    group: str | None = None,
) -> None:
    """Draw a metric-versus-parameter graph with available SEM error bars."""
    if not rows:
        raise ValueError(f"No data for plot {destination.resolve()}")
    if not any(row["mean"] is not None for row in rows):
        LOGGER.warning("All metrics undefined; skipped %s", destination)
        return
    fig, axis = plt.subplots(figsize=(6, 4))
    groups = sorted({row[group] for row in rows}) if group else [None]
    for label in groups:
        selected = [r for r in rows if not group or r[group] == label]
        selected = sorted(selected, key=lambda r: r[parameter])
        selected = [r for r in selected if r["mean"] is not None]
        if not selected:
            continue
        x = [r[parameter] for r in selected]
        y = [r["mean"] for r in selected]
        axis.plot(x, y, "o-", label=f"{group}={label}" if group else "BRAID")
        with_sem = [r for r in selected if r["sem"] is not None]
        if with_sem:
            axis.errorbar(
                [r[parameter] for r in with_sem],
                [r["mean"] for r in with_sem],
                yerr=[r["sem"] for r in with_sem],
                fmt="none",
            )
    axis.set(xlabel=parameter, ylabel=f"{target} Pearson CC")
    axis.legend()
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def plot_suite(summaries: list[dict], root: Path, settings: dict) -> None:
    """Render configured metric curves using one shared plotting function."""
    if not settings["enabled"]:
        return
    for spec in settings["curves"]:
        chosen = [
            row
            for row in summaries
            if all(
                (
                    row[k] in value
                    if isinstance(value, list)
                    else row[k] == value
                )
                for k, value in spec["where"].items()
            )
        ]
        if chosen:
            metric_curve(
                chosen,
                spec["parameter"],
                spec["where"]["target"],
                root / (spec["name"] + ".png"),
                spec.get("group"),
            )


def forecast_example(
    path: Path,
    destination: Path,
    seconds: float,
    sample_rate: float,
    horizon: int,
) -> None:
    """Plot all behavior dimensions on one shared uncluttered time interval."""
    with np.load(path) as data:
        index = list(data["horizons"]).index(horizon)
        available = np.flatnonzero(data["valid"][index])
        # The excerpt stays inside one independent 128-sample window.
        count = min(int(seconds * sample_rate), len(available))
        chosen = available[:count]
        t, truth = data["t"][chosen], data["true_Z"][chosen]
        prediction = data["Z"][index, chosen]
    labels = ["Position x", "Position y", "Velocity x", "Velocity y"]
    fig, axes = plt.subplots(truth.shape[1], 1, sharex=True, figsize=(10, 8))
    for dim, axis in enumerate(np.atleast_1d(axes)):
        axis.plot(t, truth[:, dim], label="Observed")
        axis.plot(t, prediction[:, dim], label="BRAID forecast")
        axis.set_ylabel(labels[dim])
    np.atleast_1d(axes)[0].legend()
    np.atleast_1d(axes)[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)
