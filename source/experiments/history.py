"""Render component-local epoch metrics without copying numeric histories.

history.jsonl supplies epoch/attempt/metrics. Each completed component gets
plots/attempt_<n>/*.png: total/non-step train-validation overlays and one
two-panel horizon comparison per logged metric. PNG metadata identifies the
history and style; no separate trend estimates or metric JSON are produced.
"""

import json
import logging
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .cache import file_digest, fingerprint, writer_lock
from .presentation import METRIC_LABELS, presentation, save_figure, style_axis

LOGGER = logging.getLogger(__name__)
STEP_METRIC = re.compile(r"rnn_(\d+)step_(loss|MSE|R2|CC)$")
CANONICAL = {"loss": "total_loss", "MSE": "mse", "R2": "r2", "CC": "cc"}


def metric_values(rows: list[dict], key: str) -> np.ndarray:
    """Return logged values, shape (epochs,), with warned, explicit gaps."""
    values = np.array(
        [
            np.nan if row["metrics"].get(key) is None else row["metrics"][key]
            for row in rows
        ],
        dtype=float,
    )
    if not np.isfinite(values).all():
        LOGGER.warning("Undefined history values for %s; displaying gaps.", key)
        values[~np.isfinite(values)] = np.nan
    return values


def history_figure(
    rows: list[dict],
    metric: str,
    steps: list[int],
    title: str,
    style: dict,
) -> object:
    """Build an overlay or shared-axis horizon figure from actual epoch rows."""
    epochs = np.array([row["epoch"] for row in rows])
    label = METRIC_LABELS[metric.lower()]
    if steps:
        figure, axes = plt.subplots(
            1,
            2,
            sharex=True,
            sharey=True,
            figsize=style["horizon_size"],
        )
        if max(steps) > len(style["horizon_colors"]):
            plt.close(figure)
            raise ValueError("Configure a distinct color for every horizon.")
        for axis, prefix, split in zip(
            axes,
            ("", "val_"),
            ("Training", "Validation"),
        ):
            for horizon in steps:
                key = f"{prefix}rnn_{horizon}step_{metric}"
                axis.plot(
                    epochs,
                    metric_values(rows, key),
                    color=style["horizon_colors"][horizon - 1],
                    label=f"{horizon} step",
                )
            axis.set(title=split, xlabel="Epoch", ylabel=label)
            style_axis(axis, style)
        handles, labels = axes[0].get_legend_handles_labels()
    else:
        figure, axis = plt.subplots(figsize=style["single_size"])
        for prefix, split, color in zip(
            ("", "val_"),
            ("Training", "Validation"),
            style["pair_colors"],
        ):
            axis.plot(
                epochs,
                metric_values(rows, prefix + metric),
                label=split,
                color=color,
            )
        axis.set(xlabel="Epoch", ylabel=label)
        style_axis(axis, style)
        handles, labels = axis.get_legend_handles_labels()
    if not any(
        np.isfinite(line.get_ydata()).any()
        for axis in figure.axes
        for line in axis.lines
    ):
        plt.close(figure)
        raise ValueError(f"No finite history values for {title}: {metric}")
    figure.suptitle(title, fontsize=style["title_font"])
    figure.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        fontsize=style["legend_font"],
        frameon=False,
    )
    figure.tight_layout()
    return figure


def render_component(directory: Path, settings: dict | None = None) -> None:
    """Render completed component attempts without changing their history."""
    style = presentation(settings)
    history = directory / "history.jsonl"
    rows = [json.loads(line) for line in history.read_text().splitlines()]
    if not rows:
        raise ValueError(f"Empty component history: {history.resolve()}")
    signature = fingerprint(
        dict(
            history=file_digest(history),
            style=style,
            implementation=file_digest(Path(__file__)),
            presentation=file_digest(
                Path(__file__).with_name("presentation.py")
            ),
        )
    )
    with writer_lock(directory / "plots" / ".render.lock"):
        for attempt in sorted({row["attempt"] for row in rows}):
            selected = [row for row in rows if row["attempt"] == attempt]
            keys = set().union(*(row["metrics"].keys() for row in selected))
            definitions = [
                (metric, [], filename)
                for metric, filename in CANONICAL.items()
                if metric in keys
            ]
            for metric in CANONICAL:
                steps = sorted(
                    {
                        int(match[1])
                        for key in keys
                        if (match := STEP_METRIC.fullmatch(key))
                        and match[2] == metric
                    }
                )
                if steps:
                    definitions.append(
                        (
                            metric,
                            steps,
                            f"{metric.lower()}_by_horizon",
                        )
                    )
            for metric, steps, filename in definitions:
                path = (
                    directory
                    / "plots"
                    / f"attempt_{attempt}"
                    / (filename + ".png")
                )
                if path.exists():
                    try:
                        with Image.open(path) as saved:
                            if saved.info.get("BRAID-rendering") == signature:
                                continue
                    except OSError:
                        LOGGER.warning("Redrawing damaged PNG: %s", path)
                figure = history_figure(
                    selected,
                    metric,
                    steps,
                    f"{directory.name}\n{METRIC_LABELS[metric.lower()]}",
                    style,
                )
                save_figure(figure, path, style, signature)
