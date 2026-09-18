"""Render ordered stages for a neural channel or a labeled coordinate pair.

Inputs are labeled single-channel or paired-coordinate traces, native
timestamps, and presentation settings. PNG figures have outside legends and
shared time limits. This module never reads data or fits a model.
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np

from .presentation import save_figure

SUPPORTED_KINDS = {"line", "held", "counts", "spikes"}
COORDINATE_STYLES = ("-", "--")
MINIMUM_FONT = 16


@dataclass
class PreviewStage:
    """One ordered processing stage for a single channel or coordinate.

    Time has shape (T,); values have shape (T,) or (T, K) for K labels.
    Coordinates defaults to no labels for single-channel stages.
    Spike stages use event times
    and no values; an empty event vector represents a valid silent window.
    """

    key: str
    label: str
    unit: str
    time: np.ndarray
    values: np.ndarray | None
    kind: str = "line"
    coordinates: tuple[str, ...] = ()


def signal_figure(
    stages: list[PreviewStage],
    signal: str,
    title: str,
    bounds: tuple[float, float],
    style: dict,
) -> Figure:
    """Build vertically ordered panels for one signal.

    Parameters
    ----------
    stages : list of PreviewStage
        Ordered raw, processed and fitted stages of a single signal.
    signal, title : str
        Legend identity and figure heading.
    bounds : tuple of float
        Absolute window start and exclusive end, in seconds.
    style : dict
        Font sizes, figure width, panel height and PNG DPI.

    Returns
    -------
    Figure
        Open figure; the caller saves or closes it.
    """
    if not stages or bounds[1] <= bounds[0]:
        raise ValueError("Expected stages and increasing window bounds.")
    for name in ("title_font", "label_font", "tick_font", "legend_font"):
        if style[name] < MINIMUM_FONT:
            raise ValueError(f"{name} must be at least {MINIMUM_FONT} points.")
    for stage in stages:
        if stage.coordinates and len(stage.coordinates) != 2:
            raise ValueError("Expected exactly two coordinate labels.")
        if stage.kind not in SUPPORTED_KINDS:
            raise ValueError(f"Unsupported preview kind: {stage.kind}")
        if stage.time.ndim != 1 or not np.isfinite(stage.time).all():
            raise ValueError(f"Invalid timestamp vector for {stage.key}.")
        if stage.kind != "spikes":
            if (
                stage.values is None
                or stage.values.shape
                != (
                    (len(stage.time), len(stage.coordinates))
                    if stage.coordinates
                    else stage.time.shape
                )
                or not stage.values.size
            ):
                raise ValueError(
                    f"Expected time-aligned, labeled traces: {stage.key}"
                )
            if not np.isfinite(stage.values).all():
                raise ValueError(f"Nonfinite preview values: {stage.key}")
    figure, axes = plt.subplots(
        len(stages),
        1,
        sharex=True,
        squeeze=False,
        figsize=(
            style["width_inches"],
            style["panel_height_inches"] * len(stages),
        ),
    )
    for number, (axis, stage) in enumerate(zip(axes[:, 0], stages)):
        if stage.kind == "spikes":
            axis.vlines(
                stage.time, 0, 1, color=style["pair_colors"][0],
                linewidth=1.4, label=signal
            )
            axis.set_ylim(-0.1, 1.1)
            axis.set_yticks([0, 1], labels=["", "Events"])
        else:
            labels = stage.coordinates or (signal,)
            values = (
                stage.values if stage.coordinates else stage.values[:, None]
            )
            for column, label in enumerate(labels):
                options = dict(
                    color=style["pair_colors"][column],
                    linestyle=COORDINATE_STYLES[column],
                    linewidth=1.8,
                    label=label,
                )
                if stage.kind in ("held", "counts"):
                    axis.step(
                        stage.time,
                        values[:, column],
                        where="post" if stage.kind == "held" else "pre",
                        **options,
                    )
                else:
                    axis.plot(stage.time, values[:, column], **options)
        axis.set_title(
            f"{number}. {stage.label}",
            loc="left",
            fontsize=style["label_font"],
            pad=12,
        )
        axis.set_ylabel(stage.unit, fontsize=style["label_font"])
        axis.tick_params(labelsize=style["tick_font"])
        axis.set_xlim(*bounds)
        axis.grid(axis="x", alpha=0.2)
        axis.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0,
            frameon=False,
            fontsize=style["legend_font"],
        )
        axis.ticklabel_format(axis="x", style="plain", useOffset=False)
        axis.yaxis.get_offset_text().set_fontsize(style["tick_font"])
    axes[-1, 0].set_xlabel("Session time (s)", fontsize=style["label_font"])
    figure.suptitle(title, fontsize=style["title_font"], y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.96), h_pad=2.0)
    return figure


def save_signal(
    path: Path,
    stages: list[PreviewStage],
    signal: str,
    title: str,
    bounds: tuple[float, float],
    style: dict,
) -> None:
    """Save and close one PNG, keeping the renderer free of dataset logic."""
    figure = signal_figure(stages, signal, title, bounds, style)
    save_figure(figure, path, style)
