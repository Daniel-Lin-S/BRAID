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

from .configuration import CONFIGURATION, read_yaml

METRIC_LABELS = {"loss": "Loss", "mse": "MSE", "r2": "R²", "cc": "CC"}


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
    for key in ("pair_colors", "horizon_colors"):
        if not style[key] or not all(is_color_like(c) for c in style[key]):
            raise ValueError(f"Expected valid colors in {key}.")
    if len(style["pair_colors"]) != 2:
        raise ValueError("Expected exactly two train/validation or x/y colors.")
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
