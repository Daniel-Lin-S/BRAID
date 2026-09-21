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

from .cache import atomic_json, file_digest, fingerprint, writer_lock
from .presentation import METRIC_LABELS, presentation, save_figure, style_axis

LOGGER = logging.getLogger(__name__)
STEP_METRIC = re.compile(
    r"rnn_(?:(?P<output>\d+)_)?"
    r"(?P<horizon>\d+)step_(?P<metric>loss|MSE|R2|CC)$"
)
CANONICAL = {"loss": "total_loss", "MSE": "mse", "R2": "r2", "CC": "cc"}
RENDERING_RECORD = "rendering.json"
RENDERING_RECORD_VERSION = 1


def metric_values(rows: list[dict], key: str) -> np.ndarray:
    """Return logged metric values, with explicit NaN gaps."""
    return np.array(
        [
            np.nan if row["metrics"].get(key) is None else row["metrics"][key]
            for row in rows
        ],
        dtype=float,
    )


def metric_keys(
    metric: str,
    steps: list[int],
    output: str | None,
) -> tuple[str, ...]:
    """Return the history keys contributing to one rendered figure."""
    if not steps:
        return (metric, f"val_{metric}")
    output_prefix = "" if output is None else f"{output}_"
    return tuple(
        f"{prefix}rnn_{output_prefix}{horizon}step_{metric}"
        for prefix in ("", "val_")
        for horizon in steps
    )


def has_finite_metric_values(
    rows: list[dict],
    metric: str,
    steps: list[int],
    output: str | None,
) -> bool:
    """Return whether one figure has at least one finite history value."""
    return any(
        np.isfinite(metric_values(rows, key)).any()
        for key in metric_keys(metric, steps, output)
    )


def load_skipped_metrics(path: Path, signature: str) -> dict[str, dict]:
    """Read matching intentional omissions from one component attempt."""
    if not path.is_file():
        return {}
    try:
        record = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        LOGGER.warning("Ignoring invalid rendering record %s: %s", path, error)
        return {}
    if (
        record.get("version") != RENDERING_RECORD_VERSION
        or record.get("signature") != signature
    ):
        return {}
    skipped = record.get("skipped")
    if not isinstance(skipped, dict):
        LOGGER.warning("Ignoring invalid rendering record: %s", path)
        return {}
    return skipped


def write_skipped_metrics(
    path: Path,
    signature: str,
    skipped: dict[str, dict],
) -> None:
    """Atomically publish intentional omissions for one component attempt."""
    atomic_json(
        path,
        {
            "signature": signature,
            "skipped": skipped,
            "version": RENDERING_RECORD_VERSION,
        },
    )


def history_figure(
    rows: list[dict],
    metric: str,
    steps: list[int],
    title: str,
    style: dict,
    output: str | None = None,
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
                output_prefix = "" if output is None else f"{output}_"
                key = (
                    f"{prefix}rnn_{output_prefix}{horizon}step_{metric}"
                )
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


def horizon_definitions(
    keys: set[str],
) -> list[tuple[str, list[int], str, str | None]]:
    """Group logged step metrics by metric and optional output index.

    Parameters
    ----------
    keys : set of str
        Metric names from one component attempt.

    Returns
    -------
    list of tuple
        Metric name, sorted horizons, output filename stem, and optional
        recurrent output index for each horizon plot.
    """
    groups = {}
    for key in keys:
        match = STEP_METRIC.fullmatch(key)
        if match is None:
            continue
        group = (match["metric"], match["output"])
        groups.setdefault(group, set()).add(int(match["horizon"]))
    definitions = []
    for metric in CANONICAL:
        outputs = sorted(
            (
                (output, sorted(steps))
                for (name, output), steps in groups.items()
                if name == metric
            ),
            key=lambda item: "" if item[0] is None else item[0],
        )
        for output, steps in outputs:
            qualifier = (
                ""
                if len(outputs) == 1
                else f"_output_{output or 'default'}"
            )
            definitions.append(
                (
                    metric,
                    steps,
                    f"{metric.lower()}{qualifier}_by_horizon",
                    output,
                )
            )
    return definitions


def render_component(
    directory: Path,
    settings: dict | None = None,
    regenerate: bool = False,
) -> None:
    """Render missing component plots or explicitly redraw the full set.

    Parameters
    ----------
    directory : Path
        Completed component directory containing history.jsonl.
    settings : dict, optional
        Figure presentation settings; default None uses shared settings.
    regenerate : bool, optional
        Whether to redraw valid existing PNGs; default is False.
    """
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
                (metric, [], filename, None)
                for metric, filename in CANONICAL.items()
                if metric in keys
            ]
            definitions.extend(horizon_definitions(keys))
            record_path = (
                directory
                / "plots"
                / f"attempt_{attempt}"
                / RENDERING_RECORD
            )
            skipped = load_skipped_metrics(record_path, signature)
            updated_skipped = dict(skipped)
            failures = []
            for metric, steps, filename, output in definitions:
                path = (
                    directory
                    / "plots"
                    / f"attempt_{attempt}"
                    / (filename + ".png")
                )
                if not has_finite_metric_values(
                    selected,
                    metric,
                    steps,
                    output,
                ):
                    if filename not in skipped or regenerate:
                        LOGGER.warning(
                            "Omitting all-gap history figure: component=%s "
                            "attempt=%s metric=%s "
                            "reason=no_finite_epoch_values",
                            directory,
                            attempt,
                            metric,
                        )
                    updated_skipped[filename] = {
                        "metric": metric,
                        "reason": "no_finite_epoch_values",
                    }
                    continue
                if path.exists() and not regenerate:
                    try:
                        with Image.open(path) as saved:
                            saved.verify()
                        continue
                    except OSError:
                        LOGGER.warning("Redrawing damaged PNG: %s", path)
                try:
                    figure = history_figure(
                        selected,
                        metric,
                        steps,
                        f"{directory.name}\n{METRIC_LABELS[metric.lower()]}",
                        style,
                        output,
                    )
                    save_figure(figure, path, style, signature)
                except Exception as error:
                    failures.append((filename, error))
            if updated_skipped:
                write_skipped_metrics(record_path, signature, updated_skipped)
            if failures:
                details = "; ".join(
                    f"{filename}: {error}" for filename, error in failures
                )
                raise RuntimeError(
                    f"Component rendering failed for {directory}: {details}"
                )
