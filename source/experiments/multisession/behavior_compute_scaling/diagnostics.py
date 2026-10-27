"""Publish readable checkpoint-only neural scaling diagnostics.

Usage: python -m experiments.multisession.behavior_compute_scaling.diagnostics
  --report ABSOLUTE_SOURCE_REPORT --diagnostics ABSOLUTE_DIAGNOSTICS

Inputs are checksum-valid source report tables and diagnostic summary.csv
(count, split, horizon, mean neural/readout/dummy losses), accompanied by
session_horizon_losses.csv. Output is an immutable sibling report with PNG
figures and references to input tables. Fits and existing reports are untouched.
"""

import argparse
import csv
import json
from pathlib import Path
from textwrap import fill

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.multisession.behavior_compute_scaling.plan import COUNTS
from experiments.multisession.behavior_compute_scaling.report import (
    RENDERING_STYLE, source_figures,
)
from experiments.multisession.scaling_report import COLORS
from experiments.multisession.data import SAMPLE_RATE
from experiments.multisession.store import completed, job
from experiments.storage.cache import atomic_json, fingerprint
from experiments.visualization.multisession import (
    ANNOTATION_FONT_SIZE, READABLE_TEXT_STYLE, plot_line,
)

HORIZONS = (1, 2, 4, 8)
SINGLE_SIZE = (10, 6)
PANEL_SIZE = (14, 11)
TITLE_WIDTH = 54
NOTE_WIDTH = 100
DPI = 180
MILLISECONDS_PER_SECOND = 1000
BLUE = "#2878b5"
ORANGE = "#e28528"
GREEN = "#339966"
LINE_STYLES = ("-", "--", "-.", ":")
ENDPOINT_NOTE = (
    "Largest completed budget: 800 updates (4/8/16 sessions), "
    "1600 (33 sessions); 6400 draws per recording per recurrent stage. "
    "The outer holdout also selects transfer checkpoints; selected scores "
    "are descriptive. One seed; no uncertainty estimate."
)


def read_rows(path: Path) -> list[dict]:
    """Read a nonempty diagnostic table without changing source artifacts."""
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Diagnostic table is empty: {path.resolve()}.")
    return rows


def loss_curve(
    rows: list[dict], filters: dict, column: str,
    horizons: tuple[int, ...] = HORIZONS,
) -> list[float]:
    """Sum requested horizon MSEs for each source count, rejecting gaps."""
    values = []
    for count in COUNTS:
        selected = [row for row in rows
                    if int(row["source_count"]) == count
                    and all(str(row[key]) == str(value)
                            for key, value in filters.items())
                    and int(row["horizon"]) in horizons]
        actual = [int(row["horizon"]) for row in selected]
        if sorted(actual) != sorted(horizons):
            raise ValueError(
                f"Expected horizons {horizons} for {count} sessions, "
                f"got {actual}; filters={filters}."
            )
        losses = np.asarray([float(row[column]) for row in selected])
        if not np.isfinite(losses).all() or np.any(losses <= 0):
            raise ValueError("Diagnostic log losses must be finite positive.")
        values.append(float(losses.sum()))
    return values


def configure_axis(axis: object, ylabel: str) -> None:
    """Apply source-count ticks and the shared log-axis presentation."""
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.set_xticks(COUNTS, labels=[str(count) for count in COUNTS])
    axis.set_xlim(min(COUNTS) / 1.08, max(COUNTS) * 1.08)
    axis.set_xlabel("Source sessions")
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.2)


def finish_figure(
    figure: object, path: Path, title: str, legend: bool = False,
) -> None:
    """Save one endpoint view with readable title and selection disclosure."""
    figure.suptitle(fill(title, TITLE_WIDTH), y=0.99)
    figure.text(
        0.04, 0.02, fill(ENDPOINT_NOTE, NOTE_WIDTH),
        fontsize=ANNOTATION_FONT_SIZE, va="bottom",
    )
    if legend:
        figure.axes[0].legend(
            loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False,
        )
    figure.tight_layout(rect=(0, 0.18, 1, 0.91))
    figure.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(figure)


@plt.rc_context(READABLE_TEXT_STYLE)
def endpoint_figures(
    summary: list[dict], sessions: list[dict], out: Path,
) -> None:
    """Render horizon, stage, split and recording-specific endpoint views."""
    figure, axes = plt.subplots(2, 2, figsize=PANEL_SIZE)
    for axis, horizon in zip(axes.flat, HORIZONS):
        values = loss_curve(
            summary, {"role": "outer_test"}, "mean_combined_loss",
            (horizon,),
        )
        plot_line(x=COUNTS, y=values, ax=axis, color=BLUE, marker="o")
        configure_axis(axis, "Neural prediction MSE")
        milliseconds = horizon * MILLISECONDS_PER_SECOND / SAMPLE_RATE
        axis.set_title(f"Horizon {horizon} ({milliseconds:g} ms)", pad=22)
    finish_figure(
        figure, out / "neural_loss_by_horizon.png",
        "Combined neural prediction: source scaling by horizon",
    )
    figure, axis = plt.subplots(figsize=SINGLE_SIZE)
    for column, label, color, style in (
        ("mean_readout_loss", "RNN1 neural readout", ORANGE, "--"),
        ("mean_combined_loss", "Readout + RNN2", BLUE, "-"),
    ):
        plot_line(
            x=COUNTS, y=loss_curve(summary, {"role": "outer_test"}, column),
            ax=axis, color=color, linestyle=style, marker="o", label=label,
        )
    configure_axis(axis, "Summed neural prediction MSE")
    finish_figure(
        figure, out / "neural_readout_and_combined_loss.png",
        "Neural readout versus combined prediction on the outer holdout",
        legend=True,
    )
    figure, axis = plt.subplots(figsize=SINGLE_SIZE)
    for role, label, color, style in (
        ("training", "Training", BLUE, "-"),
        ("inner_validation", "Inner validation", GREEN, "--"),
        ("outer_test", "Outer test", ORANGE, "-."),
    ):
        plot_line(
            x=COUNTS,
            y=loss_curve(summary, {"role": role}, "mean_combined_loss"),
            ax=axis, color=color, linestyle=style, marker="o", label=label,
        )
    configure_axis(axis, "Summed neural prediction MSE")
    finish_figure(
        figure, out / "neural_generalization.png",
        "Training and holdout losses on the same four source recordings",
        legend=True,
    )
    names = sorted({row["session"] for row in sessions})
    if len(names) != 4:
        raise ValueError(f"Expected four diagnostic recordings, got {names}.")
    figure, axis = plt.subplots(figsize=SINGLE_SIZE)
    for name, count, style in zip(names, COUNTS, LINE_STYLES):
        plot_line(
            x=COUNTS, y=loss_curve(
                sessions, {"role": "outer_test", "session": name},
                "combined_loss",
            ), ax=axis, color=COLORS[count], linestyle=style,
            marker="o", label=name,
        )
    configure_axis(axis, "Summed neural prediction MSE")
    finish_figure(
        figure, out / "neural_session_scaling.png",
        "Outer-holdout neural prediction for each common recording",
        legend=True,
    )


def publish(report: Path, diagnostics: Path) -> Path:
    """Validate completed inputs and publish five derived diagnostic views."""
    inputs = {}
    for path in (report, diagnostics):
        marker = completed(path)
        if marker is None:
            raise ValueError(f"Incomplete diagnostic input: {path.resolve()}.")
        inputs[str(path.resolve())] = marker["checksums"]
    provenance = json.loads((diagnostics / "identity.json").read_text())
    selections = read_rows(report / "tables" / "selection.csv")
    if {row["source_fit"] for row in selections} != {
        row["source_fit"] for row in provenance["sources"]
    }:
        raise ValueError("Diagnostic and report source fits differ.")
    summary = read_rows(diagnostics / "summary.csv")
    sessions = read_rows(diagnostics / "session_horizon_losses.csv")
    source = read_rows(report / "tables" / "source_loss.csv")
    for row in source:
        for key in ("source_count", "budget", "full_path_flops"):
            row[key] = int(row[key])
        row["chronological_source_test_loss"] = float(
            row["chronological_source_test_loss"]
        )
        row["transfer_selected"] = row["transfer_selected"] == "True"
    identity = dict(
        group="source_neural_diagnostics", inputs=inputs,
        rendering=RENDERING_STYLE, endpoint_note=ENDPOINT_NOTE,
        figures=["horizon", "readout", "generalization", "session", "compute"],
    )
    destination = report.parent / fingerprint(identity)
    with job(destination, identity) as needed:
        if needed:
            figures = destination / "figures"
            figures.mkdir()
            atomic_json(destination / "input_references.json", inputs)
            endpoint_figures(summary, sessions, figures)
            source_figures(source, figures, stages=("neural_readout",))
    atomic_json(report.parent.parent / "latest_diagnostics.json", dict(
        snapshot=str(destination.resolve()), identity=fingerprint(identity),
    ))
    print(f"Saved diagnostic figures: {(destination / 'figures').resolve()}")
    return destination


def main() -> None:
    """Resolve completed input snapshots and regenerate diagnostics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    args = parser.parse_args()
    publish(args.report.resolve(), args.diagnostics.resolve())


if __name__ == "__main__":
    main()
