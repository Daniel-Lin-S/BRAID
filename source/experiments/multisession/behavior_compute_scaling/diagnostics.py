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
    RENDERING_STYLE,
)
from experiments.multisession.scaling_report import COLORS
from experiments.multisession.data import SAMPLE_RATE
from experiments.multisession.store import completed, job
from experiments.storage.cache import atomic_json, fingerprint
from experiments.visualization.multisession import (
    ANNOTATION_FONT_SIZE, DEFAULT_LINE_WIDTH, READABLE_TEXT_STYLE,
    plot_line,
)

HORIZONS = (1, 2, 4, 8)
SINGLE_SIZE = (10, 6)
PANEL_SIZE = (14, 11)
TITLE_WIDTH = 54
NOTE_WIDTH = 100
DPI = 180
MILLISECONDS_PER_SECOND = 1000
RATIO_PADDING = 0.02
RELATIVE_PADDING = 0.10
BLUE = "#2878b5"
ORANGE = "#e28528"
GREEN = "#339966"
LINE_STYLES = ("-", "--", "-.", ":")
RATIO_NOTE = (
    "Y-axis focuses on observed MSE/dummy values. Dummy is 1 and may "
    "fall outside the displayed range; values above 1 are worse."
)
ENDPOINT_NOTE = (
    "Largest completed budget: 800 updates (4/8/16 sessions), "
    "1600 (33 sessions); 6400 draws per recording per recurrent stage. "
    "The outer holdout also selects transfer checkpoints; selected scores "
    "are descriptive. One seed; no uncertainty estimate. "
    + RATIO_NOTE
)
COMPUTE_NOTE = (
    "Five completed budgets per source count. The same outer holdout "
    "selects transfer checkpoints; selected scores are descriptive. "
    "One seed; no uncertainty estimate. "
    + RATIO_NOTE
)
BASELINE_RATIO = 1.0
Y_AXIS_ORIGIN = 0.0
DUMMY_REFERENCE_COLOR = "#333333"
BASELINE_LABEL = "MSE / dummy MSE"


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


def ratio_curve(
    rows: list[dict], filters: dict, column: str,
    horizons: tuple[int, ...] = HORIZONS,
) -> list[float]:
    """Divide per-count mean MSE by its matching dummy predictor MSE."""
    numerator = np.asarray(loss_curve(rows, filters, column, horizons))
    dummy_column = (
        "dummy_loss" if column == "combined_loss" else "mean_dummy_loss"
    )
    denominator = np.asarray(
        loss_curve(rows, filters, dummy_column, horizons)
    )
    if not np.allclose(
        denominator, denominator[0], rtol=1e-8, atol=1e-8
    ):
        raise ValueError(
            "Dummy predictor changed across source counts; "
            "a shared reference is required."
        )
    ratio = numerator / denominator
    if not np.isfinite(ratio).all() or np.any(ratio <= 0):
        raise ValueError("Expected finite positive dummy-normalized loss.")
    return ratio.tolist()


def ratio_axis(axis: object) -> None:
    """Focus on observed ratios while retaining above-dummy outcomes."""
    values = np.asarray([
        value for line in axis.lines for value in line.get_ydata()
    ], dtype=float)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("Ratio plot requires finite nonempty values.")
    if np.any(values <= 0):
        raise ValueError("Ratio plot requires positive model MSE ratios.")
    smallest, largest = float(values.min()), float(values.max())
    padding = max(
        RATIO_PADDING, (largest - smallest) * RELATIVE_PADDING
    )
    lower = max(Y_AXIS_ORIGIN, smallest - padding)
    upper = largest + padding
    if largest > BASELINE_RATIO:
        lower = min(lower, BASELINE_RATIO)
    axis.set_yscale("linear")
    axis.set_ylim(lower, upper)
    if lower <= BASELINE_RATIO <= upper:
        axis.axhline(
            BASELINE_RATIO, color=DUMMY_REFERENCE_COLOR,
            linestyle=":", linewidth=1.5,
        )
    axis.set_ylabel(BASELINE_LABEL)
    axis.grid(alpha=0.2)


def configure_axis(axis: object) -> None:
    """Use source-count ticks and the shared dummy-anchored MSE scale."""
    axis.set_xscale("log", base=2)
    axis.set_xticks(COUNTS, labels=[str(count) for count in COUNTS])
    axis.set_xlim(min(COUNTS) / 1.08, max(COUNTS) * 1.08)
    axis.set_xlabel("Source sessions")
    ratio_axis(axis)


def finish_figure(
    figure: object, path: Path, title: str,
    legend: bool = False, note: str = ENDPOINT_NOTE,
) -> None:
    """Save one endpoint view with readable title and selection disclosure."""
    figure.suptitle(fill(title, TITLE_WIDTH), y=0.99)
    figure.text(
        0.04, 0.02, fill(note, NOTE_WIDTH),
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
        values = ratio_curve(
            summary, {"role": "outer_test"}, "mean_combined_loss",
            (horizon,),
        )
        plot_line(x=COUNTS, y=values, ax=axis, color=BLUE, marker="o")
        configure_axis(axis)
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
            x=COUNTS, y=ratio_curve(summary, {"role": "outer_test"}, column),
            ax=axis, color=color, linestyle=style, marker="o", label=label,
        )
    configure_axis(axis)
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
            y=ratio_curve(summary, {"role": role}, "mean_combined_loss"),
            ax=axis, color=color, linestyle=style, marker="o", label=label,
        )
    configure_axis(axis)
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
            x=COUNTS, y=ratio_curve(
                sessions, {"role": "outer_test", "session": name},
                "combined_loss",
            ), ax=axis, color=COLORS[count], linestyle=style,
            marker="o", label=name,
        )
    configure_axis(axis)
    finish_figure(
        figure, out / "neural_session_scaling.png",
        "Outer-holdout neural prediction for each common recording",
        legend=True,
    )


@plt.rc_context(READABLE_TEXT_STYLE)
def readout_compute_figures(
    source: list[dict], sessions: list[dict], out: Path,
) -> None:
    """Draw five readout budgets per count against full-path FLOPs."""
    figure, axis = plt.subplots(figsize=SINGLE_SIZE)
    baseline_by_session = {}
    for count, style in zip(COUNTS, LINE_STYLES):
        selected = sorted(
            (row for row in source if row["stage"] == "neural_readout"
             and row["source_count"] == count),
            key=lambda row: row["full_path_flops"],
        )
        if len(selected) != 5:
            raise ValueError(
                f"Expected five readout budgets for {count} sessions."
            )
        xs, ys = [], []
        for row in selected:
            matching = [item for item in sessions
                        if item["stage"] == "neural_readout"
                        and int(item["source_count"]) == count
                        and int(item["budget"]) == row["budget"]]
            if len(matching) != 4:
                raise ValueError(
                    f"Expected four readout dummy scores for {count} "
                    f"sessions and budget {row['budget']}."
                )
            names = [item["session"] for item in matching]
            if len(set(names)) != 4:
                raise ValueError("Readout dummy scores need four sessions.")
            for item in matching:
                name = item["session"]
                value = float(item["dummy_loss"])
                if not np.isfinite(value) or value <= 0:
                    raise ValueError(
                        "Readout dummy scores must be finite positive."
                    )
                if name in baseline_by_session and not np.isclose(
                    value, baseline_by_session[name], rtol=1e-8, atol=1e-8
                ):
                    raise ValueError(
                        f"Readout dummy baseline changed for {name}."
                    )
                baseline_by_session[name] = value
            if set(names) != set(baseline_by_session):
                raise ValueError("Readout dummy sessions changed by budget.")
            dummy = np.asarray(
                [float(item["dummy_loss"]) for item in matching]
            )
            loss = row["chronological_source_test_loss"]
            ratio = loss / float(dummy.mean())
            if not np.isfinite(ratio) or ratio <= 0:
                raise ValueError("Invalid readout dummy-normalized loss.")
            xs.append(row["full_path_flops"])
            ys.append(ratio)
        plot_line(
            x=xs, y=ys, ax=axis, color=COLORS[count],
            linestyle=style, marker="o", linewidth=DEFAULT_LINE_WIDTH,
            label=f"{count} source sessions",
        )
        chosen = [index for index, row in enumerate(selected)
                  if row["transfer_selected"]]
        if len(chosen) != 1:
            raise ValueError("Expected one selected checkpoint per count.")
        axis.scatter(
            xs[chosen[0]], ys[chosen[0]], s=80, marker="s",
            color=COLORS[count], edgecolor="black", zorder=5,
        )
    axis.set_xscale("log")
    axis.set_xlabel("Full-pipeline source model FLOPs")
    ratio_axis(axis)
    finish_figure(
        figure, out / "source_neural_readout.png",
        "Neural readout versus full-pipeline training FLOPs",
        legend=True, note=COMPUTE_NOTE,
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
    source_sessions = read_rows(
        report / "tables" / "source_session_loss.csv"
    )
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
        compute_note=COMPUTE_NOTE, baseline_axis=dict(
            measure=BASELINE_LABEL,
            reference=BASELINE_RATIO,
            focus_padding=RATIO_PADDING,
            relative_padding=RELATIVE_PADDING,
            includes_reference_if_model_is_worse_than_dummy=True,
            dummy="constant training-target mean on the matching windows",
        ),
        figures=["horizon", "readout", "generalization", "session", "compute"],
    )
    destination = report.parent / fingerprint(identity)
    with job(destination, identity) as needed:
        if needed:
            figures = destination / "figures"
            figures.mkdir()
            atomic_json(destination / "input_references.json", inputs)
            endpoint_figures(summary, sessions, figures)
            readout_compute_figures(source, source_sessions, figures)
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
