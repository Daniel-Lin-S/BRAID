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
    DEFAULT_LINE_WIDTH, READABLE_TEXT_STYLE, plot_line,
)

HORIZONS = (1, 2, 4, 8)
SINGLE_SIZE = (10, 6)
PANEL_SIZE = (14, 10)
DPI = 180
MILLISECONDS_PER_SECOND = 1000
RATIO_PADDING = 0.02
MSE_PADDING = 0.02
RELATIVE_PADDING = 0.10
BLUE = "#2878b5"
ORANGE = "#e28528"
GREEN = "#339966"
LINE_STYLES = ("-", "--", "-.", ":")
BASELINE_RATIO = 1.0
Y_AXIS_ORIGIN = 0.0
DUMMY_REFERENCE_COLOR = "#333333"
BASELINE_LABEL = "Neural MSE / dummy MSE"
FIGURE_GUIDE = """# Neural source-scaling diagnostics

All five figures score predictions of the **smoothed spike-count signal**
after per-recording centering and scaling fitted on source training windows.
They do not score only the behavior-related neural variation. The full
forecast adds the RNN1 neural readout to the RNN2 residual prediction and
compares that sum with the normalized neural signal.

For each forecast horizon, MSE averages squared error over valid time points
and retained neural channels within a window, then averages windows within
each of the four common recordings and averages those recordings equally.
The dummy predicts the per-channel mean of that recording's source training
target, evaluated on the **same windows** as the model. Four figures show
model MSE divided by matching dummy MSE; the generalization figure shows
raw model MSE. The dummy-normalized figures spanning four horizons sum
horizon MSEs before division. The dummy is 1; ratios above 1 are valid
and mean that the model is worse than the dummy. Axes focus on the
observed range, so the line at 1 may be outside the view.

- `neural_loss_by_horizon.png`: RNN1 neural readout plus RNN2 residual
  correction on the fixed outer chronological test windows, separately
  at 50, 100, 200 and 400 ms.
- `neural_readout_and_combined_loss.png`: Outer-test readout alone and full
  forecast, both scored against the same normalized neural target.
- `neural_generalization.png`: Raw full-forecast MSE summed across
  four horizons, re-evaluated on training, inner-validation and
  outer-test windows of the same four recordings.
- `neural_session_scaling.png`: Outer-test full forecast, one line
  for each common recording.
- `source_neural_readout.png`: Outer-test readout ratio at five
  completed compute budgets; squares mark transfer-selected budgets.

The first four figures use each count's largest original completed budget:
800 recurrent updates for 4, 8 and 16 source sessions, and 1600 for 33.
Balanced sampling gave 6400 draws per recording per recurrent stage. The
outer test windows are the same 87 windows across source counts. These plots
use one seed and show no uncertainty interval. The outer holdout also selects
transfer checkpoints; scores at selected checkpoints are descriptive.
"""


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


def observed_limits(axis: object, minimum_padding: float) -> tuple:
    """Find finite positive plotted values and a focused linear range."""
    values = np.asarray([
        value for line in axis.lines for value in line.get_ydata()
    ], dtype=float)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("Loss plot requires finite nonempty values.")
    if np.any(values <= 0):
        raise ValueError("Loss plot requires positive MSE values.")
    smallest, largest = float(values.min()), float(values.max())
    padding = max(
        minimum_padding, (largest - smallest) * RELATIVE_PADDING
    )
    return max(Y_AXIS_ORIGIN, smallest - padding), largest + padding


def ratio_axis(axis: object) -> None:
    """Focus on observed ratios while retaining above-dummy outcomes."""
    lower, upper = observed_limits(axis, RATIO_PADDING)
    if upper > BASELINE_RATIO and lower > BASELINE_RATIO:
        lower = BASELINE_RATIO
    axis.set_yscale("linear")
    axis.set_ylim(lower, upper)
    if lower <= BASELINE_RATIO <= upper:
        axis.axhline(
            BASELINE_RATIO, color=DUMMY_REFERENCE_COLOR,
            linestyle=":", linewidth=1.5,
        )
    axis.set_ylabel(BASELINE_LABEL)
    axis.grid(alpha=0.2)


def mse_axis(axis: object) -> None:
    """Display the four-horizon neural MSE on its original linear scale."""
    axis.set_yscale("linear")
    axis.set_ylim(*observed_limits(axis, MSE_PADDING))
    axis.set_ylabel("Neural MSE, four-horizon sum")
    axis.grid(alpha=0.2)


def configure_axis(axis: object, normalized: bool = True) -> None:
    """Set source-count ticks and the requested neural-loss measure."""
    axis.set_xscale("log", base=2)
    axis.set_xticks(COUNTS, labels=[str(count) for count in COUNTS])
    axis.set_xlim(min(COUNTS) / 1.08, max(COUNTS) * 1.08)
    axis.set_xlabel("Source sessions")
    if normalized:
        ratio_axis(axis)
    else:
        mse_axis(axis)


def finish_figure(
    figure: object, path: Path, title: str, subtitle: str,
    legend: bool = False,
) -> None:
    """Save a compact diagnostic view with an explicit target subtitle."""
    figure.suptitle(title, fontsize=24, y=0.98)
    figure.text(
        0.5, 0.915, subtitle, fontsize=15,
        ha="center", va="center",
    )
    if legend:
        figure.axes[0].legend(
            loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False,
        )
    if len(figure.axes) == len(HORIZONS):
        figure.subplots_adjust(
            left=0.09, right=0.98, top=0.79, bottom=0.10,
            hspace=0.48, wspace=0.30,
        )
    else:
        figure.subplots_adjust(
            left=0.14, right=0.65 if legend else 0.96,
            top=0.81, bottom=0.18,
        )
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
        axis.set_title(f"{milliseconds:g} ms (h={horizon})", pad=10)
    finish_figure(
        figure, out / "neural_loss_by_horizon.png",
        "Full neural forecast by horizon",
        "Smoothed spikes; RNN1 readout + RNN2 residual; 87 test windows",
    )
    figure, axis = plt.subplots(figsize=SINGLE_SIZE)
    for column, label, color, style in (
        ("mean_readout_loss", "RNN1 readout", ORANGE, "--"),
        ("mean_combined_loss", "Readout + RNN2 residual", BLUE, "-"),
    ):
        plot_line(
            x=COUNTS, y=ratio_curve(summary, {"role": "outer_test"}, column),
            ax=axis, color=color, linestyle=style, marker="o", label=label,
        )
    configure_axis(axis)
    finish_figure(
        figure, out / "neural_readout_and_combined_loss.png",
        "Neural readout versus full forecast",
        "Outer test; smoothed, train-normalized spike counts",
        legend=True,
    )
    figure, axis = plt.subplots(figsize=SINGLE_SIZE)
    for role, label, color, style in (
        ("training", "Training windows", BLUE, "-"),
        ("inner_validation", "Inner validation", GREEN, "--"),
        ("outer_test", "Outer chronological test", ORANGE, "-."),
    ):
        plot_line(
            x=COUNTS,
            y=loss_curve(summary, {"role": role}, "mean_combined_loss"),
            ax=axis, color=color, linestyle=style, marker="o", label=label,
        )
    configure_axis(axis, normalized=False)
    finish_figure(
        figure, out / "neural_generalization.png",
        "Full neural forecast by data split",
        "Raw MSE on smoothed, train-normalized spike counts",
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
        "Full neural forecast by recording",
        "Outer chronological test; same four recordings at every count",
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
        "Neural readout versus training compute",
        "Outer test; squares mark transfer-selected checkpoints",
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
        rendering=RENDERING_STYLE, guide=FIGURE_GUIDE,
        baseline_axis=dict(
            measure=BASELINE_LABEL,
            reference=BASELINE_RATIO,
            focus_padding=RATIO_PADDING,
            relative_padding=RELATIVE_PADDING,
            includes_reference_if_model_is_worse_than_dummy=True,
            dummy="constant training-target mean on the matching windows",
        ),
        generalization_axis=dict(
            measure="Neural MSE, four-horizon sum",
            focus_padding=MSE_PADDING,
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
            (destination / "figure_guide.md").write_text(FIGURE_GUIDE)
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
