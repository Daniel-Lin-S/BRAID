"""Fit and compare session-private n3 widths at selected source checkpoints.

Usage: python -m experiments.multisession.behavior_compute_scaling.n3_sweep
  --stage launch|report --plan ABSOLUTE_PLAN
  --source-report ABSOLUTE_REPORT [--microbatch N]

Inputs are the selected source report, its plan, and private n3 completions
for dimensions 1, 2, 4, 8 and 16. Dimension 0 uses frozen RNN1. Output is
one immutable report with session_scores.csv, summary.csv, a guide, and six
position/velocity metric figures. Figures require every fit to complete.
"""

import argparse
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from BRAID.multisession import HORIZONS
from experiments.multisession.behavior_compute_scaling import (
    behavior_decoding_diagnostics as decoding,
    n3_session,
)
from experiments.multisession.behavior_compute_scaling.plan import COUNTS
from experiments.multisession.plan import analysis_directory
from experiments.multisession.reports import write_table
from experiments.multisession.store import completed, job
from experiments.storage.cache import atomic_json, file_digest, fingerprint
from experiments.visualization.multisession import (
    READABLE_TEXT_STYLE, plot_line,
)

SIZES = (0, 1, 2, 4, 8, 16)
FIT_SIZES = SIZES[1:]
COLORS = (
    "#222222", "#2878b5", "#e28528",
    "#339966", "#a04fa3", "#ad5b38",
)
LINE_STYLES = ("-", "--", "-.", ":", "-", "--")
MARKERS = ("o", "s", "^", "D", "v", "P")
METRICS = ("cc", "r2", "mse_ratio")
FIGURE_SIZE = (15, 11)
DPI = 180
GUIDE = """# Recorded-behavior source decoding by private n3 width

Each panel scores the same 87 chronological outer-test windows across four
common source sessions. The selected RNN1 checkpoint for each source count
is held fixed. At n3=0, prediction is the RNN1 forecast converted to the
recorded-behavior training-normalized scale. At n3>0, an independently
fitted session-private U-only n3 correction is added to that forecast.
The private n3 checkpoint is selected by that recording's inner validation.
No target adaptation is used. The source holdout was used to select the
RNN1/RNN2 transfer checkpoint, so these scores are descriptive.

Position and velocity each contain two coordinates. For each recording,
CC and R2 are coordinate means, and MSE is the mean squared error over both
coordinates and all scored test samples. Dummy MSE uses that recording's
training-target mean on the same test samples. MSE/dummy MSE is the ratio of
those two MSEs; one is dummy performance and values above one are valid.
Each plotted point equally averages the four recording-level scores.
The tables retain per-recording and per-coordinate-group evidence.
"""


def validate_components(selected: dict[int, dict]) -> dict:
    """Require checksum-valid private fits for every positive n3 width."""
    references = {}
    for count in COUNTS:
        source = selected[count]
        references[str(count)] = {}
        for index in range(n3_session.COMMON_SESSIONS):
            name = n3_session.session_name(source, index)
            references[str(count)][name] = {}
            for size in FIT_SIZES:
                path = n3_session.directory(source, index, size)
                marker = completed(path)
                if marker is None:
                    raise ValueError(
                        f"Incomplete n3 size {size} fit: {path.resolve()}."
                    )
                references[str(count)][name][str(size)] = dict(
                    component=path.name, checksums=marker["checksums"],
                )
    return references


def score_all(
    plan: dict, selected: dict[int, dict],
) -> tuple[list[dict], list[dict]]:
    """Score all widths on the exact common chronological test windows."""
    common = decoding.fixed_windows(plan, list(selected.values()))
    scores = []
    for count in COUNTS:
        source = selected[count]
        names = source["spec"]["sessions"][:n3_session.COMMON_SESSIONS]
        priors = [
            n3_session.base_forecast(
                source, index, common[name]["test"]
            )
            for index, name in enumerate(names)
        ]
        for size in SIZES:
            forecasts = priors if size == 0 else [
                n3_session.corrected(
                    source, index, common[name]["test"],
                    priors[index], size,
                )
                for index, name in enumerate(names)
            ]
            scores.extend(
                dict(row, n3_size=size)
                for row in decoding.session_scores(
                    source, common, forecasts
                )
            )
    summary = []
    for size in SIZES:
        subset = [row for row in scores if row["n3_size"] == size]
        summary.extend(
            dict(row, n3_size=size)
            for row in decoding.summarize(subset)
        )
    return scores, summary


def figures(summary: list[dict], directory: Path) -> None:
    """Save six complete four-horizon comparisons with readable axes."""
    directory.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(READABLE_TEXT_STYLE):
        for group in decoding.COORDINATE_GROUPS:
            for metric in METRICS:
                figure, axes = plt.subplots(
                    2, 2, figsize=FIGURE_SIZE
                )
                for axis, horizon in zip(axes.flat, HORIZONS):
                    observed = []
                    for index, size in enumerate(SIZES):
                        selected = sorted(
                            (
                                row for row in summary
                                if row["group"] == group
                                and row["horizon"] == horizon
                                and row["n3_size"] == size
                            ),
                            key=lambda row: row["source_count"],
                        )
                        counts = [
                            row["source_count"] for row in selected
                        ]
                        if counts != list(COUNTS):
                            raise ValueError(
                                f"Missing {group} {metric} scores for "
                                f"horizon {horizon}, n3={size}."
                            )
                        values = [row[metric] for row in selected]
                        if not np.isfinite(values).all():
                            raise ValueError(
                                f"Nonfinite {group} {metric} scores."
                            )
                        observed.extend(values)
                        plot_line(
                            x=COUNTS, y=values, ax=axis,
                            color=COLORS[index],
                            linestyle=LINE_STYLES[index],
                            marker=MARKERS[index],
                            label=f"n3={size}",
                        )
                    axis.set_xscale("log", base=2)
                    axis.set_xticks(
                        COUNTS, labels=[str(count) for count in COUNTS]
                    )
                    axis.set_xlim(
                        min(COUNTS) / 1.08, max(COUNTS) * 1.08
                    )
                    axis.set_xlabel("Source sessions")
                    axis.set_ylabel(decoding.METRIC_LABELS[metric])
                    axis.set_title(f"{horizon * 50} ms", pad=12)
                    decoding.focus_axis(axis, observed, metric)
                handles, labels = axes.flat[0].get_legend_handles_labels()
                figure.legend(
                    handles, labels, loc="upper center",
                    bbox_to_anchor=(0.5, 0.89),
                    ncol=len(SIZES), frameon=False,
                )
                title = f"{group.capitalize()} decoding by n3 width"
                figure.suptitle(
                    textwrap.fill(title, width=48), y=0.99,
                    fontsize=23,
                )
                figure.subplots_adjust(
                    left=0.10, right=0.97, top=0.77,
                    bottom=0.09, hspace=0.45, wspace=0.31,
                )
                figure.savefig(
                    directory / f"n3_{group}_{metric}.png",
                    dpi=DPI, bbox_inches="tight",
                )
                plt.close(figure)
    print(f"Saved n3 sweep figures: {directory.resolve()}")


def publish(plan_path: Path, source_report: Path) -> Path:
    """Publish one immutable report after all sweep fits complete."""
    plan = json.loads(plan_path.read_text())
    selected = n3_session.sources(plan, source_report)
    references = validate_components(selected)
    source_marker = completed(source_report)
    if source_marker is None:
        raise ValueError(f"Incomplete source report: {source_report}.")
    identity = dict(
        group="behavior_n3_size_sweep_selected_source_scaling",
        protocol=n3_session.PROTOCOL,
        sizes=SIZES,
        source_plan_checksum=file_digest(plan_path),
        source_report_checksums=source_marker["checksums"],
        components=references,
        guide=GUIDE,
    )
    root = analysis_directory(
        plan, "multi_session_behavior_compute_scaling",
        plan_path.parent.name,
    )
    destination = root / "reports" / fingerprint(identity)
    with job(destination, identity) as needed:
        if needed:
            scores, summary = score_all(plan, selected)
            write_table(destination / "session_scores.csv", scores)
            write_table(destination / "summary.csv", summary)
            atomic_json(destination / "n3_components.json", references)
            (destination / "guide.md").write_text(GUIDE)
            figures(summary, destination / "figures")
    atomic_json(root / "latest_behavior_n3_sweep.json", dict(
        snapshot=str(destination.resolve()),
        identity=fingerprint(identity),
    ))
    return destination


def main() -> None:
    """Launch a session-parallel fit sweep or publish its full report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("launch", "report"))
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--source-report", required=True, type=Path)
    parser.add_argument("--microbatch", type=int, default=4)
    args = parser.parse_args()
    plan = args.plan.resolve()
    report = args.source_report.resolve()
    if args.stage == "launch":
        n3_session.launch(
            plan, report, args.microbatch,
            sizes=FIT_SIZES, report_kind="sweep",
        )
    else:
        publish(plan, report)


if __name__ == "__main__":
    main()
