"""Score recorded behavior from saved source RNN1 checkpoints.

Usage: python -m experiments.multisession.behavior_compute_scaling.
behavior_decoding_diagnostics --plan ABSOLUTE_PLAN
  --source-report ABSOLUTE_REPORT

Inputs are a completed four-count source plan/report, checksum-valid selected
component checkpoints, and the four common Indy recordings. Output is one
immutable report containing session_scores.csv (one row per recording, count,
horizon and coordinate group), summary.csv (four-recording means), three PNG
figures, guide.md, and input checksums. No fit or adaptation is started.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from BRAID.multisession import HORIZONS, MixedForecaster
from experiments.multisession.behavior_compute_scaling.ancestry import (
    inherited_paths, restore_component,
)
from experiments.multisession.behavior_compute_scaling.plan import COUNTS
from experiments.multisession.behavior_supervision.targets import (
    RECORDED, target_mode,
)
from experiments.multisession.data import (
    Scale, session_windows, source_features,
)
from experiments.multisession.reports import write_table
from experiments.multisession.runner import dataset
from experiments.multisession.store import completed, fit_directory, job
from experiments.multisession.training import recurrent_data, stage_predictions
from experiments.reports.evaluation import score_channels
from experiments.storage.cache import atomic_json, file_digest, fingerprint
from experiments.visualization.multisession import (
    READABLE_TEXT_STYLE, plot_line,
)

COORDINATE_GROUPS = {
    "position": (0, 1),
    "velocity": (2, 3),
}
COORDINATE_NAMES = (
    "position_x", "position_y", "velocity_x", "velocity_y",
)
GROUP_COLORS = {"position": "#2878b5", "velocity": "#e28528"}
GROUP_STYLES = {"position": "-", "velocity": "--"}
METRIC_LABELS = {
    "cc": "Correlation (CC)",
    "r2": "R²",
    "mse_ratio": "MSE / dummy MSE",
}
FIGURE_SIZE = (14, 10)
DPI = 180
RELATIVE_PADDING = 0.10
MINIMUM_PADDING = 0.02
BASELINE = {"cc": 0.0, "r2": 0.0, "mse_ratio": 1.0}
COMMON_SESSION_COUNT = 4
EXPECTED_TEST_WINDOWS = 87
GUIDE = """# Recorded-behavior source decoding

The three plots score RNN1 forecasts of recorded cursor position and
backward-difference velocity on the fixed chronological test windows of
the four common source recordings. RNN1 predicts a generated behavioral
target; its saved target scaling is inverted before comparison with the
recorded behavior. Both predictions and observations remain in the
recording's training-normalized behavioral coordinates.

Each point uses the existing transfer-selected checkpoint for that source
count. That selection minimizes the sum of RNN1 and RNN2 dummy-normalized
source losses on this same holdout, so these scores are descriptive.
Only the four horizons trained here are scored: 50, 100, 200 and 400 ms.

Position and velocity each average their two coordinates within a
recording. The plotted point averages the four recording scores equally.
MSE is divided by the MSE of the recording's training-target mean on the
same test samples; one is dummy performance. Values above one are valid.
CC and R² use the same test samples and are not dummy-normalized.
The axes focus on observed values; reference lines may be outside them.
The session_scores.csv table records each recording separately.

All four fits share the exact 87-window source test set. These recordings
were included in source training on earlier chronological windows.
Each metric figure has one panel per horizon, with separate position
and velocity curves. This analysis has one seed and no matched
no-sharing control. It describes
how decoding varies with source count; it cannot isolate the effect of
weight sharing or establish transfer to new recordings.
"""


def checked_sources(plan: dict, report: Path) -> list[dict]:
    """Validate four completed fits and one fixed evaluation identity."""
    if completed(report) is None:
        raise ValueError(f"Incomplete source report: {report.resolve()}.")
    with (report / "tables" / "selection.csv").open() as stream:
        reported = list(csv.DictReader(stream))
    references = plan["source_references"]
    if len(references) != len(COUNTS):
        raise ValueError("Expected one source reference per source count.")
    sources = []
    shared_names = None
    shared_identity = None
    for key in references:
        spec = plan["jobs"][key]
        fit = fit_directory(plan, key)
        if completed(fit) is None:
            raise ValueError(f"Incomplete source fit: {fit.resolve()}.")
        names = spec["sessions"][:COMMON_SESSION_COUNT]
        if len(names) != COMMON_SESSION_COUNT or len(set(names)) != len(names):
            raise ValueError("Expected four distinct common source sessions.")
        inventory = json.loads((fit / "data.json").read_text())
        windows = {name: inventory[name]["validation"] for name in names}
        identity = fingerprint(windows)
        selection = json.loads((fit / "selection.json").read_text())
        if identity != selection["evaluation_set_id"]:
            raise ValueError(f"Source holdout identity changed: {fit}.")
        if sum(len(value) for value in windows.values()) != (
            EXPECTED_TEST_WINDOWS
        ):
            raise ValueError("Expected 87 common source test windows.")
        if shared_names is None:
            shared_names, shared_identity = names, identity
        elif names != shared_names or identity != shared_identity:
            raise ValueError("Source counts use different common test sets.")
        paths = [
            row for row in inherited_paths(fit)
            if row["budget"] == selection["selected_budget"]
        ]
        if len(paths) != 1:
            raise ValueError("Expected one selected source checkpoint path.")
        sources.append(dict(
            count=spec["source_count"], spec=spec, fit=fit,
            inventory=inventory, selection=selection, path=paths[0],
        ))
    if {row["count"] for row in sources} != set(COUNTS):
        raise ValueError("Missing or duplicated source session count.")
    reported_choices = {
        (row["source_fit"], int(row["selected_budget"]))
        for row in reported
    }
    expected_choices = {
        (row["fit"].name, row["selection"]["selected_budget"])
        for row in sources
    }
    if reported_choices != expected_choices or (
        len(reported) != len(sources)
    ):
        raise ValueError("Source report checkpoint selections differ.")
    return sorted(sources, key=lambda row: row["count"])


def fixed_windows(plan: dict, sources: list[dict]) -> dict:
    """Rebuild the four common guarded splits without writing cache files."""
    first = sources[0]["spec"]
    names = first["sessions"][:COMMON_SESSION_COUNT]
    for source in sources[1:]:
        spec = source["spec"]
        if spec["data"] != first["data"] or (
            spec["source_training"]["inner_validation_fraction"]
            != first["source_training"]["inner_validation_fraction"]
        ):
            raise ValueError("Common source preprocessing settings differ.")
    adapter = dataset(plan["settings"])
    adapter.mode = "off"
    result = {}
    for name in names:
        features = source_features(
            adapter.load(name), first["data"],
            first["source_training"]["inner_validation_fraction"],
        )
        training, _, test, record = session_windows(
            features, True, include_test=True,
        )
        for source in sources:
            if source["inventory"][name] != record:
                raise ValueError(
                    f"Rebuilt source windows differ for {name} "
                    f"at {source['count']} sessions."
                )
        result[name] = dict(training=training, test=test)
    return result


def target_scale(record: dict) -> Scale:
    """Restore a saved four-coordinate generated-target scale."""
    scale = Scale(
        mean=np.asarray(record["mean"], dtype=float),
        scale=float(record["scale"]),
        keep=np.asarray(record["keep"], dtype=bool),
    )
    if scale.mean.shape != (len(COORDINATE_NAMES),) or (
        scale.keep.shape != scale.mean.shape or not scale.keep.all()
    ):
        raise ValueError("Generated behavior target needs four coordinates.")
    if not np.isfinite(scale.mean).all() or (
        not np.isfinite(scale.scale) or scale.scale <= 0
    ):
        raise ValueError("Invalid generated behavior target scale.")
    return scale


def recorded_forecasts(
    source: dict, partitions: list[dict],
    devices: tuple[str, str] | None = None,
) -> list[np.ndarray]:
    """Restore RNN1 and forecast recorded-Z normalized session windows."""
    spec = source["spec"]
    names = spec["sessions"]
    if not partitions or len(partitions) > len(names):
        raise ValueError("Expected nonempty ordered source partitions.")
    dimensions = [
        (sum(source["inventory"][name]["scales"]["Y"]["keep"]),
         len(COORDINATE_NAMES))
        for name in names
    ]
    model = MixedForecaster(
        dimensions, spec["latent"], "A", spec["seed"] + 2,
        decoder_width=64, session_devices=devices,
    )
    component = source["path"]["stages"]["behavior_relevant"]["component_id"]
    root = source["fit"].parent.parent / "source_components"
    restore_component(model, root / component)
    input_data = [recurrent_data(row, "Z") for row in partitions]
    predicted, _ = stage_predictions(model, input_data)
    if target_mode(source["spec"]) == RECORDED:
        forecasts = predicted
    else:
        preprocessing = json.loads(
            (source["fit"] / "preprocessing.json").read_text()
        )
        scales = preprocessing["scales"]
        if len(scales) != source["count"]:
            raise ValueError(
                "Generated behavior target scales do not match count."
            )
        forecasts = [
            target_scale(scales[index]).inverse(value)
            for index, value in enumerate(predicted)
        ]
    for name, row, value in zip(names, partitions, forecasts):
        expected = (len(HORIZONS), *row["Z"].shape)
        if value.shape != expected or not np.isfinite(value).all():
            raise ValueError(
                f"Expected finite forecast shape {expected} for {name}, "
                f"got {value.shape}."
            )
    tf.keras.backend.clear_session()
    return forecasts


def selected_forecasts(source: dict, windows: dict) -> list[np.ndarray]:
    """Forecast the four common chronological source test recordings."""
    names = source["spec"]["sessions"][:COMMON_SESSION_COUNT]
    return recorded_forecasts(
        source, [windows[name]["test"] for name in names]
    )


def session_scores(
    source: dict, windows: dict, forecasts: list[np.ndarray],
) -> list[dict]:
    """Score position and velocity against observed chronological behavior."""
    rows = []
    names = source["spec"]["sessions"][:COMMON_SESSION_COUNT]
    for name, forecast in zip(names, forecasts):
        training = windows[name]["training"]["Z"]
        test = windows[name]["test"]["Z"]
        dummy = training.mean(axis=(0, 1))
        for index, horizon in enumerate(HORIZONS):
            for group, coordinates in COORDINATE_GROUPS.items():
                truth = test[:, horizon:][:, :, coordinates]
                guess = forecast[index, :, :-horizon][:, :, coordinates]
                flat_truth = truth.reshape(-1, len(coordinates))
                flat_guess = guess.reshape(-1, len(coordinates))
                context = dict(
                    session=name, fold="source", model="selected_RNN1",
                    horizon=horizon, evaluation_set="chronological_source",
                    target="behavior",
                )
                result = score_channels(
                    flat_truth, flat_guess,
                    [COORDINATE_NAMES[x] for x in coordinates], context,
                )
                baseline = float(np.square(
                    truth - dummy[list(coordinates)]
                ).mean())
                values = (
                    result["mean_cc"], result["mean_r2"],
                    result["mean_mse"], baseline,
                )
                if any(value is None for value in values) or (
                    not np.isfinite(values).all() or baseline <= 0
                ):
                    raise ValueError(
                        f"Invalid behavior score for {name}, "
                        f"{group}, horizon {horizon}."
                    )
                rows.append(dict(
                    source_count=source["count"], source_fit=source["fit"].name,
                    selected_budget=source["selection"]["selected_budget"],
                    evaluation_set_id=source["selection"]["evaluation_set_id"],
                    session=name, horizon=horizon, group=group,
                    samples=len(flat_truth),
                    cc=float(result["mean_cc"]),
                    r2=float(result["mean_r2"]),
                    mse=float(result["mean_mse"]),
                    dummy_mse=baseline,
                    mse_ratio=float(result["mean_mse"] / baseline),
                ))
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    """Average four recording scores for each count, horizon and group."""
    summary = []
    for count in COUNTS:
        for horizon in HORIZONS:
            for group in COORDINATE_GROUPS:
                selected = [
                    row for row in rows
                    if row["source_count"] == count
                    and row["horizon"] == horizon
                    and row["group"] == group
                ]
                if len(selected) != COMMON_SESSION_COUNT or (
                    len({row["session"] for row in selected})
                    != COMMON_SESSION_COUNT
                ):
                    raise ValueError(
                        f"Expected four scores for {count} sessions, "
                        f"horizon {horizon}, {group}."
                    )
                summary.append(dict(
                    source_count=count, horizon=horizon, group=group,
                    recordings=COMMON_SESSION_COUNT,
                    **{
                        metric: float(np.mean([
                            row[metric] for row in selected
                        ]))
                        for metric in ("cc", "r2", "mse", "mse_ratio")
                    },
                ))
    return summary


def focus_axis(axis: object, values: list[float], metric: str) -> None:
    """Keep the full finite observed range visible on a linear axis."""
    data = np.asarray(values, dtype=float)
    if not data.size or not np.isfinite(data).all():
        raise ValueError(f"Missing or nonfinite {metric} plot values.")
    spread = float(data.max() - data.min())
    padding = max(MINIMUM_PADDING, spread * RELATIVE_PADDING)
    lower, upper = float(data.min() - padding), float(data.max() + padding)
    anchor = BASELINE[metric]
    if lower <= anchor <= upper:
        axis.axhline(anchor, color="#333333", linestyle=":", linewidth=1.3)
    axis.set_ylim(lower, upper)
    axis.grid(alpha=0.2)


@plt.rc_context(READABLE_TEXT_STYLE)
def figures(
    summary: list[dict], directory: Path,
    subtitle: str = (
        "Transfer-selected RNN1; four common chronological test sets"
    ),
    title_prefix: str = "Recorded behavior decoding",
    compact: bool = False,
) -> None:
    """Draw one panel per trained horizon for three behavior metrics."""
    for metric, label in METRIC_LABELS.items():
        figure, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE)
        for axis, horizon in zip(axes.flat, HORIZONS):
            all_values = []
            for group in COORDINATE_GROUPS:
                selected = [
                    row for row in summary
                    if row["group"] == group and row["horizon"] == horizon
                ]
                selected.sort(key=lambda row: row["source_count"])
                if [row["source_count"] for row in selected] != list(COUNTS):
                    raise ValueError("Incomplete behavior scaling curve.")
                values = [row[metric] for row in selected]
                all_values.extend(values)
                plot_line(
                    x=COUNTS, y=values, ax=axis,
                    color=GROUP_COLORS[group],
                    linestyle=GROUP_STYLES[group],
                    marker="o", label=group.capitalize(),
                )
            axis.set_xscale("log", base=2)
            axis.set_xticks(COUNTS, labels=[str(count) for count in COUNTS])
            axis.set_xlim(min(COUNTS) / 1.08, max(COUNTS) * 1.08)
            axis.set_xlabel("Source sessions")
            axis.set_ylabel(label)
            axis.set_title(f"{horizon * 50} ms (h={horizon})", pad=10)
            focus_axis(axis, all_values, metric)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        figure.legend(
            handles, labels, loc="upper center",
            bbox_to_anchor=(0.5, 0.90 if compact else 0.87),
            ncol=2, frameon=False,
        )
        figure.suptitle(
            f"{title_prefix}: {label}",
            fontsize=23, y=0.99 if compact else 0.98,
        )
        figure.text(
            0.5, 0.945 if compact else 0.92,
            subtitle,
            fontsize=15, ha="center", va="center",
        )
        figure.subplots_adjust(
            left=0.10, right=0.97,
            top=0.82 if compact else 0.78, bottom=0.10,
            hspace=0.47, wspace=0.31,
        )
        figure.savefig(
            directory / f"behavior_{metric}.png",
            dpi=DPI, bbox_inches="tight",
        )
        plt.close(figure)


def publish(plan_path: Path, source_report: Path) -> Path:
    """Publish checksum-validated behavior figures without starting fits."""
    plan = json.loads(plan_path.read_text())
    sources = checked_sources(plan, source_report)
    report_marker = completed(source_report)
    identity = dict(
        group="recorded_behavior_source_decoding",
        plan_checksum=file_digest(plan_path),
        report_checksums=report_marker["checksums"],
        source_fits={
            str(source["count"]): completed(source["fit"])["checksums"]
            for source in sources
        },
        selection="existing_sum_of_recurrent_dummy_ratios",
        scoring="inverse_generated_target_to_recorded_behavior_v1",
        metrics=("cc", "r2", "mse_ratio"),
        horizons=HORIZONS,
        groups=COORDINATE_GROUPS,
        rendering=dict(
            text=READABLE_TEXT_STYLE, figure_size=FIGURE_SIZE, dpi=DPI,
            colors=GROUP_COLORS, styles=GROUP_STYLES,
            relative_padding=RELATIVE_PADDING,
            minimum_padding=MINIMUM_PADDING,
        ),
        guide=GUIDE,
    )
    destination = source_report.parent / fingerprint(identity)
    with job(destination, identity) as needed:
        if needed:
            windows = fixed_windows(plan, sources)
            rows = []
            for source in sources:
                rows.extend(session_scores(
                    source, windows, selected_forecasts(source, windows)
                ))
            summary = summarize(rows)
            (destination / "figures").mkdir()
            write_table(destination / "session_scores.csv", rows)
            write_table(destination / "summary.csv", summary)
            (destination / "guide.md").write_text(GUIDE)
            figures(summary, destination / "figures")
    latest = source_report.parent.parent / "latest_behavior_decoding.json"
    atomic_json(latest, dict(
        snapshot=str(destination.resolve()), identity=fingerprint(identity),
    ))
    print(
        "Saved behavioral decoding figures: "
        f"{(destination / 'figures').resolve()}"
    )
    return destination


def main() -> None:
    """Publish plots from an explicit completed plan and source report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    args = parser.parse_args()
    publish(args.plan.resolve(), args.source_report.resolve())


if __name__ == "__main__":
    main()
