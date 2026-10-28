"""Score direct-behavior source fits on the fixed four-session holdout.

Input is a completed source-only plan and its immutable source-loss report.
Output is one immutable sibling report with session_scores.csv (recording,
source count, horizon, target, CC, R2, MSE, dummy MSE and ratio), summary.csv
(equal four-recording means), guide.md, and nine target/metric PNG figures.
No fit or target adaptation is started.
"""

import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from BRAID.multisession import HORIZONS
from experiments.multisession.behavior_compute_scaling import (
    behavior_decoding_diagnostics as decoding,
)
from experiments.multisession.behavior_compute_scaling.ancestry import (
    restore_component,
)
from experiments.multisession.behavior_compute_scaling.plan import COUNTS
from experiments.multisession.behavior_supervision.models import (
    neural_readout_model, rnn1_model, rnn2_model,
)
from experiments.multisession.behavior_supervision.targets import (
    RECORDED, target_mode,
)
from experiments.multisession.data import SAMPLE_RATE
from experiments.multisession.plan import analysis_directory
from experiments.multisession.reports import write_table
from experiments.multisession.source import residual_data
from experiments.multisession.store import completed, job
from experiments.multisession.training import (
    recurrent_data, stage_predictions,
)
from experiments.reports.evaluation import score_channels
from experiments.storage.cache import atomic_json, fingerprint
from experiments.visualization.multisession import (
    READABLE_TEXT_STYLE, plot_line,
)

COMMON_SESSIONS = 4
EXPECTED_HORIZONS = (1, 2, 4, 8)
METRICS = ("cc", "r2", "mse_ratio")
GROUPS = ("position", "velocity", "neural")
COLORS = dict(
    position="#2878b5", velocity="#e28528", neural="#339966",
)
MARKERS = dict(position="o", velocity="s", neural="D")
FIGURE_SIZE = (14, 10)
DPI = 180
GUIDE = """# Direct recorded-behavior source performance

RNN1 predicts the four-coordinate, training-normalized recorded position
and velocity directly. The two n0 preprocessing stages and n3 are absent.
The full neural forecast is the RNN1 neural readout plus RNN2 residual.
Only the selected matched pipeline at each source count is shown here.
The same 87 chronological outer-test windows from the four common source
recordings are scored for every count; training and inner validation are
excluded. This holdout also selected the matched checkpoint by the sum of
RNN1 and RNN2 dummy-normalized losses, so the selected scores are descriptive.

For a recording, horizon and target group, MSE averages squared errors over
all scored time points and channels. Dummy MSE uses that recording's training
mean for each scored channel on those same test samples. Their ratio is
MSE/dummy MSE; values above one mean worse than dummy. CC and R2 average
valid per-channel values. Each plotted point averages four recording scores
equally. The four panels show the trained 50, 100, 200 and 400 ms horizons.
No target adaptation or uncertainty interval is included.
"""


def verify_stage_loss(
    source: dict, scores: list[dict], stage: str, groups: int,
) -> None:
    """Match scored horizon MSE to the selected native stage loss."""
    saved = source["path"]["stages"][stage]["score"]
    for record in saved["per_session"]:
        name = record["session"]
        rows = [row for row in scores if row["session"] == name]
        expected = len(HORIZONS) * groups
        if len(rows) != expected:
            raise ValueError(
                f"Expected {expected} {stage} scores for {name}."
            )
        reconstructed = sum(row["mse"] for row in rows) / groups
        if not np.isclose(
            reconstructed, record["loss"], rtol=1e-4, atol=1e-5
        ):
            raise ValueError(
                f"{stage} holdout loss differs from scored MSE for "
                f"{name}: saved {record['loss']}, "
                f"recomputed {reconstructed}."
            )


def full_neural_forecasts(
    source: dict, common: dict,
) -> list[np.ndarray]:
    """Restore matched RNN1, readout and RNN2 to score full neural Y."""
    spec = source["spec"]
    names = spec["sessions"]
    dimensions = [
        (
            sum(source["inventory"][name]["scales"]["Y"]["keep"]),
            sum(source["inventory"][name]["scales"]["Z"]["keep"]),
        )
        for name in names
    ]
    root = source["fit"].parent.parent / "source_components"
    stages = source["path"]["stages"]
    main = rnn1_model(dimensions, spec)
    restore_component(
        main, root / stages["behavior_relevant"]["component_id"]
    )
    readout = neural_readout_model(dimensions, spec)
    restore_component(
        readout, root / stages["neural_readout"]["component_id"]
    )
    residual = rnn2_model(dimensions, spec)
    restore_component(residual, root / stages["residual"]["component_id"])
    inputs = [
        recurrent_data(common[name]["test"])
        for name in names[:COMMON_SESSIONS]
    ]
    rows = residual_data(main, readout, inputs)
    corrections, _ = stage_predictions(residual, rows)
    forecasts = [
        prediction + row["P"]
        for prediction, row in zip(corrections, rows)
    ]
    for name, forecast in zip(names, forecasts):
        expected = (
            len(HORIZONS), *common[name]["test"]["Y"].shape
        )
        if forecast.shape != expected or not np.isfinite(forecast).all():
            raise ValueError(
                f"Invalid full neural forecast for {name}: "
                f"expected {expected}, got {forecast.shape}."
            )
    tf.keras.backend.clear_session()
    return forecasts


def neural_scores(
    source: dict, common: dict, forecasts: list[np.ndarray],
) -> list[dict]:
    """Score each recording's full normalized neural forecast."""
    names = source["spec"]["sessions"][:COMMON_SESSIONS]
    if len(forecasts) != len(names):
        raise ValueError("Expected four full neural forecasts.")
    rows = []
    for name, forecast in zip(names, forecasts):
        train = common[name]["training"]["Y"]
        test = common[name]["test"]["Y"]
        mean = train.mean(axis=(0, 1))
        for index, horizon in enumerate(HORIZONS):
            truth = test[:, horizon:]
            guess = forecast[index, :, :-horizon]
            flat_truth = truth.reshape(-1, truth.shape[-1])
            flat_guess = guess.reshape(-1, guess.shape[-1])
            result = score_channels(
                flat_truth, flat_guess,
                [f"neural_{i}" for i in range(truth.shape[-1])],
                dict(
                    session=name, fold="source", model="selected_full",
                    horizon=horizon,
                    evaluation_set="chronological_source",
                    target="neural",
                ),
            )
            dummy = float(np.square(truth - mean).mean())
            values = (
                result["mean_cc"], result["mean_r2"],
                result["mean_mse"], dummy,
            )
            if any(value is None for value in values) or (
                not np.isfinite(values).all() or dummy <= 0
            ):
                raise ValueError(
                    f"Invalid neural score for {name}, horizon {horizon}."
                )
            rows.append(dict(
                source_count=source["count"],
                source_fit=source["fit"].name,
                selected_budget=source["selection"]["selected_budget"],
                evaluation_set_id=source["selection"]["evaluation_set_id"],
                session=name, horizon=horizon, group="neural",
                samples=len(flat_truth),
                cc=float(result["mean_cc"]),
                r2=float(result["mean_r2"]),
                mse=float(result["mean_mse"]),
                dummy_mse=dummy,
                mse_ratio=float(result["mean_mse"] / dummy),
            ))
    return rows


def summarize(scores: list[dict]) -> list[dict]:
    """Equally average exactly four recording scores at each point."""
    summary = []
    for count in COUNTS:
        for horizon in EXPECTED_HORIZONS:
            for group in GROUPS:
                selected = [
                    row for row in scores
                    if row["source_count"] == count
                    and row["horizon"] == horizon
                    and row["group"] == group
                ]
                if len(selected) != COMMON_SESSIONS or (
                    len({row["session"] for row in selected})
                    != COMMON_SESSIONS
                ):
                    raise ValueError(
                        f"Expected four scores for {count}, "
                        f"horizon {horizon}, {group}."
                    )
                summary.append(dict(
                    source_count=count, horizon=horizon, group=group,
                    recordings=COMMON_SESSIONS,
                    **{
                        metric: float(np.mean([
                            row[metric] for row in selected
                        ]))
                        for metric in METRICS + ("mse",)
                    },
                ))
    return summary


def figures(summary: list[dict], directory: Path) -> None:
    """Draw separate readable target and metric horizon panels."""
    directory.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(READABLE_TEXT_STYLE):
        for group in GROUPS:
            for metric in METRICS:
                figure, axes = plt.subplots(
                    2, 2, figsize=FIGURE_SIZE
                )
                for axis, horizon in zip(axes.flat, EXPECTED_HORIZONS):
                    selected = sorted(
                        (
                            row for row in summary
                            if row["group"] == group
                            and row["horizon"] == horizon
                        ),
                        key=lambda row: row["source_count"],
                    )
                    if [row["source_count"] for row in selected] != (
                        list(COUNTS)
                    ):
                        raise ValueError(
                            f"Incomplete {group} {metric} horizon "
                            f"{horizon} curve."
                        )
                    values = [row[metric] for row in selected]
                    if not np.isfinite(values).all():
                        raise ValueError(
                            f"Nonfinite {group} {metric} values."
                        )
                    plot_line(
                        x=COUNTS, y=values, ax=axis,
                        color=COLORS[group], marker=MARKERS[group],
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
                    milliseconds = horizon * 1000 / SAMPLE_RATE
                    axis.set_title(f"{milliseconds:g} ms", pad=10)
                    decoding.focus_axis(axis, values, metric)
                title = f"{group.capitalize()} source performance"
                figure.suptitle(
                    textwrap.fill(title, 46), y=0.98, fontsize=23,
                )
                figure.subplots_adjust(
                    left=0.10, right=0.97, top=0.88,
                    bottom=0.09, hspace=0.45, wspace=0.32,
                )
                figure.savefig(
                    directory / f"{group}_{metric}.png",
                    dpi=DPI, bbox_inches="tight",
                )
                plt.close(figure)
    print(f"Saved direct-behavior performance figures: {directory.resolve()}")


def publish_direct_performance(
    plan: dict, plan_path: Path, source_report: Path,
) -> Path:
    """Publish matched behavior and full-neural scores after all sources."""
    selected = decoding.checked_sources(plan, source_report)
    if any(
        target_mode(source["spec"]) != RECORDED
        for source in selected
    ):
        raise ValueError("Direct report requires recorded RNN1 targets.")
    marker = completed(source_report)
    if marker is None:
        raise ValueError(f"Incomplete source report: {source_report}.")
    identity = dict(
        group="direct_behavior_source_performance",
        source_report_checksums=marker["checksums"],
        selected_components={
            str(source["count"]): source["path"]["stages"]
            for source in selected
        },
        guide=GUIDE,
    )
    root = analysis_directory(
        plan, plan["name"], plan_path.parent.name,
    )
    destination = root / "reports" / fingerprint(identity)
    with job(destination, identity) as needed:
        if needed:
            common = decoding.fixed_windows(plan, selected)
            scores = []
            for source in selected:
                behavior = decoding.selected_forecasts(source, common)
                behavior_scores = decoding.session_scores(
                    source, common, behavior
                )
                verify_stage_loss(
                    source, behavior_scores, "behavior_relevant", 2
                )
                scores.extend(behavior_scores)
                neural = full_neural_forecasts(source, common)
                neural_rows = neural_scores(source, common, neural)
                verify_stage_loss(source, neural_rows, "residual", 1)
                scores.extend(neural_rows)
            summary = summarize(scores)
            write_table(destination / "session_scores.csv", scores)
            write_table(destination / "summary.csv", summary)
            (destination / "guide.md").write_text(GUIDE)
            atomic_json(destination / "source_reference.json", dict(
                snapshot=str(source_report.resolve()),
            ))
            figures(summary, destination / "figures")
    atomic_json(root / "latest_performance_figures.json", dict(
        snapshot=str((destination / "figures").resolve()),
        identity=fingerprint(identity),
    ))
    return destination
