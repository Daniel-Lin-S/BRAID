"""Score direct-behavior source fits on the fixed four-session holdout.

Input is a completed source-only plan and its saved source checkpoints.
Output is one immutable report with tables for source losses, selected
checkpoints and per-recording CC, R2, MSE and dummy MSE. Its figures/
directory contains two source-loss and nine performance PNG figures.
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
from experiments.multisession.behavior_compute_scaling.plan import (
    COUNTS, RECURRENT_STAGES,
)
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
from experiments.multisession.store import (
    completed, fit_directory, job,
)
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
STAGE_LOSS_RELATIVE_TOLERANCE = 5e-4
STAGE_LOSS_ABSOLUTE_TOLERANCE = 1e-5
GUIDE = """# Direct recorded-behavior source scaling

The source_behavior_relevant.png and source_residual.png figures plot
chronological source test loss against full-pipeline training FLOPs.
Each shows seven recurrent budgets for each of four source counts.
Source loss sums squared error across the four trained horizons and
equally averages the four common recordings. The residual stage scores
the full neural prediction, including the matched neural readout.

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

The tables/ directory contains source_loss.csv, source_session_loss.csv,
selection.csv, session_scores.csv and summary.csv. The first two give
stage losses by budget and recording; selection.csv records the chosen
matched budget. The last two give performance by recording and mean.
"""


def verify_stage_loss(
    source: dict, scores: list[dict], stage: str, groups: int,
) -> None:
    """Compare scored MSE with saved loss across numerical backends."""
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
            reconstructed, record["loss"],
            rtol=STAGE_LOSS_RELATIVE_TOLERANCE,
            atol=STAGE_LOSS_ABSOLUTE_TOLERANCE,
        ):
            raise ValueError(
                f"{stage} holdout loss differs from scored MSE for "
                f"{name}: saved {record['loss']}, "
                f"recomputed {reconstructed}."
            )


def full_neural_forecasts(
    source: dict, common: dict,
) -> list[np.ndarray]:
    """Restore the selected neural forecast for four common recordings."""
    spec = source["spec"]
    names = spec["sessions"]
    objective = spec.get("objective")
    neural_only = objective in ("neural", "neural_residual")
    dimensions = [
        (
            sum(source["inventory"][name]["scales"]["Y"]["keep"]),
            sum(source["inventory"][name]["scales"]["Y"]["keep"])
            if neural_only else
            sum(source["inventory"][name]["scales"]["Z"]["keep"]),
        )
        for name in names
    ]
    root = source["fit"].parent.parent / "source_components"
    stages = source["path"]["stages"]
    inputs = [
        recurrent_data(common[name]["test"])
        for name in names[:COMMON_SESSIONS]
    ]
    if neural_only:
        from experiments.multisession.compute_scaling.self_supervised import (
            neural_model,
        )
        main = neural_model(dimensions, spec)
        restore_component(
            main, root / stages["neural"]["component_id"]
        )
        if objective == "neural":
            forecasts, _ = stage_predictions(main, inputs)
        else:
            from experiments.multisession.compute_scaling.self_supervised import (
                residual_model,
            )
            residual = residual_model(dimensions, spec)
            restore_component(
                residual, root / stages["residual"]["component_id"]
            )
            rows = residual_data(main, None, inputs)
            corrections, _ = stage_predictions(residual, rows)
            forecasts = [
                prediction + row["P"]
                for prediction, row in zip(corrections, rows)
            ]
    else:
        main = rnn1_model(dimensions, spec)
        readout = neural_readout_model(dimensions, spec)
        residual = rnn2_model(dimensions, spec)
        for model, stage in (
            (main, "behavior_relevant"),
            (readout, "neural_readout"),
            (residual, "residual"),
        ):
            restore_component(
                model, root / stages[stage]["component_id"]
            )
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


def figures(
    summary: list[dict], directory: Path,
    baseline: list[dict] | None = None,
    comparison: list[dict] | None = None,
) -> None:
    """Draw source metrics with optional direct and one-session references."""
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
                        color=(
                            "#339966" if comparison is not None
                            else COLORS[group]
                        ),
                        marker=MARKERS[group],
                        label=(
                            "Neural-pretrained BRAID"
                            if comparison is not None
                            else "Multi-session model"
                            if baseline is not None else None
                        ),
                    )
                    if comparison is not None:
                        matching = sorted(
                            (
                                row for row in comparison
                                if row["group"] == group
                                and row["horizon"] == horizon
                            ),
                            key=lambda row: row["source_count"],
                        )
                        if [row["source_count"] for row in matching] != (
                            list(COUNTS)
                        ):
                            raise ValueError(
                                "Incomplete direct comparison curve."
                            )
                        reference = [row[metric] for row in matching]
                        if not np.isfinite(reference).all():
                            raise ValueError(
                                "Nonfinite direct comparison metric."
                            )
                        values.extend(reference)
                        plot_line(
                            x=COUNTS, y=reference, ax=axis,
                            color="#2878b5", marker="D",
                            linestyle="--", label="Direct behavior",
                        )
                    if baseline is not None:
                        matching = [
                            row for row in baseline
                            if row["group"] == group
                            and row["horizon"] == horizon
                        ]
                        if len(matching) != 1 or not np.isfinite(
                            matching[0][metric]
                        ):
                            raise ValueError(
                                f"Missing one-session baseline: "
                                f"{group}, {metric}, {horizon}."
                            )
                        value = matching[0][metric]
                        values.append(value)
                        axis.axhline(
                            value, color="#555555", linestyle="--",
                            linewidth=2.2,
                            label=(
                                "Single-session direct model"
                                if comparison is not None
                                else "Mean single-session model"
                            ),
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
                if baseline is not None or comparison is not None:
                    handles, labels = (
                        axes.flat[0].get_legend_handles_labels()
                    )
                    figure.legend(
                        handles, labels, loc="upper left",
                        bbox_to_anchor=(0.79, 0.87),
                        frameon=False,
                    )
                title = f"{group.capitalize()} source performance"
                figure.suptitle(
                    textwrap.fill(title, 46), y=0.98, fontsize=23,
                )
                figure.subplots_adjust(
                    left=0.10,
                    right=(
                        0.77 if baseline is not None
                        or comparison is not None else 0.97
                    ),
                    top=0.88, bottom=0.09,
                    hspace=0.45, wspace=0.32,
                )
                figure.savefig(
                    directory / f"{group}_{metric}.png",
                    dpi=DPI, bbox_inches="tight",
                )
                plt.close(figure)
    print(f"Saved source performance figures: {directory.resolve()}")


def publish_direct_report(plan: dict, plan_path: Path) -> Path | None:
    """Publish source losses and selected performance in one snapshot."""
    from .report import source_figures, source_tables

    references = plan["source_references"]
    if len(references) != len(COUNTS):
        raise ValueError("Expected one source fit per source count.")
    if any(
        completed(fit_directory(plan, key)) is None
        for key in references
    ):
        return None
    losses, session_losses, choices = source_tables(plan)
    if len(choices) != len(COUNTS):
        raise ValueError("Incomplete source checkpoint selection.")
    selected = decoding.checked_sources(plan, choices=choices)
    if any(
        target_mode(source["spec"]) != RECORDED
        for source in selected
    ):
        raise ValueError("Direct report requires recorded RNN1 targets.")
    identity = dict(
        group="direct_behavior_complete_source_report_v1",
        source_counts=list(COUNTS),
        source_completions={
            key: completed(fit_directory(plan, key))["checksums"]
            for key in references
        },
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
            tables = destination / "tables"
            figure_dir = destination / "figures"
            tables.mkdir()
            figure_dir.mkdir()
            write_table(tables / "source_loss.csv", losses)
            write_table(
                tables / "source_session_loss.csv", session_losses
            )
            write_table(tables / "selection.csv", choices)
            write_table(tables / "session_scores.csv", scores)
            write_table(tables / "summary.csv", summary)
            (destination / "guide.md").write_text(GUIDE)
            source_figures(
                losses, figure_dir, stages=RECURRENT_STAGES,
                expected_points=len(
                    plan["settings"]["source_training"]["total_updates"]
                ), direct=True,
            )
            figures(summary, figure_dir)
    marker = completed(destination)
    if marker is None:
        raise ValueError(f"Incomplete direct report: {destination}.")
    figure_dir = (destination / "figures").resolve()
    pointer = dict(
        snapshot=str(figure_dir), identity=fingerprint(identity),
    )
    for name in (
        "latest_source_figures.json", "latest_performance_figures.json",
        "latest_figures.json",
    ):
        atomic_json(root / name, pointer)
    atomic_json(root / "latest_report.json", dict(
        snapshot=str(destination.resolve()),
        figures=str(figure_dir),
        identity=fingerprint(identity),
    ))
    return destination
