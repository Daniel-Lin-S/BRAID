"""Fit paired two- and three-stage direct-behavior session models.

Usage: python -m experiments.multisession.behavior_compute_scaling.paired_stages
  --stage audit|launch|worker|coordinator|report --plan ABSOLUTE_PLAN
  [--launch-dir ABSOLUTE_LOG_DIR] [--session-index 0..3]
  [--seed 42|43|44] [--component COMPONENT] [--microbatch N]

Input is the completed direct-source plan and its four saved single-session
split inventories. Workers rebuild normalized train/inner/test windows and
verify their exact indices and scales. Recurrent training uses train and inner
only. Output components are checksum-valid checkpoint.pkl and training.json
under stage_ablation/source_components/. Model fit manifests under
stage_ablation/fits/ reference those components and the saved split identity.
One immutable report contains neural_scores.csv, paired_differences.csv,
behavior_scores.csv, seed_summary.csv, summary.csv, guide.md, and one figure.
Campaign-local logs contain status.json, launch.json, and worker failures.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from BRAID.multisession import HORIZONS
from experiments.execution.runtime import select_gpus, thread_environment
from experiments.multisession.behavior_compute_scaling import (
    behavior_decoding_diagnostics as decoding,
    direct_report,
    single_session_baseline as baseline,
)
from experiments.multisession.behavior_supervision.models import (
    neural_readout_model, rnn1_model, rnn2_direct_model, rnn2_model,
)
from experiments.multisession.compute_scaling.ancestry import restore_component
from experiments.multisession.compute_scaling.training import (
    balanced_update, evaluate_sessions, fit_segment, gradient_function,
    optimizer_groups, segment_identity,
)
from experiments.multisession.data import SAMPLE_RATE
from experiments.multisession.plan import analysis_directory
from experiments.multisession.reports import write_table
from experiments.multisession.source import readout_data, residual_data
from experiments.multisession.store import completed, job
from experiments.multisession.training import recurrent_data, stage_predictions
from experiments.storage.cache import atomic_json, fingerprint
from experiments.visualization.multisession import (
    READABLE_TEXT_STYLE, plot_line,
)

SEEDS = (42, 43, 44)
STAGES = (
    "behavior_relevant", "neural_direct",
    "neural_readout", "residual",
)
ARCHITECTURES = {
    "two_stage": ("behavior_relevant", "neural_direct"),
    "three_stage": (
        "behavior_relevant", "neural_readout", "residual"
    ),
}
DEPENDENCIES = {
    "behavior_relevant": (),
    "neural_direct": (),
    "neural_readout": ("behavior_relevant",),
    "residual": ("behavior_relevant", "neural_readout"),
}
PREFIXES = {
    "behavior_relevant": "paired_rnn1_",
    "neural_direct": "paired_direct_neural_",
    "neural_readout": "paired_readout_",
    "residual": "paired_residual_",
}
SEED_OFFSETS = {
    "behavior_relevant": 2,
    "neural_direct": 4,
    "neural_readout": 3,
    "residual": 4,
}
MAIN_UPDATES = 2000
TOTAL_UPDATES = 2500
READOUT_UPDATES = 2000
WARMUP_UPDATES = 10
VALIDATION_EXPOSURE = 200
PATIENCE = 10
SESSION_BATCH = 8
AUDIT_UPDATES = 2
MINIMUM_FREE_MIB = 6144
GPU_MEMORY_LIMIT_MIB = 3072
MAXIMUM_GPU_UTILIZATION = 10
POLL_SECONDS = 10
LOW_CPU_PRIORITY = 10
WORKER_CPU_THREADS = 2
LOCAL_DEVICE = ("/GPU:0", "/GPU:0")
PROTOCOL = "paired_direct_stages_2500_inner_stop_v1"
METRICS = ("cc", "r2", "mse_ratio")
COLORS = {"two_stage": "#2878b5", "three_stage": "#e28528"}
STYLES = {"two_stage": "-", "three_stage": "--"}
MARKERS = {"two_stage": "o", "three_stage": "s"}
METRIC_LABELS = {
    "cc": "Correlation (CC)",
    "r2": "R²",
    "mse_ratio": "MSE / dummy MSE",
}
FIGURE_SIZE = (17, 6)
HORIZON_AXIS_LIMITS = (40, 410)
MILLISECONDS_PER_SECOND = 1000
EXPECTED_GPU_COUNT = 8
DPI = 180
AXIS_PADDING_FRACTION = 0.10
AXIS_MINIMUM_PADDING = 0.02
GUIDE = """# BRAID stage ablation

The two-stage model consists of a recorded-behavior RNN1 and an independent
direct-neural RNN2. The three-stage model uses the identical RNN1 checkpoint,
a neural readout of its latent states, and a residual RNN2. No parameters are
shared across sessions. Learned n0 preprocessing and n3 are absent. The
saved four-session split, spike smoothing, and training-only normalization
are unchanged.

Each recurrent component has at most 2,500 optimizer updates: ten warmup,
2,000 main, and 500 decay. Inner-validation loss is checked every 200
sampled windows (25 updates at batch eight); patience is ten checks from
update 25, and the best checkpoint is restored. The three-stage neural
readout retains its 2,000-update cap and the same validation cadence and
patience. The chronological outer test is used only after all fits complete.

Neural scores cover 50, 100, 200, and 400 ms. CC and R² average valid
channels within a session; MSE/dummy MSE divides model MSE by the training-
mean dummy's MSE on the same test samples. Summary points average four
sessions equally within each seed, then the three seed means equally.
Figure error bars show one sample standard deviation across four session
means, each averaged over the three seeds.
The paired table has three-stage minus two-stage scores, so positive CC
and R² differences favor three stages, while negative MSE-ratio
differences favor three stages. Behavior scores document the shared
RNN1 predictor and are not counted as an architecture difference.

This compares complete architectures. Removing the neural readout also
changes RNN2 from a state-conditioned residual predictor to an independent
direct predictor. Differences cannot be attributed to the readout alone.
"""


def paired_spec(base: dict, seed: int) -> dict:
    """Keep the reference science fixed while enabling paired stopping."""
    if seed not in SEEDS:
        raise ValueError(f"Expected one of paired seeds {SEEDS}; got {seed}.")
    if base["latent"] != 16 or base["source_count"] != 1:
        raise ValueError("Expected one latent-16 source session.")
    settings = dict(
        base["source_training"],
        checkpoints=[MAIN_UPDATES],
        total_updates=[TOTAL_UPDATES],
        maximum_updates=MAIN_UPDATES,
        warmup_updates=WARMUP_UPDATES,
    )
    if settings["session_batch_sizes"]["1"] != SESSION_BATCH or (
        settings["fixed_update_cap"] != READOUT_UPDATES
        or settings["validation_exposure"] != VALIDATION_EXPOSURE
        or settings["fixed_patience"] != PATIENCE
    ):
        raise ValueError(
            "Saved session batch, readout cap, validation cadence, "
            "or patience differs from the paired protocol."
        )
    return dict(
        base, seed=seed, implementation=PROTOCOL,
        source_training=settings,
    )


def work_items(plan: dict) -> list[dict]:
    """Resolve twelve session/seed jobs from saved split inventories."""
    items = []
    for index in range(baseline.COMMON_SESSIONS):
        base, record, previous = baseline.saved_spec(plan, index)
        if completed(previous) is None:
            raise ValueError(
                f"Incomplete three-stage session reference: "
                f"{previous.resolve()}."
            )
        for seed in SEEDS:
            items.append(dict(
                index=index, seed=seed, name=base["sessions"][0],
                spec=paired_spec(base, seed), record=record,
            ))
    if len(items) != baseline.COMMON_SESSIONS * len(SEEDS):
        raise ValueError("Expected twelve paired session/seed jobs.")
    return items


def component_root(plan: dict) -> Path:
    """Return the existing common source-component store."""
    artifact = Path(plan["settings"]["paths"]["artifact_root"])
    return artifact / "stage_ablation" / "source_components"


def component_path(
    plan: dict, item: dict, stage: str,
) -> Path | None:
    """Address a stage once all checksum-verified parents exist."""
    if stage not in STAGES:
        raise ValueError(f"Unknown paired component {stage}.")
    upstream = {}
    for name in DEPENDENCIES[stage]:
        parent = component_path(plan, item, name)
        if parent is None:
            return None
        marker = completed(parent)
        if marker is None:
            return None
        upstream[name] = marker["checksums"]
    identity = dict(
        protocol=PROTOCOL, spec=item["spec"], stage=stage,
        upstream_checksums=upstream,
    )
    stop = (
        READOUT_UPDATES if stage == "neural_readout"
        else TOTAL_UPDATES
    )
    scientific = segment_identity(
        identity, 0, stop, False, None, True,
        None if stage == "neural_readout"
        else (MAIN_UPDATES, TOTAL_UPDATES),
    )
    return (
        component_root(plan)
        / f"{PREFIXES[stage]}{fingerprint(scientific)}"
    )


def fit_stage(
    plan: dict, item: dict, stage: str, microbatch: int,
) -> Path:
    """Fit or reuse one independent native component on one GPU."""
    base, train, inner, _, record, _ = baseline.prepare(
        plan, item["index"]
    )
    if record != item["record"] or paired_spec(base, item["seed"]) != (
        item["spec"]
    ):
        raise ValueError(
            f"Paired session split or specification changed for "
            f"{item['name']}, seed {item['seed']}."
        )
    spec = item["spec"]
    dimensions = [(train["Y"].shape[-1], train["Z"].shape[-1])]
    neural = [recurrent_data(row) for row in (train, inner)]
    behavior = [recurrent_data(row, "Z") for row in (train, inner)]
    if stage == "behavior_relevant":
        model = rnn1_model(dimensions, spec, LOCAL_DEVICE)
        data = [behavior[0]]
        validation = [behavior[1]]
    elif stage == "neural_direct":
        model = rnn2_direct_model(dimensions, spec, LOCAL_DEVICE)
        data = [neural[0]]
        validation = [neural[1]]
    else:
        first = component_path(plan, item, "behavior_relevant")
        if first is None:
            raise ValueError("Neural readout requires completed RNN1.")
        main = rnn1_model(dimensions, spec, LOCAL_DEVICE)
        restore_component(main, first)
        if stage == "neural_readout":
            latent = [
                stage_predictions(main, [row])[1]
                for row in behavior
            ]
            model = neural_readout_model(
                dimensions, spec, LOCAL_DEVICE
            )
            data = readout_data([neural[0]], latent[0])
            validation = readout_data([neural[1]], latent[1])
        elif stage == "residual":
            second = component_path(
                plan, item, "neural_readout"
            )
            if second is None:
                raise ValueError(
                    "Residual RNN2 requires completed neural readout."
                )
            readout = neural_readout_model(
                dimensions, spec, LOCAL_DEVICE
            )
            restore_component(readout, second)
            model = rnn2_model(dimensions, spec, LOCAL_DEVICE)
            data = residual_data(main, readout, [neural[0]])
            validation = residual_data(
                main, readout, [neural[1]]
            )
        else:
            raise ValueError(f"Unknown paired component {stage}.")
    target = component_path(plan, item, stage)
    if target is None:
        raise ValueError(f"Parents are incomplete for {stage}.")
    identity = dict(
        protocol=PROTOCOL, spec=spec, stage=stage,
        upstream_checksums={
            name: completed(component_path(plan, item, name))[
                "checksums"
            ] for name in DEPENDENCIES[stage]
        },
    )
    settings = spec["source_training"]
    stop = (
        READOUT_UPDATES if stage == "neural_readout"
        else TOTAL_UPDATES
    )
    directory, _ = fit_segment(
        model, data, component_root(plan), identity, settings,
        spec["seed"] + SEED_OFFSETS[stage], 0, stop, None,
        LOCAL_DEVICE, microbatch, validation=validation,
        recurrent_schedule=(
            None if stage == "neural_readout"
            else (MAIN_UPDATES, TOTAL_UPDATES)
        ),
        directory_prefix=PREFIXES[stage],
    )
    if directory != target or completed(directory) is None:
        raise ValueError(
            f"Completed component identity changed: "
            f"{directory.resolve()}."
        )
    return directory


def fit_identity(
    plan: dict, item: dict, architecture: str,
) -> tuple[dict, dict] | None:
    """Bind one model fit to its completed, shared components."""
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture {architecture}.")
    components = {}
    for stage in ARCHITECTURES[architecture]:
        path = component_path(plan, item, stage)
        marker = completed(path) if path is not None else None
        if marker is None:
            return None
        components[stage] = dict(
            component_id=path.name, checksums=marker["checksums"],
        )
    identity = dict(
        protocol=PROTOCOL, architecture=architecture,
        spec=item["spec"], components=components,
    )
    return identity, components


def fit_path(
    plan: dict, item: dict, architecture: str,
) -> Path | None:
    """Return a model-fit directory once all components are complete."""
    resolved = fit_identity(plan, item, architecture)
    if resolved is None:
        return None
    identity, _ = resolved
    artifact = Path(plan["settings"]["paths"]["artifact_root"])
    return (
        artifact / "stage_ablation" / "fits"
        / f"paired_{architecture}_{fingerprint(identity)}"
    )


def finalize_fit(
    plan: dict, item: dict, architecture: str,
) -> Path | None:
    """Write one small immutable manifest after its components finish."""
    resolved = fit_identity(plan, item, architecture)
    if resolved is None:
        return None
    identity, components = resolved
    directory = fit_path(plan, item, architecture)
    if directory is None:
        raise ValueError("Resolved fit unexpectedly lacks a path.")
    with job(directory, identity) as needed:
        if needed:
            stages = {
                stage: dict(component_id=row["component_id"])
                for stage, row in components.items()
            }
            atomic_json(directory / "path.json", dict(stages=stages))
    if completed(directory) is None:
        raise ValueError(f"Incomplete paired fit: {directory.resolve()}.")
    return directory


def source_view(
    plan: dict, item: dict, architecture: str,
) -> dict:
    """Build the existing scorer's view of one completed paired fit."""
    directory = fit_path(plan, item, architecture)
    if directory is None or completed(directory) is None:
        raise ValueError(
            f"Incomplete {architecture} fit for {item['name']}, "
            f"seed {item['seed']}."
        )
    path = json.loads((directory / "path.json").read_text())
    terminal = ARCHITECTURES[architecture][-1]
    component = (
        component_root(plan)
        / path["stages"][terminal]["component_id"]
    )
    training = json.loads(
        (component / "training.json").read_text()
    )
    return dict(
        count=1, spec=item["spec"], fit=directory,
        inventory={item["name"]: item["record"]},
        path=path,
        selection=dict(
            selected_budget=training["selected_update"],
            evaluation_set_id=fingerprint({
                item["name"]: item["record"]["validation"]
            }),
        ),
    )


def direct_forecast(source: dict, outer: dict) -> list[np.ndarray]:
    """Restore the independent RNN2 and forecast normalized neural Y."""
    spec = source["spec"]
    name = spec["sessions"][0]
    retained = sum(
        source["inventory"][name]["scales"]["Y"]["keep"]
    )
    dimensions = [(retained, outer["Z"].shape[-1])]
    model = rnn2_direct_model(dimensions, spec)
    component = (
        source["fit"].parent.parent / "source_components"
        / source["path"]["stages"]["neural_direct"]["component_id"]
    )
    restore_component(model, component)
    forecast, _ = stage_predictions(
        model, [recurrent_data(outer)]
    )
    expected = (len(HORIZONS), *outer["Y"].shape)
    if forecast[0].shape != expected or not np.isfinite(
        forecast[0]
    ).all():
        raise ValueError(
            f"Expected finite direct forecast {expected} for {name}, "
            f"got {forecast[0].shape}."
        )
    tf.keras.backend.clear_session()
    return forecast


def score_models(
    plan: dict, items: list[dict],
) -> tuple[list[dict], list[dict], list[dict]]:
    """Score complete pairs once on their identical outer windows."""
    neural_rows = []
    behavior_rows = []
    difference_rows = []
    for index in range(baseline.COMMON_SESSIONS):
        _, train, _, outer, record, _ = baseline.prepare(
            plan, index
        )
        common = {}
        name = items[index * len(SEEDS)]["name"]
        common[name] = dict(training=train, test=outer)
        for item in (
            row for row in items if row["index"] == index
        ):
            if item["record"] != record:
                raise ValueError(
                    f"Saved split changed for {name}, seed "
                    f"{item['seed']}."
                )
            two = source_view(plan, item, "two_stage")
            three = source_view(plan, item, "three_stage")
            if (
                two["path"]["stages"]["behavior_relevant"]
                != three["path"]["stages"]["behavior_relevant"]
            ):
                raise ValueError(
                    f"RNN1 checkpoint differs within pair {name}, "
                    f"seed {item['seed']}."
                )
            behavior = decoding.selected_forecasts(
                three, common
            )
            for row in decoding.session_scores(
                three, common, behavior
            ):
                behavior_rows.append(dict(
                    session=name, seed=item["seed"],
                    group=row["group"],
                    horizon_ms=(
                        row["horizon"]
                        * MILLISECONDS_PER_SECOND // SAMPLE_RATE
                    ),
                    cc=row["cc"], r2=row["r2"],
                    mse_ratio=row["mse_ratio"],
                ))
            pair_scores = {}
            for architecture, source in (
                ("two_stage", two),
                ("three_stage", three),
            ):
                forecasts = (
                    direct_forecast(source, outer)
                    if architecture == "two_stage"
                    else direct_report.full_neural_forecasts(
                        source, common
                    )
                )
                scores = direct_report.neural_scores(
                    source, common, forecasts
                )
                if len(scores) != len(HORIZONS):
                    raise ValueError(
                        f"Expected four neural horizons for {name}, "
                        f"seed {item['seed']}, {architecture}."
                    )
                pair_scores[architecture] = {
                    row["horizon"]: row for row in scores
                }
                for row in scores:
                    neural_rows.append(dict(
                        architecture=architecture,
                        session=name, seed=item["seed"],
                        horizon_ms=(
                            row["horizon"]
                            * MILLISECONDS_PER_SECOND // SAMPLE_RATE
                        ),
                        cc=row["cc"], r2=row["r2"],
                        mse=row["mse"],
                        dummy_mse=row["dummy_mse"],
                        mse_ratio=row["mse_ratio"],
                        selected_update=(
                            source["selection"]["selected_budget"]
                        ),
                        source_fit=source["fit"].name,
                    ))
            for horizon in HORIZONS:
                short = pair_scores["two_stage"][horizon]
                long = pair_scores["three_stage"][horizon]
                difference_rows.append(dict(
                    session=name, seed=item["seed"],
                    horizon_ms=horizon * MILLISECONDS_PER_SECOND // SAMPLE_RATE,
                    **{
                        f"three_minus_two_{metric}": (
                            long[metric] - short[metric]
                        ) for metric in METRICS
                    },
                ))
    expected_neural = (
        baseline.COMMON_SESSIONS * len(SEEDS)
        * len(HORIZONS) * len(ARCHITECTURES)
    )
    expected_behavior = (
        baseline.COMMON_SESSIONS * len(SEEDS)
        * len(HORIZONS) * len(decoding.COORDINATE_GROUPS)
    )
    if len(neural_rows) != expected_neural or (
        len(difference_rows) * 2 != expected_neural
        or len(behavior_rows) != expected_behavior
    ):
        raise ValueError(
            "Incomplete paired neural, behavior, or difference scores."
        )
    return neural_rows, difference_rows, behavior_rows


def summarize_scores(
    rows: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Average four sessions within each seed, then three seeds."""
    seed_rows = []
    summary = []
    horizons = [x * MILLISECONDS_PER_SECOND // SAMPLE_RATE for x in HORIZONS]
    for architecture in ARCHITECTURES:
        for seed in SEEDS:
            for horizon_ms in horizons:
                selected = [
                    row for row in rows
                    if row["architecture"] == architecture
                    and row["seed"] == seed
                    and row["horizon_ms"] == horizon_ms
                ]
                if len(selected) != baseline.COMMON_SESSIONS or (
                    len({row["session"] for row in selected})
                    != baseline.COMMON_SESSIONS
                ):
                    raise ValueError(
                        f"Incomplete session scores for "
                        f"{architecture}, seed {seed}, "
                        f"{horizon_ms} ms."
                    )
                seed_rows.append(dict(
                    architecture=architecture, seed=seed,
                    horizon_ms=horizon_ms,
                    sessions=baseline.COMMON_SESSIONS,
                    **{
                        metric: float(np.mean([
                            row[metric] for row in selected
                        ])) for metric in METRICS
                    },
                ))
        for horizon_ms in horizons:
            selected = [
                row for row in seed_rows
                if row["architecture"] == architecture
                and row["horizon_ms"] == horizon_ms
            ]
            if len(selected) != len(SEEDS):
                raise ValueError(
                    f"Incomplete seed means for {architecture}, "
                    f"{horizon_ms} ms."
                )
            summary.append(dict(
                architecture=architecture,
                horizon_ms=horizon_ms,
                sessions=baseline.COMMON_SESSIONS,
                seeds=len(SEEDS),
                **{
                    metric: float(np.mean([
                        row[metric] for row in selected
                    ])) for metric in METRICS
                },
            ))
    for row in seed_rows + summary:
        if not np.isfinite([row[x] for x in METRICS]).all():
            raise FloatingPointError(
                "Nonfinite paired neural score summary."
            )
    return seed_rows, summary


def session_standard_deviations(rows: list[dict]) -> dict:
    """Compute sample SD of session means after averaging seeds."""
    result = {}
    for architecture in ARCHITECTURES:
        for horizon in HORIZONS:
            horizon_ms = horizon * MILLISECONDS_PER_SECOND // SAMPLE_RATE
            session_means = {metric: [] for metric in METRICS}
            selected = [
                row for row in rows
                if row["architecture"] == architecture
                and row["horizon_ms"] == horizon_ms
            ]
            for session in {row["session"] for row in selected}:
                session_rows = [
                    row for row in selected if row["session"] == session
                ]
                if {row["seed"] for row in session_rows} != set(SEEDS):
                    raise ValueError(
                        f"Incomplete seeds for {session}, "
                        f"{architecture}, {horizon_ms} ms."
                    )
                for metric in METRICS:
                    session_means[metric].append(float(np.mean([
                        row[metric] for row in session_rows
                    ])))
            if len(session_means[METRICS[0]]) != baseline.COMMON_SESSIONS:
                raise ValueError(
                    f"Incomplete sessions for {architecture}, "
                    f"{horizon_ms} ms."
                )
            for metric, values in session_means.items():
                value = float(np.std(values, ddof=1))
                if not np.isfinite(value):
                    raise FloatingPointError(
                        f"Nonfinite session SD for {architecture}, "
                        f"{metric}, {horizon_ms} ms."
                    )
                result[architecture, horizon_ms, metric] = value
    return result


def plot_comparison(
    summary: list[dict], rows: list[dict], directory: Path,
) -> None:
    """Draw the two neural architectures at exact forecast horizons."""
    horizons = [x * MILLISECONDS_PER_SECOND // SAMPLE_RATE for x in HORIZONS]
    deviations = session_standard_deviations(rows)
    with plt.rc_context(READABLE_TEXT_STYLE):
        figure, axes = plt.subplots(
            1, len(METRICS), figsize=FIGURE_SIZE
        )
        for axis, metric in zip(axes, METRICS):
            values = []
            for architecture in ARCHITECTURES:
                selected = sorted(
                    (
                        row for row in summary
                        if row["architecture"] == architecture
                    ),
                    key=lambda row: row["horizon_ms"],
                )
                if [row["horizon_ms"] for row in selected] != horizons:
                    raise ValueError(
                        f"Incomplete {architecture} {metric} curve."
                    )
                y = [row[metric] for row in selected]
                errors = [
                    deviations[architecture, horizon_ms, metric]
                    for horizon_ms in horizons
                ]
                values.extend(np.asarray(y) - np.asarray(errors))
                values.extend(np.asarray(y) + np.asarray(errors))
                plot_line(
                    x=horizons, y=y, ax=axis,
                    color=COLORS[architecture],
                    linestyle=STYLES[architecture],
                    marker=MARKERS[architecture],
                    label=architecture.replace("_", " ").title(),
                )
                axis.errorbar(
                    horizons, y, yerr=errors,
                    color=COLORS[architecture], fmt="none",
                    capsize=4, linewidth=1.6,
                )
            if not np.isfinite(values).all():
                raise FloatingPointError(
                    f"Nonfinite {metric} figure values."
                )
            span = max(values) - min(values)
            padding = max(
                AXIS_MINIMUM_PADDING,
                AXIS_PADDING_FRACTION * span,
            )
            axis.set_ylim(
                min(values) - padding, max(values) + padding
            )
            axis.set_xlim(*HORIZON_AXIS_LIMITS)
            axis.set_xticks(horizons)
            axis.set_xlabel("Forecast horizon (ms)")
            axis.set_ylabel(METRIC_LABELS[metric])
            axis.grid(alpha=0.25)
            legend = axis.get_legend()
            if legend is not None:
                legend.remove()
        figure.suptitle(
            textwrap.fill(
                "Neural prediction: two-stage versus "
                "three-stage BRAID",
                width=55,
            ),
            y=0.98,
        )
        figure.text(
            0.5, 0.12,
            "Error bars: ±1 SD across four sessions "
            "(each averaged over three seeds)",
            ha="center", va="center", fontsize=14,
        )
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(
            handles, labels, loc="lower center",
            bbox_to_anchor=(0.5, 0.01), ncol=2,
            frameon=False,
        )
        figure.subplots_adjust(
            left=0.08, right=0.98, top=0.77,
            bottom=0.25, wspace=0.30,
        )
        figure.savefig(
            directory / "neural_comparison.png",
            dpi=DPI, bbox_inches="tight",
        )
        plt.close(figure)


def publish(plan_path: Path) -> Path:
    """Publish only the completed two- versus three-stage comparison."""
    plan = json.loads(plan_path.read_text())
    items = work_items(plan)
    markers = {}
    for item in items:
        for architecture in ARCHITECTURES:
            directory = fit_path(plan, item, architecture)
            marker = (
                completed(directory)
                if directory is not None else None
            )
            if marker is None:
                raise ValueError(
                    f"Incomplete {architecture} fit for "
                    f"{item['name']}, seed {item['seed']}."
                )
            key = (
                f"{item['name']}:{item['seed']}:{architecture}"
            )
            markers[key] = marker["checksums"]
    identity = dict(
        group=PROTOCOL, fit_checksums=markers, guide=GUIDE,
    )
    root = analysis_directory(
        plan, "stage_ablation", plan_path.parent.name
    )
    destination = root / "reports" / fingerprint(identity)
    with job(destination, identity) as needed:
        if needed:
            neural, differences, behavior = score_models(
                plan, items
            )
            by_seed, summary = summarize_scores(neural)
            tables = destination / "tables"
            figures = destination / "figures"
            tables.mkdir()
            figures.mkdir()
            write_table(
                tables / "neural_scores.csv", neural
            )
            write_table(
                tables / "paired_differences.csv",
                differences,
            )
            write_table(
                tables / "behavior_scores.csv", behavior
            )
            write_table(
                tables / "seed_summary.csv", by_seed
            )
            write_table(
                tables / "summary.csv", summary
            )
            (destination / "guide.md").write_text(GUIDE)
            plot_comparison(summary, neural, figures)
    if completed(destination) is None:
        raise ValueError(
            f"Incomplete paired report: {destination.resolve()}."
        )
    atomic_json(root / "latest_report.json", dict(
        snapshot=str(destination.resolve()),
        figures=str((destination / "figures").resolve()),
        identity=fingerprint(identity),
    ))
    print(
        f"Paired comparison figures: "
        f"{(destination / 'figures').resolve()}"
    )
    return destination


def audit_updates(
    model: object, training: list[dict],
    validation: list[dict], settings: dict,
    seed: int, microbatch: int,
) -> None:
    """Run a few real gradients and one finite validation pass in memory."""
    optimizers = optimizer_groups(model, settings)
    rng = np.random.default_rng(seed)
    functions = [gradient_function(model, 0)]
    for _ in range(AUDIT_UPDATES):
        balanced_update(
            model, training, optimizers, rng, SESSION_BATCH,
            microbatch, LOCAL_DEVICE, functions,
        )
    losses, _ = evaluate_sessions(model, validation)
    if len(losses) != 1 or not np.isfinite(losses).all():
        raise FloatingPointError(
            "Paired audit produced invalid inner-validation loss."
        )


def audit(plan: dict, microbatch: int) -> None:
    """Exercise all paired stage objectives without writing fit artifacts."""
    if not 1 <= microbatch <= SESSION_BATCH:
        raise ValueError(
            f"Expected microbatch in 1..{SESSION_BATCH}; "
            f"got {microbatch}."
        )
    item = work_items(plan)[0]
    spec = item["spec"]
    _, train, inner, _, record, _ = baseline.prepare(
        plan, item["index"]
    )
    if record != item["record"]:
        raise ValueError("Paired audit split differs from saved reference.")
    dimensions = [(train["Y"].shape[-1], train["Z"].shape[-1])]
    neural = [recurrent_data(row) for row in (train, inner)]
    behavior = [recurrent_data(row, "Z") for row in (train, inner)]
    settings = spec["source_training"]
    main = rnn1_model(dimensions, spec, LOCAL_DEVICE)
    audit_updates(
        main, [behavior[0]], [behavior[1]], settings,
        spec["seed"] + SEED_OFFSETS["behavior_relevant"],
        microbatch,
    )
    direct = rnn2_direct_model(dimensions, spec, LOCAL_DEVICE)
    audit_updates(
        direct, [neural[0]], [neural[1]], settings,
        spec["seed"] + SEED_OFFSETS["neural_direct"],
        microbatch,
    )
    latent = [
        stage_predictions(main, [row])[1]
        for row in behavior
    ]
    readout = neural_readout_model(
        dimensions, spec, LOCAL_DEVICE
    )
    readout_train = readout_data([neural[0]], latent[0])
    readout_inner = readout_data([neural[1]], latent[1])
    audit_updates(
        readout, readout_train, readout_inner, settings,
        spec["seed"] + SEED_OFFSETS["neural_readout"],
        microbatch,
    )
    residual = rnn2_model(dimensions, spec, LOCAL_DEVICE)
    audit_updates(
        residual,
        residual_data(main, readout, [neural[0]]),
        residual_data(main, readout, [neural[1]]),
        settings, spec["seed"] + SEED_OFFSETS["residual"],
        microbatch,
    )
    print(
        "Paired in-memory audit passed all four stage objectives."
    )


def task_key(item: dict, stage: str) -> str:
    """Name one session/seed/component lifecycle entry."""
    return (
        f"{item['name']}:{item['seed']}:{stage}"
    )


def lifecycle_status(
    launch_dir: Path, states: dict[str, str],
    fits: int, phase: str,
) -> None:
    """Publish compact component and model-fit completion counts."""
    counts = {
        state: sum(value == state for value in states.values())
        for state in (
            "pending", "running", "complete",
            "failed", "blocked",
        )
    }
    atomic_json(launch_dir / "status.json", dict(
        phase=phase, components=counts,
        component_total=len(states),
        fits_complete=fits,
        fit_total=(
            baseline.COMMON_SESSIONS * len(SEEDS)
            * len(ARCHITECTURES)
        ),
        updated_at_utc=datetime.now(
            timezone.utc
        ).isoformat(),
    ))


def start_worker(
    plan_path: Path, launch_dir: Path,
    item: dict, stage: str, device: dict,
    microbatch: int,
) -> subprocess.Popen:
    """Launch one lower-priority GPU component worker."""
    env = dict(os.environ)
    env.update(thread_environment(WORKER_CPU_THREADS, 1))
    env["CUDA_VISIBLE_DEVICES"] = device["uuid"]
    env["TF_USE_LEGACY_KERAS"] = "1"
    env["PYTHONPATH"] = str(
        Path(__file__).resolve().parents[3]
    )
    env["MPLCONFIGDIR"] = "/tmp/braid_paired_mpl"
    label = (
        f"{item['index']}_{item['seed']}_{stage}"
    )
    log = launch_dir / "jobs" / label / "worker.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-m",
        "experiments.multisession.behavior_compute_scaling."
        "paired_stages",
        "--stage", "worker", "--plan", str(plan_path),
        "--launch-dir", str(launch_dir),
        "--session-index", str(item["index"]),
        "--seed", str(item["seed"]),
        "--component", stage,
        "--microbatch", str(microbatch),
    ]
    with log.open("a") as stream:
        return subprocess.Popen(
            command, env=env, stdout=stream,
            stderr=subprocess.STDOUT,
            preexec_fn=lambda: os.nice(LOW_CPU_PRIORITY),
        )


def count_fits(plan: dict, items: list[dict]) -> int:
    """Count and finalize complete paired model manifests."""
    count = 0
    for item in items:
        for architecture in ARCHITECTURES:
            directory = finalize_fit(
                plan, item, architecture
            )
            if directory is not None:
                count += 1
    return count


def coordinator(
    plan_path: Path, launch_dir: Path,
) -> None:
    """Run ready stages across free GPUs and report after all fits."""
    plan = json.loads(plan_path.read_text())
    items = work_items(plan)
    launch = json.loads(
        (launch_dir / "launch.json").read_text()
    )
    devices = launch["devices"]
    microbatch = launch["microbatch"]
    states = {
        task_key(item, stage): "pending"
        for item in items for stage in STAGES
    }
    active = {}
    while True:
        for key, (process, device, target) in list(
            active.items()
        ):
            code = process.poll()
            if code is None:
                continue
            del active[key]
            if code == 0 and completed(target) is not None:
                states[key] = "complete"
            else:
                states[key] = "failed"
                failure = dict(
                    task=key, exit_code=code,
                    component=str(target.resolve()),
                    timestamp_utc=datetime.now(
                        timezone.utc
                    ).isoformat(),
                )
                with (launch_dir / "failures.jsonl").open("a") as f:
                    f.write(json.dumps(failure) + "\n")
        busy = {device["uuid"] for _, device, _ in active.values()}
        free = [
            device for device in devices
            if device["uuid"] not in busy
        ]
        for item in items:
            for stage in STAGES:
                key = task_key(item, stage)
                if states[key] != "pending":
                    continue
                target = component_path(plan, item, stage)
                if target is None:
                    continue
                if completed(target) is not None:
                    states[key] = "complete"
                    continue
                if not free:
                    continue
                device = free.pop(0)
                process = start_worker(
                    plan_path, launch_dir, item,
                    stage, device, microbatch,
                )
                active[key] = (process, device, target)
                states[key] = "running"
        fits = count_fits(plan, items)
        if all(
            state == "complete"
            for state in states.values()
        ):
            if fits != (
                baseline.COMMON_SESSIONS * len(SEEDS)
                * len(ARCHITECTURES)
            ):
                raise ValueError(
                    "All components finished, but paired fit "
                    "manifests are incomplete."
                )
            lifecycle_status(
                launch_dir, states, fits, "reporting"
            )
            try:
                report = publish(plan_path)
            except Exception as error:
                atomic_json(
                    launch_dir / "report_error.json",
                    dict(
                        error=(
                            f"{type(error).__name__}: {error}"
                        )
                    ),
                )
                lifecycle_status(
                    launch_dir, states, fits, "report_failed"
                )
                return
            atomic_json(
                launch_dir / "report.json",
                dict(snapshot=str(report.resolve())),
            )
            lifecycle_status(
                launch_dir, states, fits, "complete"
            )
            return
        if not active:
            ready = any(
                states[task_key(item, stage)] == "pending"
                and component_path(
                    plan, item, stage
                ) is not None
                for item in items for stage in STAGES
            )
            if not ready:
                for item in items:
                    for stage in STAGES:
                        key = task_key(item, stage)
                        if states[key] == "pending":
                            states[key] = "blocked"
                lifecycle_status(
                    launch_dir, states, fits, "failed"
                )
                return
        lifecycle_status(launch_dir, states, fits, "active")
        time.sleep(POLL_SECONDS)


def launch(plan_path: Path, microbatch: int) -> Path:
    """Audit in memory, then detach independent component workers."""
    plan = json.loads(plan_path.read_text())
    work_items(plan)
    if not 1 <= microbatch <= SESSION_BATCH:
        raise ValueError(
            f"Expected microbatch in 1..{SESSION_BATCH}; "
            f"got {microbatch}."
        )
    visibility = os.environ.get("CUDA_VISIBLE_DEVICES")
    count = (
        len([part for part in visibility.split(",") if part.strip()])
        if visibility is not None else EXPECTED_GPU_COUNT
    )
    devices = select_gpus("auto", count)
    selected = [
        device for device in devices
        if device["free_memory_mib"] >= MINIMUM_FREE_MIB
        and device["utilization_percent"]
        <= MAXIMUM_GPU_UTILIZATION
    ]
    if not selected:
        raise RuntimeError(
            "No idle GPU has sufficient free memory "
            "for paired stages."
        )
    audit_env = dict(os.environ)
    audit_env.update(thread_environment(WORKER_CPU_THREADS, 1))
    audit_env["CUDA_VISIBLE_DEVICES"] = selected[0]["uuid"]
    audit_env["TF_USE_LEGACY_KERAS"] = "1"
    audit_env["PYTHONPATH"] = str(
        Path(__file__).resolve().parents[3]
    )
    audit_env["MPLCONFIGDIR"] = "/tmp/braid_paired_mpl"
    subprocess.run(
        [
            sys.executable, "-m",
            "experiments.multisession.behavior_compute_scaling."
            "paired_stages",
            "--stage", "audit", "--plan", str(plan_path),
            "--microbatch", str(microbatch),
        ],
        env=audit_env, check=True,
    )
    root = (
        Path(__file__).resolve().parents[4]
        / "logs" / "stage_ablation"
    )
    stamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")
    launch_dir = (
        root / f"stage_ablation_{stamp}_{os.getpid()}"
    )
    launch_dir.mkdir(parents=True)
    atomic_json(launch_dir / "launch.json", dict(
        plan=str(plan_path.resolve()),
        devices=selected, microbatch=microbatch,
        seeds=list(SEEDS), protocol=PROTOCOL,
    ))
    env = dict(os.environ)
    env["PYTHONPATH"] = str(
        Path(__file__).resolve().parents[3]
    )
    env["TF_USE_LEGACY_KERAS"] = "1"
    env["MPLCONFIGDIR"] = "/tmp/braid_paired_mpl"
    command = [
        sys.executable, "-m",
        "experiments.multisession.behavior_compute_scaling."
        "paired_stages",
        "--stage", "coordinator",
        "--plan", str(plan_path),
        "--launch-dir", str(launch_dir),
    ]
    with (launch_dir / "coordinator.log").open("a") as stream:
        process = subprocess.Popen(
            command, env=env, stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            preexec_fn=lambda: os.nice(LOW_CPU_PRIORITY),
        )
    atomic_json(
        launch_dir / "pid.json",
        dict(pid=process.pid),
    )
    print(
        f"Detached stage ablation campaign: "
        f"{launch_dir.resolve()}"
    )
    return launch_dir


def main() -> None:
    """Dispatch audit, launch, component, coordinator, or report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", required=True,
        choices=(
            "audit", "launch", "worker",
            "coordinator", "report",
        ),
    )
    parser.add_argument(
        "--plan", type=Path, required=True
    )
    parser.add_argument("--launch-dir", type=Path)
    parser.add_argument("--session-index", type=int)
    parser.add_argument(
        "--seed", type=int, choices=SEEDS
    )
    parser.add_argument(
        "--component", choices=STAGES
    )
    parser.add_argument(
        "--microbatch", type=int, default=4
    )
    args = parser.parse_args()
    plan_path = args.plan.resolve()
    if args.stage == "launch":
        launch(plan_path, args.microbatch)
        return
    if args.stage == "report":
        publish(plan_path)
        return
    plan = json.loads(plan_path.read_text())
    if args.stage == "coordinator":
        if args.launch_dir is None:
            raise ValueError(
                "Coordinator requires --launch-dir."
            )
        coordinator(
            plan_path, args.launch_dir.resolve()
        )
        return
    visible = tf.config.list_physical_devices("GPU")
    if not visible:
        raise RuntimeError(
            "Audit and component workers require one GPU."
        )
    tf.config.set_logical_device_configuration(
        visible[0], [
            tf.config.LogicalDeviceConfiguration(
                memory_limit=GPU_MEMORY_LIMIT_MIB
            )
        ],
    )
    if args.stage == "audit":
        audit(plan, args.microbatch)
        return
    if (
        args.session_index is None
        or args.seed is None
        or args.component is None
        or args.launch_dir is None
    ):
        raise ValueError(
            "Worker requires launch directory, session index, "
            "seed, and component."
        )
    matches = [
        item for item in work_items(plan)
        if item["index"] == args.session_index
        and item["seed"] == args.seed
    ]
    if len(matches) != 1:
        raise ValueError(
            "Expected one session/seed component job."
        )
    path = fit_stage(
        plan, matches[0], args.component,
        args.microbatch,
    )
    print(f"Completed component: {path.resolve()}")


if __name__ == "__main__":
    main()
