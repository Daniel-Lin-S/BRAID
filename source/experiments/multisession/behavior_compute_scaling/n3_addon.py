"""Fit and report the input-only behavioral post-model on saved sources.

Usage: python -m experiments.multisession.behavior_compute_scaling.n3_addon
  --stage audit|launch|worker|coordinator|report --plan ABSOLUTE_PLAN
  --source-report ABSOLUTE_REPORT [--count 4|8|16|33]

Inputs are a completed source plan/report, selected immutable RNN1 checkpoints,
training-only source windows, and task-target U. Each source count gets one
checksum-validated n3 component in the common source_components store. A full
report under the existing analysis root contains per-session and aggregate
behavior metrics, component references, and two groups of figures. The launch
log contains only status and worker output; it never stores model artifacts.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import textwrap
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from BRAID.multisession import HORIZONS, MixedForecaster
from experiments.execution.runtime import select_gpus, thread_environment
from experiments.multisession.behavior_compute_scaling.ancestry import (
    restore_component,
)
from experiments.multisession.behavior_compute_scaling import (
    behavior_decoding_diagnostics as decoding,
)
from experiments.multisession.behavior_compute_scaling.pipeline import (
    resolved_schedule,
)
from experiments.multisession.behavior_compute_scaling.plan import COUNTS
from experiments.multisession.behavior_compute_scaling.training import (
    balanced_update, evaluate_sessions, fit_segment, gradient_function,
    optimizer_groups,
)
from experiments.multisession.data import session_windows, source_features
from experiments.multisession.log_paths import source_session_log_path
from experiments.multisession.plan import analysis_directory
from experiments.multisession.reports import write_table
from experiments.multisession.runner import dataset
from experiments.multisession.store import completed, fit_directory, job
from experiments.multisession.training import (
    forward_inputs, stage_predictions,
)
from experiments.storage.cache import atomic_json, file_digest, fingerprint
from experiments.visualization.multisession import (
    READABLE_TEXT_STYLE, plot_line,
)

N3_SIZE = 1
N3_SEED_OFFSET = 5
N3_DECODER_WIDTH = 64
AUDIT_UPDATES = 3
AUDIT_COUNT = 4
GPUS_PER_WORKER = 1
GPU_INVENTORY_COUNT = 8
WORKER_CPU_THREADS = 2
MINIMUM_FREE_MIB = 4096
POLL_SECONDS = 10
LOW_CPU_PRIORITY = 10
PROTOCOL = "frozen_rnn_input_only_n3_v1"
VARIANTS = ("RNN1", "RNN1 + n3")
VARIANT_STYLES = {"RNN1": "--", "RNN1 + n3": "-"}
GROUP_MARKERS = {"position": "o", "velocity": "s"}
GUIDE = """# Behavioral source decoding with BRAID n3

Only the existing transfer-selected source checkpoint per source count is
used. Its RNN1 and RNN2 weights remain frozen. The one-state, session-private
post-model n3 receives task target U, predicts a behavioral correction, and
is trained under the source fixed-stage exposure cap and inner-validation
selection. RNN2 predicts neural activity and does not supply a behavioral
correction in this campaign.

For each horizon h, RNN1's generated-target prediction is inverted to the
training-normalized recorded-behavior scale, P_h. The n3 training target is
recorded Z on that same scale, and its additive prior is P_h. Thus the n3
objective is sum_h mean[(Z_{t+h} - P_{t+h} - n3_h(U))^2]. The final forecast
is P_h + n3_h(U). No recorded Z enters n3 at inference.

The four common source recordings have the same guarded 87-window outer
chronological test set at every source count. Inner validation selects n3
checkpoints; the outer test also selected the upstream transfer checkpoints.
These outer-test scores are therefore descriptive. Each plotted source-count
point is an equal mean of four session scores. CC and R2 are per-coordinate
means. MSE is per-coordinate mean squared error; MSE/dummy MSE divides by
the same session and horizon's training-target-mean predictor error. Values
above one are valid. The comparison uses identical windows and source RNN1
checkpoints with and without n3; no target adaptation is performed.
"""


def source_entry(plan: dict, source_report: Path, count: int) -> dict:
    """Return one checksum-validated selected source path."""
    entries = decoding.checked_sources(plan, source_report)
    matches = [entry for entry in entries if entry["count"] == count]
    if len(matches) != 1:
        raise ValueError(f"Expected one selected source with {count} sessions.")
    return matches[0]


def component_root(source: dict) -> Path:
    """Return the common source component store."""
    return source["fit"].parent.parent / "source_components"


def n3_settings(source: dict) -> dict:
    """Resolve the same fixed-stage exposure protocol as preprocessing."""
    settings = resolved_schedule(
        source["spec"]["source_training"], source["count"]
    )
    if settings["fixed_update_cap"] <= 0:
        raise ValueError("n3 fixed-stage update cap must be positive.")
    return settings


def n3_identity(source: dict) -> dict:
    """Bind n3 to its scientific inputs and selected RNN1 checkpoint."""
    component = source["path"]["stages"]["behavior_relevant"][
        "component_id"
    ]
    marker = completed(component_root(source) / component)
    if marker is None:
        raise ValueError(f"Incomplete selected RNN1 component: {component}.")
    return dict(
        protocol=PROTOCOL, stage="behavior_post_n3",
        source_fit=source["fit"].name,
        source_count=source["count"],
        sessions=source["spec"]["sessions"],
        data_identity=fingerprint(source["inventory"]),
        evaluation_set_id=source["selection"]["evaluation_set_id"],
        selected_budget=source["selection"]["selected_budget"],
        upstream_checkpoint=component,
        upstream_checksums=marker["checksums"],
        generated_scales_checksum=file_digest(
            source["fit"] / "preprocessing.json"
        ),
        latent=N3_SIZE, decoder_width=N3_DECODER_WIDTH,
        sharing="none", input="U", output="recorded_Z_residual",
        seed=source["spec"]["seed"] + N3_SEED_OFFSET,
        horizons=HORIZONS, source_training=n3_settings(source),
    )


def n3_directory(source: dict) -> Path:
    """Compute the fixed-stage component path without starting a fit."""
    identity = dict(
        n3_identity(source), start=0,
        stop=n3_settings(source)["fixed_update_cap"],
        branch=False, parent=None, parent_checksums=None,
        fixed_stage=True,
    )
    return component_root(source) / fingerprint(identity)


def n3_model(
    source: dict, devices: tuple[str, str] | None = None,
) -> MixedForecaster:
    """Build native-style input-only one-state session-private recurrences."""
    count = source["count"]
    return MixedForecaster(
        [(0, len(decoding.COORDINATE_NAMES))] * count,
        N3_SIZE, "none", source["spec"]["seed"] + N3_SEED_OFFSET,
        decoder_width=N3_DECODER_WIDTH,
        input_dimensions=[2] * count,
        forward_dimensions=2, session_devices=devices,
    )


def source_windows(plan: dict, source: dict) -> tuple[list, list]:
    """Rebuild and verify every source recording's saved split inventory."""
    adapter = dataset(plan["settings"])
    adapter.mode = "off"
    train, inner = [], []
    for name in source["spec"]["sessions"]:
        features = source_features(
            adapter.load(name), source["spec"]["data"],
            source["spec"]["source_training"][
                "inner_validation_fraction"
            ],
        )
        training, validation, _, inventory = session_windows(
            features, True, include_test=True,
        )
        if inventory != source["inventory"][name]:
            raise ValueError(f"Saved source windows changed for {name}.")
        if not len(training["Z"]) or not len(validation["Z"]):
            raise ValueError(f"Empty n3 training or inner validation: {name}.")
        train.append(training)
        inner.append(validation)
    return train, inner


def post_data(partitions: list[dict], forecasts: list[np.ndarray]) -> list:
    """Build U-only inputs and additive RNN1 prior for n3 training."""
    if len(partitions) != len(forecasts):
        raise ValueError("n3 partitions and RNN1 forecasts differ in count.")
    rows = []
    for partition, forecast in zip(partitions, forecasts):
        target = partition["Z"]
        if forecast.shape != (len(HORIZONS), *target.shape):
            raise ValueError("n3 prior shape differs from recorded behavior.")
        if not np.isfinite(target).all() or not np.isfinite(forecast).all():
            raise ValueError("n3 target or RNN1 prior is nonfinite.")
        rows.append(dict(
            I=partition["U"], F=forward_inputs(partition["U"]),
            T=target, P=forecast,
        ))
    return rows


def fit_one(
    plan: dict, source_report: Path, count: int,
    devices: tuple[str, str], microbatch: int,
) -> Path:
    """Fit one immutable selected-path n3 component, or reuse completion."""
    source = source_entry(plan, source_report, count)
    directory = n3_directory(source)
    if completed(directory) is not None:
        return directory
    train, inner = source_windows(plan, source)
    priors_train = decoding.recorded_forecasts(source, train, devices)
    priors_inner = decoding.recorded_forecasts(source, inner, devices)
    model = n3_model(source, devices)
    settings = n3_settings(source)
    result, _ = fit_segment(
        model, post_data(train, priors_train), component_root(source),
        n3_identity(source), settings,
        source["spec"]["seed"] + N3_SEED_OFFSET,
        0, settings["fixed_update_cap"], None,
        devices, microbatch, validation=post_data(inner, priors_inner),
    )
    if result != directory or completed(result) is None:
        raise ValueError("n3 completion path or checksum differs.")
    return result


def audit_one(
    plan: dict, source_report: Path, devices: tuple[str, str],
    microbatch: int,
) -> dict:
    """Time three n3 updates and one inner check without saving artifacts."""
    source = source_entry(plan, source_report, AUDIT_COUNT)
    started = time.monotonic()
    train, inner = source_windows(plan, source)
    priors_train = decoding.recorded_forecasts(source, train, devices)
    priors_inner = decoding.recorded_forecasts(source, inner, devices)
    model = n3_model(source, devices)
    data = post_data(train, priors_train)
    validation = post_data(inner, priors_inner)
    prepared = time.monotonic()
    optimizers = optimizer_groups(model, n3_settings(source))
    functions = [
        gradient_function(model, index) for index in range(source["count"])
    ]
    rng = np.random.default_rng(source["spec"]["seed"] + N3_SEED_OFFSET)
    for _ in range(AUDIT_UPDATES):
        balanced_update(
            model, data, optimizers, rng,
            n3_settings(source)["session_batch_sizes"][str(AUDIT_COUNT)],
            microbatch, devices, functions,
        )
    checked = time.monotonic()
    losses, _ = evaluate_sessions(model, validation)
    if not np.isfinite(losses).all():
        raise ValueError("n3 audit inner-validation loss is nonfinite.")
    return dict(
        preparation_seconds=prepared - started,
        update_seconds=(checked - prepared) / AUDIT_UPDATES,
        validation_seconds=time.monotonic() - checked,
        audit_updates=AUDIT_UPDATES, validation_losses=losses,
    )


def corrected_forecasts(
    source: dict, windows: dict, priors: list[np.ndarray],
) -> list[np.ndarray]:
    """Apply saved n3 correction using only U at inference."""
    model = n3_model(source)
    restore_component(model, n3_directory(source))
    names = source["spec"]["sessions"][:decoding.COMMON_SESSION_COUNT]
    data = post_data([windows[name]["test"] for name in names], priors)
    corrections, _ = stage_predictions(model, data)
    result = [base + residual for base, residual in zip(priors, corrections)]
    if any(not np.isfinite(row).all() for row in result):
        raise ValueError("Nonfinite n3-corrected behavioral prediction.")
    tf.keras.backend.clear_session()
    return result


def comparison_figures(summary: list[dict], directory: Path) -> None:
    """Plot paired RNN1 versus RNN1+n3 session-count curves."""
    with plt.rc_context(READABLE_TEXT_STYLE):
        for metric, label in decoding.METRIC_LABELS.items():
            figure, axes = plt.subplots(2, 2, figsize=decoding.FIGURE_SIZE)
            for axis, horizon in zip(axes.flat, HORIZONS):
                values = []
                for variant in VARIANTS:
                    for group in decoding.COORDINATE_GROUPS:
                        selected = sorted(
                            (row for row in summary
                             if row["variant"] == variant
                             and row["group"] == group
                             and row["horizon"] == horizon),
                            key=lambda row: row["source_count"],
                        )
                        counts = [row["source_count"] for row in selected]
                        if counts != list(COUNTS):
                            raise ValueError("Incomplete paired n3 comparison.")
                        scores = [row[metric] for row in selected]
                        values.extend(scores)
                        plot_line(
                            x=COUNTS, y=scores, ax=axis,
                            color=decoding.GROUP_COLORS[group],
                            linestyle=VARIANT_STYLES[variant],
                            marker=GROUP_MARKERS[group],
                            label=f"{group.capitalize()}: {variant}",
                        )
                axis.set_xscale("log", base=2)
                axis.set_xticks(COUNTS, labels=[str(x) for x in COUNTS])
                axis.set_xlim(min(COUNTS) / 1.08, max(COUNTS) * 1.08)
                axis.set_xlabel("Source sessions")
                axis.set_ylabel(label)
                axis.set_title(f"{horizon * 50} ms", pad=10)
                decoding.focus_axis(axis, values, metric)
            handles, labels = axes.flat[0].get_legend_handles_labels()
            figure.legend(
                handles, labels, loc="upper center",
                bbox_to_anchor=(0.5, 0.90), ncol=2, frameon=False,
            )
            figure.suptitle(
                textwrap.fill(f"Behavior: n3 comparison, {label}", 55),
                fontsize=23, y=0.99,
            )
            figure.subplots_adjust(
                left=0.10, right=0.97, top=0.77, bottom=0.10,
                hspace=0.47, wspace=0.31,
            )
            figure.savefig(
                directory / f"n3_comparison_{metric}.png",
                dpi=decoding.DPI, bbox_inches="tight",
            )
            plt.close(figure)


def publish(plan_path: Path, source_report: Path) -> Path:
    """Publish two behavior-only groups when all four n3 fits complete."""
    plan = json.loads(plan_path.read_text())
    sources = decoding.checked_sources(plan, source_report)
    components = {}
    for source in sources:
        directory = n3_directory(source)
        marker = completed(directory)
        if marker is None:
            raise ValueError(
                f"Cannot report incomplete n3 fit: {directory.resolve()}."
            )
        components[str(source["count"])] = dict(
            component=directory.name, checksums=marker["checksums"],
        )
    report_marker = completed(source_report)
    identity = dict(
        group="behavior_n3_selected_source_scaling", protocol=PROTOCOL,
        source_plan_checksum=file_digest(plan_path),
        source_report_checksums=report_marker["checksums"],
        components=components, guide=GUIDE,
        figure_style=READABLE_TEXT_STYLE,
    )
    root = analysis_directory(
        plan, "multi_session_behavior_compute_scaling",
        plan_path.parent.name,
    )
    destination = root / "reports" / fingerprint(identity)
    with job(destination, identity) as needed:
        if needed:
            windows = decoding.fixed_windows(plan, sources)
            scores = []
            for source in sources:
                priors = decoding.selected_forecasts(source, windows)
                forecasts = corrected_forecasts(source, windows, priors)
                for variant, values in zip(VARIANTS, (priors, forecasts)):
                    for row in decoding.session_scores(source, windows, values):
                        scores.append(dict(row, variant=variant))
            summary = []
            for variant in VARIANTS:
                subset = [row for row in scores if row["variant"] == variant]
                summary.extend(
                    dict(row, variant=variant)
                    for row in decoding.summarize(subset)
                )
            figures_path = destination / "figures"
            figures_path.mkdir()
            write_table(destination / "session_scores.csv", scores)
            write_table(destination / "summary.csv", summary)
            atomic_json(destination / "n3_components.json", components)
            (destination / "guide.md").write_text(GUIDE)
            corrected = [
                row for row in summary if row["variant"] == "RNN1 + n3"
            ]
            decoding.figures(
                corrected, figures_path,
                subtitle="Transfer-selected RNN1 + U-only n3; common test set",
                title_prefix="Behavior decoding with n3",
                compact=True,
            )
            comparison_figures(summary, figures_path)
    pointer = root / "latest_behavior_n3.json"
    atomic_json(pointer, dict(
        snapshot=str(destination.resolve()), identity=fingerprint(identity),
    ))
    print(
        "Saved n3 behavioral figures: "
        f"{(destination / 'figures').resolve()}"
    )
    return destination


def coordinator(
    plan_path: Path, source_report: Path, launch: Path,
    devices: list[dict], microbatch: int,
) -> None:
    """Run four independent source-count workers and then publish figures."""
    processes = {}
    status = {str(count): "pending" for count in COUNTS}
    atomic_json(launch / "status.json", dict(state="active", members=status))
    for index, count in enumerate(COUNTS):
        assigned = devices[
            index * GPUS_PER_WORKER:(index + 1) * GPUS_PER_WORKER
        ]
        if len(assigned) != GPUS_PER_WORKER or any(
            device["free_memory_mib"] < MINIMUM_FREE_MIB
            for device in assigned
        ):
            raise RuntimeError(f"Insufficient free GPU memory for {count}.")
        env = dict(os.environ)
        env.update(thread_environment(WORKER_CPU_THREADS, 1))
        env["CUDA_VISIBLE_DEVICES"] = ",".join(
            device["uuid"] for device in assigned
        )
        env["TF_USE_LEGACY_KERAS"] = "1"
        env["MPLCONFIGDIR"] = "/tmp/braid_n3_matplotlib"
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[3])
        command = [
            sys.executable, "-m",
            "experiments.multisession.behavior_compute_scaling.n3_addon",
            "--stage", "worker", "--plan", str(plan_path),
            "--source-report", str(source_report),
            "--count", str(count), "--microbatch", str(microbatch),
        ]
        source = source_entry(plan, source_report, count)
        log = source_session_log_path(
            launch, count, "all_sessions",
            n3_directory(source).name, N3_SIZE,
        )
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a") as stream:
            process = subprocess.Popen(
                command, env=env, stdout=stream,
                stderr=subprocess.STDOUT,
                preexec_fn=lambda: os.nice(LOW_CPU_PRIORITY),
            )
        processes[count] = process
        status[str(count)] = "running"
        atomic_json(launch / "status.json", dict(
            state="active", members=status,
        ))
    while processes:
        for count, process in list(processes.items()):
            code = process.poll()
            if code is None:
                continue
            del processes[count]
            status[str(count)] = "complete" if code == 0 else "failed"
            atomic_json(launch / "status.json", dict(
                state="active", members=status,
            ))
        if processes:
            time.sleep(POLL_SECONDS)
    report = None
    if all(value == "complete" for value in status.values()):
        try:
            report = publish(plan_path, source_report)
        except Exception as error:
            atomic_json(launch / "status.json", dict(
                state="report_failed", members=status,
                error=f"{type(error).__name__}: {error}",
            ))
            raise
        state = "complete"
    else:
        state = "failed"
    atomic_json(launch / "status.json", dict(
        state=state, members=status,
        report=str(report.resolve()) if report else None,
    ))


def launch(
    plan_path: Path, source_report: Path, microbatch: int,
) -> Path:
    """Detach a four-member n3 queue after GPU inventory validation."""
    plan = json.loads(plan_path.read_text())
    decoding.checked_sources(plan, source_report)
    if microbatch <= 0:
        raise ValueError("n3 microbatch must be positive.")
    inventory = select_gpus("auto", GPU_INVENTORY_COUNT)
    devices = sorted(
        inventory,
        key=lambda row: (
            row["utilization_percent"], -row["free_memory_mib"]
        ),
    )[:len(COUNTS) * GPUS_PER_WORKER]
    launch_root = Path(__file__).resolve().parents[4] / "logs"
    launch_root /= "multi_session_training"
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    launch_dir = launch_root / f"n3_{now}_{os.getpid()}"
    launch_dir.mkdir(parents=True)
    atomic_json(launch_dir / "launch.json", dict(
        plan=str(plan_path), source_report=str(source_report),
        devices=devices, microbatch=microbatch,
    ))
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[3])
    env["TF_USE_LEGACY_KERAS"] = "1"
    env["MPLCONFIGDIR"] = "/tmp/braid_n3_matplotlib"
    command = [
        sys.executable, "-m",
        "experiments.multisession.behavior_compute_scaling.n3_addon",
        "--stage", "coordinator", "--plan", str(plan_path),
        "--source-report", str(source_report),
        "--launch-dir", str(launch_dir),
        "--microbatch", str(microbatch),
    ]
    with (launch_dir / "coordinator.log").open("a") as stream:
        process = subprocess.Popen(
            command, env=env, stdout=stream,
            stderr=subprocess.STDOUT, start_new_session=True,
            preexec_fn=lambda: os.nice(LOW_CPU_PRIORITY),
        )
    atomic_json(launch_dir / "pid.json", dict(pid=process.pid))
    print(f"Detached n3 queue: {launch_dir.resolve()}")
    return launch_dir


def main() -> None:
    """Run one explicitly selected n3 add-on lifecycle stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=(
        "audit", "launch", "worker", "coordinator", "report",
    ))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--count", type=int, choices=COUNTS)
    parser.add_argument("--launch-dir", type=Path)
    parser.add_argument("--microbatch", type=int, default=4)
    args = parser.parse_args()
    plan_path = args.plan.resolve()
    source_report = args.source_report.resolve()
    plan = json.loads(plan_path.read_text())
    devices = ("/GPU:0", "/GPU:0")
    if args.stage in ("audit", "worker"):
        visible = tf.config.list_physical_devices("GPU")
        if not visible:
            raise RuntimeError("n3 audit or worker requires a visible GPU.")
        if len(visible) > 1:
            devices = ("/GPU:0", "/GPU:1")
    if args.stage == "report":
        publish(plan_path, source_report)
    elif args.stage == "launch":
        launch(plan_path, source_report, args.microbatch)
    elif args.stage == "coordinator":
        if args.launch_dir is None:
            raise ValueError("Coordinator requires --launch-dir.")
        settings = json.loads(
            (args.launch_dir / "launch.json").read_text()
        )
        coordinator(
            plan_path, source_report, args.launch_dir,
            settings["devices"], args.microbatch,
        )
    elif args.stage == "audit":
        print(json.dumps(audit_one(
            plan, source_report, devices, args.microbatch,
        )))
    else:
        if args.count is None:
            raise ValueError("Worker requires --count.")
        directory = fit_one(
            plan, source_report, args.count,
            devices, args.microbatch,
        )
        print(f"Completed n3 fit: {directory.resolve()}")


if __name__ == "__main__":
    main()
