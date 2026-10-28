"""Fit session-private n3 corrections for selected behavioral sources.

Usage: python -m experiments.multisession.behavior_compute_scaling.n3_session
  --stage audit|launch|worker|coordinator|report --plan ABSOLUTE_PLAN
  --source-report ABSOLUTE_REPORT [--count N --session-index I]

Inputs are four checksum-valid selected source checkpoints and the four common
recordings per count. Sixteen independent n3 fits live in the common source
component store. The launch status records per-session update, validation,
completion and failure. Once every fit completes, one immutable report under
the existing analysis root contains behavioral tables and the n3-only and
paired comparison figures.
"""

import argparse
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np
import tensorflow as tf

from BRAID.multisession import HORIZONS, MixedForecaster
from experiments.execution.runtime import select_gpus, thread_environment
from experiments.multisession.behavior_compute_scaling import (
    behavior_decoding_diagnostics as decoding,
    n3_addon,
)
from experiments.multisession.behavior_compute_scaling.plan import COUNTS
from experiments.multisession.behavior_compute_scaling.training import (
    balanced_update, evaluate_sessions, fit_segment, gradient_function,
    optimizer_groups, validation_due,
)
from experiments.multisession.data import session_windows, source_features
from experiments.multisession.plan import analysis_directory
from experiments.multisession.reports import write_table
from experiments.multisession.runner import dataset
from experiments.multisession.store import completed, job, load_pickle
from experiments.multisession.training import recurrent_data, stage_predictions
from experiments.storage.cache import atomic_json, file_digest, fingerprint

PROTOCOL = "frozen_rnn_input_only_n3_per_session_v2"
COMMON_SESSIONS = 4
GPU_INVENTORY_COUNT = 8
MINIMUM_FREE_MIB = 4096
POLL_SECONDS = 5
WORKER_CPU_THREADS = 2
LOW_CPU_PRIORITY = 10
AUDIT_UPDATES = 3
AUDIT_COUNT = 4
N3_SIZE = 1
N3_DECODER_WIDTH = 64
N3_SEED_OFFSET = 5
TASK_WIDTH = 2
BEHAVIOR_WIDTH = 4
PROGRESS_PATTERN = re.compile(
    r"update=(?P<update>\d+)(?:/(?P<cap>\d+))? "
    r"(?:(?:train=(?P<train>\S+) inner=(?P<inner>\S+) "
    r"best=(?P<best>\d+) stale=(?P<stale>\d+))|"
    r"(?:loss=(?P<loss>\S+) lr=\S+))"
)
GUIDE = """# Recorded-behavior source decoding with n3

Each curve uses the previously transfer-selected RNN1 checkpoint for 4, 8,
16, or 33 source sessions. RNN1, RNN2 and preprocessing are frozen. Only the
four source recordings appearing in every source set receive an n3 fit. The
other source recordings cannot affect these session-private n3 corrections.

Each n3 is an independently fitted one-state, input-only BRAID post-model.
Its input is task target U, and its output is added to RNN1's forecast after
that forecast is converted to the recorded-behavior training-normalized
scale. For horizon h, the training loss is the window mean of
(Z_{t+h} - P_{t+h} - n3_h(U))^2, summed over 50/100/200/400 ms. No recorded
behavior is an inference input. Each session independently selects its own
checkpoint by its inner-validation loss; the same per-session exposure caps,
validation update boundaries, patience and learning rate as the original
fixed-stage protocol apply.

All plotted scores use the identical 87 chronological outer-test windows of
the four common source recordings for every source count. RNN1/RNN2 transfer
selection used this outer holdout, so scores are descriptive. CC and R2 are
means of the two coordinates per group. MSE is the mean squared error in
recorded-behavior training-normalized units; MSE/dummy MSE divides by the
same session and horizon's training-target-mean predictor error. Values above
one are valid. Each plotted point is an equal mean of four session scores.
The paired comparison uses the same RNN1 checkpoints and test windows, with
and without n3. No target adaptation is performed.
"""


def sources(plan: dict, source_report: Path) -> dict[int, dict]:
    """Validate four selected paths and return them by source count."""
    selected = decoding.checked_sources(plan, source_report)
    return {row["count"]: row for row in selected}


def session_name(source: dict, index: int) -> str:
    """Identify one of the four fixed common source recordings."""
    if index not in range(COMMON_SESSIONS):
        raise ValueError(f"Expected common session index 0-3, got {index}.")
    return source["spec"]["sessions"][index]


def session_settings(source: dict) -> dict:
    """Preserve original cap and validation-update boundaries for one fit."""
    settings = dict(n3_addon.n3_settings(source))
    count = source["count"]
    local = settings["session_batch_sizes"][str(count)]
    original_interval = settings["validation_exposure"]
    original_batch = count * local
    settings["session_batch_sizes"] = {"1": local}
    settings["validation_exposure"] = original_interval / count
    cap = settings["fixed_update_cap"]
    for update in range(1, cap + 1):
        expected = validation_due(
            update, original_batch, original_interval, cap
        )
        actual = validation_due(
            update, local, settings["validation_exposure"], cap
        )
        if actual != expected:
            raise ValueError(
                f"Per-session validation check changed at update {update}."
            )
    return settings


def session_seed(source: dict, index: int) -> int:
    """Keep each session's native private-map initialization distinct."""
    return source["spec"]["seed"] + N3_SEED_OFFSET + index


def session_model(
    source: dict, index: int,
    device: tuple[str, str] | None = None,
) -> MixedForecaster:
    """Build one U-only additive n3 state of width one."""
    return MixedForecaster(
        [(0, BEHAVIOR_WIDTH)], N3_SIZE, "none",
        session_seed(source, index), decoder_width=N3_DECODER_WIDTH,
        input_dimensions=[TASK_WIDTH],
        forward_dimensions=TASK_WIDTH,
        session_devices=device,
    )


def source_component(source: dict) -> Path:
    """Validate the selected upstream RNN1 checkpoint."""
    key = source["path"]["stages"]["behavior_relevant"][
        "component_id"
    ]
    path = n3_addon.component_root(source) / key
    if completed(path) is None:
        raise ValueError(f"Incomplete selected RNN1 checkpoint: {path}.")
    return path


def identity(source: dict, index: int) -> dict:
    """Identify one scientific n3 fit without runtime GPU metadata."""
    name = session_name(source, index)
    upstream = source_component(source)
    return dict(
        protocol=PROTOCOL, stage="behavior_post_n3",
        source_fit=source["fit"].name,
        source_count=source["count"],
        source_session=name, session_index=index,
        data_identity=fingerprint(source["inventory"][name]),
        evaluation_set_id=source["selection"]["evaluation_set_id"],
        selected_budget=source["selection"]["selected_budget"],
        upstream_checkpoint=upstream.name,
        upstream_checksums=completed(upstream)["checksums"],
        generated_scales_checksum=file_digest(
            source["fit"] / "preprocessing.json"
        ),
        latent=N3_SIZE, decoder_width=N3_DECODER_WIDTH,
        sharing="none", input="U", output="recorded_Z_residual",
        seed=session_seed(source, index), horizons=HORIZONS,
        source_training=session_settings(source),
    )


def directory(source: dict, index: int) -> Path:
    """Resolve one immutable fit path without fitting it."""
    scientific = dict(
        identity(source, index), start=0,
        stop=session_settings(source)["fixed_update_cap"],
        branch=False, parent=None, parent_checksums=None,
        fixed_stage=True,
    )
    return n3_addon.component_root(source) / fingerprint(scientific)


def windows(
    plan: dict, source: dict, index: int,
) -> tuple[dict, dict]:
    """Load and verify one session's training and inner-validation windows."""
    name = session_name(source, index)
    adapter = dataset(plan["settings"])
    adapter.mode = "off"
    features = source_features(
        adapter.load(name), source["spec"]["data"],
        source["spec"]["source_training"][
            "inner_validation_fraction"
        ],
    )
    train, inner, _, record = session_windows(
        features, True, include_test=True,
    )
    if record != source["inventory"][name]:
        raise ValueError(f"Saved source windows changed for {name}.")
    if not len(train["Z"]) or not len(inner["Z"]):
        raise ValueError(f"Empty n3 train or inner validation: {name}.")
    return train, inner


def selected_session_weights(weights: dict, index: int) -> dict:
    """Extract one private map set and its shared A from RNN1 weights."""
    selected = {
        f"0/{full_key.split('/', 1)[1]}": value
        for full_key, value in weights.items()
        if full_key.startswith(f"{index}/")
    }
    expected = {f"0/{key}" for key in ("A", "K", "Afw", "Kfw", "C")}
    if set(selected) != expected:
        raise ValueError(f"Expected RNN1 maps {expected}, got {set(selected)}.")
    return selected


def base_forecast(
    source: dict, index: int, partition: dict,
) -> np.ndarray:
    """Forecast one source session using its saved RNN1 private maps."""
    name = session_name(source, index)
    neural_width = sum(
        source["inventory"][name]["scales"]["Y"]["keep"]
    )
    model = MixedForecaster(
        [(neural_width, BEHAVIOR_WIDTH)],
        source["spec"]["latent"], "A",
        source["spec"]["seed"] + 2,
        decoder_width=N3_DECODER_WIDTH,
    )
    payload = load_pickle(source_component(source) / "checkpoint.pkl")
    selected = selected_session_weights(payload["weights"], index)
    model.restore(selected)
    raw, _ = stage_predictions(model, [recurrent_data(partition, "Z")])
    scales = json.loads(
        (source["fit"] / "preprocessing.json").read_text()
    )["scales"]
    if len(scales) != source["count"]:
        raise ValueError("Generated target scales differ from source count.")
    prediction = decoding.target_scale(scales[index]).inverse(raw[0])
    expected = (len(HORIZONS), *partition["Z"].shape)
    if prediction.shape != expected or not np.isfinite(prediction).all():
        raise ValueError(f"Invalid RNN1 prediction for {name}: {expected}.")
    tf.keras.backend.clear_session()
    return prediction


def fit_one(
    plan: dict, source: dict, index: int, microbatch: int,
) -> Path:
    """Fit or reuse one independently selected n3 session checkpoint."""
    target = directory(source, index)
    if completed(target) is not None:
        return target
    train, inner = windows(plan, source, index)
    training = n3_addon.post_data(
        [train], [base_forecast(source, index, train)]
    )
    validation = n3_addon.post_data(
        [inner], [base_forecast(source, index, inner)]
    )
    model = session_model(source, index, ("/GPU:0", "/GPU:0"))
    settings = session_settings(source)
    result, _ = fit_segment(
        model, training, n3_addon.component_root(source),
        identity(source, index), settings, session_seed(source, index),
        0, settings["fixed_update_cap"], None,
        ("/GPU:0", "/GPU:0"), microbatch,
        validation=validation,
    )
    if result != target or completed(result) is None:
        raise ValueError("Session n3 component path or checksum differs.")
    return result


def audit(
    plan: dict, source: dict, index: int, microbatch: int,
) -> dict:
    """Time three updates and validation without persisting audit artifacts."""
    start = time.monotonic()
    train, inner = windows(plan, source, index)
    training = n3_addon.post_data(
        [train], [base_forecast(source, index, train)]
    )
    validation = n3_addon.post_data(
        [inner], [base_forecast(source, index, inner)]
    )
    prepared = time.monotonic()
    settings = session_settings(source)
    model = session_model(source, index, ("/GPU:0", "/GPU:0"))
    optimizers = optimizer_groups(model, settings)
    rng = np.random.default_rng(session_seed(source, index))
    functions = [gradient_function(model, 0)]
    for _ in range(AUDIT_UPDATES):
        balanced_update(
            model, training, optimizers, rng,
            settings["session_batch_sizes"]["1"], microbatch,
            ("/GPU:0", "/GPU:0"), functions,
        )
    updated = time.monotonic()
    losses, _ = evaluate_sessions(model, validation)
    return dict(
        preparation_seconds=prepared - start,
        update_seconds=(updated - prepared) / AUDIT_UPDATES,
        validation_seconds=time.monotonic() - updated,
        validation_losses=losses,
    )


def corrected(
    source: dict, index: int, partition: dict,
    prior: np.ndarray,
) -> np.ndarray:
    """Add one saved session's U-only n3 prediction to RNN1."""
    model = session_model(source, index)
    path = directory(source, index)
    if completed(path) is None:
        raise ValueError(f"Incomplete n3 session component: {path}.")
    model.restore(load_pickle(path / "checkpoint.pkl")["weights"])
    data = n3_addon.post_data([partition], [prior])
    residual, _ = stage_predictions(model, data)
    result = prior + residual[0]
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite n3-corrected forecast.")
    tf.keras.backend.clear_session()
    return result


def publish(plan_path: Path, source_report: Path) -> Path:
    """Publish both behavioral figure groups after all 16 completions."""
    plan = json.loads(plan_path.read_text())
    selected = sources(plan, source_report)
    references = {}
    for count, source in selected.items():
        references[str(count)] = {}
        for index in range(COMMON_SESSIONS):
            name = session_name(source, index)
            path = directory(source, index)
            marker = completed(path)
            if marker is None:
                raise ValueError(
                    f"Cannot publish with incomplete n3 fit: {path}."
                )
            references[str(count)][name] = dict(
                component=path.name, checksums=marker["checksums"],
            )
    report_marker = completed(source_report)
    identity_record = dict(
        group="behavior_n3_per_session_selected_source_scaling",
        protocol=PROTOCOL,
        source_plan_checksum=file_digest(plan_path),
        source_report_checksums=report_marker["checksums"],
        components=references, guide=GUIDE,
    )
    root = analysis_directory(
        plan, "multi_session_behavior_compute_scaling",
        plan_path.parent.name,
    )
    destination = root / "reports" / fingerprint(identity_record)
    with job(destination, identity_record) as needed:
        if needed:
            common = decoding.fixed_windows(plan, list(selected.values()))
            scores = []
            for count in COUNTS:
                source = selected[count]
                names = source["spec"]["sessions"][:COMMON_SESSIONS]
                priors = [
                    base_forecast(source, index, common[name]["test"])
                    for index, name in enumerate(names)
                ]
                forecasts = [
                    corrected(
                        source, index, common[name]["test"], priors[index]
                    )
                    for index, name in enumerate(names)
                ]
                for variant, values in zip(
                    n3_addon.VARIANTS, (priors, forecasts)
                ):
                    scores.extend(
                        dict(row, variant=variant)
                        for row in decoding.session_scores(
                            source, common, values
                        )
                    )
            summary = []
            for variant in n3_addon.VARIANTS:
                subset = [
                    row for row in scores if row["variant"] == variant
                ]
                summary.extend(
                    dict(row, variant=variant)
                    for row in decoding.summarize(subset)
                )
            figures = destination / "figures"
            figures.mkdir()
            write_table(destination / "session_scores.csv", scores)
            write_table(destination / "summary.csv", summary)
            atomic_json(destination / "n3_components.json", references)
            (destination / "guide.md").write_text(GUIDE)
            decoding.figures(
                [row for row in summary
                 if row["variant"] == "RNN1 + n3"],
                figures,
                subtitle="Transfer-selected RNN1 + U-only n3; common test set",
                title_prefix="Behavior decoding with n3",
                compact=True,
            )
            n3_addon.comparison_figures(summary, figures)
    atomic_json(root / "latest_behavior_n3.json", dict(
        snapshot=str(destination.resolve()),
        identity=fingerprint(identity_record),
    ))
    print(f"Saved n3 behavioral figures: {(destination / 'figures').resolve()}")
    return destination


def key(count: int, index: int) -> str:
    """Return a stable launch-local member label."""
    return f"{count}:{index}"


def log_progress(path: Path) -> dict:
    """Read exact last logged update and loss from a live worker."""
    progress = {}
    if not path.exists():
        return progress
    for line in path.read_text(errors="replace").splitlines():
        match = PROGRESS_PATTERN.search(line)
        if match is None:
            continue
        values = match.groupdict()
        progress["update"] = int(values["update"])
        score = values["train"] or values["loss"]
        progress["training_loss"] = float(score)
        if values["inner"] is not None:
            progress["inner_validation_loss"] = float(values["inner"])
            progress["best_update"] = int(values["best"])
            progress["stale_checks"] = int(values["stale"])
    return progress


def member_record(source: dict, index: int) -> dict:
    """Describe one session's independent fixed-stage fit."""
    return dict(
        state="pending", source_count=source["count"],
        session=session_name(source, index), session_index=index,
        update=0, cap=session_settings(source)["fixed_update_cap"],
        component=directory(source, index).name,
    )


def publish_status(launch: Path, members: dict, state: str) -> None:
    """Write current per-session state and separate completion counts."""
    counts = {
        value: sum(row["state"] == value for row in members.values())
        for value in ("pending", "running", "complete", "failed")
    }
    atomic_json(launch / "status.json", dict(
        state=state, members=members, counts=counts,
        observed_at_utc=datetime.now(timezone.utc).isoformat(),
    ))


def worker_command(
    plan_path: Path, source_report: Path, count: int,
    index: int, microbatch: int,
) -> list[str]:
    """Build one independent session worker invocation."""
    return [
        sys.executable, "-m",
        "experiments.multisession.behavior_compute_scaling.n3_session",
        "--stage", "worker", "--plan", str(plan_path),
        "--source-report", str(source_report),
        "--count", str(count), "--session-index", str(index),
        "--microbatch", str(microbatch),
    ]


def coordinator(
    plan_path: Path, source_report: Path, launch: Path,
    devices: list[dict], microbatch: int,
) -> None:
    """Schedule independent sessions and report only full membership."""
    plan = json.loads(plan_path.read_text())
    selected = sources(plan, source_report)
    tasks = [
        (count, index)
        for count in COUNTS for index in range(COMMON_SESSIONS)
    ]
    tasks.sort(
        key=lambda item: -session_settings(
            selected[item[0]]
        )["fixed_update_cap"]
    )
    members = {
        key(count, index): member_record(selected[count], index)
        for count, index in tasks
    }
    pending = []
    for count, index in tasks:
        label = key(count, index)
        path = directory(selected[count], index)
        if completed(path) is None:
            pending.append((count, index))
        else:
            training = json.loads((path / "training.json").read_text())
            members[label].update(
                state="complete", update=training["stop"],
                selected_update=training["selected_update"],
                inner_validation_loss=training["best_validation_loss"],
            )
    active = {}
    publish_status(launch, members, "active")
    while pending or active:
        for device in devices:
            slot = device["index"]
            if slot in active or not pending:
                continue
            if device["free_memory_mib"] < MINIMUM_FREE_MIB:
                continue
            count, index = pending.pop(0)
            label = key(count, index)
            env = dict(os.environ)
            env.update(thread_environment(WORKER_CPU_THREADS, 1))
            env["CUDA_VISIBLE_DEVICES"] = device["uuid"]
            env["TF_USE_LEGACY_KERAS"] = "1"
            env["PYTHONPATH"] = str(Path(__file__).resolve().parents[3])
            env["MPLCONFIGDIR"] = "/tmp/braid_n3_matplotlib"
            log = launch / f"n3_{count}_session_{index}.log"
            with log.open("a") as stream:
                process = subprocess.Popen(
                    worker_command(
                        plan_path, source_report, count, index,
                        microbatch,
                    ),
                    env=env, stdout=stream,
                    stderr=subprocess.STDOUT,
                    preexec_fn=lambda: os.nice(LOW_CPU_PRIORITY),
                )
            active[slot] = (count, index, process, log, time.monotonic())
            members[label].update(state="running", gpu_index=slot)
        for slot, item in list(active.items()):
            count, index, process, log, started = item
            label = key(count, index)
            members[label].update(log_progress(log))
            members[label]["elapsed_seconds"] = round(
                time.monotonic() - started, 1
            )
            code = process.poll()
            if code is None:
                continue
            del active[slot]
            path = directory(selected[count], index)
            marker = completed(path) if code == 0 else None
            if marker is None:
                members[label]["state"] = "failed"
                members[label]["exit_code"] = code
                continue
            training = json.loads((path / "training.json").read_text())
            members[label].update(
                state="complete", update=training["stop"],
                selected_update=training["selected_update"],
                inner_validation_loss=training["best_validation_loss"],
                training_flops=training["total_flops"],
            )
        publish_status(launch, members, "active")
        if pending or active:
            time.sleep(POLL_SECONDS)
    if any(row["state"] == "failed" for row in members.values()):
        publish_status(launch, members, "failed")
        return
    try:
        report = publish(plan_path, source_report)
    except Exception as error:
        publish_status(launch, members, "report_failed")
        atomic_json(launch / "report_error.json", dict(
            error=f"{type(error).__name__}: {error}",
        ))
        raise
    publish_status(launch, members, "complete")
    atomic_json(launch / "report.json", dict(path=str(report.resolve())))


def launch(
    plan_path: Path, source_report: Path, microbatch: int,
) -> Path:
    """Detach a session-granular queue on available GPUs."""
    plan = json.loads(plan_path.read_text())
    sources(plan, source_report)
    if microbatch <= 0:
        raise ValueError("n3 microbatch must be positive.")
    devices = select_gpus("auto", GPU_INVENTORY_COUNT)
    if any(row["free_memory_mib"] < MINIMUM_FREE_MIB for row in devices):
        raise RuntimeError("One or more GPUs lack n3 worker free memory.")
    launch_root = Path(__file__).resolve().parents[4]
    launch_root /= "logs/multi_session_training"
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    launch_dir = launch_root / f"n3_session_{now}_{os.getpid()}"
    launch_dir.mkdir(parents=True)
    atomic_json(launch_dir / "launch.json", dict(
        plan=str(plan_path), source_report=str(source_report),
        devices=devices, microbatch=microbatch,
        protocol=PROTOCOL,
    ))
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[3])
    env["TF_USE_LEGACY_KERAS"] = "1"
    env["MPLCONFIGDIR"] = "/tmp/braid_n3_matplotlib"
    command = [
        sys.executable, "-m",
        "experiments.multisession.behavior_compute_scaling.n3_session",
        "--stage", "coordinator", "--plan", str(plan_path),
        "--source-report", str(source_report),
        "--launch-dir", str(launch_dir),
        "--microbatch", str(microbatch),
    ]
    with (launch_dir / "coordinator.log").open("a") as stream:
        process = subprocess.Popen(
            command, env=env, stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            preexec_fn=lambda: os.nice(LOW_CPU_PRIORITY),
        )
    atomic_json(launch_dir / "pid.json", dict(pid=process.pid))
    print(f"Detached per-session n3 queue: {launch_dir.resolve()}")
    return launch_dir


def main() -> None:
    """Dispatch one explicitly selected session add-on lifecycle stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=(
        "audit", "launch", "worker", "coordinator", "report",
    ))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--count", type=int, choices=COUNTS)
    parser.add_argument("--session-index", type=int)
    parser.add_argument("--microbatch", type=int, default=4)
    parser.add_argument("--launch-dir", type=Path)
    args = parser.parse_args()
    plan_path = args.plan.resolve()
    source_report = args.source_report.resolve()
    plan = json.loads(plan_path.read_text())
    if args.stage == "launch":
        launch(plan_path, source_report, args.microbatch)
    elif args.stage == "report":
        publish(plan_path, source_report)
    elif args.stage == "coordinator":
        if args.launch_dir is None:
            raise ValueError("Coordinator requires --launch-dir.")
        record = json.loads(
            (args.launch_dir / "launch.json").read_text()
        )
        coordinator(
            plan_path, source_report, args.launch_dir,
            record["devices"], args.microbatch,
        )
    else:
        if args.count is None or args.session_index is None:
            raise ValueError("Session worker/audit requires count and index.")
        visible = tf.config.list_physical_devices("GPU")
        if not visible:
            raise RuntimeError("n3 session worker requires a visible GPU.")
        for device in visible:
            tf.config.experimental.set_memory_growth(device, True)
        source = sources(plan, source_report)[args.count]
        if args.stage == "audit":
            print(json.dumps(audit(
                plan, source, args.session_index, args.microbatch,
            )))
        else:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s %(levelname)s %(message)s",
            )
            result = fit_one(
                plan, source, args.session_index, args.microbatch,
            )
            print(f"Completed n3 session fit: {result.resolve()}")


if __name__ == "__main__":
    main()
