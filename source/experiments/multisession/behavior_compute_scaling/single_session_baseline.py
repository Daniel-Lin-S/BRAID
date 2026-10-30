"""Fit direct-target single-session baselines and publish unified figures.

Usage: python -m experiments.multisession.behavior_compute_scaling.
single_session_baseline --stage audit|launch|worker|coordinator|report
  --plan ABSOLUTE_PLAN --source-report ABSOLUTE_REPORT
  [--session-index 0..3] [--microbatch N]

Input is the direct source plan and its complete unified report. Four
independent source fits use the common recordings and saved split protocol.
Outputs are checksum-valid source components/fits, campaign-local lifecycle
logs, and a new immutable report containing 11 source/performance figures,
source/score tables, baseline_scores.csv, and guide.md.
"""

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import tensorflow as tf

from experiments.execution.runtime import (
    select_gpus, thread_environment,
)
from experiments.multisession.behavior_compute_scaling import (
    behavior_decoding_diagnostics as decoding,
    direct_report,
)
from experiments.multisession.behavior_compute_scaling.pipeline import (
    fixed_targets, run_stage,
)
from experiments.multisession.behavior_compute_scaling.plan import (
    COUNTS, RECURRENT_STAGES,
)
from experiments.multisession.compute_scaling.branching import (
    resolved_schedule,
)
from experiments.multisession.compute_scaling.training import (
    balanced_update, gradient_function, optimizer_groups,
)
from experiments.multisession.data import (
    session_windows, source_features,
)
from experiments.multisession.plan import analysis_directory
from experiments.multisession.reports import write_table
from experiments.multisession.runner import dataset
from experiments.multisession.source import readout_data, residual_data
from experiments.multisession.store import (
    completed, fit_directory, job,
)
from experiments.multisession.training import (
    recurrent_data, stage_predictions,
)
from experiments.storage.cache import atomic_json, fingerprint

COMMON_SESSIONS = 4
MATCHED_MAIN = 1920
MATCHED_TOTAL = 2400
MATCHED_PROTOCOL = "architecture_matched_inner_stopped_2400_v1"
MAIN_UPDATES = 640
TOTAL_UPDATES = 800
SESSION_BATCH = 8
FIXED_CAP = 2000
FIXED_EXPOSURE = SESSION_BATCH * FIXED_CAP
VALIDATION_EXPOSURE = 200
AUDIT_UPDATES = 3
MINIMUM_FREE_MIB = 4096
POLL_SECONDS = 10
LOW_CPU_PRIORITY = 10
WORKER_CPU_THREADS = 2
LOCAL_DEVICE = ("/GPU:0", "/GPU:0")
BASELINE_PROTOCOL = "direct_single_session_800_updates_v1"
GUIDE = """# Direct behavior source scaling with single-session baseline

All eleven figures share one figures/ directory. The source_loss panels
show the four multi-session models' seven recurrent budgets against training
FLOPs. The nine performance panels show their selected checkpoints and a
gray horizontal line: the equal mean of four independently fitted models,
one per common source recording. Position and velocity score recorded
behavior in each recording's training-normalized units. Neural performance
scores the full readout plus RNN2 forecast against normalized spike counts.

Each single-session RNN1 and RNN2 uses 640 main updates plus 160 decay
updates at batch 8. Its matched neural readout has a 2,000-update cap,
inner-validation checks every 200 sampled windows, patience 10, and restores
its best checkpoint. The 16,000-window readout exposure cap matches the
four-session campaign's per-recording exposure. The single-session models
are selected without looking at the outer test set. The multi-session
checkpoint is selected using this same holdout, so its score is descriptive.

Each panel uses the four trained horizons: 50, 100, 200, and 400 ms. CC and
R2 average valid channels within each recording, then the four recordings.
MSE/dummy MSE divides each recording's model MSE by its training-mean dummy
MSE on the same test samples before averaging recordings. Values above one
are valid. No target adaptation or uncertainty interval is included.
"""


def reference(plan: dict) -> tuple[str, dict]:
    """Return the completed four-source fit defining common windows."""
    matches = [
        key for key in plan["source_references"]
        if plan["jobs"][key]["source_count"] == COMMON_SESSIONS
    ]
    if len(matches) != 1:
        raise ValueError("Expected one four-source reference fit.")
    key = matches[0]
    directory = fit_directory(plan, key)
    if completed(directory) is None:
        raise ValueError(f"Incomplete four-source fit: {directory}.")
    return key, json.loads((directory / "data.json").read_text())


def saved_spec(
    plan: dict, index: int, matched: bool = False,
) -> tuple[dict, dict, Path]:
    """Resolve a completed reference session without loading its arrays."""
    key, inventory = reference(plan)
    original = plan["jobs"][key]
    names = original["sessions"][:COMMON_SESSIONS]
    if index not in range(len(names)):
        raise ValueError(f"Expected session index 0-3, got {index}.")
    name = names[index]
    record = inventory[name]
    settings = dict(
        original["source_training"],
        batch_size=SESSION_BATCH,
        session_batch_sizes={"1": SESSION_BATCH},
        checkpoints=[MAIN_UPDATES],
        total_updates=[TOTAL_UPDATES],
        update_multipliers={"1": 1},
        fixed_exposure=FIXED_EXPOSURE,
        validation_exposure=VALIDATION_EXPOSURE,
        maximum_updates=MAIN_UPDATES,
    )
    spec = dict(
        original, source_count=1, sessions=[name],
        sources={name: original["sources"][name]},
        implementation=BASELINE_PROTOCOL,
        source_training=resolved_schedule(settings, 1),
        data_identity=fingerprint(record),
    )
    if spec["source_training"]["fixed_update_cap"] != FIXED_CAP:
        raise ValueError("Single-session readout cap is not 2,000.")
    if matched:
        spec.pop("continuation", None)
        spec["implementation"] = MATCHED_PROTOCOL
        spec["initialization_session_index"] = index
        spec["initialization_dimensions"] = [
            [sum(inventory[item]["scales"][key]["keep"])
             for key in ("Y", "Z")]
            for item in names
        ]
        spec["source_training"].update(
            checkpoints=[MATCHED_MAIN], total_updates=[MATCHED_TOTAL],
            maximum_updates=MATCHED_MAIN,
            recurrent_early_stopping="inner_validation",
        )
    identity = dict(protocol=spec["implementation"], spec=spec)
    artifact = Path(plan["settings"]["paths"]["artifact_root"])
    directory = (
        artifact / "multi_session_training" / "fits"
        / f"source_single_{fingerprint(identity)}"
    )
    return spec, record, directory


def prepare(
    plan: dict, index: int, matched: bool = False,
) -> tuple[dict, dict, dict, dict, dict, Path]:
    """Rebuild one recording's exact saved train/inner/test split."""
    spec, saved, directory = saved_spec(plan, index, matched)
    name = spec["sessions"][0]
    adapter = dataset(plan["settings"])
    adapter.mode = "off"
    features = source_features(
        adapter.load(name), spec["data"],
        spec["source_training"]["inner_validation_fraction"],
    )
    train, inner, outer, record = session_windows(
        features, True, include_test=True,
    )
    if record != saved:
        raise ValueError(f"Single-session split changed for {name}.")
    if not all(len(row["Z"]) for row in (train, inner, outer)):
        raise ValueError(f"Empty source split for {name}.")
    return spec, train, inner, outer, record, directory


def initialized_model(
    factory: object, dimensions: list, spec: dict,
) -> object:
    """Match each private map's initialization to the four-source model.

    ``dimensions`` contains singleton (neural, behavior) widths. The
    reference model is constructed only to extract its initial weights;
    it is never fitted and allocates no optimizer state.
    """
    model = factory(dimensions, spec, LOCAL_DEVICE)
    if "initialization_session_index" not in spec:
        return model
    from .n3_session import selected_session_weights

    reference_model = factory(
        spec["initialization_dimensions"], spec, LOCAL_DEVICE
    )
    index = spec["initialization_session_index"]
    weights = reference_model.weights()
    selected = (
        {"0": weights[str(index)]} if str(index) in weights
        else selected_session_weights(weights, index)
    )
    model.restore(selected)
    return model


def source_view(
    spec: dict, record: dict, directory: Path,
) -> dict:
    """Read a single-recording path with inner-only checkpoint selection."""
    name = spec["sessions"][0]
    path = json.loads((directory / "path.json").read_text())
    return dict(
        count=1, spec=spec, fit=directory, path=path,
        inventory={name: record},
        selection=dict(
            selected_budget=path["budget"],
            selection_role="inner_validation",
            evaluation_set_id=fingerprint({name: record["validation"]}),
        ),
    )


def fit_one(
    plan: dict, index: int, microbatch: int, matched: bool = False,
) -> Path:
    """Fit or reuse one independent matched three-stage path."""
    spec, train, inner, outer, record, directory = prepare(
        plan, index, matched
    )
    identity = dict(protocol=spec["implementation"], spec=spec)
    with job(directory, identity) as needed:
        if not needed:
            return directory
        root = directory.parent.parent / "source_components"
        dimensions = [(train["Y"].shape[-1], train["Z"].shape[-1])]
        from experiments.multisession.behavior_supervision.models import (
            neural_readout_model, rnn1_model, rnn2_model,
        )
        scoring = inner if matched else outer
        prepared, stages, parent, _ = fixed_targets(
            ([train], [inner], [scoring]), dimensions,
            spec["sessions"], spec, directory, root,
            LOCAL_DEVICE, microbatch,
            initialize=lambda factory: initialized_model(
                factory, dimensions, spec
            ),
        )
        main_updates = MATCHED_MAIN if matched else MAIN_UPDATES
        total_updates = MATCHED_TOTAL if matched else TOTAL_UPDATES
        schedule = (main_updates, total_updates) if matched else None
        main = initialized_model(rnn1_model, dimensions, spec)
        behavior = [
            recurrent_data(part[0], "Z") for part in prepared
        ]
        first, parent = run_stage(
            "behavior_relevant", main, [behavior[0]],
            [behavior[2]], spec["sessions"], spec, root,
            parent, main_updates, LOCAL_DEVICE, microbatch,
            validation=[behavior[1]] if matched else None,
            stopping_schedule=schedule,
        )
        neural = [
            recurrent_data(row) for row in (train, inner, scoring)
        ]
        latent = [
            stage_predictions(main, [row])[1]
            for row in behavior
        ]
        readout = initialized_model(neural_readout_model, dimensions, spec)
        readout_inputs = [
            readout_data([row], states)
            for row, states in zip(neural, latent)
        ]
        second, parent = run_stage(
            "neural_readout", readout, readout_inputs[0],
            readout_inputs[2], spec["sessions"], spec, root,
            parent, 0, LOCAL_DEVICE, microbatch,
            validation=readout_inputs[1],
        )
        residual = initialized_model(rnn2_model, dimensions, spec)
        third, _ = run_stage(
            "residual", residual,
            residual_data(main, readout, [neural[0]]),
            residual_data(main, readout, [neural[2]]),
            spec["sessions"], spec, root, parent,
            main_updates, LOCAL_DEVICE, microbatch,
            validation=(
                residual_data(main, readout, [neural[1]]) if matched else None
            ),
            stopping_schedule=schedule,
        )
        stages.update(
            behavior_relevant=first, neural_readout=second,
            residual=third,
        )
        atomic_json(directory / "data.json", {
            spec["sessions"][0]: record,
        })
        atomic_json(directory / "path.json", dict(
            budget=main_updates, total_updates=total_updates,
            stages=stages,
            full_path_flops=sum(
                row["component_flops"] for row in stages.values()
            ),
        ))
        from experiments.multisession.behavior_supervision.targets import (
            GENERATED, target_mode,
        )
        if matched and target_mode(spec) == GENERATED:
            from . import n3_session

            correction = n3_session.fit_one(
                plan, source_view(spec, record, directory), 0,
                microbatch, size=16,
            )
            atomic_json(directory / "n3.json", dict(
                component=correction.name,
                checksums=completed(correction)["checksums"],
            ))
    return directory


def scored_baseline(plan: dict) -> tuple[list[dict], list[dict]]:
    """Score four independent fits on their matched outer windows."""
    rows = []
    for index in range(COMMON_SESSIONS):
        spec, train, _, outer, record, directory = prepare(
            plan, index
        )
        if completed(directory) is None:
            raise ValueError(f"Incomplete single-session fit: {directory}.")
        name = spec["sessions"][0]
        path = json.loads((directory / "path.json").read_text())
        inventory = {name: record}
        selection = dict(
            selected_budget=path["budget"],
            evaluation_set_id=fingerprint({
                name: record["validation"]
            }),
        )
        source = dict(
            count=1, spec=spec, fit=directory,
            inventory=inventory, path=path, selection=selection,
        )
        common = {name: dict(training=train, test=outer)}
        behavior = decoding.selected_forecasts(source, common)
        behavior_rows = decoding.session_scores(
            source, common, behavior
        )
        direct_report.verify_stage_loss(
            source, behavior_rows, "behavior_relevant", 2
        )
        neural = direct_report.full_neural_forecasts(source, common)
        neural_rows = direct_report.neural_scores(
            source, common, neural
        )
        direct_report.verify_stage_loss(
            source, neural_rows, "residual", 1
        )
        rows.extend(behavior_rows + neural_rows)
    summary = []
    for group in direct_report.GROUPS:
        for horizon in direct_report.EXPECTED_HORIZONS:
            selected = [
                row for row in rows if row["group"] == group
                and row["horizon"] == horizon
            ]
            if len(selected) != COMMON_SESSIONS or len({
                row["session"] for row in selected
            }) != COMMON_SESSIONS:
                raise ValueError("Missing single-session baseline scores.")
            summary.append(dict(
                group=group, horizon=horizon,
                recordings=COMMON_SESSIONS,
                **{
                    metric: float(np.mean([
                        row[metric] for row in selected
                    ]))
                    for metric in direct_report.METRICS
                },
            ))
    return rows, summary


def publish(plan_path: Path, source_report: Path) -> Path:
    """Publish source and performance panels with the complete baseline."""
    from .report import source_figures, source_tables

    plan = json.loads(plan_path.read_text())
    source_marker = completed(source_report)
    if source_marker is None:
        raise ValueError(f"Incomplete source report: {source_report}.")
    fits = [prepare(plan, i)[-1] for i in range(COMMON_SESSIONS)]
    markers = [completed(path) for path in fits]
    if any(marker is None for marker in markers):
        raise ValueError("All four single-session fits must finish first.")
    identity = dict(
        group="direct_source_with_single_session_baseline_v2",
        source_report_checksums=source_marker["checksums"],
        baseline_checksums={
            path.name: marker["checksums"]
            for path, marker in zip(fits, markers)
        },
        guide=GUIDE,
    )
    root = analysis_directory(
        plan, plan["name"], plan_path.parent.name
    )
    destination = root / "reports" / fingerprint(identity)
    with job(destination, identity) as needed:
        if needed:
            sources, sessions, choices = source_tables(plan)
            with (source_report / "tables" / "summary.csv").open() as f:
                direct = list(csv.DictReader(f))
            with (source_report / "tables" / "session_scores.csv").open() as f:
                direct_sessions = list(csv.DictReader(f))
            if len(direct_sessions) != (
                len(COUNTS) * COMMON_SESSIONS
                * len(direct_report.GROUPS)
                * len(direct_report.EXPECTED_HORIZONS)
            ):
                raise ValueError("Incomplete direct per-session scores.")
            if len(direct) != (
                len(COUNTS) * len(direct_report.GROUPS)
                * len(direct_report.EXPECTED_HORIZONS)
            ):
                raise ValueError("Incomplete direct performance table.")
            summary = [dict(
                row, source_count=int(row["source_count"]),
                horizon=int(row["horizon"]),
                **{
                    metric: float(row[metric])
                    for metric in direct_report.METRICS
                },
            ) for row in direct]
            baseline_rows, baseline = scored_baseline(plan)
            tables = destination / "tables"
            figure_dir = destination / "figures"
            tables.mkdir()
            figure_dir.mkdir()
            write_table(tables / "source_loss.csv", sources)
            write_table(
                tables / "source_session_loss.csv", sessions
            )
            write_table(tables / "selection.csv", choices)
            write_table(tables / "session_scores.csv", direct_sessions)
            write_table(tables / "baseline_scores.csv", baseline_rows)
            write_table(tables / "baseline_summary.csv", baseline)
            write_table(tables / "summary.csv", summary)
            (destination / "guide.md").write_text(GUIDE)
            source_figures(
                sources, figure_dir, stages=RECURRENT_STAGES,
                expected_points=len(
                    plan["settings"]["source_training"]["total_updates"]
                ), direct=True,
            )
            direct_report.figures(summary, figure_dir, baseline)
    if completed(destination) is None:
        raise ValueError(f"Incomplete baseline report: {destination}.")
    figure_dir = (destination / "figures").resolve()
    pointer = dict(
        snapshot=str(figure_dir), identity=fingerprint(identity),
    )
    for name in (
        "latest_figures.json", "latest_source_figures.json",
        "latest_performance_figures.json",
    ):
        atomic_json(root / name, pointer)
    atomic_json(root / "latest_report.json", dict(
        snapshot=str(destination.resolve()),
        figures=str(figure_dir), identity=fingerprint(identity),
    ))
    return destination


def audit(plan: dict, microbatch: int, matched: bool = False) -> None:
    """Run three RNN1 updates without creating a fit artifact."""
    from experiments.multisession.behavior_supervision.models import (
        rnn1_model,
    )

    from experiments.multisession.behavior_supervision.models import (
        preprocessing_model,
    )
    from experiments.multisession.behavior_supervision.targets import (
        GENERATED, target_mode,
    )

    spec, train, _, _, _, _ = prepare(plan, 0, matched)
    generated = matched and target_mode(spec) == GENERATED
    factory = preprocessing_model if generated else rnn1_model
    model = initialized_model(
        factory,
        [(train["Y"].shape[-1], train["Z"].shape[-1])],
        spec,
    )
    optimizers = optimizer_groups(model, spec["source_training"])
    rng = np.random.default_rng(spec["seed"] + 2)
    functions = [gradient_function(model, 0)]
    for _ in range(AUDIT_UPDATES):
        balanced_update(
            model, [recurrent_data(train, "Y" if generated else "Z")],
            optimizers,
            rng, SESSION_BATCH, microbatch, LOCAL_DEVICE,
            functions,
        )
    print("Single-session first-stage audit: three updates succeeded.")


def status(launch: Path, states: dict[str, str], phase: str) -> None:
    """Publish lifecycle counts without model metrics."""
    counts = {
        state: sum(value == state for value in states.values())
        for state in ("pending", "running", "complete", "failed")
    }
    atomic_json(launch / "status.json", dict(
        phase=phase, counts=counts, total=len(states),
        updated_at_utc=datetime.now(timezone.utc).isoformat(),
    ))


def worker_environment(device: dict | None = None) -> dict:
    """Limit worker threads and optionally bind one GPU UUID."""
    env = dict(os.environ)
    env.update(thread_environment(WORKER_CPU_THREADS, 1))
    env["CUDA_VISIBLE_DEVICES"] = "" if device is None else device["uuid"]
    env["TF_USE_LEGACY_KERAS"] = "1"
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[3])
    env["MPLCONFIGDIR"] = "/tmp/braid_single_mpl"
    return env


def run_task_queue(
    tasks: list[dict], devices: list[dict], launch_dir: Path,
) -> dict[str, str]:
    """Schedule independent commands and validate immutable completions.

    Parameters
    ----------
    tasks : list of dict
        Each entry has a unique key, command argv, and target directory.
    devices : list of dict
        One selected GPU UUID per concurrent worker.
    launch_dir : Path
        Lifecycle logs; worker logs are nested under jobs/<task key>/.

    Returns
    -------
    dict
        Terminal complete/failed state for every requested task.
    """
    if not tasks or not devices:
        raise ValueError("A task queue requires tasks and available devices.")
    states = {task["key"]: "pending" for task in tasks}
    if len(states) != len(tasks):
        raise ValueError("Duplicate task keys in fit queue.")
    active = {}
    while True:
        for key, (process, device, target) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            del active[key]
            states[key] = (
                "complete" if code == 0 and completed(target) is not None
                else "failed"
            )
            if states[key] == "failed":
                with (launch_dir / "failures.jsonl").open("a") as stream:
                    stream.write(json.dumps(dict(
                        task=key, exit_code=code,
                        artifact=str(target.resolve()),
                    )) + "\n")
        busy = {device["uuid"] for _, device, _ in active.values()}
        free = [device for device in devices if device["uuid"] not in busy]
        for task in tasks:
            key, target = task["key"], Path(task["target"])
            if states[key] != "pending":
                continue
            if completed(target) is not None:
                states[key] = "complete"
                continue
            if not free:
                continue
            device = free.pop(0)
            log = launch_dir / "jobs" / key / "worker.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("a") as stream:
                process = subprocess.Popen(
                    task["command"], env=worker_environment(device),
                    stdout=stream, stderr=subprocess.STDOUT,
                    preexec_fn=lambda: os.nice(LOW_CPU_PRIORITY),
                )
            active[key] = (process, device, target)
            states[key] = "running"
        status(launch_dir, states, "active")
        if not active:
            return states
        time.sleep(POLL_SECONDS)


def coordinator(
    plan_path: Path, source_report: Path, launch: Path,
    microbatch: int,
) -> None:
    """Run each recording on one GPU and publish after all finish."""
    plan = json.loads(plan_path.read_text())
    devices = json.loads((launch / "launch.json").read_text())["devices"]
    tasks = []
    for index in range(COMMON_SESSIONS):
        spec, _, directory = saved_spec(plan, index)
        tasks.append(dict(
            key=spec["sessions"][0], target=str(directory),
            command=[
                sys.executable, "-m",
                "experiments.multisession.behavior_compute_scaling."
                "single_session_baseline", "--stage", "worker",
                "--plan", str(plan_path),
                "--source-report", str(source_report),
                "--session-index", str(index),
                "--microbatch", str(microbatch),
            ],
        ))
    states = run_task_queue(tasks, devices, launch)
    if "failed" in states.values():
        status(launch, states, "failed")
        return
    try:
        report = publish(plan_path, source_report)
    except Exception as error:
        atomic_json(launch / "report_error.json", dict(
            error=f"{type(error).__name__}: {error}",
        ))
        status(launch, states, "report_failed")
        return
    atomic_json(launch / "report.json", dict(snapshot=str(report.resolve())))
    status(launch, states, "complete")


def launch(
    plan_path: Path, source_report: Path, microbatch: int,
) -> Path:
    """Detach four independent single-GPU fits after audit."""
    if microbatch <= 0 or microbatch > SESSION_BATCH:
        raise ValueError("Expected microbatch between 1 and 8.")
    plan = json.loads(plan_path.read_text())
    if completed(source_report) is None:
        raise ValueError(f"Incomplete direct source report: {source_report}.")
    devices = select_gpus("auto", 8)
    candidates = sorted(
        (device for device in devices
         if device["free_memory_mib"] >= MINIMUM_FREE_MIB),
        key=lambda item: item["free_memory_mib"], reverse=True,
    )
    if len(candidates) < COMMON_SESSIONS:
        raise RuntimeError("Fewer than four GPUs have enough free memory.")
    selected = candidates[:COMMON_SESSIONS]
    root = Path(__file__).resolve().parents[4]
    root /= "logs/multi_session_training"
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = root / f"single_session_{now}_{os.getpid()}"
    directory.mkdir(parents=True)
    atomic_json(directory / "launch.json", dict(
        plan=str(plan_path.resolve()),
        source_report=str(source_report.resolve()),
        devices=selected, microbatch=microbatch,
    ))
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[3])
    env["TF_USE_LEGACY_KERAS"] = "1"
    env["MPLCONFIGDIR"] = "/tmp/braid_single_mpl"
    command = [
        sys.executable, "-m",
        "experiments.multisession.behavior_compute_scaling."
        "single_session_baseline",
        "--stage", "coordinator", "--plan", str(plan_path),
        "--source-report", str(source_report),
        "--launch-dir", str(directory),
        "--microbatch", str(microbatch),
    ]
    with (directory / "coordinator.log").open("a") as stream:
        process = subprocess.Popen(
            command, env=env, stdout=stream,
            stderr=subprocess.STDOUT, start_new_session=True,
            preexec_fn=lambda: os.nice(LOW_CPU_PRIORITY),
        )
    atomic_json(directory / "pid.json", dict(pid=process.pid))
    print(f"Detached single-session baselines: {directory.resolve()}")
    return directory


def main() -> None:
    """Dispatch baseline audit, fits, or complete report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", required=True,
        choices=("audit", "launch", "worker", "coordinator", "report"),
    )
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--source-report", required=True, type=Path)
    parser.add_argument("--session-index", type=int)
    parser.add_argument("--microbatch", type=int, default=4)
    parser.add_argument("--launch-dir", type=Path)
    args = parser.parse_args()
    plan_path = args.plan.resolve()
    source_report = args.source_report.resolve()
    plan = json.loads(plan_path.read_text())
    if args.stage == "launch":
        launch(plan_path, source_report, args.microbatch)
    elif args.stage == "coordinator":
        if args.launch_dir is None:
            raise ValueError("Coordinator requires --launch-dir.")
        coordinator(
            plan_path, source_report, args.launch_dir.resolve(),
            args.microbatch,
        )
    elif args.stage == "report":
        publish(plan_path, source_report)
    else:
        if args.session_index is None:
            raise ValueError("Worker and audit require session index.")
        visible = tf.config.list_physical_devices("GPU")
        if not visible:
            raise RuntimeError("Single-session worker needs one GPU.")
        tf.config.experimental.set_memory_growth(visible[0], True)
        if args.stage == "audit":
            audit(plan, args.microbatch)
        else:
            fit_one(plan, args.session_index, args.microbatch)


if __name__ == "__main__":
    main()
