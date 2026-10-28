"""Schedule direct-behavior source budgets on four independent GPUs.

Usage: python -m experiments.multisession.behavior_compute_scaling.budget_queue
       --plan ABSOLUTE_PLAN --detach

Input is a published direct-behavior source-only plan. Output is a detached
campaign log with lifecycle status, nested task logs, and frozen source code.
Completed scientific artifacts remain in the plan's shared artifact root.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.execution.runtime import select_gpus, thread_environment
from experiments.multisession.behavior_compute_scaling.budget_parallel import (
    budget_directory, fit_budget, fit_prefixes, finish_source,
    prefix_component, validate_plan,
)
from experiments.multisession.compatibility import (
    freeze_source, verify_source_code,
)
from experiments.multisession.plan import PRIMARY
from experiments.multisession.queue_status import publish_status
from experiments.multisession.store import completed, fit_directory
from experiments.storage.cache import atomic_json, fingerprint, writer_lock

LOGGER = logging.getLogger(__name__)
GPU_COUNT = 4
POLL_SECONDS = 5
LOW_CPU_PRIORITY = 10
TASK_KINDS = ("prefix", "budget", "final")


def task_plan(plan: dict) -> dict:
    """Build runtime tasks without changing scientific source identities."""
    jobs = {}
    for source_key in plan["source_references"]:
        spec = plan["jobs"][source_key]
        multiplier = spec["source_training"]["update_multipliers"][
            str(spec["source_count"])
        ]
        budgets = [
            value * multiplier
            for value in spec["source_training"]["checkpoints"]
        ]
        for kind, budget in [
            ("prefix", None),
            *(("budget", value) for value in budgets),
            ("final", None),
        ]:
            identity = dict(
                source_fit=source_key, task_kind=kind, budget=budget,
            )
            task_key = fingerprint(identity)
            jobs[task_key] = dict(
                spec, kind="source", task_kind=kind,
                source_fit=source_key, budget=budget,
            )
    return dict(jobs=jobs)


def task_complete(
    plan: dict, launch: Path, spec: dict,
) -> bool:
    """Validate a task's durable scientific or prefix record."""
    source = spec["source_fit"]
    if completed(fit_directory(plan, source)) is not None:
        return True
    kind = spec["task_kind"]
    if kind == "budget":
        return completed(
            budget_directory(plan, source, spec["budget"])
        ) is not None
    if kind == "final":
        return False
    multiplier = spec["source_training"]["update_multipliers"][
        str(spec["source_count"])
    ]
    return all(
        prefix_component(plan, launch, source, budget * multiplier)
        is not None
        for budget in spec["source_training"]["checkpoints"]
    )


def ready_tasks(
    plan: dict, launch: Path, tasks: dict,
    states: dict[str, str], pending: set[str], active: dict,
) -> list[str]:
    """Release a budget when its main prefix becomes checksum-valid."""
    counts = Counter(
        tasks[key]["source_count"] for key, _ in active.values()
    )
    result = []
    for key in pending:
        spec = tasks[key]
        source = spec["source_fit"]
        kind = spec["task_kind"]
        if kind == "budget" and prefix_component(
            plan, launch, source, spec["budget"]
        ) is None:
            continue
        if kind == "final":
            related = [
                other for other, row in tasks.items()
                if row["source_fit"] == source and other != key
            ]
            if any(states[other] != "complete" for other in related):
                continue
        result.append(key)
    rank = dict(budget=0, prefix=1, final=2)
    return sorted(result, key=lambda key: (
        counts[tasks[key]["source_count"]],
        rank[tasks[key]["task_kind"]],
        -tasks[key]["source_count"],
        tasks[key]["budget"] or 0,
    ))


def task_log(launch: Path, key: str, spec: dict) -> Path:
    """Nest a runtime task log by source count and task type."""
    budget = "all" if spec["budget"] is None else str(spec["budget"])
    return (
        launch / "jobs" / "source_budget"
        / f"sources{spec['source_count']}" / spec["task_kind"]
        / f"{budget}__{key}.log"
    )


def start_task(
    plan_path: Path, launch: Path, source_hash: str,
    task_key: str, spec: dict, device: dict, threads: int,
) -> subprocess.Popen:
    """Launch one source task with exactly one visible GPU."""
    log = task_log(launch, task_key, spec)
    log.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.update(thread_environment(threads, 1))
    environment["CUDA_VISIBLE_DEVICES"] = device["uuid"]
    environment["TF_USE_LEGACY_KERAS"] = "1"
    environment["MPLCONFIGDIR"] = str((launch / "mpl_cache").resolve())
    command = [
        sys.executable, "-m",
        "experiments.multisession.behavior_compute_scaling.budget_queue",
        "--plan", str(plan_path.resolve()),
        "--launch", str(launch.resolve()),
        "--source-sha256", source_hash,
        "--task-kind", spec["task_kind"],
        "--source-key", spec["source_fit"],
    ]
    if spec["budget"] is not None:
        command.extend(["--budget", str(spec["budget"])])
    with log.open("a") as stream:
        process = subprocess.Popen(
            command, env=environment, stdout=stream,
            stderr=subprocess.STDOUT,
            preexec_fn=lambda: os.nice(LOW_CPU_PRIORITY),
        )
    process.log = log
    LOGGER.info(
        "Started %s count=%d budget=%s pid=%d gpu=%d",
        spec["task_kind"], spec["source_count"],
        spec["budget"], process.pid, device["index"],
    )
    return process


def run_queue(
    plan: dict, path: Path, launch: Path, source_hash: str,
) -> None:
    """Schedule ready paths while keeping only four one-GPU tasks live."""
    from experiments.multisession.runner import publish_reports, reconcile

    validate_plan(plan)
    task_spec = task_plan(plan)
    tasks = task_spec["jobs"]
    devices = select_gpus(plan["settings"]["runtime"]["device"], GPU_COUNT)
    atomic_json(launch / "devices.json", devices)
    states = {
        key: (
            "complete" if task_complete(plan, launch, spec) else "pending"
        )
        for key, spec in tasks.items()
    }
    pending = {
        key for key, state in states.items() if state == "pending"
    }
    active = {}
    with writer_lock(path.parent / "budget_coordinator.lock", blocking=False):
        while pending or active:
            for slot, device in enumerate(devices):
                if slot in active:
                    continue
                ready = ready_tasks(
                    plan, launch, tasks, states, pending, active
                )
                if not ready:
                    continue
                key = ready[0]
                active[slot] = (
                    key, start_task(
                        path, launch, source_hash, key, tasks[key], device,
                        plan["settings"]["runtime"]["cpu_threads"],
                    )
                )
                pending.remove(key)
                states[key] = "running"
            for slot, (key, process) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                del active[slot]
                try:
                    valid = task_complete(plan, launch, tasks[key])
                except (ValueError, OSError):
                    LOGGER.exception("Invalid completed task %s", key)
                    valid = False
                states[key] = (
                    "complete" if code == 0 and valid else "failed"
                )
                LOGGER.info(
                    "%s count=%d budget=%s %s exit=%d",
                    tasks[key]["task_kind"], tasks[key]["source_count"],
                    tasks[key]["budget"], states[key], code,
                )
            if pending and not active and not ready_tasks(
                plan, launch, tasks, states, pending, active
            ):
                for key in pending:
                    states[key] = "blocked"
                pending.clear()
            phase = "active" if pending or active else "finished"
            publish_status(
                launch, task_spec, states, pending, active, phase
            )
            if pending or active:
                time.sleep(POLL_SECONDS)
        source_states = {}
        for source in plan["source_references"]:
            completed_fit = completed(fit_directory(plan, source))
            source_states[source] = (
                "complete" if completed_fit is not None else "failed"
            )
        reconcile(plan, path, source_states)
        if all(state == "complete" for state in source_states.values()):
            publish_reports(plan, path)
    LOGGER.info("Budget task states: %s", dict(Counter(states.values())))


def configure_worker(threads: int) -> None:
    """Require one physical GPU and bounded TensorFlow threads."""
    import tensorflow as tf

    devices = tf.config.list_physical_devices("GPU")
    if len(devices) != 1:
        raise RuntimeError(f"Expected one visible GPU, got {len(devices)}.")
    tf.config.experimental.set_memory_growth(devices[0], True)
    tf.config.set_soft_device_placement(False)
    tf.config.threading.set_intra_op_parallelism_threads(threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)


def main() -> None:
    """Detach a budget queue or execute one assigned source task."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--launch", type=Path)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--source-sha256")
    parser.add_argument("--task-kind", choices=TASK_KINDS)
    parser.add_argument("--source-key")
    parser.add_argument("--budget", type=int)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    path = args.plan.resolve()
    plan = json.loads(path.read_text())
    validate_plan(plan)
    if args.task_kind is not None:
        if args.launch is None or args.source_key not in plan["jobs"]:
            parser.error("Worker requires launch and source key.")
        if args.source_sha256 is None:
            parser.error("Worker requires a frozen-source checksum.")
        verify_source_code(args.source_sha256)
        configure_worker(plan["settings"]["runtime"]["cpu_threads"])
        if args.task_kind == "prefix":
            fit_prefixes(plan, args.launch.resolve(), args.source_key)
        elif args.task_kind == "budget":
            if args.budget is None:
                parser.error("Budget worker requires --budget.")
            fit_budget(
                plan, args.launch.resolve(), args.source_key,
                args.budget,
            )
        else:
            finish_source(plan, args.source_key)
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    launch = args.launch or (
        Path(plan["settings"]["paths"]["log_root"]).resolve()
        / PRIMARY / f"behavior_budget_{stamp}_{os.getpid()}"
    )
    launch.mkdir(parents=True, exist_ok=True)
    if args.detach:
        source_hash = freeze_source(launch)
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str((launch / "source").resolve())
        command = [
            sys.executable, "-m",
            "experiments.multisession.behavior_compute_scaling.budget_queue",
            "--plan", str(path), "--launch", str(launch.resolve()),
            "--source-sha256", source_hash,
        ]
        with (launch / "coordinator.log").open("a") as stream:
            process = subprocess.Popen(
                command, env=environment, stdout=stream,
                stderr=subprocess.STDOUT, start_new_session=True,
            )
        atomic_json(launch / "launch.json", dict(
            pid=process.pid, plan=str(path), command=command,
            source_sha256=source_hash,
        ))
        print(f"Launched pid={process.pid}; logs: {launch.resolve()}")
        return
    if args.source_sha256 is None:
        parser.error("Coordinator requires a frozen-source checksum.")
    verify_source_code(args.source_sha256)
    run_queue(plan, path, launch.resolve(), args.source_sha256)


if __name__ == "__main__":
    main()
