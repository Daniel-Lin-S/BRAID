"""Schedule independent two-GPU source fits and ready one-GPU targets.

Inputs are an immutable multi-session source/target plan and standard launch directory.
Outputs are the common status.json, job logs and reports. Source jobs reserve
two physical GPUs; dependent target jobs can start on freed GPUs immediately.
Completed checksum-valid fits are reused, with no duplicate worker launch.
"""

from collections import Counter
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.execution.runtime import select_gpus, thread_environment
from experiments.multisession.log_paths import job_label, job_log_path
from experiments.multisession.queue import adopt_workers
from experiments.multisession.queue_status import publish_status
from experiments.multisession.store import completed, fit_directory
from experiments.storage.cache import atomic_json, writer_lock

LOGGER = logging.getLogger(__name__)
GPU_COUNT = 8
SOURCE_GPU_COUNT = 2
POLL_SECONDS = 5
LOW_CPU_PRIORITY = 10
MINIMUM_FREE_MEMORY_MIB = 4096


def ordered_jobs(plan: dict) -> list[str]:
    """Prioritize independent sources then dependency-ready target fits."""
    rank = dict(source=0, preprocessing=1, adaptation=2, scratch=3)
    return sorted(
        plan["jobs"],
        key=lambda key: (
            rank[plan["jobs"][key]["kind"]],
            plan["jobs"][key].get("source_count", 0),
            plan["jobs"][key].get("session", ""),
            plan["jobs"][key].get("fold", -1),
            plan["jobs"][key].get("fraction", 0),
            key,
        ),
    )


def hold_source_slot(
    spec: dict, free_slots: int, source_ready: bool,
    adaptation_started: set[int],
) -> bool:
    """Reserve the last GPU for a waiting two-GPU source fit."""
    if not source_ready or free_slots >= SOURCE_GPU_COUNT:
        return False
    return (
        spec["kind"] != "adaptation"
        or spec["source_count"] in adaptation_started
    )


def start_worker(
    plan: dict, path: Path, launch: Path, key: str,
    devices: list[dict], threads: int, source_sha256: str | None,
) -> subprocess.Popen:
    """Launch one standard checksum-validating fit worker."""
    spec = plan["jobs"][key]
    log = job_log_path(launch, key, spec)
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(thread_environment(threads, 1))
    env["CUDA_VISIBLE_DEVICES"] = ",".join(
        device["uuid"] for device in devices
    )
    env["TF_USE_LEGACY_KERAS"] = "1"
    env["MPLCONFIGDIR"] = str((launch / "mpl_cache").resolve())
    command = [
        sys.executable, "-m", "experiments.multisession.runner",
        "--stage", "worker", "--plan", str(path.resolve()),
        "--job", key, "--threads", str(threads),
    ]
    if source_sha256 is not None:
        command.extend(["--source-sha256", source_sha256])
    with log.open("a") as stream:
        process = subprocess.Popen(
            command, env=env, stdout=stream,
            stderr=subprocess.STDOUT,
            preexec_fn=lambda: os.nice(LOW_CPU_PRIORITY),
        )
    process.log = log
    LOGGER.info(
        "Started %s pid=%d GPUs=%s log=%s",
        job_label(key, spec), process.pid,
        [device["index"] for device in devices], log.resolve(),
    )
    return process


def runnable(
    plan: dict, states: dict[str, str], pending: set[str],
) -> list[str]:
    """List jobs whose entire dependency path is complete."""
    return [
        key for key in ordered_jobs(plan)
        if key in pending
        and all(
            states[parent] == "complete"
            for parent in plan["jobs"][key]["dependencies"]
        )
    ]


def run_compute_queue(
    plan: dict, path: Path, launch: Path, settings: dict,
    source_sha256: str | None,
    previous_launch: Path | None = None,
) -> None:
    """Run checksum-aware source/target queue over eight GPU slots."""
    from experiments.multisession.runner import (
        publish_reports, reconcile,
    )

    if settings["runtime"].get("gpus_per_source") != SOURCE_GPU_COUNT:
        raise ValueError("Compute-scaling source requires two GPUs.")
    if previous_launch is None:
        devices = select_gpus(settings["runtime"]["device"], GPU_COUNT)
        active = {}
        allocations = {}
    else:
        devices, active = adopt_workers(plan, previous_launch)
        saved = json.loads(
            (previous_launch / "allocations.json").read_text()
        )
        allocations = {
            key: saved[key] for key, _ in active.values()
        }
        used = [slot for slots in allocations.values() for slot in slots]
        if len(used) != len(set(used)):
            raise ValueError("Adopted workers overlap GPU slots.")
        if any(
            len(allocations[key]) != (
                SOURCE_GPU_COUNT
                if plan["jobs"][key]["kind"] == "source" else 1
            ) for key in allocations
        ):
            raise ValueError("Adopted GPU allocation differs by fit kind.")
    atomic_json(launch / "devices.json", devices)
    atomic_json(launch / "allocations.json", allocations)
    states = reconcile(plan, path)
    pending = {
        key for key, state in states.items() if state != "complete"
    }
    for key in pending:
        states[key] = "pending"
    for key, _ in active.values():
        pending.discard(key)
        states[key] = "running"
    failed = set()
    source_reported = False
    adaptation_started = {
        plan["jobs"][key]["source_count"]
        for key, state in states.items()
        if plan["jobs"][key]["kind"] == "adaptation"
        and state in ("complete", "running")
    }
    with writer_lock(path.parent / "coordinator.lock", blocking=False):
        while pending or active:
            current = {
                device["uuid"]: device for device in select_gpus(
                    settings["runtime"]["device"], GPU_COUNT
                )
            }
            occupied = {
                slot for slots in allocations.values()
                for slot in slots
            }
            free = [
                index for index, device in enumerate(devices)
                if index not in occupied
                and current[device["uuid"]]["free_memory_mib"]
                >= MINIMUM_FREE_MEMORY_MIB
            ]
            ready = runnable(plan, states, pending)
            source_ready = {
                key for key in ready
                if plan["jobs"][key]["kind"] == "source"
            }
            for key in ready:
                spec = plan["jobs"][key]
                slots_needed = (
                    SOURCE_GPU_COUNT if spec["kind"] == "source" else 1
                )
                if len(free) < slots_needed or hold_source_slot(
                    spec, len(free), bool(source_ready),
                    adaptation_started,
                ):
                    continue
                slots = free[:slots_needed]
                free = free[slots_needed:]
                process = start_worker(
                    plan, path, launch, key,
                    [devices[index] for index in slots],
                    settings["runtime"]["cpu_threads"], source_sha256,
                )
                active[slots[0]] = (key, process)
                allocations[key] = slots
                pending.remove(key)
                states[key] = "running"
                if spec["kind"] == "source":
                    source_ready.remove(key)
                elif spec["kind"] == "adaptation":
                    adaptation_started.add(spec["source_count"])
                atomic_json(launch / "allocations.json", allocations)
            changed = False
            for slot, (key, process) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                del active[slot]
                del allocations[key]
                atomic_json(launch / "allocations.json", allocations)
                try:
                    valid = completed(fit_directory(plan, key))
                except (ValueError, OSError):
                    LOGGER.exception("Invalid completed fit %s", key)
                    valid = None
                states[key] = (
                    "complete" if code == 0 and valid else "failed"
                )
                if states[key] == "failed":
                    failed.add(key)
                LOGGER.info(
                    "%s %s exit=%d", job_label(key, plan["jobs"][key]),
                    states[key], code,
                )
                changed = True
            blocked = {
                key for key in pending
                if any(
                    parent in failed
                    for parent in plan["jobs"][key]["dependencies"]
                )
            }
            for key in blocked:
                pending.remove(key)
                failed.add(key)
                states[key] = "blocked"
            publish_status(
                launch, plan, states, pending, active, "active"
            )
            if changed:
                reconcile(plan, path, states)
            if not source_reported and all(
                states[key] == "complete" for key in plan["source_references"]
            ):
                publish_reports(plan, path)
                source_reported = True
            if pending and not active and not runnable(
                plan, states, pending
            ):
                raise RuntimeError(
                    "No runnable jobs remain; inspect source failures."
                )
            if pending or active:
                time.sleep(POLL_SECONDS)
        reconcile(plan, path, states)
        publish_reports(plan, path)
        publish_status(
            launch, plan, states, pending, active, "finished"
        )
    LOGGER.info("Final fit states: %s", dict(Counter(states.values())))
