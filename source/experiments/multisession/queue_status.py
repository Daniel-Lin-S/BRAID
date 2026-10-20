"""Publish and inspect live multi-session campaign progress.

Input is a coordinator launch with status.json, launch.json, and worker PIDs.
Output is a liveness-checked progress mapping. Saved counts are snapshots;
only verified live PIDs are reported as currently running.
"""

from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from experiments.storage.cache import atomic_json
from experiments.multisession.log_paths import job_label
from experiments.multisession.queue import process_state


def publish_status(
    launch: Path,
    plan: dict,
    states: dict[str, str],
    pending: set[str],
    active: dict,
    phase: str,
) -> None:
    """Write durable queue counts and PID identities after each update.

    Parameters
    ----------
    launch : Path
        Absolute campaign log directory.
    plan : dict
        Immutable job definitions keyed by fit ID.
    states : dict of str to str
        Most recently reconciled queue states.
    pending : set of str
        Unstarted jobs, including queued retries.
    active : dict
        Worker slot to (job ID, process) mapping.
    phase : str
        Active or finished coordinator phase.
    """
    if phase not in ("active", "finished"):
        raise ValueError(f"Unknown coordinator phase: {phase}.")
    counts = Counter(states.values())
    summary = dict(
        total=len(plan["jobs"]),
        finished=counts["complete"] + counts["failed"]
        + counts["blocked"],
        complete=counts["complete"],
        failed=counts["failed"],
        blocked=counts["blocked"],
        running=len(active),
        remaining=len(pending),
    )
    accounted = sum(
        summary[key]
        for key in ("complete", "failed", "blocked", "running", "remaining")
    )
    if accounted != summary["total"]:
        raise ValueError(f"Queue counts do not cover {summary['total']} jobs.")
    workers = {}
    for slot, (key, process) in active.items():
        state = process_state(process.pid)
        workers[str(slot)] = dict(
            job=key, pid=process.pid,
            start_ticks=state[1] if state is not None else None,
            label=job_label(key, plan["jobs"][key]),
            log=str(process.log),
        )
    coordinator = process_state(os.getpid())
    stamp = datetime.now(timezone.utc).isoformat()
    status = dict(
        phase=phase,
        updated_at_utc=stamp,
        coordinator_pid=os.getpid(),
        coordinator_start_ticks=(
            coordinator[1] if coordinator is not None else None
        ),
        counts=summary,
        active=workers,
        failures=sorted(
            key for key, state in states.items()
            if state in ("failed", "blocked")
        ),
    )
    atomic_json(launch / "status.json", status)
    record = launch / "launch.json"
    if record.is_file():
        metadata = json.loads(record.read_text())
        metadata.update(counts=summary, updated_at_utc=stamp)
        atomic_json(record, metadata)


def inspect_status(launch: Path) -> dict:
    """Return saved counts with current coordinator and worker liveness."""
    status = json.loads((launch / "status.json").read_text())
    coordinator = process_state(status["coordinator_pid"])
    live = (
        coordinator is not None
        and coordinator[0] not in ("T", "t", "Z")
        and coordinator[1] == status.get("coordinator_start_ticks")
    )
    active = {}
    for slot, worker in status.get("active", {}).items():
        state = process_state(worker["pid"])
        if (
            state is not None
            and state[0] not in ("T", "t", "Z")
            and state[1] == worker.get("start_ticks")
        ):
            active[slot] = worker
    counts = dict(status["counts"])
    counts.setdefault("blocked", 0)
    counts.setdefault(
        "total", counts["complete"] + counts["failed"]
        + counts["blocked"] + len(status.get("active", {}))
    )
    counts.setdefault(
        "remaining", counts["total"] - counts["complete"]
        - counts["failed"] - counts["blocked"]
        - len(status.get("active", {}))
    )
    counts["running"] = len(active)
    counts["remaining"] += len(status.get("active", {})) - len(active)
    counts["finished"] = (
        counts["complete"] + counts["failed"] + counts["blocked"]
    )
    return dict(
        phase=status.get("phase", "active") if live else "stopped",
        snapshot_at_utc=status.get("updated_at_utc"),
        coordinator_live=live,
        counts=counts,
        active=active,
    )
