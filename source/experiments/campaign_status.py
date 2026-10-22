"""Track checkpointed campaign jobs without model-specific imports.

Inputs are a StatusPlan of preprocessing paths, fit paths, reuse rules, and
coordinator module name. Job directories contain status.json and complete.json;
launch.json contains process metadata. Outputs are live counts in launch.json
with artifact references resolved by the supplied plan. Status monitoring
never moves checkpoints, figures, or TensorBoard streams.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
from importlib import import_module
import json
from pathlib import Path
from threading import Event
import time

from experiments.storage.cache import atomic_json


POLL_SECONDS = 15
STATES = ("finished", "running", "interrupted", "failed", "pending")


@dataclass(frozen=True)
class FitJob:
    """One fit directory or a validated preprocessing-backed reuse."""

    directory: Path | None
    preprocessing_index: int | None = None
    verified_reuse: bool = False
    reused: bool = False


@dataclass(frozen=True)
class StatusPlan:
    """Paths and process identity for one model-independent campaign."""

    preprocessing: tuple[Path, ...]
    fits: tuple[FitJob, ...]
    report_marker: Path
    coordinator_module: str

    def __post_init__(self) -> None:
        if not self.preprocessing or not self.fits:
            raise ValueError("Campaign status needs nonempty job sets.")
        if not self.coordinator_module:
            raise ValueError("Coordinator module cannot be empty.")
        for fit in self.fits:
            if fit.directory is None:
                index = fit.preprocessing_index
                if index is None or not 0 <= index < len(self.preprocessing):
                    raise ValueError(
                        "Reused fit needs a valid preprocessing index."
                    )


def writer_active(directory: Path) -> bool:
    """Check whether a worker holds this job's sibling lock."""
    path = directory.with_suffix(".lock")
    if not path.is_file():
        return False
    with path.open("r") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    return False


def job_state(
    directory: Path, active_since_ns: int | None = None,
) -> str:
    """Read a job marker, checking live locks for running records."""
    if (directory / "complete.json").is_file():
        return "finished"
    status = directory / "status.json"
    if not status.is_file():
        return "pending"
    state = json.loads(status.read_text()).get("state")
    if state not in ("running", "failed"):
        raise ValueError(
            f"Expected running or failed in {status.resolve()}, "
            f"got {state!r} without complete.json."
        )
    if state == "running" and active_since_ns is not None:
        if (
            status.stat().st_mtime_ns < active_since_ns
            or not writer_active(directory)
        ):
            return "interrupted"
    return state


def count_states(states: list[str]) -> dict[str, int]:
    """Count every planned job and its remaining completion work."""
    totals = Counter(states)
    unknown = set(totals) - set(STATES)
    if unknown:
        raise ValueError(f"Unexpected campaign states: {sorted(unknown)}.")
    result = {state: totals[state] for state in STATES}
    result["total"] = len(states)
    result["remaining"] = len(states) - result["finished"]
    return result


def snapshot(
    root: Path, plan: StatusPlan, active: bool | None = None,
    active_since_ns: int | None = None,
) -> dict:
    """Summarize all jobs in a campaign plan.

    Parameters
    ----------
    root : Path
        Campaign directory containing launch.json.
    plan : StatusPlan
        Explicit preprocessing and fit jobs.
    active : bool, optional
        Coordinator activity. Default is None.
    active_since_ns : int, optional
        Coordinator start time in nanoseconds. Default is None.

    Returns
    -------
    dict
        Timestamped preprocessing and fit counts.
    """
    prep = [
        job_state(directory, active_since_ns)
        for directory in plan.preprocessing
    ]
    fit_states = []
    reused_finished = 0
    for fit in plan.fits:
        if fit.directory is None:
            state = (
                "finished" if fit.verified_reuse
                or prep[fit.preprocessing_index] == "finished"
                else "pending"
            )
        else:
            state = job_state(fit.directory, active_since_ns)
        fit_states.append(state)
        reused_finished += (
            state == "finished"
            and (fit.reused or (
                fit.directory is not None and fit.directory.is_symlink()
            ))
        )
    if active is False:
        prep = [
            "interrupted" if state == "running" else state
            for state in prep
        ]
        fit_states = [
            "interrupted" if state == "running" else state
            for state in fit_states
        ]
    fits = count_states(fit_states)
    fits["reused_finished"] = reused_finished
    fits["local_finished"] = fits["finished"] - reused_finished
    return {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "preprocessing": count_states(prep),
        "fits": fits,
        "report_finished": plan.report_marker.is_file(),
    }


def _process_active(record: dict, module: str) -> bool:
    """Check whether the recorded coordinator is visible and alive."""
    for name in ("pid", "first_fit_pid"):
        pid = record.get(name)
        if not isinstance(pid, int) or pid < 1:
            continue
        try:
            command = Path(f"/proc/{pid}/cmdline").read_bytes()
        except (FileNotFoundError, ProcessLookupError):
            continue
        if module.encode() in command:
            return True
    return False


def _process_start_ns(record: dict) -> int:
    """Return the live coordinator start time in nanoseconds."""
    pid = record.get("pid")
    if not isinstance(pid, int) or pid < 1:
        raise ValueError("Active campaign needs a valid coordinator PID.")
    try:
        return Path(f"/proc/{pid}").stat().st_ctime_ns
    except (FileNotFoundError, ProcessLookupError) as error:
        raise RuntimeError(
            f"Cannot read start time for campaign PID {pid}."
        ) from error


def _any_writer_active(plan: StatusPlan) -> bool:
    """Detect live work when its coordinator PID is hidden."""
    paths = list(plan.preprocessing)
    paths += [
        fit.directory for fit in plan.fits
        if fit.directory is not None
    ]
    return any(writer_active(path) for path in paths)


def refresh(root: Path, plan: StatusPlan) -> dict:
    """Atomically add current planned-job counts to launch.json."""
    path = root / "launch.json"
    record = json.loads(path.read_text()) if path.is_file() else {}
    visible = (
        "ended_at_utc" not in record
        and _process_active(record, plan.coordinator_module)
    )
    active = visible or _any_writer_active(plan)
    active_since_ns = (
        _process_start_ns(record) if visible
        else 0 if active else None
    )
    record.update(snapshot(
        root, plan, active=active, active_since_ns=active_since_ns
    ))
    record["campaign_active"] = active
    atomic_json(path, record)
    return record


def watch(
    root: Path, plan: StatusPlan, stop: Event | None = None,
) -> None:
    """Refresh status until the coordinator exits or stop is set."""
    while True:
        record = refresh(root, plan)
        if stop is None and not record["campaign_active"]:
            return
        if stop is None:
            time.sleep(POLL_SECONDS)
        elif stop.wait(POLL_SECONDS):
            refresh(root, plan)
            return


def main() -> None:
    """Refresh or watch a study's generic campaign status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--watch", action="store_true")
    arguments = parser.parse_args()
    root = arguments.root.resolve()
    study = import_module(arguments.study)
    launch = root / "launch.json"
    record = json.loads(launch.read_text()) if launch.is_file() else {}
    keys = record.get("reused_completed_keys")
    reused = set(keys) if keys is not None else None
    plan = study.status_plan(root, reused)
    if arguments.watch:
        watch(root, plan)
    else:
        print(json.dumps(refresh(root, plan), indent=2))


if __name__ == "__main__":
    main()
