"""Adopt verified workers and avoid shared-source lock contention.

Inputs are an earlier launch's status, GPU records, and immutable plan.
Only stopped or exited coordinators may hand workers to a successor. Worker
PID start times guard against PID reuse; completion records decide success.
"""

import json
from pathlib import Path

from experiments.multisession.store import fit_directory


def process_state(pid: int) -> tuple[str, str] | None:
    """Return Linux process state/start ticks, or None after process exit."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except FileNotFoundError:
        return None
    return fields[0], fields[19]


class AdoptedWorker:
    """Observe an existing worker without signaling or restarting it."""

    def __init__(self, pid: int, plan: dict, key: str) -> None:
        self.pid = pid
        self.directory = fit_directory(plan, key)
        state = process_state(pid)
        self.started = state[1] if state is not None else None

    def poll(self) -> int | None:
        """Return completion-derived exit status once this PID has exited."""
        state = process_state(self.pid)
        if state and state[0] != "Z" and state[1] == self.started:
            return None
        return 0 if (self.directory / "complete.json").is_file() else 1


def adopt_workers(plan: dict, launch: Path) -> tuple[list, dict]:
    """Validate source plan, process ownership, and checkpoint locations."""
    status = json.loads((launch / "status.json").read_text())
    coordinator = process_state(status["coordinator_pid"])
    if coordinator and coordinator[0] not in ("T", "t", "Z"):
        raise RuntimeError("Previous coordinator must be stopped or exited.")
    record = json.loads((launch / "launch.json").read_text())
    old_path = Path(record["plan"])
    previous = json.loads(old_path.read_text())
    devices = json.loads((launch / "devices.json").read_text())
    active = {}
    for slot, worker in status["active"].items():
        key, pid = worker["job"], worker["pid"]
        if plan["jobs"].get(key) != previous["jobs"][key]:
            raise ValueError(f"Cannot adopt changed or excluded fit {key}.")
        if fit_directory(plan, key) != fit_directory(previous, key):
            raise ValueError(f"Cannot relocate active fit {key}.")
        state = process_state(pid)
        if state and state[0] != "Z":
            command = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
            words = [word.decode() for word in command if word]
            if (
                "experiments.multisession.runner" not in words
                or words[words.index("--job") + 1] != key
                or Path(words[words.index("--plan") + 1]) != old_path
            ):
                raise ValueError(f"PID {pid} is not the expected fit worker.")
        process = AdoptedWorker(pid, plan, key)
        process.log = Path(worker["log"])
        active[int(slot)] = (key, process)
    return devices, active


def source_resource(spec: dict) -> tuple | None:
    """Group supervised sources sharing expensive private preprocessing."""
    if spec["kind"] == "source" and spec.get("objective") == "behavior":
        return spec["seed"], tuple(spec["sessions"])
    return None


def resource_available(key: str, jobs: dict, active: dict) -> bool:
    """Keep shared-component waiters out of allocated GPU worker slots."""
    resource = source_resource(jobs[key])
    return resource is None or all(
        source_resource(jobs[other]) != resource
        for other, _ in active.values()
    )
