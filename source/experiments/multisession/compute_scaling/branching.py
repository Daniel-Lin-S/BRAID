"""Resolve source budgets, branch checkpoints, and path arithmetic.

Inputs are balanced per-session windows, native models, and immutable
component directories. Outputs are restored or fitted branch paths,
resolved per-count schedules, and exact FLOP/exposure accounting.
"""

import json
from pathlib import Path

import numpy as np

from experiments.multisession.compute_scaling.ancestry import prefix_flops
from experiments.multisession.compute_scaling.training import fit_segment
from experiments.multisession.store import load_pickle

def resolved_schedule(settings: dict, count: int) -> dict:
    """Resolve per-count updates without changing session-local sampling."""
    result = dict(settings)
    multiplier = settings["update_multipliers"][str(count)]
    result["checkpoints"] = [x * multiplier for x in settings["checkpoints"]]
    result["total_updates"] = [
        x * multiplier for x in settings["total_updates"]
    ]
    result["warmup_updates"] *= multiplier
    batch = count * settings["session_batch_sizes"][str(count)]
    result["fixed_update_cap"] = settings["fixed_exposure"] // batch
    return result


def fit_component_budget(
    model: object, train: list[dict], root: Path, identity: dict,
    settings: dict, seed: int, budget: int,
    devices: tuple[str, str], microbatch: int,
    resume: Path | None = None,
) -> tuple[Path, int]:
    """Restore or train main prefix and one budget-matched decay branch."""
    parent = resume
    total = prefix_flops(resume) if resume is not None else 0
    start = (load_pickle(resume / "checkpoint.pkl")["update"]
             if resume is not None else 0)
    if start >= budget:
        raise ValueError("Continuation budget must exceed its main prefix.")
    for stop in settings["checkpoints"]:
        if stop <= start:
            continue
        if stop > budget:
            break
        directory, _ = fit_segment(
            model, train, root, identity, settings, seed,
            start, stop, parent,
            devices, microbatch,
        )
        parent = directory
        start = stop
        total += load_flops(directory)
    index = settings["checkpoints"].index(budget)
    end = settings["total_updates"][index]
    branch_settings = dict(settings, branch_updates=end - budget)
    branch, _ = fit_segment(
        model, train, root, identity, branch_settings, seed,
        budget, end, parent,
        devices, microbatch, branch=True,
    )
    return branch, total + load_flops(branch)


def load_flops(directory: Path) -> int:
    """Read validated component arithmetic from a completed segment."""
    value = json.loads(
        (directory / "training.json").read_text()
    )["total_flops"]
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"Invalid component FLOPs: {directory.resolve()}.")
    return value


def component_exposure(directory: Path, data: list[dict]) -> list[dict]:
    """Audit draws and distinct training windows along one stage path."""
    counts = [np.zeros(len(row["T"]), dtype=np.int64) for row in data]
    while directory is not None:
        record = json.loads((directory / "training.json").read_text())
        for row in record["history"]:
            for values, selected in zip(counts, row["sampled_windows"]):
                np.add.at(values, selected, 1)
        identity = json.loads((directory / "identity.json").read_text())
        parent = identity["parent"]
        directory = directory.parent / parent if parent else None
    return [dict(
        available_windows=len(values), total_draws=int(values.sum()),
        distinct_windows=int(np.count_nonzero(values)),
        mean_repetitions=float(values.mean()),
        maximum_repetitions=int(values.max()),
    ) for values in counts]
