"""Validate and reuse immutable source paths for longer recurrent budgets.

Source continuation.json references the ancestor fit and its checksum map.
Only new paths are stored in budgets.json; inherited_paths resolves earlier
rows without duplicating them. Component main-prefix ancestry supplies exact
Adam/sampling restoration and counts each required training segment once.
"""

import json
from pathlib import Path

from experiments.multisession.store import completed, load_pickle
from experiments.storage.cache import fingerprint


def checked_reference(fits: Path, reference: dict) -> Path:
    """Resolve a completed fit whose payloads match pinned checksums."""
    directory = fits / reference["source_fit"]
    marker = completed(directory)
    if marker is None or marker["checksums"] != reference["checksums"]:
        raise ValueError(
            f"Source ancestor is incomplete or changed: {directory.resolve()}."
        )
    return directory


def inherited_paths(directory: Path) -> list[dict]:
    """Read chronological budget paths through checksum-pinned ancestors."""
    parts = []
    visited = set()
    while True:
        if directory in visited:
            raise ValueError("Cyclic source continuation ancestry.")
        visited.add(directory)
        parts.append(json.loads((directory / "budgets.json").read_text()))
        reference = directory / "continuation.json"
        if not reference.exists():
            break
        directory = checked_reference(
            directory.parent, json.loads(reference.read_text())
        )
    paths = [row for part in reversed(parts) for row in part]
    budgets = [row["budget"] for row in paths]
    if not budgets or budgets != sorted(set(budgets)):
        raise ValueError("Expected distinct increasing source budgets.")
    return paths


def validate_continuation(spec: dict, records: dict, fits: Path) -> Path:
    """Reject a source continuation if scientific inputs or splits differ."""
    directory = checked_reference(fits, spec["continuation"])
    previous = json.loads((directory / "identity.json").read_text())
    for key in (
        "sessions", "sources", "data", "model", "latent", "seed",
        "source_count", "sharing", "objective",
    ):
        if spec[key] != previous[key]:
            raise ValueError(f"Source continuation changed {key}.")
    saved = json.loads((directory / "data.json").read_text())
    if fingerprint(saved) != fingerprint(records):
        raise ValueError("Source continuation changed window splits or scales.")
    return directory


def restore_component(model: object, directory: Path) -> None:
    """Restore a checksum-valid checkpoint without creating a fit."""
    if completed(directory) is None:
        raise ValueError(f"Incomplete component: {directory.resolve()}.")
    model.restore(load_pickle(directory / "checkpoint.pkl")["weights"])


def main_prefix(branch: Path) -> Path:
    """Resolve the constant-rate prefix preceding a saved decay branch."""
    if completed(branch) is None:
        raise ValueError(f"Incomplete decay branch: {branch.resolve()}.")
    identity = json.loads((branch / "identity.json").read_text())
    if not identity["branch"] or not identity["parent"]:
        raise ValueError("Expected a decay branch with a main-prefix parent.")
    parent = branch.parent / identity["parent"]
    if completed(parent) is None:
        raise ValueError(f"Incomplete main prefix: {parent.resolve()}.")
    return parent


def prefix_flops(directory: Path) -> int:
    """Count each required constant-rate prefix segment, excluding branches."""
    total = 0
    visited = set()
    while directory is not None:
        if directory in visited:
            raise ValueError("Cyclic component prefix ancestry.")
        visited.add(directory)
        if completed(directory) is None:
            raise ValueError(f"Incomplete prefix: {directory.resolve()}.")
        identity = json.loads((directory / "identity.json").read_text())
        if identity["branch"]:
            raise ValueError("A main-prefix path cannot include decay work.")
        training = json.loads((directory / "training.json").read_text())
        flops = training["total_flops"]
        if not isinstance(flops, int) or flops <= 0:
            raise ValueError("Expected positive integer prefix training FLOPs.")
        total += flops
        parent = identity["parent"]
        directory = directory.parent / parent if parent else None
    return total
