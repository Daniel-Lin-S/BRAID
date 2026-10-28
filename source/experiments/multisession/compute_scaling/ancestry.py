"""Restore immutable component checkpoints and count main-prefix FLOPs.

Inputs are checksum-valid component directories with identity, training,
and checkpoint records. Outputs are restored model weights or prefix paths
and arithmetic counts. Source-fit continuation belongs to the campaign.
"""

import json
from pathlib import Path

from experiments.multisession.store import completed, load_pickle

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
