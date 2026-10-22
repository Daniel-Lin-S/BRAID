"""Connect compact launch logs to the shared analysis and fit stores.

launch.json contains artifact_root and analysis_directory. The referenced
analysis manifest uses the shared specification/members format; each member
has fit, fit_id, role and reused fields. Fit references are relative to the
artifact root. No model payloads or source snapshots belong in launch logs.
"""

from __future__ import annotations

import json
from pathlib import Path

from experiments.storage.analysis import read_manifest
from experiments.storage.cache import atomic_json


def validate_roots(paths: dict) -> None:
    """Reject overlapping log and artifact/cache roots before any writes.

    Parameters
    ----------
    paths : dict
        Explicit log_root, artifact_root and cache_root locations.
    """
    logs = Path(paths["log_root"]).resolve()
    for name in ("artifact_root", "cache_root"):
        root = Path(paths[name]).resolve()
        if root.is_relative_to(logs) or logs.is_relative_to(root):
            raise ValueError(
                f"Expected separate {name} and log_root; got "
                f"{root} and {logs}."
            )


def bind_analysis(root: Path, paths: dict, analysis: Path) -> None:
    """Add shared artifact references to a launch without changing its fits.

    Parameters
    ----------
    root : Path
        Launch log directory beneath the configured log_root.
    paths : dict
        User-configured storage roots.
    analysis : Path
        Existing shared analysis directory with a validated manifest.
    """
    validate_roots(paths)
    if not root.resolve().is_relative_to(Path(paths["log_root"]).resolve()):
        raise ValueError(f"Launch is outside log_root: {root.resolve()}.")
    artifact = Path(paths["artifact_root"]).resolve()
    if not analysis.resolve().is_relative_to(artifact / "analysis"):
        raise ValueError(f"Analysis is outside artifact root: {analysis}.")
    read_manifest(analysis)
    path = root / "launch.json"
    record = json.loads(path.read_text()) if path.exists() else {}
    values = dict(
        artifact_root=str(artifact), analysis_directory=str(analysis.resolve())
    )
    for key, value in values.items():
        if key in record and record[key] != value:
            raise ValueError(f"Launch {key} differs at {path.resolve()}.")
    record.update(values)
    atomic_json(path, record)


def campaign_analysis(root: Path) -> Path:
    """Resolve and validate a launch's shared analysis reference."""
    path = root / "launch.json"
    record = json.loads(path.read_text()) if path.exists() else {}
    if not {"artifact_root", "analysis_directory"} <= record.keys():
        raise ValueError(
            f"Missing shared artifact references in {path.resolve()}; "
            "initialize storage or migrate existing artifacts first."
        )
    artifact = Path(record["artifact_root"]).resolve()
    analysis = Path(record["analysis_directory"]).resolve()
    if (
        not analysis.is_relative_to(artifact / "analysis")
        or analysis.is_relative_to(root.resolve())
    ):
        raise ValueError(f"Invalid analysis reference: {analysis}.")
    read_manifest(analysis)
    return analysis


def campaign_member(root: Path, key: str) -> tuple[Path, dict]:
    """Return a member's shared fit path and immutable membership fields."""
    analysis = campaign_analysis(root)
    artifact = analysis.parents[2]
    members = read_manifest(analysis)["members"]
    if key not in members:
        raise ValueError(f"Unknown analysis member {key!r} in {analysis}.")
    member = members[key]
    fit = (artifact / member["fit"]).resolve()
    if not fit.is_relative_to(artifact / "experiments"):
        raise ValueError(f"Fit reference leaves shared experiments: {fit}.")
    if fit.name != member["fit_id"]:
        raise ValueError(f"Fit reference has inconsistent ID: {fit}.")
    return fit, member
