"""Publish experiment membership separately from immutable shared fits.

analysis/<name>/<analysis-id>/manifest.json holds a specification and members
keyed by case/session/fold. Each member references its fit and prediction
bundle, with relative metrics paths and checksums when evaluation completes.
Completed member records and metric rows are never overwritten.
"""

import copy
import json
import logging
from pathlib import Path
import uuid

from .artifacts import scientific_source, validate_completion
from .cache import atomic_json, file_digest, fingerprint, writer_lock

ANALYSIS_IMPLEMENTATION = "analysis_implementation"
ANALYSIS_SCHEMA = 1
LOGGER = logging.getLogger(__name__)
MISSING_SETTING = "<missing>"


def analysis_plotting(settings: dict) -> dict:
    """Remove preview invocation controls from analysis rendering metadata."""
    result = copy.deepcopy(settings)
    result.pop("previews", None)
    return result


def analysis_settings(snapshots: dict) -> dict:
    """Extract evaluation/report specifications without invocation settings."""
    experiment = snapshots["experiment"]
    return dict(
        schema=ANALYSIS_SCHEMA, name=experiment["name"],
        suite=experiment["suite"], evaluation=snapshots["evaluation"],
        data=scientific_source(snapshots["data"]),
        model=snapshots["model"], seed=experiment["seed"],
        model_plugin=experiment["model_plugin"],
        report_plugin=experiment.get("report_plugin"),
        analysis_implementation=snapshots.get("analysis_implementation"),
    )


def member_key(case: dict, session: str, fold: int) -> str:
    """Return readable experiment membership, not another model identity."""
    name = case["name"]
    for value in (name, session):
        if Path(value).name != value or value in (".", ".."):
            raise ValueError(f"Invalid artifact directory component: {value}")
    return f"{name}/{session}/fold_{fold}"


def initialize_analysis(
    root: Path, snapshots: dict, members: dict[str, dict],
) -> Path:
    """Publish a specification-addressed manifest of all requested fits."""
    if not members:
        raise ValueError("Cannot publish analysis with no requested fits.")
    spec = dict(settings=analysis_settings(snapshots), members=members)
    directory = (
        root / "analysis" / snapshots["experiment"]["name"]
        / fingerprint(spec)
    ).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "manifest.json"
    with writer_lock(directory / "manifest.lock"):
        if path.exists():
            if json.loads(path.read_text())["specification"] != spec:
                raise ValueError(f"Analysis specification mismatch: {path}")
            manifest = json.loads(path.read_text())
        else:
            manifest = dict(
                specification=spec, members=copy.deepcopy(members),
            )
        rendering = dict(
            plotting=analysis_plotting(snapshots["plotting"])
        )
        if manifest.get("rendering") != rendering:
            manifest["rendering"] = rendering
            atomic_json(
                path, manifest,
            )
    return directory


def read_manifest(directory: Path) -> dict:
    """Read and validate the immutable analysis specification."""
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    if fingerprint(manifest["specification"]) != directory.name:
        raise ValueError(f"Invalid analysis specification: {path}")
    expected = manifest["specification"]["members"]
    if set(manifest["members"]) != set(expected):
        raise ValueError(
            f"Analysis membership differs from specification: {path}"
        )
    for key, original in expected.items():
        member = manifest["members"][key]
        if any(
            member.get(field) != value
            for field, value in original.items() if field != "state"
        ):
            raise ValueError(f"Analysis member provenance changed: {key}")
    return manifest


def record_plotting(directory: Path, settings: dict) -> None:
    """Record current plot settings without changing scientific membership."""
    with writer_lock(directory / "manifest.lock"):
        manifest = read_manifest(directory)
        rendering = manifest.setdefault("rendering", {})
        current = analysis_plotting(settings)
        if rendering.get("plotting") != current:
            rendering["plotting"] = current
            atomic_json(directory / "manifest.json", manifest)


def prepare_metrics(directory: Path, key: str) -> Path:
    """Quarantine an unpublished metrics payload before retrying evaluation.

    The caller holds this member's evaluation lock and has verified that the
    member is incomplete. Completed metrics must never enter this function.
    """
    destination = directory / "metrics" / key
    path = destination / "metrics.json"
    if path.exists():
        quarantine = directory / "quarantine" / key / uuid.uuid4().hex
        quarantine.mkdir(parents=True)
        path.rename(quarantine / path.name)
        LOGGER.warning("Quarantined unpublished metrics at %s", quarantine)
    return destination


def update_member(directory: Path, key: str, values: dict) -> None:
    """Append completion or incomplete-state evidence under a writer lock."""
    with writer_lock(directory / "manifest.lock"):
        manifest = read_manifest(directory)
        member = manifest["members"][key]
        if member.get("state") == "complete":
            raise ValueError(
                f"Cannot overwrite completed analysis member: {key}"
            )
        member.update(values)
        atomic_json(directory / "manifest.json", manifest)


def validate_member(directory: Path, member: dict) -> Path:
    """Validate a completed metrics payload and return its absolute path."""
    path = (directory / member["metrics"]).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError(f"Metrics path is outside its analysis: {path}")
    if file_digest(path) != member["metrics_sha256"]:
        raise ValueError(f"Completed metrics checksum mismatch: {path}")
    root = directory.parents[2]
    fit = (root / member["fit"]).resolve()
    if not fit.is_relative_to(root / "experiments"):
        raise ValueError(f"Fit reference is outside shared experiments: {fit}")
    if fit.name != member["fit_id"]:
        raise ValueError(f"Fit reference does not match its ID: {fit}")
    validate_completion(fit)
    predictions = (root / member["predictions"]).resolve()
    if not predictions.is_relative_to(fit / "predictions"):
        raise ValueError(
            f"Prediction reference is outside its fit: {predictions}"
        )
    if file_digest(predictions) != member["predictions_sha256"]:
        raise ValueError(f"Prediction checksum mismatch: {predictions}")
    return path


def setting_differences(
    saved: object, current: object, prefix: str = "",
) -> list[tuple[str, object, object]]:
    """Return differing leaf parameters between saved and current settings.

    Parameters
    ----------
    saved : object
        JSON-compatible settings stored in an analysis manifest.
    current : object
        JSON-compatible settings resolved for the current invocation.
    prefix : str, optional
        Dotted parent path used during recursive comparison, by default "".

    Returns
    -------
    list of tuple
        Dotted parameter path, saved value and current value for each
        difference.
    """
    if not isinstance(saved, dict) or not isinstance(current, dict):
        return [] if saved == current else [(prefix, saved, current)]
    differences = []
    for key in sorted(set(saved) | set(current)):
        path = f"{prefix}.{key}" if prefix else key
        if key not in saved:
            differences.append((path, MISSING_SETTING, current[key]))
        elif key not in current:
            differences.append((path, saved[key], MISSING_SETTING))
        else:
            differences.extend(
                setting_differences(saved[key], current[key], path)
            )
    return differences


def format_setting_difference(
    difference: tuple[str, object, object],
) -> str:
    """Format one saved-versus-current parameter difference."""
    path, saved, current = difference
    return (
        f"{path}: saved={json.dumps(saved, sort_keys=True)}, "
        f"current={json.dumps(current, sort_keys=True)}"
    )


def warn_implementation_mismatch(
    directory: Path, saved: dict, current: dict,
) -> None:
    """Warn when plotting metrics produced by another implementation."""
    recorded = saved.get(ANALYSIS_IMPLEMENTATION)
    expected = current.get(ANALYSIS_IMPLEMENTATION)
    if recorded == expected:
        return
    LOGGER.warning(
        "Plotting saved analysis %s across an analysis implementation "
        "mismatch: saved=%r, current=%r. Saved metrics will not be rescored.",
        directory.name,
        recorded,
        expected,
    )


def find_analysis(
    root: Path, snapshots: dict, analysis_id: str | None = None,
) -> Path:
    """Resolve saved analysis for plotting without loading data or fitting."""
    parent = root / "analysis" / snapshots["experiment"]["name"]
    paths = sorted(parent.glob("*/manifest.json"))
    if not paths:
        raise ValueError(
            "No saved analysis folders containing manifest.json under "
            f"{parent.resolve()}."
        )
    current = analysis_settings(snapshots)
    if analysis_id:
        selected = [
            path.parent for path in paths if path.parent.name == analysis_id
        ]
        if not selected:
            raise ValueError(
                f"Saved analysis --analysis-id {analysis_id!r} was not found "
                f"under {parent.resolve()}."
            )
        directory = selected[0]
        saved = read_manifest(directory)["specification"]["settings"]
        warn_implementation_mismatch(directory, saved, current)
        return directory
    exact = []
    compatible = []
    mismatches = []
    for path in paths:
        saved = read_manifest(path.parent)["specification"]["settings"]
        differences = setting_differences(saved, current)
        if not differences:
            exact.append(path.parent)
        elif all(
            difference[0] == ANALYSIS_IMPLEMENTATION
            for difference in differences
        ):
            compatible.append((path.parent, saved))
        else:
            mismatches.append((len(differences), path.parent, differences))
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        identifiers = ", ".join(path.name for path in exact)
        raise ValueError(
            f"Multiple compatible saved analyses under {parent.resolve()}: "
            f"{identifiers}. Select one with --analysis-id."
        )
    if len(compatible) == 1:
        directory, saved = compatible[0]
        warn_implementation_mismatch(directory, saved, current)
        return directory
    if len(compatible) > 1:
        identifiers = ", ".join(
            directory.name for directory, _ in compatible
        )
        raise ValueError(
            "Multiple plot-compatible saved analyses with different analysis "
            f"implementations under {parent.resolve()}: {identifiers}. "
            "Select one with --analysis-id."
        )
    _, closest, differences = min(
        mismatches, key=lambda item: (item[0], item[1].name),
    )
    details = "; ".join(
        format_setting_difference(difference)
        for difference in differences
    )
    raise ValueError(
        f"Saved analyses exist under {parent.resolve()}, but none match the "
        f"current settings. Closest analysis {closest.name} differs at: "
        f"{details}. To plot it, pass --analysis-id {closest.name}."
    )
