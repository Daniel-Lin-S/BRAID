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

ANALYSIS_SCHEMA = 1
LOGGER = logging.getLogger(__name__)


def analysis_settings(snapshots: dict) -> dict:
    """Extract evaluation/report specifications without invocation settings."""
    experiment = snapshots["experiment"]
    return dict(
        schema=ANALYSIS_SCHEMA, name=experiment["name"],
        suite=experiment["suite"], evaluation=snapshots["evaluation"],
        plotting=snapshots["plotting"],
        previews=snapshots["data"].get("previews"),
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
        else:
            atomic_json(
                path,
                dict(specification=spec, members=copy.deepcopy(members)),
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


def find_analysis(
    root: Path, snapshots: dict, analysis_id: str | None = None,
) -> Path:
    """Resolve saved analysis for plotting without loading data or fitting."""
    parent = root / "analysis" / snapshots["experiment"]["name"]
    candidates = []
    for path in parent.glob("*/manifest.json"):
        if analysis_id and path.parent.name != analysis_id:
            continue
        manifest = read_manifest(path.parent)
        if analysis_id or (
            manifest["specification"]["settings"]
            == analysis_settings(snapshots)
        ):
            candidates.append(path.parent)
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one saved analysis under {parent.resolve()}, "
            f"found {len(candidates)}; select it with --analysis-id."
        )
    return candidates[0]
