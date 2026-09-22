"""Render selected previews from saved analyses and validated caches.

The plot-stage preview mode builds missing scientific caches with reuse
semantics, publishes shared preprocessing views in the cache root, and writes
checkpoint-derived fitted views only below selected completed fits.
"""

import copy
import json
import logging
import os
from pathlib import Path

import numpy as np

from .analysis import read_manifest
from .artifacts import resolved_fit, scientific_source
from .cache import fingerprint
from .contracts import FeatureSet, plugin
from .diagnostics import render_safely
from .populations import population_columns
from .presentation import presentation
from .previews import (
    fitted_previews,
    preprocessing_previews,
    preview_windows,
)

LOGGER = logging.getLogger(__name__)
PREVIEW_CACHE_VERSION = 1


def _analysis_cases(manifest: dict) -> list[dict]:
    """Return unique cases in saved analysis membership order."""
    cases = {}
    for member in manifest["members"].values():
        case = member["case"]
        previous = cases.setdefault(case["name"], case)
        if previous != case:
            raise ValueError(
                f"Analysis case {case['name']!r} has conflicting settings."
            )
    if not cases:
        raise ValueError("Saved analysis contains no model cases.")
    return list(cases.values())


def _selected_values(
    requested: list | None,
    available: set,
    label: str,
) -> list:
    """Validate an explicit selection or return every available value."""
    values = sorted(available) if requested is None else requested
    missing = set(values) - available
    if missing:
        raise ValueError(
            f"Selected preview {label} values are absent from the analysis: "
            f"{sorted(missing)}."
        )
    return values


def _render_settings(
    snapshots: dict,
    cases: list[dict],
) -> dict:
    """Resolve rendering settings and the whole-suite context requirement."""
    settings = copy.deepcopy(snapshots["plotting"]["previews"])
    settings.pop("enabled")
    settings.pop("selection")
    settings["presentation"] = presentation(
        snapshots["plotting"].get("presentation")
    )
    settings["adapter"] = snapshots["data"].get("preview_adapter")
    settings["context_samples"] = max(
        resolved_fit(dict(configurations=snapshots, case=case))["args_base"][
            "sequence_length"
        ]
        for case in cases
    )
    return settings


def _population_destination(
    cache_root: Path,
    session: str,
    fold_number: int,
    fold: FeatureSet,
    columns: np.ndarray,
) -> Path:
    """Address one population by fold provenance and ordered channel IDs."""
    if fold.path is None:
        raise ValueError(
            "Preprocessing previews require a published fold cache."
        )
    manifest = json.loads((fold.path / "manifest.json").read_text())
    identity = {
        "version": PREVIEW_CACHE_VERSION,
        "fold_identity": scientific_source(manifest["identity"]),
        "fold_sha256": manifest["sha256"],
        "channel_ids": fold.arrays["ids"][columns].tolist(),
    }
    return (
        cache_root
        / "previews"
        / "preprocessing"
        / session
        / f"fold_{fold_number}"
        / fingerprint(identity)
    ).resolve()


def _members_by_identity(manifest: dict) -> dict[tuple[str, int, str], dict]:
    """Index saved members by session, fold, and readable case name."""
    result = {}
    for member in manifest["members"].values():
        key = (member["session"], member["fold"], member["case"]["name"])
        if key in result:
            raise ValueError(f"Duplicate analysis preview member: {key}.")
        result[key] = member
    return result


def render_previews(
    snapshots: dict,
    analysis_root: Path,
) -> dict[str, int]:
    """Render selected preprocessing and fitted previews without fitting.

    Parameters
    ----------
    snapshots : dict
        Resolved experiment, data, plotting, path, and runtime settings.
    analysis_root : Path
        Existing analysis whose membership identifies fitted model cases.

    Returns
    -------
    dict of int
        Completed preprocessing and fitted rendering counts.

    Raises
    ------
    RuntimeError
        One or more independent cache or rendering operations failed.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TF_USE_LEGACY_KERAS"] = "1"
    manifest = read_manifest(analysis_root)
    cases = _analysis_cases(manifest)
    selected = snapshots["plotting"]["previews"]["selection"]
    available_sessions = {
        member["session"] for member in manifest["members"].values()
    }
    sessions = _selected_values(
        selected["sessions"], available_sessions, "session"
    )
    available_folds = {
        member["fold"] for member in manifest["members"].values()
    }
    folds = _selected_values(selected["folds"], available_folds, "fold")
    case_names = {case["name"] for case in cases}
    fitted_cases = selected["cases"] or []
    _selected_values(fitted_cases, case_names, "case")

    data = copy.deepcopy(snapshots["data"])
    data["cache_mode"] = "reuse"
    dataset = plugin(data["plugin"], settings=data)
    dataset_sessions = set(dataset.sessions())
    missing_sessions = set(sessions) - dataset_sessions
    if missing_sessions:
        raise ValueError(
            "Selected preview sessions are absent from the dataset: "
            f"{sorted(missing_sessions)}."
        )
    from .runner import population_order

    shared = population_order(
        dataset.inventory(), snapshots["experiment"]["seed"]
    )
    settings = _render_settings(snapshots, cases)
    regenerate = (
        snapshots["runtime"]["preview_regeneration"] == "all"
    )
    cache_root = Path(snapshots["paths"]["cache_root"]).resolve()
    artifact_root = Path(snapshots["paths"]["artifact_root"]).resolve()
    members = _members_by_identity(manifest)
    failures = []
    unavailable = []
    completed = dict(preprocessing=0, fitted=0)

    for session_name in sessions:
        try:
            session = dataset.load(session_name)
        except Exception as error:
            failures.append(
                f"Session preview cache {session_name}: "
                f"{type(error).__name__}: {error}"
            )
            continue
        for fold_number in folds:
            try:
                fold = dataset.fold(
                    session,
                    fold_number,
                    data["infer_velocity"],
                )
                windows = preview_windows(fold, settings)
                populations = {}
                for case in cases:
                    columns = population_columns(
                        fold.arrays["ids"].tolist(),
                        shared,
                        case["population_scale"],
                    )
                    populations.setdefault(tuple(columns.tolist()), columns)
            except Exception as error:
                failures.append(
                    f"Fold preview cache {session_name}/fold_{fold_number}: "
                    f"{type(error).__name__}: {error}"
                )
                continue
            for columns in populations.values():
                destination = _population_destination(
                    cache_root,
                    session_name,
                    fold_number,
                    fold,
                    columns,
                )
                before = len(failures)
                render_safely(
                    failures,
                    f"Preprocessing preview {destination}",
                    preprocessing_previews,
                    session,
                    fold,
                    settings,
                    destination,
                    columns,
                    windows,
                    regenerate,
                )
                completed["preprocessing"] += int(len(failures) == before)
            for case_name in fitted_cases:
                identity = (session_name, fold_number, case_name)
                member = members.get(identity)
                if member is None or member.get("state") != "complete":
                    unavailable.append(
                        f"{session_name}/fold_{fold_number}/{case_name}"
                    )
                    continue
                run = (artifact_root / member["fit"]).resolve()
                experiments = (artifact_root / "experiments").resolve()
                if not run.is_relative_to(experiments):
                    failures.append(
                        f"Fitted preview {identity}: fit path is outside "
                        "the experiments root."
                    )
                    continue
                before = len(failures)
                render_safely(
                    failures,
                    f"Fitted preview {run}",
                    fitted_previews,
                    session,
                    fold,
                    settings,
                    run,
                    snapshots["experiment"]["preview_model_plugin"],
                    windows,
                    regenerate,
                )
                completed["fitted"] += int(len(failures) == before)
    if unavailable:
        LOGGER.warning(
            "Skipped %d unavailable fitted preview selections because their "
            "analysis members are not complete: %s",
            len(unavailable),
            ", ".join(unavailable),
        )
    if failures:
        raise RuntimeError(
            "Preview rendering failed after independent selections were "
            "attempted:\n" + "\n".join(failures)
        )
    return completed
