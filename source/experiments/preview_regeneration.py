"""Regenerate one configured run's previews from saved artifacts only.

Inputs are a source_run reference, explicit channel/window selections, saved
session/fold caches, checkpoint and fitted excerpts. Outputs are immutable
preview revisions under the configured analysis root. No data extraction,
normalizer fitting, model reconstruction or training is performed here.
"""

import json
import logging
import os
from pathlib import Path

import numpy as np

from .artifacts import artifact_path, validate_completion
from .cache import file_digest, fingerprint, load_entry
from .contracts import plugin
from .preview_publication import publish_previews

LOGGER = logging.getLogger(__name__)
TIME_TOLERANCE = 1e-8


def regenerate_previews(settings: dict) -> Path:
    """Render the configured source run without invoking training.

    Parameters
    ----------
    settings : dict
        Resolved experiment and data configuration, including previews.

    Returns
    -------
    Path
        Absolute completed preview revision directory.
    """
    preview = settings["data"]["previews"]
    if not preview["enabled"]:
        raise ValueError("Enable data previews before requesting regeneration.")
    reference = preview.get("source_run")
    if not reference or not Path(reference).is_absolute():
        raise ValueError(
            "Set an absolute previews.source_run in configuration."
        )
    run = Path(reference).resolve()
    validate_completion(run)
    runtime = json.loads((artifact_path(run, "runtime.json")).read_text())
    if not runtime.get("cache"):
        raise ValueError("Source run has no saved fold-cache reference.")
    fold_path = Path(runtime["cache"])
    fold_manifest = json.loads((fold_path / "manifest.json").read_text())
    session_path = Path(fold_manifest["metadata"]["session_cache"])
    session_manifest = json.loads((session_path / "manifest.json").read_text())
    sources = [p for p in run.rglob("*") if p.is_file()]
    fitted_reference = artifact_path(run, "fitted_excerpts.json")
    if fitted_reference.exists():
        reference = json.loads(fitted_reference.read_text())
        directory = Path(reference["directory"])
        sources.extend([directory / "arrays.npz", directory / "manifest.json"])
    for cache in (session_path, fold_path):
        sources.extend([cache / "arrays.npz", cache / "manifest.json"])
    hashes = {str(p.resolve()): file_digest(p) for p in sorted(sources)}
    fold = load_entry(fold_path, fold_manifest["identity"])
    session = load_entry(session_path, session_manifest["identity"])
    requested = preview.get("window_ranges")
    if not requested or len(requested) != preview["windows"]:
        raise ValueError("Specify every saved preview window in window_ranges.")
    windows = []
    time = fold.arrays["t"]
    interval = 1 / fold.metadata["settings"]["sampling_rate_hz"]
    for start, stop in requested:
        indices = np.flatnonzero(
            (time >= start - TIME_TOLERANCE) & (time < stop - TIME_TOLERANCE)
        )
        if (
            not len(indices)
            or not np.isclose(stop - start, preview["seconds"])
            or not np.isclose(time[indices[0]], start)
            or not np.isclose(time[indices[-1]] + interval, stop)
            or not np.allclose(np.diff(time[indices]), interval)
        ):
            raise ValueError(
                f"Saved time grid cannot support window {start}–{stop}."
            )
        windows.append(indices)
    # Checkpoint unpickling imports TensorFlow classes, but only NumPy maps run.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TF_USE_LEGACY_KERAS"] = "1"
    fitted = plugin(
        settings["experiment"]["preview_model_plugin"],
        source_run=run,
        features=fold,
        windows=windows,
    )
    with np.load(
        artifact_path(run, "selection.npz"), allow_pickle=False
    ) as saved:
        columns = saved["selected_columns"]
    destination = (
        Path(settings["paths"]["artifact_root"])
        / "analysis" / settings["experiment"]["name"]
        / fingerprint(dict(stage="preview", fit_id=run.name, previews=preview))
        / "previews"
        / fold.metadata["session"]
        / f"fold_{fold.metadata['fold']}"
    )
    result = publish_previews(
        session,
        fold,
        preview,
        destination,
        windows,
        columns,
        fitted,
        artifact_path(run, "model.p"),
        hashes,
    )
    LOGGER.info("Completed artifact-only previews: %s", result)
    return result
