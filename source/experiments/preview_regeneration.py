"""Redraw fit-owned previews using saved caches and checkpoint inference.

The preview stage uses previews.source_run and optional explicit channel_ids
and train/validation/test window_ranges. Outputs replace only data_preview
rendering payloads. No fitting, scientific-cache writes or analysis creation
occur. Local source-run configuration must remain outside version control.
"""

import json
import logging
import os
from pathlib import Path

import numpy as np

from .artifacts import artifact_path, validate_completion
from .cache import file_digest, load_entry
from .presentation import presentation
from .previews import fitted_previews, preprocessing_previews

LOGGER = logging.getLogger(__name__)


def regenerate_previews(settings: dict) -> Path:
    """Render a completed fit in place without changing scientific payloads."""
    preview = dict(settings["data"]["previews"])
    if not preview["enabled"]:
        raise ValueError("Enable data previews before requesting regeneration.")
    reference = preview.get("source_run")
    if not reference or not Path(reference).is_absolute():
        raise ValueError(
            "Set an absolute previews.source_run in configuration."
        )
    run = Path(reference).resolve()
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TF_USE_LEGACY_KERAS"] = "1"
    completion = validate_completion(run)
    runtime = json.loads(artifact_path(run, "runtime.json").read_text())
    if not runtime.get("cache"):
        raise ValueError("Source run has no saved fold-cache reference.")
    fold_path = Path(runtime["cache"])
    fold_manifest = json.loads((fold_path / "manifest.json").read_text())
    session_path = Path(fold_manifest["metadata"]["session_cache"])
    session_manifest = json.loads((session_path / "manifest.json").read_text())
    sources = [
        path / filename
        for path in (session_path, fold_path)
        for filename in ("manifest.json", "arrays.npz")
    ]
    hashes = {path: file_digest(path) for path in sources}
    fold = load_entry(fold_path, fold_manifest["identity"])
    session = load_entry(session_path, session_manifest["identity"])
    preview["presentation"] = presentation(
        settings["plotting"].get("presentation")
    )
    arguments = json.loads(artifact_path(run, "fit_arguments.json").read_text())
    preview["context_samples"] = arguments["args_base"]["sequence_length"]
    with np.load(
        artifact_path(run, "selection.npz"), allow_pickle=False
    ) as data:
        columns = data["selected_columns"]
    preprocessing_previews(session, fold, preview, run, columns)
    fitted_previews(
        session,
        fold,
        preview,
        run,
        settings["experiment"]["preview_model_plugin"],
        columns,
    )
    if validate_completion(run) != completion or any(
        file_digest(path) != digest for path, digest in hashes.items()
    ):
        raise ValueError(
            "Scientific source artifacts changed during rendering."
        )
    result = run / "data_preview"
    LOGGER.info("Completed data previews: %s", result)
    return result
