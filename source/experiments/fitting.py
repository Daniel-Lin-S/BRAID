"""Persist fit and inference completion independently of analysis.

Fit artifacts use the registered configuration/provenance/data/checkpoints
groups. Each predictions/test/horizons_<steps> bundle contains predictions.npz
(Y/Z/X forecasts shaped (H, T, C), valid masks (H, T), truths (T, C),
times/indices (T,), horizons (H,) and training means (C,)) and completion.json
(input provenance and payload checksum). C varies with the predicted target.
"""

import argparse
import json
import logging
import os
from pathlib import Path
import uuid

import numpy as np
import yaml

from .artifacts import (
    ARTIFACT_PATHS, artifact_path, fit_identity, fit_seed, prepare_run,
    prepare_model_settings, effective_model_configuration,
    validate_completion,
    completed_fit,
)
from .cache import atomic_json, file_digest
from .coordination import shared_writer_lock
from .contracts import FeatureSet, Model, plugin
from .windows import window_indices

LOGGER = logging.getLogger(__name__)


def canonical_horizons(horizons: list[int]) -> list[int]:
    """Validate and order distinct positive inference horizons."""
    if (
        not horizons or any(type(h) is not int or h < 1 for h in horizons)
        or len(set(horizons)) != len(horizons)
    ):
        raise ValueError("Expected unique positive integer forecast horizons.")
    return sorted(horizons)


def make_backend(identity: dict, run: Path) -> Model:
    """Construct the configured fitting adapter using its recorded seed."""
    return plugin(
        identity["configurations"]["experiment"]["model_plugin"],
        configuration=str(artifact_path(run, "model_configuration.yaml")),
        overrides=identity["model_overrides"], seed=fit_seed(identity),
    )


def ensure_fit(
    run: Path, identity: dict, features: FeatureSet, columns: np.ndarray,
    arguments: argparse.Namespace, gpu: dict | None,
) -> bool:
    """Reuse a validated fit or complete only its unfinished training work.

    Parameters
    ----------
    run : Path
        Canonical fit-ID directory.
    identity : dict
        Resolved numerical inputs and recorded configuration.
    features : FeatureSet
        Time-first arrays and training/validation/test split roles.
    columns : ndarray, shape (C,)
        Ordered neural input/output columns.
    arguments : Namespace
        Invocation stage and logging preferences.
    gpu : dict or None
        Diagnostic allocation record, excluded from fit identity.

    Returns
    -------
    bool
        True only when a fit was completed during this invocation.
    """
    completion = artifact_path(run, "fit_complete.json")
    if completed_fit(run, identity):
        LOGGER.info("Reusing completed fit %s", run.resolve())
        return False
    if arguments.stage == "evaluate" and not completion.exists():
        raise ValueError(f"Evaluation requires a completed fit: {run}")
    prepare_model_settings(run.parents[2], identity)
    run.mkdir(parents=True, exist_ok=True)
    with shared_writer_lock(run / "fit.lock", run.name, "fit"):
        if completed_fit(run, identity):
            LOGGER.info("Reusing completed fit %s", run)
            return False
        if arguments.stage == "evaluate":
            raise ValueError(f"Evaluation requires a completed fit: {run}")
        prepare_run(run)
        atomic_json(
            artifact_path(run, "identity.json"),
            dict(fit_id=run.name, identity=fit_identity(identity)),
        )
        effective = effective_model_configuration(identity)
        artifact_path(run, "model_configuration.yaml").write_text(
            yaml.safe_dump(effective)
        )
        resolved = dict(identity["configurations"], model=effective)
        atomic_json(
            artifact_path(run, "resolved_configurations.json"),
            resolved,
        )
        atomic_json(
            artifact_path(run, "runtime.json"),
            dict(
                gpu=gpu, pid=os.getpid(),
                tensorboard=bool(
                    getattr(arguments, "tensorboard", False)
                ),
                cache=str(features.path) if features.path else None,
                text_log=str((
                    arguments.log_directory / "sessions"
                    / f"{features.metadata['session']}.log"
                ).resolve()),
            ),
        )
        atomic_json(
            artifact_path(run, "seed.json"), dict(seed=fit_seed(identity))
        )
        np.savez_compressed(
            artifact_path(run, "selection.npz"),
            selected_columns=columns,
            channel_ids=features.arrays["ids"][columns],
            unit_dimensions=features.arrays["units"][columns],
            indices=features.arrays["indices"], role=features.arrays["role"],
            segment=features.arrays["segment"],
        )
        status = artifact_path(run, "status.json")
        atomic_json(status, dict(state="running", pid=os.getpid()))
        try:
            from .restart import prepare_components

            prepare_components(run / "components")
            from BRAID.tools.tensorboard import tensorboard_scope

            backend = make_backend(identity, run)
            with tensorboard_scope(
                bool(getattr(arguments, "tensorboard", False))
            ):
                backend.fit(
                    features,
                    columns,
                    identity["case"]["dimensions"],
                    run,
                )
                backend.save(artifact_path(run, "model.p"))
            atomic_json(status, dict(state="complete"))
            checksums = {
                name: file_digest(artifact_path(run, name))
                for name in ARTIFACT_PATHS
                if name != "fit_complete.json"
                and artifact_path(run, name).exists()
            }
            atomic_json(
                completion, dict(fit_id=run.name, checksums=checksums)
            )
            return True
        except BaseException as error:
            atomic_json(
                status, dict(state="failed", error=f"{type(error).__name__}: "
                             f"{error}"),
            )
            raise


def prediction_directory(run: Path, horizons: list[int]) -> Path:
    """Resolve a readable inference bundle beneath its owning fit."""
    label = "_".join(str(h) for h in canonical_horizons(horizons))
    return run / "predictions" / "test" / f"horizons_{label}"


def validate_predictions(directory: Path, expected: dict) -> Path | None:
    """Validate an existing prediction completion without invoking a model."""
    completion = directory / "completion.json"
    if not completion.exists():
        return None
    record = json.loads(completion.read_text())
    from .implementation import compatible_inference

    recorded = record["identity"]
    numerical = "inference_implementation"
    compatible = (
        set(recorded) == set(expected)
        and {k: v for k, v in recorded.items() if k != numerical}
        == {k: v for k, v in expected.items() if k != numerical}
        and compatible_inference(
            recorded.get(numerical), expected.get(numerical)
        )
    )
    if not compatible:
        raise ValueError(f"Prediction provenance mismatch: {directory}")
    path = directory / "predictions.npz"
    if file_digest(path) != record["sha256"]:
        raise ValueError(f"Prediction checksum mismatch: {path}")
    return path


def validate_forecast_arrays(
    predictions: dict[str, np.ndarray], samples: int, neural: int,
    behavior: int, latent: int, horizons: list[int], length: int,
) -> None:
    """Validate forecast dimensions, finite values and window validity.

    Parameters
    ----------
    predictions : dict of ndarray
        Y/Z/X arrays shaped (H, T, C) and boolean valid shaped (H, T).
    samples : int
        Number T of held-out samples in complete independent windows.
    neural, behavior, latent : int
        Expected channel counts for Y, Z and X respectively.
    horizons : list of int
        H positive forecast offsets, each shorter than a window.
    length : int
        Samples per independent window.
    """
    if samples < 1 or length < 1 or samples % length:
        raise ValueError("Expected nonempty complete prediction windows.")
    canonical_horizons(horizons)
    if max(horizons) >= length:
        raise ValueError("Forecast horizons must be shorter than a window.")
    for name, channels in (("Y", neural), ("Z", behavior), ("X", latent)):
        expected = (len(horizons), samples, channels)
        values = predictions.get(name)
        if values is None or values.shape != expected:
            actual = None if values is None else values.shape
            raise ValueError(
                f"Expected {name} forecast shape {expected}, got {actual}."
            )
        if not values.size or not np.isfinite(values).all():
            raise ValueError(f"Empty or nonfinite {name} forecasts.")
    valid = predictions.get("valid")
    expected = np.array([
        np.arange(samples) % length >= horizon for horizon in horizons
    ])
    if valid is None or valid.dtype != np.bool_ or not np.array_equal(
        valid, expected
    ):
        raise ValueError(
            f"Expected boolean valid mask shaped {expected.shape} "
            "matching independent window boundaries and forecast horizons."
        )


def prediction_arrays(
    backend: Model, run: Path, identity: dict, features: FeatureSet,
    columns: np.ndarray, horizons: list[int],
) -> dict[str, np.ndarray]:
    """Forecast held-out windows from a validated saved checkpoint."""
    test = window_indices(features, 2, backend.length).ravel()
    arrays = features.arrays
    truth_y = arrays["Y"][test][:, columns]
    predictions = backend.predict(truth_y, arrays["U"][test], horizons)
    validate_forecast_arrays(
        predictions, len(test), len(columns), arrays["Z"].shape[1],
        identity["case"]["dimensions"]["nx"], horizons, backend.length,
    )
    with np.load(artifact_path(run, "fit_indices.npz")) as fitted:
        source_indices = fitted["training"].ravel()
    training = np.searchsorted(arrays["indices"], source_indices)
    if (
        not len(training) or np.any(training >= len(arrays["indices"]))
        or not np.array_equal(arrays["indices"][training], source_indices)
    ):
        raise ValueError(f"Saved training indices do not match inputs: {run}")
    payload = dict(
        predictions, true_Y=truth_y, true_Z=arrays["Z"][test],
        t=arrays["t"][test], source_indices=arrays["indices"][test],
        horizons=np.asarray(horizons),
        baseline_Y=arrays["Y"][training][:, columns].mean(axis=0),
        baseline_Z=arrays["Z"][training].mean(axis=0),
    )
    for name, values in payload.items():
        if not values.size or not np.isfinite(values).all():
            raise ValueError(f"Empty or nonfinite prediction payload: {name}")
    return payload


def ensure_predictions(
    run: Path, identity: dict, features: FeatureSet, columns: np.ndarray,
    horizons: list[int],
) -> Path:
    """Reuse or publish inference without writing any completed fit payload."""
    validate_completion(run)
    horizons = canonical_horizons(horizons)
    directory = prediction_directory(run, horizons)
    expected = dict(
        fit_id=run.name,
        checkpoint_sha256=file_digest(artifact_path(run, "model.p")),
        source=fit_identity(identity)["source"], horizons=horizons,
        inference_implementation=identity["configurations"].get(
            "inference_implementation"
        ),
    )
    path = validate_predictions(directory, expected)
    if path is not None:
        return path
    with shared_writer_lock(
        directory.parent / f"{directory.name}.lock",
        run.name,
        "prediction",
    ):
        path = validate_predictions(directory, expected)
        if path is not None:
            return path
        if directory.exists():
            quarantine = directory.parent / "quarantine"
            quarantine.mkdir(exist_ok=True)
            directory.rename(
                quarantine / f"{directory.name}-{uuid.uuid4().hex}"
            )
            LOGGER.warning(
                "Quarantined incomplete prediction bundle beneath %s",
                quarantine.resolve(),
            )
        directory.mkdir()
        backend = make_backend(identity, run)
        backend.load(artifact_path(run, "model.p"))
        arrays = prediction_arrays(
            backend, run, identity, features, columns, horizons
        )
        path = directory / "predictions.npz"
        np.savez_compressed(path, **arrays)
        atomic_json(
            directory / "completion.json",
            dict(identity=expected, sha256=file_digest(path)),
        )
        return path
