"""Reuse immutable target preprocessing predictions and native snapshots.

A prediction cache contains identity.json, behavior.npz (training/validation
arrays shaped samples by behavior coordinates), and complete.json checksums.
Identity includes checkpoint, array content/dtype/shape, and implementation.
Snapshots retain native {'model': MainModel} serialization without destroying
live TensorFlow graphs or rebuilding graphs merely to save their weights.
"""

from copy import copy
import hashlib
from pathlib import Path

import numpy as np

from BRAID.BRAIDModel import runPredict
from BRAID.MainModel import MainModel
from experiments.storage.cache import file_digest, fingerprint
from experiments.multisession.store import completed, dump_pickle, job

CACHE_NAMESPACE = "BRAID_preprocessing_predictions"


def array_identity(values: np.ndarray) -> dict:
    """Hash finite dimension-first data, including its exact dtype and shape."""
    values = np.ascontiguousarray(values)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("Preprocessing cache inputs must be nonempty/finite.")
    return dict(
        shape=list(values.shape),
        dtype=values.dtype.str,
        sha256=hashlib.sha256(memoryview(values).cast("B")).hexdigest(),
    )


def preprocessing_predictions(
    model: MainModel,
    arrays: dict[str, np.ndarray],
    preprocessing: Path,
    implementation: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute each selected subset's frozen predictions exactly once.

    Parameters
    ----------
    model : MainModel
        Native preprocessing checkpoint, optionally not yet reconstructed.
    arrays : dict of ndarray
        Y, Z, U and their validation arrays, shape (coordinates, samples).
    preprocessing : Path
        Immutable dependency directory containing model.p.
    implementation : str
        Numerical implementation fingerprint for cache invalidation.

    Returns
    -------
    training, validation : ndarray
        Native one-step behavior predictions, shape (samples, coordinates).
    """
    identity = dict(
        checkpoint=file_digest(preprocessing / "model.p"),
        implementation=implementation,
        arrays={key: array_identity(value) for key, value in arrays.items()},
    )
    directory = preprocessing.parent.parent / CACHE_NAMESPACE / fingerprint(
        identity
    )
    with job(directory, identity) as needed:
        if needed:
            model.restoreModels()
            predictions = {}
            for role, suffix in (
                ("training", ""), ("validation", "_validation")
            ):
                result = runPredict(
                    model,
                    Y=arrays["Y" + suffix].T,
                    Z=arrays["Z" + suffix].T,
                    U=arrays["U" + suffix].T,
                    YType="cont", ZType="cont", useXFilt=False,
                    missing_marker=model.missing_marker,
                )[0]
                predictions[role] = (
                    result[0] if isinstance(result, list) else result
                )
            validate_predictions(predictions, arrays)
            np.savez(directory / "behavior.npz", **predictions)
    if completed(directory) is None:
        raise RuntimeError(f"Incomplete preprocessing cache: {directory}")
    with np.load(directory / "behavior.npz", allow_pickle=False) as archive:
        predictions = dict(archive)
    validate_predictions(predictions, arrays)
    return predictions["training"], predictions["validation"]


def validate_predictions(predictions: dict, arrays: dict) -> None:
    """Reject cache shape mismatches and nonfinite behavior predictions."""
    for role, suffix in (("training", ""), ("validation", "_validation")):
        expected = arrays["Z" + suffix].T.shape
        actual = predictions[role]
        if actual.shape != expected or not np.isfinite(actual).all():
            raise ValueError(
                f"Expected finite {role} predictions shaped {expected}, "
                f"got {actual.shape}."
            )


def checkpoint_copy(model: MainModel, memo: dict | None = None) -> MainModel:
    """Copy native model containers, replacing live graphs with weight arrays.

    Shared preprocessing references stay shared in the serialized object.
    Only containers are copied; large data arrays are not duplicated.
    """
    if memo is None:
        memo = {}
    if id(model) in memo:
        return memo[id(model)]
    snapshot = copy(model)
    memo[id(model)] = snapshot
    for name, value in vars(model).items():
        if isinstance(value, MainModel):
            setattr(snapshot, name, checkpoint_copy(value, memo))
        elif hasattr(value, "get_recreation_info"):
            record = value.get_recreation_info()
            record.constructor_kwargs = dict(record.constructor_kwargs)
            for setting in (
                "steps_ahead", "batch_size", "multi_step_with_A_KC"
            ):
                if setting in record.constructor_kwargs:
                    record.constructor_kwargs[setting] = getattr(
                        value, setting
                    )
            setattr(snapshot, name, record)
    return snapshot


def save_snapshot(model: MainModel, path: Path) -> None:
    """Atomically save a native checkpoint while keeping live graphs intact."""
    dump_pickle(path, dict(model=checkpoint_copy(model)))
