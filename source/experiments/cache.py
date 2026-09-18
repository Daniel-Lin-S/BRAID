"""Atomic, provenance-addressed NumPy feature caches.

Each completed directory has arrays.npz (named, non-object arrays) and
manifest.json (identity, metadata, payload checksum, completion state).
Only validated completed entries are reused. Locks serialize writers;
failed writes remain unpublished and are removed by their owning writer.
"""

from contextlib import contextmanager
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile
from typing import Callable, Iterator

import numpy as np

from .contracts import FeatureSet
from artifact_io import file_digest

LOGGER = logging.getLogger(__name__)
CACHE_VERSION = 1


def fingerprint(value: object) -> str:
    """Hash a JSON-compatible configuration using canonical key ordering."""
    encoded = json.dumps(value, sort_keys=True, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    """Publish strict JSON atomically; reject unsupported/nonfinite values."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def writer_lock(path: Path, blocking: bool = True) -> Iterator[None]:
    """Serialize writers for one cache entry or experiment result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
        try:
            fcntl.flock(stream.fileno(), flags)
        except BlockingIOError as error:
            raise RuntimeError(
                f"Cannot acquire lock; active writer at {path.resolve()}"
            ) from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def validate_arrays(arrays: dict[str, np.ndarray]) -> None:
    """Reject empty, object-valued, or nonfinite cached arrays."""
    if not arrays:
        raise ValueError("Feature extraction produced no arrays.")
    for name, array in arrays.items():
        if not array.size or array.dtype.hasobject:
            raise ValueError(f"Invalid cache array {name}: {array.shape}.")
        if np.issubdtype(array.dtype, np.number):
            if not np.isfinite(array).all():
                raise ValueError(f"Nonfinite values in cache array {name}.")


def load_entry(directory: Path, identity: dict) -> FeatureSet:
    """Read a completed cache after checking identity and payload checksum."""
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["identity"] != identity or not manifest["complete"]:
        raise ValueError(f"Cache provenance mismatch: {directory.resolve()}")
    payload = directory / "arrays.npz"
    if file_digest(payload) != manifest["sha256"]:
        raise ValueError(f"Cache checksum mismatch: {payload.resolve()}")
    with np.load(payload, allow_pickle=False) as archive:
        arrays = dict(archive)
    validate_arrays(arrays)
    return FeatureSet(arrays, manifest["metadata"], directory)


def cached(
    root: Path,
    kind: str,
    identity: dict,
    mode: str,
    build: Callable[[], FeatureSet],
) -> FeatureSet:
    """Build or reuse a complete cache entry according to an explicit mode.

    Parameters
    ----------
    root : Path
        Absolute cache root.
    kind : str
        Transformation namespace, such as session or fold.
    identity : dict
        Complete deterministic transformation provenance.
    mode : {'reuse', 'rebuild', 'off'}
        Reuse validates payloads; rebuild explicitly replaces that identity.
    build : callable
        Returns arrays and metadata only when extraction is necessary.

    Returns
    -------
    FeatureSet
        Finite arrays, provenance, and the published cache directory.
    """
    if mode not in {"reuse", "rebuild", "off"}:
        raise ValueError(f"Unsupported cache mode: {mode}.")
    if mode == "off":
        result = build()
        validate_arrays(result.arrays)
        return result
    identity = dict(identity, cache_version=CACHE_VERSION)
    directory = root.resolve() / kind / fingerprint(identity)
    with writer_lock(directory.with_suffix(".lock")):
        if directory.exists() and mode == "reuse":
            LOGGER.info("Reusing cache %s", directory)
            return load_entry(directory, identity)
        result = build()
        validate_arrays(result.arrays)
        directory.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(dir=directory.parent))
        try:
            payload = staging / "arrays.npz"
            np.savez_compressed(payload, **result.arrays)
            atomic_json(
                staging / "manifest.json",
                dict(
                    complete=True,
                    identity=identity,
                    metadata=result.metadata,
                    sha256=file_digest(payload),
                ),
            )
            if directory.exists():
                shutil.rmtree(directory)
            os.replace(staging, directory)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        result.path = directory
        return result
