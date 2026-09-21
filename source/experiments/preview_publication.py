"""Publish fit-owned previews at stable preprocessing/fitted destinations.

Each destination holds manifest.json and train/validation/test folders with
PNG figures and excerpts.npz. Manifests describe sources, selections, stages,
style and checksums. Only rendering payloads are replaced under a writer lock;
scientific artifacts and old-layout directories are never modified.
"""

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

from .cache import atomic_json, file_digest, fingerprint, writer_lock
from .contracts import FeatureSet, plugin
from .previews import SPLIT_NAMES
from .signal_previews import preview_columns, render_window, window_excerpts

PREVIEW_VERSION = 6


def _preview_adapter(settings: dict):
    """Construct an optional modality-specific preview adapter."""
    reference = settings.get("adapter")
    if reference is None:
        return None
    if not isinstance(reference, str) or ":" not in reference:
        raise ValueError("previews.adapter must have the form module:class.")
    adapter = plugin(reference, settings=settings)
    for attribute in (
        "implementation_identity",
        "window_excerpts",
        "render_window",
        "window_metadata",
    ):
        if not hasattr(adapter, attribute):
            raise TypeError(f"Preview adapter {reference!r} lacks {attribute}.")
    return adapter


def publish_previews(
    session: FeatureSet,
    fold: FeatureSet,
    settings: dict,
    root: Path,
    windows: list[np.ndarray],
    available: np.ndarray | None = None,
    fitted: list[dict] | None = None,
    checkpoint: Path | None = None,
    source_checksums: dict[str, str] | None = None,
) -> Path:
    """Stage all figures before replacing owned files and publishing manifest.

    Parameters
    ----------
    session, fold : FeatureSet
        Cached native and split-local arrays.
    settings : dict
        Preview selection and resolved presentation settings.
    root : Path
        Exact preprocessing or fitted destination.
    windows : list of ndarray
        Train, validation, test indices, each shape (N,).
    available : ndarray, optional
        Ordered fitted population, shape (C,); default is every channel.
    fitted : list of dict, optional
        Fitted stages per window; default is preprocessing-only.
    checkpoint : Path, optional
        Fitted source checkpoint; default is absent.
    source_checksums : dict, optional
        Additional protected file digests; default is absent.

    Returns
    -------
    Path
        Absolute completed rendering destination.
    """
    if len(windows) != len(SPLIT_NAMES) or (
        fitted is not None and len(fitted) != len(windows)
    ):
        raise ValueError("Expected three split windows and matching stages.")
    for role, indices in enumerate(windows):
        if not len(indices) or not np.all(fold.arrays["role"][indices] == role):
            raise ValueError(f"Invalid {SPLIT_NAMES[role]} preview indices.")
    columns = preview_columns(fold, settings, available)
    adapter = _preview_adapter(settings)
    protected = dict(source_checksums or {})
    for features in (session, fold):
        if features.path is not None:
            for name in ("manifest.json", "arrays.npz"):
                path = (features.path / name).resolve()
                protected[str(path)] = file_digest(path)
    if checkpoint is not None:
        protected[str(checkpoint.resolve())] = file_digest(checkpoint)
    identity = {
        "version": PREVIEW_VERSION,
        "settings": settings,
        "fold": fold.metadata,
        "source_checksums": protected,
        "channels": fold.arrays["ids"][columns].tolist(),
        "indices": [fold.arrays["indices"][w].tolist() for w in windows],
        "checkpoint_sha256": file_digest(checkpoint) if checkpoint else None,
        "fitted_arrays": [
            {
                key: {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
                }
                for key, value in window.items()
            }
            for window in fitted or []
        ],
        "implementation": {
            name: file_digest(Path(__file__).with_name(name))
            for name in (
                "preview_publication.py",
                "signal_previews.py",
                "preview_rendering.py",
                "presentation.py",
            )
        },
        "adapter_implementation": (
            adapter.implementation_identity() if adapter else None
        ),
    }
    destination = root.resolve()
    with writer_lock(destination.parent / f".{destination.name}.lock"):
        manifest_path = destination / "manifest.json"
        previous = (
            json.loads(manifest_path.read_text())
            if manifest_path.exists()
            else {}
        )
        if (
            previous.get("identity") == identity
            and previous.get("complete")
            and all(
                (destination / name).is_file()
                and file_digest(destination / name) == digest
                for name, digest in previous["checksums"].items()
            )
        ):
            return destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(
                prefix=".rendering-",
                dir=destination.parent,
            )
        )
        try:
            records = []
            for number, (split, indices) in enumerate(
                zip(SPLIT_NAMES, windows)
            ):
                start = float(fold.arrays["t"][indices[0]])
                stop = start + settings["seconds"]
                arguments = (
                    session,
                    fold,
                    indices,
                    columns,
                    stop,
                    None if fitted is None else fitted[number],
                )
                excerpts = (
                    adapter.window_excerpts(*arguments)
                    if adapter
                    else window_excerpts(*arguments)
                )
                render = adapter.render_window if adapter else render_window
                figures = render(
                    staging / split,
                    excerpts,
                    fold.metadata["session"],
                    (start, stop),
                    settings["presentation"],
                )
                record = {
                    "directory": split,
                    "start": start,
                    "stop": stop,
                    "source_indices": excerpts["source_indices"].tolist(),
                    "channel_ids": excerpts["channel_ids"].tolist(),
                    "figures": figures,
                }
                if adapter:
                    record.update(adapter.window_metadata(excerpts))
                else:
                    record["unit_dimensions"] = excerpts[
                        "unit_dimensions"
                    ].tolist()
                records.append(record)
            checksums = {
                str(path.relative_to(staging)): file_digest(path)
                for path in sorted(staging.rglob("*"))
                if path.is_file()
            }
            for name, digest in protected.items():
                if file_digest(Path(name)) != digest:
                    raise ValueError(f"Source changed during rendering: {name}")
            manifest = {
                "complete": True,
                "identity": identity,
                "rendering_id": fingerprint(identity),
                "windows": records,
                "source_files_unchanged": True,
                "session_cache": str(session.path) if session.path else None,
                "fold_cache": str(fold.path) if fold.path else None,
                "checkpoint": str(checkpoint.resolve()) if checkpoint else None,
                "checksums": checksums,
            }
            for name in checksums:
                target = destination / name
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.is_symlink():
                    raise ValueError(f"Refusing symlink output: {target}")
                os.replace(staging / name, target)
            for name in previous.get("checksums", {}).keys() - checksums.keys():
                target = (destination / name).resolve()
                if not target.is_relative_to(destination):
                    raise ValueError(f"Invalid rendering output: {target}")
                if target.suffix not in (".png", ".npz"):
                    raise ValueError(f"Unexpected rendering payload: {target}")
                target.unlink(missing_ok=True)
            atomic_json(manifest_path, manifest)
        finally:
            shutil.rmtree(staging)
    return destination
