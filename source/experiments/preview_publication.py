"""Publish reusable per-signal previews as validated immutable revisions.

Each revision contains index.md, manifest.json and named window folders.
The manifest records source identities, stage labels, numerical/PNG checksums
and rendering settings. Publication is atomic and guarded by a writer lock.
"""

import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

from .cache import atomic_json, file_digest, fingerprint, writer_lock
from .contracts import FeatureSet
from .signal_previews import preview_columns, render_window, window_excerpts

PREVIEW_VERSION = 4


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
    """Publish one complete data-only or checkpoint-specific preview revision.

    Parameters
    ----------
    session, fold : FeatureSet
        Validated native-session and processed-fold arrays.
    settings : dict
        Selection and presentation configuration.
    root : Path
        Parent directory for immutable revisions.
    windows : list of ndarray, each shape (T,)
        Fold indices for selected time windows.
    available : ndarray, optional
        Model population columns; default is all channels.
    fitted : list of dict, optional
        Per-window fitted arrays from the model adapter; default is absent.
    checkpoint : Path, optional
        Source checkpoint for fitted stages; default is absent.

    source_checksums : dict, optional
        Source file hashes to verify before publication; default is absent.

    Returns
    -------
    Path
        Absolute directory of a completed, checksum-verified revision.
    """
    if not windows or (fitted is not None and len(fitted) != len(windows)):
        raise ValueError(
            "Expected nonempty windows and matching fitted arrays."
        )
    columns = preview_columns(fold, settings, available)
    identity = dict(
        version=PREVIEW_VERSION,
        settings=settings,
        fold=fold.metadata,
        source_checksums=source_checksums or {},
        channels=fold.arrays["ids"][columns].tolist(),
        indices=[fold.arrays["indices"][w].tolist() for w in windows],
        checkpoint_sha256=file_digest(checkpoint) if checkpoint else None,
        implementation={
            name: file_digest(Path(__file__).with_name(name))
            for name in (
                "preview_publication.py",
                "signal_previews.py",
                "preview_rendering.py",
            )
        },
    )
    destination = (
        root.resolve()
        / f"presentation_v{PREVIEW_VERSION}_{fingerprint(identity)[:16]}"
    )
    with writer_lock(destination.with_suffix(".lock")):
        if destination.exists():
            manifest = json.loads((destination / "manifest.json").read_text())
            if not manifest["complete"] or manifest["identity"] != identity:
                raise ValueError(
                    f"Incomplete/incompatible previews: {destination}"
                )
            for name, digest in manifest["checksums"].items():
                if file_digest(destination / name) != digest:
                    raise ValueError(
                        f"Corrupt preview artifact: {destination / name}"
                    )
            return destination
        root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".rendering-", dir=root))
        try:
            records = []
            for number, indices in enumerate(windows):
                start = float(fold.arrays["t"][indices[0]])
                stop = start + settings["seconds"]
                name = f"window_{number + 1:02d}_{start:.3f}s-{stop:.3f}s"
                excerpts = window_excerpts(
                    session,
                    fold,
                    indices,
                    columns,
                    stop,
                    None if fitted is None else fitted[number],
                )
                figures = render_window(
                    staging / name,
                    excerpts,
                    fold.metadata["session"],
                    (start, stop),
                    settings["presentation"],
                )
                records.append(
                    dict(
                        directory=name,
                        start=start,
                        stop=stop,
                        source_indices=excerpts["source_indices"].tolist(),
                        channel_ids=excerpts["channel_ids"].tolist(),
                        unit_dimensions=excerpts["unit_dimensions"].tolist(),
                        figures=figures,
                    )
                )
            lines = [
                f"# {fold.metadata['session']} data previews",
                "",
                "Neural channels are separate; x/y coordinates share panels.",
                "",
            ]
            for record in records:
                lines.extend([f"## {record['directory']}", ""])
                for figure in record["figures"]:
                    link = f"{record['directory']}/{figure['file']}"
                    lines.append(f"- [{figure['signal']}]({link})")
                lines.append("")
            (staging / "index.md").write_text("\n".join(lines))
            checksums = {
                str(p.relative_to(staging)): file_digest(p)
                for p in sorted(staging.rglob("*"))
                if p.is_file()
            }
            for name, digest in (source_checksums or {}).items():
                if file_digest(Path(name)) != digest:
                    raise ValueError(f"Source changed during rendering: {name}")
            atomic_json(
                staging / "manifest.json",
                dict(
                    complete=True,
                    identity=identity,
                    windows=records,
                    source_files_unchanged=True,
                    session_cache=str(session.path) if session.path else None,
                    fold_cache=str(fold.path) if fold.path else None,
                    checkpoint=str(checkpoint.resolve())
                    if checkpoint
                    else None,
                    checksums=checksums,
                ),
            )
            os.replace(staging, destination)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    return destination
