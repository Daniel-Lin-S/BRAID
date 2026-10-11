"""Immutable, checksum-validated jobs for multi-session training.

Each job directory has identity.json, status.json, payload files, and an
atomic complete.json listing SHA256 checksums. Incomplete attempts move to
quarantine before retry; completed payloads are never overwritten.
"""

from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
import os
import pickle
import time
import uuid

from experiments.storage.cache import (
    atomic_json,
    file_digest,
    fingerprint,
    writer_lock,
)


FIT_STORES = {
    "scratch": "experiments/BRAID_matched_scratch",
    "preprocessing": "experiments/BRAID_matched_preprocessing",
}
DEFAULT_FIT_STORE = "multi_session_training/fits"


def fit_directory(plan: dict, key: str) -> Path:
    """Resolve a fit from plan storage metadata, outside scientific identity.

    Plans without storage metadata retain their original artifact locations.
    Scratch checkpoints and preprocessing dependencies have shared namespaces
    under experiments, accessible without an analysis-specific directory.
    """
    kind = plan["jobs"][key]["kind"]
    relative = plan.get("storage", {}).get(kind, DEFAULT_FIT_STORE)
    return (
        Path(plan["settings"]["paths"]["artifact_root"]) / relative / key
    ).resolve()


def dump_pickle(path: Path, value: object) -> None:
    """Atomically write an internal model/array payload."""
    temporary = path.with_suffix(path.suffix + ".pending")
    with temporary.open("wb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def load_pickle(path: Path) -> object:
    """Load a trusted local checkpoint created by this experiment."""
    with path.open("rb") as stream:
        return pickle.load(stream)


@lru_cache(maxsize=100000)
def versioned_digest(path: Path, size: int, modified: int, changed: int) -> str:
    """Cache a verified payload hash until its filesystem version changes."""
    return file_digest(path)


def completed(directory: Path) -> dict | None:
    """Validate every completed payload, refusing corrupt completions."""
    import json

    marker = directory / "complete.json"
    if not marker.exists():
        return None
    record = json.loads(marker.read_text())
    if not record.get("checksums"):
        raise ValueError(f"Empty completion checksums: {marker.resolve()}")
    for name, digest in record["checksums"].items():
        path = directory / name
        if not path.is_file():
            raise ValueError(f"Missing completed payload: {path.resolve()}")
        stat = path.stat()
        actual = versioned_digest(
            path,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        )
        if actual != digest:
            raise ValueError(f"Completed checksum mismatch: {path.resolve()}")
    return record


def quarantine_incomplete_components(directory: Path) -> list[Path]:
    """Isolate interrupted native stages before retrying a target fit.

    Parameters
    ----------
    directory : Path
        Target fit directory; stages reside in the sibling component store.

    Returns
    -------
    list[Path]
        Absolute quarantine paths for stages lacking a completion record.
    """
    root = directory.parent.parent / "target_components" / directory.name
    if not root.exists():
        return []
    moved = []
    for group in root.iterdir():
        if not group.is_dir():
            continue
        for stage in group.iterdir():
            if not stage.is_dir() or (stage / "complete.json").exists():
                continue
            quarantine = root.parent / "quarantine"
            quarantine.mkdir(exist_ok=True)
            destination = quarantine / (
                f"{directory.name}-{group.name}-{stage.name}-"
                f"{time.time_ns()}-{uuid.uuid4().hex[:8]}"
            )
            stage.rename(destination)
            moved.append(destination)
    return moved


def finish(directory: Path, metadata: dict) -> None:
    """Publish a completion only after all payloads are durable."""
    if (directory / "complete.json").exists():
        raise ValueError(f"Refusing to overwrite completion: {directory}")
    files = [
        p
        for p in directory.rglob("*")
        if p.is_file()
        and p.name not in ("status.json", "complete.json")
        and not p.name.endswith(".lock")
    ]
    if not files:
        raise ValueError(f"Cannot complete empty job: {directory}")
    atomic_json(
        directory / "complete.json",
        dict(
            **metadata,
            checksums={
                str(p.relative_to(directory)): file_digest(p) for p in files
            },
        ),
    )


@contextmanager
def job(directory: Path, identity: dict):
    """Lock a job, reuse completion, and isolate unfinished attempts."""
    directory.parent.mkdir(parents=True, exist_ok=True)
    with writer_lock(directory.with_suffix(".lock")):
        record = completed(directory)
        if record is not None:
            if record["identity_hash"] != fingerprint(identity):
                raise ValueError(f"Job identity mismatch: {directory}")
            yield False
            return
        if directory.exists():
            quarantine = directory.parent / "quarantine"
            quarantine.mkdir(exist_ok=True)
            directory.rename(
                quarantine
                / (f"{directory.name}-{time.time_ns()}-{uuid.uuid4().hex[:8]}")
            )
        directory.mkdir()
        atomic_json(directory / "identity.json", identity)
        atomic_json(directory / "status.json", dict(state="running"))
        try:
            yield True
            finish(directory, dict(identity_hash=fingerprint(identity)))
            atomic_json(directory / "status.json", dict(state="complete"))
        except BaseException as error:
            atomic_json(
                directory / "status.json",
                dict(
                    state="failed",
                    error=f"{type(error).__name__}: {error}",
                ),
            )
            raise
