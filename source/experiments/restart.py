"""Clean incomplete component outputs before a locked experiment restart.

Input is the components directory of a compatible fold. Completed checkpoint
and history checksums are validated before any deletion. Incomplete components
retain component.json; recognized partial outputs are deleted, never copied.
The caller must hold the fold run lock throughout preparation and fitting.
"""

import json
from pathlib import Path
import re
import shutil

from .cache import file_digest

PARTIAL_FILES = {
    "history.jsonl",
    "initial.weights.h5",
    "latest.weights.h5",
    "pending.weights.h5",
    "completed.weights.h5",
    "completed.pending.weights.h5",
    "complete.pending.json",
    "architecture.json",
}
TIMESTAMP = re.compile(r"\d{8}-\d{6}")
PARTIAL_DIRECTORIES = {
    "train",
    "validation",
}


def prepare_components(root: Path) -> None:
    """Validate all components, then remove recognized incomplete outputs."""
    deletions = []
    for definition in sorted(root.rglob("component.json")):
        directory = definition.parent
        entries = list(directory.iterdir())
        if any(path.is_symlink() for path in entries):
            raise ValueError(f"Component contains a symbolic link: {directory}")
        complete = directory / "complete.json"
        if complete.exists():
            record = json.loads(complete.read_text())
            checksums = record.get("checksums", {})
            required = {"completed.weights.h5", "history.jsonl"}
            if not required.issubset(checksums):
                raise ValueError(f"Incomplete completion checksums: {complete}")
            for name, digest in checksums.items():
                if Path(name).name != name:
                    raise ValueError(f"Invalid checksum filename: {complete}")
                if file_digest(directory / name) != digest:
                    raise ValueError(
                        f"Completed checksum mismatch: {directory / name}"
                    )
            continue
        for path in entries:
            if path.name == "component.json":
                continue
            known_file = (
                path.name in PARTIAL_FILES
                or path.name.startswith("events.out.tfevents.")
                or re.fullmatch(r"loss_attempt_\d+\.png", path.name)
            )
            known_directory = TIMESTAMP.fullmatch(path.name)
            if not (
                (path.is_file() and known_file)
                or (
                    path.is_dir()
                    and (known_directory or path.name in PARTIAL_DIRECTORIES)
                )
            ):
                raise ValueError(f"Unrecognized incomplete artifact: {path}")
            deletions.append(path)
    for path in deletions:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
