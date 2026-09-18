"""Hash saved artifacts without loading their complete contents into memory.

Input is a filesystem path; output is a hexadecimal content digest used in
cache, checkpoint and completion manifests. This module has no ML imports.
"""

import hashlib
from pathlib import Path

BUFFER_SIZE = 8 * 1024 * 1024


def file_digest(path: Path, algorithm: str = "sha256") -> str:
    """Hash a file incrementally without retaining its bytes in memory."""
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(BUFFER_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()
