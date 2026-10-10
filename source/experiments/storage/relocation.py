"""Preserve cache identity for certified import-only package relocations.

Exact relocated-file checksums map to their prior flat-file checksums.
Any subsequent source edit automatically receives its new checksum.
This registry never changes a saved artifact or accepts numerical changes.
"""

from pathlib import Path

from .cache import file_digest

RELOCATIONS = dict([
    (
        "95a1d8cb89d6f3ddb954127fe8930a52b528ceaf4b63aba63201f314623c5e44",
        "16ca0cc011025084b132c3d183b079b2d1c4541e3a3949c6dc501ede4086840a",
    ),
    (
        "72c2995eba389227952169aacd1d6c7baff058cb42bef76aeb06402f8e9b50b0",
        "aa26a72e6f73d9a6b98b85768aae95913ba50518894acc64a7f627be7c8b3893",
    ),
    (
        "7c46cf33acdc78ea5d502832dec74890211bcc9937a8f976b0dc08ad0a7c9f2b",
        "4779e57d70c84328da68ca992562b20002df32e2b839c24ecb2b59b4643e07ef",
    ),
])


def preprocessing_digest(path: Path) -> str:
    """Return the scientific cache digest for a relocated adapter file."""
    digest = file_digest(path)
    return RELOCATIONS.get(digest, digest)
