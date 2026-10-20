"""Freeze and checksum campaign code and repository YAML defaults.

Input is a campaign root. Output is a frozen source/ and assets/ snapshot,
plus a digest covering Python and YAML files. The digest is audit metadata;
scientific fit IDs use explicit active module versions.
"""

from pathlib import Path
import shutil

from experiments.storage.cache import file_digest, fingerprint


SOURCE_ROOT = Path(__file__).resolve().parents[1]


def source_code_digest(root: Path | None = None) -> str:
    """Hash source and YAML defaults for a launch, excluding fit ID."""
    root = SOURCE_ROOT if root is None else root
    files = sorted(root.rglob("*.py"))
    if not files:
        raise ValueError(f"No Python source beneath {root.resolve()}.")
    configuration = root.parent / "assets" / "config"
    files.extend(sorted(configuration.rglob("*.yaml")))
    return fingerprint({
        str(path.relative_to(root.parent)): file_digest(path)
        for path in files
    })


def freeze_source(launch: Path) -> str:
    """Copy code and default configs into a launch-local immutable snapshot."""
    destination = launch / "source"
    assets = launch / "assets"
    if SOURCE_ROOT.resolve() != destination.resolve():
        if destination.exists() or assets.exists():
            raise FileExistsError(
                f"Launch source snapshot already exists: {launch.resolve()}."
            )
        shutil.copytree(
            SOURCE_ROOT, destination,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        shutil.copytree(SOURCE_ROOT.parent / "assets", assets)
    if not (assets / "config" / "default.yaml").is_file():
        raise FileNotFoundError(
            f"Missing frozen BRAID default configuration: {assets}."
        )
    return source_code_digest(destination)


def verify_source_code(expected: str) -> None:
    """Reject a launch whose frozen code or defaults changed."""
    actual = source_code_digest()
    if expected != actual:
        raise ValueError(
            f"Launch source checksum mismatch: expected {expected}, "
            f"got {actual}."
        )
