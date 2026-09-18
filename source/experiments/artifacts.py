"""Resolve structured run artifacts and read historical flat layouts.

New runs group settings in configuration/, identities and execution state in
provenance/, selections in data/, fitted models in checkpoints/, scores and
predictions in evaluation/, and fit summaries in diagnostics/. Readers accept
historical flat runs without modifying them. Model directory identities omit
machine paths, session selections and presentation preferences.
"""

import copy
import json
from importlib import import_module
from pathlib import Path

from .cache import fingerprint, file_digest
from .contracts import FeatureSet


ARTIFACT_GROUPS = {
    "configuration": (
        "model_configuration.yaml",
        "resolved_configurations.json",
        "fit_arguments.json",
    ),
    "provenance": (
        "identity.json",
        "runtime.json",
        "seed.json",
        "status.json",
        "preview_reference.json",
        "fitted_excerpts.json",
        "training_deviations.json",
        "fit_complete.json",
    ),
    "data": ("selection.npz", "fit_indices.npz"),
    "checkpoints": ("model.p",),
    "evaluation": ("metrics.json", "predictions.npz"),
    "diagnostics": ("stage_loss_summary.json",),
}
ARTIFACT_PATHS = {
    name: Path(group) / name
    for group, names in ARTIFACT_GROUPS.items()
    for name in names
}


def resolved_fit(identity: dict, features: FeatureSet | None = None) -> dict:
    """Obtain semantic fit arguments through the configured model adapter."""
    snapshots = identity["configurations"]
    arguments = copy.deepcopy(identity.get("resolved_fit"))
    if arguments is None:
        module, name = snapshots["experiment"]["model_plugin"].split(":")
        adapter = getattr(import_module(module), name)
        arguments = adapter.resolve_fit_configuration(
            snapshots["model"],
            identity["case"].get("dimensions", {}),
            identity.get("model_overrides"),
            features,
        )
    for key in ("verbose", "save_logs", "epoch_artifacts", "clear_graph"):
        arguments.get("args_base", {}).pop(key, None)
    return arguments


def artifact_path(run: Path, name: str) -> Path:
    """Resolve a named artifact, preferring an existing historical file.

    Parameters
    ----------
    run : Path
        Root of a single run, never a model-settings directory.
    name : str
        Registered artifact filename.

    Returns
    -------
    Path
        Absolute canonical path; this function never creates directories.
    """
    if name not in ARTIFACT_PATHS:
        raise ValueError(f"Unknown run artifact: {name}")
    legacy = run / name
    current = run / ARTIFACT_PATHS[name]
    if legacy.exists() and current.exists():
        raise ValueError(
            f"Ambiguous duplicate artifact: {run.resolve()}/{name}"
        )
    return (legacy if legacy.exists() else current).resolve()


def prepare_run(run: Path) -> None:
    """Create the named artifact groups for a new or incomplete run."""
    for group in ARTIFACT_GROUPS:
        (run / group).mkdir(parents=True, exist_ok=True)


def discover_runs(root: Path) -> list[Path]:
    """Find new and historical runs through their identity documents."""
    runs = set()
    for path in root.rglob("identity.json"):
        run = (
            path.parent.parent
            if path.parent.name == "provenance"
            else path.parent
        )
        if artifact_path(run, "identity.json") == path.resolve():
            runs.add(run.resolve())
    return sorted(runs)


def model_identity(snapshots: dict, case: dict) -> dict:
    """Extract settings shared across sessions and folds for one model case."""
    data = copy.deepcopy(snapshots["data"])
    for key in ("root", "cache_root", "cache_mode", "previews"):
        data.pop(key, None)
    experiment = snapshots["experiment"]
    model = resolved_fit(dict(configurations=snapshots, case=case))
    return dict(
        model=model,
        data=data,
        population_scale=case.get("population_scale", 1.0),
        evaluation=snapshots["evaluation"],
        seed=experiment["seed"],
        model_plugin=experiment["model_plugin"],
        versions=snapshots.get("versions"),
    )


def model_directory(root: Path, snapshots: dict, case: dict) -> Path:
    """Return a readable model-settings directory with a stable fingerprint."""
    return (
        root
        / f"{case['name']}-{fingerprint(model_identity(snapshots, case))[:16]}"
    )


def compatibility_identity(identity: dict) -> dict:
    """Compare numerical settings and source provenance across layouts.

    Paths, presentation and invocation settings do not change a fit. Package
    versions and all scientific settings remain part of compatibility.
    """
    source = copy.deepcopy(identity["source"])
    source.pop("source", None)
    source.pop("session_cache", None)
    if "sha256" in source.get("identity", {}):
        source["identity"].pop("source", None)
    for key in ("settings", "identity"):
        if key in source:
            for field in (
                "root",
                "cache_root",
                "cache_mode",
                "previews",
                "path",
            ):
                source[key].pop(field, None)
    model = model_identity(identity["configurations"], identity["case"])
    model["model"] = resolved_fit(identity)
    return dict(
        model=model,
        fold=identity["fold"],
        source=source,
        selected_ids=identity["selected_ids"],
        fit_seed=fit_seed(identity),
        # Overrides have already been applied to the resolved arguments.
    )


def validate_completion(run: Path, status: dict) -> None:
    """Check every registered completed payload before reusing a run."""
    if not status.get("checksums"):
        raise ValueError(f"Completion has no payload checksums: {run}")
    for name, digest in status["checksums"].items():
        path = artifact_path(run, name)
        if file_digest(path) != digest:
            raise ValueError(f"Completed artifact checksum mismatch: {path}")


def compatible_run(root: Path, identity: dict) -> Path | None:
    """Find an existing compatible completed or interrupted run, read-only."""
    expected = compatibility_identity(identity)
    complete, partial = [], []
    for run in discover_runs(root):
        existing = json.loads(artifact_path(run, "identity.json").read_text())
        fit_path = artifact_path(run, "fit_arguments.json")
        if "resolved_fit" not in existing and fit_path.exists():
            existing["resolved_fit"] = json.loads(fit_path.read_text())
        if compatibility_identity(existing) != expected:
            continue
        path = artifact_path(run, "status.json")
        status = json.loads(path.read_text()) if path.exists() else {}
        if status.get("state") == "complete":
            validate_completion(run, status)
            complete.append(run)
        else:
            partial.append(run)
    if len(complete) > 1 or (not complete and len(partial) > 1):
        raise ValueError(
            "Multiple compatible runs found; "
            "choose an unambiguous artifact root."
        )
    return (complete or partial or [None])[0]


def fit_seed(identity: dict) -> int:
    """Resolve the per-fit seed used by BRAID, including historical cases."""
    return int(
        fingerprint(
            dict(
                session=identity["source"]["session"],
                fold=identity["fold"],
                case=identity["case"],
                seed=identity["configurations"]["experiment"]["seed"],
            )
        )[:8],
        16,
    ) % (2**31 - 1)
