"""Address immutable fits independently of experiment membership.

Fits live in experiments/<model>_<settings-hash>/<session>/fold_<k>/<fit-id>/.
model_settings.json beside the session directories contains settings_hash
and the complete canonical fitting recipe in settings. The
configuration, provenance, data and checkpoints groups contain settings,
identity/completion documents, selected channels/indices and model.p.
Predictions are completed bundles beneath each fit. Analysis never
contributes to fit identity and no other artifact layout is read.
"""

import copy
import json
from importlib import import_module
from pathlib import Path
import re

from .cache import atomic_json, fingerprint, file_digest, writer_lock
from .contracts import FeatureSet, Model
from .populations import POPULATION_SELECTION_POLICY

FIT_SCHEMA = 1
SEED_MODULUS = 2**31 - 1
PRESENTATION_KEYS = ("verbose", "save_logs", "epoch_artifacts", "clear_graph")
LOCATION_KEYS = ("root", "cache_root", "cache_mode", "previews", "path")
MODEL_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")
SETTINGS_FILENAME = "model_settings.json"
SETTINGS_LOCK = "model_settings.lock"
ARTIFACT_GROUPS = {
    "configuration": (
        "model_configuration.yaml", "resolved_configurations.json",
        "fit_arguments.json",
    ),
    "provenance": (
        "identity.json", "runtime.json", "seed.json", "status.json",
        "fitted_excerpts.json", "training_deviations.json",
        "fit_complete.json",
    ),
    "data": ("selection.npz", "fit_indices.npz"),
    "checkpoints": ("model.p",),
}
ARTIFACT_PATHS = {
    name: Path(group) / name
    for group, names in ARTIFACT_GROUPS.items()
    for name in names
}


def model_adapter(snapshots: dict) -> type[Model]:
    """Load an adapter and validate its explicitly declared public name.

    Parameters
    ----------
    snapshots : dict
        Resolved configurations containing experiment.model_plugin.

    Returns
    -------
    type of Model
        Adapter class with a nonempty, filesystem-safe model_name.
    """
    reference = snapshots["experiment"]["model_plugin"]
    module, separator, name = reference.partition(":")
    if not separator or not module or not name:
        raise ValueError(
            f"Expected module:class model plugin, got {reference!r}."
        )
    adapter = getattr(import_module(module), name)
    public_name = getattr(adapter, "model_name", None)
    if (
        not isinstance(public_name, str)
        or MODEL_NAME_PATTERN.fullmatch(public_name) is None
    ):
        raise ValueError(
            f"Model adapter {reference!r} must declare a nonempty "
            "filesystem-safe model_name using letters, digits, '-' or '_'."
        )
    return adapter


def resolved_fit(identity: dict, features: FeatureSet | None = None) -> dict:
    """Resolve effective numerical arguments, excluding presentation flags."""
    snapshots = identity["configurations"]
    arguments = copy.deepcopy(identity.get("resolved_fit"))
    if arguments is None:
        adapter = model_adapter(snapshots)
        arguments = adapter.resolve_fit_configuration(
            snapshots["model"], identity["case"].get("dimensions", {}),
            identity.get("model_overrides"), features,
        )
    for key in PRESENTATION_KEYS:
        arguments.get("args_base", {}).pop(key, None)
    return arguments


def artifact_path(run: Path, name: str) -> Path:
    """Return the absolute path of a registered fit artifact."""
    if name not in ARTIFACT_PATHS:
        raise ValueError(f"Unknown fit artifact: {name}")
    return (run / ARTIFACT_PATHS[name]).resolve()


def prepare_run(run: Path) -> None:
    """Create groups belonging to an incomplete fit."""
    for group in ARTIFACT_GROUPS:
        (run / group).mkdir(parents=True, exist_ok=True)


def scientific_source(source: dict) -> dict:
    """Remove storage locations and rendering preferences from provenance."""
    result = copy.deepcopy(source)
    if "sha256" in result:
        result.pop("source", None)
    for key in (*LOCATION_KEYS, "source", "session_cache"):
        if key == "source" and isinstance(result.get(key), dict):
            result[key] = scientific_source(result[key])
        else:
            result.pop(key, None)
    for key in ("settings", "identity"):
        if isinstance(result.get(key), dict):
            result[key] = scientific_source(result[key])
    return result


def model_identity(snapshots: dict, case: dict) -> dict:
    """Collect numerical context shared by seed and recipe resolution."""
    return dict(
        model=resolved_fit(dict(configurations=snapshots, case=case)),
        data=scientific_source(snapshots["data"]),
        seed=snapshots["experiment"]["seed"],
        model_plugin=snapshots["experiment"]["model_plugin"],
        versions=snapshots.get("versions"),
        fitting_implementation=snapshots.get("fitting_implementation"),
    )


def model_settings(
    snapshots: dict, case: dict, overrides: dict | None = None,
) -> dict:
    """Resolve the complete session-independent fitting recipe.

    Parameters
    ----------
    snapshots : dict
        Resolved model, preprocessing, seed and implementation settings.
    case : dict
        One case's dimension overrides and population_scale, not its sweep.
    overrides : dict, optional
        Fitting overrides; default is None. Presentation flags are excluded.

    Returns
    -------
    dict
        Canonical recipe using the configured batch limit, without capping
        against a particular session's available windows.
    """
    recipe = model_identity(snapshots, case)
    recipe["model"] = resolved_fit(dict(
        configurations=snapshots, case=case, model_overrides=overrides,
    ))
    recipe["data"].pop("sessions", None)
    scale = case["population_scale"]
    if isinstance(scale, bool) or not isinstance(scale, (int, float)):
        raise ValueError(f"Expected numeric population_scale, got {scale!r}.")
    if not 0 < scale <= 1:
        raise ValueError(f"Expected population_scale in (0, 1], got {scale!r}.")
    recipe["population"] = dict(
        scale=float(scale), selection_policy=POPULATION_SELECTION_POLICY,
    )
    return recipe


def settings_record(identity: dict) -> dict:
    """Build the canonical recipe document referenced by one fit.

    Parameters
    ----------
    identity : dict
        Fit inputs with configurations, case and optional model_overrides.

    Returns
    -------
    dict
        settings_hash (full SHA-256) and its canonical settings payload.
    """
    settings = model_settings(
        identity["configurations"], identity["case"],
        identity.get("model_overrides"),
    )
    return dict(settings_hash=fingerprint(settings), settings=settings)


def fit_inputs(identity: dict) -> dict:
    """Collect effective fitting inputs before deriving the random seed."""
    snapshots = identity["configurations"]
    model = model_identity(snapshots, identity["case"])
    model["model"] = resolved_fit(identity)
    return dict(
        schema=FIT_SCHEMA, model=model, fold=identity["fold"],
        source=scientific_source(identity["source"]),
        selected_ids=identity["selected_ids"],
    )


def fit_seed(identity: dict) -> int:
    """Derive a seed independently of sweep labels and scoring."""
    return int(fingerprint(fit_inputs(identity))[:8], 16) % SEED_MODULUS


def fit_identity(identity: dict) -> dict:
    """Return the sole canonical identity used for fit completion reuse."""
    return dict(
        fit_inputs(identity), fit_seed=fit_seed(identity),
        settings_hash=settings_record(identity)["settings_hash"],
    )


def model_directory(
    root: Path, snapshots: dict, case: dict, overrides: dict | None = None,
) -> Path:
    """Address a complete fitting recipe by public model name and SHA-256.

    Parameters
    ----------
    root : Path
        Shared artifact root.
    snapshots, case : dict
        Resolved configuration and this case's training settings.
    overrides : dict, optional
        Fitting overrides; default is None.

    Returns
    -------
    Path
        Absolute experiments/<model_name>_<full-settings-hash> directory.
    """
    adapter = model_adapter(snapshots)
    digest = fingerprint(model_settings(snapshots, case, overrides))
    return (root / "experiments" / f"{adapter.model_name}_{digest}").resolve()


def validate_model_settings(directory: Path) -> dict:
    """Validate immutable settings metadata against its directory name.

    Parameters
    ----------
    directory : Path
        Model-settings directory containing model_settings.json.

    Returns
    -------
    dict
        Validated settings_hash and settings document.
    """
    path = directory / SETTINGS_FILENAME
    record = json.loads(path.read_text())
    if set(record) != {"settings_hash", "settings"}:
        raise ValueError(f"Invalid model settings document: {path.resolve()}")
    settings = record["settings"]
    digest = fingerprint(settings)
    adapter = model_adapter(dict(
        experiment=dict(model_plugin=settings["model_plugin"])
    ))
    if (
        record["settings_hash"] != digest
        or directory.name != f"{adapter.model_name}_{digest}"
    ):
        raise ValueError(f"Model settings hash mismatch: {path.resolve()}")
    return record


def prepare_model_settings(directory: Path, identity: dict) -> None:
    """Publish a recipe once, rejecting existing missing or conflicting data.

    Parameters
    ----------
    directory : Path
        Canonical model-settings directory, not a fit directory.
    identity : dict
        Resolved fit inputs whose configured recipe must match the directory.
    """
    expected = settings_record(identity)
    adapter = model_adapter(identity["configurations"])
    if directory.name != (
        f"{adapter.model_name}_{expected['settings_hash']}"
    ):
        raise ValueError(f"Unexpected model settings directory: {directory}")
    with writer_lock(directory / SETTINGS_LOCK):
        path = directory / SETTINGS_FILENAME
        if path.exists():
            if validate_model_settings(directory) != expected:
                raise ValueError(f"Conflicting model settings: {path}")
        else:
            if any(item.name != SETTINGS_LOCK for item in directory.iterdir()):
                raise ValueError(f"Missing model settings metadata: {path}")
            atomic_json(path, expected)


def fit_directory(root: Path, identity: dict) -> Path:
    """Resolve a fit independently of analysis configuration and labels."""
    return (
        model_directory(
            root, identity["configurations"], identity["case"],
            identity.get("model_overrides"),
        )
        / identity["source"]["session"] / f"fold_{identity['fold']}"
        / fingerprint(fit_identity(identity))
    )


def validate_completion(run: Path) -> dict:
    """Validate every completed fit payload without writing to the fit."""
    settings = validate_model_settings(run.parents[2])
    path = artifact_path(run, "fit_complete.json")
    record = json.loads(path.read_text())
    required = {
        "model.p", "identity.json", "fit_indices.npz", "selection.npz",
        "model_configuration.yaml",
    }
    if (
        record.get("fit_id") != run.name
        or not required.issubset(record.get("checksums", {}))
    ):
        raise ValueError(f"Invalid fit completion record: {path}")
    for name, digest in record["checksums"].items():
        if file_digest(artifact_path(run, name)) != digest:
            raise ValueError(f"Completed fit checksum mismatch: {run / name}")
    identity = json.loads(artifact_path(run, "identity.json").read_text())
    if (
        identity["fit_id"] != run.name
        or fingerprint(identity["identity"]) != run.name
        or identity["identity"]["settings_hash"] != settings["settings_hash"]
    ):
        raise ValueError(f"Invalid completed fit identity: {run}")
    return record


def completed_fit(run: Path, identity: dict) -> bool:
    """Identify a reusable fit without creating directories or lock files.

    Parameters
    ----------
    run : Path
        Exact configuration-resolved fit directory; no fallback lookup.
    identity : dict
        Currently resolved scientific configuration and fold provenance.

    Returns
    -------
    bool
        False if incomplete; True only after identity and checksum checks.
    """
    if not artifact_path(run, "fit_complete.json").exists():
        return False
    validate_completion(run)
    recorded = json.loads(artifact_path(run, "identity.json").read_text())
    if recorded["identity"] != fit_identity(identity):
        raise ValueError(f"Completed fit identity mismatch: {run.resolve()}")
    return True
