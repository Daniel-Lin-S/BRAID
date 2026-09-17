"""Run pluggable experiments from a manifest and local machine settings.

Inputs: data/model/evaluation/plotting configurations and optional CLI
profile overrides. Outputs: immutable configuration-addressed run folders,
cache references, component histories/checkpoints, predictions, metrics,
and summary/plot artifacts. Text logs are routed separately.

Usage: python -m experiments.runner --experiment YAML --stage STAGE
Paths printed to the console are always absolute.
"""

import argparse
import copy
import json
import logging
from importlib.metadata import version
import platform
import os
from pathlib import Path
import random
import subprocess

import numpy as np
import yaml

from .cache import atomic_json, file_digest, fingerprint, writer_lock
from .contracts import Dataset, FeatureSet, plugin
from .previews import preprocessing_previews
from .runtime import (
    case_logging,
    configure_gpu,
    configure_logging,
    launch_directory,
)
from .configuration import (
    argument_parser,
    resolve_configuration,
    REPOSITORY,
)
from .windows import window_indices

LOGGER = logging.getLogger(__name__)


def population_order(inventory: dict[str, list[str]], seed: int) -> list[str]:
    """Prioritize common IDs then remaining IDs, using one seeded ordering."""
    populations = [set(ids) for ids in inventory.values()]
    common = sorted(set.intersection(*populations))
    remaining = sorted(set.union(*populations) - set(common))
    rng = np.random.default_rng(seed)
    return list(rng.permutation(common)) + list(rng.permutation(remaining))


def run_case(
    dataset: Dataset,
    session: FeatureSet,
    fold: int,
    case: dict,
    settings: dict,
    arguments: argparse.Namespace,
    shared_order: list[str],
    snapshots: dict,
    gpu: dict,
    root: Path,
) -> None:
    """Fit or evaluate one configuration through the model protocol."""
    from .evaluation import evaluate_forecasts

    features = dataset.fold(session, fold, settings["velocity"])
    preview_settings = copy.deepcopy(settings["previews"])
    if arguments.no_previews:
        preview_settings["enabled"] = False
    preprocessing_previews(
        session, features, preview_settings, root / "uncached_previews"
    )
    ids = features.arrays["ids"].tolist()
    ordered = [ids.index(name) for name in shared_order if name in ids]
    count = int(np.floor(len(ordered) * case["population_scale"]))
    smallest = int(np.floor(len(ordered) * settings["neural_scoring_fraction"]))
    if min(count, smallest) < 1:
        raise ValueError("Population selection contains no neural outputs.")
    columns = np.array(ordered[:count])
    model_overrides = dict(epoch_artifacts=True)
    model_overrides["verbose"] = arguments.log_level in ("DEBUG", "INFO")
    numeric_settings = copy.deepcopy(snapshots)
    numeric_settings.pop("plotting")
    numeric_settings["data"].pop("previews")
    numeric_settings.pop("paths")
    numeric_settings.pop("runtime")
    identity = dict(
        configurations=numeric_settings,
        case=case,
        fold=fold,
        source=features.metadata,
        selected_ids=[ids[i] for i in columns],
        model_overrides={
            k: v for k, v in model_overrides.items() if k != "verbose"
        },
    )
    key = fingerprint(identity)
    directory = root / session.metadata["session"] / f"fold_{fold}" / key[:16]
    directory.mkdir(parents=True, exist_ok=True)
    log_directory = (
        arguments.log_directory
        / "sessions"
        / session.metadata["session"]
        / f"fold_{fold}"
        / f"{case['name']}-{key[:16]}"
    )
    with writer_lock(directory / "run.lock"), case_logging(log_directory):
        status_path = directory / "status.json"
        if status_path.exists():
            status = json.loads(status_path.read_text())
            if status["state"] == "complete":
                recorded = json.loads((directory / "identity.json").read_text())
                if recorded != identity:
                    raise ValueError("Completed experiment identity mismatch.")
                for filename, checksum in status["checksums"].items():
                    if file_digest(directory / filename) != checksum:
                        raise ValueError(
                            "Completed artifact checksum mismatch: "
                            f"{directory / filename}"
                        )
                LOGGER.info("Reusing completed experiment %s", directory)
                return
        model_configuration = directory / "model_configuration.yaml"
        model_configuration.write_text(yaml.safe_dump(snapshots["model"]))
        atomic_json(directory / "identity.json", identity)
        atomic_json(directory / "resolved_configurations.json", snapshots)
        atomic_json(
            directory / "runtime.json",
            dict(
                gpu=gpu,
                pid=os.getpid(),
                seed=settings["seed"],
                code_commit=subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True
                ).strip(),
                code_diff=subprocess.check_output(
                    ["git", "diff"], cwd=REPOSITORY, text=True
                ),
                cache=str(features.path) if features.path else None,
                text_log_directory=str(log_directory),
                independent_windows=True,
            ),
        )
        np.savez_compressed(
            directory / "selection.npz",
            selected_columns=columns,
            channel_ids=features.arrays["ids"][columns],
            unit_dimensions=features.arrays["units"][columns],
            indices=features.arrays["indices"],
            role=features.arrays["role"],
            segment=features.arrays["segment"],
        )
        atomic_json(status_path, dict(state="running", pid=os.getpid()))
        try:
            run_seed = int(
                fingerprint(
                    dict(
                        session=session.metadata["session"],
                        fold=fold,
                        case=case,
                        seed=settings["seed"],
                    )
                )[:8],
                16,
            ) % (2**31 - 1)
            atomic_json(directory / "seed.json", {"seed": run_seed})
            backend = plugin(
                settings["model_plugin"],
                configuration=str(model_configuration),
                overrides=model_overrides,
                seed=run_seed,
                previews=preview_settings,
            )
            checkpoint = directory / "model.p"
            if arguments.stage == "evaluate":
                backend.load(checkpoint)
            else:
                backend.fit(features, columns, case["dimensions"], directory)
                backend.save(checkpoint)
                if preview_settings["enabled"]:
                    from .previews import fitted_previews

                    revision = fitted_previews(
                        session,
                        features,
                        preview_settings,
                        directory,
                        snapshots["experiment"]["preview_model_plugin"],
                        columns,
                    )
                    atomic_json(
                        directory / "preview_reference.json",
                        {"directory": str(revision)},
                    )
            from .history import summarize_histories

            summarize_histories(directory, plots=not arguments.no_plots)
            test = window_indices(features, 2, backend.length).ravel()
            arrays = features.arrays
            truth = {
                name: arrays[name][test] for name in ("Y", "Z", "t", "indices")
            }
            truth["Y"] = truth["Y"][:, columns]
            predictions = backend.predict(
                truth["Y"], arrays["U"][test], settings["horizons"]
            )
            # Validate native save/load reconstruction on a real test window.
            restored = plugin(
                settings["model_plugin"],
                configuration=str(model_configuration),
                overrides=model_overrides,
                seed=run_seed,
                previews=preview_settings,
            )
            restored.load(checkpoint)
            length = backend.length
            check = restored.predict(
                truth["Y"][:length],
                arrays["U"][test[:length]],
                settings["horizons"],
            )
            for name in ("Y", "Z"):
                np.testing.assert_allclose(
                    check[name],
                    predictions[name][:, :length],
                    rtol=1e-5,
                    atol=1e-5,
                )
            metadata = dict(
                session=session.metadata["session"],
                fold=fold,
                configuration=case["name"],
                hyper_parameters=case["dimensions"],
                **case["summary_parameters"],
                population_scale=case["population_scale"],
                neural_channels=len(columns),
                behavior_dimensions=truth["Z"].shape[1],
            )
            with np.load(directory / "fit_indices.npz") as fit_indices:
                source_indices = fit_indices["training"].ravel()
            training = np.searchsorted(arrays["indices"], source_indices)
            baseline = dict(
                Y=arrays["Y"][training][:, columns].mean(axis=0),
                Z=arrays["Z"][training].mean(axis=0),
            )
            evaluate_forecasts(
                predictions,
                truth,
                settings["horizons"],
                np.arange(smallest),
                metadata,
                directory,
                baseline,
            )
            atomic_json(
                status_path,
                dict(
                    state="complete",
                    pid=os.getpid(),
                    checkpoint_reload_verified=True,
                    checksums={
                        filename: file_digest(directory / filename)
                        for filename in (
                            "model.p",
                            "metrics.json",
                            "predictions.npz",
                        )
                    },
                ),
            )
            LOGGER.info("Completed experiment %s", directory)
        except BaseException as error:
            atomic_json(
                status_path,
                dict(
                    state="failed",
                    pid=os.getpid(),
                    error=f"{type(error).__name__}: {error}",
                ),
            )
            raise


def main() -> None:
    """Execute the selected experiment stage using resolved configuration."""
    arguments = argument_parser().parse_args()
    snapshots = resolve_configuration(arguments)
    if arguments.dry_run:
        print(json.dumps(snapshots, indent=2))
        return
    inherited = os.environ.get("NHP_LAUNCH_LOG_DIR")
    arguments.log_directory = (
        Path(inherited).resolve()
        if inherited
        else launch_directory(snapshots, arguments.stage)
    )
    runtime = snapshots["runtime"]
    arguments.log_level = runtime["log_level"]
    configure_logging(arguments.log_directory, arguments.log_level)
    if arguments.stage == "preview":
        from .preview_regeneration import regenerate_previews

        regenerate_previews(snapshots)
        return
    experiment = snapshots["experiment"]
    data = snapshots["data"]
    evaluation = snapshots["evaluation"]
    plotting = snapshots["plotting"]
    snapshots["implementation"] = fingerprint(
        {
            str(path.relative_to(REPOSITORY)): file_digest(path)
            for path in sorted((REPOSITORY / "source").rglob("*.py"))
        }
    )
    snapshots["versions"] = dict(
        python=platform.python_version(),
        packages={
            name: version(name)
            for name in (
                "tensorflow",
                "tf-keras",
                "numpy",
                "scipy",
                "h5py",
                "scikit-learn",
                "PyYAML",
            )
        },
    )
    root = Path(snapshots["paths"]["artifact_root"]) / experiment["name"]
    root.mkdir(parents=True, exist_ok=True)
    if arguments.stage == "plot":
        plugin(
            experiment["report_plugin"],
            root=root,
            settings=plotting,
            sample_rate=data["sampling_rate_hz"],
        )
        return
    dataset = plugin(data["plugin"], settings=data)
    sessions = dataset.sessions()
    selected = (
        arguments.session or experiment["selection"]["sessions"] or sessions
    )
    if set(selected) - set(sessions):
        raise ValueError("Requested session is not in the configured dataset.")
    inventory = dataset.inventory()
    atomic_json(
        root / "sessions.json",
        dict(
            ordered_sessions=sessions,
            selected_sessions=selected,
            inventory=inventory,
        ),
    )
    shared = population_order(inventory, experiment["seed"])
    random.seed(experiment["seed"])
    np.random.seed(experiment["seed"])
    gpu = None
    if arguments.stage != "preprocess":
        gpu = configure_gpu(runtime["device"], runtime["cpu_threads"])
    cases = (
        []
        if arguments.stage == "preprocess"
        else plugin(experiment["suite_plugin"], settings=experiment["suite"])
    )
    settings = dict(
        evaluation,
        seed=experiment["seed"],
        model_plugin=experiment["model_plugin"],
        velocity=data["infer_velocity"],
        previews=data["previews"],
    )
    folds = (
        arguments.fold
        or experiment["selection"]["folds"]
        or list(range(data["cv"]["folds"]))
    )
    for session in selected:
        source = dataset.load(session)
        for fold in folds:
            if arguments.stage == "preprocess":
                features = dataset.fold(source, fold, data["infer_velocity"])
                preprocessing_previews(
                    source,
                    features,
                    data["previews"],
                    root / "uncached_previews",
                )
                LOGGER.info(
                    "Cached %s fold %s at %s", session, fold, features.path
                )
                continue
            for case in cases:
                run_case(
                    dataset,
                    source,
                    fold,
                    case,
                    settings,
                    arguments,
                    shared,
                    snapshots,
                    gpu,
                    root,
                )
    if arguments.stage != "preprocess":
        plugin(
            experiment["report_plugin"],
            root=root,
            settings=plotting,
            sample_rate=data["sampling_rate_hz"],
        )


if __name__ == "__main__":
    main()
