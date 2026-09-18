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

from .artifacts import (
    artifact_path,
    compatible_run,
    compatibility_identity,
    resolved_fit,
    fit_seed,
    model_directory,
    model_identity,
    prepare_run,
)
from .cache import atomic_json, file_digest, fingerprint, writer_lock
from .contracts import Dataset, FeatureSet, plugin
from .previews import preprocessing_previews
from .runtime import (
    session_logging,
    lifecycle_scope,
    configure_device,
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
) -> bool:
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
    identity["resolved_fit"] = resolved_fit(identity, features)
    key = fingerprint(compatibility_identity(identity))
    model_root = model_directory(root, snapshots, case)
    directory = (
        model_root / session.metadata["session"] / f"fold_{fold}" / key[:16]
    )
    existing = compatible_run(root, identity)
    if existing is not None:
        directory = existing
    from .model_summary import register_model_run

    register_model_run(model_root, directory)
    directory.mkdir(parents=True, exist_ok=True)
    session_log = (
        arguments.log_directory
        / "sessions"
        / f"{session.metadata['session']}.log"
    )
    LOGGER.info("Model case %s fold %s at %s", case["name"], fold, directory)
    with writer_lock(directory / "run.lock"):
        status_path = artifact_path(directory, "status.json")
        if status_path.exists():
            status = json.loads(status_path.read_text())
            if status["state"] == "complete":
                recorded = json.loads(
                    (artifact_path(directory, "identity.json")).read_text()
                )
                fit_path = artifact_path(directory, "fit_arguments.json")
                if "resolved_fit" not in recorded and fit_path.exists():
                    recorded["resolved_fit"] = json.loads(fit_path.read_text())
                if compatibility_identity(recorded) != compatibility_identity(
                    identity
                ):
                    raise ValueError("Completed experiment identity mismatch.")
                for filename, checksum in status["checksums"].items():
                    if (
                        file_digest(artifact_path(directory, filename))
                        != checksum
                    ):
                        raise ValueError(
                            "Completed artifact checksum mismatch: "
                            f"{directory / filename}"
                        )
                LOGGER.info("Reusing completed experiment %s", directory)
                return False
        prepare_run(directory)
        model_configuration = artifact_path(
            directory, "model_configuration.yaml"
        )
        model_configuration.write_text(yaml.safe_dump(snapshots["model"]))
        atomic_json(artifact_path(directory, "identity.json"), identity)
        atomic_json(
            artifact_path(directory, "resolved_configurations.json"), snapshots
        )
        atomic_json(
            artifact_path(directory, "runtime.json"),
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
                text_log=str(session_log.resolve()),
                independent_windows=True,
            ),
        )
        np.savez_compressed(
            artifact_path(directory, "selection.npz"),
            selected_columns=columns,
            channel_ids=features.arrays["ids"][columns],
            unit_dimensions=features.arrays["units"][columns],
            indices=features.arrays["indices"],
            role=features.arrays["role"],
            segment=features.arrays["segment"],
        )
        atomic_json(status_path, dict(state="running", pid=os.getpid()))
        try:
            run_seed = fit_seed(identity)
            atomic_json(
                artifact_path(directory, "seed.json"), {"seed": run_seed}
            )
            backend = plugin(
                settings["model_plugin"],
                configuration=str(model_configuration),
                overrides=model_overrides,
                seed=run_seed,
                previews=preview_settings,
            )
            checkpoint = artifact_path(directory, "model.p")
            fit_complete = artifact_path(directory, "fit_complete.json")
            if arguments.stage == "evaluate" or fit_complete.exists():
                if fit_complete.exists():
                    fit_record = json.loads(fit_complete.read_text())
                    if (
                        file_digest(checkpoint)
                        != fit_record["checkpoint_sha256"]
                    ):
                        raise ValueError(
                            f"Fitted checkpoint checksum mismatch: {checkpoint}"
                        )
                backend.load(checkpoint)
            else:
                backend.fit(features, columns, case["dimensions"], directory)
                backend.save(checkpoint)
                atomic_json(
                    fit_complete, {"checkpoint_sha256": file_digest(checkpoint)}
                )
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
                    artifact_path(directory, "preview_reference.json"),
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
                configuration=model_root.name,
                case_name=case["name"],
                model_settings=model_identity(snapshots, case),
                hyper_parameters=case["dimensions"],
                **case["summary_parameters"],
                population_scale=case["population_scale"],
                neural_channels=len(columns),
                behavior_dimensions=truth["Z"].shape[1],
            )
            with np.load(
                artifact_path(directory, "fit_indices.npz")
            ) as fit_indices:
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
                        filename: file_digest(
                            artifact_path(directory, filename)
                        )
                        for filename in (
                            "model.p",
                            "metrics.json",
                            "predictions.npz",
                        )
                    },
                ),
            )
            LOGGER.info("Completed experiment %s", directory)
            return True
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
        gpu = configure_device(
            runtime["device"],
            runtime["cpu_threads"],
            runtime["cpu_interop_threads"],
        )
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
    from .model_summary import register_model_work, write_model_summary

    if cases:
        for case in cases:
            register_model_work(root, snapshots, case, selected, folds)
            write_model_summary(model_directory(root, snapshots, case))
    for session in selected:
        with lifecycle_scope(arguments.log_directory, session):
            with session_logging(arguments.log_directory, session):
                source = dataset.load(session)
                for fold in folds:
                    with lifecycle_scope(
                        arguments.log_directory,
                        session,
                        fold,
                    ) as fold_state:
                        if arguments.stage == "preprocess":
                            features = dataset.fold(
                                source,
                                fold,
                                data["infer_velocity"],
                            )
                            preprocessing_previews(
                                source,
                                features,
                                data["previews"],
                                root / "uncached_previews",
                            )
                            LOGGER.info(
                                "Cached fold %s at %s", fold, features.path
                            )
                            continue
                        changed = []
                        for case in cases:
                            try:
                                changed.append(
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
                                )
                            finally:
                                write_model_summary(
                                    model_directory(root, snapshots, case),
                                )
                        fold_state["skipped"] = not any(changed)
    if arguments.stage != "preprocess":
        plugin(
            experiment["report_plugin"],
            root=root,
            settings=plotting,
            sample_rate=data["sampling_rate_hz"],
        )


if __name__ == "__main__":
    main()
