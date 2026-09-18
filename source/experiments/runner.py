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

import numpy as np

from .artifacts import fit_directory, resolved_fit
from .cache import file_digest, writer_lock
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
)

LOGGER = logging.getLogger(__name__)


def population_order(inventory: dict[str, list[str]], seed: int) -> list[str]:
    """Prioritize common IDs then remaining IDs, using one seeded ordering."""
    if not inventory or any(not ids for ids in inventory.values()):
        raise ValueError("Expected a nonempty inventory for every session.")
    if any(len(set(ids)) != len(ids) for ids in inventory.values()):
        raise ValueError("Session inventory contains duplicate channel IDs.")
    populations = [set(ids) for ids in inventory.values()]
    common = sorted(set.intersection(*populations))
    remaining = sorted(set.union(*populations) - set(common))
    rng = np.random.default_rng(seed)
    return list(rng.permutation(common)) + list(rng.permutation(remaining))


def case_identity(
    features: FeatureSet, fold: int, case: dict, snapshots: dict,
    shared_order: list[str],
) -> tuple[dict, np.ndarray]:
    """Resolve effective fitting inputs before any artifact lookup."""
    from .populations import population_columns

    ids = features.arrays["ids"].tolist()
    columns = population_columns(ids, shared_order, case["population_scale"])
    identity = dict(
        configurations=snapshots, case=case, fold=fold,
        source=features.metadata, selected_ids=[ids[i] for i in columns],
        model_overrides=dict(epoch_artifacts=True),
    )
    identity["resolved_fit"] = resolved_fit(identity, features)
    return identity, columns


def plan_analysis(
    dataset: Dataset, sessions: list[str], folds: list[int],
    cases: list[dict], snapshots: dict, shared_order: list[str], root: Path,
) -> Path:
    """Resolve all requested fit references and scoring IDs before training."""
    from .analysis import initialize_analysis, member_key
    from .populations import scoring_channels

    members = {}
    for name in sessions:
        session = dataset.load(name)
        for fold in folds:
            features = dataset.fold(
                session, fold, snapshots["data"]["infer_velocity"]
            )
            common = scoring_channels(
                features.arrays["ids"].tolist(), shared_order, cases
            )
            for case in cases:
                identity, _ = case_identity(
                    features, fold, case, snapshots, shared_order
                )
                run = fit_directory(root, identity)
                members[member_key(case, name, fold)] = dict(
                    case=case, session=name, fold=fold, fit_id=run.name,
                    fit=str(run.relative_to(root.resolve())),
                    common_ids=common, state="pending",
                )
    return initialize_analysis(root, snapshots, members)


def run_case(
    dataset: Dataset, session: FeatureSet, fold: int, case: dict,
    settings: dict, arguments: argparse.Namespace, shared_order: list[str],
    snapshots: dict, gpu: dict | None, root: Path, analysis_root: Path,
) -> bool:
    """Complete a shared fit and publish this analysis's scoring separately."""
    from .analysis import member_key, read_manifest
    from .fitting import canonical_horizons, ensure_fit, ensure_predictions

    features = dataset.fold(session, fold, settings["velocity"])
    identity, columns = case_identity(
        features, fold, case, snapshots, shared_order
    )
    run = fit_directory(root, identity)
    key = member_key(case, session.metadata["session"], fold)
    member = read_manifest(analysis_root)["members"][key]
    if member["fit_id"] != run.name:
        raise ValueError(f"Analysis fit provenance changed: {run}")
    previews = copy.deepcopy(settings["previews"])
    if arguments.no_previews:
        previews["enabled"] = False
    trained = ensure_fit(
        run, identity, features, columns, arguments, previews, gpu
    )
    horizons = canonical_horizons(settings["horizons"])
    predictions = ensure_predictions(
        run, identity, features, columns, horizons
    )
    scored = score_member(
        analysis_root, key, run, predictions, identity, horizons, root
    )
    render_case(
        session, features, previews, run, columns, snapshots,
        arguments, analysis_root, key,
    )
    return trained or scored


def score_member(
    analysis_root: Path, key: str, run: Path, predictions: Path,
    identity: dict, horizons: list[int], root: Path,
) -> bool:
    """Publish missing analysis metrics from a completed forecast bundle."""
    from .analysis import (
        prepare_metrics, read_manifest, update_member, validate_member,
    )
    from .evaluation import evaluate_forecasts
    from .populations import scoring_indices

    case, fold = identity["case"], identity["fold"]
    with writer_lock(analysis_root / "metrics" / key / "evaluation.lock"):
        member = read_manifest(analysis_root)["members"][key]
        if member.get("state") == "complete":
            validate_member(analysis_root, member)
            return False
        try:
            with np.load(predictions, allow_pickle=False) as saved:
                payload = dict(saved)
            truth = dict(
                Y=payload["true_Y"], Z=payload["true_Z"],
                t=payload["t"], indices=payload["source_indices"],
            )
            selected_ids = identity["selected_ids"]
            metadata = dict(
                session=member["session"], fold=fold, fit_id=run.name,
                configuration=case["name"], case_name=case["name"],
                hyper_parameters=case["dimensions"],
                **case["summary_parameters"],
                population_scale=case["population_scale"],
                neural_channels=len(selected_ids),
                behavior_dimensions=truth["Z"].shape[1],
                selected_channel_ids=selected_ids,
            )
            destination = prepare_metrics(analysis_root, key)
            common = scoring_indices(selected_ids, member["common_ids"])
            evaluate_forecasts(
                payload, truth, horizons, common, metadata, destination,
                dict(Y=payload["baseline_Y"], Z=payload["baseline_Z"]),
            )
            metrics = destination / "metrics.json"
            update_member(
                analysis_root, key,
                dict(
                    state="complete",
                    metrics=str(metrics.relative_to(analysis_root)),
                    metrics_sha256=file_digest(metrics),
                    predictions=str(predictions.relative_to(root.resolve())),
                    predictions_sha256=file_digest(predictions),
                ),
            )
        except BaseException as error:
            update_member(
                analysis_root, key,
                dict(state="failed", error=f"{type(error).__name__}: {error}"),
            )
            raise
    return True


def render_case(
    session: FeatureSet, features: FeatureSet, previews: dict, run: Path,
    columns: np.ndarray, snapshots: dict, arguments: argparse.Namespace,
    analysis_root: Path, key: str,
) -> None:
    """Render analysis-owned figures without modifying shared fit payloads."""
    if previews["enabled"]:
        from .previews import fitted_previews

        preprocessing_previews(
            session, features, previews, analysis_root / "previews" / key
        )
        fitted_previews(
            session, features, previews, run,
            snapshots["experiment"]["preview_model_plugin"], columns,
            analysis_root / "previews" / key,
        )
    if not arguments.no_plots:
        from .history import summarize_histories

        destination = analysis_root / "plots" / "training" / key
        if not (destination / "stage_loss_summary.json").exists():
            summarize_histories(run, destination, plots=True)


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
    from .implementation import implementation_signatures

    snapshots.update(implementation_signatures())
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
    root = Path(snapshots["paths"]["artifact_root"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if arguments.stage == "plot":
        from .analysis import find_analysis

        analysis_root = find_analysis(root, snapshots, arguments.analysis_id)
        plugin(
            experiment["report_plugin"],
            root=analysis_root,
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
    analysis_root = None
    if cases:
        analysis_root = plan_analysis(
            dataset, selected, folds, cases, snapshots, shared, root
        )
        LOGGER.info("Analysis artifacts: %s", analysis_root)
        from .model_summary import write_model_summary

        write_model_summary(analysis_root)
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
                                root / "analysis" / experiment["name"],
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
                                        dataset, source, fold, case, settings,
                                        arguments, shared, snapshots, gpu,
                                        root, analysis_root,
                                    )
                                )
                            except BaseException as error:
                                from .analysis import (
                                    member_key, read_manifest, update_member,
                                )

                                key = member_key(case, session, fold)
                                member = read_manifest(
                                    analysis_root
                                )["members"][key]
                                if member["state"] != "complete":
                                    update_member(
                                        analysis_root, key,
                                        dict(
                                            state="failed",
                                            error=str(error),
                                        ),
                                    )
                                raise
                            finally:
                                write_model_summary(analysis_root)
                        fold_state["skipped"] = not any(changed)
    if arguments.stage != "preprocess":
        plugin(
            experiment["report_plugin"],
            root=analysis_root,
            settings=plotting,
            sample_rate=data["sampling_rate_hz"],
        )


if __name__ == "__main__":
    main()
