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
import os
from pathlib import Path
import random

import numpy as np

from .artifacts import fit_directory, prepare_model_settings, resolved_fit
from .cache import file_digest, writer_lock
from .contracts import Dataset, FeatureSet, plugin
from .previews import preprocessing_previews
from .runtime import (
    session_logging,
    lifecycle_scope,
    stage_scope,
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
    features: FeatureSet,
    fold: int,
    case: dict,
    snapshots: dict,
    shared_order: list[str],
) -> tuple[dict, np.ndarray]:
    """Resolve effective fitting inputs before any artifact lookup."""
    from .populations import population_columns

    ids = features.arrays["ids"].tolist()
    columns = population_columns(ids, shared_order, case["population_scale"])
    identity = dict(
        configurations=snapshots,
        case=case,
        fold=fold,
        source=features.metadata,
        selected_ids=[ids[i] for i in columns],
        model_overrides=dict(epoch_artifacts=True),
    )
    identity["resolved_fit"] = resolved_fit(identity, features)
    return identity, columns


def plan_analysis(
    dataset: Dataset,
    sessions: list[str],
    folds: list[int],
    cases: list[dict],
    snapshots: dict,
    shared_order: list[str],
    root: Path,
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
                    case=case,
                    session=name,
                    fold=fold,
                    fit_id=run.name,
                    fit=str(run.relative_to(root.resolve())),
                    common_ids=common,
                    state="pending",
                )
    return initialize_analysis(root, snapshots, members)


def run_case(
    dataset: Dataset,
    session: FeatureSet,
    fold: int,
    case: dict,
    settings: dict,
    arguments: argparse.Namespace,
    shared_order: list[str],
    snapshots: dict,
    gpu: dict | None,
    root: Path,
    analysis_root: Path,
    rendering_errors: list[str] | None = None,
    progress: dict | None = None,
) -> bool:
    """Complete a shared fit and publish this analysis's scoring separately."""
    from .analysis import member_key, read_manifest
    from .fitting import canonical_horizons, ensure_fit, ensure_predictions
    from .diagnostics import history_monitor, render_safely
    from .presentation import presentation

    progress = {} if progress is None else progress
    progress["phase"] = "setup"
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
    style = presentation(snapshots["plotting"].get("presentation"))
    previews["presentation"] = style
    previews["context_samples"] = identity["resolved_fit"]["args_base"][
        "sequence_length"
    ]
    if arguments.no_previews:
        previews["enabled"] = False
    errors = [] if rendering_errors is None else rendering_errors
    prepare_model_settings(run.parents[2], identity)
    render_safely(
        errors,
        f"Preprocessing previews {run.resolve()}",
        preprocessing_previews,
        session,
        features,
        previews,
        run,
        columns,
    )
    progress["phase"] = "fit"
    with history_monitor(
        run,
        style,
        errors,
        enabled=not arguments.no_plots and snapshots["plotting"]["enabled"],
        regenerate=(
            snapshots.get("runtime", {}).get(
                "figure_regeneration", "incomplete"
            )
            == "all"
        ),
    ):
        trained = ensure_fit(run, identity, features, columns, arguments, gpu)
    from .previews import fitted_previews

    if previews["enabled"]:
        render_safely(
            errors,
            f"Fitted previews {run.resolve()}",
            fitted_previews,
            session,
            features,
            previews,
            run,
            snapshots["experiment"]["preview_model_plugin"],
            columns,
        )
    horizons = canonical_horizons(settings["horizons"])
    progress["phase"] = "prediction"
    predictions = ensure_predictions(
        run, identity, features, columns, horizons
    )
    progress["phase"] = "evaluation"
    scored = score_member(
        analysis_root, key, run, predictions, identity, horizons, root
    )
    progress["phase"] = "rendering"
    if rendering_errors is None and errors:
        raise RuntimeError("Rendering failed: " + "; ".join(errors))
    return trained or scored


def score_member(
    analysis_root: Path,
    key: str,
    run: Path,
    predictions: Path,
    identity: dict,
    horizons: list[int],
    root: Path,
) -> bool:
    """Publish missing analysis metrics from a completed forecast bundle."""
    from .analysis import (
        prepare_metrics,
        read_manifest,
        update_member,
        validate_member,
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
                Y=payload["true_Y"],
                Z=payload["true_Z"],
                t=payload["t"],
                indices=payload["source_indices"],
            )
            selected_ids = identity["selected_ids"]
            metadata = dict(
                session=member["session"],
                fold=fold,
                fit_id=run.name,
                configuration=case["name"],
                case_name=case["name"],
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
                payload,
                truth,
                horizons,
                common,
                metadata,
                destination,
                dict(Y=payload["baseline_Y"], Z=payload["baseline_Z"]),
            )
            metrics = destination / "metrics.json"
            update_member(
                analysis_root,
                key,
                dict(
                    state="complete", error=None, error_type=None,
                    failure_phase=None,
                    metrics=str(metrics.relative_to(analysis_root)),
                    metrics_sha256=file_digest(metrics),
                    predictions=str(predictions.relative_to(root.resolve())),
                    predictions_sha256=file_digest(predictions),
                ),
            )
        except BaseException as error:
            update_member(
                analysis_root,
                key,
                dict(state="failed", error=f"{type(error).__name__}: {error}"),
            )
            raise
    return True


def run_session(
    session: str, dataset: Dataset, folds: list[int], cases: list[dict],
    settings: dict, arguments: argparse.Namespace, shared: list[str],
    snapshots: dict, gpu: dict | None, root: Path,
    analysis_root: Path | None, stop: object = None,
) -> dict:
    """Execute one session, returning attempts and contained error messages.

    The caller exclusively owns this session's log and supplies an optional
    shared cancellation event. Arrays remain local to the session worker.
    """
    from .diagnostics import render_safely
    from .presentation import presentation

    data, plotting = snapshots["data"], snapshots["plotting"]
    rendering_errors, model_errors, attempted = [], [], set()
    session_context = (
        stage_scope(
            arguments.log_directory,
            "preprocess",
            session=session,
        )
        if arguments.stage == "preprocess"
        else lifecycle_scope(arguments.log_directory, session)
    )
    with session_context as session_state:
        with session_logging(arguments.log_directory, session):
            source = dataset.load(session)
            for fold in folds:
                fold_context = (
                    stage_scope(
                        arguments.log_directory,
                        "preprocess",
                        session=session,
                        fold=fold,
                    )
                    if arguments.stage == "preprocess"
                    else lifecycle_scope(
                        arguments.log_directory,
                        session,
                        fold,
                    )
                )
                with fold_context as fold_state:
                    if arguments.stage == "preprocess":
                        features = dataset.fold(
                            source,
                            fold,
                            data["infer_velocity"],
                        )
                        for case in cases:
                            identity, columns = case_identity(
                                features, fold, case, snapshots, shared
                            )
                            run = fit_directory(root, identity)
                            prepare_model_settings(run.parents[2], identity)
                            previews = dict(
                                data["previews"],
                                presentation=presentation(
                                    plotting.get("presentation")
                                ),
                                context_samples=identity["resolved_fit"][
                                    "args_base"
                                ]["sequence_length"],
                            )
                            previous_errors = len(rendering_errors)
                            render_safely(
                                rendering_errors,
                                str(run.resolve()),
                                preprocessing_previews,
                                source,
                                features,
                                previews,
                                run,
                                columns,
                            )
                            outcome = (
                                "failed"
                                if len(rendering_errors) > previous_errors
                                else "completed"
                            )
                            fold_state[outcome] += 1
                            session_state[outcome] += 1
                        LOGGER.info(
                            "Cached fold %s at %s", fold, features.path
                        )
                        continue
                    from .analysis import (
                        member_key, read_manifest, update_member,
                    )

                    for case in cases:
                        if stop is not None and stop.is_set():
                            raise KeyboardInterrupt
                        key = member_key(case, session, fold)
                        with lifecycle_scope(
                            arguments.log_directory, session, fold,
                            case["name"],
                        ) as model_state:
                            try:
                                changed = run_case(
                                    dataset, source, fold, case, settings,
                                    arguments, shared, snapshots, gpu,
                                    root, analysis_root, rendering_errors,
                                    model_state,
                                )
                            except Exception as error:
                                phase = model_state.get("phase", "setup")
                                message = (
                                    f"{key} [{phase}]: "
                                    f"{type(error).__name__}: {error}"
                                )
                                model_errors.append(message)
                                model_state["failed"] = 1
                                LOGGER.exception(
                                    "Model failed: %s", message
                                )
                                member = read_manifest(analysis_root)[
                                    "members"
                                ][key]
                                if member["state"] != "complete":
                                    update_member(
                                        analysis_root, key,
                                        dict(
                                            state="failed",
                                            failure_phase=phase,
                                            error_type=type(error).__name__,
                                            error=str(error),
                                        ),
                                    )
                            else:
                                model_state["skipped"] = not changed
                                outcome = (
                                    "completed" if changed else "reused"
                                )
                                model_state[outcome] = 1
                            for outcome in (
                                "completed", "reused", "failed"
                            ):
                                count = model_state.get(outcome, 0)
                                fold_state[outcome] = (
                                    fold_state.get(outcome, 0) + count
                                )
                                session_state[outcome] = (
                                    session_state.get(outcome, 0) + count
                                )
                        attempted.add(key)
    return dict(
        attempted=attempted, model_errors=model_errors,
        rendering_errors=rendering_errors,
    )


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

        with stage_scope(arguments.log_directory, "preview") as state:
            regenerate_previews(snapshots)
            state["completed"] = 1
        return
    experiment = snapshots["experiment"]
    data = snapshots["data"]
    evaluation = snapshots["evaluation"]
    plotting = snapshots["plotting"]
    from .implementation import implementation_signatures, scientific_versions

    snapshots.update(implementation_signatures())
    snapshots["versions"] = scientific_versions()
    root = Path(snapshots["paths"]["artifact_root"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if arguments.stage == "plot":
        from .analysis import find_analysis

        with stage_scope(
            arguments.log_directory,
            "plot",
            analysis_id=arguments.analysis_id,
        ) as state:
            analysis_root = find_analysis(
                root, snapshots, arguments.analysis_id
            )
            plugin(
                experiment["report_plugin"],
                root=analysis_root,
                settings=plotting,
                sample_rate=data["sampling_rate_hz"],
                regenerate=(
                    runtime.get("figure_regeneration", "incomplete") == "all"
                ),
            )
            state["completed"] = 1
        return
    if (arguments.stage == "fit"
            and runtime.get("parallel_workers", 1) > 1):
        from .parallel import run_parallel

        run_parallel(snapshots, arguments, root)
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
    cases = plugin(experiment["suite_plugin"], settings=experiment["suite"])
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
    from .diagnostics import render_safely

    rendering_errors = []
    model_errors = []
    attempted = set()
    rendered = set()
    if arguments.stage != "preprocess":
        analysis_root = plan_analysis(
            dataset, selected, folds, cases, snapshots, shared, root
        )
        LOGGER.info("Analysis artifacts: %s", analysis_root)
        from .model_summary import write_model_summary

        write_model_summary(analysis_root)

    def report(result: dict) -> None:
        """Collect session outcomes and render newly eligible comparisons."""
        attempted.update(result["attempted"])
        model_errors.extend(result["model_errors"])
        rendering_errors.extend(result["rendering_errors"])
        if arguments.stage != "preprocess":
            render_safely(
                rendering_errors, "Analysis report", plugin,
                experiment["report_plugin"], root=analysis_root,
                settings=plotting, sample_rate=data["sampling_rate_hz"],
                attempted=attempted, rendered=rendered,
                regenerate=runtime.get("figure_regeneration") == "all",
            )

    for session in selected:
        report(run_session(
            session, dataset, folds, cases, settings, arguments, shared,
            snapshots, gpu, root, analysis_root,
        ))
    if model_errors or rendering_errors:
        details = model_errors + rendering_errors
        raise RuntimeError(
            "Scientific results remain saved; experiment failures:\n"
            + "\n".join(details)
        )


if __name__ == "__main__":
    main()
