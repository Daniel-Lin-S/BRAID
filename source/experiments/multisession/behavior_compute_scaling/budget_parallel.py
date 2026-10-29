"""Run independently checkpointed source compute-budget tasks.

Input is the existing immutable direct-behavior source plan. Prefix tasks
write checksum-valid RNN1 main checkpoints. Budget tasks write one matched
RNN1/readout/RNN2 path under source_budget_paths/<source-fit>/; final tasks
assemble the ordinary source fit for the existing report. Runtime readiness
markers live under the launch's tasks/ directory and contain no fit metrics.
"""

from pathlib import Path
import json

import numpy as np

from experiments.multisession.behavior_compute_scaling.plan import (
    COMPONENTS, DIRECT_CAMPAIGN, RECURRENT_STAGES,
)
from experiments.multisession.store import (
    completed, dump_pickle, fit_directory, job,
)
from experiments.storage.cache import atomic_json, fingerprint

COMMON_SESSIONS = 4
COMMON_WINDOWS = 87
TASK_PROTOCOL = "direct_behavior_budget_tasks_v1"
NEURAL_TASK_PROTOCOL = "single_neural_budget_tasks_v2"
LOCAL_DEVICE = ("/GPU:0", "/GPU:0")


def validate_plan(plan: dict) -> None:
    """Require the existing four-count direct source-only design."""
    from experiments.multisession.compute_scaling.self_supervised import (
        CAMPAIGN as NEURAL_CAMPAIGN,
    )
    if plan.get("name") == NEURAL_CAMPAIGN:
        if not plan.get("source_only"):
            raise ValueError("Neural budget plan must be source-only.")
        sources = plan["source_references"]
        counts = [plan["jobs"][key]["source_count"] for key in sources]
        if len(sources) != 4 or sorted(counts) != [4, 8, 16, 33]:
            raise ValueError("Expected four neural source counts.")
        if any(
            plan["jobs"][key]["objective"] != "neural"
            for key in sources
        ):
            raise ValueError("Neural budget plan has behavior objective.")
        return
    if plan.get("name") != DIRECT_CAMPAIGN or not plan.get("source_only"):
        raise ValueError("Budget tasks require the direct source-only plan.")
    sources = plan["source_references"]
    counts = [plan["jobs"][key]["source_count"] for key in sources]
    if len(sources) != 4 or sorted(counts) != [4, 8, 16, 33]:
        raise ValueError("Expected four direct source counts.")
    if any(
        plan["jobs"][key]["behavior_supervision"]["target"]
        != "recorded" for key in sources
    ):
        raise ValueError("Budget tasks require recorded behavior targets.")


def source_arrays(plan: dict, key: str) -> tuple:
    """Rebuild the exact train, inner, and outer source windows."""
    from experiments.multisession.data import (
        session_windows, source_features,
    )
    from experiments.multisession.runner import dataset

    spec = plan["jobs"][key]
    adapter = dataset(plan["settings"])
    train, inner, outer, records = [], [], [], {}
    for name in spec["sessions"]:
        features = source_features(
            adapter.load(name), spec["data"],
            spec["source_training"]["inner_validation_fraction"],
        )
        training, validation, test, record = session_windows(
            features, True, include_test=True,
        )
        train.append(training)
        inner.append(validation)
        outer.append(test)
        records[name] = record
    if sum(
        len(records[name]["validation"])
        for name in spec["sessions"][:COMMON_SESSIONS]
    ) != COMMON_WINDOWS:
        raise ValueError("Expected 87 fixed outer source windows.")
    existing = fit_directory(plan, key) / "data.json"
    if existing.is_file() and json.loads(existing.read_text()) != records:
        raise ValueError("Source windows differ from interrupted fit.")
    return train, inner, outer, records


def resolved_spec(spec: dict, records: dict) -> dict:
    """Bind deterministic split identity to count-specific updates."""
    from experiments.multisession.compute_scaling.branching import (
        resolved_schedule,
    )

    return dict(
        spec,
        source_training=resolved_schedule(
            spec["source_training"], spec["source_count"]
        ),
        data_identity=fingerprint(records),
    )


def progress_path(launch: Path, key: str, budget: int) -> Path:
    """Locate a runtime marker for a completed main prefix."""
    return launch / "tasks" / "prefix" / key / f"{budget}.json"


def prefix_component(
    plan: dict, launch: Path, key: str, budget: int,
) -> Path | None:
    """Resolve a checksum-valid RNN1 main checkpoint marker."""
    marker = progress_path(launch, key, budget)
    if not marker.is_file():
        return None
    record = json.loads(marker.read_text())
    component = (
        Path(plan["settings"]["paths"]["artifact_root"])
        / "multi_session_training" / "source_components"
        / record["component_id"]
    )
    completion = completed(component)
    if completion is None or fingerprint(completion["checksums"]) != (
        record["checksum"]
    ):
        raise ValueError(f"Invalid RNN1 prefix marker: {marker.resolve()}.")
    return component


def fit_prefixes(plan: dict, launch: Path, key: str) -> None:
    """Extend the RNN1 main run, releasing each saved checkpoint."""
    if plan["name"] != DIRECT_CAMPAIGN:
        from experiments.multisession.compute_scaling.self_supervised import (
            fit_prefixes as neural_prefixes,
        )
        neural_prefixes(plan, launch, key)
        return
    from experiments.multisession.behavior_compute_scaling.pipeline import (
        stage_identity,
    )
    from experiments.multisession.behavior_supervision.models import (
        rnn1_model,
    )
    from experiments.multisession.compute_scaling.training import fit_segment
    from experiments.multisession.training import recurrent_data

    train, _, _, records = source_arrays(plan, key)
    spec = resolved_spec(plan["jobs"][key], records)
    dimensions = [
        (row["Y"].shape[-1], row["Z"].shape[-1]) for row in train
    ]
    model = rnn1_model(dimensions, spec, LOCAL_DEVICE)
    data = [recurrent_data(row, "Z") for row in train]
    root = (
        Path(plan["settings"]["paths"]["artifact_root"])
        / "multi_session_training" / "source_components"
    )
    identity = stage_identity(spec, "behavior_relevant", None)
    seed = spec["seed"] + COMPONENTS.index("behavior_relevant")
    previous = None
    start = 0
    microbatch = spec["source_training"]["session_batch_sizes"][
        str(spec["source_count"])
    ]
    for stop in spec["source_training"]["checkpoints"]:
        directory, _ = fit_segment(
            model, data, root, identity, spec["source_training"],
            seed, start, stop, previous, LOCAL_DEVICE, microbatch,
        )
        marker = completed(directory)
        if marker is None:
            raise ValueError(f"Incomplete RNN1 prefix: {directory.resolve()}.")
        destination = progress_path(launch, key, stop)
        destination.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(destination, dict(
            component_id=directory.name,
            checksum=fingerprint(marker["checksums"]),
        ))
        previous, start = directory, stop


def budget_identity(plan: dict, key: str, budget: int) -> dict:
    """Identify a matched budget independently of its device assignment."""
    spec = plan["jobs"][key]
    if budget not in (
        value * spec["source_training"]["update_multipliers"][
            str(spec["source_count"])
        ] for value in spec["source_training"]["checkpoints"]
    ):
        raise ValueError(f"Unknown source budget {budget}.")
    return dict(
        protocol=(
            TASK_PROTOCOL if plan["name"] == DIRECT_CAMPAIGN
            else NEURAL_TASK_PROTOCOL
        ), source_fit=key,
        implementation=spec["implementation"], budget=budget,
    )


def budget_directory(plan: dict, key: str, budget: int) -> Path:
    """Locate one immutable matched-path completion."""
    identity = budget_identity(plan, key, budget)
    reused = plan.get("continuation", {}).get(key, {}).get(
        "budgets", {}
    ).get(str(budget))
    if reused is not None:
        return Path(reused)
    root = Path(plan["settings"]["paths"]["artifact_root"])
    return (
        root / "multi_session_training" / "source_budget_paths" / key
        / f"budget{budget}-{fingerprint(identity)}"
    )


def fit_budget(
    plan: dict, launch: Path, key: str, budget: int,
) -> None:
    """Fit or reuse one matched RNN1/readout/RNN2 budget path."""
    if plan["name"] != DIRECT_CAMPAIGN:
        from experiments.multisession.compute_scaling.self_supervised import (
            fit_budget as neural_budget,
        )
        neural_budget(plan, launch, key, budget)
        return
    from experiments.multisession.behavior_compute_scaling.pipeline import (
        run_stage,
    )
    from experiments.multisession.behavior_supervision.models import (
        neural_readout_model, rnn1_model, rnn2_model,
    )
    from experiments.multisession.source import readout_data, residual_data
    from experiments.multisession.training import (
        recurrent_data, stage_predictions,
    )

    current = prefix_component(plan, launch, key, budget)
    if current is None:
        raise ValueError(f"RNN1 prefix {budget} is not ready.")
    destination = budget_directory(plan, key, budget)
    with job(destination, budget_identity(plan, key, budget)) as needed:
        if not needed:
            return
        train, inner, outer, records = source_arrays(plan, key)
        spec = resolved_spec(plan["jobs"][key], records)
        settings = spec["source_training"]
        index = settings["checkpoints"].index(budget)
        previous = (
            prefix_component(plan, launch, key, settings["checkpoints"][
                index - 1
            ]) if index else None
        )
        if index and previous is None:
            raise ValueError("Previous RNN1 prefix is missing.")
        dimensions = [
            (row["Y"].shape[-1], row["Z"].shape[-1]) for row in train
        ]
        partitions = (train, inner, outer)
        neural = [
            [recurrent_data(row) for row in part] for part in partitions
        ]
        behavior = [
            [recurrent_data(row, "Z") for row in part]
            for part in partitions
        ]
        root = destination.parent.parent.parent / "source_components"
        names = spec["sessions"][:COMMON_SESSIONS]
        microbatch = settings["session_batch_sizes"][str(len(train))]
        main = rnn1_model(dimensions, spec, LOCAL_DEVICE)
        main_record, parent = run_stage(
            "behavior_relevant", main, behavior[0], behavior[2],
            names, spec, root, None, budget,
            LOCAL_DEVICE, microbatch, resume=previous,
        )
        if completed(current) is None:
            raise ValueError("RNN1 main prefix changed during budget fit.")
        states = [stage_predictions(main, part)[1] for part in behavior]
        readout = neural_readout_model(dimensions, spec, LOCAL_DEVICE)
        readout_inputs = [
            readout_data(part, latent)
            for part, latent in zip(neural, states)
        ]
        readout_record, parent = run_stage(
            "neural_readout", readout, readout_inputs[0],
            readout_inputs[2], names, spec, root, parent, 0,
            LOCAL_DEVICE, microbatch, validation=readout_inputs[1],
        )
        residual = rnn2_model(dimensions, spec, LOCAL_DEVICE)
        residual_record, _ = run_stage(
            "residual", residual,
            residual_data(main, readout, neural[0]),
            residual_data(main, readout, neural[2]),
            names, spec, root, parent, budget,
            LOCAL_DEVICE, microbatch,
        )
        stages = dict(
            behavior_relevant=main_record,
            neural_readout=readout_record,
            residual=residual_record,
        )
        total = sum(row["component_flops"] for row in stages.values())
        score = sum(
            stages[stage]["score"]["mean_ratio"]
            for stage in RECURRENT_STAGES
        )
        if total <= 0 or not np.isfinite(score):
            raise ValueError("Invalid budget FLOPs or selection score.")
        path = dict(
            budget=budget,
            total_updates=settings["total_updates"][index],
            branch_updates=settings["total_updates"][index] - budget,
            stages=stages, full_path_flops=total,
            selection_score=score, preprocessing_reference=None,
        )
        atomic_json(destination / "path.json", path)


def finish_source(plan: dict, key: str) -> None:
    """Assemble seven completed paths into the ordinary source fit."""
    if plan["name"] != DIRECT_CAMPAIGN:
        from experiments.multisession.compute_scaling.self_supervised import (
            finish_source as finish_neural,
        )
        finish_neural(plan, key)
        return
    from experiments.multisession.behavior_supervision.models import (
        rnn1_model, rnn2_model,
    )
    from experiments.multisession.compute_scaling.ancestry import (
        restore_component,
    )

    spec = plan["jobs"][key]
    multiplier = spec["source_training"]["update_multipliers"][
        str(spec["source_count"])
    ]
    budgets = [
        value * multiplier for value in spec["source_training"]["checkpoints"]
    ]
    paths = []
    for budget in budgets:
        directory = budget_directory(plan, key, budget)
        if completed(directory) is None:
            raise ValueError(f"Incomplete matched budget {budget}.")
        paths.append(json.loads((directory / "path.json").read_text()))
    train, _, _, records = source_arrays(plan, key)
    names = spec["sessions"][:COMMON_SESSIONS]
    fixed_set = fingerprint({
        name: records[name]["validation"] for name in names
    })
    selected = min(
        paths,
        key=lambda row: (row["selection_score"], row["full_path_flops"]),
    )
    dimensions = [
        (row["Y"].shape[-1], row["Z"].shape[-1]) for row in train
    ]
    root = (
        Path(plan["settings"]["paths"]["artifact_root"])
        / "multi_session_training" / "source_components"
    )
    main = rnn1_model(dimensions, spec, LOCAL_DEVICE)
    residual = rnn2_model(dimensions, spec, LOCAL_DEVICE)
    for model, stage in (
        (main, "behavior_relevant"), (residual, "residual")
    ):
        restore_component(
            model, root / selected["stages"][stage]["component_id"]
        )
    identity = dict(spec, parent_checksums={})
    directory = fit_directory(plan, key)
    with job(directory, identity) as needed:
        if not needed:
            return
        atomic_json(directory / "data.json", records)
        atomic_json(directory / "budgets.json", paths)
        atomic_json(directory / "selection.json", dict(
            selected_budget=selected["budget"],
            selection_score=selected["selection_score"],
            selected_full_path_flops=selected["full_path_flops"],
            evaluation_sessions=names,
            evaluation_window_counts={
                name: len(records[name]["validation"]) for name in names
            },
            evaluation_set_id=fixed_set,
            holdout_role=(
                "chronological_source_test_and_checkpoint_selection"
            ),
        ))
        dump_pickle(directory / "shared.pkl", dict(
            stage1=main.transferable(),
            stage2=residual.transferable(),
        ))
