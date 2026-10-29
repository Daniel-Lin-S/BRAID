"""Fit one neural-only multi-session source backbone.

Inputs are a source-only plan, guarded Y/U source windows, and saved prefix
components. Outputs in the shared component store are neural
checkpoints. Source fits store budgets,
inner-validation selection, split inventories, and transferable weights.
No outer-test array enters model fitting or checkpoint selection.
"""

import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from BRAID.multisession import MixedForecaster
from experiments.multisession.behavior_compute_scaling.plan import (
    COUNTS, LATENT, SEED,
)
from experiments.multisession.behavior_compute_scaling.pipeline import (
    run_stage, stage_identity,
)
from experiments.multisession.behavior_supervision.models import (
    STAGE2_SHARING,
)
from experiments.multisession.compute_scaling.branching import (
    resolved_schedule,
)
from experiments.multisession.compute_scaling.training import fit_segment
from experiments.multisession.data import session_windows, source_features
from experiments.multisession.plan import PRIMARY
from experiments.multisession.runner import dataset
from experiments.multisession.store import (
    completed, dump_pickle, fit_directory, job,
)
from experiments.multisession.training import recurrent_data
from experiments.storage.cache import atomic_json, fingerprint
from experiments.multisession.compute_scaling.ancestry import (
    restore_component,
)

CAMPAIGN = "multi_session_self_supervised_scaling"
PROTOCOL = "single_neural_recurrent_compute_v2"
IMPLEMENTATION = fingerprint(dict(
    objective="single_neural_forecaster_shared_core_v2",
    compute="branching_flops_v1",
    selection="four_common_inner_raw_neural_v1",
))
SHARING = STAGE2_SHARING
COMMON = 4
OUTER_WINDOWS = 87
LOCAL_DEVICE = ("/GPU:0", "/GPU:0")


def build_plan(settings: dict, sessions: dict, reference: dict) -> dict:
    """Build four nested neural-only source fits from the direct source sets."""
    if settings["name"] != CAMPAIGN or (
        settings["latent"] != LATENT or settings["seeds"] != [SEED]
    ):
        raise ValueError("Expected self-supervised latent-16 seed-42 design.")
    if settings["source_training"]["protocol"] != PROTOCOL:
        raise ValueError("Expected neural-only compute protocol.")
    schedule = json.loads(json.dumps(settings["source_training"]))
    if schedule != dict(
        reference["settings"]["source_training"], protocol=PROTOCOL
    ):
        raise ValueError("Neural source schedule differs from direct ladder.")
    old = reference["settings"]
    for field in ("data", "model", "held_out_sessions"):
        if settings[field] != old[field]:
            raise ValueError(f"Self-supervised {field} differs from direct.")
    if settings["paths"]["artifact_root"] != (
        old["paths"]["artifact_root"]
    ):
        raise ValueError("Artifact root differs from reference campaign.")
    jobs, sources = {}, []
    for key in reference["source_references"]:
        original = reference["jobs"][key]
        if completed(fit_directory(reference, key)) is None:
            raise ValueError(f"Incomplete direct source reference: {key}.")
        if any(
            sessions.get(name) != original["sources"][name]
            for name in original["sessions"]
        ):
            raise ValueError("Source recording fingerprint changed.")
        spec = deepcopy(original)
        spec.update(
            objective="neural", sharing=SHARING,
            source_training=deepcopy(settings["source_training"]),
            implementation=IMPLEMENTATION, dependencies=[],
        )
        spec.pop("behavior_supervision", None)
        spec.pop("continuation", None)
        identifier = fingerprint(spec)
        if identifier in jobs:
            raise ValueError("Duplicate neural-only source fit identity.")
        jobs[identifier] = spec
        sources.append(identifier)
    counts = sorted(row["source_count"] for row in jobs.values())
    if counts != list(COUNTS):
        raise ValueError(f"Expected source counts {COUNTS}, got {counts}.")
    return dict(
        name=CAMPAIGN, source_only=True,
        settings=deepcopy(settings),
        storage=deepcopy(reference.get("storage", {})),
        jobs=jobs, source_references=sources,
        analyses={CAMPAIGN: sources},
        analysis_paths={CAMPAIGN: CAMPAIGN},
        held_out_sessions=list(settings["held_out_sessions"]),
    )


def neural_model(
    dimensions: list[tuple[int, int]], spec: dict,
    devices: tuple[str, str] | None = None,
) -> MixedForecaster:
    """Build one neural forecaster with a shared recurrent core."""
    return MixedForecaster(
        dimensions, spec["latent"], SHARING,
        spec["seed"], session_devices=devices,
    )


def source_arrays(plan: dict, key: str) -> tuple:
    """Rebuild Y/U only for source fits; record guarded split indices."""
    spec = plan["jobs"][key]
    adapter = dataset(plan["settings"])
    adapter.mode = "off"
    train, inner, records = [], [], {}
    for name in spec["sessions"]:
        features = source_features(
            adapter.load(name), spec["data"],
            spec["source_training"]["inner_validation_fraction"],
        )
        training, validation, _, record = session_windows(
            features, False, include_test=True,
        )
        if "Z" in training or "Z" in record["scales"]:
            raise ValueError("Behavior entered neural-only source windows.")
        train.append(training)
        inner.append(validation)
        records[name] = record
    if sum(
        len(records[name]["validation"])
        for name in spec["sessions"][:COMMON]
    ) != OUTER_WINDOWS:
        raise ValueError("Expected 87 fixed outer source windows.")
    existing = fit_directory(plan, key) / "data.json"
    if existing.is_file() and json.loads(existing.read_text()) != records:
        raise ValueError("Saved neural-only source split changed.")
    return train, inner, records


def resolved_spec(spec: dict, records: dict) -> dict:
    """Bind resolved updates and current split identity to a component."""
    return dict(
        spec,
        source_training=resolved_schedule(
            spec["source_training"], spec["source_count"]
        ),
        data_identity=fingerprint(records),
    )


def prefix_identity(spec: dict) -> dict:
    """Identify the single neural recurrent objective."""
    return stage_identity(spec, "neural", None)


def fit_prefixes(plan: dict, launch: Path, key: str) -> None:
    """Save seven reusable main-run neural checkpoints."""
    from ..behavior_compute_scaling.budget_parallel import (
        prefix_component, progress_path,
    )
    train, _, records = source_arrays(plan, key)
    spec = resolved_spec(plan["jobs"][key], records)
    dimensions = [(row["Y"].shape[-1],) * 2 for row in train]
    model = neural_model(dimensions, spec, LOCAL_DEVICE)
    data = [recurrent_data(row) for row in train]
    root = (
        Path(plan["settings"]["paths"]["artifact_root"])
        / PRIMARY / "source_components"
    )
    settings = spec["source_training"]
    local = settings["session_batch_sizes"][str(spec["source_count"])]
    previous, start = None, 0
    for stop in settings["checkpoints"]:
        component, _ = fit_segment(
            model, data, root, prefix_identity(spec), settings,
            spec["seed"], start, stop, previous,
            LOCAL_DEVICE, local,
        )
        marker = completed(component)
        if marker is None:
            raise ValueError("Incomplete neural prefix.")
        progress = progress_path(launch, key, stop)
        progress.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(progress, dict(
            component_id=component.name,
            checksum=fingerprint(marker["checksums"]),
        ))
        if prefix_component(plan, launch, key, stop) != component:
            raise ValueError("Neural prefix marker differs from component.")
        previous, start = component, stop


def fit_budget(
    plan: dict, launch: Path, key: str, budget: int,
) -> None:
    """Fit one branched neural budget using inner validation."""
    from ..behavior_compute_scaling.budget_parallel import (
        budget_directory, budget_identity, prefix_component,
    )
    current = prefix_component(plan, launch, key, budget)
    if current is None:
        raise ValueError(f"Neural prefix {budget} is not ready.")
    destination = budget_directory(plan, key, budget)
    with job(destination, budget_identity(plan, key, budget)) as needed:
        if not needed:
            return
        train, inner, records = source_arrays(plan, key)
        spec = resolved_spec(plan["jobs"][key], records)
        settings = spec["source_training"]
        index = settings["checkpoints"].index(budget)
        previous = (
            prefix_component(
                plan, launch, key, settings["checkpoints"][index - 1]
            ) if index else None
        )
        if index and previous is None:
            raise ValueError("Previous neural prefix is missing.")
        dimensions = [(row["Y"].shape[-1],) * 2 for row in train]
        neural_train = [recurrent_data(row) for row in train]
        neural_inner = [recurrent_data(row) for row in inner]
        names = spec["sessions"][:COMMON]
        root = destination.parent.parent.parent / "source_components"
        local = settings["session_batch_sizes"][str(len(train))]
        main = neural_model(dimensions, spec, LOCAL_DEVICE)
        main_record, _ = run_stage(
            "neural", main, neural_train, neural_inner, names,
            spec, root, None, budget, LOCAL_DEVICE, local,
            resume=previous,
        )
        if completed(current) is None:
            raise ValueError("Neural prefix changed during budget fit.")
        score = main_record["score"]["mean_loss"]
        total = main_record["component_flops"]
        if not np.isfinite(score) or score <= 0 or total <= 0:
            raise ValueError("Invalid inner neural loss or backbone FLOPs.")
        atomic_json(destination / "path.json", dict(
            budget=budget,
            total_updates=settings["total_updates"][index],
            branch_updates=settings["total_updates"][index] - budget,
            stages=dict(neural=main_record),
            full_path_flops=total, selection_score=score,
            selection_role="four_common_inner_validation",
        ))


def finish_source(plan: dict, key: str) -> None:
    """Freeze the best inner-validation backbone without test scoring."""
    from ..behavior_compute_scaling.budget_parallel import (
        budget_directory,
    )
    spec = plan["jobs"][key]
    multiplier = spec["source_training"]["update_multipliers"][
        str(spec["source_count"])
    ]
    budgets = [
        x * multiplier for x in spec["source_training"]["checkpoints"]
    ]
    paths = []
    for budget in budgets:
        directory = budget_directory(plan, key, budget)
        if completed(directory) is None:
            raise ValueError(f"Incomplete neural budget {budget}.")
        paths.append(json.loads((directory / "path.json").read_text()))
    train, inner, records = source_arrays(plan, key)
    chosen = min(
        paths,
        key=lambda row: (
            row["selection_score"], row["full_path_flops"]
        ),
    )
    names = spec["sessions"][:COMMON]
    inner_id = fingerprint({
        name: records[name]["inner_validation"] for name in names
    })
    outer_id = fingerprint({
        name: records[name]["validation"] for name in names
    })
    dimensions = [(row["Y"].shape[-1],) * 2 for row in train]
    root = (
        Path(plan["settings"]["paths"]["artifact_root"])
        / PRIMARY / "source_components"
    )
    main = neural_model(dimensions, spec, LOCAL_DEVICE)
    restore_component(
        main, root / chosen["stages"]["neural"]["component_id"]
    )
    directory = fit_directory(plan, key)
    with job(directory, dict(spec, parent_checksums={})) as needed:
        if not needed:
            return
        atomic_json(directory / "data.json", records)
        atomic_json(directory / "budgets.json", paths)
        atomic_json(directory / "selection.json", dict(
            selected_budget=chosen["budget"],
            selection_score=chosen["selection_score"],
            selected_full_path_flops=chosen["full_path_flops"],
            selection_set_id=inner_id,
            selection_role="four_common_inner_validation",
            evaluation_sessions=names,
            evaluation_set_id=outer_id,
            evaluation_window_counts={
                name: len(records[name]["validation"]) for name in names
            },
        ))
        dump_pickle(directory / "shared.pkl", main.transferable())


