"""Fit matched behavioral source paths and score the fixed holdout.

Inputs are normalized train, inner-validation, and chronological-test windows
for nested Indy sessions. Generated-target preprocessing is shared across
budgets; each RNN1 branch has a matched readout and RNN2 branch. Immutable
segments live in source_components. A source fit stores split/scaling records,
component paths, losses, exposure, FLOPs, selection, and transferable weights.
Generated mode also stores preprocessing and target-scaling records.
"""

from pathlib import Path
import json

import numpy as np

from BRAID.multisession import HORIZONS
from experiments.multisession.behavior_compute_scaling.plan import (
    COMPONENTS, RECURRENT_STAGES,
)
from experiments.multisession.behavior_supervision.models import (
    STAGE1_SHARING, STAGE2_SHARING,
    neural_readout_model, preprocessing_model,
    preprocessing_readout_model, rnn1_model, rnn2_model,
)
from experiments.multisession.behavior_supervision.targets import (
    RECORDED, prepare_targets, target_mode,
)
from experiments.multisession.compute_scaling.training import (
    evaluate_sessions, fit_segment,
)
from experiments.multisession.compute_scaling.branching import (
    component_exposure, fit_component_budget, load_flops,
    resolved_schedule,
)
from experiments.multisession.source import (
    readout_data, residual_data,
)
from experiments.multisession.store import dump_pickle
from experiments.multisession.behavior_compute_scaling.ancestry import (
    inherited_paths, main_prefix, restore_component,
    validate_continuation,
)
from experiments.multisession.training import (
    recurrent_data, stage_predictions,
)
from experiments.storage.cache import atomic_json, fingerprint


def score_component(
    model: object, training: list[dict], holdout: list[dict],
    names: list[str],
) -> dict:
    """Evaluate native stage loss and training-mean dummy on common four."""
    if len(names) != 4:
        raise ValueError("Expected exactly four common source sessions.")
    losses, evaluation_flops = evaluate_sessions(
        model, holdout[:len(names)]
    )
    rows = []
    for index, name in enumerate(names):
        target = training[index]["T"]
        truth = holdout[index]["T"]
        if not len(target) or not len(truth):
            raise ValueError(f"Empty source train or holdout: {name}.")
        mean = target.mean(axis=(0, 1))
        loss = losses[index]
        dummy = 0.0
        for horizon in HORIZONS:
            error = mean - truth[:, horizon:]
            dummy += float(np.mean(np.square(error), axis=(1, 2)).mean())
        if not np.isfinite([loss, dummy]).all():
            raise FloatingPointError(f"Nonfinite source holdout: {name}.")
        if loss <= 0 or dummy <= 0:
            raise ValueError(
                f"Nonpositive source holdout or dummy loss: {name}."
            )
        ratio = loss / dummy
        if not np.isfinite(ratio) or ratio <= 0:
            raise ValueError(f"Invalid source dummy ratio: {name}.")
        rows.append(dict(
            session=name, windows=len(truth), loss=loss,
            dummy_loss=dummy, ratio=ratio,
        ))
    mean_loss = float(np.mean([row["loss"] for row in rows]))
    mean_ratio = float(np.mean([row["ratio"] for row in rows]))
    if not np.isfinite([mean_loss, mean_ratio]).all():
        raise FloatingPointError("Nonfinite average source test loss.")
    return dict(
        per_session=rows, mean_loss=mean_loss,
        mean_ratio=mean_ratio, evaluation_flops=evaluation_flops,
    )


def stage_identity(spec: dict, stage: str, parent: str | None) -> dict:
    """Bind source component identity to scientific inputs and ancestry."""
    return dict(
        source_count=spec["source_count"],
        sessions=spec["sessions"],
        sources=spec["sources"],
        data=spec["data"],
        model=spec["model"],
        latent=spec["latent"],
        seed=spec["seed"],
        sharing=(
            STAGE1_SHARING if stage == "behavior_relevant"
            else STAGE2_SHARING if stage == "residual" else "none"
        ),
        source_training=spec["source_training"],
        implementation=spec["implementation"],
        stage=stage,
        upstream_checkpoint=parent,
        data_identity=spec["data_identity"],
    )


def run_stage(
    stage: str, model: object, training: list[dict],
    holdout: list[dict], names: list[str], spec: dict,
    root: Path, parent: str | None, budget: int,
    devices: tuple[str, str], microbatch: int,
    validation: list[dict] | None = None,
    reuse: dict | None = None, resume: Path | None = None,
) -> tuple[dict, str]:
    """Fit one matched stage branch and record fixed-set stage loss."""
    if reuse is not None:
        restore_component(model, root / reuse["component_id"])
        return reuse, reuse["component_id"]
    stage_index = COMPONENTS.index(stage)
    identity = stage_identity(spec, stage, parent)
    settings = spec["source_training"]
    if stage in RECURRENT_STAGES:
        directory, flops = fit_component_budget(
            model, training, root, identity, settings,
            spec["seed"] + stage_index, budget, devices, microbatch,
            resume=resume,
        )
    else:
        if validation is None:
            raise ValueError(f"Fixed stage {stage} requires inner validation.")
        directory, _ = fit_segment(
            model, training, root, identity, settings,
            spec["seed"] + stage_index, 0, settings["fixed_update_cap"],
            None, devices, microbatch, validation=validation,
        )
        flops = load_flops(directory)
    score = score_component(model, training, holdout, names)
    record = dict(
        stage=stage, component_id=directory.name,
        component_flops=flops, score=score,
        exposure=component_exposure(directory, training),
    )
    return record, directory.name


def fit_source_campaign(
    train: list[dict], holdout: list[dict], records: dict,
    directory: Path, component_root: Path, spec: dict,
    devices: tuple[str, str], microbatch: int,
    validation: list[dict],
) -> None:
    """Fit matched recurrent budgets using the declared behavior target.

    Parameters
    ----------
    train, holdout, validation : list of dict
        Per-session Y/U/Z arrays shaped (windows, 128, channels).
    records : dict
        Original window indices and train-only normalization per session.
    directory, component_root : Path
        Source fit and reusable immutable component store.
    spec : dict
        Scientific source definition, including schedules and sharing.
    devices : tuple of str
        Two devices; sessions remain assigned by index modulo two.
    microbatch : int
        Maximum session-local windows before adaptive accumulation.
    """
    spec = dict(
        spec, source_training=resolved_schedule(
            spec["source_training"], spec["source_count"]
        ), data_identity=fingerprint(records),
    )
    inherited = []
    reused = {}
    resume = None
    if "continuation" in spec:
        ancestor = validate_continuation(spec, records, directory.parent)
        inherited = inherited_paths(ancestor)
        reused = inherited[-1]["stages"]
        resume = main_prefix(
            component_root / reused["behavior_relevant"]["component_id"]
        )
        atomic_json(directory / "continuation.json", spec["continuation"])
    names = spec["sessions"][:4]
    if len(names) != 4 or len(set(names)) != 4:
        raise ValueError("Expected four distinct common source sessions.")
    split_identity = fingerprint({
        name: records[name]["validation"] for name in names
    })
    if sum(len(records[name]["validation"]) for name in names) != 87:
        raise ValueError("Expected exactly 87 common outer test windows.")
    component_dimensions = [
        (row["Y"].shape[-1], row["Z"].shape[-1]) for row in train
    ]
    partitions = (train, validation, holdout)
    neural = [[recurrent_data(row) for row in part] for part in partitions]
    mode = target_mode(spec)
    direct_target = mode == RECORDED
    if direct_target:
        prepared, _ = prepare_targets(partitions, mode)
        base_stages = {}
        pre_parent = None
        target_reference = None
    else:
        pre = preprocessing_model(component_dimensions, spec, devices)
        pre_record, parent = run_stage(
            "preprocessing_neural", pre, neural[0], neural[2], names,
            spec, component_root, None, 0, devices, microbatch,
            validation=neural[1], reuse=reused.get("preprocessing_neural"),
        )
        pre_states = [stage_predictions(pre, part)[1] for part in neural]
        pre_readout = preprocessing_readout_model(
            component_dimensions, spec, devices
        )
        behavior = [
            readout_data([recurrent_data(row, "Z") for row in part], states)
            for part, states in zip(partitions, pre_states)
        ]
        behavior_record, pre_parent = run_stage(
            "preprocessing_behavior", pre_readout, behavior[0], behavior[2],
            names, spec, component_root, parent, 0, devices, microbatch,
            validation=behavior[1], reuse=reused.get("preprocessing_behavior"),
        )
        prepared, scales = prepare_targets(
            partitions, mode, pre_states, pre_readout
        )
        atomic_json(directory / "preprocessing.json", dict(
            neural=pre_record["component_id"],
            behavior=behavior_record["component_id"], scales=scales,
            target_identity=fingerprint(dict(
                parent=pre_parent, scales=scales, data=spec["data_identity"],
            )),
        ))
        if inherited:
            saved_preprocessing = json.loads(
                (ancestor / "preprocessing.json").read_text()
            )
            current_preprocessing = json.loads(
                (directory / "preprocessing.json").read_text()
            )
            if saved_preprocessing != current_preprocessing:
                raise ValueError(
                    "Continuation changed fixed generated targets."
                )
        base_stages = dict(
            preprocessing_neural=pre_record,
            preprocessing_behavior=behavior_record,
        )
        target_reference = "preprocessing.json"
    stage_data = [
        [recurrent_data(row, "Z") for row in part] for part in prepared
    ]
    main = rnn1_model(component_dimensions, spec, devices)
    main_initial = main.weights()
    paths = []
    prior_budget = inherited[-1]["budget"] if inherited else 0
    settings = spec["source_training"]
    for budget, total in zip(
        settings["checkpoints"], settings["total_updates"]
    ):
        if budget <= prior_budget:
            continue
        main.restore(main_initial)
        stages = dict(base_stages)
        record, parent = run_stage(
            "behavior_relevant", main, stage_data[0], stage_data[2], names,
            spec, component_root, pre_parent, budget, devices, microbatch,
            resume=resume,
        )
        stages["behavior_relevant"] = record
        main_states = [stage_predictions(main, part)[1] for part in stage_data]
        readout = neural_readout_model(
            component_dimensions, spec, devices
        )
        readout_inputs = [
            readout_data(part, states)
            for part, states in zip(neural, main_states)
        ]
        record, parent = run_stage(
            "neural_readout", readout, readout_inputs[0], readout_inputs[2],
            names, spec, component_root, parent, 0, devices, microbatch,
            validation=readout_inputs[1],
        )
        stages["neural_readout"] = record
        residual = rnn2_model(component_dimensions, spec, devices)
        record, _ = run_stage(
            "residual", residual,
            residual_data(main, readout, neural[0]),
            residual_data(main, readout, neural[2]), names,
            spec, component_root, parent, budget, devices, microbatch,
        )
        stages["residual"] = record
        total_flops = sum(row["component_flops"] for row in stages.values())
        criterion = sum(
            stages[stage]["score"]["mean_ratio"] for stage in RECURRENT_STAGES
        )
        if not np.isfinite(criterion) or total_flops <= 0:
            raise ValueError("Invalid source selection score or FLOPs.")
        paths.append(dict(
            budget=budget, total_updates=total, branch_updates=total - budget,
            stages=stages, full_path_flops=total_flops,
            selection_score=criterion,
            preprocessing_reference=target_reference,
        ))
    selected = min(
        inherited + paths,
        key=lambda row: (row["selection_score"], row["full_path_flops"]),
    )
    for model, stage in ((main, "behavior_relevant"), (residual, "residual")):
        restore_component(
            model, component_root / selected["stages"][stage]["component_id"]
        )
    dump_pickle(directory / "shared.pkl", dict(
        stage1=main.transferable(), stage2=residual.transferable(),
    ))
    atomic_json(directory / "budgets.json", paths)
    atomic_json(directory / "selection.json", dict(
        selected_budget=selected["budget"],
        selection_score=selected["selection_score"],
        selected_full_path_flops=selected["full_path_flops"],
        evaluation_sessions=names,
        evaluation_window_counts={
            name: len(records[name]["validation"]) for name in names
        },
        evaluation_set_id=split_identity,
        holdout_role=(
            "chronological_source_test_and_checkpoint_selection"
            if direct_target else
            "chronological_source_test_and_transfer_selection"
        ),
    ))
