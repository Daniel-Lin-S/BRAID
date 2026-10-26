"""Build matched behavioral source paths and score the fixed source holdout.

Inputs are normalized train/chronological-holdout arrays for nested Indy
sessions. Immutable component segments live in source_components; the source
fit stores data.json, budgets.json, selection.json and shared.pkl. Every
budget path has five components whose data depend on that budget's upstream
branch. The four common recordings' held-out windows select transfer weights.
"""

from pathlib import Path

import numpy as np

from BRAID.multisession import HORIZONS, MixedForecaster
from experiments.multisession.behavior_compute_scaling.plan import (
    CHECKPOINTS, COMPONENTS, STAGE1_SHARING, STAGE2_SHARING,
)
from experiments.multisession.behavior_compute_scaling.training import (
    fit_segment, selected_batch, session_loss,
)
from experiments.multisession.source import (
    BEHAVIOR_WIDTH, NEURAL_WIDTH, PREPROCESSING_SIZE,
    behavior_targets, readout_data, residual_data,
)
from experiments.multisession.store import dump_pickle
from experiments.multisession.training import (
    ReadoutStage, recurrent_data, stage_predictions,
)
from experiments.multisession.data import Scale
from experiments.storage.cache import atomic_json, fingerprint

EVALUATION_BATCH = 16


def fit_component_budget(
    model: object, train: list[dict], root: Path, identity: dict,
    settings: dict, seed: int, budget: int,
    devices: tuple[str, str], microbatch: int,
) -> tuple[Path, int]:
    """Restore or train main prefix and one budget-matched decay branch."""
    parent = None
    total = 0
    for stop in CHECKPOINTS:
        if stop > budget:
            break
        directory, _ = fit_segment(
            model, train, root, identity, settings, seed,
            stop - CHECKPOINTS[0], stop, parent,
            devices, microbatch,
        )
        parent = directory
        total += load_flops(directory)
    branch, _ = fit_segment(
        model, train, root, identity, settings, seed,
        budget, budget + settings["branch_updates"], parent,
        devices, microbatch, branch=True,
    )
    return branch, total + load_flops(branch)


def load_flops(directory: Path) -> int:
    """Read validated component arithmetic from a completed segment."""
    import json

    value = json.loads(
        (directory / "training.json").read_text()
    )["total_flops"]
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"Invalid component FLOPs: {directory.resolve()}.")
    return value


def score_component(
    model: object, training: list[dict], holdout: list[dict],
    names: list[str],
) -> dict:
    """Evaluate native stage loss and training-mean dummy on common four."""
    if len(names) != 4:
        raise ValueError("Expected exactly four common source sessions.")
    rows = []
    for index, name in enumerate(names):
        target = training[index]["T"]
        truth = holdout[index]["T"]
        if not len(target) or not len(truth):
            raise ValueError(f"Empty source train or holdout: {name}.")
        mean = target.mean(axis=(0, 1))
        loss_sum = 0.0
        for start in range(0, len(truth), EVALUATION_BATCH):
            chosen = np.arange(
                start, min(start + EVALUATION_BATCH, len(truth))
            )
            batch = selected_batch(holdout[index], chosen)
            loss_sum += float(session_loss(model, index, batch).numpy())
        loss = loss_sum / len(truth)
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
        mean_ratio=mean_ratio,
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
    )


def run_stage(
    stage: str, model: object, training: list[dict],
    holdout: list[dict], names: list[str], spec: dict,
    root: Path, parent: str | None, budget: int,
    devices: tuple[str, str], microbatch: int,
) -> tuple[dict, str]:
    """Fit one matched stage branch and record fixed-set stage loss."""
    stage_index = COMPONENTS.index(stage)
    identity = stage_identity(spec, stage, parent)
    directory, flops = fit_component_budget(
        model, training, root, identity, spec["source_training"],
        spec["seed"] + stage_index, budget, devices, microbatch,
    )
    score = score_component(model, training, holdout, names)
    record = dict(
        stage=stage, component_id=directory.name,
        component_flops=flops, score=score,
    )
    return record, directory.name


def fit_source_campaign(
    train: list[dict], holdout: list[dict], records: dict,
    directory: Path, component_root: Path, spec: dict,
    devices: tuple[str, str], microbatch: int,
) -> None:
    """Fit five dependency-consistent paths and publish selected maps."""
    names = spec["sessions"][:4]
    if len(names) != 4 or len(set(names)) != 4:
        raise ValueError("Expected four distinct common source sessions.")
    split_identity = fingerprint({
        name: records[name]["validation"] for name in names
    })
    if any(
        not records[name]["validation"] for name in names
    ):
        raise ValueError("Common source holdout has empty windows.")
    dimensions = [
        (row["Y"].shape[-1], row["Y"].shape[-1]) for row in train
    ]
    neural_train = [recurrent_data(row) for row in train]
    neural_test = [recurrent_data(row) for row in holdout]
    pre = MixedForecaster(
        dimensions, PREPROCESSING_SIZE, "none", spec["seed"]
    )
    paths = []
    transferable = {}
    for budget in CHECKPOINTS:
        stages = {}
        record, parent = run_stage(
            "preprocessing_neural", pre, neural_train,
            neural_test, names, spec, component_root,
            None, budget, devices, microbatch,
        )
        stages["preprocessing_neural"] = record
        _, pre_train = stage_predictions(pre, neural_train)
        _, pre_test = stage_predictions(pre, neural_test)
        pre_readout = ReadoutStage(
            [
                (PREPROCESSING_SIZE, row["Z"].shape[-1])
                for row in train
            ],
            BEHAVIOR_WIDTH, spec["seed"] + 1, behavior=True,
        )
        ztrain = [recurrent_data(row, "Z") for row in train]
        ztest = [recurrent_data(row, "Z") for row in holdout]
        record, parent = run_stage(
            "preprocessing_behavior", pre_readout,
            readout_data(ztrain, pre_train),
            readout_data(ztest, pre_test),
            names, spec, component_root, parent,
            budget, devices, microbatch,
        )
        stages["preprocessing_behavior"] = record
        prepared_train = behavior_targets(
            pre_readout, pre_train, train
        )
        prepared_test = behavior_targets(
            pre_readout, pre_test, holdout
        )
        scales = []
        for source, test in zip(prepared_train, prepared_test):
            scale = Scale.fit(source["Z"], "Z")
            source["Z"] = scale.apply(source["Z"])
            test["Z"] = scale.apply(test["Z"])
            scales.append(scale.record())
        stage_train = [
            recurrent_data(row, "Z") for row in prepared_train
        ]
        stage_test = [
            recurrent_data(row, "Z") for row in prepared_test
        ]
        main = MixedForecaster(
            [
                (row["Y"].shape[-1], row["Z"].shape[-1])
                for row in train
            ],
            spec["latent"], STAGE1_SHARING,
            spec["seed"] + 2, decoder_width=BEHAVIOR_WIDTH,
        )
        record, parent = run_stage(
            "behavior_relevant", main, stage_train,
            stage_test, names, spec, component_root,
            parent, budget, devices, microbatch,
        )
        stages["behavior_relevant"] = record
        _, main_train = stage_predictions(main, stage_train)
        _, main_test = stage_predictions(main, stage_test)
        readout = ReadoutStage(
            [
                (spec["latent"], row["Y"].shape[-1])
                for row in train
            ],
            NEURAL_WIDTH, spec["seed"] + 3,
        )
        record, parent = run_stage(
            "neural_readout", readout,
            readout_data(neural_train, main_train),
            readout_data(neural_test, main_test),
            names, spec, component_root, parent,
            budget, devices, microbatch,
        )
        stages["neural_readout"] = record
        residual = MixedForecaster(
            dimensions, spec["latent"], STAGE2_SHARING,
            spec["seed"] + 4,
            input_dimensions=[
                row[0] + spec["latent"] + 2
                for row in dimensions
            ],
            forward_dimensions=spec["latent"] + 2,
        )
        record, _ = run_stage(
            "residual", residual,
            residual_data(main, readout, neural_train),
            residual_data(main, readout, neural_test),
            names, spec, component_root,
            parent, budget, devices, microbatch,
        )
        stages["residual"] = record
        total_flops = sum(
            row["component_flops"] for row in stages.values()
        )
        criterion = (
            stages["behavior_relevant"]["score"]["mean_ratio"]
            + stages["residual"]["score"]["mean_ratio"]
        )
        if not np.isfinite(criterion) or total_flops <= 0:
            raise ValueError("Invalid source selection score or FLOPs.")
        paths.append(dict(
            budget=budget, branch_updates=spec["source_training"][
                "branch_updates"
            ],
            stages=stages, full_path_flops=total_flops,
            selection_score=criterion,
            behavior_scales=scales,
        ))
        transferable[budget] = dict(
            stage1=main.transferable(),
            stage2=residual.transferable(),
        )
    selected = min(
        paths, key=lambda row: (
            row["selection_score"], row["full_path_flops"]
        )
    )
    dump_pickle(
        directory / "shared.pkl", transferable[selected["budget"]]
    )
    atomic_json(directory / "budgets.json", paths)
    atomic_json(directory / "selection.json", dict(
        selected_budget=selected["budget"],
        selection_score=selected["selection_score"],
        selected_full_path_flops=selected["full_path_flops"],
        evaluation_sessions=names,
        evaluation_window_counts={
            name: len(records[name]["validation"])
            for name in names
        },
        evaluation_set_id=split_identity,
        holdout_role="chronological_source_test_and_transfer_selection",
    ))
