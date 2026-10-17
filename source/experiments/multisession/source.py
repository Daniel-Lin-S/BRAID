"""Source neural pretraining and five-component supervision sensitivity.

A source job consumes session windows with training-only normalization.
Its shared.pkl stores only transferable Afw/Kfw/C arrays. Component records
reference immutable stage checkpoints, histories, and exposure accounting.
The neural-only branch never constructs or reads a behavior objective.
"""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from BRAID.multisession import HORIZONS, MixedForecaster

from experiments.storage.cache import atomic_json, fingerprint
from experiments.multisession.data import Scale
from experiments.multisession.store import (
    completed, dump_pickle, job, load_pickle,
)
from experiments.multisession.training import (
    ReadoutStage,
    recurrent_data,
    stage_predictions,
    train_stage,
)

PREPROCESSING_SIZE = 150
NEURAL_WIDTH = 256
BEHAVIOR_WIDTH = 64
COMPONENT_COUNT = 5
NEURAL_STAGE_COUNT = 2


def fit_component(
    model: object,
    train: list,
    validation: list,
    root: Path,
    identity: dict,
    settings: dict,
    seed: int,
) -> dict:
    """Reuse or complete a separately addressed source component."""
    identity = dict(identity, settings=settings)
    if "parent" in identity:
        parent = completed(root / identity["parent"])
        if parent is None:
            raise RuntimeError("Source component dependency is incomplete.")
        identity["parent_checksums"] = parent["checksums"]
    directory = root / fingerprint(identity)
    with job(directory, identity):
        result = train_stage(
            model, train, validation, directory, settings, seed
        )
    return dict(component_id=directory.name, training=result)


def readout_data(data: list[dict], states: list[np.ndarray]) -> list[dict]:
    """Attach horizon-first latent states to a fixed decoder objective."""
    return [dict(row, I=np.moveaxis(x, 0, -1)) for row, x in zip(data, states)]


def aligned_one_step(states: np.ndarray) -> np.ndarray:
    """Align filtered x(t+1|t) as x(t|t-1), resetting each window."""
    return np.concatenate(
        (np.zeros_like(states[:, :1]), states[:, :-1]), axis=1
    )


def behavior_targets(
    readout: ReadoutStage, states: list[np.ndarray], raw: list[dict]
) -> list[dict]:
    """Construct preprocessed behavior from past-neural latent states."""
    result = []
    for model, x, values in zip(readout.models, states, raw):
        z = model.apply_func(aligned_one_step(x[0])).numpy()
        result.append(dict(values, Z=z))
    return result


def residual_data(
    main: MixedForecaster,
    decoder: ReadoutStage | None,
    data: list[dict],
) -> list[dict]:
    """Prepare residual neural inputs and additive stage-1 predictions."""
    from experiments.multisession.training import select_batch, MAX_HORIZON

    all_states = [[] for _ in data]
    all_predictions = [[] for _ in data]
    for start in range(0, max(len(d["T"]) for d in data), 32):
        indices = [np.arange(start, min(start + 32, len(d["T"]))) for d in data]
        inputs, forward, _, _ = select_batch(data, indices)
        predictions, states = main.forecast(
            inputs, forward, tuple(range(1, MAX_HORIZON + 1))
        )
        for i in range(len(data)):
            if len(indices[i]):
                all_states[i].append(states[i].numpy())
                if decoder is None:
                    all_predictions[i].append(predictions[i].numpy())
    result = []
    readouts = decoder.models if decoder is not None else [None] * len(data)
    for parts, forecasts, row, readout in zip(
        all_states, all_predictions, data, readouts
    ):
        x = np.concatenate(parts, axis=1)
        if readout is None:
            predicted = np.concatenate(forecasts, axis=1)
            prior = predicted[np.asarray(HORIZONS) - 1]
        else:
            prior = np.stack(
                [readout.apply_func(x[h - 1]).numpy() for h in HORIZONS]
            )
        result.append(
            dict(
                row,
                I=np.concatenate((aligned_one_step(x[0]), row["I"]), -1),
                F=np.concatenate((np.moveaxis(x, 0, -1), row["F"]), axis=2),
                P=prior,
            )
        )
    return result


def restore_first_stage(
    model: MixedForecaster,
    directory: Path,
    source_id: str,
    stage_settings: dict,
) -> dict:
    """Restore a completed A component after checking the stage cap."""
    source_directory = directory.parent / source_id
    if completed(source_directory) is None:
        raise ValueError(f"Incomplete first-stage source: {source_directory}")
    records = json.loads(
        (source_directory / "components.json").read_text()
    )
    record = records["neural"]
    component = (
        directory.parent.parent / "source_components"
        / record["component_id"]
    )
    if completed(component) is None:
        raise ValueError(f"Incomplete first-stage component: {component}")
    payload = load_pickle(component / "completed.pkl")
    prior = payload["training"]
    cap = stage_settings["maximum_updates"]
    if prior["updates"] > cap:
        raise ValueError(
            f"First stage used {prior['updates']} updates, above {cap}."
        )
    comparable = dict(prior["settings"], maximum_updates=cap)
    if comparable != stage_settings:
        raise ValueError("First-stage training settings differ.")
    model.restore(payload["weights"])
    return record


def fit_source(
    train: list[dict],
    validation: list[dict],
    directory: Path,
    component_root: Path,
    identity: dict,
    settings: dict,
) -> None:
    """Execute the configured objective and publish transferable weights."""
    if identity["objective"] not in (
        "neural", "neural_residual", "behavior"
    ):
        raise ValueError(
            f"Unknown source objective {identity['objective']}."
        )
    latent = identity["latent"]
    sharing, seed = identity["sharing"], identity["seed"]
    supervised = identity["objective"] == "behavior"
    two_stage = identity["objective"] == "neural_residual"
    stage_settings = deepcopy(settings)
    if supervised:
        stage_settings["maximum_updates"] //= COMPONENT_COUNT
    if two_stage:
        stage_settings["maximum_updates"] //= NEURAL_STAGE_COUNT
    base = {k: v for k, v in identity.items() if k != "sharing"}
    records = {}
    dimensions = [(d["Y"].shape[-1], d["Y"].shape[-1]) for d in train]
    neural_train = [recurrent_data(d) for d in train]
    neural_val = [recurrent_data(d) for d in validation]
    if not supervised:
        stage1_sharing = (
            identity.get("stage1_sharing", "AAK")
            if two_stage else sharing
        )
        model = MixedForecaster(dimensions, latent, stage1_sharing, seed)
        if two_stage and "stage1_source" in identity:
            records["neural"] = restore_first_stage(
                model, directory, identity["stage1_source"], stage_settings
            )
        else:
            records["neural"] = fit_component(
                model,
                neural_train,
                neural_val,
                component_root,
                dict(identity, stage="neural"),
                stage_settings,
                seed,
            )
        if two_stage:
            residual = MixedForecaster(
                dimensions,
                latent,
                identity.get("stage2_sharing", "AAK"),
                seed + 4,
                input_dimensions=[
                    d[0] + latent + 2 for d in dimensions
                ],
                forward_dimensions=latent + 2,
            )
            records["residual"] = fit_component(
                residual,
                residual_data(model, None, neural_train),
                residual_data(model, None, neural_val),
                component_root,
                dict(
                    identity,
                    stage="residual",
                    parent=records["neural"]["component_id"],
                ),
                stage_settings,
                seed + 4,
            )
    else:
        pre = MixedForecaster(dimensions, PREPROCESSING_SIZE, "none", seed)
        records["preprocessing_neural"] = fit_component(
            pre,
            neural_train,
            neural_val,
            component_root,
            dict(base, stage="preprocessing_neural"),
            stage_settings,
            seed,
        )
        _, pre_train = stage_predictions(pre, neural_train)
        _, pre_val = stage_predictions(pre, neural_val)
        pre_readout = ReadoutStage(
            [(PREPROCESSING_SIZE, d["Z"].shape[-1]) for d in train],
            BEHAVIOR_WIDTH,
            seed + 1,
            behavior=True,
        )
        ztrain = [recurrent_data(d, "Z") for d in train]
        zval = [recurrent_data(d, "Z") for d in validation]
        records["preprocessing_behavior"] = fit_component(
            pre_readout,
            readout_data(ztrain, pre_train),
            readout_data(zval, pre_val),
            component_root,
            dict(
                base,
                stage="preprocessing_behavior",
                parent=records["preprocessing_neural"]["component_id"],
            ),
            stage_settings,
            seed + 1,
        )
        ztrain = behavior_targets(pre_readout, pre_train, train)
        zval = behavior_targets(pre_readout, pre_val, validation)
        scales = []
        for tr, va in zip(ztrain, zval):
            scale = Scale.fit(tr["Z"], "Z")
            tr["Z"], va["Z"] = scale.apply(tr["Z"]), scale.apply(va["Z"])
            scales.append(scale.record())
        atomic_json(directory / "preprocessed_behavior_scaling.json", scales)
        main_train = [recurrent_data(d, "Z") for d in ztrain]
        main_val = [recurrent_data(d, "Z") for d in zval]
        model = MixedForecaster(
            [(d["Y"].shape[-1], d["Z"].shape[-1]) for d in train],
            latent,
            sharing,
            seed + 2,
            decoder_width=BEHAVIOR_WIDTH,
        )
        records["behavior_relevant"] = fit_component(
            model,
            main_train,
            main_val,
            component_root,
            dict(
                identity,
                stage="behavior_relevant",
                parent=records["preprocessing_behavior"]["component_id"],
            ),
            stage_settings,
            seed + 2,
        )
        _, main_states = stage_predictions(model, main_train)
        _, main_val_states = stage_predictions(model, main_val)
        neural_readout = ReadoutStage(
            [(latent, d["Y"].shape[-1]) for d in train],
            NEURAL_WIDTH,
            seed + 3,
        )
        records["neural_readout"] = fit_component(
            neural_readout,
            readout_data(neural_train, main_states),
            readout_data(neural_val, main_val_states),
            component_root,
            dict(
                identity,
                stage="neural_readout",
                parent=records["behavior_relevant"]["component_id"],
            ),
            stage_settings,
            seed + 3,
        )
        residual = MixedForecaster(
            dimensions,
            latent,
            "none",
            seed + 4,
            input_dimensions=[d[0] + latent + 2 for d in dimensions],
            forward_dimensions=latent + 2,
        )
        # Neural targets remain unchanged; source behavior states only
        # supply the native stage-2 filtering and forecast dependencies.
        residual_train = residual_data(model, neural_readout, neural_train)
        residual_val = residual_data(model, neural_readout, neural_val)
        records["residual"] = fit_component(
            residual,
            residual_train,
            residual_val,
            component_root,
            dict(
                identity,
                stage="residual",
                parent=records["neural_readout"]["component_id"],
            ),
            stage_settings,
            seed + 4,
        )
    transferable = model.transferable()
    if two_stage:
        transferable = dict(
            stage1=transferable,
            stage2=residual.transferable(),
        )
    dump_pickle(directory / "shared.pkl", transferable)
    atomic_json(directory / "components.json", records)
