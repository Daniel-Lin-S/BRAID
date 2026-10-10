"""Adapt frozen source dynamics through the native single-session BRAID path.

Inputs are an immutable target subset, native model configuration, and optional
shared source weights. Outputs include native model.p, component histories,
selection.npz, training budget accounting, and test predictions/metrics.
Target preprocessing is an independently completed, reusable dependency.
"""

from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np
import tensorflow as tf

from BRAID.BRAIDModel import BRAIDModel
from BRAID.MainModel import MainModel
from BRAID.config import resolve_braid_fit_arguments

from experiments.models.braid_backend import BRAIDBackend, MISSING_MARKER
from experiments.storage.cache import atomic_json, file_digest
from experiments.contracts import FeatureSet
from experiments.reports.evaluation import score_channels
from experiments.datasets.nhp import TEST
from experiments.multisession.data import LENGTH
from experiments.datasets.windows import window_indices


def fit_arguments(
    configuration: dict, batch: int, latent: int, epochs: int
) -> dict:
    """Resolve the fixed native architecture and target optimization."""
    configuration = deepcopy(configuration)
    for name in ("stage_1", "stage_2"):
        configuration["model"][name]["neural_decoder"]["hidden_size"] = 256
    configuration["forecast"]["use_observed_input_in_dynamics"] = True
    arguments = resolve_braid_fit_arguments(configuration)
    arguments.update(nx=latent * 2, n1=latent, n3=None, n_pre=150)
    arguments["args_base"].update(
        batch_size=batch,
        epochs=epochs,
        verbose=False,
        epoch_artifacts=True,
        clear_graph=False,
    )
    return arguments


def training_arrays(
    features: FeatureSet, training: np.ndarray, validation: np.ndarray
) -> dict:
    """Extract dimension-first arrays from the exact selected windows."""
    result = {}
    for name in ("Y", "Z", "U"):
        result[name] = features.arrays[name][training.ravel()].T
        result[name + "_validation"] = features.arrays[name][
            validation.ravel()
        ].T
    return result


def component_directory(directory: Path) -> Path:
    """Preserve completed native stages independently of fit attempts.

    Incomplete component leaves move to quarantine before native fitting.
    Completed leaves are validated and reused by BRAID's native callbacks.
    """
    root = directory.parent.parent / "target_components" / directory.name
    if root.exists():
        for history in list(root.rglob("history.jsonl")):
            marker = history.parent / "complete.json"
            if marker.exists():
                record = json.loads(marker.read_text())
                for name, digest in record["checksums"].items():
                    if file_digest(history.parent / name) != digest:
                        raise ValueError(
                            f"Corrupt target component: {marker.resolve()}"
                        )
                continue
            quarantine = root.parent / "quarantine"
            quarantine.mkdir(exist_ok=True)
            history.parent.rename(
                quarantine
                / (f"{directory.name}-{history.parent.name}-{time.time_ns()}")
            )
    root.mkdir(parents=True, exist_ok=True)
    return root


def save_component_records(
    directory: Path, root: Path, arguments: dict, model: MainModel
) -> None:
    """Persist dependency checksums, fitted normalization and spent budget."""
    from experiments.multisession.store import dump_pickle

    records = {
        str(path.parent.relative_to(root)): json.loads(path.read_text())
        for path in root.rglob("complete.json")
    }
    if not records:
        raise ValueError("Target fit completed without component records.")
    atomic_json(directory / "components.json", records)
    atomic_json(
        directory / "budget.json",
        history_budget(
            root,
            arguments["args_base"]["epochs"],
        ),
    )
    normalization = {
        name: getattr(model, name, None)
        for name in ("YPrepMap", "UPrepMap", "ZPrepMap")
    }
    dump_pickle(directory / "normalization.pkl", normalization)


def fit_preprocessing(
    arrays: dict, arguments: dict, directory: Path, seed: int
) -> None:
    """Fit only the reusable native n_pre=150 preprocessing model."""
    tf.keras.utils.set_random_seed(seed)
    components = component_directory(directory)
    model = MainModel(
        log_dir=str(components),
        artifact_role="behaviour_preprocess",
        missing_marker=MISSING_MARKER,
    )
    base = deepcopy(arguments["args_base"])
    base.update(model2_Cz_Full=False, allow_nonzero_Cz2=True, has_UFT_reg=False)
    model.fit(
        **arrays, nx=150, n1=0, YType="cont", ZType="cont", UType="cont", **base
    )
    if model.hasBlownUp():
        raise FloatingPointError("Target preprocessing fit is nonfinite.")
    model.saveToFile(str(directory / "model.p"))
    save_component_records(directory, components, arguments, model)


def history_budget(directory: Path, allowed_epochs: int) -> dict:
    """Collect actual epoch counts and available optimizer-step evidence."""
    components = {}
    for path in directory.rglob("history.jsonl"):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        components[str(path.parent.relative_to(directory))] = dict(
            epochs=len(rows),
            allowed_epochs=allowed_epochs,
            optimizer_updates=rows[-1]["optimizer_updates"] if rows else 0,
            last_epoch=rows[-1]["epoch"] if rows else None,
        )
    return dict(components=components)


def fit_target(
    arrays: dict,
    arguments: dict,
    directory: Path,
    preprocessing: Path,
    weights: dict,
    seed: int,
) -> BRAIDModel:
    """Fit target-private components while imported modules stay frozen."""
    tf.keras.utils.set_random_seed(seed)
    pre = MainModel.loadFromFile(str(preprocessing / "model.p"))
    # Loading the dependency must not change paired main-stage initialization.
    tf.keras.utils.set_random_seed(seed)
    components = component_directory(directory)
    model = BRAIDModel(
        log_dir=str(components),
        missing_marker=MISSING_MARKER,
        preprocessing_model=pre,
        frozen_stage1=weights,
    )
    model.fit(**arrays, YType="cont", ZType="cont", UType="cont", **arguments)
    if model.hasBlownUp():
        raise FloatingPointError("Nonfinite target BRAID fit.")
    for name, expected in weights.items():
        actual = getattr(model.sId.model1.rnn.cell, name).get_weights()
        for left, right in zip(expected, actual):
            np.testing.assert_array_equal(left, right)
    model.saveToFile(str(directory / "model.p"))
    model.restoreModels()
    save_component_records(directory, components, arguments, model.sId)
    return model


def evaluate_target(
    model: BRAIDModel,
    features: FeatureSet,
    directory: Path,
    identity: dict,
    horizons: list[int],
) -> None:
    """Save fixed-window forecasts and original-unit per-dimension scores."""
    test = window_indices(features, TEST, LENGTH).ravel()
    arrays = features.arrays
    backend = object.__new__(BRAIDBackend)
    backend.model, backend.length = model, LENGTH
    predicted = backend.predict(arrays["Y"][test], arrays["U"][test], horizons)
    np.savez_compressed(
        directory / "predictions.npz",
        **predicted,
        true_Y=arrays["Y"][test],
        true_Z=arrays["Z"][test],
        indices=arrays["indices"][test],
        horizons=np.asarray(horizons),
    )
    rows = []
    for hi, horizon in enumerate(horizons):
        valid = predicted["valid"][hi]
        for name, kind in (("Y", "neural"), ("Z", "behavior")):
            ids = (
                arrays["ids"].tolist()
                if name == "Y"
                else ["position_x", "position_y", "velocity_x", "velocity_y"]
            )
            context = dict(
                session=identity["session"],
                fold=identity["fold"],
                model=identity["kind"],
                horizon=horizon,
                evaluation_set="full",
                target=kind,
            )
            result = score_channels(
                arrays[name][test][valid],
                predicted[name][hi, valid],
                ids,
                context,
            )
            rows.append(
                dict(
                    horizon=horizon,
                    target=kind,
                    samples=int(valid.sum()),
                    coordinates=ids,
                    **result,
                )
            )
    atomic_json(directory / "metrics.json", rows)
