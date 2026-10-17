"""Evaluate saved source checkpoints on common training-session validation.

Input is a completed sensitivity plan and source fit checkpoints. Output is
one checksum-validated derived cache with metrics.json (one source session
and horizon per row) and losses.json (best macro-validation objective per
source fit). Source data and fit artifacts are read but never modified.
"""

import json
from pathlib import Path

import numpy as np

from BRAID.multisession import HORIZONS, MixedForecaster
from experiments.multisession.data import session_windows, source_features
from experiments.multisession.runner import dataset
from experiments.multisession.sensitivity import LATENT_SIZE
from experiments.multisession.source import residual_data
from experiments.multisession.store import (
    completed, fit_directory, job, load_pickle,
)
from experiments.multisession.training import recurrent_data, stage_predictions
from experiments.reports.evaluation import score_channels
from experiments.storage.cache import atomic_json, file_digest, fingerprint


def _component(plan: dict, fit_id: str, stage: str) -> dict:
    """Read one completed source stage and its native weight arrays."""
    fit = fit_directory(plan, fit_id)
    records = json.loads((fit / "components.json").read_text())
    stage_id = records[stage]["component_id"]
    directory = fit.parent.parent / "source_components" / stage_id
    if completed(directory) is None:
        raise ValueError(f"Incomplete source component: {directory}")
    return load_pickle(directory / "completed.pkl")


def _restore(
    payload: dict, dimensions: list, sharing: str, seed: int,
    *, residual: bool = False,
) -> MixedForecaster:
    """Construct only the common four session paths from saved mappings."""
    latent = LATENT_SIZE
    widths = {}
    if residual:
        widths = dict(
            input_dimensions=[d[0] + latent + 2 for d in dimensions],
            forward_dimensions=latent + 2,
        )
    model = MixedForecaster(dimensions, latent, sharing, seed, **widths)
    selected = {
        name: values for name, values in payload["weights"].items()
        if int(name.split("/", 1)[0]) < len(dimensions)
    }
    model.restore(selected)
    return model


def _evaluate_fit(
    plan: dict, fit_id: str, values: list[dict], names: list[str],
) -> tuple[list[dict], dict]:
    """Score a fitted source predictor on the same four validation sessions."""
    spec = plan["jobs"][fit_id]
    dimensions = [(v["Y"].shape[-1], v["Y"].shape[-1]) for v in values]
    sharing = (
        spec.get("stage1_sharing", "AAK")
        if spec["objective"] == "neural_residual"
        else spec["sharing"]
    )
    main = _restore(
        _component(plan, fit_id, "neural"),
        dimensions, sharing, spec["seed"],
    )
    inputs = [recurrent_data(v) for v in values]
    stage = "neural"
    if spec["objective"] == "neural_residual":
        residual_inputs = residual_data(main, None, inputs)
        residual = _restore(
            _component(plan, fit_id, "residual"),
            dimensions, spec.get("stage2_sharing", "AAK"),
            spec["seed"] + 4, residual=True,
        )
        corrections, _ = stage_predictions(residual, residual_inputs)
        predictions = [
            correction + row["P"]
            for correction, row in zip(corrections, residual_inputs)
        ]
        stage = "residual"
    else:
        predictions, _ = stage_predictions(main, inputs)
    rows = []
    for session, value, forecast in zip(names, values, predictions):
        for index, horizon in enumerate(HORIZONS):
            truth = value["Y"][:, horizon:].reshape(
                -1, value["Y"].shape[-1]
            )
            predicted = forecast[index, :, :-horizon].reshape(
                -1, value["Y"].shape[-1]
            )
            context = dict(
                session=session,
                fold="source_validation",
                model=spec["sharing"],
                horizon=horizon,
                evaluation_set="source_validation",
                target="neural",
            )
            result = score_channels(
                truth, predicted,
                [str(i) for i in range(truth.shape[1])],
                context,
            )
            rows.append(dict(
                fit_id=fit_id,
                objective=spec["objective"],
                sharing=spec["sharing"],
                source_count=spec["source_count"],
                session=session,
                horizon=horizon,
                cc=result["mean_cc"],
                valid_channels=result["valid_cc_channels"],
            ))
    training = _component(plan, fit_id, stage)["training"]
    loss = dict(
        fit_id=fit_id,
        objective=spec["objective"],
        sharing=spec["sharing"],
        source_count=spec["source_count"],
        best_update=training["best_update"],
        updates=training["updates"],
        macro_validation_loss=training["best_loss"],
    )
    return rows, loss


def collect_source_validation(plan: dict) -> tuple[list[dict], list[dict]]:
    """Reuse cached source-validation metrics once every source is complete."""
    ids = plan.get("source_references", [])
    if not ids or any(
        completed(fit_directory(plan, key)) is None for key in ids
    ):
        return [], []
    identity = dict(
        fits=[
            (key, completed(fit_directory(plan, key))["checksums"][
                "components.json"
            ])
            for key in ids
        ],
        evaluator=file_digest(Path(__file__)),
    )
    root = Path(plan["settings"]["paths"]["artifact_root"])
    directory = (
        root / "multi_session_training" / "source_validation"
        / fingerprint(identity)
    )
    with job(directory, identity) as needed:
        if needed:
            anchor = plan["jobs"][ids[0]]["sessions"][:4]
            adapter = dataset(plan["settings"])
            values = []
            for name in anchor:
                features = source_features(
                    adapter.load(name), plan["settings"]["data"]
                )
                _, validation, _ = session_windows(features, False)
                values.append(validation)
            rows, losses = [], []
            for key in ids:
                if plan["jobs"][key]["sessions"][:4] != anchor:
                    raise ValueError(
                        f"Source validation anchor differs in fit {key}."
                    )
                scored, loss = _evaluate_fit(plan, key, values, anchor)
                rows.extend(scored)
                losses.append(loss)
            if not rows or not losses:
                raise ValueError("Source validation cache would be empty.")
            atomic_json(directory / "metrics.json", rows)
            atomic_json(directory / "losses.json", losses)
    return (
        json.loads((directory / "metrics.json").read_text()),
        json.loads((directory / "losses.json").read_text()),
    )
