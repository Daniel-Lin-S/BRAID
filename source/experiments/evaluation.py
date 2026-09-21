"""Score saved forecasts and aggregate independent session means.

metrics.json contains one row per horizon and neural evaluation set, with
per-channel CC/R2/MSE, aggregate values, validity counts, and baseline errors.
summary.csv retains configuration, horizon, target, metric, mean, SEM, and
number of sessions. Undefined metrics are explicit null values with warnings.
"""

import csv
import json
import logging
from pathlib import Path

import numpy as np

from BRAID.tools.evaluation import evalPrediction

from .analysis import read_manifest, validate_member
from .cache import atomic_json

LOGGER = logging.getLogger(__name__)
METRICS = {"cc": "CC", "r2": "R2", "mse": "MSE"}


def _evaluation_context(context: dict) -> str:
    """Format stable evaluation identifiers for diagnostic warnings."""
    fields = (
        "session", "fold", "model", "horizon", "evaluation_set", "target",
    )
    return " ".join(f"{field}={context[field]}" for field in fields)


def score_channels(
    truth: np.ndarray,
    predicted: np.ndarray,
    channel_ids: list[str],
    context: dict,
) -> dict:
    """Score channels and diagnose expected and unexpected exclusions.

    Parameters
    ----------
    truth : ndarray, shape (samples, channels)
        Finite held-out observations.
    predicted : ndarray, shape (samples, channels)
        Finite forecasts aligned with ``truth``.
    channel_ids : list of str
        Stable identifiers for the channel axis.
    context : dict
        Session, fold, model, horizon, evaluation set and target identifiers.

    Returns
    -------
    dict
        Per-channel values, finite-channel means and validity counts.
    """
    if (
        truth.ndim != 2 or truth.shape != predicted.shape
        or len(truth) < 2 or truth.shape[1] < 1
    ):
        raise ValueError(
            f"Expected matching nonempty forecasts: "
            f"{truth.shape}, {predicted.shape}."
        )
    if not np.isfinite(truth).all() or not np.isfinite(predicted).all():
        raise ValueError("Scored observations and forecasts must be finite.")
    if len(channel_ids) != truth.shape[1]:
        raise ValueError(
            f"Expected {truth.shape[1]} channel IDs, got {len(channel_ids)}."
        )
    required = {
        "session", "fold", "model", "horizon", "evaluation_set", "target",
    }
    if set(context) != required:
        raise ValueError(
            f"Expected evaluation context fields {sorted(required)}, "
            f"got {sorted(context)}."
        )
    output = {}
    flat = np.ptp(truth, axis=0) == 0
    if flat.any():
        affected = [channel_ids[i] for i in np.flatnonzero(flat)]
        LOGGER.warning(
            "Ignoring flat truth channels for CC and R2 because their "
            "observed range is zero; %s affected_count=%d "
            "total_channels=%d affected_channels=%s",
            _evaluation_context(context), len(affected), len(channel_ids),
            affected,
        )
    for name, measure in METRICS.items():
        values = np.asarray(evalPrediction(truth, predicted, measure))
        values = np.atleast_1d(values).astype(float)
        if values.shape != (truth.shape[1],):
            raise ValueError(
                f"Expected {truth.shape[1]} {name} values, got "
                f"shape {values.shape}."
            )
        if name in ("cc", "r2"):
            values[flat] = np.nan
        good = np.isfinite(values)
        unexpected = ~good & (~flat if name in ("cc", "r2") else True)
        if np.any(unexpected):
            indices = np.flatnonzero(unexpected)
            LOGGER.warning(
                "Unexpected nonfinite %s values although truth and "
                "predictions are finite and truth range is nonzero; %s "
                "affected_count=%d total_channels=%d affected_channels=%s "
                "values=%s",
                name.upper(),
                _evaluation_context(context),
                len(indices),
                len(channel_ids),
                [channel_ids[i] for i in indices],
                [str(values[i]) for i in indices],
            )
        output[f"per_dimension_{name}"] = [
            float(v) if ok else None for v, ok in zip(values, good)
        ]
        output[f"mean_{name}"] = (
            float(values[good].mean()) if good.any() else None
        )
        output[f"valid_{name}_channels"] = int(good.sum())
    return output


def evaluate_forecasts(
    predictions: dict,
    truth: dict,
    horizons: list[int],
    common: np.ndarray | None,
    metadata: dict,
    directory: Path,
    baseline: dict,
) -> list[dict]:
    """Score full outputs and optionally the experiment's common channels."""
    groups = [("full", np.arange(truth["Y"].shape[1]))]
    if common is not None:
        if not len(common) or len(np.unique(common)) != len(common):
            raise ValueError("Expected nonempty unique common-channel indices.")
        if np.any(common < 0) or np.any(common >= truth["Y"].shape[1]):
            raise ValueError("Common indices are outside the fitted outputs.")
        groups.append(("common", common))
    rows = []
    for number, horizon in enumerate(horizons):
        valid = predictions["valid"][number]
        for name, selected in groups:
            shared_context = dict(
                session=metadata["session"],
                fold=metadata["fold"],
                model=metadata["configuration"],
                horizon=horizon,
                evaluation_set=name,
            )
            behavior = score_channels(
                truth["Z"][valid],
                predictions["Z"][number, valid],
                [f"behavior_{i}" for i in range(truth["Z"].shape[1])],
                dict(shared_context, target="behavior"),
            )
            channel_ids = [
                metadata["selected_channel_ids"][i] for i in selected
            ]
            neural = score_channels(
                truth["Y"][valid][:, selected],
                predictions["Y"][number, valid][:, selected],
                channel_ids,
                dict(shared_context, target="neural"),
            )
            row = dict(
                metadata,
                horizon=horizon,
                evaluation_set=name,
                behavior=behavior,
                neural=neural,
                samples=int(valid.sum()),
                scored_channel_ids=channel_ids,
            )
            for target, columns in (
                ("Y", selected),
                ("Z", np.arange(truth["Z"].shape[1])),
            ):
                observed = truth[target][valid][:, columns]
                mse = np.mean(
                    (observed - baseline[target][columns]) ** 2, axis=0
                )
                if not np.isfinite(mse).all():
                    raise ValueError(f"Nonfinite {target} baseline error.")
                row[f"{target}_baseline_mse"] = mse.tolist()
            rows.append(row)
    atomic_json(directory / "metrics.json", rows)
    return rows


def collect_results(root: Path) -> list[dict]:
    """Read completed metrics only from this analysis's explicit membership."""
    rows = []
    for member in read_manifest(root)["members"].values():
        if member.get("state") != "complete":
            continue
        path = validate_member(root, member)
        result = json.loads(path.read_text())
        if not result:
            raise ValueError(f"Empty completed metrics: {path}")
        if any(
            row["fit_id"] != member["fit_id"]
            or row["session"] != member["session"]
            or row["fold"] != member["fold"]
            or row["configuration"] != member["case"]["name"]
            for row in result
        ):
            raise ValueError(
                f"Metrics do not match analysis membership: {path}"
            )
        rows.extend(result)
    if not rows:
        raise ValueError(f"No completed evaluation results: {root.resolve()}")
    return rows


def aggregate_folds(rows: list[dict]) -> list[dict]:
    """Calculate per-session fold statistics for in-memory plotting.

    Parameters
    ----------
    rows : list of dict
        Completed evaluation rows from individual folds.

    Returns
    -------
    list of dict
        Per-session means and sample standard deviations. No files are written.
    """
    groups = {}
    names = (
        "session",
        "configuration",
        "population_scale",
        "nx",
        "n1",
        "horizon",
        "evaluation_set",
        "target",
        "metric",
    )
    for row in rows:
        for target in ("behavior", "neural"):
            for metric in METRICS:
                key = (
                    row["session"],
                    row["configuration"],
                    row["population_scale"],
                    row["nx"],
                    row["n1"],
                    row["horizon"],
                    row["evaluation_set"],
                    target,
                    metric,
                )
                groups.setdefault(key, []).append(
                    row[target][f"mean_{metric}"]
                )
    summaries = []
    for key, values in sorted(groups.items()):
        finite = np.asarray(
            [value for value in values if value is not None], dtype=float
        )
        if finite.size and not np.isfinite(finite).all():
            identity = dict(zip(names, key))
            raise ValueError(f"Nonfinite fold aggregate for {identity}.")
        summaries.append(
            dict(
                zip(names, key),
                mean=float(finite.mean()) if finite.size else None,
                std=(
                    float(finite.std(ddof=1))
                    if finite.size > 1
                    else None
                ),
                folds=int(finite.size),
            )
        )
    return summaries


def aggregate(rows: list[dict], destination: Path) -> list[dict]:
    """Average folds per session, then compute sample SEM across sessions."""
    if not rows:
        raise ValueError("Cannot aggregate empty evaluation rows.")
    groups = {}
    for row in rows:
        for target in ("behavior", "neural"):
            for metric in METRICS:
                key = (
                    row["configuration"],
                    row["population_scale"],
                    row["nx"],
                    row["n1"],
                    row["horizon"],
                    row["evaluation_set"],
                    target,
                    metric,
                )
                groups.setdefault(key, {}).setdefault(
                    row["session"], []
                ).append(row[target][f"mean_{metric}"])
    summaries = []
    names = (
        "configuration",
        "population_scale",
        "nx",
        "n1",
        "horizon",
        "evaluation_set",
        "target",
        "metric",
    )
    for key, sessions in sorted(groups.items()):
        valid = all(
            all(v is not None for v in values) for values in sessions.values()
        )
        means = (
            np.array([np.mean(values) for values in sessions.values()])
            if valid
            else np.array([])
        )
        mean = float(means.mean()) if valid else None
        sem = (
            float(means.std(ddof=1) / np.sqrt(len(means)))
            if valid and len(means) > 1
            else None
        )
        summaries.append(
            dict(
                zip(names, key),
                mean=mean,
                sem=sem,
                sessions=len(sessions),
                fold_counts={s: len(v) for s, v in sessions.items()},
                aggregation="mean_folds_then_sample_sem_sessions",
            )
        )
    destination.mkdir(parents=True, exist_ok=True)
    atomic_json(destination / "summary.json", summaries)
    atomic_json(destination / "raw_metrics.json", rows)
    with (destination / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    with (destination / "raw_metrics.csv").open("w", newline="") as stream:
        fields = [
            "session",
            "fold",
            "configuration",
            "population_scale",
            "nx",
            "n1",
            "horizon",
            "evaluation_set",
            "target",
            "metric",
            "value",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            base = {key: row[key] for key in fields[:8]}
            for target in ("neural", "behavior"):
                for metric in METRICS:
                    writer.writerow(
                        dict(
                            base,
                            target=target,
                            metric=metric,
                            value=row[target][f"mean_{metric}"],
                        )
                    )
    return summaries
