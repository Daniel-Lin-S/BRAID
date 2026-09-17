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

from .cache import atomic_json

LOGGER = logging.getLogger(__name__)
METRICS = {"cc": "CC", "r2": "R2", "mse": "MSE"}


def score_channels(truth: np.ndarray, predicted: np.ndarray) -> dict:
    """Reuse BRAID metrics while marking undefined channel scores explicitly."""
    if truth.shape != predicted.shape or len(truth) < 2:
        raise ValueError(
            f"Expected matching nonempty forecasts: "
            f"{truth.shape}, {predicted.shape}."
        )
    if not np.isfinite(truth).all() or not np.isfinite(predicted).all():
        raise ValueError("Scored observations and forecasts must be finite.")
    output = {}
    flat = np.ptp(truth, axis=0) == 0
    for name, measure in METRICS.items():
        values = np.asarray(evalPrediction(truth, predicted, measure))
        values = np.atleast_1d(values).astype(float)
        if name in ("cc", "r2"):
            values[flat] = np.nan
        good = np.isfinite(values)
        if not good.all():
            LOGGER.warning(
                "%s undefined for %d/%d channels",
                name,
                int((~good).sum()),
                len(good),
            )
        output[f"per_dimension_{name}"] = [
            float(v) if ok else None for v, ok in zip(values, good)
        ]
        output[f"mean_{name}"] = float(values.mean()) if good.all() else None
        output[f"valid_{name}_channels"] = int(good.sum())
    return output


def evaluate_forecasts(
    predictions: dict,
    truth: dict,
    horizons: list[int],
    common: np.ndarray,
    metadata: dict,
    directory: Path,
    baseline: dict,
) -> list[dict]:
    """Score both full-output and fixed-smallest-population neural forecasts."""
    rows = []
    for number, horizon in enumerate(horizons):
        valid = predictions["valid"][number]
        behavior = score_channels(
            truth["Z"][valid], predictions["Z"][number, valid]
        )
        for name, selected in (
            ("full", np.arange(truth["Y"].shape[1])),
            ("smallest", common),
        ):
            neural = score_channels(
                truth["Y"][valid][:, selected],
                predictions["Y"][number, valid][:, selected],
            )
            row = dict(
                metadata,
                horizon=horizon,
                evaluation_set=name,
                behavior=behavior,
                neural=neural,
                samples=int(valid.sum()),
            )
            for target, columns in (
                ("Y", selected),
                ("Z", np.arange(truth["Z"].shape[1])),
            ):
                observed = truth[target][valid][:, columns]
                mse = np.mean(
                    (observed - baseline[target][columns]) ** 2, axis=0
                )
                row[f"{target}_baseline_mse"] = mse.tolist()
            rows.append(row)
    atomic_json(directory / "metrics.json", rows)
    np.savez_compressed(
        directory / "predictions.npz",
        **predictions,
        true_Y=truth["Y"],
        true_Z=truth["Z"],
        t=truth["t"],
        source_indices=truth["indices"],
        horizons=horizons,
    )
    return rows


def collect_results(root: Path) -> list[dict]:
    """Read only completed evaluation artifacts, excluding unfinished runs."""
    rows = []
    completed = set()
    for path in sorted(root.glob("*/fold_*/*/metrics.json")):
        status = path.parent / "status.json"
        if (
            status.exists()
            and json.loads(status.read_text())["state"] == "complete"
        ):
            result = json.loads(path.read_text())
            if not result:
                raise ValueError(f"Empty completed metrics: {path.resolve()}")
            key = tuple(
                result[0][name]
                for name in (
                    "session",
                    "fold",
                    "configuration",
                )
            )
            if key in completed:
                raise ValueError(
                    "Multiple completed configurations for the same "
                    "session/fold/case; use a separate artifact root: "
                    f"{path.parent.resolve()}"
                )
            completed.add(key)
            rows.extend(result)
    if not rows:
        raise ValueError(
            f"No completed evaluation results under {root.resolve()}"
        )
    return rows


def aggregate(rows: list[dict], destination: Path) -> list[dict]:
    """Average folds per session, then compute sample SEM across sessions."""
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
