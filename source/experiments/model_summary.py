"""Maintain model-level completion and performance tables from saved runs.

model_manifest.json identifies model settings and expected session/fold pairs.
model_summary.csv has rows per run, horizon, evaluation set and output target,
with CC/R2/MSE, training-mean baseline MSE and valid-channel counts. Missing or
failed folds have status rows with empty metric cells, never fabricated scores.
Summary publication is atomic and serialized across concurrent launches.
"""

import csv
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from .artifacts import artifact_path, discover_runs, model_directory
from .artifacts import model_identity, validate_completion
from .cache import atomic_json, writer_lock

FIELDS = (
    "session",
    "fold",
    "run_identity",
    "status",
    "horizon",
    "evaluation_set",
    "target",
    "cc",
    "r2",
    "mse",
    "baseline_mse",
    "samples",
    "valid_cc_channels",
    "valid_r2_channels",
    "valid_mse_channels",
    "metrics_defined",
    "details",
)


def register_model_work(
    root: Path,
    snapshots: dict,
    case: dict,
    sessions: list[str],
    folds: list[int],
) -> None:
    """Register requested session/fold pairs without discarding prior work."""
    directory = model_directory(root, snapshots, case)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "model_manifest.json"
    identity = model_identity(snapshots, case)
    pairs = {(s, f) for s in sessions for f in folds}
    with writer_lock(directory / "summary.lock"):
        if path.exists():
            existing = json.loads(path.read_text())
            if existing["identity"] != identity:
                raise ValueError(
                    f"Model settings mismatch: {directory.resolve()}"
                )
            pairs.update(tuple(pair) for pair in existing["expected"])
        references = existing.get("runs", []) if path.exists() else []
        atomic_json(
            path,
            dict(identity=identity, expected=sorted(pairs), runs=references),
        )


def completed_rows(run: Path, status: dict) -> list[dict]:
    """Validate completed artifacts and flatten per-target aggregate metrics."""
    validate_completion(run, status)
    path = artifact_path(run, "metrics.json")
    metrics = json.loads(path.read_text())
    if not metrics:
        raise ValueError(f"Completed metrics are empty: {path}")
    rows = []
    for item in metrics:
        for target, short in (("neural", "Y"), ("behavior", "Z")):
            scores = item[target]
            baseline = np.asarray(item[f"{short}_baseline_mse"], dtype=float)
            if not baseline.size or not np.isfinite(baseline).all():
                raise ValueError(f"Invalid baseline scores in {path}")
            row = dict(
                session=item["session"],
                fold=item["fold"],
                run_identity=run.name,
                status="complete",
                horizon=item["horizon"],
                evaluation_set=item["evaluation_set"],
                target=target,
                baseline_mse=float(baseline.mean()),
                samples=item["samples"],
                details=str(path),
            )
            for metric in ("cc", "r2", "mse"):
                value = scores[f"mean_{metric}"]
                if value is not None and not np.isfinite(value):
                    raise ValueError(f"Nonfinite {metric} in {path}")
                row[metric] = value
                row[f"valid_{metric}_channels"] = scores[
                    f"valid_{metric}_channels"
                ]
            row["metrics_defined"] = all(
                row[m] is not None for m in ("cc", "r2", "mse")
            )
            rows.append(row)
    return rows


def write_model_summary(directory: Path) -> None:
    """Publish a complete model table with explicit missing/failed fold rows."""
    with writer_lock(directory / "summary.lock"):
        manifest = json.loads((directory / "model_manifest.json").read_text())
        expected = {tuple(pair) for pair in manifest["expected"]}
        found = set()
        completed = set()
        rows = []
        for run in sorted(
            set(discover_runs(directory))
            | {Path(p) for p in manifest.get("runs", [])}
        ):
            identity = json.loads(
                artifact_path(run, "identity.json").read_text()
            )
            if (
                model_identity(identity["configurations"], identity["case"])
                != manifest["identity"]
            ):
                raise ValueError(f"Incompatible run in model directory: {run}")
            pair = (identity["source"]["session"], identity["fold"])
            found.add(pair)
            status_path = artifact_path(run, "status.json")
            status = (
                json.loads(status_path.read_text())
                if status_path.exists()
                else {"state": "incomplete"}
            )
            if status["state"] == "complete":
                if pair in completed:
                    raise ValueError(
                        f"Multiple completed runs for {pair}: {directory}"
                    )
                completed.add(pair)
                result = completed_rows(run, status)
                if any((row["session"], row["fold"]) != pair for row in result):
                    raise ValueError(
                        f"Metrics do not match run identity: {run}"
                    )
                rows.extend(result)
            else:
                rows.append(
                    dict(
                        session=pair[0],
                        fold=pair[1],
                        run_identity=run.name,
                        status=status["state"],
                        details=str(status_path),
                    )
                )
        rows.extend(
            dict(session=s, fold=f, status="missing")
            for s, f in sorted(expected - found)
        )
        if not rows:
            raise ValueError(f"No expected or recorded work: {directory}")
        fd, temporary = tempfile.mkstemp(dir=directory, suffix=".csv.tmp")
        try:
            with os.fdopen(fd, "w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            os.replace(temporary, directory / "model_summary.csv")
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


def register_model_run(directory: Path, run: Path) -> None:
    """Reference one canonical run, including compatible historical layouts."""
    with writer_lock(directory / "summary.lock"):
        path = directory / "model_manifest.json"
        manifest = json.loads(path.read_text())
        manifest["runs"] = sorted(
            set(manifest.get("runs", [])) | {str(run.resolve())}
        )
        atomic_json(path, manifest)
