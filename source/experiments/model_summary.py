"""Publish per-fit status and score tables for one analysis manifest.

summaries/model_summary.csv has rows per member, horizon, scoring set and
target. Pending/failed members retain explicit status with no invented scores.
"""

import csv
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from .analysis import read_manifest, validate_member
from .cache import writer_lock

FIELDS = (
    "configuration", "session", "fold", "fit_id", "status", "horizon",
    "evaluation_set", "target", "cc", "r2", "mse", "baseline_mse", "samples",
    "valid_cc_channels", "valid_r2_channels", "valid_mse_channels",
    "metrics_defined", "details",
)


def completed_rows(directory: Path, member: dict) -> list[dict]:
    """Flatten validated metrics without touching their owning fit."""
    path = validate_member(directory, member)
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
                configuration=item["configuration"],
                session=item["session"], fold=item["fold"],
                fit_id=member["fit_id"], status="complete",
                horizon=item["horizon"], evaluation_set=item["evaluation_set"],
                target=target, baseline_mse=float(baseline.mean()),
                samples=item["samples"], details=str(path),
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
    """Atomically publish analysis-scoped completion and score rows."""
    destination = directory / "summaries"
    destination.mkdir(parents=True, exist_ok=True)
    with writer_lock(destination / "summary.lock"):
        manifest = read_manifest(directory)
        rows = []
        for member in manifest["members"].values():
            if member.get("state") == "complete":
                rows.extend(completed_rows(directory, member))
            else:
                rows.append(dict(
                    configuration=member["case"]["name"],
                    session=member["session"], fold=member["fold"],
                    fit_id=member["fit_id"], status=member["state"],
                    details=member.get("error"),
                ))
        if not rows:
            raise ValueError(f"No requested analysis work: {directory}")
        fd, temporary = tempfile.mkstemp(dir=destination, suffix=".csv.tmp")
        try:
            with os.fdopen(fd, "w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            os.replace(temporary, destination / "model_summary.csv")
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
