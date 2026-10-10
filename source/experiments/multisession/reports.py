"""Publish independent scaling and two-condition sensitivity reports.

Each analysis owns immutable reports/<snapshot>/ tables and figures.
raw_metrics.csv contains fit/session/fold/seed, design, metric and value;
per_dimension.csv retains coordinate scores. summary.csv averages repeats
and folds within sessions before computing mean/SEM across sessions.
paired_gain.csv compares each transfer fit with its matched scratch control.
latest_report.json points to the newest nonempty completed snapshot.
"""

from collections import defaultdict
import csv
import json
import logging
from pathlib import Path

import numpy as np

from experiments.storage.cache import atomic_json, fingerprint
from experiments.visualization.multisession import figures
from experiments.multisession.store import completed, fit_directory, job

LOGGER = logging.getLogger(__name__)
DESIGN = (
    "objective",
    "latent",
    "sharing",
    "source_count",
    "fraction",
    "target",
    "horizon",
    "metric",
)
PAIR = (
    "session",
    "fold",
    "seed",
    "fraction",
    "latent",
    "target",
    "horizon",
    "metric",
)


def write_table(path: Path, rows: list[dict]) -> None:
    """Write a nonempty table with stable columns and explicit null values."""
    if not rows:
        raise ValueError(f"Refusing empty report table: {path.resolve()}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def session_means(rows: list[dict]) -> list[dict]:
    """Retain one mean and validity count per target session/condition."""
    groups = defaultdict(list)
    for row in rows:
        key = tuple(row[k] for k in (*DESIGN, "session"))
        groups[key].append(row["value"])
    result = []
    for key, values in sorted(groups.items()):
        finite = [value for value in values if value is not None]
        if finite and not np.isfinite(finite).all():
            raise ValueError("Nonfinite aggregate input.")
        result.append(
            dict(
                zip((*DESIGN, "session"), key),
                value=float(np.mean(finite)) if finite else None,
                valid_fits=len(finite),
                completed_fits=len(values),
            )
        )
    return result


def summarize(rows: list[dict]) -> list[dict]:
    """Aggregate correlated fold/repeat scores before session-level SEM."""
    groups = defaultdict(list)
    for row in session_means(rows):
        groups[tuple(row[k] for k in DESIGN)].append(row)
    result = []
    for key, sessions in sorted(groups.items()):
        valid = [row for row in sessions if row["value"] is not None]
        if not valid:
            continue
        means = np.asarray([row["value"] for row in valid])
        result.append(
            dict(
                zip(DESIGN, key),
                mean=float(means.mean()),
                sem=float(means.std(ddof=1) / np.sqrt(len(means)))
                if len(means) > 1
                else None,
                sessions=len(means),
                session_ids=";".join(sorted(row["session"] for row in valid)),
                contributing_fits=sum(row["valid_fits"] for row in valid),
                invalid_fits=sum(
                    row["completed_fits"] - row["valid_fits"]
                    for row in sessions
                ),
            )
        )
    return result


def paired_gains(rows: list[dict]) -> list[dict]:
    """Calculate within-session/fold/seed differences from matched scratch."""
    baseline = {
        tuple(row[k] for k in PAIR): row
        for row in rows
        if row["sharing"] == "scratch"
    }
    result = []
    for row in rows:
        if row["sharing"] == "scratch":
            continue
        scratch = baseline.get(tuple(row[k] for k in PAIR))
        if scratch is None or scratch["value"] is None or row["value"] is None:
            continue
        result.append(
            dict(
                row,
                scratch_fit=scratch["fit_id"],
                scratch_value=scratch["value"],
                difference=row["value"] - scratch["value"],
            )
        )
    return result


def collect(plan: dict, members: list[str]) -> tuple[list, list, list]:
    """Read metrics only from checksum-validated completed member fits."""
    rows, dimensions, identities = [], [], []
    for key in members:
        directory = fit_directory(plan, key)
        record = completed(directory)
        if record is None:
            continue
        path = directory / "metrics.json"
        if not path.exists():
            raise ValueError(f"Completed evaluation lacks metrics: {directory}")
        identities.append((key, record["checksums"]["metrics.json"]))
        spec = plan["jobs"][key]
        base = {
            k: spec[k]
            for k in (
                "session",
                "fold",
                "seed",
                "fraction",
                "latent",
                "objective",
                "sharing",
                "source_count",
            )
        }
        for item in json.loads(path.read_text()):
            for metric in ("cc", "r2", "mse"):
                common = dict(
                    base,
                    fit_id=key,
                    target=item["target"],
                    horizon=item["horizon"],
                    metric=metric,
                )
                rows.append(
                    dict(
                        common,
                        value=item[f"mean_{metric}"],
                        valid_dimensions=item[f"valid_{metric}_channels"],
                        scored_samples=item["samples"],
                    )
                )
                for coordinate, value in zip(
                    item["coordinates"],
                    item[f"per_dimension_{metric}"],
                ):
                    dimensions.append(
                        dict(common, coordinate=coordinate, value=value)
                    )
    return rows, dimensions, identities



def cost_tables(plan: dict, members: list[str], snapshot: Path) -> None:
    """Separate source costs from paired target component allowances."""
    source_ids = {
        plan["jobs"][key]["source_fit"]
        for key in members
        if "source_fit" in plan["jobs"][key]
    }
    sources, targets = [], []
    for key in sorted(source_ids):
        directory = fit_directory(plan, key)
        if completed(directory) is None:
            continue
        for stage, record in json.loads(
            (directory / "components.json").read_text()
        ).items():
            training = record["training"]
            sources.append(
                dict(
                    fit_id=key,
                    component=stage,
                    component_id=record["component_id"],
                    allowed_updates=training["settings"]["maximum_updates"],
                    updates=training["updates"],
                    window_exposures=sum(training["window_exposures"]),
                    elapsed_seconds=training["elapsed_seconds"],
                )
            )
    dependencies = {plan["jobs"][key]["preprocessing"] for key in members}
    for key in sorted(set(members) | dependencies):
        directory = fit_directory(plan, key)
        if completed(directory) is None:
            continue
        budget = json.loads((directory / "budget.json").read_text())
        for stage, values in budget["components"].items():
            targets.append(
                dict(
                    fit_id=key,
                    kind=plan["jobs"][key]["kind"],
                    component=stage,
                    **values,
                )
            )
    if sources:
        write_table(snapshot / "source_costs.csv", sources)
    if targets:
        write_table(snapshot / "target_costs.csv", targets)


def report(plan: dict, plan_path: Path) -> None:
    """Publish nonempty immutable report snapshots for all three analyses."""
    root = Path(plan["settings"]["paths"]["artifact_root"]) / "analysis"
    for name, members in plan["analyses"].items():
        rows, dimensions, identities = collect(plan, members)
        if not rows:
            LOGGER.info("%s: no completed target metrics yet.", name)
            continue
        summary = summarize(rows)
        if not summary:
            LOGGER.warning("%s: no finite metrics to summarize.", name)
            continue
        analysis = root / name / plan_path.parent.name
        snapshot = analysis / "reports" / fingerprint(identities)
        with job(snapshot, dict(members=identities)) as needed:
            if not needed:
                atomic_json(
                    analysis / "latest_report.json",
                    dict(
                        snapshot=str(snapshot.relative_to(analysis)),
                        completed_fits=len(identities),
                        expected_fits=len(members),
                    ),
                )
                continue
            write_table(snapshot / "raw_metrics.csv", rows)
            write_table(snapshot / "per_dimension.csv", dimensions)
            write_table(snapshot / "summary.csv", summary)
            write_table(snapshot / "session_metrics.csv", session_means(rows))
            cost_tables(plan, members, snapshot)
            gains = paired_gains(rows)
            if gains:
                write_table(snapshot / "paired_gain.csv", gains)
                gain_rows = [
                    dict(row, value=row["difference"]) for row in gains
                ]
                write_table(
                    snapshot / "paired_gain_summary.csv", summarize(gain_rows)
                )
            figures(summary, name, snapshot, plan["settings"])
        atomic_json(
            analysis / "latest_report.json",
            dict(
                snapshot=str(snapshot.relative_to(analysis)),
                completed_fits=len(identities),
                expected_fits=len(members),
            ),
        )
