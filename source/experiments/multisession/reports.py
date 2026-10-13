"""Publish independent scaling and two-condition sensitivity reports.

Each analysis owns one rolling reports/live-<id>/ and one immutable final
reports/<fit-set-hash>/ with tables and figures.
raw_metrics.csv contains fit/session/fold/seed, design, metric and value;
per_dimension.csv retains coordinate scores. summary.csv averages repeats
and folds within sessions before computing mean/SEM across sessions.
paired_gain.csv compares each transfer fit with its matched scratch control.
latest_report.json points to the validated live or final report. Figure-only
revisions live under figures/ and latest_figures.json, without copying tables.
"""

from collections import defaultdict
import csv
import json
import logging
from pathlib import Path
import shutil
import uuid

import numpy as np

from experiments.storage.cache import (
    atomic_json,
    file_digest,
    fingerprint,
    writer_lock,
)
from experiments.visualization.multisession import figures
from experiments.multisession.store import completed, fit_directory, job
from experiments.multisession.plan import CAPACITY, SUPERVISION

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


def write_payload(
    plan: dict,
    name: str,
    members: list[str],
    rows: list[dict],
    dimensions: list[dict],
    summary: list[dict],
    directory: Path,
) -> None:
    """Write one report's tables and figures into an empty directory."""
    write_table(directory / "raw_metrics.csv", rows)
    write_table(directory / "per_dimension.csv", dimensions)
    write_table(directory / "summary.csv", summary)
    write_table(directory / "session_metrics.csv", session_means(rows))
    cost_tables(plan, members, directory)
    gains = paired_gains(rows)
    if gains:
        write_table(directory / "paired_gain.csv", gains)
        gain_rows = [dict(row, value=row["difference"]) for row in gains]
        write_table(
            directory / "paired_gain_summary.csv", summarize(gain_rows)
        )
    figures(summary, name, directory, plan["settings"])


def report_path(analysis: Path, record: dict) -> Path:
    """Resolve a report pointer without allowing paths outside its root."""
    relative = Path(record["snapshot"])
    if (
        relative.is_absolute()
        or relative.parts[:1] != ("reports",)
        or ".." in relative.parts
    ):
        raise ValueError(f"Invalid report snapshot path: {relative}")
    return analysis / relative


def ready_checksums(directory: Path, fit_set_hash: str) -> None:
    """Validate a staged live report and record its payload checksums."""
    required = (
        "raw_metrics.csv",
        "per_dimension.csv",
        "summary.csv",
        "session_metrics.csv",
    )
    for name in required:
        path = directory / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing or empty report payload: {path}")
    files = [p for p in directory.rglob("*") if p.is_file()]
    atomic_json(
        directory / "ready.json",
        dict(
            fit_set_hash=fit_set_hash,
            checksums={
                str(p.relative_to(directory)): file_digest(p) for p in files
            },
        ),
    )


def valid_live(directory: Path, fit_set_hash: str) -> bool:
    """Check that the published interim report is complete and unchanged."""
    marker = directory / "ready.json"
    if not marker.is_file():
        return False
    record = json.loads(marker.read_text())
    if record.get("fit_set_hash") != fit_set_hash:
        return False
    checksums = record.get("checksums", {})
    if not checksums:
        return False
    for name, digest in checksums.items():
        path = directory / name
        if not path.is_file() or file_digest(path) != digest:
            return False
    return True


def remove_old_live(reports: Path, keep: Path | None) -> None:
    """Remove only superseded, replaceable interim report directories."""
    for path in reports.glob("live-*"):
        if path == keep:
            continue
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"Unexpected live report path: {path}")
        shutil.rmtree(path)


def publish_live(
    plan: dict,
    name: str,
    members: list[str],
    rows: list[dict],
    dimensions: list[dict],
    summary: list[dict],
    analysis: Path,
    fit_set_hash: str,
    completed_fits: int,
) -> None:
    """Replace an interim report only after its new payload is validated."""
    reports = analysis / "reports"
    stage = reports / f"live-{uuid.uuid4().hex}"
    stage.mkdir()
    try:
        write_payload(
            plan, name, members, rows, dimensions, summary, stage
        )
        ready_checksums(stage, fit_set_hash)
        if not valid_live(stage, fit_set_hash):
            raise ValueError(f"Invalid staged report: {stage}")
        atomic_json(
            analysis / "latest_report.json",
            dict(
                snapshot=str(stage.relative_to(analysis)),
                completed_fits=completed_fits,
                expected_fits=len(members),
                fit_set_hash=fit_set_hash,
                mode="live",
            ),
        )
    except BaseException:
        shutil.rmtree(stage)
        raise
    remove_old_live(reports, stage)


def publish_final(
    plan: dict,
    name: str,
    members: list[str],
    rows: list[dict],
    dimensions: list[dict],
    summary: list[dict],
    analysis: Path,
    identities: list,
    fit_set_hash: str,
) -> None:
    """Publish one checksum-validated immutable final report."""
    snapshot = analysis / "reports" / fit_set_hash
    with job(snapshot, dict(members=identities)) as needed:
        if needed:
            write_payload(
                plan, name, members, rows, dimensions, summary, snapshot
            )
    if completed(snapshot) is None:
        raise ValueError(f"Final report lacks completion: {snapshot}")
    atomic_json(
        analysis / "latest_report.json",
        dict(
            snapshot=str(snapshot.relative_to(analysis)),
            completed_fits=len(identities),
            expected_fits=len(members),
            fit_set_hash=fit_set_hash,
            mode="final",
        ),
    )
    remove_old_live(analysis / "reports", None)


def report(
    plan: dict, plan_path: Path, final: bool = False
) -> None:
    """Publish rolling interim reports or immutable final reports."""
    root = Path(plan["settings"]["paths"]["artifact_root"]) / "analysis"
    for name, members in plan["analyses"].items():
        analysis = root / name / plan_path.parent.name
        reports = analysis / "reports"
        reports.mkdir(parents=True, exist_ok=True)
        with writer_lock(reports / "report.lock"):
            rows, dimensions, identities = collect(plan, members)
            if not rows:
                LOGGER.info("%s: no completed target metrics yet.", name)
                continue
            fit_set_hash = fingerprint(identities)
            pointer = analysis / "latest_report.json"
            if pointer.is_file():
                previous = json.loads(pointer.read_text())
                if previous.get("fit_set_hash") == fit_set_hash:
                    path = report_path(analysis, previous)
                    if previous.get("mode") == "final":
                        if completed(path) is not None:
                            continue
                    elif not final and valid_live(path, fit_set_hash):
                        remove_old_live(reports, path)
                        continue
            summary = summarize(rows)
            if not summary:
                LOGGER.warning("%s: no finite metrics to summarize.", name)
                continue
            if final:
                publish_final(
                    plan, name, members, rows, dimensions, summary,
                    analysis, identities, fit_set_hash,
                )
            else:
                publish_live(
                    plan, name, members, rows, dimensions, summary,
                    analysis, fit_set_hash, len(identities),
                )


def read_summary(path: Path) -> list[dict]:
    """Read final aggregate scores for figure-only publication.

    Parameters
    ----------
    path : Path
        Final summary.csv with design keys, mean, SEM, and session counts.

    Returns
    -------
    list[dict]
        Numeric fields restored for the sensitivity renderer.
    """
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Final summary has no rows: {path.resolve()}")
    for row in rows:
        for name in ("latent", "source_count", "horizon"):
            row[name] = int(row[name])
        for name in ("fraction", "mean"):
            row[name] = float(row[name])
        row["sem"] = float(row["sem"]) if row["sem"] else None
        values = [row["fraction"], row["mean"]]
        if row["sem"] is not None:
            values.append(row["sem"])
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite final summary row: {path.resolve()}")
    return rows


def refresh_sensitivity_figures(plan: dict, plan_path: Path) -> None:
    """Publish immutable figure revisions from completed sensitivity reports.

    The final report and its tables remain untouched. Each sensitivity root
    gains figures/scratch-baselines-<id>/ and a latest_figures.json pointer.
    """
    root = Path(plan["settings"]["paths"]["artifact_root"]) / "analysis"
    renderer = Path(figures.__code__.co_filename)
    for name in (CAPACITY, SUPERVISION):
        analysis = root / name / plan_path.parent.name
        pointer = analysis / "latest_report.json"
        if not pointer.is_file():
            raise ValueError(
                f"Missing final report pointer: {pointer.resolve()}"
            )
        record = json.loads(pointer.read_text())
        snapshot = report_path(analysis, record)
        if record.get("mode") != "final" or completed(snapshot) is None:
            raise ValueError(f"Sensitivity report is not final: {analysis}")
        summary_path = snapshot / "summary.csv"
        identity = dict(
            analysis=name,
            final_report_checksum=file_digest(summary_path),
            renderer_checksum=file_digest(renderer),
            fractions=plan["settings"]["target_fractions"],
            source_counts=plan["settings"]["source_counts"],
            horizons=plan["settings"]["evaluation_horizons"],
        )
        directory = (
            analysis / "figures"
            / f"scratch-baselines-{fingerprint(identity)[:12]}"
        )
        families = (
            ("capacity",)
            if name == CAPACITY
            else ("supervision", "decoder")
        )
        expected = {
            f"{family}_{target}_h{horizon}.png"
            for family in families
            for target in ("behavior", "neural")
            for horizon in identity["horizons"]
        }
        with job(directory, identity) as needed:
            if needed:
                figures(
                    read_summary(summary_path),
                    name,
                    directory,
                    plan["settings"],
                )
                missing = expected - {
                    path.name for path in directory.glob("*.png")
                }
                if missing:
                    raise ValueError(
                        f"Missing sensitivity figures: {sorted(missing)}"
                    )
        atomic_json(
            analysis / "latest_figures.json",
            dict(
                snapshot=str(directory.relative_to(analysis)),
                source_report=record["snapshot"],
                figure_count=len(expected),
            ),
        )
