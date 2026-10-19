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
from math import comb
from itertools import combinations
import csv
import json
import logging
import os
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
from experiments.multisession.plan import (
    CAPACITY, SUPERVISION, analysis_directory,
)
from experiments.multisession.sensitivity import (
    CAMPAIGN, DIRECTION, FILTER, RESIDUAL, FILTER_PROFILES,
    RESIDUAL_PROFILES, SOURCE_COUNT,
)

LOGGER = logging.getLogger(__name__)
RESIDUAL_STAGE_COUNT = 2
RESIDUAL_REPORT_PROFILES = ("scratch", *RESIDUAL_PROFILES)
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


def dummy_mse_by_coordinate(
    plan: dict, spec: dict, fit: Path, cache: dict
) -> dict:
    """Evaluate an independent training-label draw on fixed test truth."""
    key = (
        spec["session"], spec["fold"], spec["fraction"],
        spec["preprocessing"],
    )
    if key in cache:
        return cache[key]
    from experiments.multisession.runner import dataset

    if "_adapter" not in cache:
        cache["_adapter"] = dataset(plan["settings"])
    adapter = cache["_adapter"]
    features = adapter.fold(
        adapter.load(spec["session"]), spec["fold"], True
    )
    selection = fit_directory(plan, spec["preprocessing"])
    with np.load(selection / "selection.npz") as chosen:
        selected = chosen["training"].ravel()
    indices = features.arrays["indices"]
    positions = np.searchsorted(indices, selected)
    if not len(selected) or np.any(positions >= len(indices)):
        raise ValueError(
            f"Selected training indices exceed {spec['session']}."
        )
    if not np.array_equal(indices[positions], selected):
        raise ValueError(
            f"Selected training indices do not match {spec['session']}."
        )
    baselines = {}
    with np.load(fit / "predictions.npz") as predictions:
        horizons = predictions["horizons"].tolist()
        valid = predictions["valid"]
        for name, target in (("Y", "neural"), ("Z", "behavior")):
            train = features.arrays[name][positions]
            mean = np.mean(train, axis=0)
            variance = np.mean((train - mean) ** 2, axis=0)
            truth = predictions[f"true_{name}"]
            for index, horizon in enumerate(horizons):
                if not np.any(valid[index]):
                    raise ValueError(
                        f"No valid test samples at horizon {horizon}."
                    )
                error = np.mean(
                    (truth[valid[index]] - mean) ** 2, axis=0
                ) + variance
                if not np.isfinite(error).all():
                    raise ValueError(
                        f"Nonfinite dummy MSE for {spec['session']}."
                    )
                baselines[target, horizon] = error
                centered = truth[valid[index]] - np.mean(
                    truth[valid[index]], axis=0
                )
                test_variance = np.mean(centered ** 2, axis=0)
                scored = (
                    np.isfinite(error)
                    & np.isfinite(test_variance)
                    & (test_variance > 0)
                )
                if not np.any(scored):
                    raise ValueError(
                        f"No valid dummy R² coordinates: "
                        f"{spec['session']}."
                    )
                dummy_r2 = 1 - error[scored] / test_variance[scored]
                if not np.isfinite(dummy_r2).all():
                    raise ValueError(
                        f"Nonfinite dummy R² for {spec['session']}."
                    )
                baselines[target, horizon, "r2"] = float(
                    np.mean(dummy_r2)
                )
    cache[key] = baselines
    return baselines


def paired_comparisons(
    rows: list[dict], profiles: tuple[str, ...]
) -> list[dict]:
    """Exact paired sign tests with Holm correction across three metrics."""
    metrics = ("cc", "r2", "mse_dummy_improvement")
    grouped = defaultdict(list)
    for row in rows:
        if row["metric"] in metrics and row["value"] is not None:
            key = (
                row["sharing"], row["target"], row["fraction"],
                row["horizon"], row["metric"], row["session"],
            )
            grouped[key].append(row["value"])
    means = {
        key: float(np.mean(values))
        for key, values in grouped.items()
    }
    cells = defaultdict(dict)
    for left, right in combinations(profiles, 2):
        for target in ("neural", "behavior"):
            for fraction in sorted({row["fraction"] for row in rows}):
                for horizon in sorted({row["horizon"] for row in rows}):
                    cell = left, right, target, fraction, horizon
                    for metric in metrics:
                        sessions = sorted({
                            key[-1] for key in means
                            if key[:5] == (
                                left, target, fraction, horizon, metric
                            )
                        } & {
                            key[-1] for key in means
                            if key[:5] == (
                                right, target, fraction, horizon, metric
                            )
                        })
                        if not sessions:
                            continue
                        differences = np.asarray([
                            means[
                                right, target, fraction, horizon,
                                metric, session,
                            ] - means[
                                left, target, fraction, horizon,
                                metric, session,
                            ]
                            for session in sessions
                        ])
                        if not np.isfinite(differences).all():
                            raise ValueError("Nonfinite paired difference.")
                        nonzero = differences[differences != 0]
                        n = len(nonzero)
                        tail = min(
                            int(np.sum(nonzero > 0)),
                            int(np.sum(nonzero < 0)),
                        )
                        raw_p = min(
                            1.0,
                            2.0 * sum(comb(n, k) for k in range(tail + 1))
                            / (2 ** n),
                        )
                        cells[cell][metric] = dict(
                            profile_a=left,
                            profile_b=right,
                            target=target,
                            fraction=fraction,
                            horizon=horizon,
                            metric=metric,
                            sessions=len(sessions),
                            nonzero_sessions=n,
                            mean_difference=float(differences.mean()),
                            median_difference=float(np.median(differences)),
                            raw_p=raw_p,
                        )
    result = []
    for family in cells.values():
        if set(family) != set(metrics):
            continue
        ordered = sorted(
            family.values(), key=lambda row: row["raw_p"]
        )
        adjusted = 0.0
        for index, row in enumerate(ordered):
            adjusted = max(
                adjusted,
                min(1.0, (len(metrics) - index) * row["raw_p"]),
            )
            result.append(dict(row, adjusted_p=adjusted))
    return result


def collect(
    plan: dict, members: list[str], baseline_cache: dict | None = None
) -> tuple[list, list, list]:
    """Read metrics only from checksum-validated completed member fits."""
    rows, dimensions, identities = [], [], []
    if baseline_cache is None:
        baseline_cache = {}
    from experiments.multisession.scaling import CAMPAIGN as SCALING
    from experiments.multisession.capacity import CAMPAIGN as CAPACITY

    normalize_mse = (
        plan["name"] in (CAMPAIGN, SCALING, CAPACITY)
        and "loco_targets" in plan
    )
    for key in members:
        directory = fit_directory(plan, key)
        record = completed(directory)
        if record is None:
            continue
        path = directory / "metrics.json"
        if not path.exists():
            raise ValueError(f"Completed evaluation lacks metrics: {directory}")
        spec = plan["jobs"][key]
        identity = [key, record["checksums"]["metrics.json"]]
        if normalize_mse:
            prep = completed(
                fit_directory(plan, spec["preprocessing"])
            )
            identity.extend((
                record["checksums"]["predictions.npz"],
                prep["checksums"]["selection.npz"],
            ))
            dummy = dummy_mse_by_coordinate(
                plan, spec, directory, baseline_cache
            )
        identities.append(tuple(identity))
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
            if normalize_mse:
                error = dummy[item["target"], item["horizon"]]
                model = np.asarray(
                    item["per_dimension_mse"], dtype=float
                )
                if len(error) != len(model):
                    raise ValueError(
                        f"Dummy/model coordinate count differs: {directory}"
                    )
                valid = np.isfinite(error) & (error > 0)
                if not np.any(valid):
                    raise ValueError(
                        f"No valid dummy MSE coordinates: {directory}"
                    )
                if not np.isfinite(model[valid]).all():
                    raise ValueError(
                        f"Nonfinite model MSE coordinates: {directory}"
                    )
                if not np.all(valid):
                    LOGGER.warning(
                        "%s: excluded %d invalid dummy coordinates.",
                        directory, int(np.sum(~valid)),
                    )
                improvement = 100 * (1 - model[valid] / error[valid])
                common = dict(
                    base,
                    fit_id=key,
                    target=item["target"],
                    horizon=item["horizon"],
                    metric="mse_dummy_improvement",
                )
                rows.append(dict(
                    common,
                    value=float(np.mean(improvement)),
                    valid_dimensions=int(np.sum(valid)),
                    scored_samples=item["samples"],
                ))
                for coordinate, value in zip(
                    np.asarray(item["coordinates"])[valid],
                    improvement,
                ):
                    dimensions.append(dict(
                        common, coordinate=coordinate,
                        value=float(value),
                    ))
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
    dummy_cache: dict | None = None,
    pending_128: bool = False,
) -> None:
    """Write one report's tables and figures into an empty directory."""
    kind = plan.get("analysis_kinds", {}).get(name, name)
    from experiments.multisession.scaling import CAMPAIGN as SCALING

    from experiments.multisession.capacity import CAMPAIGN as CAPACITY

    if kind == CAPACITY:
        from experiments.multisession.capacity_report import (
            write_capacity_payload,
        )

        write_capacity_payload(
            plan, name, members, rows, dimensions, summary,
            directory, dummy_cache, pending_128,
        )
        return
    if kind == SCALING:
        from experiments.multisession.scaling_report import (
            write_scaling_payload,
        )

        write_scaling_payload(
            plan, name, members, rows, dimensions, summary, directory
        )
        return
    if kind in (FILTER, RESIDUAL):
        write_sharing_payload(
            plan, kind, members, rows, dimensions, summary, directory
        )
        return
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


def write_sharing_payload(
    plan: dict,
    name: str,
    members: list[str],
    rows: list[dict],
    dimensions: list[dict],
    summary: list[dict],
    directory: Path,
) -> None:
    """Write nested tables and figures for the two new analysis roots."""
    from experiments.visualization.sharing_sensitivity import (
        source_figures, target_figures,
    )
    from experiments.multisession.source_validation import (
        collect_source_validation,
    )

    tables = directory / "tables"
    figures_root = directory / "figures"
    tables.mkdir()
    figures_root.mkdir()
    conditions = (
        FILTER_PROFILES if name == FILTER else RESIDUAL_REPORT_PROFILES
    )
    ceiling = plan["settings"]["source_training"]["maximum_updates"]
    source_update_ceiling = {
        profile: (
            [ceiling, ceiling // RESIDUAL_STAGE_COUNT]
            if profile in RESIDUAL_PROFILES[1:] else [ceiling]
        )
        for profile in conditions if profile != "scratch"
    }
    atomic_json(
        tables / "design.json",
        dict(
            conditions=conditions,
            source_count=SOURCE_COUNT,
            source_update_ceiling=source_update_ceiling,
            interpretation=(
                "Stage-1 sharing with matched scratch controls."
                if name == FILTER
                else (
                    "Stage 1 fixed at A; stage-2 sharing varies "
                    "with matched scratch controls."
                )
            ),
        ),
    )
    write_table(tables / "raw_metrics.csv", rows)
    write_table(tables / "per_dimension.csv", dimensions)
    write_table(tables / "summary.csv", summary)
    write_table(tables / "session_metrics.csv", session_means(rows))
    cost_tables(plan, members, tables)
    gains = paired_gains(rows)
    if gains:
        write_table(tables / "paired_gain.csv", gains)
        gain_rows = [dict(row, value=row["difference"]) for row in gains]
        write_table(
            tables / "paired_gain_summary.csv", summarize(gain_rows)
        )
    comparisons = paired_comparisons(rows, conditions)
    if comparisons:
        write_table(
            tables / "paired_comparisons.csv", comparisons
        )
    target_figures(summary, name, figures_root, plan["settings"])
    if name == FILTER:
        source_rows, source_losses = collect_source_validation(plan)
        if source_rows:
            write_table(
                tables / "source_validation_metrics.csv", source_rows
            )
            write_table(
                tables / "source_validation_loss.csv", source_losses
            )
            source_figures(
                source_rows, source_losses,
                figures_root / "source_validation",
            )


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
    table_root = directory / "tables"
    if not table_root.is_dir():
        table_root = directory
    for name in required:
        path = table_root / name
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


def publish_current_paths(analysis: Path, snapshot: Path) -> None:
    """Point stable capacity figure and table paths at a validated report.

    Parameters
    ----------
    analysis : Path
        Direction-specific analysis root.
    snapshot : Path
        Validated live or immutable final report directory.
    """
    for name in ("figures", "tables"):
        target = analysis / name
        if target.exists() and not target.is_symlink():
            raise ValueError(
                f"Cannot replace non-symlink report path: {target}"
            )
        link = analysis / f".{name}-{uuid.uuid4().hex}"
        destination = snapshot.relative_to(analysis) / name
        try:
            os.symlink(destination, link)
            os.replace(link, target)
        finally:
            if link.is_symlink():
                link.unlink()


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
    dummy_cache: dict | None = None,
    pending_128: bool = False,
    expected_fits: int | None = None,
) -> None:
    """Replace an interim report only after its new payload is validated."""
    reports = analysis / "reports"
    stage = reports / f"live-{uuid.uuid4().hex}"
    stage.mkdir()
    try:
        write_payload(
            plan, name, members, rows, dimensions, summary, stage,
            dummy_cache, pending_128,
        )
        ready_checksums(stage, fit_set_hash)
        if not valid_live(stage, fit_set_hash):
            raise ValueError(f"Invalid staged report: {stage}")
        from experiments.multisession.capacity import (
            CAMPAIGN as CAPACITY_SCALING,
        )
        atomic_json(
            analysis / "latest_report.json",
            dict(
                snapshot=str(stage.relative_to(analysis)),
                completed_fits=completed_fits,
                expected_fits=(
                    len(members) if expected_fits is None
                    else expected_fits
                ),
                phase_expected_fits=(
                    len(members) if pending_128 else None
                ),
                fit_set_hash=fit_set_hash,
                mode="live",
            ),
        )
        if plan["name"] == CAPACITY_SCALING:
            publish_current_paths(analysis, stage)
    except BaseException:
        pointer = analysis / "latest_report.json"
        current = (
            json.loads(pointer.read_text())
            if pointer.is_file() else {}
        )
        if current.get("snapshot") != str(stage.relative_to(analysis)):
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
    dummy_cache: dict | None = None,
) -> None:
    """Publish one checksum-validated immutable final report."""
    snapshot = analysis / "reports" / fit_set_hash
    with job(snapshot, dict(members=identities)) as needed:
        if needed:
            write_payload(
                plan, name, members, rows, dimensions, summary,
                snapshot, dummy_cache,
            )
    if completed(snapshot) is None:
        raise ValueError(f"Final report lacks completion: {snapshot}")
    from experiments.multisession.capacity import (
        CAMPAIGN as CAPACITY_SCALING,
    )
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
    if plan["name"] == CAPACITY_SCALING:
        publish_current_paths(analysis, snapshot)
    remove_old_live(analysis / "reports", None)


def report_members(
    plan: dict, name: str, members: list[str]
) -> list[str]:
    """Include the shared scratch controls in residual report payloads.

    Parameters
    ----------
    plan : dict
        Combined sharing plan with filter and residual memberships.
    name : str
        Analysis name, including transfer direction when applicable.
    members : list[str]
        Residual adaptation fit identities.

    Returns
    -------
    list[str]
        Adaptation identities followed by matching scratch identities.
    """
    kind = plan.get("analysis_kinds", {}).get(name, name)
    if kind != RESIDUAL:
        return members
    filter_name = name.replace(RESIDUAL, FILTER, 1)
    filter_members = plan["analyses"].get(filter_name)
    if filter_members is None:
        raise ValueError(f"Missing filter controls for {name}.")
    controls = [
        key for key in filter_members
        if plan["jobs"][key]["kind"] == "scratch"
    ]
    fields = ("session", "fold", "fraction", "seed", "latent")
    control_targets = {
        tuple(plan["jobs"][key][field] for field in fields)
        for key in controls
    }
    residual_targets = {
        tuple(plan["jobs"][key][field] for field in fields)
        for key in members
    }
    if (
        len(controls) != len(control_targets)
        or control_targets != residual_targets
    ):
        raise ValueError(
            f"Residual targets lack one-to-one scratch controls: {name}."
        )
    return [*members, *controls]


def scratch_controls_ready(
    plan: dict, name: str, members: list[str], analysis: Path
) -> bool:
    """Hold Loco reports until every matched scratch fit is complete.

    Parameters
    ----------
    plan : dict
        Combined sharing plan and fit locations.
    name : str
        Analysis name including its target direction.
    members : list[str]
        Report fit identities including matched scratch controls.
    analysis : Path
        Analysis directory containing the current report pointer.

    Returns
    -------
    bool
        Whether all scratch controls can appear in tables and figures.
    """
    from experiments.multisession.scaling import CAMPAIGN as SCALING

    from experiments.multisession.capacity import CAMPAIGN as CAPACITY

    scaling = plan["name"] in (SCALING, CAPACITY)
    if not scaling and (
        plan["name"] != CAMPAIGN
        or not name.endswith(f"_{DIRECTION}")
    ):
        return True
    controls = [
        key for key in members
        if plan["jobs"][key]["kind"] == "scratch"
    ]
    expected = (
        len(plan["held_out_sessions"])
        if scaling and name.startswith("Indy → Indy")
        else len(plan["loco_targets"])
    )
    expected *= len(plan["settings"]["target_folds"])
    expected *= len(plan["settings"]["target_fractions"])
    if plan["name"] == CAPACITY:
        expected *= len({
            plan["jobs"][key]["latent"] for key in members
        })
    if len(controls) != expected:
        raise ValueError(
            f"Expected {expected} Loco scratch controls for {name}; "
            f"found {len(controls)}."
        )
    remaining = sum(
        completed(fit_directory(plan, key)) is None
        for key in controls
    )
    if not remaining:
        return True
    pointer = analysis / "latest_report.json"
    if pointer.is_file():
        record = json.loads(pointer.read_text())
        if record.get("mode") == "live":
            pointer.unlink()
    LOGGER.info(
        "%s: awaiting %d scratch fits before publishing figures.",
        name, remaining,
    )
    return False


def capacity_report_members(
    plan: dict, members: list[str]
) -> tuple[list[str], bool] | None:
    """Select a complete 32/64 phase or complete full capacity report."""
    from experiments.multisession.capacity import (
        COUNTS, LATENTS, PHASE_WIDTHS, TARGETS_PER_DIRECTION,
    )

    full_expected = TARGETS_PER_DIRECTION * (len(COUNTS) + 1)
    full_expected *= len(LATENTS)
    if len(members) != full_expected:
        raise ValueError(
            f"Expected {full_expected} capacity members; "
            f"found {len(members)}."
        )
    phase = [
        key for key in members
        if plan["jobs"][key]["latent"] in PHASE_WIDTHS
    ]
    phase_expected = TARGETS_PER_DIRECTION * (len(COUNTS) + 1)
    phase_expected *= len(PHASE_WIDTHS)
    if len(phase) != phase_expected:
        raise ValueError(
            f"Expected {phase_expected} 32/64 capacity members; "
            f"found {len(phase)}."
        )
    fields = ("session", "fold", "seed", "fraction", "latent")
    controls = set()
    adaptations = defaultdict(set)
    for key in members:
        spec = plan["jobs"][key]
        target = tuple(spec[field] for field in fields)
        if spec["kind"] == "scratch":
            if target in controls:
                raise ValueError(f"Duplicate capacity control: {target}.")
            controls.add(target)
        elif spec["kind"] == "adaptation":
            adaptations[target].add(spec["source_count"])
        else:
            raise ValueError(
                f"Unexpected capacity report kind: {spec['kind']}."
            )
    if set(adaptations) != controls or any(
        counts != set(COUNTS) for counts in adaptations.values()
    ):
        raise ValueError(
            "Capacity adaptations lack matched scratch controls."
        )
    if all(
        completed(fit_directory(plan, key)) is not None
        for key in members
    ):
        return members, False
    if all(
        completed(fit_directory(plan, key)) is not None
        for key in phase
    ):
        return phase, True
    return None


def report(
    plan: dict, plan_path: Path, final: bool = False
) -> None:
    """Publish rolling interim reports or immutable final reports."""
    for name, members in plan["analyses"].items():
        members = report_members(plan, name, members)
        expected_fits = len(members)
        pending_128 = False
        kind = plan.get("analysis_kinds", {}).get(name, name)
        from experiments.multisession.capacity import (
            CAMPAIGN as CAPACITY_SCALING,
        )
        if kind == CAPACITY_SCALING:
            selected = capacity_report_members(plan, members)
            if selected is None:
                LOGGER.info(
                    "%s: awaiting complete 32/64-width membership.",
                    name,
                )
                continue
            members, pending_128 = selected
        analysis = analysis_directory(
            plan, name, plan_path.parent.name
        )
        reports = analysis / "reports"
        reports.mkdir(parents=True, exist_ok=True)
        with writer_lock(reports / "report.lock"):
            if not scratch_controls_ready(
                plan, name, members, analysis
            ):
                continue
            dummy_cache = {}
            rows, dimensions, identities = collect(
                plan, members, dummy_cache
            )
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
                            if kind == CAPACITY_SCALING:
                                publish_current_paths(analysis, path)
                            continue
                    elif not final and valid_live(path, fit_set_hash):
                        if kind == CAPACITY_SCALING:
                            publish_current_paths(analysis, path)
                        remove_old_live(reports, path)
                        continue
            summary = summarize(rows)
            if not summary:
                LOGGER.warning("%s: no finite metrics to summarize.", name)
                continue
            if (final or kind == CAPACITY_SCALING) and not pending_128:
                publish_final(
                    plan, name, members, rows, dimensions, summary,
                    analysis, identities, fit_set_hash, dummy_cache,
                )
            else:
                publish_live(
                    plan, name, members, rows, dimensions, summary,
                    analysis, fit_set_hash, len(identities),
                    dummy_cache, pending_128, expected_fits,
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
