"""Report BRAID structure sweep metrics and comparison figures.

Inputs are validated analysis manifest members and their metrics.json rows.
summaries/structure_raw.csv contains case, session, fold, horizon, target,
metric, status and value. summaries/structure_summary.csv contains the same
case/horizon/target/metric keys with cross-session mean, SEM and member counts.
plots contains aggregate figures and per-session figures under
plots/sessions/<session>.
"""

import csv
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from .analysis import read_manifest
from .cache import writer_lock
from .evaluation import collect_results
from .reporting import braid_report
from .structure_plots import (
    render_dynamic_comparison_figures,
    render_session_structure_figures,
    render_structure_figures,
)

METRICS = ("cc", "r2", "mse")
TARGETS = ("behavior", "neural")
RAW_FIELDS = (
    "configuration", "dynamics", "encoder", "decoder", "session",
    "fold", "horizon", "target", "metric", "status", "value",
)
SUMMARY_FIELDS = (
    "configuration", "dynamics", "encoder", "decoder", "horizon",
    "target", "metric", "mean", "sem", "contributing_sessions",
    "completed_members", "failed_members", "pending_members",
)


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    """Atomically publish one table with a fixed column schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".csv.tmp")
    try:
        with os.fdopen(fd, "w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _completed_lookup(rows: list[dict]) -> dict:
    """Index validated full-population metrics by case and held-out split."""
    lookup = {}
    for row in rows:
        if row["evaluation_set"] != "full":
            continue
        key = (
            row["configuration"], row["session"], row["fold"],
            row["horizon"],
        )
        if key in lookup:
            raise ValueError(f"Duplicate structure metric row for {key}.")
        lookup[key] = row
    return lookup


def _raw_rows(manifest: dict, completed: dict) -> list[dict]:
    """Expand all expected members, including failed and pending ones."""
    horizons = manifest["specification"]["settings"]["evaluation"]["horizons"]
    rows = []
    for member in manifest["members"].values():
        case = member["case"]
        structure = case["structure"]
        for horizon in horizons:
            key = (
                case["name"], member["session"], member["fold"], horizon,
            )
            metric_row = completed.get(key)
            if member["state"] == "complete" and metric_row is None:
                raise ValueError(
                    f"Completed structure member lacks horizon {key}."
                )
            for target in TARGETS:
                for metric in METRICS:
                    value = (
                        metric_row[target][f"mean_{metric}"]
                        if metric_row is not None else None
                    )
                    rows.append(dict(
                        configuration=case["name"],
                        dynamics=structure["dynamics"],
                        encoder=structure["encoder"],
                        decoder=structure["decoder"],
                        session=member["session"], fold=member["fold"],
                        horizon=horizon, target=target, metric=metric,
                        status=member["state"], value=value,
                    ))
    return rows


def _summary_rows(raw: list[dict], aggregates: list[dict]) -> list[dict]:
    """Join existing fold-then-session aggregates with member status."""
    lookup = {
        (
            row["configuration"], row["horizon"], row["target"],
            row["metric"],
        ): row
        for row in aggregates if row["evaluation_set"] == "full"
    }
    groups = {}
    for row in raw:
        key = (
            row["configuration"], row["horizon"], row["target"],
            row["metric"],
        )
        groups.setdefault(key, []).append(row)
    output = []
    for key, members in sorted(groups.items()):
        aggregate = lookup.get(key, {})
        first = members[0]
        output.append(dict(
            configuration=key[0], dynamics=first["dynamics"],
            encoder=first["encoder"], decoder=first["decoder"],
            horizon=key[1], target=key[2], metric=key[3],
            mean=aggregate.get("mean"), sem=aggregate.get("sem"),
            contributing_sessions=aggregate.get(
                "contributing_sessions", 0
            ),
            completed_members=sum(
                row["status"] == "complete" for row in members
            ),
            failed_members=sum(
                row["status"] == "failed" for row in members
            ),
            pending_members=sum(
                row["status"] not in ("complete", "failed")
                for row in members
            ),
        ))
    return output


def _session_summary_rows(raw: list[dict]) -> list[dict]:
    """Average completed folds and calculate sample SD within each session.

    Parameters
    ----------
    raw : list of dict
        Per-fold structure rows, including missing and failed rows.

    Returns
    -------
    list of dict
        One row per session/case/horizon/target/metric with ``mean`` and
        fold-level ``std``. Missing folds remain represented in the counts.
    """
    groups = {}
    for row in raw:
        key = (
            row["session"], row["configuration"], row["dynamics"],
            row["encoder"], row["decoder"], row["horizon"],
            row["target"], row["metric"],
        )
        groups.setdefault(key, []).append(row)
    summaries = []
    for key, members in sorted(groups.items()):
        values = np.asarray([
            row["value"] for row in members
            if row["value"] is not None
        ], dtype=float)
        if values.size and not np.isfinite(values).all():
            raise ValueError(
                f"Nonfinite structure fold metrics for {key[:6]}."
            )
        summaries.append(dict(
            session=key[0], configuration=key[1], dynamics=key[2],
            encoder=key[3], decoder=key[4], horizon=key[5],
            target=key[6], metric=key[7],
            mean=float(values.mean()) if values.size else None,
            std=(float(values.std(ddof=1)) if values.size > 1 else None),
            completed_folds=int(values.size),
            failed_folds=sum(row["status"] == "failed" for row in members),
            pending_folds=sum(
                row["status"] not in ("complete", "failed")
                for row in members
            ),
        ))
    return summaries


def structure_report(
    root: Path, settings: dict, sample_rate: float,
    attempted: set[str] | None = None, rendered: set[str] | None = None,
    regenerate: bool = False,
) -> None:
    """Write all sweep tables and render ready horizon comparisons.

    Parameters
    ----------
    root : Path
        Analysis directory containing immutable member metrics.
    settings : dict
        Plot style and structure horizon options.
    sample_rate : float
        Aligned sample rate in Hz.
    attempted : set of str, optional
        Members attempted in this invocation; default None uses saved state.
    rendered : set of str, optional
        Invocation-local figure names already drawn; default None.
    regenerate : bool, optional
        Existing reporting regeneration switch; default False.
    """
    braid_report(
        root, settings, sample_rate, attempted, rendered, regenerate
    )
    manifest = read_manifest(root)
    has_complete = any(
        member["state"] == "complete"
        for member in manifest["members"].values()
    )
    completed = (
        _completed_lookup(collect_results(root)) if has_complete else {}
    )
    raw = _raw_rows(manifest, completed)
    aggregates = (
        json.loads((root / "summaries" / "summary.json").read_text())
        if has_complete else []
    )
    summaries = _summary_rows(raw, aggregates)
    session_summaries = _session_summary_rows(raw)
    with writer_lock(root / "summaries" / "structure.lock"):
        _write_csv(root / "summaries" / "structure_raw.csv", RAW_FIELDS, raw)
        _write_csv(
            root / "summaries" / "structure_summary.csv",
            SUMMARY_FIELDS, summaries,
        )
    ready = (
        set(manifest["members"]) <= attempted
        if attempted is not None else all(
            member["state"] in ("complete", "failed")
            for member in manifest["members"].values()
        )
    )
    if ready and settings["structure_horizons"]["enabled"]:
        render_structure_figures(
            root / "plots", summaries, settings, sample_rate,
            rendered=rendered,
        )
        render_dynamic_comparison_figures(
            root / "plots", summaries, settings, sample_rate,
            rendered=rendered,
        )
        render_session_structure_figures(
            root / "plots" / "sessions", session_summaries, settings,
            sample_rate, rendered=rendered,
        )
