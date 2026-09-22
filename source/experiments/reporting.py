"""BRAID-specific artifact reports, selected through the reporting plugin.

Inputs are completed metrics.json/predictions.npz artifacts. Outputs include
raw_metrics.csv/json, summary.csv/json, population_summary.csv, and the
configured figure suite. Incomplete folds are reported, never filled in.
"""

import csv
from pathlib import Path

from .analysis import read_manifest, record_plotting
from .evaluation import aggregate, aggregate_folds, collect_results
from .outliers import resolve_outliers
from .plots import PlotRenderingError, plot_suite


def attempt_plot_suite(
    errors: list[str], label: str, *args: object, **kwargs: object,
) -> None:
    """Run one plot suite and retain renderer-local failures for the end."""
    try:
        plot_suite(*args, **kwargs)
    except PlotRenderingError as error:
        errors.append(f"{label}: {error}")


def braid_report(
    root: Path, settings: dict, sample_rate: float,
    attempted: set[str] | None = None, rendered: set[str] | None = None,
    regenerate: bool = False,
) -> None:
    """Publish tables and comparisons whose dependencies are terminal.

    Parameters
    ----------
    root : Path
        Analysis directory with a validated membership manifest.
    settings : dict
        Plot selections and presentation settings.
    sample_rate : float
        Dataset sampling rate in Hz.
    attempted : set of str, optional
        Members attempted in this invocation. None uses saved terminal states.
    rendered : set of str, optional
        Figure names already attempted in this invocation; default None.
        Updated in place to avoid repeated rendering and failure warnings.
    regenerate : bool, optional
        Replace existing comparison figures; default False generates only
        missing figures.
    """
    from .model_summary import write_model_summary

    record_plotting(root, settings)
    write_model_summary(root)
    manifest = read_manifest(root)
    destination = root / "summaries"
    completed = any(
        member["state"] == "complete"
        for member in manifest["members"].values()
    )
    rows = collect_results(root) if completed else []
    outliers = resolve_outliers(settings, rows) if rows else ()
    summaries = aggregate(rows, destination, outliers) if rows else []
    table = {}
    for row in summaries:
        if (
            row["nx"] not in (16, 64)
            or row["horizon"] != 4
            or row["evaluation_set"] != "common"
            or row["metric"] != "cc"
        ):
            continue
        key = (
            row["population_scale"],
            row["nx"],
            row["n1"],
            row.get("n2", 0),
        )
        item = table.setdefault(
            key,
            dict(
                population_scale=key[0], nx=key[1], n1=key[2], n2=key[3],
            ),
        )
        for field in ("mean", "sem"):
            item[f"{row['target']}_cc_4step_{field}"] = row[field]
    if table:
        with (destination / "population_summary.csv").open(
            "w", newline=""
        ) as stream:
            writer = csv.DictWriter(
                stream, fieldnames=list(next(iter(table.values())))
            )
            writer.writeheader()
            writer.writerows(table.values())
    rendering_errors = []
    expected = expected_comparisons(manifest, attempted)
    attempt_plot_suite(
        rendering_errors,
        "aggregate",
        summaries,
        root / "plots",
        settings,
        expected,
        rendered=rendered,
        regenerate=regenerate,
    )
    fold_summaries = aggregate_folds(rows, outliers)
    session_expected = expected_comparisons(
        manifest, attempted, per_session=True
    )
    sessions = sorted({
        member["session"] for member in manifest["members"].values()
    })
    for session in sessions:
        attempt_plot_suite(
            rendering_errors,
            f"session/{session}",
            [row for row in fold_summaries if row["session"] == session],
            root / "plots" / "sessions" / session,
            settings,
            [row for row in session_expected if row["session"] == session],
            rendered=rendered,
            namespace=f"session/{session}",
            regenerate=regenerate,
        )
    if rendering_errors:
        raise PlotRenderingError(
            "Comparison rendering failed after all suites were attempted: "
            + "; ".join(rendering_errors)
        )


def expected_comparisons(
    manifest: dict, attempted: set[str] | None = None,
    per_session: bool = False,
) -> list[dict]:
    """Expand comparison membership with pending and failed contributions.

    Parameters
    ----------
    manifest : dict
        Analysis specification and per-session/fold/model member records.
    attempted : set of str, optional
        Current invocation's finished attempts; None uses saved states.
    per_session : bool, optional
        Keep each session's fold dependencies separate; default False.

    Returns
    -------
    list of dict
        One record per aggregate point, with pending, expected/completed
        counts and failed member keys in addition to metric selection fields.
    """
    horizons = manifest["specification"]["settings"]["evaluation"]["horizons"]
    expected = {}
    for key, member in manifest["members"].items():
        case = member["case"]
        state = member["state"]
        pending = state not in ("complete", "failed") or (
            attempted is not None and key not in attempted
        )
        scoring = ["full", "common"] if member["common_ids"] else ["full"]
        for horizon in horizons:
            for evaluation_set in scoring:
                for target in ("neural", "behavior"):
                    for metric in ("cc", "r2", "mse"):
                        candidate = dict(case.get("summary_parameters", {}))
                        candidate.update(
                            configuration=case["name"],
                            nx=case["dimensions"]["nx"],
                            population_scale=case["population_scale"],
                            horizon=horizon,
                            evaluation_set=evaluation_set,
                            target=target,
                            metric=metric,
                        )
                        if per_session:
                            candidate["session"] = member["session"]
                        group = tuple(candidate.values())
                        point = expected.setdefault(
                            group, dict(
                                candidate,
                                pending=False,
                                expected_count=0,
                                completed_count=0,
                                completed_members=[],
                                failed_members=[],
                            ),
                        )
                        point["pending"] |= pending
                        point["expected_count"] += 1
                        point["completed_count"] += int(state == "complete")
                        if state == "complete":
                            point["completed_members"].append(key)
                        if state == "failed" and not pending:
                            point["failed_members"].append(key)
    return list(expected.values())
