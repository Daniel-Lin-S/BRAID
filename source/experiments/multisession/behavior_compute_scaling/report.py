"""Publish immutable source-compute and held-out transfer reports.

Inputs are completed source and target fits from the campaign plan. Outputs
under analysis/<campaign>/<plan>/reports/<content-id> contain source_loss.csv,
source_session_loss.csv, transfer_metrics.csv, paired_gain.csv, and PNG figures.
latest_report.json points to a checksum-valid immutable snapshot. Source test
curves use the chronological holdout also used for transfer selection.
"""

import json
from pathlib import Path
from textwrap import fill

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from experiments.multisession.behavior_compute_scaling.ancestry import (
    inherited_paths,
)
from experiments.multisession.behavior_compute_scaling.plan import (
    CAMPAIGN, COMPONENTS, COUNTS, RECURRENT_STAGES,
)
from experiments.multisession.plan import analysis_directory
from experiments.multisession.reports import (
    collect, paired_gains, summarize, write_table,
)
from experiments.multisession.store import completed, fit_directory, job
from experiments.storage.cache import atomic_json, fingerprint
from experiments.multisession.scaling_report import COLORS
from experiments.visualization.multisession import (
    ANNOTATION_FONT_SIZE, DEFAULT_LINE_WIDTH, READABLE_TEXT_STYLE, plot_line,
)

EXPECTED_COMMON_WINDOWS = 87
SOURCE_FIGURE_SIZE = (8, 5)
TITLE_WIDTH = 42
NOTE_WIDTH = 78
RENDERING_STYLE = dict(
    text=READABLE_TEXT_STYLE, annotation_font=ANNOTATION_FONT_SIZE,
    title_width=TITLE_WIDTH, note_width=NOTE_WIDTH,
)
TARGET_COLOR = "#2878b5"
SCRATCH_COLOR = "#222222"
SELECTED_MARKER = "s"
STAGE_LABELS = {
    "behavior_relevant": "RNN1", "residual": "RNN2",
}
TARGET_TYPES = ("neural", "behavior")
TARGET_METRICS = ("cc", "r2", "mse")
INTERIM_COUNTS = (4, 8, 16)
INTERIM_NOTE = "Interim: 33-session fits pending."
LINE_STYLES = {4: "-", 8: "--", 16: "-.", 33: ":"}


def source_tables(
    plan: dict, counts: tuple[int, ...] = COUNTS,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Collect checkpoint, session-loss and selection provenance."""
    losses, sessions, choices = [], [], []
    expected_set = None
    for key in plan["source_references"]:
        spec = plan["jobs"][key]
        if spec["source_count"] not in counts:
            continue
        directory = fit_directory(plan, key)
        completion = completed(directory)
        if completion is None:
            continue
        selection = json.loads(
            (directory / "selection.json").read_text()
        )
        identity = selection["evaluation_set_id"]
        if expected_set is None:
            expected_set = identity
        elif identity != expected_set:
            raise ValueError("Source test window sets differ by budget.")
        inventory = json.loads((directory / "data.json").read_text())
        windows = {
            name: inventory[name]["validation"]
            for name in selection["evaluation_sessions"]
        }
        if fingerprint(windows) != identity:
            raise ValueError("Source test window identity changed.")
        if {
            name: len(values) for name, values in windows.items()
        } != selection["evaluation_window_counts"]:
            raise ValueError("Source test window counts changed.")
        if sum(len(value) for value in windows.values()) != (
            EXPECTED_COMMON_WINDOWS
        ):
            raise ValueError(
                "Expected 87 fixed chronological source test windows."
            )
        choices.append(dict(
            source_fit=key, source_count=spec["source_count"],
            selected_budget=selection["selected_budget"],
            selection_score=selection["selection_score"],
            selected_full_path_flops=selection[
                "selected_full_path_flops"
            ],
            evaluation_set_id=identity,
            source_completion_checksum=fingerprint(
                completion["checksums"]
            ),
            interpretation=(
                "Selected source test score is descriptive because "
                "the same holdout selected the transfer checkpoint."
            ),
        ))
        for path in inherited_paths(directory):
            total = path["full_path_flops"]
            if not isinstance(total, int) or total <= 0:
                raise ValueError("Invalid full-pipeline FLOP axis.")
            for stage in COMPONENTS:
                record = path["stages"][stage]
                component = (
                    directory.parent.parent / "source_components"
                    / record["component_id"]
                )
                marker = completed(component)
                if marker is None:
                    raise ValueError(
                        f"Incomplete source branch {component.resolve()}."
                    )
                score = record["score"]
                loss = score["mean_loss"]
                if not np.isfinite(loss) or loss <= 0:
                    raise ValueError("Invalid source log-loss axis.")
                common = dict(
                    source_fit=key,
                    source_count=spec["source_count"],
                    budget=path["budget"],
                    branch_updates=path["branch_updates"],
                    stage=stage,
                    full_path_flops=total,
                    component_flops=record["component_flops"],
                    chronological_source_test_loss=loss,
                    evaluation_set_id=identity,
                    component_id=record["component_id"],
                    component_checksum=fingerprint(
                        marker["checksums"]
                    ),
                    transfer_selected=(
                        path["budget"]
                        == selection["selected_budget"]
                    ),
                    test_holdout_selects_transfer=True,
                )
                losses.append(common)
                for item in score["per_session"]:
                    sessions.append(dict(
                        common, session=item["session"],
                        windows=item["windows"],
                        raw_loss=item["loss"],
                        dummy_loss=item["dummy_loss"],
                        ratio=item["ratio"],
                    ))
    return losses, sessions, choices


@plt.rc_context(READABLE_TEXT_STYLE)
def source_figures(
    rows: list[dict], directory: Path,
    counts: tuple[int, ...] = COUNTS,
    stages: tuple[str, ...] = COMPONENTS,
    expected_points: int = 5,
) -> None:
    """Draw complete log-log source panels for requested stages."""
    if not rows:
        raise ValueError("Cannot plot source loss without completed rows.")
    for stage in stages:
        figure, axis = plt.subplots(figsize=SOURCE_FIGURE_SIZE)
        stage_rows = [row for row in rows if row["stage"] == stage]
        for count in counts:
            selected = sorted(
                (
                    row for row in stage_rows
                    if row["source_count"] == count
                ),
                key=lambda row: row["full_path_flops"],
            )
            if len(selected) != expected_points:
                raise ValueError(
                    f"Expected {expected_points} source FLOP points "
                    f"for {count} sessions, got {len(selected)}."
                )
            xs = [row["full_path_flops"] for row in selected]
            ys = [
                row["chronological_source_test_loss"]
                for row in selected
            ]
            if not np.isfinite(xs + ys).all() or min(xs + ys) <= 0:
                raise ValueError(
                    "Source log axes require finite positive values."
                )
            plot_line(
                x=xs, y=ys, ax=axis, linestyle=LINE_STYLES[count],
                color=COLORS[count], marker="o",
                linewidth=DEFAULT_LINE_WIDTH,
                label=f"{count} source sessions",
            )
            chosen = [
                row for row in selected if row["transfer_selected"]
            ]
            if len(chosen) != 1:
                raise ValueError("Expected one selected source checkpoint.")
            axis.scatter(
                chosen[0]["full_path_flops"],
                chosen[0]["chronological_source_test_loss"],
                color=COLORS[count], marker=SELECTED_MARKER,
                s=80, edgecolor="black", zorder=5,
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("Full-pipeline source model FLOPs")
        axis.set_ylabel("Chronological source test loss")
        stage_label = STAGE_LABELS.get(stage, stage.replace("_", " ").title())
        axis.set_title(
            fill(
                f"{stage_label}: fixed four-session "
                "chronological source test"
                + (" (interim)" if counts != COUNTS else ""),
                TITLE_WIDTH,
            ),
            pad=22,
        )
        axis.text(
            0, -0.32,
            fill(
                "The same holdout selects transfer checkpoints; selected "
                "source scores are descriptive.", NOTE_WIDTH,
            ) + (f"\n{INTERIM_NOTE}" if counts != COUNTS else ""),
            transform=axis.transAxes, fontsize=ANNOTATION_FONT_SIZE,
        )
        axis.legend(
            loc="upper left", bbox_to_anchor=(1.01, 1),
            frameon=False,
        )
        figure.savefig(
            directory / f"source_{stage}.png",
            dpi=180, bbox_inches="tight",
        )
        plt.close(figure)


@plt.rc_context(READABLE_TEXT_STYLE)
def target_figures(
    summary: list[dict], gain_summary: list[dict], directory: Path,
    settings: dict, counts: tuple[int, ...] = COUNTS,
) -> None:
    """Plot absolute and paired transfer metrics by source session count."""
    if not summary:
        return
    for target in TARGET_TYPES:
        for metric in TARGET_METRICS:
            for fraction in settings["target_fractions"]:
                figure, axes = plt.subplots(
                    2, 3, figsize=(14, 8), sharex=True,
                )
                for axis, horizon in zip(
                    axes.flat, settings["evaluation_horizons"]
                ):
                    selected = [
                        row for row in summary
                        if row["target"] == target
                        and row["metric"] == metric
                        and row["fraction"] == fraction
                        and row["horizon"] == horizon
                    ]
                    source = sorted(
                        (
                            row for row in selected
                            if row["sharing"] != "scratch"
                        ),
                        key=lambda row: row["source_count"],
                    )
                    scratch = [
                        row for row in selected
                        if row["sharing"] == "scratch"
                    ]
                    if source:
                        plot_line(
                            x=[row["source_count"] for row in source],
                            y=[row["mean"] for row in source],
                            ax=axis, marker="o",
                            linewidth=DEFAULT_LINE_WIDTH,
                            color=TARGET_COLOR, label="Transfer",
                        )
                    if scratch:
                        if len(scratch) != 1:
                            raise ValueError(
                                "Expected one matched scratch baseline."
                            )
                        axis.axhline(
                            scratch[0]["mean"],
                            color=SCRATCH_COLOR, linestyle="--",
                            linewidth=DEFAULT_LINE_WIDTH,
                            label="Matched scratch",
                        )
                    axis.set_title(f"Horizon {horizon}")
                    axis.set_xticks(counts)
                    axis.set_xlim(min(counts) - 1, max(counts) + 1)
                    axis.set_xlabel("Source sessions")
                    axis.set_ylabel(metric.upper())
                    if axis.get_legend() is not None:
                        axis.get_legend().remove()
                handles, labels = axes.flat[0].get_legend_handles_labels()
                if handles:
                    figure.legend(
                        handles, labels, loc="upper left",
                        bbox_to_anchor=(0.82, 0.88), frameon=False,
                    )
                figure.suptitle(
                    fill(
                        f"{target.title()} {metric.upper()}, "
                        f"{fraction:.0%} target labels"
                        + (f" ({INTERIM_NOTE})"
                           if counts != COUNTS else ""),
                        TITLE_WIDTH,
                    ),
                    y=1.04,
                )
                figure.tight_layout(rect=(0, 0, 0.80, 1))
                figure.savefig(
                    directory / (
                        f"transfer_{target}_{metric}_"
                        f"{int(fraction * 100)}.png"
                    ),
                    dpi=180, bbox_inches="tight",
                )
                plt.close(figure)
                draw_gains(
                    gain_summary, directory, target,
                    metric, fraction, settings, counts,
                )


@plt.rc_context(READABLE_TEXT_STYLE)
def draw_gains(
    summary: list[dict], directory: Path, target: str,
    metric: str, fraction: float, settings: dict,
    counts: tuple[int, ...] = COUNTS,
) -> None:
    """Plot paired transfer-minus-scratch gain at all six horizons."""
    if not summary:
        return
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    for axis, horizon in zip(
        axes.flat, settings["evaluation_horizons"]
    ):
        selected = sorted(
            (
                row for row in summary
                if row["target"] == target
                and row["metric"] == metric
                and row["fraction"] == fraction
                and row["horizon"] == horizon
            ),
            key=lambda row: row["source_count"],
        )
        if selected:
            plot_line(
                x=[row["source_count"] for row in selected],
                y=[row["mean"] for row in selected],
                ax=axis, marker="o", linewidth=DEFAULT_LINE_WIDTH,
                color=TARGET_COLOR,
            )
        axis.axhline(0, color=SCRATCH_COLOR, linestyle="--")
        axis.set_title(f"Horizon {horizon}")
        axis.set_xticks(counts)
        axis.set_xlim(min(counts) - 1, max(counts) + 1)
        axis.set_xlabel("Source sessions")
        axis.set_ylabel(f"Improvement over scratch ({metric.upper()})")
    figure.suptitle(
        fill(
            f"Paired {target} {metric.upper()}, "
            f"{fraction:.0%} target labels"
            + (f" ({INTERIM_NOTE})" if counts != COUNTS else ""),
            TITLE_WIDTH,
        ),
        y=1.04,
    )
    figure.tight_layout()
    figure.savefig(
        directory / (
            f"paired_{target}_{metric}_{int(fraction * 100)}.png"
        ),
        dpi=180, bbox_inches="tight",
    )
    plt.close(figure)


def validate_target_coverage(
    summary: list[dict], gains: list[dict], settings: dict,
    counts: tuple[int, ...] = COUNTS,
) -> None:
    """Require every source-count and scratch curve at every horizon."""
    conditions = {
        (target, metric, fraction, horizon)
        for target in TARGET_TYPES
        for metric in TARGET_METRICS
        for fraction in settings["target_fractions"]
        for horizon in settings["evaluation_horizons"]
    }
    expected_source = {
        (*condition, count)
        for condition in conditions for count in counts
    }
    source = {
        (row["target"], row["metric"], row["fraction"],
         row["horizon"], row["source_count"])
        for row in summary if row["sharing"] != "scratch"
    }
    scratch = {
        (row["target"], row["metric"], row["fraction"],
         row["horizon"])
        for row in summary if row["sharing"] == "scratch"
    }
    paired = {
        (row["target"], row["metric"], row["fraction"],
         row["horizon"], row["source_count"])
        for row in gains
    }
    if (
        source != expected_source or scratch != conditions
        or paired != expected_source
        or len(summary) != len(expected_source) + len(conditions)
        or len(gains) != len(expected_source)
    ):
        raise ValueError(
            "Completed target fits lack requested source-count, scratch, "
            "or paired-gain figure coverage."
        )


def report_members(
    plan: dict, counts: tuple[int, ...],
) -> list[str]:
    """Select complete-design or authorized interim report members."""
    if counts not in (COUNTS, INTERIM_COUNTS):
        raise ValueError(f"Unsupported report source counts: {counts}.")
    if counts != COUNTS and revised_protocol(plan):
        raise ValueError(
            "Interim figures are not authorized for this protocol."
        )
    members = plan["analyses"][CAMPAIGN]
    if not members:
        raise ValueError("Compute-scaling analysis has no planned fits.")
    return [
        key for key in members
        if plan["jobs"][key]["kind"] in ("preprocessing", "scratch")
        or plan["jobs"][key].get("source_count") in counts
    ]


def campaign_complete(
    plan: dict, final: bool, counts: tuple[int, ...] = COUNTS,
) -> bool:
    """Require every selected fit before publishing campaign figures."""
    members = report_members(plan, counts)
    missing = [
        key for key in members
        if not (fit_directory(plan, key) / "complete.json").is_file()
    ]
    if missing:
        if final:
            raise ValueError(
                f"Cannot publish compute-scaling figures: "
                f"{len(missing)} of {len(members)} fits are incomplete."
            )
        return False
    for key in members:
        if completed(fit_directory(plan, key)) is None:
            raise ValueError(f"Invalid completed campaign fit: {key}.")
    return True


def revised_protocol(plan: dict) -> bool:
    """Identify the explicitly configured fixed-preprocessing protocol."""
    return bool(plan.get("settings", {}).get("source_training", {}).get(
        "protocol"
    ))


def source_report(
    plan: dict, plan_path: Path,
    counts: tuple[int, ...] = COUNTS,
) -> Path | None:
    """Publish one source-only snapshot after its counts finish."""
    if counts not in (COUNTS, INTERIM_COUNTS):
        raise ValueError(f"Unsupported source-only counts: {counts}.")
    keys = [
        key for key in plan["source_references"]
        if plan["jobs"][key]["source_count"] in counts
    ]
    if any(completed(fit_directory(plan, key)) is None for key in keys):
        return None
    sources, sessions, choices = source_tables(plan, counts)
    if len(choices) != len(counts):
        raise ValueError(
            f"Source figure group requires counts {counts}."
        )
    root = analysis_directory(plan, CAMPAIGN, plan_path.parent.name)
    identity = dict(
        group="recurrent_source", source_counts=list(counts),
        rendering=RENDERING_STYLE,
        protocol=plan["settings"]["source_training"],
        sources=[(row["source_fit"], row["component_checksum"])
                 for row in sources],
    )
    snapshot = root / "reports" / fingerprint(identity)
    with job(snapshot, identity) as needed:
        if needed:
            tables, figures = snapshot / "tables", snapshot / "figures"
            tables.mkdir()
            figures.mkdir()
            write_table(tables / "source_loss.csv", sources)
            write_table(tables / "source_session_loss.csv", sessions)
            write_table(tables / "selection.csv", choices)
            diagnostics = []
            for key in keys:
                spec = plan["jobs"][key]
                fit = fit_directory(plan, key)
                paths = inherited_paths(fit)
                seen = set()
                for path in paths:
                    for stage, row in path["stages"].items():
                        if row["component_id"] in seen:
                            continue
                        seen.add(row["component_id"])
                        component = (fit.parent.parent / "source_components"
                                     / row["component_id"])
                        training = json.loads(
                            (component / "training.json").read_text()
                        )
                        for name, exposure in zip(
                            spec["sessions"], row["exposure"]
                        ):
                            diagnostics.append(dict(
                                source_count=spec["source_count"], stage=stage,
                                component_id=row["component_id"], session=name,
                                actual_stop=training["stop"],
                                selected_update=training["selected_update"],
                                component_flops=row["component_flops"],
                                inner_evaluation_flops=training[
                                    "evaluation_flops"
                                ],
                                outer_evaluation_flops=row["score"][
                                    "evaluation_flops"
                                ], **exposure,
                            ))
            write_table(tables / "component_diagnostics.csv", diagnostics)
            atomic_json(snapshot / "design.json", dict(
                source_counts=list(counts), source_components=RECURRENT_STAGES,
                rendering=RENDERING_STYLE,
                planned_source_counts=list(COUNTS),
                report_scope=("complete" if counts == COUNTS else "interim"),
                expected_common_windows=EXPECTED_COMMON_WINDOWS,
                x_axis="Full-pipeline training FLOPs through actual stopping",
                holdout_role="Chronological source test and transfer selection",
                selected_score_interpretation="Descriptive after selection",
                evaluation_flops_in_training_axis=False,
            ))
            source_figures(
                sources, figures, counts=counts, stages=RECURRENT_STAGES,
                expected_points=len(
                    plan["settings"]["source_training"]["total_updates"]
                ),
            )
            print(f"Saved source figures: {figures.resolve()}")
    pointer = dict(
        snapshot=str((snapshot / "figures").resolve()),
        identity=fingerprint(identity),
    )
    atomic_json(root / "latest_source_figures.json", pointer)
    atomic_json(root / "latest_figures.json", pointer)
    return snapshot


def report(
    plan: dict, plan_path: Path, final: bool = False,
    counts: tuple[int, ...] = COUNTS,
) -> None:
    """Publish a complete design or the authorized three-count interim."""
    source_snapshot = None
    if revised_protocol(plan):
        if counts != COUNTS:
            raise ValueError(
                "Interim figures are not authorized for this protocol."
            )
        source_snapshot = source_report(plan, plan_path)
    if plan.get("source_only"):
        return
    if not campaign_complete(plan, final or counts != COUNTS, counts):
        return
    root = analysis_directory(
        plan, CAMPAIGN, plan_path.parent.name
    )
    sources, sessions, choices = source_tables(plan, counts)
    if not sources:
        raise ValueError("Completed campaign has no source loss rows.")
    members = [
        key for key in report_members(plan, counts)
        if plan["jobs"][key]["kind"] in ("scratch", "adaptation")
    ]
    rows, dimensions, identities = collect(plan, members)
    if len(identities) != len(members):
        raise ValueError(
            f"Expected {len(members)} completed target fits, "
            f"got {len(identities)}."
        )
    if len(choices) != len(counts):
        raise ValueError(
            f"Expected {len(counts)} completed source fits, "
            f"got {len(choices)}."
        )
    summary = summarize(rows) if rows else []
    gains = paired_gains(rows) if rows else []
    for row in gains:
        row["improvement"] = (
            -row["difference"] if row["metric"] == "mse"
            else row["difference"]
        )
        row["favorable_direction"] = (
            "lower" if row["metric"] == "mse" else "higher"
        )
    gain_summary = summarize([
        dict(row, value=row["improvement"]) for row in gains
    ]) if gains else []
    validate_target_coverage(
        summary, gain_summary, plan["settings"], counts
    )
    identity = dict(
        source=[
            (row["source_fit"], row["component_checksum"])
            for row in sources
        ],
        targets=identities, rendering=RENDERING_STYLE,
    )
    if counts != COUNTS:
        identity["interim_source_counts"] = list(counts)
    snapshot = root / "reports" / fingerprint(identity)
    with job(snapshot, identity) as needed:
        if needed:
            tables = snapshot / "tables"
            images = snapshot / "figures"
            tables.mkdir()
            images.mkdir()
            if source_snapshot is None:
                write_table(tables / "source_loss.csv", sources)
                write_table(tables / "source_session_loss.csv", sessions)
                write_table(tables / "selection.csv", choices)
            if rows:
                write_table(tables / "transfer_metrics.csv", rows)
                write_table(tables / "transfer_summary.csv", summary)
                write_table(
                    tables / "transfer_dimensions.csv", dimensions
                )
            if gains:
                write_table(tables / "paired_gain.csv", gains)
                write_table(
                    tables / "paired_gain_summary.csv", gain_summary
                )
            atomic_json(snapshot / "design.json", dict(
                source_holdout_role=(
                    "chronological source test and transfer "
                    "checkpoint selection"
                ),
                selected_score_interpretation=(
                    "Descriptive because the same holdout selects "
                    "the transfer checkpoint."
                ),
                expected_common_windows=EXPECTED_COMMON_WINDOWS,
                source_counts=list(counts),
                planned_source_counts=list(COUNTS),
                report_scope=(
                    "interim" if counts != COUNTS else "complete"
                ),
                source_components=list(COMPONENTS),
                complete_target_fits=len(identities),
            ))
            if source_snapshot is None:
                source_figures(sources, images, counts)
            else:
                atomic_json(snapshot / "source_reference.json", dict(
                    snapshot=str(source_snapshot.resolve()),
                ))
            if rows:
                target_figures(
                    summary, gain_summary, images,
                    plan["settings"], counts,
                )
    atomic_json(root / "latest_report.json", dict(
        snapshot=str(snapshot.resolve()),
        identity=fingerprint(identity),
    ))
    atomic_json(root / "latest_figures.json", dict(
        snapshot=str((snapshot / "figures").resolve()),
        source_snapshot=(str((source_snapshot / "figures").resolve())
                         if source_snapshot is not None else None),
        identity=fingerprint(identity),
    ))
    print(f"Saved scaling figures: {(snapshot / 'figures').resolve()}")
