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
from experiments.multisession.behavior_compute_scaling.plan import (
    CAMPAIGN, COMPONENTS, COUNTS,
)
from experiments.multisession.plan import analysis_directory
from experiments.multisession.reports import (
    collect, paired_gains, summarize, write_table,
)
from experiments.multisession.store import completed, fit_directory, job
from experiments.storage.cache import atomic_json, fingerprint
from experiments.multisession.scaling_report import COLORS
from experiments.visualization.multisession import (
    DEFAULT_LINE_WIDTH, plot_line,
)

EXPECTED_COMMON_WINDOWS = 87
SOURCE_FIGURE_SIZE = (8, 5)
TITLE_WIDTH = 58
TARGET_COLOR = "#2878b5"
SCRATCH_COLOR = "#222222"
SELECTED_MARKER = "s"


def source_tables(plan: dict) -> tuple[list[dict], list[dict], list[dict]]:
    """Collect checkpoint, session-loss and selection provenance."""
    losses, sessions, choices = [], [], []
    expected_set = None
    for key in plan["source_references"]:
        directory = fit_directory(plan, key)
        completion = completed(directory)
        if completion is None:
            continue
        spec = plan["jobs"][key]
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
        for path in json.loads(
            (directory / "budgets.json").read_text()
        ):
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


def source_figures(rows: list[dict], directory: Path) -> None:
    """Draw one log-log four-count source test panel per stage."""
    if not rows:
        return
    for stage in COMPONENTS:
        figure, axis = plt.subplots(figsize=SOURCE_FIGURE_SIZE)
        stage_rows = [row for row in rows if row["stage"] == stage]
        for count in COUNTS:
            selected = sorted(
                (
                    row for row in stage_rows
                    if row["source_count"] == count
                ),
                key=lambda row: row["full_path_flops"],
            )
            if not selected:
                continue
            if len(selected) != 5:
                raise ValueError(
                    f"Expected five source FLOP points for {count} sessions."
                )
            xs = [row["full_path_flops"] for row in selected]
            ys = [
                row["chronological_source_test_loss"]
                for row in selected
            ]
            plot_line(
                x=xs, y=ys, ax=axis,
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
        axis.set_title(
            fill(
                f"{stage.replace('_', ' ').title()}: fixed four-session "
                "chronological source test",
                TITLE_WIDTH,
            ),
            pad=22,
        )
        axis.text(
            0, -0.25,
            "The same holdout selects transfer checkpoints; selected "
            "source scores are descriptive.",
            transform=axis.transAxes, fontsize=9,
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


def target_figures(
    summary: list[dict], gain_summary: list[dict], directory: Path,
    settings: dict,
) -> None:
    """Plot absolute and paired transfer metrics by source session count."""
    if not summary:
        return
    for target in ("neural", "behavior"):
        for metric in ("cc", "r2", "mse"):
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
                    axis.set_xticks(COUNTS)
                    axis.set_xlim(min(COUNTS) - 1, max(COUNTS) + 1)
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
                        f"{fraction:.0%} target labels",
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
                    metric, fraction, settings,
                )


def draw_gains(
    summary: list[dict], directory: Path, target: str,
    metric: str, fraction: float, settings: dict,
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
        axis.set_xticks(COUNTS)
        axis.set_xlim(min(COUNTS) - 1, max(COUNTS) + 1)
        axis.set_xlabel("Source sessions")
        axis.set_ylabel(f"Improvement over scratch ({metric.upper()})")
    figure.suptitle(
        fill(
            f"Paired {target} {metric.upper()}, "
            f"{fraction:.0%} target labels",
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


def report(plan: dict, plan_path: Path, final: bool = False) -> None:
    """Publish a new immutable analysis snapshot when fits advance."""
    root = analysis_directory(
        plan, CAMPAIGN, plan_path.parent.name
    )
    sources, sessions, choices = source_tables(plan)
    if not sources:
        return
    members = [
        key for key in plan["analyses"][CAMPAIGN]
        if plan["jobs"][key]["kind"] in (
            "scratch", "adaptation"
        )
    ]
    rows, dimensions, identities = collect(plan, members)
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
    identity = dict(
        source=[
            (row["source_fit"], row["component_checksum"])
            for row in sources
        ],
        targets=identities,
    )
    snapshot = root / "reports" / fingerprint(identity)
    with job(snapshot, identity) as needed:
        if needed:
            tables = snapshot / "tables"
            images = snapshot / "figures"
            tables.mkdir()
            images.mkdir()
            write_table(tables / "source_loss.csv", sources)
            write_table(
                tables / "source_session_loss.csv", sessions
            )
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
                source_counts=list(COUNTS),
                source_components=list(COMPONENTS),
                complete_target_fits=len(identities),
            ))
            source_figures(sources, images)
            if rows:
                target_figures(
                    summary, gain_summary, images,
                    plan["settings"],
                )
    atomic_json(root / "latest_report.json", dict(
        snapshot=str(snapshot.resolve()),
        identity=fingerprint(identity),
    ))
    atomic_json(root / "latest_figures.json", dict(
        snapshot=str((snapshot / "figures").resolve()),
        identity=fingerprint(identity),
    ))
    print(f"Saved scaling figures: {(snapshot / 'figures').resolve()}")
