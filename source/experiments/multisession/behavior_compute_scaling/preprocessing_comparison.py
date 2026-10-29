"""Compare saved source models with and without behavior preprocessing.

Usage: python -m experiments.multisession.behavior_compute_scaling.
preprocessing_comparison --with-plan ABSOLUTE_PLAN
  --with-source-report ABSOLUTE_REPORT --with-n3-report ABSOLUTE_REPORT
  --direct-plan ABSOLUTE_PLAN --direct-report ABSOLUTE_REPORT

Inputs are two checksum-valid source reports and the n3=16 sweep report.
Output is an immutable sibling report under the direct campaign's reports/
with per-recording scores, paired differences, four-session means, selected
checkpoint references, a guide, and nine target/metric PNG figures.
No model is fitted or adapted.
"""

import argparse
import csv
import json
from pathlib import Path
from textwrap import fill

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.multisession.behavior_compute_scaling import (
    behavior_decoding_diagnostics as decoding,
    direct_report,
)
from experiments.multisession.behavior_compute_scaling.ancestry import (
    inherited_paths,
)
from experiments.multisession.behavior_compute_scaling.plan import COUNTS
from experiments.multisession.plan import analysis_directory
from experiments.multisession.reports import write_table
from experiments.multisession.store import completed, job
from experiments.storage.cache import atomic_json, fingerprint
from experiments.visualization.multisession import (
    READABLE_TEXT_STYLE, plot_line,
)

N3_WIDTH = 16
COMMON_SESSIONS = 4
HORIZONS = (1, 2, 4, 8)
GROUPS = ("position", "velocity", "neural")
METRICS = ("cc", "r2", "mse_ratio")
VARIANTS = ("with_preprocessing_n3_16", "direct_no_n3")
STYLES = {
    VARIANTS[0]: dict(
        color="#7551a8", marker="o", linestyle="-",
        label="Preprocessing + n₃=16",
    ),
    VARIANTS[1]: dict(
        color="#2878b5", marker="D", linestyle="--",
        label="Direct behavior, n₃=0",
    ),
}
FIGURE_SIZE = (15, 10)
DPI = 180
GUIDE = """# Behavioral preprocessing comparison

The purple model has the original n0=150 neural preprocessing, generated
behavior target, and independently fitted session-private n3=16. The blue
model predicts recorded behavior directly and has no n0 or n3. Both use
n1=n2=16, seed 42, A1,fw plus RNN2 sharing, the same nested source sets,
and the same 87 chronological windows in four common recordings.

Each source count uses the checkpoint minimizing the sum of RNN1 and RNN2
mean session dummy-normalized outer-test losses among the five recurrent
budgets common to both campaigns. The direct campaign's saved original
selection used seven budgets; this report reselects from five without
changing fits. The same holdout selects these checkpoints, so the plotted
selected scores are descriptive. Fixed-stage training and n3 add compute
only to the purple model. The two pipelines differ in both preprocessing
and n3; their difference cannot be attributed to preprocessing alone.

Behavior metrics score recorded position and velocity separately. Neural
metrics score the full RNN1 readout plus RNN2 residual against smoothed,
training-normalized spike counts. All four trained horizons are shown:
50, 100, 200 and 400 ms. Each plotted point is an equal mean of four
recording scores. MSE/dummy MSE uses each recording's training-target-mean
predictor on the same test samples; values above one are valid. CC and R2
are channel-mean metrics. No target adaptation or uncertainty interval is
included. Paired differences in paired_scores.csv are direct minus
preprocessing; lower MSE ratio and higher CC/R2 are preferable.
"""


def read_rows(path: Path) -> list[dict]:
    """Read a nonempty CSV table from a completed report."""
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Missing comparison rows: {path.resolve()}.")
    return rows


def checked_report(path: Path) -> dict:
    """Require a checksum-valid immutable input report."""
    marker = completed(path)
    if marker is None:
        raise ValueError(f"Incomplete input report: {path.resolve()}.")
    return marker


def matched_sources(
    with_sources: list[dict], direct_sources: list[dict],
) -> list[dict]:
    """Reselect direct checkpoints within the five shared budgets."""
    paired = []
    for original, direct in zip(with_sources, direct_sources):
        if original["count"] != direct["count"]:
            raise ValueError("Source count pairing changed.")
        for field in ("sessions", "latent", "sharing", "seed"):
            if original["spec"][field] != direct["spec"][field]:
                raise ValueError(
                    f"Source {original['count']} has different {field}."
                )
        if (
            original["selection"]["evaluation_set_id"]
            != direct["selection"]["evaluation_set_id"]
        ):
            raise ValueError("Chronological evaluation windows differ.")
        spec = original["spec"]["source_training"]
        multiplier = spec["update_multipliers"][str(original["count"])]
        budgets = {
            int(value) * int(multiplier)
            for value in spec["total_updates"]
        }
        candidates = [
            row for row in inherited_paths(direct["fit"])
            if row["total_updates"] in budgets
        ]
        if len(candidates) != len(budgets):
            raise ValueError(
                f"Expected {len(budgets)} common budgets for "
                f"{direct['count']} sessions, got {len(candidates)}."
            )
        chosen = min(
            candidates,
            key=lambda row: (
                row["selection_score"], row["full_path_flops"]
            ),
        )
        updated = dict(direct)
        updated["path"] = chosen
        updated["selection"] = dict(
            direct["selection"], selected_budget=chosen["budget"],
            selection_score=chosen["selection_score"],
            selected_full_path_flops=chosen["full_path_flops"],
        )
        paired.append(updated)
    return paired


def n3_behavior(
    report: Path, sources: list[dict],
) -> list[dict]:
    """Load the existing n3=16 per-recording performance only."""
    rows = read_rows(report / "session_scores.csv")
    selected = [
        row for row in rows if int(row["n3_size"]) == N3_WIDTH
    ]
    expected = len(COUNTS) * COMMON_SESSIONS * len(HORIZONS) * 2
    if len(selected) != expected:
        raise ValueError(
            f"Expected {expected} n3=16 behavior scores, "
            f"got {len(selected)}."
        )
    source_map = {row["count"]: row for row in sources}
    parsed = []
    for row in selected:
        count = int(row["source_count"])
        source = source_map[count]
        if (
            row["source_fit"] != source["fit"].name
            or int(row["selected_budget"])
            != source["selection"]["selected_budget"]
            or row["evaluation_set_id"]
            != source["selection"]["evaluation_set_id"]
        ):
            raise ValueError("n3=16 source checkpoint differs.")
        parsed.append(dict(
            variant=VARIANTS[0], source_count=count,
            source_fit=row["source_fit"],
            selected_budget=int(row["selected_budget"]),
            evaluation_set_id=row["evaluation_set_id"],
            session=row["session"], horizon=int(row["horizon"]),
            group=row["group"], samples=int(row["samples"]),
            **{
                metric: float(row[metric])
                for metric in METRICS + ("mse", "dummy_mse")
            },
        ))
    return parsed


def scored_models(
    direct_plan: dict, with_sources: list[dict],
    direct_sources: list[dict], n3_report: Path,
) -> list[dict]:
    """Score selected full neural models and direct behavior."""
    common = decoding.fixed_windows(direct_plan, direct_sources)
    scores = n3_behavior(n3_report, with_sources)
    for variant, sources in (
        (VARIANTS[0], with_sources),
        (VARIANTS[1], direct_sources),
    ):
        for source in sources:
            neural = direct_report.full_neural_forecasts(source, common)
            rows = direct_report.neural_scores(source, common, neural)
            direct_report.verify_stage_loss(
                source, rows, "residual", 1
            )
            scores.extend(dict(row, variant=variant) for row in rows)
            if variant == VARIANTS[1]:
                behavior = decoding.selected_forecasts(source, common)
                rows = decoding.session_scores(
                    source, common, behavior
                )
                direct_report.verify_stage_loss(
                    source, rows, "behavior_relevant", 2
                )
                scores.extend(
                    dict(row, variant=variant) for row in rows
                )
    return scores


def paired_scores(scores: list[dict]) -> list[dict]:
    """Check identical scoring samples and pair the two variants."""
    indexed = {}
    for row in scores:
        key = (
            row["variant"], row["source_count"], row["session"],
            row["horizon"], row["group"],
        )
        if key in indexed:
            raise ValueError(f"Duplicate comparison score: {key}.")
        indexed[key] = row
    expected = len(VARIANTS) * len(COUNTS) * COMMON_SESSIONS
    expected *= len(HORIZONS) * len(GROUPS)
    if len(indexed) != expected:
        raise ValueError(
            f"Expected {expected} comparison scores, got {len(indexed)}."
        )
    paired = []
    for row in scores:
        if row["variant"] != VARIANTS[0]:
            continue
        opposite = indexed.get((
            VARIANTS[1], row["source_count"], row["session"],
            row["horizon"], row["group"],
        ))
        if opposite is None or (
            row["samples"] != opposite["samples"]
            or row["evaluation_set_id"]
            != opposite["evaluation_set_id"]
            or not np.isclose(
                row["dummy_mse"], opposite["dummy_mse"],
                rtol=1e-5, atol=1e-6,
            )
        ):
            raise ValueError(
                "Paired models use different test samples or dummy loss."
            )
        paired.append(dict(
            source_count=row["source_count"],
            session=row["session"], horizon=row["horizon"],
            group=row["group"], samples=row["samples"],
            **{
                f"direct_minus_preprocessing_{metric}": (
                    opposite[metric] - row[metric]
                )
                for metric in METRICS
            },
        ))
    return paired


def comparison_figures(summary: list[dict], directory: Path) -> None:
    """Plot paired pipelines by source count for all trained horizons."""
    directory.mkdir()
    with plt.rc_context(READABLE_TEXT_STYLE):
        for group in GROUPS:
            for metric in METRICS:
                figure, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE)
                for axis, horizon in zip(axes.flat, HORIZONS):
                    observed = []
                    for variant in VARIANTS:
                        selected = sorted(
                            (
                                row for row in summary
                                if row["variant"] == variant
                                and row["group"] == group
                                and row["horizon"] == horizon
                            ),
                            key=lambda row: row["source_count"],
                        )
                        if [
                            row["source_count"] for row in selected
                        ] != list(COUNTS):
                            raise ValueError(
                                f"Missing {variant} {group} "
                                f"horizon {horizon} scores."
                            )
                        values = [row[metric] for row in selected]
                        if not np.isfinite(values).all():
                            raise ValueError("Nonfinite comparison metric.")
                        observed.extend(values)
                        plot_line(
                            x=COUNTS, y=values, ax=axis,
                            **STYLES[variant],
                        )
                    axis.set_xscale("log", base=2)
                    axis.set_xticks(
                        COUNTS, labels=[str(x) for x in COUNTS]
                    )
                    axis.set_xlim(
                        min(COUNTS) / 1.08,
                        max(COUNTS) * 1.08,
                    )
                    axis.set_xlabel("Source sessions")
                    axis.set_ylabel(decoding.METRIC_LABELS[metric])
                    axis.set_title(f"{horizon * 50} ms", pad=10)
                    decoding.focus_axis(axis, observed, metric)
                figure.suptitle(
                    fill(
                        f"{group.capitalize()}: preprocessing "
                        "and direct behavior",
                        45,
                    ),
                    y=0.98, fontsize=23,
                )
                handles, labels = axes.flat[0].get_legend_handles_labels()
                for axis in axes.flat:
                    legend = axis.get_legend()
                    if legend is not None:
                        legend.remove()
                figure.legend(
                    handles, labels, loc="upper left",
                    bbox_to_anchor=(0.79, 0.88), frameon=False,
                )
                figure.subplots_adjust(
                    left=0.09, right=0.77, top=0.88,
                    bottom=0.09, hspace=0.45, wspace=0.34,
                )
                figure.savefig(
                    directory / f"{group}_{metric}.png",
                    dpi=DPI, bbox_inches="tight",
                )
                plt.close(figure)
    print(f"Saved preprocessing comparison figures: {directory.resolve()}")


def publish(
    with_plan_path: Path, with_source_report: Path,
    with_n3_report: Path, direct_plan_path: Path,
    direct_report_path: Path,
) -> Path:
    """Publish checksum-pinned comparison without refitting models."""
    reports = (
        with_source_report, with_n3_report, direct_report_path,
    )
    markers = [checked_report(path) for path in reports]
    with_plan = json.loads(with_plan_path.read_text())
    direct_plan = json.loads(direct_plan_path.read_text())
    if with_plan["settings"]["data"] != direct_plan["settings"]["data"]:
        raise ValueError("Source data preprocessing settings differ.")
    with_sources = decoding.checked_sources(
        with_plan, with_source_report
    )
    direct_sources = decoding.checked_sources(
        direct_plan, direct_report_path
    )
    matched = matched_sources(with_sources, direct_sources)
    identity = dict(
        group="behavior_preprocessing_comparison_v1",
        input_checksums={
            path.name: marker["checksums"]
            for path, marker in zip(reports, markers)
        },
        matched_direct={
            str(source["count"]): dict(
                budget=source["path"]["budget"],
                stages=source["path"]["stages"],
            )
            for source in matched
        },
        guide=GUIDE,
    )
    root = analysis_directory(
        direct_plan, direct_plan["name"],
        direct_plan_path.parent.name,
    )
    destination = root / "reports" / fingerprint(identity)
    with job(destination, identity) as needed:
        if needed:
            scores = scored_models(
                direct_plan, with_sources, matched, with_n3_report
            )
            paired = paired_scores(scores)
            summary = [
                dict(row, variant=variant)
                for variant in VARIANTS
                for row in direct_report.summarize([
                    score for score in scores
                    if score["variant"] == variant
                ])
            ]
            tables = destination / "tables"
            tables.mkdir()
            write_table(tables / "session_scores.csv", scores)
            write_table(tables / "paired_scores.csv", paired)
            write_table(tables / "summary.csv", summary)
            write_table(tables / "selected_checkpoints.csv", [
                dict(
                    source_count=original["count"],
                    with_budget=original["path"]["budget"],
                    direct_budget=direct["path"]["budget"],
                    with_total_updates=original["path"][
                        "total_updates"
                    ],
                    direct_total_updates=direct["path"][
                        "total_updates"
                    ],
                    with_full_path_flops=original["path"][
                        "full_path_flops"
                    ],
                    direct_full_path_flops=direct["path"][
                        "full_path_flops"
                    ],
                )
                for original, direct in zip(with_sources, matched)
            ])
            (destination / "guide.md").write_text(GUIDE)
            atomic_json(destination / "inputs.json", dict(
                with_plan=str(with_plan_path.resolve()),
                with_source_report=str(with_source_report.resolve()),
                with_n3_report=str(with_n3_report.resolve()),
                direct_plan=str(direct_plan_path.resolve()),
                direct_report=str(direct_report_path.resolve()),
            ))
            comparison_figures(summary, destination / "figures")
    if completed(destination) is None:
        raise ValueError(f"Incomplete comparison: {destination}.")
    atomic_json(root / "latest_preprocessing_comparison.json", dict(
        snapshot=str(destination.resolve()),
        figures=str((destination / "figures").resolve()),
        identity=fingerprint(identity),
    ))
    return destination


def main() -> None:
    """Publish a matched selected-checkpoint comparison report."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "with-plan", "with-source-report", "with-n3-report",
        "direct-plan", "direct-report",
    ):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    publish(
        args.with_plan.resolve(), args.with_source_report.resolve(),
        args.with_n3_report.resolve(), args.direct_plan.resolve(),
        args.direct_report.resolve(),
    )


if __name__ == "__main__":
    main()
