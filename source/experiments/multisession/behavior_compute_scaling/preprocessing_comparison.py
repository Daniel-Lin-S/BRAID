"""Compare saved source models with and without behavior preprocessing.

Usage: python -m experiments.multisession.behavior_compute_scaling.
preprocessing_comparison --with-plan ABSOLUTE_PLAN
  --with-source-report ABSOLUTE_REPORT [--with-n3-report ABSOLUTE_REPORT]
  --direct-plan ABSOLUTE_PLAN --direct-report ABSOLUTE_REPORT

Inputs are two checksum-valid source reports and completed n3=16 fits.
An optional n3 sweep report supplies previously scored behavior metrics.
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
    n3_session,
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
COMPARISON_REPORT_PREFIX = "comparison-preprocessed-n3-16-vs-direct-best-7"
COMMON_SESSIONS = 4
BUDGET_POOL = (50, 100, 200, 400, 800, 1600, 2400)
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

Each campaign uses its saved best checkpoint at each source count. The
selection minimizes the sum of RNN1 and RNN2 mean session dummy-normalized
outer-test losses over all completed budgets in that campaign, with lower
FLOPs breaking an exact tie. Both campaigns use total recurrent budgets
50/100/200/400/800/1600/2400, doubled for 33 sessions. Each selects its best
budget independently. The same holdout selects these checkpoints,
so selected scores are descriptive. The n3=16 correction is applied to the
preprocessing campaign's selected model. The pipelines differ in
preprocessing and n3, and selected updates and FLOPs may differ; their
difference cannot be attributed to preprocessing alone.

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


def validate_selected_sources(
    with_sources: list[dict], direct_sources: list[dict],
) -> None:
    """Require compatible data and each campaign's saved best checkpoint.

    Parameters
    ----------
    with_sources, direct_sources : list of dict
        Four sources ordered by session count, containing completed fit
        paths, scientific specifications, and saved checkpoint selections.
    """
    if len(with_sources) != len(COUNTS) or (
        len(direct_sources) != len(COUNTS)
    ):
        raise ValueError("Expected four selected sources per campaign.")
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
        for source in (original, direct):
            paths = inherited_paths(source["fit"])
            multiplier = 2 if source["count"] == 33 else 1
            expected = [budget * multiplier for budget in BUDGET_POOL]
            actual = [path["total_updates"] for path in paths]
            if actual != expected:
                raise ValueError(
                    f"Expected complete budget pool {expected}, "
                    f"got {actual} for {source['fit'].resolve()}."
                )
            chosen = min(
                paths,
                key=lambda row: (
                    row["selection_score"], row["full_path_flops"]
                ),
            )
            if chosen["budget"] != source["path"]["budget"]:
                raise ValueError(
                    "Saved checkpoint is not the best completed budget "
                    f"for {source['fit'].resolve()}."
                )


def n3_behavior(
    report: Path | None, sources: list[dict], common: dict,
) -> list[dict]:
    """Score recorded behavior using matching saved private n3 models."""
    if report is None:
        scores = []
        for source in sources:
            priors = decoding.selected_forecasts(source, common)
            names = source["spec"]["sessions"][:COMMON_SESSIONS]
            predictions = [
                n3_session.corrected(
                    source, index, common[name]["test"],
                    priors[index], N3_WIDTH,
                )
                for index, name in enumerate(names)
            ]
            scores.extend(
                dict(row, variant=VARIANTS[0])
                for row in decoding.session_scores(
                    source, common, predictions
                )
            )
        return scores
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
    direct_sources: list[dict], n3_report: Path | None,
) -> list[dict]:
    """Score selected full neural models and direct behavior."""
    common = decoding.fixed_windows(direct_plan, direct_sources)
    scores = n3_behavior(n3_report, with_sources, common)
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


def n3_references(sources: list[dict]) -> dict:
    """Pin all sixteen selected private correction checkpoints."""
    references = {}
    for source in sources:
        references[str(source["count"])] = {}
        for index in range(COMMON_SESSIONS):
            name = n3_session.session_name(source, index)
            path = n3_session.directory(source, index, N3_WIDTH)
            marker = checked_report(path)
            references[str(source["count"])][name] = dict(
                component=path.name, checksums=marker["checksums"],
            )
    return references


def publish(
    with_plan_path: Path, with_source_report: Path,
    with_n3_report: Path | None, direct_plan_path: Path,
    direct_report_path: Path,
) -> Path:
    """Publish checksum-pinned comparison without refitting models."""
    reports = [with_source_report, direct_report_path]
    if with_n3_report is not None:
        reports.append(with_n3_report)
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
    validate_selected_sources(with_sources, direct_sources)
    identity = dict(
        group="behavior_preprocessing_comparison_seven_budgets_v3",
        budget_pool=list(BUDGET_POOL),
        n3_components=n3_references(with_sources),
        selection="saved_best_over_all_completed_budgets",
        input_checksums={
            path.name: marker["checksums"]
            for path, marker in zip(reports, markers)
        },
        selected_direct={
            str(source["count"]): dict(
                budget=source["path"]["budget"],
                stages=source["path"]["stages"],
            )
            for source in direct_sources
        },
        guide=GUIDE,
    )
    root = analysis_directory(
        direct_plan, direct_plan["name"],
        direct_plan_path.parent.name,
    )
    report_hash = fingerprint(identity)
    reports_root = root / "reports"
    destination = reports_root / (
        f"{COMPARISON_REPORT_PREFIX}-{report_hash}"
    )
    legacy = reports_root / report_hash
    if not destination.exists() and completed(legacy) is not None:
        if destination.is_symlink():
            raise ValueError(
                f"Broken comparison report alias: {destination.absolute()}."
            )
        destination.symlink_to(legacy.name, target_is_directory=True)
    with job(destination, identity) as needed:
        if needed:
            scores = scored_models(
                direct_plan, with_sources, direct_sources, with_n3_report
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
                for original, direct in zip(with_sources, direct_sources)
            ])
            (destination / "guide.md").write_text(GUIDE)
            atomic_json(destination / "inputs.json", dict(
                with_plan=str(with_plan_path.resolve()),
                with_source_report=str(with_source_report.resolve()),
                with_n3_report=(
                    str(with_n3_report.resolve())
                    if with_n3_report is not None else None
                ),
                direct_plan=str(direct_plan_path.resolve()),
                direct_report=str(direct_report_path.resolve()),
            ))
            comparison_figures(summary, destination / "figures")
    if completed(destination) is None:
        raise ValueError(f"Incomplete comparison: {destination}.")
    atomic_json(root / "latest_preprocessing_comparison.json", dict(
        snapshot=str(destination.absolute()),
        figures=str((destination / "figures").absolute()),
        identity=fingerprint(identity),
    ))
    return destination


def main() -> None:
    """Compare each campaign at its saved best-budget checkpoints."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "with-plan", "with-source-report", "with-n3-report",
        "direct-plan", "direct-report",
    ):
        parser.add_argument(
            f"--{name}", required=name != "with-n3-report", type=Path
        )
    args = parser.parse_args()
    publish(
        args.with_plan.resolve(), args.with_source_report.resolve(),
        args.with_n3_report.resolve() if args.with_n3_report else None,
        args.direct_plan.resolve(),
        args.direct_report.resolve(),
    )


if __name__ == "__main__":
    main()
