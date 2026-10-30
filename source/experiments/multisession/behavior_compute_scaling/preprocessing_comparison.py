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
The default saved mode does not fit models. Use --stage matched-launch to
fit inner-stopped architecture controls, reselect saved sources on inner
validation and queue one complete report; --stage matched-report regenerates
that report from completed fits. --microbatch changes runtime accumulation.
Lifecycle logs contain no performance tables; private worker logs are nested
by model family and recording. No held-out session adaptations are launched.
"""

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
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
    single_session_baseline as baseline,
)
from experiments.multisession.behavior_compute_scaling.ancestry import (
    inherited_paths,
)
from experiments.multisession.behavior_compute_scaling.plan import COUNTS
from experiments.multisession.behavior_compute_scaling.pipeline import (
    inner_source_choices,
)
from experiments.multisession.plan import analysis_directory
from experiments.multisession.reports import write_table
from experiments.multisession.store import completed, job
from experiments.storage.cache import atomic_json, fingerprint
from experiments.visualization.multisession import (
    READABLE_TEXT_STYLE, plot_line,
)

MODULE = (
    "experiments.multisession.behavior_compute_scaling."
    "preprocessing_comparison"
)
MATCHED_PREFIX = "comparison-inner-selected-matched-baselines-2400"
MATCHED_PROTOCOL = "inner_selected_architecture_controls_session_sd_v1"
GPU_COUNT = 4
# GPU audit: 540 MiB resident; reserve space for downstream graph caches.
MATCHED_MEMORY_MIB = 2048
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



MATCHED_GUIDE = """# Inner-selected BRAID with matched single-session controls

All source-loss and performance figures are in figures/. Colored curves are
five-stage preprocessing BRAID plus n3=16, and three-stage direct BRAID
without n3. Gray horizontal lines are their separate single-session controls.
Both use n1=n2=16, seed 42, identical channel counts, recorded-behavior scales,
guarded splits, and 87 outer-test windows across the same four recordings.

Multi-session checkpoints minimize the sum of mean session RNN1 and full
neural RNN2 dummy-normalized losses on inner validation across seven budgets
50/100/200/400/800/1600/2400 (doubled for 33 sessions). For stage k, recording
s, and horizon h, L_ks = sum_h mean((prediction_ks,h - target_ks,h)^2).
D_ks replaces predictions with that recording's training-target mean.
Selection minimizes mean_s(L_1s/D_1s) + mean_s(L_2s/D_2s), breaking ties by
lower full-pipeline training FLOPs. The preprocessing model's RNN1 selection
target is generated behavior; the direct model's target is recorded behavior.
Both RNN2 losses score the full neural forecast. n3 does not rank backbones.

Each single-session recurrent stage has a cap of 2400 updates at batch 8,
with 1920 warmup/constant-rate updates and 480 linear-decay updates. The first
10 updates warm Adam from 1e-7 to 1e-3. Inner validation is checked every 25
updates and at the cap; patience is 10 checks, and the minimum-loss checkpoint
is restored. Fixed stages and n3 use constant 1e-3 and a 2000-update cap,
with the same inner stopping. Actual updates, best updates, and exposure are
in the training table. Initial private weights match the corresponding
recording in the four-source reference. Single-session stages are fitted
independently; they share no parameters across recordings.

Controls match the four-source reference's fixed-stage exposure: 16000 draws
per recording. Saved multi-session fixed-stage caps are 16000/8000/4000/1936
per recording for 4/8/16/33 sources, and n3 follows these count-specific caps.
The recurrent maximum exposure is 19200 draws per recording for all counts,
but saved multi-session recurrent fits have scheduled endpoints, while the
controls use the requested inner early stopping. These are architecture- and
maximum-exposure-matched controls, not identical stopping trajectories.

For position, velocity and neural predictions, MSE_s,h = mean((yhat-y)^2)
in the recording's training-normalized observed coordinates. Dummy MSE_s,h
uses its observed training-target mean on exactly the same test samples.
MSE/dummy MSE is computed within each recording, then averaged equally across
four recordings. CC and R2 average channels within a recording. Generated
behavior scaling is inverted before adding n3 and scoring recorded behavior.
Ratios above one are valid. Raw MSE, dummy MSE, and recording scores are saved.

Plotted means are sum_s m_s / 4. Error bars and gray bands are +/- one sample
session SD: sqrt(sum_s (m_s - mean)^2 / 3). They measure recording variability,
not uncertainty across seeds or statistical significance. Paired differences
against each family's control are retained per recording. Source loss panels
use each family's native training target, so their RNN1 loss scales differ.

All new selection and early stopping use inner validation; outer test is
scored after selection. These saved multi-session models' outer results were
historically inspected and used for transfer selection. Retrospective inner
reselection cannot make that history blind. No unseen-session adaptations
are queued. A seen-session test score may select a model only for a separate
unseen-session transfer evaluation, not for these seen-session comparisons.
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


def comparison_figures(
    summary: list[dict], directory: Path,
    controls: list[dict] | None = None,
) -> None:
    """Plot paired pipelines by source count for all trained horizons."""
    directory.mkdir(exist_ok=True)
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
                        if controls is not None:
                            deviations = np.asarray([
                                row[f"{metric}_sd"] for row in selected
                            ])
                            if not np.isfinite(deviations).all():
                                raise ValueError("Nonfinite session SD.")
                            axis.errorbar(
                                COUNTS, values, yerr=deviations, fmt="none",
                                ecolor=STYLES[variant]["color"],
                                capsize=4, elinewidth=1.6,
                            )
                            observed.extend(np.asarray(values) - deviations)
                            observed.extend(np.asarray(values) + deviations)
                            matches = [
                                row for row in controls
                                if row["variant"] == variant
                                and row["group"] == group
                                and row["horizon"] == horizon
                            ]
                            if len(matches) != 1:
                                raise ValueError("Missing matched control.")
                            control = matches[0]
                            mean = control[metric]
                            sd = control[f"{metric}_sd"]
                            observed.extend((mean - sd, mean + sd))
                            style = dict(STYLES[variant])
                            style.update(
                                color=("#555555" if variant == VARIANTS[0]
                                       else "#999999"),
                                label=("Single-session: preprocessing + n₃"
                                       if variant == VARIANTS[0] else
                                       "Single-session: direct behavior"),
                            )
                            plot_line(
                                x=COUNTS, y=[mean] * len(COUNTS),
                                ax=axis, **style,
                            )
                            axis.fill_between(
                                COUNTS, mean - sd, mean + sd,
                                color=style["color"], alpha=0.08,
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
                    handles, labels,
                    loc=("lower center" if controls is not None
                         else "upper left"),
                    bbox_to_anchor=(
                        (0.5, 0.01) if controls is not None else (0.79, 0.88)
                    ),
                    ncol=2 if controls is not None else 1, frameon=False,
                    title=("Mean ± one session SD (4 recordings)"
                           if controls is not None else None),
                    title_fontsize=16,
                )
                figure.subplots_adjust(
                    left=0.09, right=0.97 if controls is not None else 0.77,
                    top=0.88, bottom=0.22 if controls is not None else 0.09,
                    hspace=0.45, wspace=0.34,
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


def matched_sources(inputs: dict) -> tuple[dict, dict, Path]:
    """Cache inner-only selections independently of figures and private fits.

    Input keys are with_plan/with_source_report/direct_plan/direct_report
    absolute paths. The immutable selection cache stores budget identifiers,
    aggregate criteria and per-session raw/dummy losses; it never copies
    checkpoints or window inventories.
    """
    plans, groups, checksums = {}, {}, {}
    for variant, prefix in zip(VARIANTS, ("with", "direct")):
        plan_path = Path(inputs[f"{prefix}_plan"])
        report = Path(inputs[
            "with_source_report" if prefix == "with" else "direct_report"
        ])
        plans[variant] = json.loads(plan_path.read_text())
        sources = decoding.checked_sources(plans[variant], report)
        for source in sources:
            source["paths"] = inherited_paths(source["fit"])
            checksums[source["fit"].name] = checked_report(
                source["fit"]
            )["checksums"]
        groups[variant] = sources
    validate_selected_sources(*[groups[key] for key in VARIANTS])
    direct_plan = plans[VARIANTS[1]]
    root = analysis_directory(
        direct_plan, direct_plan["name"],
        Path(inputs["direct_plan"]).parent.name,
    )
    identity = dict(
        protocol=MATCHED_PROTOCOL, source_checksums=checksums,
        selection="inner_joint_dummy_normalized", budget_pool=BUDGET_POOL,
    )
    cache = root / "selections" / f"inner-{fingerprint(identity)}"
    with job(cache, identity) as needed:
        if needed:
            common = decoding.fixed_windows(
                direct_plan, groups[VARIANTS[0]] + groups[VARIANTS[1]]
            )
            for variant in VARIANTS:
                selected, rows = inner_source_choices(groups[variant], common)
                atomic_json(cache / f"{variant}.json", [dict(
                    source_count=source["count"],
                    selection=source["selection"],
                ) for source in selected])
                write_table(cache / f"{variant}_ranking.csv", rows)
                write_table(cache / f"{variant}_sessions.csv", [
                    dict(source_count=source["count"], **row)
                    for source in selected
                    for row in source["inner_session_scores"]
                ])
    for variant in VARIANTS:
        saved = json.loads((cache / f"{variant}.json").read_text())
        indexed = {row["source_count"]: row for row in saved}
        for source in groups[variant]:
            selection = indexed[source["count"]]["selection"]
            matches = [path for path in source["paths"]
                       if path["budget"] == selection["selected_budget"]]
            if len(matches) != 1:
                raise ValueError("Inner selection has no unique saved path.")
            source.update(selection=selection, path=matches[0])
    return plans, groups, cache


def control_scores(plans: dict, groups: dict) -> tuple[list[dict], list[Path]]:
    """Score completed architecture-matched controls on recorded quantities."""
    scores, fits = [], []
    for variant in VARIANTS:
        plan = plans[variant]
        reference = groups[variant][0]
        for index in range(COMMON_SESSIONS):
            spec, train, _, outer, record, directory = baseline.prepare(
                plan, index, matched=True
            )
            checked_report(directory)
            fits.append(directory)
            source = baseline.source_view(spec, record, directory)
            name = spec["sessions"][0]
            if record != reference["inventory"][name]:
                raise ValueError(f"Baseline split/scale differs for {name}.")
            common = {name: dict(training=train, test=outer)}
            predictions = decoding.selected_forecasts(source, common)
            if variant == VARIANTS[0]:
                predictions = [n3_session.corrected(
                    source, 0, outer, predictions[0], N3_WIDTH
                )]
            rows = decoding.session_scores(source, common, predictions)
            neural = direct_report.full_neural_forecasts(source, common)
            rows += direct_report.neural_scores(source, common, neural)
            scores.extend(dict(row, variant=variant) for row in rows)
    return scores, fits


def session_summary(rows: list[dict]) -> list[dict]:
    """Compute equal recording means and sample SD, never seed uncertainty."""
    grouped = {}
    for row in rows:
        key = tuple(row[field] for field in (
            "variant", "source_count", "group", "horizon"
        ))
        grouped.setdefault(key, []).append(row)
    result = []
    for (variant, count, group, horizon), records in sorted(grouped.items()):
        if len(records) != COMMON_SESSIONS or len({
            row["session"] for row in records
        }) != COMMON_SESSIONS:
            raise ValueError("Mean/SD requires four distinct recordings.")
        statistics = {}
        for metric in METRICS + ("mse", "dummy_mse"):
            values = np.asarray([row[metric] for row in records], dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite {metric} in recording scores.")
            statistics[metric] = float(values.mean())
            statistics[f"{metric}_sd"] = float(values.std(ddof=1))
        result.append(dict(
            variant=variant, source_count=count, group=group,
            horizon=horizon, recordings=COMMON_SESSIONS, **statistics,
        ))
    return result


def baseline_differences(
    scores: list[dict], controls: list[dict],
) -> list[dict]:
    """Pair each multi-session score to its own recording/family control."""
    indexed = {
        (row["variant"], row["session"], row["group"], row["horizon"]): row
        for row in controls
    }
    result = []
    for row in scores:
        control = indexed[(
            row["variant"], row["session"], row["group"], row["horizon"]
        )]
        if row["samples"] != control["samples"] or not np.isclose(
            row["dummy_mse"], control["dummy_mse"], rtol=1e-5, atol=1e-6,
        ):
            raise ValueError("Control and source score different test samples.")
        result.append(dict(
            variant=row["variant"], source_count=row["source_count"],
            session=row["session"], group=row["group"], horizon=row["horizon"],
            **{f"model_minus_control_{metric}": row[metric] - control[metric]
               for metric in METRICS},
        ))
    return result


def training_records(fits: list[Path]) -> list[dict]:
    """Audit actual and restored exposure, including all post-best updates."""
    rows = []
    for fit in fits:
        spec = json.loads((fit / "identity.json").read_text())["spec"]
        path = json.loads((fit / "path.json").read_text())
        stages = dict(path["stages"])
        if (fit / "n3.json").exists():
            stages["n3"] = dict(component_id=json.loads(
                (fit / "n3.json").read_text()
            )["component"])
        for stage, record in stages.items():
            component = fit.parent.parent / "source_components"
            component /= record["component_id"]
            checked_report(component)
            training = json.loads((component / "training.json").read_text())
            rows.append(dict(
                fit=fit.name, session=spec["sessions"][0], stage=stage,
                component=component.name,
                requested_updates=training["requested_stop"],
                actual_updates=training["stop"],
                best_update=training["selected_update"],
                actual_draws=sum(
                    len(row["sampled_windows"][0])
                    for row in training["history"]
                ),
                best_inner_loss=training["best_validation_loss"],
                training_flops=training["total_flops"],
                evaluation_flops=training["evaluation_flops"],
            ))
    return rows


def publish_matched(inputs: dict) -> Path:
    """Publish all comparison groups only after controls and n3 finish."""
    from .report import source_figures, source_tables

    plans, groups, cache = matched_sources(inputs)
    fits = [
        baseline.saved_spec(plans[variant], index, matched=True)[-1]
        for variant in VARIANTS for index in range(COMMON_SESSIONS)
    ]
    identity = dict(
        protocol=MATCHED_PROTOCOL, guide=MATCHED_GUIDE,
        selection_checksums=checked_report(cache)["checksums"],
        baseline_checksums={
            fit.name: checked_report(fit)["checksums"] for fit in fits
        },
        n3_components=n3_references(groups[VARIANTS[0]]),
    )
    root = cache.parent.parent
    destination = root / "reports" / f"{MATCHED_PREFIX}-{fingerprint(identity)}"
    with job(destination, identity) as needed:
        if needed:
            scores = scored_models(
                plans[VARIANTS[1]], groups[VARIANTS[0]],
                groups[VARIANTS[1]], None,
            )
            controls, _ = control_scores(plans, groups)
            summary = session_summary(scores)
            control_summary = session_summary(controls)
            tables = destination / "tables"
            figures = destination / "figures"
            tables.mkdir()
            figures.mkdir()
            for name, rows in (
                ("session_scores", scores), ("summary", summary),
                ("baseline_scores", controls),
                ("baseline_summary", control_summary),
                ("paired_scores", paired_scores(scores)),
                ("paired_baseline_gains",
                 baseline_differences(scores, controls)),
                ("baseline_training", training_records(fits)),
            ):
                write_table(tables / f"{name}.csv", rows)
            write_table(tables / "selection.csv", [dict(
                variant=variant, source_count=source["count"],
                selected_budget=source["path"]["budget"],
                total_updates=source["path"]["total_updates"],
                selection_set_id=source["selection"]["selection_set_id"],
                inner_score=source["selection"]["selection_score"],
                full_path_flops=source["path"]["full_path_flops"],
            ) for variant in VARIANTS for source in groups[variant]])
            for variant in VARIANTS:
                rows, _, _ = source_tables(plans[variant])
                chosen = {source["count"]: source["path"]["budget"]
                          for source in groups[variant]}
                for row in rows:
                    row["transfer_selected"] = (
                        row["budget"] == chosen[row["source_count"]]
                    )
                    row["test_holdout_selects_transfer"] = False
                    row["selection_role"] = "inner_validation"
                write_table(tables / f"{variant}_source_loss.csv", rows)
                source_figures(
                    rows, figures, stages=("behavior_relevant", "residual"),
                    expected_points=len(BUDGET_POOL), direct=True,
                    filename_prefix=f"{variant}_",
                    stage_labels={
                        "behavior_relevant": (
                            "Five-stage RNN1 generated-behavior loss"
                            if variant == VARIANTS[0] else
                            "Three-stage RNN1 recorded-behavior loss"
                        ),
                        "residual": (
                            "Five-stage full neural loss"
                            if variant == VARIANTS[0] else
                            "Three-stage full neural loss"
                        ),
                    },
                )
            comparison_figures(summary, figures, control_summary)
            (destination / "guide.md").write_text(MATCHED_GUIDE)
            atomic_json(destination / "inputs.json", dict(
                inputs, selection_cache=str(cache.resolve()),
                baseline_fits=[str(fit.resolve()) for fit in fits],
            ))
    checked_report(destination)
    pointer = dict(
        snapshot=str(destination.resolve()),
        figures=str((destination / "figures").resolve()),
        identity=fingerprint(identity),
    )
    atomic_json(root / "latest_preprocessing_comparison.json", pointer)
    atomic_json(root / "latest_matched_baseline_comparison.json", pointer)
    return destination


def matched_command(inputs: dict, stage: str) -> list[str]:
    """Build explicit argv without shell interpolation or local defaults."""
    command = [
        sys.executable, "-m", MODULE, "--stage", stage,
    ]
    for key in (
        "with_plan", "with_source_report", "direct_plan", "direct_report"
    ):
        command.extend(["--" + key.replace("_", "-"), inputs[key]])
    return command


def matched_tasks(inputs: dict, microbatch: int) -> list[dict]:
    """Resolve eight controls and sixteen independently reusable n3 fits."""
    plans, groups, _ = matched_sources(inputs)
    tasks = []
    for variant in VARIANTS:
        for index in range(COMMON_SESSIONS):
            spec, _, target = baseline.saved_spec(
                plans[variant], index, matched=True
            )
            tasks.append(dict(
                key=f"controls/{variant}/{spec['sessions'][0]}",
                target=str(target.resolve()), kind="control",
                variant=variant, index=index,
            ))
    for source in groups[VARIANTS[0]]:
        for index in range(COMMON_SESSIONS):
            tasks.append(dict(
                key=(f"n3/source_{source['count']}/"
                     f"{source['spec']['sessions'][index]}"),
                target=str(n3_session.directory(
                    source, index, N3_WIDTH
                ).resolve()),
                kind="n3", count=source["count"], index=index,
            ))
    for index, task in enumerate(tasks):
        task["command"] = matched_command(inputs, "matched-worker") + [
            "--task-index", str(index), "--microbatch", str(microbatch),
        ]
    return tasks


def matched_worker(inputs: dict, task_index: int, microbatch: int) -> None:
    """Run one independent recording fit, preserving completed components."""
    tasks = matched_tasks(inputs, microbatch)
    if task_index not in range(len(tasks)):
        raise ValueError(f"Invalid matched task index {task_index}.")
    task = tasks[task_index]
    if completed(Path(task["target"])) is not None:
        return
    if task["kind"] == "control":
        key = (
            "with_plan" if task["variant"] == VARIANTS[0] else "direct_plan"
        )
        plan = json.loads(Path(inputs[key]).read_text())
        result = baseline.fit_one(plan, task["index"], microbatch, matched=True)
    else:
        plans, groups, _ = matched_sources(inputs)
        source = next(row for row in groups[VARIANTS[0]]
                      if row["count"] == task["count"])
        result = n3_session.fit_one(
            plans[VARIANTS[0]], source, task["index"], microbatch, N3_WIDTH,
        )
    if result.resolve() != Path(task["target"]) or completed(result) is None:
        raise ValueError("Matched task did not complete its expected artifact.")


def matched_coordinator(
    inputs: dict, launch_dir: Path, microbatch: int,
) -> None:
    """Freeze selections, dispatch private fits, then publish the full group."""
    launch = json.loads((launch_dir / "launch.json").read_text())
    devices = launch["devices"]
    states = {"preparation": "running"}
    baseline.status(launch_dir, states, "selecting_inner_checkpoints")
    try:
        log = launch_dir / "jobs" / "selection" / "worker.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a") as stream:
            subprocess.run(
                matched_command(inputs, "matched-prepare"),
                env=baseline.worker_environment(devices[0]),
                stdout=stream, stderr=subprocess.STDOUT, check=True,
            )
        tasks = matched_tasks(inputs, microbatch)
        states = baseline.run_task_queue(tasks, devices, launch_dir)
        if "failed" in states.values():
            baseline.status(launch_dir, states, "failed")
            return
        baseline.status(launch_dir, states, "reporting")
        report = publish_matched(inputs)
        atomic_json(launch_dir / "report.json", dict(
            snapshot=str(report.resolve()),
        ))
        baseline.status(launch_dir, states, "complete")
    except Exception as error:
        atomic_json(launch_dir / "error.json", dict(
            error=f"{type(error).__name__}: {error}",
        ))
        baseline.status(launch_dir, states, "failed")
        raise


def launch_matched(inputs: dict, microbatch: int) -> Path:
    """Audit both first stages without retained fits and detach four GPUs."""
    from experiments.execution.runtime import select_gpus

    if not 1 <= microbatch <= baseline.SESSION_BATCH:
        raise ValueError("Expected microbatch between one and eight.")
    candidates = sorted(
        (device for device in select_gpus("auto", 8)
         if device["free_memory_mib"] >= MATCHED_MEMORY_MIB),
        key=lambda device: device["free_memory_mib"], reverse=True,
    )
    if len(candidates) < GPU_COUNT:
        raise RuntimeError("Matched controls require four available GPUs.")
    devices = candidates[:GPU_COUNT]
    subprocess.run(
        matched_command(inputs, "matched-audit")
        + ["--microbatch", str(microbatch)],
        env=baseline.worker_environment(devices[0]), check=True,
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    launch_dir = (
        Path(__file__).resolve().parents[4] / "logs/multi_session_training"
        / f"matched_baselines_{stamp}_{os.getpid()}"
    )
    launch_dir.mkdir(parents=True)
    atomic_json(launch_dir / "launch.json", dict(
        inputs=inputs, devices=devices, microbatch=microbatch,
        protocol=MATCHED_PROTOCOL, memory_budget_mib=MATCHED_MEMORY_MIB,
    ))
    command = matched_command(inputs, "matched-coordinator") + [
        "--launch-dir", str(launch_dir), "--microbatch", str(microbatch),
    ]
    with (launch_dir / "coordinator.log").open("a") as stream:
        process = subprocess.Popen(
            command, env=baseline.worker_environment(), stdout=stream,
            stderr=subprocess.STDOUT, start_new_session=True,
            preexec_fn=lambda: os.nice(baseline.LOW_CPU_PRIORITY),
        )
    atomic_json(launch_dir / "pid.json", dict(pid=process.pid))
    print(f"Detached matched baseline campaign: {launch_dir.resolve()}")
    return launch_dir


def main() -> None:
    """Compare each campaign at its saved best-budget checkpoints."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="saved", choices=(
        "saved", "matched-launch", "matched-audit", "matched-prepare",
        "matched-worker", "matched-coordinator", "matched-report",
    ))
    parser.add_argument("--microbatch", type=int, default=4)
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--launch-dir", type=Path)
    for name in (
        "with-plan", "with-source-report", "with-n3-report",
        "direct-plan", "direct-report",
    ):
        parser.add_argument(
            f"--{name}", required=name != "with-n3-report", type=Path
        )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    inputs = {
        key: str(getattr(args, key).resolve()) for key in (
            "with_plan", "with_source_report", "direct_plan", "direct_report"
        )
    }
    if args.stage != "saved":
        if args.stage in ("matched-audit", "matched-prepare", "matched-worker"):
            import tensorflow as tf

            devices = tf.config.list_physical_devices("GPU")
            if not devices:
                raise RuntimeError("Matched fit/selection worker needs a GPU.")
            tf.config.experimental.set_memory_growth(devices[0], True)
        if args.stage == "matched-launch":
            launch_matched(inputs, args.microbatch)
        elif args.stage == "matched-audit":
            for key in ("with_plan", "direct_plan"):
                baseline.audit(
                    json.loads(Path(inputs[key]).read_text()),
                    args.microbatch, matched=True,
                )
        elif args.stage == "matched-prepare":
            matched_sources(inputs)
        elif args.stage == "matched-worker":
            matched_worker(inputs, args.task_index, args.microbatch)
        elif args.stage == "matched-report":
            print(publish_matched(inputs).resolve())
        else:
            if args.launch_dir is None:
                raise ValueError("Coordinator requires --launch-dir.")
            matched_coordinator(
                inputs, args.launch_dir.resolve(), args.microbatch
            )
        return
    publish(
        args.with_plan.resolve(), args.with_source_report.resolve(),
        args.with_n3_report.resolve() if args.with_n3_report else None,
        args.direct_plan.resolve(),
        args.direct_report.resolve(),
    )


if __name__ == "__main__":
    main()
