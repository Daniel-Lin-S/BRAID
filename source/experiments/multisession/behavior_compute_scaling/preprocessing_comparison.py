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
--stage paired-gain-report --matched-report ABSOLUTE_REPORT publishes paired
recording views from saved complete scores without fitting.
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
from matplotlib.lines import Line2D
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
from experiments.multisession.store import completed, fit_directory, job
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
GAIN_PREFIX = "comparison-paired-gains-single-and-four-source"
GAIN_PROTOCOL = "paired_recording_gain_views_v1"
GAIN_MARKERS = ("o", "s", "^", "D")
GAIN_REFERENCES = ("single_session", "four_source")
SIGN_TOLERANCE = 1e-6
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
    absolute paths; historical report paths are provenance only. Completed
    source fits define membership. The cache stores budget identifiers,
    aggregate criteria and per-session raw/dummy losses; it never copies
    checkpoints or window inventories.
    """
    plans, groups, checksums = {}, {}, {}
    for variant, prefix in zip(VARIANTS, ("with", "direct")):
        plan_path = Path(inputs[f"{prefix}_plan"])
        plan = json.loads(plan_path.read_text())
        plans[variant] = plan
        choices = []
        for key in plan["source_references"]:
            fit = fit_directory(plan, key)
            checked_report(fit)
            saved = json.loads((fit / "selection.json").read_text())
            choices.append(dict(
                source_fit=key, selected_budget=saved["selected_budget"],
            ))
        sources = decoding.checked_sources(plan, choices=choices)
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
        if "preparation" in states:
            states["preparation"] = "failed"
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


GAIN_GUIDE = """# Paired recording gains for BRAID source scaling

The nine `vs_single` figures compare each multi-session protocol with its
architecture-matched, independently fitted single-session BRAID on the same
four recordings. The nine `vs_four` figures show the additional change from
four to 8, 16, or 33 source recordings. Both protocols appear in every panel:
purple is five-stage preprocessing BRAID plus n3=16; blue is three-stage
direct-behavior BRAID without n3. Four thin marked lines show individual
recordings; a thick line is their equal mean. The gray zero line means no
change. The four-source gain is exactly zero at four sources by definition.

For recording s, count c, metric m and reference r, the gain is
`m(s,c) - m(s,r)` for CC and R2; for MSE/dummy MSE it is
`m(s,r) - m(s,c)`. Thus positive means improvement for every metric.
Single-session reference r=1 uses the matched control for that protocol and
recording. Four-source reference r=4 uses that protocol's four-source fit.
MSE/dummy MSE first divides each recording's model MSE by its own observed
training-target-mean dummy MSE on the identical chronological test samples.
Ratios above one remain valid. Positive/negative session counts classify
gains outside an absolute tolerance of 1e-6; near-zero gains are neutral.

The figures preserve all four trained horizons, position, velocity and full
neural forecast, with CC, R2 and MSE/dummy MSE. Exact paired gains, means,
minima, maxima and sign counts are in the tables. Source-loss panels link to
the immutable input snapshot, and the original absolute-score tables remain
there. These four seen recordings were present in source training on earlier
chronological windows. Checkpoints and early stopping were selected using
inner validation. Earlier historical outer-test inspection still limits a
blind-test claim. One fit per configuration cannot estimate variation from
training seeds; the four recording lines expose heterogeneity rather than
formal statistical significance. No new model fit or adaptation is run.
"""


def gain_key(row: dict) -> tuple:
    """Identify one protocol, recording, coordinate group and horizon."""
    return (
        row["variant"], row["session"], row["group"],
        int(row["horizon"]),
    )


def checked_gain_scores(
    rows: list[dict], counts: tuple[int, ...],
) -> dict[tuple, dict]:
    """Validate complete finite per-recording metrics and index them.

    The returned dictionary keys append source count to ``gain_key``.
    Expected rows cover two protocols, four recordings, three groups,
    four horizons and every requested source count.
    """
    indexed = {}
    for row in rows:
        count = int(row["source_count"])
        if count not in counts or row["variant"] not in VARIANTS:
            raise ValueError("Unexpected protocol or source count in scores.")
        key = (*gain_key(row), count)
        if key in indexed:
            raise ValueError(f"Duplicate recording score {key}.")
        if int(row["samples"]) <= 0:
            raise ValueError(f"Empty scored recording {key}.")
        values = [float(row[metric]) for metric in METRICS]
        values += [float(row["mse"]), float(row["dummy_mse"])]
        if not np.isfinite(values).all() or values[-1] <= 0:
            raise ValueError(f"Invalid recording metric or dummy MSE {key}.")
        indexed[key] = row
    expected = len(VARIANTS) * COMMON_SESSIONS
    expected *= len(GROUPS) * len(HORIZONS) * len(counts)
    if len(indexed) != expected:
        raise ValueError(
            f"Expected {expected} complete recording scores, "
            f"got {len(indexed)}."
        )
    return indexed


def paired_gain_rows(
    scores: list[dict], controls: list[dict], reference: str,
) -> list[dict]:
    """Subtract a recording-matched control or four-source score.

    Parameters
    ----------
    scores : list of dict
        Selected multi-session per-recording scores at counts 4/8/16/33.
    controls : list of dict
        One single-session score per recording and protocol.
    reference : str
        ``single_session`` or ``four_source``.

    Returns
    -------
    list of dict
        One positive-is-better gain per metric and saved source score.
    """
    if reference not in GAIN_REFERENCES:
        raise ValueError(f"Unknown paired reference {reference}.")
    source = checked_gain_scores(scores, COUNTS)
    baseline = checked_gain_scores(controls, (1,))
    rows = []
    for key, model in sorted(source.items()):
        variant, session, group, horizon, count = key
        baseline_count = 1 if reference == "single_session" else 4
        paired = (baseline if baseline_count == 1 else source)[
            (variant, session, group, horizon, baseline_count)
        ]
        if (
            int(model["samples"]) != int(paired["samples"])
            or not np.isclose(
                float(model["dummy_mse"]), float(paired["dummy_mse"]),
                rtol=1e-5, atol=1e-6,
            )
        ):
            raise ValueError(
                f"Paired model and reference use different samples: {key}."
            )
        if reference == "four_source" and (
            model["evaluation_set_id"] != paired["evaluation_set_id"]
        ):
            raise ValueError(f"Source test identity changed for {key}.")
        for metric in METRICS:
            value = float(model[metric])
            anchor = float(paired[metric])
            gain = (
                anchor - value if metric == "mse_ratio"
                else value - anchor
            )
            if not np.isfinite(gain):
                raise FloatingPointError(f"Nonfinite paired gain {key}.")
            rows.append(dict(
                reference=reference, variant=variant,
                source_count=count, session=session,
                group=group, horizon=horizon, metric=metric,
                gain=gain, model_score=value, reference_score=anchor,
                samples=int(model["samples"]),
            ))
    return rows


def gain_summary(rows: list[dict]) -> list[dict]:
    """Summarize paired effects across the four visible recordings."""
    grouped = {}
    for row in rows:
        key = tuple(row[field] for field in (
            "reference", "variant", "source_count", "group",
            "horizon", "metric",
        ))
        grouped.setdefault(key, []).append(row)
    summary = []
    for key, values in sorted(grouped.items()):
        sessions = {row["session"] for row in values}
        if len(values) != COMMON_SESSIONS or len(sessions) != COMMON_SESSIONS:
            raise ValueError(f"Expected four paired recordings for {key}.")
        gains = np.asarray([row["gain"] for row in values], dtype=float)
        if not np.isfinite(gains).all():
            raise FloatingPointError(f"Nonfinite paired gains for {key}.")
        summary.append(dict(
            zip(("reference", "variant", "source_count", "group",
                 "horizon", "metric"), key),
            mean_gain=float(gains.mean()),
            minimum_gain=float(gains.min()),
            maximum_gain=float(gains.max()),
            improved_sessions=int(np.sum(gains > SIGN_TOLERANCE)),
            worsened_sessions=int(np.sum(gains < -SIGN_TOLERANCE)),
            near_zero_sessions=int(np.sum(
                np.abs(gains) <= SIGN_TOLERANCE
            )),
        ))
    return summary


def gain_figure(
    rows: list[dict], summary: list[dict], reference: str,
    group: str, metric: str, directory: Path,
) -> Path:
    """Draw four horizon panels with both model families and four sessions."""
    if reference not in GAIN_REFERENCES:
        raise ValueError(f"Unknown gain reference {reference}.")
    names = sorted({row["session"] for row in rows})
    if len(names) != COMMON_SESSIONS:
        raise ValueError("Expected four recording names in gain figure.")
    with plt.rc_context(READABLE_TEXT_STYLE):
        figure, axes = plt.subplots(2, 2, figsize=(16, 11))
        for axis, horizon in zip(axes.flat, HORIZONS):
            extent = [0.0]
            for variant in VARIANTS:
                style = STYLES[variant]
                for index, name in enumerate(names):
                    selected = sorted((row for row in rows if (
                        row["reference"] == reference
                        and row["variant"] == variant
                        and row["session"] == name
                        and row["group"] == group
                        and row["horizon"] == horizon
                        and row["metric"] == metric
                    )), key=lambda row: row["source_count"])
                    if [
                        row["source_count"] for row in selected
                    ] != list(COUNTS):
                        raise ValueError("Incomplete recording gain curve.")
                    values = [row["gain"] for row in selected]
                    extent.extend(values)
                    plot_line(
                        x=COUNTS, y=values, ax=axis,
                        color=style["color"],
                        linestyle=style["linestyle"],
                        marker=GAIN_MARKERS[index],
                        linewidth=1.3, markersize=5, alpha=0.48,
                        label="_nolegend_",
                    )
                selected = sorted((row for row in summary if (
                    row["reference"] == reference
                    and row["variant"] == variant
                    and row["group"] == group
                    and row["horizon"] == horizon
                    and row["metric"] == metric
                )), key=lambda row: row["source_count"])
                if [row["source_count"] for row in selected] != list(COUNTS):
                    raise ValueError("Incomplete four-recording mean curve.")
                values = [row["mean_gain"] for row in selected]
                extent.extend(values)
                plot_line(
                    x=COUNTS, y=values, ax=axis,
                    color=style["color"],
                    linestyle=style["linestyle"],
                    marker=style["marker"], linewidth=3.5,
                    markersize=10, label="_nolegend_",
                )
            data = np.asarray(extent, dtype=float)
            if not np.isfinite(data).all():
                raise FloatingPointError("Nonfinite gain figure values.")
            padding = max(0.01, 0.08 * float(np.ptp(data)))
            axis.set_ylim(float(data.min() - padding),
                          float(data.max() + padding))
            axis.set_xscale("log", base=2)
            axis.set_xlim(min(COUNTS) / 1.08, max(COUNTS) * 1.08)
            axis.set_xticks(COUNTS, labels=[str(x) for x in COUNTS])
            axis.axhline(0, color="#777777", linestyle=":", linewidth=1.5)
            axis.set_title(f"{horizon * 50} ms", pad=11)
            axis.set_xlabel("Source sessions")
            axis.set_ylabel(
                f"Improvement in {decoding.METRIC_LABELS[metric]}"
            )
            axis.grid(alpha=0.15)
        label = (
            "matched single-session BRAID" if reference == "single_session"
            else "the four-source model"
        )
        figure.suptitle(
            fill(f"{group.capitalize()}: gain over {label}", 54),
            fontsize=22, y=0.97,
        )
        handles = [
            Line2D([0], [0], color=STYLES[variant]["color"],
                   marker=STYLES[variant]["marker"],
                   linestyle=STYLES[variant]["linestyle"],
                   linewidth=3.5, markersize=9,
                   label=STYLES[variant]["label"] + " mean")
            for variant in VARIANTS
        ]
        handles.extend(
            Line2D([0], [0], color="#777777", linestyle="-",
                   marker=GAIN_MARKERS[index], linewidth=1.3,
                   markersize=7, label=(
                       f"{name[5:9]}-{name[9:11]}-{name[11:13]}"
                       f" ({name[-2:]})"
                   ))
            for index, name in enumerate(names)
        )
        figure.legend(
            handles=handles, loc="lower center",
            bbox_to_anchor=(0.5, 0.015), ncol=3, frameon=False,
            title="Thick: four-recording mean   Thin: individual recording",
        )
        figure.subplots_adjust(
            left=0.09, right=0.98, top=0.87, bottom=0.20,
            hspace=0.43, wspace=0.29,
        )
        output = directory / f"{group}_{metric}_vs_{reference}.png"
        figure.savefig(output, dpi=DPI, bbox_inches="tight")
        plt.close(figure)
    return output


def publish_gain_views(report: Path) -> Path:
    """Publish complete paired views from saved scores without fitting.

    Input is one checksum-valid matched comparison report with the source
    and single-session per-recording score tables. Output is a sibling
    immutable report with paired tables and 18 performance figures. The four
    source-loss panels are linked to the input report and checksum-pinned.
    """
    marker = checked_report(report)
    identity = dict(
        protocol=GAIN_PROTOCOL, source=report.name,
        source_checksums=marker["checksums"],
        references=list(GAIN_REFERENCES), groups=list(GROUPS),
        horizons=list(HORIZONS), metrics=list(METRICS),
        style="four_recording_curves_and_mean",
        guide=GAIN_GUIDE,
    )
    destination = report.parent / f"{GAIN_PREFIX}-{fingerprint(identity)}"
    with job(destination, identity) as needed:
        if needed:
            scores = read_rows(report / "tables" / "session_scores.csv")
            controls = read_rows(report / "tables" / "baseline_scores.csv")
            rows = [
                row for reference in GAIN_REFERENCES
                for row in paired_gain_rows(scores, controls, reference)
            ]
            summary = gain_summary(rows)
            tables = destination / "tables"
            figures = destination / "figures"
            tables.mkdir()
            figures.mkdir()
            write_table(tables / "paired_recording_gains.csv", rows)
            write_table(tables / "paired_gain_summary.csv", summary)
            for name in (
                "direct_no_n3_source_behavior_relevant.png",
                "direct_no_n3_source_residual.png",
                "with_preprocessing_n3_16_source_behavior_relevant.png",
                "with_preprocessing_n3_16_source_residual.png",
            ):
                target = report / "figures" / name
                if not target.is_file():
                    raise ValueError(f"Missing source-loss panel {target}.")
                (figures / name).symlink_to(target)
            for reference in GAIN_REFERENCES:
                for group in GROUPS:
                    for metric in METRICS:
                        gain_figure(
                            rows, summary, reference, group, metric, figures
                        )
            (destination / "guide.md").write_text(GAIN_GUIDE)
            atomic_json(destination / "inputs.json", dict(
                matched_report=str(report.resolve()),
                matched_checksums=marker["checksums"],
            ))
    checked_report(destination)
    pointer = dict(
        snapshot=str(destination.resolve()),
        figures=str((destination / "figures").resolve()),
        identity=fingerprint(identity),
    )
    root = report.parent.parent
    for name in (
        "latest_paired_gain_comparison.json",
        "latest_matched_baseline_comparison.json",
        "latest_preprocessing_comparison.json",
        "latest_report.json",
        "latest_figures.json",
        "latest_source_figures.json",
        "latest_performance_figures.json",
    ):
        atomic_json(root / name, pointer)
    figure_dir = (destination / "figures").resolve()
    print(f"Saved paired comparison figures: {figure_dir}")
    return destination


def main() -> None:
    """Compare each campaign at its saved best-budget checkpoints."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="saved", choices=(
        "saved", "matched-launch", "matched-audit", "matched-prepare",
        "matched-worker", "matched-coordinator", "matched-report",
        "paired-gain-report",
    ))
    parser.add_argument("--microbatch", type=int, default=4)
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--launch-dir", type=Path)
    parser.add_argument("--matched-report", type=Path)
    for name in (
        "with-plan", "with-source-report", "with-n3-report",
        "direct-plan", "direct-report",
    ):
        parser.add_argument(
            f"--{name}", type=Path
        )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.stage == "paired-gain-report":
        if args.matched_report is None:
            raise ValueError("Paired gains require --matched-report.")
        publish_gain_views(args.matched_report.resolve())
        return
    required = (
        "with_plan", "with_source_report", "direct_plan", "direct_report"
    )
    missing = [key for key in required if getattr(args, key) is None]
    if missing:
        raise ValueError(f"Missing required comparison inputs: {missing}.")
    inputs = {
        key: str(getattr(args, key).resolve()) for key in required
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
