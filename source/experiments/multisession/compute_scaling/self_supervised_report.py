"""Publish complete one-stage source scaling and seen-session BRAID analysis.

Inputs are four completed neural source fits, four completed full BRAID paths,
and a completed direct-behavior reference report. Outputs are a checksum-pinned
report with source losses, inner selections, per-session performances, and ten
figures. The latest pointer changes only after report completion.
"""

import csv
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from BRAID.multisession import HORIZONS
from experiments.multisession.behavior_compute_scaling import (
    behavior_decoding_diagnostics as decoding,
    direct_report,
)
from experiments.multisession.behavior_compute_scaling.ancestry import (
    inherited_paths, restore_component,
)
from experiments.multisession.behavior_compute_scaling.pipeline import (
    inner_source_choices as inner_direct_choices,
    score_component,
)
from experiments.multisession.behavior_compute_scaling.report import (
    source_figures,
)
from experiments.multisession.behavior_compute_scaling.plan import COUNTS
from experiments.multisession.compute_scaling.self_supervised import (
    CAMPAIGN, COMMON, neural_model,
)
from experiments.multisession.behavior_supervision.source_adaptation import (
    full_directory, session_directory,
)
from experiments.multisession.behavior_supervision.models import (
    STAGE2_SHARING,
)
from experiments.multisession.data import session_windows, source_features
from experiments.multisession.plan import analysis_directory
from experiments.multisession.reports import write_table
from experiments.multisession.runner import dataset
from experiments.multisession.store import (
    completed, fit_directory, job,
)
from experiments.multisession.training import recurrent_data
from experiments.storage.cache import atomic_json, file_digest, fingerprint

REPORT_PREFIX = "self-supervised-single-backbone-full-braid"
GROUPS = ("position", "velocity", "neural")
METRICS = ("cc", "r2", "mse_ratio")
VARIANTS = ("self_supervised", "direct_behavior")
GUIDE = """# Single-stage neural pretraining and seen-session BRAID

The source forecaster filters training-normalized neural activity Y with task
input U and predicts Y at 50, 100, 200, and 400 ms. Its A, Afw, and Kfw
maps are shared across source recordings; neural input K and neural output
C are recording-specific. No observed behavior, behavioral normalization,
or behavioral objective enters source pretraining. Each update draws equally
from every recording. Seven branched budgets are fitted per source count.

The single source_neural panel plots raw mean neural chronological test loss
on the same 87 windows from the four common recordings against neural
backbone training FLOPs. Both axes are logarithmic. The selected checkpoint
minimizes raw mean neural loss on those recordings' inner-validation windows;
an exact tie uses lower FLOPs. Selection is frozen before any outer scoring.

From each selected checkpoint, a no-preprocessing three-stage BRAID is fitted
on the four seen recordings. Its behavioral RNN1 starts from the shared
A/Afw/Kfw core and each recording's private neural input K. The recurrent
core stays frozen; private K is fine-tuned and a new behavior decoder C is
trained. Four session paths fit independently for 800 RNN1 updates, then
train private neural readouts with inner-validation early stopping. A shared
RNN2 is fitted for 800 updates after all four paths complete. RNN2 predicts
an additive neural correction. No n0 or n3 is fitted. Adaptation FLOPs are
recorded separately from the source panel's x-axis.

The nine performance figures show the resulting full BRAID forecast of
recorded position, velocity, and neural activity at all four horizons, each
with CC, R2, and MSE divided by the recording's training-mean dummy MSE.
The blue direct-behavior reference reselects saved checkpoints on the same
inner windows using its original joint dummy-normalized behavior/neural
criterion; no direct model is refitted. The gray horizontal line is the
existing single-session direct-model mean. Raw MSE and each recording's
score remain in tables. Values above one are valid MSE/dummy ratios.

The direct reference's outer results were previously inspected and used for
historical checkpoint selection. Retrospective inner reselection cannot
restore a blind historical test. No held-out session adaptation is included.
"""


def read_rows(path: Path) -> list[dict]:
    """Read a nonempty CSV table from an immutable input report."""
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Empty input table: {path.resolve()}.")
    return rows


def checked_sources(plan: dict) -> list[dict]:
    """Validate all selected source fits and seven matched budget paths."""
    sources = []
    for key in plan["source_references"]:
        fit = fit_directory(plan, key)
        if completed(fit) is None:
            raise ValueError(f"Incomplete neural source fit: {fit}.")
        spec = plan["jobs"][key]
        inventory = json.loads((fit / "data.json").read_text())
        selection = json.loads((fit / "selection.json").read_text())
        paths = json.loads((fit / "budgets.json").read_text())
        multiplier = spec["source_training"]["update_multipliers"][
            str(spec["source_count"])
        ]
        expected = [
            value * multiplier
            for value in spec["source_training"]["total_updates"]
        ]
        if [row["total_updates"] for row in paths] != expected:
            raise ValueError("Neural source budget pool differs.")
        chosen = min(
            paths, key=lambda row: (
                row["selection_score"], row["full_path_flops"]
            )
        )
        if chosen["budget"] != selection["selected_budget"]:
            raise ValueError("Saved neural checkpoint selection differs.")
        names = spec["sessions"][:COMMON]
        if selection["selection_set_id"] != fingerprint({
            name: inventory[name]["inner_validation"] for name in names
        }):
            raise ValueError("Inner neural selection set changed.")
        if selection["evaluation_set_id"] != fingerprint({
            name: inventory[name]["validation"] for name in names
        }):
            raise ValueError("Outer neural evaluation set changed.")
        sources.append(dict(
            count=spec["source_count"], spec=spec, fit=fit,
            inventory=inventory, selection=selection, path=chosen,
            paths=paths,
        ))
    sources.sort(key=lambda row: row["count"])
    if [row["count"] for row in sources] != list(COUNTS):
        raise ValueError("Expected four neural source counts.")
    identities = {row["selection"]["evaluation_set_id"] for row in sources}
    if len(identities) != 1:
        raise ValueError("Outer neural test sets differ by source count.")
    return sources


def common_windows(
    plan: dict, sources: list[dict], direct: list[dict],
) -> dict:
    """Rebuild identical train/inner/test arrays with behavior for reporting."""
    first = sources[0]["spec"]
    names = first["sessions"][:COMMON]
    adapter = dataset(plan["settings"])
    adapter.mode = "off"
    common = {}
    for name in names:
        features = source_features(
            adapter.load(name), first["data"],
            first["source_training"]["inner_validation_fraction"],
        )
        train, inner, test, record = session_windows(
            features, True, include_test=True,
        )
        for source in sources:
            saved = source["inventory"][name]
            for field in ("training", "inner_validation", "validation"):
                if saved[field] != record[field]:
                    raise ValueError(
                        f"Neural {field} windows changed for {name}."
                    )
            for field in ("Y", "U"):
                if saved["scales"][field] != record["scales"][field]:
                    raise ValueError(
                        f"Neural {field} scale changed for {name}."
                    )
        for source in direct:
            if source["inventory"][name] != record:
                raise ValueError(
                    f"Direct source split or scale changed for {name}."
                )
        common[name] = dict(training=train, inner=inner, test=test)
    if sum(len(row["test"]["Y"]) for row in common.values()) != 87:
        raise ValueError("Expected 87 fixed outer-test windows.")
    return common


def outer_source_losses(
    sources: list[dict], common: dict,
) -> tuple[list[dict], list[dict]]:
    """Score all seven neural paths after source selection is frozen."""
    losses, sessions = [], []
    for source in sources:
        spec = source["spec"]
        names = spec["sessions"][:COMMON]
        dimensions = [
            (sum(source["inventory"][name]["scales"]["Y"]["keep"]),) * 2
            for name in spec["sessions"]
        ]
        model = neural_model(dimensions, spec)
        root = source["fit"].parent.parent / "source_components"
        training = [
            recurrent_data(common[name]["training"]) for name in names
        ]
        test = [
            recurrent_data(common[name]["test"]) for name in names
        ]
        for path in source["paths"]:
            component = root / path["stages"]["neural"]["component_id"]
            marker = completed(component)
            if marker is None:
                raise ValueError("Incomplete neural source component.")
            restore_component(model, component)
            score = score_component(model, training, test, names)
            losses.append(dict(
                source_count=source["count"],
                source_fit=source["fit"].name,
                budget=path["budget"],
                total_updates=path["total_updates"],
                stage="neural",
                full_path_flops=path["full_path_flops"],
                chronological_source_test_loss=score["mean_loss"],
                evaluation_set_id=source["selection"][
                    "evaluation_set_id"
                ],
                component_id=component.name,
                component_checksum=fingerprint(marker["checksums"]),
                transfer_selected=(
                    path["budget"]
                    == source["selection"]["selected_budget"]
                ),
            ))
            sessions.extend(dict(
                source_count=source["count"],
                budget=path["budget"], stage="neural", **row,
            ) for row in score["per_session"])
        tf.keras.backend.clear_session()
    return losses, sessions


def adapted_sources(plan: dict, sources: list[dict]) -> tuple:
    """Validate full seen-session BRAID fits and score-ready inventories."""
    adapted, completions = [], {}
    for source in sources:
        key = source["key"]
        directory = full_directory(plan, key)
        marker = completed(directory)
        if marker is None:
            raise ValueError(f"Incomplete seen-session BRAID: {directory}.")
        full = json.loads((directory / "path.json").read_text())
        inventory = json.loads((directory / "data.json").read_text())
        names = source["spec"]["sessions"][:COMMON]
        if set(inventory) != set(names):
            raise ValueError("Seen-session BRAID inventory differs.")
        for name in names:
            original = source["inventory"][name]
            for role in ("training", "inner_validation", "validation"):
                if inventory[name][role] != original[role]:
                    raise ValueError("Seen-session BRAID split changed.")
        spec = dict(
            source["spec"], objective="behavior",
            behavior_supervision=dict(target="recorded"),
            stage1_sharing=STAGE2_SHARING, sessions=names,
        )
        adapted.append(dict(
            source, spec=spec, inventory=inventory, path=full,
        ))
        completions[str(source["count"])] = marker["checksums"]
    return adapted, completions


def score_selected(
    sources: list[dict], direct: list[dict], common: dict,
) -> list[dict]:
    """Score both selected variants on matching outer-test samples."""
    scores = []
    for variant, collection in zip(VARIANTS, (sources, direct)):
        for source in collection:
            neural = direct_report.full_neural_forecasts(source, common)
            rows = direct_report.neural_scores(source, common, neural)
            if variant == VARIANTS[1]:
                direct_report.verify_stage_loss(
                    source, rows, "residual", 1
                )
            scores.extend(dict(row, variant=variant) for row in rows)
            behavior = decoding.selected_forecasts(source, common)
            rows = decoding.session_scores(source, common, behavior)
            if variant == VARIANTS[1]:
                direct_report.verify_stage_loss(
                    source, rows, "behavior_relevant", 2
                )
            scores.extend(dict(row, variant=variant) for row in rows)
    expected = len(VARIANTS) * len(COUNTS) * COMMON
    expected *= len(HORIZONS) * len(GROUPS)
    if len(scores) != expected:
        raise ValueError("Incomplete selected source score coverage.")
    lookup = {}
    for row in scores:
        key = (
            row["variant"], row["source_count"], row["session"],
            row["horizon"], row["group"],
        )
        if key in lookup:
            raise ValueError("Duplicate selected source performance row.")
        lookup[key] = row
    for row in scores:
        if row["variant"] != VARIANTS[0]:
            continue
        key = (
            VARIANTS[1], row["source_count"], row["session"],
            row["horizon"], row["group"],
        )
        other = lookup[key]
        if row["samples"] != other["samples"] or (
            not np.isclose(
                row["dummy_mse"], other["dummy_mse"],
                rtol=1e-5, atol=1e-6,
            )
        ):
            raise ValueError("Comparison test samples or dummy loss differ.")
    return scores


def session_training_rows(plan: dict, sources: list[dict]) -> list[dict]:
    """Collect four private stage paths per source count for the report."""
    rows = []
    for source in sources:
        for index in range(COMMON):
            directory = session_directory(plan, source["key"], index)
            record = json.loads((directory / "path.json").read_text())
            expected = source["spec"]["sessions"][index]
            if record["session"] != expected or (
                "source_count" in record
            ):
                raise ValueError("Seen-session training row differs.")
            rows.append(dict(record, source_count=source["count"]))
    return rows


def publish(plan: dict, plan_path: Path) -> Path:
    """Publish one complete immutable analysis after all fits finish."""
    if plan["name"] != CAMPAIGN:
        raise ValueError("Expected neural-only source scaling plan.")
    direct_plan_path = Path(
        plan["settings"]["source_reference_plan"]
    ).resolve()
    direct_report_path = Path(
        plan["settings"]["direct_reference_report"]
    ).resolve()
    direct_plan = json.loads(direct_plan_path.read_text())
    direct_marker = completed(direct_report_path)
    if direct_marker is None:
        raise ValueError("Direct comparison report is incomplete.")
    direct = decoding.checked_sources(direct_plan, direct_report_path)
    for source in direct:
        source["paths"] = inherited_paths(source["fit"])
    sources = checked_sources(plan)
    for source, key in zip(sources, sorted(
        plan["source_references"],
        key=lambda value: plan["jobs"][value]["source_count"],
    )):
        source["key"] = key
        source["plan"] = plan
    common = common_windows(plan, sources, direct)
    adapted, full_completions = adapted_sources(plan, sources)
    sessions = {}
    for source in sources:
        for index in range(COMMON):
            directory = session_directory(plan, source["key"], index)
            marker = completed(directory)
            if marker is None:
                raise ValueError(f"Incomplete seen-session path: {directory}.")
            sessions[f"{source['count']}:{index}"] = marker["checksums"]
    identity = dict(
        group="single_neural_source_and_seen_full_braid_v2",
        source_plan_checksum=file_digest(plan_path),
        source_completions={
            str(source["count"]): completed(source["fit"])["checksums"]
            for source in sources
        },
        direct_plan_checksum=file_digest(direct_plan_path),
        direct_report_checksums=direct_marker["checksums"],
        seen_session_paths=sessions,
        seen_full_paths=full_completions, guide=GUIDE,
    )
    root = analysis_directory(plan, CAMPAIGN, plan_path.parent.name)
    destination = root / "reports" / (
        f"{REPORT_PREFIX}-{fingerprint(identity)}"
    )
    with job(destination, identity) as needed:
        if needed:
            direct_choices, direct_inner = inner_direct_choices(
                direct, common
            )
            source_losses, source_sessions = outer_source_losses(
                sources, common
            )
            scores = score_selected(
                adapted, direct_choices, common
            )
            summaries = []
            for variant in VARIANTS:
                subset = [
                    row for row in scores if row["variant"] == variant
                ]
                summaries.extend(dict(
                    row, variant=variant
                ) for row in direct_report.summarize(subset))
            baseline = [
                dict(row, **{
                    field: float(row[field]) for field in METRICS
                }, horizon=int(row["horizon"]))
                for row in read_rows(
                    direct_report_path / "tables" /
                    "baseline_summary.csv"
                )
            ]
            tables = destination / "tables"
            tables.mkdir()
            write_table(tables / "source_loss.csv", source_losses)
            write_table(
                tables / "source_session_loss.csv", source_sessions
            )
            write_table(tables / "direct_inner_selection.csv", direct_inner)
            write_table(tables / "session_scores.csv", scores)
            write_table(tables / "summary.csv", summaries)
            write_table(tables / "baseline_summary.csv", baseline)
            write_table(tables / "selection.csv", [
                dict(
                    source_count=source["count"],
                    self_budget=source["path"]["budget"],
                    self_total_updates=source["path"]["total_updates"],
                    self_inner_raw_neural_loss=source["path"][
                        "selection_score"
                    ],
                    direct_budget=other["path"]["budget"],
                    direct_total_updates=other["path"]["total_updates"],
                    direct_inner_joint_score=next(
                        row["joint_inner_score"] for row in direct_inner
                        if row["source_count"] == source["count"]
                        and row["budget"] == other["path"]["budget"]
                    ),
                    direct_outer_history=(
                        "Previously inspected; inner reselection "
                        "does not restore blinding."
                    ),
                ) for source, other in zip(sources, direct_choices)
            ])
            write_table(
                tables / "seen_session_training.csv",
                session_training_rows(plan, sources),
            )
            write_table(tables / "seen_full_paths.csv", [
                dict(source_count=source["count"], **json.loads((
                    full_directory(plan, source["key"]) / "path.json"
                ).read_text()))
                for source in sources
            ])
            (destination / "guide.md").write_text(GUIDE)
            atomic_json(destination / "inputs.json", dict(
                plan=str(plan_path.resolve()),
                direct_plan=str(direct_plan_path),
                direct_report=str(direct_report_path),
            ))
            figures = destination / "figures"
            figures.mkdir()
            source_figures(
                source_losses, figures,
                stages=("neural",),
                expected_points=len(
                    plan["settings"]["source_training"]["total_updates"]
                ), direct=True,
                stage_labels={"neural": "Neural backbone forecast"},
                x_label="Full-backbone training FLOPs",
            )
            self_summary = [
                row for row in summaries
                if row["variant"] == VARIANTS[0]
            ]
            direct_summary = [
                row for row in summaries
                if row["variant"] == VARIANTS[1]
            ]
            direct_report.figures(
                self_summary, figures,
                baseline=baseline, comparison=direct_summary,
            )
    if completed(destination) is None:
        raise ValueError("Incomplete neural-only source report.")
    atomic_json(root / "latest_report.json", dict(
        snapshot=str(destination.resolve()),
        figures=str((destination / "figures").resolve()),
        identity=fingerprint(identity),
    ))
    return destination
