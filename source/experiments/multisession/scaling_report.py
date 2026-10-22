"""Render matched BRAID scaling reports for each adaptation direction.

Inputs are completed fit metrics, prediction arrays, and source histories.
Output tables contain raw CC, R², MSE, dummy-normalized MSE, paired relative
percentages, and selected-checkpoint source exposures. PNG figures plot the
three comparable metrics across source counts for each target fraction.
"""

from collections import defaultdict
import json
import logging
from pathlib import Path

import numpy as np

from experiments.multisession.scaling import COUNTS
from experiments.multisession.store import completed, fit_directory
from experiments.visualization.multisession import curve
from experiments.storage.cache import (
    atomic_json,
    file_digest,
    fingerprint,
)

LOGGER = logging.getLogger(__name__)
METRICS = {
    "cc": "CC",
    "r2": "R²",
    "mse_dummy_improvement": "MSE improvement over dummy (%)",
}
COLORS = {4: "#8046a5", 8: "#2878b5", 16: "#339966", 33: "#e28528"}
GAIN_THRESHOLD_PERCENT = 10.0
DENOMINATOR_EPSILON = 1e-10
SOURCE_LOSS_SMOOTHING_UPDATES = 100


def dummy_scores(
    plan: dict, members: list[str], cache: dict | None = None
) -> dict:
    """Compute the training-label dummy reference for matched scratch fits.

    Parameters
    ----------
    plan : dict
        Campaign plan containing target preprocessing and fit locations.
    members : list[str]
        Matched adaptation and scratch fit identifiers.
    cache : dict, optional
        Previously calculated dummy scores keyed by target selection.

    Returns
    -------
    dict
        Dummy CC, R², and MSE improvement by matched target identity.
    """
    from experiments.multisession.reports import dummy_mse_by_coordinate

    if cache is None:
        cache = {}
    scores = {}
    for key in members:
        spec = plan["jobs"][key]
        if spec["kind"] != "scratch":
            continue
        fit = fit_directory(plan, key)
        if completed(fit) is None:
            raise ValueError(f"Missing matched scratch fit: {fit}.")
        baseline = dummy_mse_by_coordinate(plan, spec, fit, cache)
        matched = (
            spec["session"], spec["fold"], spec["seed"],
            spec["fraction"], spec["latent"],
        )
        for target in ("neural", "behavior"):
            for horizon in spec["horizons"]:
                scores[matched, target, horizon, "cc"] = 0.0
                scores[matched, target, horizon, "r2"] = baseline[
                    target, horizon, "r2"
                ]
                scores[
                    matched, target, horizon, "mse_dummy_improvement"
                ] = 0.0
    return scores


def relative_gains(rows: list[dict], dummy: dict) -> list[dict]:
    """Report signed transfer gains against scratch-to-dummy distance."""
    identity = (
        "session", "fold", "seed", "fraction", "latent",
        "target", "horizon", "metric",
    )
    scratch = {
        tuple(row[field] for field in identity): row
        for row in rows if row["sharing"] == "scratch"
    }
    gains = []
    unavailable = defaultdict(int)
    for row in rows:
        metric = row["metric"]
        if row["sharing"] == "scratch" or metric not in METRICS:
            continue
        key = tuple(row[field] for field in identity)
        control = scratch.get(key)
        if control is None:
            raise ValueError(f"Missing matched scratch metric: {key}.")
        target_key = key[:5], key[5], key[6], metric
        dummy_value = dummy[target_key]
        reference = control["value"]
        value = row["value"]
        reason = None
        if value is None or reference is None:
            reason = "nonfinite_fit_metric"
        elif dummy_value is None:
            reason = "nonfinite_dummy_metric"
        else:
            denominator = reference - dummy_value
            if not np.isfinite(denominator):
                reason = "nonfinite_denominator"
            elif denominator <= DENOMINATOR_EPSILON:
                reason = "scratch_did_not_beat_dummy"
        if reason is not None:
            unavailable[reason] += 1
            percent = None
        else:
            percent = 100.0 * (value - reference) / denominator
            if not np.isfinite(percent):
                raise ValueError(f"Nonfinite relative gain: {key}.")
        gains.append(dict(
            fit_id=row["fit_id"],
            scratch_fit=control["fit_id"],
            session=row["session"], fold=row["fold"],
            seed=row["seed"], fraction=row["fraction"],
            source_count=row["source_count"],
            target=row["target"], horizon=row["horizon"],
            metric=metric, transfer_value=value,
            scratch_value=reference, dummy_value=dummy_value,
            relative_gain_percent=percent,
            meaningful_gain=(
                percent >= GAIN_THRESHOLD_PERCENT
                if percent is not None else None
            ),
            unavailable_reason=reason,
        ))
    if unavailable:
        LOGGER.warning(
            "Undefined scratch-to-dummy gains: %s", dict(unavailable)
        )
    return gains


def selected_exposures(plan: dict, members: list[str]) -> list[dict]:
    """Count sampled windows through each selected source checkpoint."""
    source_ids = {
        plan["jobs"][key]["source_fit"]
        for key in members
        if "source_fit" in plan["jobs"][key]
    }
    result = []
    for source_id in sorted(source_ids):
        fit = fit_directory(plan, source_id)
        if completed(fit) is None:
            continue
        spec = plan["jobs"][source_id]
        records = json.loads((fit / "components.json").read_text())
        for stage, record in records.items():
            component = (
                fit.parent.parent / "source_components"
                / record["component_id"]
            )
            if completed(component) is None:
                raise ValueError(f"Incomplete source component: {component}.")
            training = record["training"]
            best = training["best_update"]
            exposures = np.zeros(spec["source_count"], dtype=np.int64)
            with (component / "history.jsonl").open() as stream:
                for step, line in enumerate(stream, start=1):
                    if step > best:
                        break
                    sampled = json.loads(line)["sampled_windows"]
                    exposures += [len(windows) for windows in sampled]
            expected = best * training["settings"]["batch_size"]
            if int(exposures.sum()) != expected:
                raise ValueError(
                    f"Selected source exposure mismatch: {component}."
                )
            for index, name in enumerate(spec["sessions"]):
                result.append(dict(
                    source_fit=source_id,
                    component=stage,
                    source_count=spec["source_count"],
                    session=name,
                    best_update=best,
                    final_update=training["updates"],
                    selected_windows=int(exposures[index]),
                    final_windows=training["window_exposures"][index],
                    validation_interval=(
                        training["settings"]["validation_interval"]
                    ),
                ))
    return result


def scaling_figures(
    summary: list[dict], direction: str, root: Path, settings: dict
) -> None:
    """Plot all targets, metrics, horizons, and label fractions."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for target in ("neural", "behavior"):
        for metric, label in METRICS.items():
            directory = root / "target_transfer" / target / metric
            directory.mkdir(parents=True, exist_ok=True)
            for horizon in settings["evaluation_horizons"]:
                figure, axes = plt.subplots(
                    1, 3, figsize=(15, 4.2), sharey=True
                )
                any_transfer = False
                for axis, fraction in zip(
                    axes, settings["target_fractions"]
                ):
                    common = dict(
                        target=target, metric=metric,
                        horizon=horizon, fraction=fraction,
                    )
                    control = [
                        row for row in summary
                        if row["sharing"] == "scratch"
                        and all(row[field] == value
                                for field, value in common.items())
                    ]
                    if len(control) != 1:
                        raise ValueError(
                            f"Expected one scratch summary for {common}."
                        )
                    axis.axhline(
                        control[0]["mean"], color="#222222",
                        linestyle="--", label="Scratch BRAID",
                    )
                    any_transfer |= curve(
                        axis, summary,
                        dict(common, sharing="A2_A2_fw_K2_fw"),
                        "source_count", list(COUNTS),
                        "dynamic parameter sharing", "#8046a5",
                    )
                    axis.set_xscale("log", base=2)
                    axis.set_xticks(COUNTS)
                    axis.set_xticklabels([str(count) for count in COUNTS])
                    axis.set_title(f"Target labels: {fraction:.0%}")
                    axis.set_xlabel("Indy source sessions")
                    axis.grid(alpha=0.2)
                    axis.legend(loc="best")
                axes[0].set_ylabel(label)
                figure.suptitle(f"{direction}: {target}, horizon {horizon}")
                figure.tight_layout()
                if any_transfer:
                    figure.savefig(
                        directory / f"h{horizon}.png", dpi=150
                    )
                plt.close(figure)


def source_loss_histories(plan: dict) -> tuple[dict, list[dict]]:
    """Load and validate the saved source training loss histories.

    Parameters
    ----------
    plan : dict
        Completed multi-session source scaling plan.

    Returns
    -------
    tuple[dict, list[dict]]
        Histories keyed by component name and checksums used to identify the
        figure inputs. Each history contains one-dimensional update and loss
        arrays for one source count.
    """
    histories = {"neural": [], "residual": []}
    checksums = []
    for fit_id in plan.get("source_references", []):
        spec = plan["jobs"][fit_id]
        fit = fit_directory(plan, fit_id)
        if completed(fit) is None:
            raise ValueError(f"Incomplete source fit for loss plot: {fit}")
        components = json.loads((fit / "components.json").read_text())
        for stage in histories:
            if stage not in components:
                raise ValueError(
                    f"Source fit {fit_id} has no {stage} component."
                )
            component_id = components[stage]["component_id"]
            component = (
                fit.parent.parent / "source_components" / component_id
            )
            if completed(component) is None:
                raise ValueError(
                    f"Incomplete {stage} source component: {component}"
                )
            history_path = component / "history.jsonl"
            updates = []
            losses = []
            validation_updates = []
            validation_losses = []
            with history_path.open() as stream:
                for line in stream:
                    record = json.loads(line)
                    updates.append(int(record["update"]))
                    losses.append(float(record["loss"]))
                    if "validation_loss" in record:
                        validation_updates.append(int(record["update"]))
                        validation_losses.append(
                            float(record["validation_loss"])
                        )
            update_array = np.asarray(updates, dtype=int)
            loss_array = np.asarray(losses, dtype=float)
            validation_update_array = np.asarray(
                validation_updates, dtype=int
            )
            validation_loss_array = np.asarray(
                validation_losses, dtype=float
            )
            training = components[stage]["training"]
            expected_updates = int(training["updates"])
            best_index = (
                int(np.argmin(validation_loss_array))
                if len(validation_loss_array)
                else -1
            )
            if (
                expected_updates < 1
                or int(training["best_update"]) < 1
                or int(training["best_update"]) > expected_updates
                or len(update_array) != expected_updates
                or not np.array_equal(
                    update_array, np.arange(1, expected_updates + 1)
                )
                or not np.isfinite(loss_array).all()
                or len(validation_update_array) == 0
                or not np.isfinite(validation_loss_array).all()
                or int(validation_update_array[best_index])
                != int(training["best_update"])
                or not np.isclose(
                    validation_loss_array[best_index],
                    float(training["best_loss"]),
                )
            ):
                raise ValueError(
                    f"Invalid {stage} training or validation loss history "
                    f"in {component}; expected {expected_updates} finite "
                    "training updates and a validation minimum matching "
                    "the selected checkpoint."
                )
            histories[stage].append(dict(
                source_count=int(spec["source_count"]),
                updates=update_array,
                losses=loss_array,
                validation_updates=validation_update_array,
                validation_losses=validation_loss_array,
                best_update=int(training["best_update"]),
            ))
            checksums.append(dict(
                source_count=int(spec["source_count"]),
                stage=stage,
                checksum=file_digest(history_path),
            ))
    for stage, values in histories.items():
        counts = [row["source_count"] for row in values]
        if sorted(counts) != sorted(plan["settings"]["source_counts"]):
            raise ValueError(
                f"Expected one {stage} loss history per source count; "
                f"found {counts}."
            )
    return histories, checksums


def plot_source_loss_histories(
    histories: dict, directory: Path
) -> None:
    """Plot training and validation losses by stage and source count."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {"neural": "Stage 1 neural", "residual": "Stage 2 residual"}
    for stage, rows in histories.items():
        figure, axis = plt.subplots(figsize=(12, 5))
        for row in sorted(rows, key=lambda item: item["source_count"]):
            color = COLORS[row["source_count"]]
            updates = row["updates"]
            losses = row["losses"]
            window = min(SOURCE_LOSS_SMOOTHING_UPDATES, len(losses))
            smooth = np.convolve(
                losses, np.ones(window, dtype=float) / window, mode="valid"
            )
            smooth_updates = updates[window - 1:]
            axis.plot(updates, losses, color=color, alpha=0.14)
            axis.plot(
                smooth_updates,
                smooth,
                color=color,
                label=(
                    f"{row['source_count']} sessions, training "
                    f"({SOURCE_LOSS_SMOOTHING_UPDATES}-update mean)"
                ),
            )
            axis.plot(
                row["validation_updates"],
                row["validation_losses"],
                color=color,
                linestyle="--",
                marker="o",
                markersize=3,
                label=f"{row['source_count']} sessions, validation",
            )
            axis.axvline(
                row["best_update"],
                color=color,
                linestyle=":",
                alpha=0.55,
            )
        axis.set_title(f"{labels[stage]} source loss")
        axis.set_xlabel("Training update")
        axis.set_ylabel("Loss")
        axis.grid(alpha=0.2)
        axis.legend(
            title="Indy source sessions",
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0,
            fontsize=8,
        )
        figure.tight_layout(rect=(0, 0, 0.72, 1))
        figure.savefig(
            directory / f"{stage}_loss.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(figure)


def refresh_scaling_figures(plan: dict, plan_path: Path) -> None:
    """Publish figure-only revisions from each final scaling report.

    Parameters
    ----------
    plan : dict
        Completed multi-session source scaling plan.
    plan_path : Path
        Path to the immutable plan JSON.

    Notes
    -----
    Existing report snapshots and their figures are left unchanged. New
    figures are written under each analysis root and linked by a pointer.
    """
    import csv

    from experiments.multisession.plan import analysis_directory
    from experiments.multisession.reports import report_path
    from experiments.multisession.store import job

    renderer = Path(__file__)
    plan_id = plan_path.parent.name
    histories, history_checksums = source_loss_histories(plan)
    expected_names = {
        f"target_transfer/{target}/{metric}/h{horizon}.png"
        for target in ("neural", "behavior")
        for metric in METRICS
        for horizon in plan["settings"]["evaluation_horizons"]
    }
    expected_names.update({
        "source_training/neural_loss.png",
        "source_training/residual_loss.png",
    })
    for direction in plan["analysis_paths"]:
        analysis = analysis_directory(plan, direction, plan_id)
        pointer = analysis / "latest_report.json"
        if not pointer.is_file():
            raise ValueError(
                f"Missing final scaling report pointer: {pointer.resolve()}"
            )
        record = json.loads(pointer.read_text())
        snapshot = report_path(analysis, record)
        if record.get("mode") != "final" or completed(snapshot) is None:
            raise ValueError(f"Scaling report is not final: {analysis}")
        summary_path = snapshot / "tables" / "summary.csv"
        with summary_path.open(newline="") as stream:
            summary = list(csv.DictReader(stream))
        if not summary:
            raise ValueError(f"Final scaling summary is empty: {summary_path}")
        for row in summary:
            row["source_count"] = int(row["source_count"])
            row["horizon"] = int(row["horizon"])
            row["fraction"] = float(row["fraction"])
            row["mean"] = float(row["mean"])
            row["sem"] = float(row["sem"]) if row["sem"] else None
        identity = dict(
            analysis=direction,
            report_checksum=file_digest(summary_path),
            renderer_checksum=file_digest(renderer),
            fractions=plan["settings"]["target_fractions"],
            horizons=plan["settings"]["evaluation_horizons"],
            source_counts=plan["settings"]["source_counts"],
            label="dynamic parameter sharing",
            source_history_checksums=history_checksums,
        )
        directory = (
            analysis / "figures"
            / f"dynamic-sharing-{fingerprint(identity)[:12]}"
        )
        with job(directory, identity) as needed:
            if needed:
                scaling_figures(
                    summary, direction, directory, plan["settings"]
                )
                loss_directory = directory / "source_training"
                loss_directory.mkdir(parents=True)
                plot_source_loss_histories(histories, loss_directory)
                missing = expected_names - {
                    str(path.relative_to(directory))
                    for path in directory.rglob("*.png")
                }
                if missing:
                    raise ValueError(
                        f"Missing scaling figures: {sorted(missing)}"
                    )
        atomic_json(
            analysis / "latest_figures.json",
            dict(
                snapshot=str(directory.relative_to(analysis)),
                source_report=record["snapshot"],
                figure_count=len(expected_names),
                label="dynamic parameter sharing",
            ),
        )


def write_scaling_payload(
    plan: dict, direction: str, members: list[str],
    rows: list[dict], dimensions: list[dict],
    summary: list[dict], directory: Path
) -> None:
    """Write scratch-inclusive scaling tables and figures."""
    from experiments.multisession.reports import (
        cost_tables, session_means, write_table,
    )

    tables = directory / "tables"
    figures = directory / "figures"
    tables.mkdir()
    figures.mkdir()
    write_table(tables / "raw_metrics.csv", rows)
    write_table(tables / "per_dimension.csv", dimensions)
    write_table(tables / "summary.csv", summary)
    write_table(tables / "session_metrics.csv", session_means(rows))
    cost_tables(plan, members, tables)
    gains = relative_gains(rows, dummy_scores(plan, members))
    if gains:
        write_table(tables / "relative_gain.csv", gains)
    exposure = selected_exposures(plan, members)
    if exposure:
        write_table(tables / "source_exposure.csv", exposure)
    scaling_figures(summary, direction, figures, plan["settings"])
