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

LOGGER = logging.getLogger(__name__)
METRICS = {
    "cc": "CC",
    "r2": "R²",
    "mse_dummy_improvement": "MSE improvement over dummy (%)",
}
COLORS = {4: "#8046a5", 8: "#2878b5", 16: "#339966", 33: "#e28528"}
GAIN_THRESHOLD_PERCENT = 10.0
DENOMINATOR_EPSILON = 1e-10


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
                        "A1_fw + all RNN2 dynamics", "#8046a5",
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
