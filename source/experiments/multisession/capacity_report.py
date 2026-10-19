"""Report the Indy source-count by latent-width comparison.

Input is a capacity plan plus completed fit metrics and source histories.
Output contains raw and matched comparison CSV tables, 36 transfer plots
per direction, source training/validation loss, and exposure tables.
"""

from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from experiments.multisession.capacity import COUNTS, LATENTS
from experiments.multisession.scaling_report import (
    GAIN_THRESHOLD_PERCENT,
    METRICS,
    dummy_scores,
    relative_gains,
    selected_exposures,
)
from experiments.multisession.store import completed, fit_directory

COLORS = {
    "scratch": "#242424",
    4: "#8046a5",
    33: "#e28528",
}
SOURCE_LOSS_SMOOTHING_UPDATES = 100


def comparison_rows(rows: list[dict], dummy: dict) -> list[dict]:
    """Pair source-count effects and capacity interactions by target fit."""
    fields = (
        "session", "fold", "seed", "fraction", "latent",
        "target", "horizon", "metric",
    )
    selected = {
        tuple(row[field] for field in fields) + (row["source_count"],):
        row for row in rows if row["metric"] in METRICS
    }
    result = []
    for key, latter in selected.items():
        if key[-1] != COUNTS[1]:
            continue
        first = selected.get(key[:-1] + (COUNTS[0],))
        scratch = selected.get(key[:-1] + (0,))
        if first is None or scratch is None:
            continue
        dummy_value = dummy[key[:5], key[5], key[6], key[7]]
        denominator = (
            scratch["value"] - dummy_value
            if scratch["value"] is not None
            and dummy_value is not None else None
        )
        percent = None
        reason = None
        if scratch["value"] is None or dummy_value is None:
            reason = "nonfinite_reference_metric"
        elif not np.isfinite(denominator) or denominator <= 1e-10:
            reason = "scratch_did_not_beat_dummy"
        elif first["value"] is None or latter["value"] is None:
            reason = "nonfinite_fit_metric"
        else:
            percent = 100 * (
                latter["value"] - first["value"]
            ) / denominator
            if not np.isfinite(percent):
                raise ValueError(f"Nonfinite source-count effect: {key}.")
        result.append(dict(
            session=key[0], fold=key[1], seed=key[2],
            fraction=key[3], latent=key[4], target=key[5],
            horizon=key[6], metric=key[7],
            four_fit=first["fit_id"], thirty_three_fit=latter["fit_id"],
            scratch_fit=scratch["fit_id"],
            four_value=first["value"],
            thirty_three_value=latter["value"],
            scratch_value=scratch["value"],
            dummy_value=dummy_value,
            source_count_effect_percent=percent,
            meaningful_effect=(
                abs(percent) >= GAIN_THRESHOLD_PERCENT
                if percent is not None else None
            ),
            unavailable_reason=reason,
        ))
    effects = {
        (row["session"], row["fold"], row["seed"],
         row["fraction"], row["target"], row["horizon"],
         row["metric"], row["latent"]): row
        for row in result
    }
    for key, row in effects.items():
        baseline = effects.get(key[:-1] + (LATENTS[0],))
        value = row["source_count_effect_percent"]
        reference = (
            baseline["source_count_effect_percent"]
            if baseline is not None else None
        )
        interaction = (
            value - reference
            if value is not None and reference is not None else None
        )
        row["interaction_vs_32_percent"] = interaction
        row["meaningful_interaction"] = (
            abs(interaction) >= GAIN_THRESHOLD_PERCENT
            if interaction is not None else None
        )
    return result


def capacity_figures(
    summary: list[dict], direction: str, root: Path, settings: dict
) -> None:
    """Plot scratch and 4/33-source curves at all horizons and metrics."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for target in ("neural", "behavior"):
        for metric, label in METRICS.items():
            directory = root / "target_transfer" / target / metric
            directory.mkdir(parents=True, exist_ok=True)
            for horizon in settings["evaluation_horizons"]:
                figure, axes = plt.subplots(
                    1, 3, figsize=(15, 4), sharey=True
                )
                for axis, fraction in zip(
                    axes, settings["target_fractions"]
                ):
                    for count in (0, *COUNTS):
                        chosen = {
                            row["latent"]: row
                            for row in summary
                            if row["target"] == target
                            and row["metric"] == metric
                            and row["horizon"] == horizon
                            and row["fraction"] == fraction
                            and row["source_count"] == count
                        }
                        widths = [x for x in LATENTS if x in chosen]
                        if not widths:
                            continue
                        axis.errorbar(
                            widths,
                            [chosen[x]["mean"] for x in widths],
                            yerr=[
                                chosen[x]["sem"] or 0 for x in widths
                            ],
                            color=COLORS["scratch" if count == 0
                                         else count],
                            marker="o",
                            linestyle="--" if count == 0 else "-",
                            capsize=3,
                            label=("Scratch BRAID" if count == 0
                                   else f"{count} Indy sources"),
                        )
                    axis.set_xticks(LATENTS)
                    axis.set_xlabel("n1 = n2 latent width")
                    axis.set_title(f"Target labels: {fraction:.0%}")
                    axis.grid(alpha=0.2)
                axes[0].set_ylabel(label)
                handles, labels = axes[-1].get_legend_handles_labels()
                figure.legend(
                    handles, labels, loc="center left",
                    bbox_to_anchor=(0.83, 0.5),
                )
                figure.suptitle(
                    f"{direction}: {target}, horizon {horizon}"
                )
                figure.tight_layout(rect=(0, 0, 0.82, 0.94))
                figure.savefig(directory / f"h{horizon}.png", dpi=150)
                plt.close(figure)


def source_histories(plan: dict) -> dict:
    """Read finite training and validation losses for completed sources."""
    histories = defaultdict(list)
    for source_id in plan["source_references"]:
        source = fit_directory(plan, source_id)
        if completed(source) is None:
            continue
        spec = plan["jobs"][source_id]
        components = json.loads((source / "components.json").read_text())
        for stage in ("neural", "residual"):
            record = components[stage]
            component = (
                source.parent.parent / "source_components"
                / record["component_id"]
            )
            if completed(component) is None:
                raise ValueError(f"Incomplete source component: {component}")
            train_x, train_y, valid_x, valid_y = [], [], [], []
            with (component / "history.jsonl").open() as stream:
                for line in stream:
                    item = json.loads(line)
                    train_x.append(item["update"])
                    train_y.append(item["loss"])
                    if "validation_loss" in item:
                        valid_x.append(item["update"])
                        valid_y.append(item["validation_loss"])
            training = record["training"]
            if (
                len(train_x) != training["updates"]
                or not valid_x
                or not np.isfinite(train_y).all()
                or not np.isfinite(valid_y).all()
                or valid_x[int(np.argmin(valid_y))]
                != training["best_update"]
            ):
                raise ValueError(f"Invalid source loss history: {component}")
            histories[stage].append(dict(
                source_count=spec["source_count"],
                latent=spec["latent"],
                train_x=np.asarray(train_x),
                train_y=np.asarray(train_y),
                valid_x=valid_x, valid_y=valid_y,
                best_update=training["best_update"],
            ))
    return histories


def source_loss_figures(histories: dict, root: Path) -> None:
    """Plot training and validation loss with legends outside the axes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root.mkdir(parents=True, exist_ok=True)
    for stage, rows in histories.items():
        figure, axis = plt.subplots(figsize=(12, 5))
        for row in sorted(rows, key=lambda x: (
            x["source_count"], x["latent"]
        )):
            color = COLORS[row["source_count"]]
            dash = {32: "-", 64: "--", 128: ":"}[row["latent"]]
            losses = row["train_y"]
            window = min(SOURCE_LOSS_SMOOTHING_UPDATES, len(losses))
            smooth = np.convolve(
                losses, np.ones(window) / window, mode="valid"
            )
            label = (
                f"{row['source_count']} sources, "
                f"width {row['latent']}"
            )
            axis.plot(
                row["train_x"][window - 1:], smooth,
                color=color, linestyle=dash,
                label=f"{label}, train",
            )
            axis.plot(
                row["valid_x"], row["valid_y"],
                color=color, linestyle=dash,
                marker="o", markersize=2,
                label=f"{label}, validation",
            )
        axis.set_xlabel("Training update")
        axis.set_ylabel("Loss")
        axis.set_title(f"Source {stage} training and validation loss")
        axis.grid(alpha=0.2)
        axis.legend(
            loc="upper left", bbox_to_anchor=(1.02, 1),
            borderaxespad=0, fontsize=8,
        )
        figure.tight_layout(rect=(0, 0, 0.72, 1))
        figure.savefig(
            root / f"{stage}_loss.png", dpi=150,
            bbox_inches="tight",
        )
        plt.close(figure)


def write_capacity_payload(
    plan: dict, direction: str, members: list[str],
    rows: list[dict], dimensions: list[dict],
    summary: list[dict], directory: Path
) -> None:
    """Write matched raw, relative, exposure, and figure artifacts."""
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
    dummy = dummy_scores(plan, members)
    gains = relative_gains(rows, dummy)
    for row in gains:
        row["latent"] = plan["jobs"][row["fit_id"]]["latent"]
    if gains:
        write_table(tables / "relative_gain.csv", gains)
    effects = comparison_rows(rows, dummy)
    if effects:
        write_table(tables / "source_count_effect.csv", effects)
    exposure = selected_exposures(plan, members)
    if exposure:
        for row in exposure:
            row["latent"] = plan["jobs"][row["source_fit"]]["latent"]
        write_table(tables / "source_exposure.csv", exposure)
    capacity_figures(summary, direction, figures, plan["settings"])
    source_loss_figures(
        source_histories(plan), figures / "source_training_loss"
    )
