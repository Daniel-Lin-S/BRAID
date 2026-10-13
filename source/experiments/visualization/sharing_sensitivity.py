"""Render nested target-transfer and source-validation sensitivity figures.

Inputs are summary rows, source-validation rows and the analysis identifier.
Outputs are PNG figures under figures/target_transfer/<target>/ and
figures/source_validation/. Tables remain separate under tables/.
"""

from pathlib import Path

import numpy as np

from experiments.multisession.sensitivity import FILTER, RESIDUAL
from experiments.visualization.multisession import curve

COLORS = {
    "AK": "#2878b5",
    "AAK": "#e28528",
    "AAK2": "#339966",
    "A": "#8046a5",
    "scratch": "#222222",
}
LABELS = {
    "A": "A1_fw",
    "AK": "A1_fw + K1_fw",
    "AAK": "A1 + A1_fw + K1_fw",
    "AAK2": "Stage 1 + shared RNN2 dynamics",
    "scratch": "Scratch BRAID",
}
HORIZONS = (1, 2, 4, 8)
SOURCE_COUNTS = (4, 8, 16, 33)


def _axis(axis: object, ticks: list, x_label: str, y_label: str) -> None:
    """Apply consistent logarithmic positioning and readable tick labels."""
    axis.set_xscale("log", base=2)
    axis.set_xticks(ticks)
    axis.set_xticklabels([str(value) for value in ticks])
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.grid(alpha=0.2)


def target_figures(
    summary: list[dict], analysis: str, root: Path, settings: dict
) -> None:
    """Compare exactly two transfers with a matched scratch reference."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    profiles = ("AK", "AAK") if analysis == FILTER else ("AAK", "AAK2")
    fractions = settings["target_fractions"]
    for target in ("neural", "behavior"):
        directory = root / "target_transfer" / target
        directory.mkdir(parents=True, exist_ok=True)
        for horizon in settings["evaluation_horizons"]:
            figure, axis = plt.subplots(figsize=(7.5, 4.8))
            common = dict(target=target, metric="cc", horizon=horizon)
            for profile in profiles:
                curve(
                    axis, summary,
                    dict(common, sharing=profile),
                    "fraction", fractions, LABELS[profile], COLORS[profile],
                )
            curve(
                axis, summary, dict(common, sharing="scratch"),
                "fraction", fractions, LABELS["scratch"],
                COLORS["scratch"], "--",
            )
            _axis(
                axis, fractions, "Target labels (%)", "CC",
            )
            axis.set_xticklabels([f"{100 * x:g}" for x in fractions])
            axis.set_title(f"{target.capitalize()} · horizon {horizon}")
            axis.legend(loc="best")
            figure.tight_layout()
            figure.savefig(directory / f"h{horizon}.png", dpi=150)
            plt.close(figure)


def source_figures(
    metrics: list[dict], losses: list[dict], root: Path
) -> None:
    """Show common-session CC and all-session source loss scaling."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root.mkdir(parents=True, exist_ok=True)
    for horizon in HORIZONS:
        figure, axis = plt.subplots(figsize=(7.5, 4.8))
        for profile in ("A", "AK", "AAK", "AAK2"):
            by_count = []
            for count in SOURCE_COUNTS:
                values = [
                    row["cc"] for row in metrics
                    if row["sharing"] == profile
                    and row["source_count"] == count
                    and row["horizon"] == horizon
                    and row["cc"] is not None
                ]
                if values:
                    by_count.append((count, float(np.mean(values))))
            if by_count:
                x, y = zip(*by_count)
                axis.plot(
                    x, y, marker="o", color=COLORS[profile],
                    label=LABELS[profile],
                )
                errors = []
                for count in x:
                    values = [
                        row["cc"] for row in metrics
                        if row["sharing"] == profile
                        and row["source_count"] == count
                        and row["horizon"] == horizon
                        and row["cc"] is not None
                    ]
                    errors.append(
                        float(np.std(values, ddof=1) / np.sqrt(len(values)))
                        if len(values) > 1 else 0.0
                    )
                axis.errorbar(
                    x, y, yerr=errors, color=COLORS[profile],
                    fmt="none", capsize=3,
                )
        _axis(axis, list(SOURCE_COUNTS), "Source sessions", "Neural CC")
        axis.set_title(
            f"Same four source validation sessions · horizon {horizon}"
        )
        axis.legend(loc="best")
        figure.tight_layout()
        figure.savefig(root / f"common_four_neural_h{horizon}.png", dpi=150)
        plt.close(figure)
    figure, axis = plt.subplots(figsize=(7.5, 4.8))
    for profile in ("A", "AK", "AAK", "AAK2"):
        selected = sorted(
            (
                row["source_count"], row["macro_validation_loss"]
            )
            for row in losses if row["sharing"] == profile
        )
        if selected:
            x, y = zip(*selected)
            axis.plot(
                x, y, marker="o", color=COLORS[profile],
                label=LABELS[profile],
            )
    _axis(
        axis, list(SOURCE_COUNTS),
        "Source sessions", "Best macro-validation loss",
    )
    axis.set_title("All source sessions · sum of horizon MSE")
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(root / "macro_loss.png", dpi=150)
    plt.close(figure)
