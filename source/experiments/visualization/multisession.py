"""Render source/target scaling and exactly paired sensitivity comparisons.

Input summary rows contain condition, target, horizon, session mean and SEM.
Output PNG files separate neural/behavior targets and horizons; primary axes
use source counts or label percentages, sensitivity colors encode conditions.
"""

from pathlib import Path

import numpy as np

from experiments.multisession.plan import PRIMARY, SUPERVISION

COLORS = {
    "A": "#2878b5",
    "AK": "#e28528",
    "AKC": "#339966",
    "scratch": "#222222",
}
DEFAULT_LINE_WIDTH = 2.2


def plot_line(*, x: list, y: list, ax: object, **style: object) -> None:
    """Draw a campaign line with shared width and available styling."""
    style.setdefault("linewidth", DEFAULT_LINE_WIDTH)
    try:
        import seaborn as sns
    except ModuleNotFoundError:
        ax.plot(x, y, **style)
    else:
        sns.lineplot(x=x, y=y, ax=ax, **style)


LABELS = {
    "A": "A1_fw",
    "AK": "A1_fw + K1_fw",
    "AKC": "A1_fw + K1_fw + Cz",
    "scratch": "Scratch BRAID",
}



def curve(
    axis: object,
    summary: list,
    filters: dict,
    parameter: str,
    ticks: list,
    label: str,
    color: str,
    style: str = "-",
) -> bool:
    """Draw one available condition without inventing missing scores."""
    selected = [
        row
        for row in summary
        if all(row.get(k) == v for k, v in filters.items())
    ]
    x, y, errors = [], [], []
    for tick in ticks:
        row = next((r for r in selected if r[parameter] == tick), None)
        if row is not None:
            x.append(tick)
            y.append(row["mean"])
            errors.append(row["sem"])
    if not x:
        return False
    axis.plot(x, y, color=color, linestyle=style, marker="o", label=label)
    valid = [i for i, error in enumerate(errors) if error is not None]
    if valid:
        axis.errorbar(
            np.asarray(x)[valid],
            np.asarray(y)[valid],
            yerr=np.asarray(errors)[valid].astype(float),
            color=color,
            fmt="none",
            capsize=3,
        )
    return True


def figures(
    summary: list[dict], analysis: str, directory: Path, settings: dict
) -> None:
    """Render primary scaling or paired sensitivity CC comparisons."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fractions = settings["target_fractions"]
    counts = settings["source_counts"]
    for target in ("behavior", "neural"):
        for horizon in settings["evaluation_horizons"]:
            common = dict(target=target, horizon=horizon, metric="cc")
            families = (
                ("source", "target")
                if analysis == PRIMARY
                else (
                    ("supervision", "decoder")
                    if analysis == SUPERVISION
                    else ("capacity",)
                )
            )
            for family in families:
                panels = (
                    fractions
                    if family == "source"
                    else counts
                    if family == "target"
                    else ["AK"]
                    if family == "decoder"
                    else ["A", "AK"]
                )
                columns = min(3, len(panels))
                rows = int(np.ceil(len(panels) / columns))
                figure, axes = plt.subplots(
                    rows,
                    columns,
                    squeeze=False,
                    figsize=(5 * columns, 3.8 * rows),
                )
                any_data = False
                for axis, panel in zip(axes.flat, panels):
                    filters = dict(common, latent=32, objective="neural")
                    if family in ("source", "target"):
                        parameter = (
                            "source_count" if family == "source" else "fraction"
                        )
                        ticks = counts if family == "source" else fractions
                        filters[
                            "fraction" if family == "source" else "source_count"
                        ] = panel
                        for sharing in ("A", "AK"):
                            any_data |= curve(
                                axis,
                                summary,
                                dict(filters, sharing=sharing),
                                parameter,
                                ticks,
                                LABELS[sharing],
                                COLORS[sharing],
                            )
                        if family == "source":
                            scratch = [
                                r
                                for r in summary
                                if all(
                                    r[k] == v
                                    for k, v in dict(
                                        common,
                                        latent=32,
                                        fraction=panel,
                                        sharing="scratch",
                                    ).items()
                                )
                            ]
                            if scratch:
                                axis.axhline(
                                    scratch[0]["mean"],
                                    color="black",
                                    linestyle="--",
                                    label="Scratch BRAID",
                                )
                                any_data = True
                        else:
                            any_data |= curve(
                                axis,
                                summary,
                                dict(common, latent=32, sharing="scratch"),
                                parameter,
                                ticks,
                                "Scratch BRAID",
                                "black",
                                "--",
                            )
                        axis.set_title(
                            f"Labels: {panel:.0%}"
                            if family == "source"
                            else f"Source sessions: {panel}"
                        )
                    else:
                        parameter, ticks = "fraction", fractions
                        filters.update(source_count=max(counts), sharing=panel)
                        if family == "capacity":
                            comparisons = [
                                (
                                    dict(filters, latent=n),
                                    f"{n} → {n}+{n}",
                                    color,
                                )
                                for n, color in (
                                    (32, "#2878b5"),
                                    (64, "#e28528"),
                                )
                            ]
                        elif family == "supervision":
                            comparisons = [
                                (
                                    dict(filters, objective=objective),
                                    label,
                                    color,
                                )
                                for objective, label, color in (
                                    ("neural", "Neural-only", "#2878b5"),
                                    (
                                        "behavior",
                                        "Behavior-supervised",
                                        "#e28528",
                                    ),
                                )
                            ]
                        else:
                            comparisons = [
                                (
                                    dict(
                                        filters, objective="behavior", sharing=s
                                    ),
                                    LABELS[s],
                                    COLORS[s],
                                )
                                for s in ("AK", "AKC")
                            ]
                        for where, label, color in comparisons:
                            any_data |= curve(
                                axis,
                                summary,
                                where,
                                parameter,
                                ticks,
                                label,
                                color,
                            )
                        if family == "capacity":
                            for latent, color in (
                                (32, "#2878b5"),
                                (64, "#e28528"),
                            ):
                                any_data |= curve(
                                    axis,
                                    summary,
                                    dict(
                                        common,
                                        latent=latent,
                                        sharing="scratch",
                                    ),
                                    parameter,
                                    ticks,
                                    f"Scratch {latent}+{latent}",
                                    color,
                                    "--",
                                )
                        else:
                            any_data |= curve(
                                axis,
                                summary,
                                dict(
                                    common,
                                    latent=32,
                                    sharing="scratch",
                                ),
                                parameter,
                                ticks,
                                LABELS["scratch"],
                                COLORS["scratch"],
                                "--",
                            )
                        axis.set_title(
                            LABELS[panel]
                            if family != "decoder"
                            else "Behavior-supervised decoder sharing"
                        )
                    axis.set_xscale("log", base=2)
                    axis.set_xticks(ticks)
                    axis.set_xticklabels(
                        [
                            str(t)
                            if parameter == "source_count"
                            else f"{100 * t:g}"
                            for t in ticks
                        ]
                    )
                    axis.set_xlabel(
                        "Source sessions"
                        if parameter == "source_count"
                        else "Target labels (%)"
                    )
                    axis.set_ylabel("CC")
                    axis.grid(alpha=0.2)
                for axis in list(axes.flat)[len(panels) :]:
                    axis.set_visible(False)
                handles = {}
                for axis in axes.flat:
                    h, lab = axis.get_legend_handles_labels()
                    for handle, label in zip(h, lab):
                        handles[label] = handle
                if any_data:
                    figure.suptitle(
                        f"{target.capitalize()} · horizon {horizon}"
                    )
                    legend_columns = (
                        len(handles)
                        if analysis == PRIMARY
                        else 1 if family == "decoder"
                        else min(2, len(handles))
                    )
                    figure.legend(
                        handles.values(),
                        handles.keys(),
                        loc="lower center",
                        ncol=legend_columns,
                    )
                    bottom = (
                        0.08 if analysis == PRIMARY
                        else 0.23 if family == "decoder"
                        else 0.16
                    )
                    figure.tight_layout(rect=(0, bottom, 1, 0.95))
                    figure.savefig(
                        directory / f"{family}_{target}_h{horizon}.png", dpi=150
                    )
                plt.close(figure)
