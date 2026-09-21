"""Render global summaries and in-memory per-session fold comparisons.

Outputs are named target_metric_vs_parameter_selection.png figures. Failed
observations remain annotated gaps; pending comparisons are deferred. This
module does not load checkpoints, fit models or produce individual-fold plots.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .cache import fingerprint
from .presentation import METRIC_LABELS, presentation, save_figure, style_axis

LOGGER = logging.getLogger(__name__)
READINESS_FIELDS = {
    "pending", "expected_count", "completed_count", "failed_members",
}
FAILURE_MARKER_HEIGHT = 0.03
FAILURE_LABEL_OFFSET = (0, 12)
MISSING_LABEL_OFFSET = (0, 10)
PARAMETER_LABELS = {
    "horizon": "Forecast horizon (steps)",
    "nx": "Latent dimension (nx)",
    "population_scale": "Neural population (%)",
}


def curve_specs(settings: dict) -> list[dict]:
    """Expand configured comparison designs into separate metric figures."""
    result = []
    for curve in settings.get("curves", []):
        for metric in settings.get("metrics", [curve["where"]["metric"]]):
            if metric not in ("cc", "r2", "mse"):
                raise ValueError(f"Unsupported comparison metric: {metric}")
            where = dict(curve["where"], metric=metric)
            parameter = curve["parameter"]
            selection = (
                f"nx{where['nx']}"
                if parameter == "horizon"
                else f"horizon{where['horizon']}"
            )
            axis = (
                "population" if parameter == "population_scale" else parameter
            )
            name = (
                f"{where['target']}_{metric}_vs_{axis}_{selection}_"
                f"{where['evaluation_set']}"
            )
            result.append(dict(curve, where=where, name=name))
    return result


def matches(row: dict, where: dict) -> bool:
    """Match scalar and multi-value selections without dropping missing rows."""
    return all(
        row[key] in value if isinstance(value, list) else row[key] == value
        for key, value in where.items()
    )


def metric_curve(
    rows: list[dict],
    spec: dict,
    style: dict,
    partial: bool = False,
) -> object:
    """Build one metric figure from means, uncertainties and x values."""
    if not rows or not any(
        row["mean"] is not None and np.isfinite(row["mean"]) for row in rows
    ):
        raise ValueError(f"No finite results for {spec['name']}.")
    parameter, group = spec["parameter"], spec.get("group")
    metric, target = spec["where"]["metric"], spec["where"]["target"]
    figure, axis = plt.subplots(figsize=style["single_size"])
    groups = sorted({r[group] for r in rows}) if group else [None]
    for number, value in enumerate(groups):
        selected = sorted(
            [r for r in rows if not group or r[group] == value],
            key=lambda row: row[parameter],
        )
        x = np.array([row[parameter] for row in selected], dtype=float)
        if parameter == "population_scale":
            x *= 100
        y = np.array(
            [
                np.nan if row["mean"] is None else row["mean"]
                for row in selected
            ],
            dtype=float,
        )
        if not np.isfinite(y).all():
            y[~np.isfinite(y)] = np.nan
        color_index = {16: 0, 64: 1}.get(value, number)
        color = style["pair_colors"][color_index % len(style["pair_colors"])]
        axis.plot(
            x,
            y,
            "o-",
            label=f"{group}={value}" if group else "BRAID",
            color=color,
        )
        for index, row in enumerate(selected):
            failures = row.get("failed_members", [])
            if not failures:
                continue
            label = (
                f"{row['completed_count']}/{row['expected_count']} results"
                f"\n{len(failures)} failed"
            )
            if np.isfinite(y[index]):
                axis.annotate(
                    label, (x[index], y[index]), xytext=FAILURE_LABEL_OFFSET,
                    textcoords="offset points", ha="center", color=color,
                )
            else:
                transform = axis.get_xaxis_transform()
                axis.plot(
                    [x[index]], [FAILURE_MARKER_HEIGHT], "x", color=color,
                    transform=transform,
                )
                axis.annotate(
                    label, (x[index], FAILURE_MARKER_HEIGHT),
                    xycoords=transform, xytext=MISSING_LABEL_OFFSET,
                    textcoords="offset points", ha="center", color=color,
                )
        uncertainty = "std" if any("std" in row for row in selected) else "sem"
        valid_uncertainty = [
            i
            for i, row in enumerate(selected)
            if np.isfinite(y[i])
            and row.get(uncertainty) is not None
            and np.isfinite(row[uncertainty])
        ]
        if valid_uncertainty:
            axis.errorbar(
                x[valid_uncertainty],
                y[valid_uncertainty],
                yerr=[selected[i][uncertainty] for i in valid_uncertainty],
                fmt="none",
                color=color,
                capsize=4,
            )
    ticks = sorted({row[parameter] for row in rows})
    if parameter == "nx":
        axis.set_xscale("log", base=2)
    elif parameter == "population_scale":
        ticks = [100 * value for value in ticks]
    axis.set_xticks(ticks, labels=[f"{value:g}" for value in ticks])
    axis.set(
        xlabel=PARAMETER_LABELS[parameter],
        ylabel=f"{target.capitalize()} {METRIC_LABELS[metric]}",
    )
    style_axis(axis, style)
    fixed = (
        f"nx={spec['where']['nx']}"
        if parameter == "horizon"
        else f"horizon={spec['where']['horizon']}"
    )
    title = (
        f"{target.capitalize()} {METRIC_LABELS[metric]}\n"
        f"{fixed}; {spec['where']['evaluation_set']} scoring"
    )
    context_label = spec.get("context_label")
    if context_label:
        title += f"\n{context_label}"
    figure.suptitle(
        title + ("\n(partial: failed experiments)" if partial else ""),
        fontsize=style["title_font"],
        wrap=True,
    )
    axis.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        frameon=False,
        fontsize=style["legend_font"],
    )
    figure.tight_layout()
    return figure


def comparison_rows(
    chosen: list[dict], dependencies: list[dict], name: str,
) -> list[dict]:
    """Attach failure counts to means and reject unexplained missing metrics.

    Parameters
    ----------
    chosen : list of dict
        Available aggregate metric rows for one figure.
    dependencies : list of dict
        Terminal expected points with contribution counts and failure keys.
    name : str
        Figure identifier used in validation errors.

    Returns
    -------
    list of dict
        Available means plus explicit gaps caused by failed experiments.
    """
    result = [dict(row) for row in chosen]
    for candidate in dependencies:
        criteria = {
            k: v for k, v in candidate.items() if k not in READINESS_FIELDS
        }
        row = next((
            row for row in result
            if all(row[k] == v for k, v in criteria.items())
        ), None)
        if row is None:
            if not candidate.get("failed_members"):
                raise ValueError(
                    f"Missing completed result in {name}: {criteria}"
                )
            row = dict(criteria, mean=None, sem=None, std=None)
            result.append(row)
        row.update({
            k: candidate[k] for k in READINESS_FIELDS if k in candidate
        })
    for row in result:
        if row["mean"] is None or not np.isfinite(row["mean"]):
            if not row.get("failed_members") or row.get("completed_count", 0):
                raise ValueError(f"Undefined completed metric in {name}: {row}")
    return result


def _comparison_signature(rows: list[dict], dependencies: list[dict]) -> str:
    """Fingerprint numerical values and terminal dependency states."""
    return fingerprint(dict(
        rows=sorted(rows, key=repr),
        dependencies=sorted(dependencies, key=repr),
    ))


def _current_figure(path: Path, signature: str) -> bool:
    """Return whether a readable PNG represents the current comparison."""
    try:
        with Image.open(path) as saved:
            recorded = saved.info.get("BRAID-rendering")
            saved.verify()
    except OSError:
        LOGGER.warning("Redrawing damaged comparison PNG: %s", path)
        return False
    return recorded == signature


def plot_suite(
    summaries: list[dict],
    root: Path,
    settings: dict,
    expected: list[dict] | None = None,
    pending: bool = False,
    rendered: set[str] | None = None,
    namespace: str | None = None,
    regenerate: bool = False,
) -> None:
    """Publish terminal comparisons, annotating only failed contributions.

    Parameters
    ----------
    summaries : list of dict
        Aggregate metric means and uncertainty values.
    root : Path
        Destination for comparison PNG files.
    settings : dict
        Enabled flag, curve selections and presentation preferences.
    expected : list of dict, optional
        Per-point dependencies and readiness; default None.
    pending : bool, optional
        Defer without explicit dependencies when True; default False.
    rendered : set of str, optional
        Qualified figure names already handled in this invocation; default None.
    namespace : str, optional
        Figure namespace used for session-specific rendering; default None.
    regenerate : bool, optional
        Replace all figures; default False repairs missing, damaged or stale
        figures while preserving current ones.
    """
    if not settings["enabled"]:
        return
    style = presentation(settings.get("presentation"))
    errors = []
    warning_details = []
    warning_members = set()
    for configured_spec in curve_specs(settings):
        spec = dict(configured_spec)
        name = spec["name"]
        render_key = f"{namespace}/{name}" if namespace else name
        path = root / f"{name}.png"
        if rendered is not None and render_key in rendered:
            continue
        dependencies = [
            row for row in expected or [] if matches(row, spec["where"])
        ]
        if any(row.get("pending", False) for row in dependencies) or (
            pending and expected is None
        ):
            continue
        chosen = [row for row in summaries if matches(row, spec["where"])]
        try:
            rows = comparison_rows(chosen, dependencies, name)
            signature = _comparison_signature(rows, dependencies)
            if path.exists() and not regenerate and _current_figure(
                path, signature
            ):
                if rendered is not None:
                    rendered.add(render_key)
                continue
            if rendered is not None:
                rendered.add(render_key)
            failures = sorted({
                key for row in dependencies
                for key in row.get("failed_members", [])
            })
            if failures:
                counts = ", ".join(
                    f"{row[spec['parameter']]}="
                    f"{row['completed_count']}/{row['expected_count']}"
                    for row in dependencies if row.get("failed_members")
                )
                warning_details.append(f"{name} ({counts})")
                warning_members.update(failures)
                if not any(
                    row["mean"] is not None and np.isfinite(row["mean"])
                    for row in rows
                ):
                    continue
            if namespace and namespace.startswith("session/"):
                spec["context_label"] = namespace.removeprefix("session/")
            figure = metric_curve(rows, spec, style, partial=bool(failures))
            save_figure(figure, path, style, signature)
        except Exception as error:
            LOGGER.exception("Comparison failed: %s", render_key)
            errors.append(str(error))
    if warning_members:
        LOGGER.warning(
            "Incomplete comparisons %s; failed members: %s",
            "; ".join(warning_details),
            ", ".join(sorted(warning_members)),
        )
    if errors:
        raise RuntimeError("Comparison rendering failed: " + "; ".join(errors))
