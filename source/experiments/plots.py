"""Render global summaries and in-memory per-session fold comparisons.

Outputs are named target_metric_vs_parameter_selection.png figures. Failed
observations are reported through warnings; pending comparisons are deferred.
module does not load checkpoints, fit models or produce individual-fold plots.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .cache import fingerprint
from .presentation import (
    METRIC_LABELS, outlier_rendering, presentation, save_figure,
    style_axis,
)

LOGGER = logging.getLogger(__name__)
READINESS_FIELDS = {
    "pending", "expected_count", "completed_count", "completed_members",
    "failed_members",
}
RESIDUAL_BRANCH = "residual"
MAIN_ONLY_STYLE = ("-", "o")
RESIDUAL_STYLE = ("--", "s")
GROUP_LINESTYLES = ("-", "--", "-.", ":")
GROUP_MARKERS = ("o", "s", "D", "^", "v", "P", "X", "*")
OUTLIER_LABEL_OFFSET_POINTS = 8
OUTLIER_LABEL_STACK_POINTS = 14
PARAMETER_LABELS = {
    "horizon": "Forecast horizon (steps)",
    "nx": "Latent dimension (nx)",
    "population_scale": "Neural population (%)",
}


class PlotRenderingError(RuntimeError):
    """Report renderer-local failures after all eligible figures run."""


def curve_specs(settings: dict) -> list[dict]:
    """Expand configured comparison designs into separate metric figures."""
    result = []
    for curve in settings.get("curves", []):
        for metric in settings.get("metrics", [curve["where"]["metric"]]):
            if metric not in ("cc", "r2", "mse"):
                raise ValueError(f"Unsupported comparison metric: {metric}")
            where = dict(curve["where"], metric=metric)
            parameter = curve["parameter"]
            group = curve.get("group")
            panel = curve.get("panel")
            branch = curve.get("branch")
            if branch is not None and branch != RESIDUAL_BRANCH:
                raise ValueError(f"Unsupported comparison branch: {branch!r}")
            if branch is not None and not group:
                raise ValueError("A branched comparison requires a group.")
            if panel:
                if not group:
                    raise ValueError("A panel comparison requires a group.")
                if not isinstance(where.get(panel), list):
                    raise ValueError(
                        f"Panel field {panel!r} requires a list selection."
                    )
                name = f"{metric}_vs_{parameter}_by_{group}"
                result.append(dict(curve, where=where, name=name))
                continue
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
    rows: list[dict], spec: dict, style: dict,
    outlier_style: dict | None = None,
) -> object:
    """Build one metric figure from means, uncertainties and x values."""
    if not rows or not any(
        row["mean"] is not None and np.isfinite(row["mean"]) for row in rows
    ):
        raise ValueError(f"No finite results for {spec['name']}.")
    metric = spec["where"]["metric"]
    outlier_style = outlier_rendering(
        outlier_style or style.get("outlier_rendering")
    )
    panel = spec.get("panel")
    panels = spec["where"][panel] if panel else [None]
    figure, axes = plt.subplots(
        1,
        len(panels),
        figsize=(style["horizon_size"] if panel else style["single_size"]),
        squeeze=False,
    )
    for axis, panel_value in zip(axes[0], panels):
        panel_rows = (
            [row for row in rows if row[panel] == panel_value]
            if panel else rows
        )
        _metric_axis(
            axis, panel_rows, spec, style, panel_value, outlier_style
        )
    title = _metric_title(spec, metric)
    context_label = spec.get("context_label")
    if context_label:
        title += f"\n{context_label}"
    figure.suptitle(title, fontsize=style["title_font"], wrap=True)
    if panel:
        handles, labels = axes[0][0].get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(0.99, 0.5),
            frameon=False,
            fontsize=style["legend_font"],
        )
    else:
        axes[0][0].legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            frameon=False,
            fontsize=style["legend_font"],
        )
    figure.tight_layout()
    return figure


def _metric_axis(
    axis: object,
    rows: list[dict],
    spec: dict,
    style: dict,
    panel_value: object,
    outlier_style: dict,
) -> None:
    """Render grouped metric series for one target panel."""
    parameter = spec["parameter"]
    metric = spec["where"]["metric"]
    for value, branch, selected, number in _comparison_series(rows, spec):
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
        color, linestyle, marker = _group_style(
            spec, value, number, style, branch
        )
        axis.plot(
            x,
            y,
            marker=marker,
            linestyle=linestyle,
            label=_series_label(spec, value, branch, selected),
            color=color,
        )
        _plot_uncertainty(axis, selected, x, y, color)
    ticks = sorted({row[parameter] for row in rows})
    if parameter == "nx":
        axis.set_xscale("log", base=2)
    elif parameter == "population_scale":
        ticks = [100 * value for value in ticks]
    axis.set_xticks(ticks, labels=[f"{value:g}" for value in ticks])
    axis.set(
        xlabel=PARAMETER_LABELS[parameter],
        ylabel=(
            METRIC_LABELS[metric]
            if panel_value is not None
            else f"{spec['where']['target'].capitalize()} "
            f"{METRIC_LABELS[metric]}"
        ),
    )
    if panel_value is not None:
        axis.set_title(str(panel_value).capitalize())
    _render_outlier_arrows(axis, rows, parameter, outlier_style)
    style_axis(axis, style)


def _render_outlier_arrows(
    axis: object, rows: list[dict], parameter: str, style: dict,
) -> None:
    """Zoom to normal summaries and mark selected finite values at an edge."""
    events = [
        (row, event)
        for row in rows
        for event in row.get("outliers", [])
    ]
    if not events:
        return
    bounds = []
    for row in rows:
        mean = row.get("mean")
        if mean is None or not np.isfinite(mean):
            continue
        uncertainty = row.get("std", row.get("sem"))
        if uncertainty is None or not np.isfinite(uncertainty):
            uncertainty = 0.0
        bounds.extend((mean - uncertainty, mean + uncertainty))
    if not bounds:
        raise ValueError(
            "Cannot render configured outlier arrows without finite "
            "normal observations."
        )
    lower, upper = min(bounds), max(bounds)
    span = upper - lower
    if not np.isfinite(span) or span <= 0:
        raise ValueError(
            "Cannot render configured outlier arrows with a zero normal "
            "metric range."
        )
    padding = span * style["zoom_padding_fraction"]
    lower_edge, upper_edge = lower - padding, upper + padding
    axis.set_ylim(lower_edge, upper_edge)
    digits = style["label_significant_figures"]
    arrows = []
    for row, event in events:
        value = event["value"]
        x = row[parameter] * (100 if parameter == "population_scale" else 1)
        if value < lower:
            direction, edge = "↓", lower_edge
        elif value > upper:
            direction, edge = "↑", upper_edge
        else:
            raise ValueError(
                "Configured outlier is inside the normal display range: "
                f"{event['member']} value={value!r}."
            )
        arrows.append((x, direction, edge, row, event))
    arrows.sort(key=lambda item: (
        item[0], item[1], item[3].get("nx", -1), item[4]["member"],
    ))
    offsets = {}
    for x, direction, edge, _row, event in arrows:
        key = (x, direction)
        offset = offsets.get(key, 0)
        offsets[key] = offset + 1
        sign = 1 if direction == "↓" else -1
        axis.annotate(
            f"{direction} {event['value']:.{digits}g}",
            xy=(x, edge),
            xytext=(0, sign * (
                OUTLIER_LABEL_OFFSET_POINTS
                + offset * OUTLIER_LABEL_STACK_POINTS
            )),
            textcoords="offset points",
            ha="center",
            va="bottom" if direction == "↓" else "top",
            arrowprops=dict(arrowstyle="->", color="black"),
            annotation_clip=False,
        )


def _comparison_series(
    rows: list[dict],
    spec: dict,
) -> list[tuple[object, bool | None, list[dict], int]]:
    """Return deterministic comparison series, optionally split by residuals."""
    group = spec.get("group")
    values = sorted({row[group] for row in rows}) if group else [None]
    if spec.get("branch") is None:
        return [
            (
                value,
                None,
                sorted(
                    [row for row in rows if not group or row[group] == value],
                    key=lambda row: row[spec["parameter"]],
                ),
                number,
            )
            for number, value in enumerate(values)
        ]
    result = []
    for number, value in enumerate(values):
        selected = [
            row for row in rows if not group or row[group] == value
        ]
        main_only, residual = _residual_branches(selected, spec, value)
        if main_only:
            result.append((value, False, main_only, number))
        if residual:
            if spec["parameter"] == "nx":
                residual = _add_shared_predecessor(main_only, residual)
            result.append((value, True, residual, number))
    return result


def _residual_branches(
    rows: list[dict],
    spec: dict,
    group_value: object,
) -> tuple[list[dict], list[dict]]:
    """Classify rows by n2 and reject ambiguous same-nx branch members."""
    grouped = {False: {}, True: {}}
    group = spec.get("group")
    parameter = spec["parameter"]
    for row in rows:
        if "n1" not in row or "n2" not in row:
            raise ValueError(
                "Residual comparison requires n1 and n2 metric metadata."
            )
        n1, n2, nx = row["n1"], row["n2"], row["nx"]
        if (
            isinstance(n1, bool)
            or isinstance(n2, bool)
            or not isinstance(n1, int)
            or not isinstance(n2, int)
            or n1 < 1
            or n2 < 0
            or n1 + n2 != nx
        ):
            raise ValueError(
                "Invalid residual dimensions for comparison: "
                f"nx={nx!r}, n1={n1!r}, n2={n2!r}."
            )
        residual = n2 > 0
        previous = grouped[residual].setdefault(nx, [])
        if previous and (
            previous[0]["n1"] != n1 or previous[0]["n2"] != n2
        ):
            label = f"{group}={group_value}" if group else "comparison"
            raise ValueError(
                f"Ambiguous residual branch for {label}, nx={nx}: "
                f"(n1, n2)=({previous[0]['n1']}, {previous[0]['n2']}) "
                f"and ({n1}, {n2})."
            )
        previous.append(row)
    return (
        _flatten_residual_branch(grouped[False], parameter, group, group_value),
        _flatten_residual_branch(grouped[True], parameter, group, group_value),
    )


def _flatten_residual_branch(
    rows: dict[int, list[dict]],
    parameter: str,
    group: str | None,
    group_value: object,
) -> list[dict]:
    """Order one residual branch and reject repeated points on a line."""
    result = []
    for nx in sorted(rows):
        values = sorted(rows[nx], key=lambda row: row[parameter])
        if parameter == "nx" and len(values) > 1:
            label = f"{group}={group_value}" if group else "comparison"
            n1, n2 = values[0]["n1"], values[0]["n2"]
            raise ValueError(
                f"Duplicate residual branch point for {label}, nx={nx}, "
                f"n1={n1}, n2={n2}."
            )
        result.extend(values)
    return result


def _add_shared_predecessor(
    main_only: list[dict],
    residual: list[dict],
) -> list[dict]:
    """Anchor a residual branch at its last completed shared predecessor."""
    if not main_only or not residual:
        return residual
    first_residual_nx = residual[0]["nx"]
    candidates = [
        row for row in main_only
        if row["nx"] < first_residual_nx
        and row["mean"] is not None
        and np.isfinite(row["mean"])
        and not row.get("failed_members")
    ]
    if not candidates:
        return residual
    return [candidates[-1], *residual]


def _plot_uncertainty(
    axis: object,
    selected: list[dict],
    x: np.ndarray,
    y: np.ndarray,
    color: str,
) -> None:
    """Render finite fold standard deviations or session standard errors."""
    uncertainty = "std" if any("std" in row for row in selected) else "sem"
    valid = [
        index
        for index, row in enumerate(selected)
        if np.isfinite(y[index])
        and row.get(uncertainty) is not None
        and np.isfinite(row[uncertainty])
    ]
    if valid:
        axis.errorbar(
            x[valid],
            y[valid],
            yerr=[selected[index][uncertainty] for index in valid],
            fmt="none",
            color=color,
            capsize=4,
        )


def _group_style(
    spec: dict,
    value: object,
    number: int,
    style: dict,
    branch: bool | None,
) -> tuple[str, str, str]:
    """Return redundant color, line and marker encodings for one series."""
    if branch is not None:
        colors = style["horizon_colors"]
        color = colors[number % len(colors)]
        linestyle, marker = RESIDUAL_STYLE if branch else MAIN_ONLY_STYLE
        return color, linestyle, marker
    group = spec.get("group")
    if not spec.get("panel"):
        color_index = {16: 0, 64: 1}.get(value, number)
        color = style["pair_colors"][color_index % len(style["pair_colors"])]
        return color, "-", "o"
    color_index = number
    if (
        not isinstance(color_index, int)
        or color_index < 0
        or color_index >= len(style["horizon_colors"])
    ):
        raise ValueError(
            f"No configured comparison color for {group}={value}."
        )
    return (
        style["horizon_colors"][color_index],
        GROUP_LINESTYLES[number % len(GROUP_LINESTYLES)],
        GROUP_MARKERS[number % len(GROUP_MARKERS)],
    )


def _series_label(
    spec: dict,
    value: object,
    branch: bool | None,
    selected: list[dict],
) -> str:
    """Format a concise legend label for a configured comparison series."""
    group = spec.get("group")
    if branch is None:
        return _group_label(group, value)
    branch_label = "residual" if branch else "main only"
    if group == "horizon":
        return f"{_group_label(group, value)}, {branch_label}"
    if group == "nx":
        row = selected[-1]
        return f"nx={value}, n1={row['n1']}, n2={row['n2']}"
    return f"{_group_label(group, value)}, {branch_label}"


def _group_label(group: str | None, value: object) -> str:
    """Format a concise legend label for one comparison series."""
    if group == "horizon":
        suffix = "step" if value == 1 else "steps"
        return f"{value} {suffix}"
    if group == "nx":
        return f"nx={value}"
    return "BRAID" if group is None else f"{group}={value}"


def _metric_title(spec: dict, metric: str) -> str:
    """Describe fixed selections or the two varying comparison dimensions."""
    if spec.get("panel"):
        title = (
            f"{METRIC_LABELS[metric]} by {PARAMETER_LABELS[spec['parameter']]}"
            f" and {PARAMETER_LABELS[spec['group']]}\n"
            f"{spec['where']['evaluation_set']} scoring"
        )
        if spec.get("branch") == RESIDUAL_BRANCH:
            title += "\nmain-only and residual branches"
    else:
        parameter = spec["parameter"]
        target = spec["where"]["target"]
        fixed = (
            f"nx={spec['where']['nx']}"
            if parameter == "horizon"
            else f"horizon={spec['where']['horizon']}"
        )
        title = (
            f"{target.capitalize()} {METRIC_LABELS[metric]}\n"
            f"{fixed}; {spec['where']['evaluation_set']} scoring"
        )
    return title


def comparison_rows(
    chosen: list[dict], dependencies: list[dict], name: str,
) -> list[dict]:
    """Attach terminal dependency and missing-metric metadata to means.

    Parameters
    ----------
    chosen : list of dict
        Available aggregate metric rows for one figure.
    dependencies : list of dict
        Terminal expected points with contribution counts and member keys.
    name : str
        Figure identifier used in validation warnings.

    Returns
    -------
    list of dict
        Available means plus explicit gaps for unavailable results.
    """
    result = [dict(row) for row in chosen]
    for candidate in dependencies:
        criteria = {
            key: value for key, value in candidate.items()
            if key not in READINESS_FIELDS
        }
        row = next((
            row for row in result
            if all(row[key] == value for key, value in criteria.items())
        ), None)
        if row is None:
            row = dict(
                criteria,
                mean=None,
                sem=None,
                std=None,
                missing_members=list(candidate.get("completed_members", [])),
            )
            result.append(row)
        row.update({
            key: candidate[key]
            for key in READINESS_FIELDS if key in candidate
        })
    for row in result:
        row.setdefault("missing_members", [])
        mean = row.get("mean")
        if mean is not None and not np.isfinite(mean):
            row["mean"] = None
        if (
            row.get("mean") is None
            and not row["missing_members"]
            and not row.get("failed_members")
        ):
            row["missing_members"] = list(
                row.get("completed_members", [f"unidentified:{name}"])
            )
    return result

def _comparison_signature(
    rows: list[dict], dependencies: list[dict], spec: dict,
    outlier_style: dict,
) -> str:
    """Fingerprint values, dependencies and semantic curve configuration."""
    return fingerprint(dict(
        rows=sorted(rows, key=repr),
        dependencies=sorted(dependencies, key=repr),
        curve={
            key: value for key, value in spec.items()
            if key != "context_label"
        },
        outlier_rendering=outlier_style,
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
    """Publish terminal comparisons and warn about unavailable results.

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
    outlier_style = outlier_rendering(settings.get("outlier_rendering"))
    rendering_style = dict(style, outlier_rendering=outlier_style)
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
            signature = _comparison_signature(
                rows, dependencies, spec, outlier_style
            )
            if path.exists() and not regenerate and _current_figure(
                path, signature
            ):
                if rendered is not None:
                    rendered.add(render_key)
                continue
            if rendered is not None:
                rendered.add(render_key)
            failures = sorted({
                member for row in rows
                for member in row.get("failed_members", [])
            })
            missing = sorted({
                member for row in rows
                for member in row.get("missing_members", [])
            })
            if failures or missing:
                counts = []
                if failures:
                    counts.append(f"{len(failures)} failed")
                if missing:
                    counts.append(f"{len(missing)} missing")
                warning_details.append(
                    f"{render_key} ({', '.join(counts)})"
                )
                warning_members.update(failures)
                warning_members.update(missing)
                if not any(
                    row.get("mean") is not None
                    and np.isfinite(row["mean"])
                    for row in rows
                ):
                    continue
            if namespace and namespace.startswith("session/"):
                spec["context_label"] = namespace.removeprefix("session/")
            figure = metric_curve(rows, spec, rendering_style)
            save_figure(figure, path, style, signature)
        except Exception as error:
            LOGGER.exception("Comparison failed: %s", render_key)
            errors.append(str(error))
    if warning_members:
        LOGGER.warning(
            "Incomplete comparisons %s; affected members: %s",
            "; ".join(warning_details),
            ", ".join(sorted(warning_members)),
        )
    if errors:
        raise PlotRenderingError(
            "Comparison rendering failed: " + "; ".join(errors)
        )
