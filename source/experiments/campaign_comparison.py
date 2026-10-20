"""Build matched, checksum-validated comparison tables and horizon curves.

Inputs are a ComparisonPlan and completed fit directories containing
metrics.json, complete.json, and checked payloads. Outputs are raw scores,
session means, matched gains, horizon figures, and report_complete.json under
the requested report root. Metric rows require target, horizon, and
mean_<metric> fields. Existing completed reports are validated and
never overwritten.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np

from experiments.multisession.store import completed
from experiments.storage.cache import atomic_json, file_digest


@dataclass(frozen=True)
class ComparisonPlan:
    """Explicit design for a matched multi-session comparison."""

    tasks: tuple[tuple[str, int, int], ...]
    sessions: tuple[str, ...]
    cases: tuple[str, ...]
    case_labels: dict[str, str]
    candidate_cases: tuple[str, ...]
    control_cases: tuple[str, ...]
    candidate_column: str
    targets: tuple[str, ...]
    horizons: tuple[int, ...]
    metrics: tuple[str, ...]
    smaller_is_better: tuple[str, ...]
    bin_ms: int
    iqr_multiplier: float

    def __post_init__(self) -> None:
        if not self.tasks or len(self.sessions) < 2:
            raise ValueError(
                "Comparison needs tasks and at least two sessions."
            )
        if not self.cases or not self.targets or not self.metrics:
            raise ValueError(
                "Comparison cases, targets, and metrics are empty."
            )
        if not self.horizons or min(self.horizons) < 1:
            raise ValueError("Comparison horizons must be positive.")
        if set(self.cases) != set(self.case_labels):
            raise ValueError("Case labels must cover each case exactly.")
        if not self.candidate_cases or not self.control_cases:
            raise ValueError("Candidate and control cases are required.")
        if set(self.candidate_cases) & set(self.control_cases):
            raise ValueError("Candidate and control cases must be distinct.")
        if not set(self.candidate_cases + self.control_cases) <= set(
            self.cases
        ):
            raise ValueError("Unknown candidate or control case.")
        if not set(self.smaller_is_better) <= set(self.metrics):
            raise ValueError("Unknown smaller-is-better metric.")
        if self.bin_ms < 1 or self.iqr_multiplier <= 0:
            raise ValueError(
                "Bin duration and IQR multiplier must be positive."
            )


def read_metrics(
    directory: Path, plan: ComparisonPlan,
) -> dict[tuple[str, int, str], float]:
    """Read every finite target, horizon, and metric score."""
    rows = json.loads((directory / "metrics.json").read_text())
    result = {}
    for row in rows:
        for metric in plan.metrics:
            value = row[f"mean_{metric}"]
            if value is None or not math.isfinite(value):
                raise ValueError(
                    f"Nonfinite {metric} in {directory.resolve()}."
                )
            key = row["target"], row["horizon"], metric
            if key in result:
                raise ValueError(
                    f"Duplicate score {key} in {directory.resolve()}."
                )
            result[key] = value
    expected = {
        (target, horizon, metric)
        for target in plan.targets for horizon in plan.horizons
        for metric in plan.metrics
    }
    if set(result) != expected:
        raise ValueError(
            f"Expected {len(expected)} score entries in "
            f"{directory.resolve()}, got {len(result)}."
        )
    return result


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    """Atomically save nonempty tabular results with fixed columns."""
    if not rows:
        raise ValueError(f"Refusing empty table: {path.resolve()}")
    temporary = path.with_suffix(path.suffix + ".pending")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _raw_rows(
    plan: ComparisonPlan,
    fit_path: Callable[[str, int, int, str], Path],
) -> list[dict]:
    """Read only checksum-complete fits for every planned case."""
    rows = []
    missing = []
    for session, fold, seed in plan.tasks:
        for case in plan.cases:
            directory = fit_path(session, fold, seed, case)
            if completed(directory) is None:
                missing.append((session, fold, seed, case))
                continue
            for (target, horizon, metric), value in read_metrics(
                directory, plan
            ).items():
                rows.append(dict(
                    session=session, fold=fold, seed=seed,
                    case=case, target=target, horizon=horizon,
                    metric=metric, value=value,
                ))
    if missing:
        raise RuntimeError(
            f"Comparison incomplete: {len(missing)} of "
            f"{len(plan.cases) * len(plan.tasks)} fits missing; "
            f"first={missing[0]}."
        )
    return rows


def _session_rows(
    rows: list[dict], plan: ComparisonPlan,
) -> list[dict]:
    """Average replicated fits within each session and design cell."""
    grouped = {}
    for row in rows:
        key = tuple(row[name] for name in (
            "session", "case", "target", "horizon", "metric"
        ))
        grouped.setdefault(key, []).append(row["value"])
    expected = len(plan.tasks) // len(plan.sessions)
    if expected < 1 or len(plan.tasks) % len(plan.sessions):
        raise ValueError("Sessions lack balanced fold and seed tasks.")
    result = [
        dict(
            session=key[0], case=key[1], target=key[2],
            horizon=key[3], metric=key[4],
            value=float(np.mean(values)), fits=len(values),
        )
        for key, values in grouped.items()
    ]
    if any(row["fits"] != expected for row in result):
        raise ValueError("A session curve lacks paired folds or seeds.")
    return result


def _paired_rows(
    rows: list[dict], plan: ComparisonPlan,
) -> list[dict]:
    """Compute oriented, within-fit candidate minus control gains."""
    by_fit = {
        tuple(row[name] for name in (
            "session", "fold", "seed", "case", "target",
            "horizon", "metric"
        )): row["value"]
        for row in rows
    }
    paired = []
    for session, fold, seed in plan.tasks:
        for candidate in plan.candidate_cases:
            for control in plan.control_cases:
                for target in plan.targets:
                    for horizon in plan.horizons:
                        for metric in plan.metrics:
                            prefix = (session, fold, seed)
                            suffix = (target, horizon, metric)
                            left = by_fit[prefix + (candidate,) + suffix]
                            right = by_fit[prefix + (control,) + suffix]
                            gain = (
                                right - left
                                if metric in plan.smaller_is_better
                                else left - right
                            )
                            paired.append(dict(
                                session=session, fold=fold, seed=seed,
                                **{plan.candidate_column: candidate},
                                control=control, target=target,
                                horizon=horizon, metric=metric, gain=gain,
                            ))
    return paired


def _session_gain_rows(
    rows: list[dict], plan: ComparisonPlan,
) -> list[dict]:
    """Average matched gains within each session and case pair."""
    grouped = {}
    for row in rows:
        key = tuple(row[name] for name in (
            "session", plan.candidate_column, "control", "target",
            "horizon", "metric"
        ))
        grouped.setdefault(key, []).append(row["gain"])
    expected = len(plan.tasks) // len(plan.sessions)
    result = [
        dict(
            session=key[0], **{plan.candidate_column: key[1]},
            control=key[2], target=key[3], horizon=key[4],
            metric=key[5], gain=float(np.mean(values)),
            pairs=len(values),
        )
        for key, values in grouped.items()
    ]
    if any(row["pairs"] != expected for row in result):
        raise ValueError("A session gain lacks paired folds or seeds.")
    return result


def report(
    root: Path, plan: ComparisonPlan,
    fit_path: Callable[[str, int, int, str], Path],
) -> None:
    """Publish validated comparison tables, figures, and completion."""
    marker = root / "report_complete.json"
    if marker.exists():
        record = json.loads(marker.read_text())
        for name, digest in record["checksums"].items():
            if file_digest(root / name) != digest:
                raise ValueError(
                    f"Completed report checksum mismatch: "
                    f"{(root / name).resolve()}"
                )
        return
    raw = _raw_rows(plan, fit_path)
    root.mkdir(parents=True, exist_ok=True)
    write_csv(
        root / "raw_scores.csv", raw,
        ["session", "fold", "seed", "case", "target",
         "horizon", "metric", "value"],
    )
    session = _session_rows(raw, plan)
    write_csv(
        root / "session_scores.csv", session,
        ["session", "case", "target", "horizon", "metric",
         "value", "fits"],
    )
    paired = _paired_rows(raw, plan)
    write_csv(
        root / "paired_gains.csv", paired,
        ["session", "fold", "seed", plan.candidate_column,
         "control", "target", "horizon", "metric", "gain"],
    )
    gains = _session_gain_rows(paired, plan)
    write_csv(
        root / "session_paired_gains.csv", gains,
        ["session", plan.candidate_column, "control", "target",
         "horizon", "metric", "gain", "pairs"],
    )
    figures(root, session, plan)
    outputs = [
        "raw_scores.csv", "session_scores.csv",
        "paired_gains.csv", "session_paired_gains.csv",
    ]
    outputs += [
        f"{target}_{metric}.png"
        for target in plan.targets for metric in plan.metrics
    ]
    atomic_json(
        marker,
        dict(
            fits=len(plan.cases) * len(plan.tasks),
            sessions=len(plan.sessions),
            checksums={
                name: file_digest(root / name) for name in outputs
            },
        ),
    )


def _curve_points(
    rows: list[dict], target: str, metric: str, case: str,
    plan: ComparisonPlan,
) -> list[dict]:
    """Aggregate session means and SEM at each forecast horizon."""
    points = []
    for horizon in plan.horizons:
        values = np.asarray([
            next(
                row["value"] for row in rows
                if row["session"] == session
                and row["case"] == case
                and row["target"] == target
                and row["metric"] == metric
                and row["horizon"] == horizon
            )
            for session in plan.sessions
        ])
        points.append(dict(
            case=case, horizon=horizon,
            mean=float(values.mean()),
            sem=float(values.std(ddof=1) / np.sqrt(len(plan.sessions))),
        ))
    return points


def figures(
    root: Path, rows: list[dict], plan: ComparisonPlan,
) -> None:
    """Draw mean and session-SEM horizon curves for each score."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from experiments.configuration import CONFIGURATION, read_yaml
    from experiments.reports.outliers import zoom_limits
    from experiments.visualization.presentation import (
        configure_metric_axis, r2_display_limits,
    )

    style = read_yaml(CONFIGURATION / "plotting" / "style.yaml")
    axes_style = style["metric_axes"]
    display = style["presentation"]
    colors = display["horizon_colors"]
    labels = [
        f"{step}\n{step * plan.bin_ms} ms"
        for step in plan.horizons
    ]
    for target in plan.targets:
        for metric in plan.metrics:
            curves = {
                case: _curve_points(rows, target, metric, case, plan)
                for case in plan.cases
            }
            points = [
                point for curve in curves.values()
                for point in curve
            ]
            lower, upper, robust_low, robust_high = zoom_limits(
                points, plan.iqr_multiplier
            )
            inliers = [
                point for point in points
                if robust_low <= point["mean"] <= robust_high
            ]
            if metric == "r2":
                visible_low, visible_high = r2_display_limits(
                    inliers, axes_style
                )
                lower = max(lower, visible_low)
                upper = min(upper, visible_high)
            elif metric == "mse" and lower <= 0:
                positive = [
                    point["mean"] for point in inliers
                    if point["mean"] > 0
                ]
                if not positive:
                    raise ValueError("MSE plot has no positive means.")
                lower = min(positive) * axes_style["mse"][
                    "uncertainty_lower_fraction"
                ]
            figure, axis = plt.subplots(
                figsize=display["single_size"]
            )
            for index, case in enumerate(plan.cases):
                curve = curves[case]
                x = np.asarray(plan.horizons, dtype=float)
                y = np.asarray([p["mean"] for p in curve])
                sem = np.asarray([p["sem"] for p in curve])
                inlier = (
                    (y >= robust_low) & (y <= robust_high)
                )
                color = colors[index]
                axis.plot(
                    x, np.where(inlier, y, np.nan),
                    marker=display["group_markers"][index],
                    linestyle=display["group_linestyles"][
                        index % len(display["group_linestyles"])
                    ],
                    color=color, label=plan.case_labels[case],
                )
                axis.fill_between(
                    x, np.where(inlier, y - sem, np.nan),
                    np.where(inlier, y + sem, np.nan),
                    color=color, alpha=0.15,
                )
                for position in np.flatnonzero(~inlier):
                    value = y[position]
                    edge = upper if value > robust_high else lower
                    axis.scatter(
                        [x[position]], [edge],
                        color=color,
                        marker="^" if value > robust_high else "v",
                        clip_on=False,
                    )
                    axis.annotate(
                        f"{value:.3g}", (x[position], edge),
                        xytext=(5, 8 + index * 10),
                        textcoords="offset points", color=color,
                        clip_on=False,
                    )
            axis.set_xscale("log", base=2)
            axis.set_xticks(plan.horizons, labels)
            axis.set_xlim(
                plan.horizons[0] / 1.2,
                plan.horizons[-1] * 1.2,
            )
            axis.set_ylim(lower, upper)
            configure_metric_axis(axis, metric, axes_style)
            axis.set_xlabel("Forecast horizon (steps / ms)")
            axis.set_title(f"{target.title()} prediction: mean ± SEM")
            axis.grid(alpha=0.2)
            axis.legend(frameon=False)
            figure.tight_layout()
            figure.savefig(
                root / f"{target}_{metric}.png",
                dpi=display["dpi"],
            )
            plt.close(figure)
