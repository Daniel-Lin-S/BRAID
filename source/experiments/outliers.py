"""Validate explicit finite-metric display exclusions for plot reporting.

Input rules live under ``plotting.outliers`` and select completed metric rows.
The output rules are immutable descriptions used only while aggregating and
rendering. Raw metric artifacts and member completion payloads are unchanged.
"""

from dataclasses import dataclass
from typing import Iterable

import numpy as np


TARGETS = frozenset(("neural", "behavior"))
METRICS = frozenset(("cc", "r2", "mse"))
EVALUATION_SETS = frozenset(("full", "common"))
REQUIRED_FIELDS = frozenset((
    "member", "horizon", "evaluation_set", "targets", "metrics",
    "aggregate_exclude", "session_zoom", "reason",
))


@dataclass(frozen=True)
class OutlierRule:
    """One immutable finite-metric selector used only for presentation."""

    member: str
    horizon: int
    evaluation_set: str
    targets: tuple[str, ...]
    metrics: tuple[str, ...]
    aggregate_exclude: bool
    session_zoom: bool
    reason: str


def resolve_outliers(
    settings: dict, rows: list[dict],
) -> tuple[OutlierRule, ...]:
    """Parse rules and require exactly one finite raw result per selection."""
    configured = settings.get("outliers", [])
    if configured is None:
        configured = []
    if not isinstance(configured, list):
        raise ValueError("plotting.outliers must be a list of mappings.")
    rules = tuple(
        _parse_rule(item, index) for index, item in enumerate(configured)
    )
    _validate_unique_rules(rules)
    _validate_matches(rules, rows)
    return rules


def matching_rule(
    rules: Iterable[OutlierRule], row: dict, target: str, metric: str,
) -> OutlierRule | None:
    """Return the sole matching rule for one raw target metric, if any."""
    matches = [
        rule for rule in rules
        if rule.member == member_id(row)
        and rule.horizon == row["horizon"]
        and rule.evaluation_set == row["evaluation_set"]
        and target in rule.targets
        and metric in rule.metrics
    ]
    if len(matches) > 1:
        raise ValueError(
            "Ambiguous plotting.outliers rules select "
            f"{_result_id(row, target, metric)}."
        )
    return matches[0] if matches else None


def member_id(row: dict) -> str:
    """Return the stable member identifier for one completed metric row."""
    return f"{row['configuration']}/{row['session']}/fold_{row['fold']}"


def _parse_rule(value: object, index: int) -> OutlierRule:
    """Validate one YAML rule without inspecting saved metric rows."""
    label = f"plotting.outliers[{index}]"
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping.")
    if set(value) != REQUIRED_FIELDS:
        missing = sorted(REQUIRED_FIELDS - set(value))
        unknown = sorted(set(value) - REQUIRED_FIELDS)
        raise ValueError(
            f"{label} fields are invalid; missing={missing}, unknown={unknown}."
        )
    member = value["member"]
    parts = member.split("/") if isinstance(member, str) else []
    if len(parts) != 3 or not all(parts) or not parts[2].startswith("fold_"):
        raise ValueError(
            f"{label}.member must be configuration/session/fold_number."
        )
    fold_text = parts[2].removeprefix("fold_")
    if not fold_text.isdecimal():
        raise ValueError(f"{label}.member fold must be a nonnegative integer.")
    horizon = value["horizon"]
    if type(horizon) is not int or horizon < 1:
        raise ValueError(f"{label}.horizon must be a positive integer.")
    evaluation_set = value["evaluation_set"]
    if evaluation_set not in EVALUATION_SETS:
        raise ValueError(
            f"{label}.evaluation_set must be full or common, got "
            f"{evaluation_set!r}."
        )
    targets = _choices(value["targets"], TARGETS, f"{label}.targets")
    metrics = _choices(value["metrics"], METRICS, f"{label}.metrics")
    for field in ("aggregate_exclude", "session_zoom"):
        if type(value[field]) is not bool:
            raise ValueError(f"{label}.{field} must be true or false.")
    reason = value["reason"]
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError(f"{label}.reason must be a nonempty string.")
    return OutlierRule(
        member=member,
        horizon=horizon,
        evaluation_set=evaluation_set,
        targets=targets,
        metrics=metrics,
        aggregate_exclude=value["aggregate_exclude"],
        session_zoom=value["session_zoom"],
        reason=reason,
    )


def _choices(
    value: object, allowed: frozenset[str], label: str,
) -> tuple[str, ...]:
    """Validate a nonempty unique list of supported selector choices."""
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a nonempty list.")
    if any(item not in allowed for item in value):
        raise ValueError(f"{label} contains unsupported values: {value!r}.")
    if len(set(value)) != len(value):
        raise ValueError(f"{label} contains duplicate values: {value!r}.")
    return tuple(value)


def _validate_unique_rules(rules: tuple[OutlierRule, ...]) -> None:
    """Reject repeated selectors before examining metric rows."""
    seen = set()
    for rule in rules:
        key = (
            rule.member, rule.horizon, rule.evaluation_set,
            rule.targets, rule.metrics,
        )
        if key in seen:
            raise ValueError(f"Duplicate plotting.outliers selector: {key!r}.")
        seen.add(key)


def _validate_matches(rules: tuple[OutlierRule, ...], rows: list[dict]) -> None:
    """Require every selected raw result to be unique and finite."""
    claimed = set()
    for rule in rules:
        for target in rule.targets:
            for metric in rule.metrics:
                matches = [
                    row for row in rows
                    if matching_rule((rule,), row, target, metric) is not None
                ]
                label = (
                    f"{rule.member} horizon={rule.horizon} "
                    f"evaluation_set={rule.evaluation_set} "
                    f"target={target} metric={metric}"
                )
                if not matches:
                    raise ValueError(
                        "Unmatched plotting.outliers selector: " + label + "."
                    )
                if len(matches) != 1:
                    raise ValueError(
                        "Ambiguous plotting.outliers selector: " + label + "."
                    )
                metric_value = matches[0][target][f"mean_{metric}"]
                if metric_value is None or not np.isfinite(metric_value):
                    raise ValueError(
                        "plotting.outliers requires a finite selected metric: "
                        + label + "."
                    )
                if label in claimed:
                    raise ValueError(
                        "Overlapping plotting.outliers selector: " + label + "."
                    )
                claimed.add(label)


def _result_id(row: dict, target: str, metric: str) -> str:
    """Format a selected raw result for a validation error."""
    return (
        f"{member_id(row)} horizon={row['horizon']} "
        f"evaluation_set={row['evaluation_set']} target={target} "
        f"metric={metric}"
    )
