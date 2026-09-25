"""Resolve finite-metric exclusions used by comparison reporting.

Inputs are exact-member or session/fold slice selectors from
``plotting.outliers``, an optional horizon-wise statistical screen, completed
metric rows, and optional analysis membership. Outputs are atomic
member/target/metric rules that remain outside the normal per-session display
range. Scientific metric artifacts are never modified.
"""

import logging
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

LOGGER = logging.getLogger(__name__)
TARGETS = frozenset(("neural", "behavior"))
METRICS = frozenset(("cc", "r2", "mse"))
EVALUATION_SETS = frozenset(("full", "common"))
COMMON_FIELDS = frozenset((
    "horizon", "evaluation_set", "targets", "metrics",
    "aggregate_exclude", "session_zoom", "reason",
))
SLICE_FIELDS = frozenset(("session", "fold"))
SCREEN_FIELDS = frozenset((
    "targets", "metrics", "horizons", "evaluation_set",
    "standard_deviation_threshold", "standard_error_threshold",
    "aggregate_exclude", "session_zoom", "reason",
))


@dataclass(frozen=True)
class OutlierRule:
    """One atomic finite-metric selector used only for presentation."""

    member: str
    horizon: int
    evaluation_set: str
    targets: tuple[str, ...]
    metrics: tuple[str, ...]
    aggregate_exclude: bool
    session_zoom: bool
    reason: str


@dataclass(frozen=True)
class _ConfiguredRule:
    """One validated exact-member or session/fold selector."""

    member: str | None
    session: str | None
    fold: int | None
    horizon: int
    evaluation_set: str
    targets: tuple[str, ...]
    metrics: tuple[str, ...]
    aggregate_exclude: bool
    session_zoom: bool
    reason: str


def resolve_outliers(
    settings: dict,
    rows: list[dict],
    members: dict[str, dict] | None = None,
    attempted: set[str] | None = None,
    warned: set[str] | None = None,
) -> tuple[OutlierRule, ...]:
    """Resolve ready selectors and restore candidates in the normal range.

    Parameters
    ----------
    settings : dict
        Plotting settings containing optional ``outliers`` selectors.
    rows : list of dict
        Completed raw metric rows available to this report.
    members : dict of str to dict, optional
        Analysis membership used to expand slices and defer pending members.
    attempted : set of str, optional
        Members attempted in this invocation. Completed historical members
        outside this set remain pending until a plot-only invocation.
    warned : set of str, optional
        Invocation-local diagnostic keys used to suppress repeated warnings.

    Returns
    -------
    tuple of OutlierRule
        Atomic effective exclusions with finite matching metric values.
    """
    configured = settings.get("outliers", [])
    if configured is None:
        configured = []
    if not isinstance(configured, list):
        raise ValueError("plotting.outliers must be a list of mappings.")
    selectors = tuple(
        _parse_rule(item, index) for index, item in enumerate(configured)
    )
    _validate_unique_selectors(selectors)
    rules = _expand_rules(selectors, rows, members, attempted)
    rules += _screen_rules(settings.get("outlier_screen"), rows)
    _validate_matches(rules, rows)
    return _restore_normal_values(rules, rows, warned)


def matching_rule(
    rules: Iterable[OutlierRule], row: dict, target: str, metric: str,
) -> OutlierRule | None:
    """Return the sole matching effective rule for one raw metric."""
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


def _parse_rule(value: object, index: int) -> _ConfiguredRule:
    """Validate one selector without inspecting saved metric rows."""
    label = f"plotting.outliers[{index}]"
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping.")
    has_member = "member" in value
    has_slice = "slice" in value
    if has_member == has_slice:
        raise ValueError(
            f"{label} must contain exactly one of member or slice."
        )
    required = COMMON_FIELDS | ({"member"} if has_member else {"slice"})
    if set(value) != required:
        missing = sorted(required - set(value))
        unknown = sorted(set(value) - required)
        raise ValueError(
            f"{label} fields are invalid; missing={missing}, "
            f"unknown={unknown}."
        )
    member = _parse_member(value["member"], label) if has_member else None
    session, fold = (
        (None, None)
        if has_member else _parse_slice(value["slice"], label)
    )
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
    return _ConfiguredRule(
        member=member,
        session=session,
        fold=fold,
        horizon=horizon,
        evaluation_set=evaluation_set,
        targets=targets,
        metrics=metrics,
        aggregate_exclude=value["aggregate_exclude"],
        session_zoom=value["session_zoom"],
        reason=reason,
    )


def _parse_member(value: object, label: str) -> str:
    """Validate one exact analysis member identifier."""
    parts = value.split("/") if isinstance(value, str) else []
    if len(parts) != 3 or not all(parts) or not parts[2].startswith("fold_"):
        raise ValueError(
            f"{label}.member must be configuration/session/fold_number."
        )
    if not parts[2].removeprefix("fold_").isdecimal():
        raise ValueError(f"{label}.member fold must be a nonnegative integer.")
    return value


def _parse_slice(value: object, label: str) -> tuple[str, int]:
    """Validate one selector spanning every case in a session and fold."""
    if not isinstance(value, dict) or set(value) != SLICE_FIELDS:
        raise ValueError(
            f"{label}.slice must contain exactly session and fold."
        )
    session = value["session"]
    fold = value["fold"]
    if not isinstance(session, str) or not session.strip():
        raise ValueError(f"{label}.slice.session must be a nonempty string.")
    if type(fold) is not int or fold < 0:
        raise ValueError(
            f"{label}.slice.fold must be a nonnegative integer."
        )
    return session, fold


def _choices(
    value: object, allowed: frozenset[str], label: str,
) -> tuple[str, ...]:
    """Validate a nonempty unique list of supported choices."""
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a nonempty list.")
    if any(item not in allowed for item in value):
        raise ValueError(f"{label} contains unsupported values: {value!r}.")
    if len(set(value)) != len(value):
        raise ValueError(f"{label} contains duplicate values: {value!r}.")
    return tuple(value)


def _positive_threshold(value: object, label: str) -> float:
    """Validate one positive finite statistical threshold."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not np.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{label} must be a positive finite number.")
    return float(value)


def _screen_settings(value: object) -> dict | None:
    """Validate the optional horizon-wise median-distance screen."""
    if value is None:
        return None
    label = "plotting.outlier_screen"
    if not isinstance(value, dict) or set(value) != SCREEN_FIELDS:
        fields = set(value) if isinstance(value, dict) else set()
        raise ValueError(
            f"{label} fields are invalid; "
            f"missing={sorted(SCREEN_FIELDS - fields)}, "
            f"unknown={sorted(fields - SCREEN_FIELDS)}."
        )
    horizons = value["horizons"]
    if (
        not isinstance(horizons, list)
        or not horizons
        or any(type(item) is not int or item < 1 for item in horizons)
        or len(set(horizons)) != len(horizons)
    ):
        raise ValueError(f"{label}.horizons must be unique positive integers.")
    evaluation_set = value["evaluation_set"]
    if evaluation_set not in EVALUATION_SETS:
        raise ValueError(
            f"{label}.evaluation_set must be full or common, got "
            f"{evaluation_set!r}."
        )
    for field in ("aggregate_exclude", "session_zoom"):
        if type(value[field]) is not bool:
            raise ValueError(f"{label}.{field} must be true or false.")
    reason = value["reason"]
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError(f"{label}.reason must be a nonempty string.")
    return dict(
        targets=_choices(value["targets"], TARGETS, f"{label}.targets"),
        metrics=_choices(value["metrics"], METRICS, f"{label}.metrics"),
        horizons=tuple(horizons),
        evaluation_set=evaluation_set,
        standard_deviation_threshold=_positive_threshold(
            value["standard_deviation_threshold"],
            f"{label}.standard_deviation_threshold",
        ),
        standard_error_threshold=_positive_threshold(
            value["standard_error_threshold"],
            f"{label}.standard_error_threshold",
        ),
        aggregate_exclude=value["aggregate_exclude"],
        session_zoom=value["session_zoom"],
        reason=reason,
    )


def _screen_rules(
    value: object, rows: list[dict],
) -> tuple[OutlierRule, ...]:
    """Select finite values far from a horizon-wise metric median."""
    settings = _screen_settings(value)
    if settings is None:
        return ()
    groups: dict[tuple[str, str, int], list[tuple[dict, float]]] = {}
    for row in rows:
        if (
            row["evaluation_set"] != settings["evaluation_set"]
            or row["horizon"] not in settings["horizons"]
        ):
            continue
        for target in settings["targets"]:
            for metric in settings["metrics"]:
                value = row[target][f"mean_{metric}"]
                if value is not None and np.isfinite(value):
                    groups.setdefault(
                        (target, metric, row["horizon"]), []
                    ).append((row, float(value)))
    rules = []
    for (target, metric, horizon), grouped_values in sorted(groups.items()):
        values = list(grouped_values)
        if len(values) < 2:
            raise ValueError(
                "plotting.outlier_screen requires at least two finite values "
                f"for target={target} metric={metric} horizon={horizon}."
            )
        while len(values) >= 2:
            array = np.asarray([item[1] for item in values], dtype=float)
            median = float(np.median(array))
            standard_deviation = float(array.std(ddof=1))
            standard_error = standard_deviation / np.sqrt(len(array))
            selected = [
                (row, metric_value)
                for row, metric_value in values
                if (
                    abs(metric_value - median)
                    > settings["standard_deviation_threshold"]
                    * standard_deviation
                    or abs(metric_value - median)
                    > settings["standard_error_threshold"] * standard_error
                )
            ]
            if not selected:
                break
            selected_members = {
                member_id(row) for row, _metric_value in selected
            }
            for row, _metric_value in selected:
                rules.append(OutlierRule(
                    member=member_id(row),
                    horizon=horizon,
                    evaluation_set=settings["evaluation_set"],
                    targets=(target,),
                    metrics=(metric,),
                    aggregate_exclude=settings["aggregate_exclude"],
                    session_zoom=settings["session_zoom"],
                    reason=settings["reason"],
                ))
            values = [
                item for item in values
                if member_id(item[0]) not in selected_members
            ]
    return tuple(rules)


def _selector_key(rule: _ConfiguredRule) -> tuple:
    """Return one immutable configured-selector identity."""
    selector = (
        ("member", rule.member)
        if rule.member is not None
        else ("slice", rule.session, rule.fold)
    )
    return (
        selector,
        rule.horizon,
        rule.evaluation_set,
        rule.targets,
        rule.metrics,
    )


def _validate_unique_selectors(
    rules: tuple[_ConfiguredRule, ...],
) -> None:
    """Reject repeated configured selectors before expansion."""
    seen = set()
    for rule in rules:
        key = _selector_key(rule)
        if key in seen:
            raise ValueError(f"Duplicate plotting.outliers selector: {key!r}.")
        seen.add(key)


def _candidate_members(
    rule: _ConfiguredRule,
    rows: list[dict],
    members: dict[str, dict] | None,
) -> list[str]:
    """Return exact member keys selected before readiness filtering."""
    if rule.member is not None:
        if members is not None and rule.member not in members:
            raise ValueError(
                "Unknown plotting.outliers member selector: "
                f"{rule.member}."
            )
        return [rule.member]
    universe = members or {
        member_id(row): {
            "session": row["session"],
            "fold": row["fold"],
            "state": "complete",
        }
        for row in rows
    }
    selected = sorted(
        key for key, value in universe.items()
        if value["session"] == rule.session and value["fold"] == rule.fold
    )
    if not selected:
        raise ValueError(
            "Unmatched plotting.outliers slice selector: "
            f"session={rule.session} fold={rule.fold}."
        )
    return selected


def _member_ready(
    key: str,
    members: dict[str, dict] | None,
    attempted: set[str] | None,
) -> bool:
    """Return whether a selected member can provide metrics in this report."""
    if members is None:
        return True
    state = members[key]["state"]
    if state == "failed":
        return False
    if state != "complete":
        return False
    return attempted is None or key in attempted


def _expand_rules(
    selectors: tuple[_ConfiguredRule, ...],
    rows: list[dict],
    members: dict[str, dict] | None,
    attempted: set[str] | None,
) -> tuple[OutlierRule, ...]:
    """Expand ready selectors into atomic member/target/metric rules."""
    result = []
    for selector in selectors:
        for member in _candidate_members(selector, rows, members):
            if not _member_ready(member, members, attempted):
                continue
            for target in selector.targets:
                for metric in selector.metrics:
                    result.append(OutlierRule(
                        member=member,
                        horizon=selector.horizon,
                        evaluation_set=selector.evaluation_set,
                        targets=(target,),
                        metrics=(metric,),
                        aggregate_exclude=selector.aggregate_exclude,
                        session_zoom=selector.session_zoom,
                        reason=selector.reason,
                    ))
    return tuple(result)


def _validate_matches(
    rules: tuple[OutlierRule, ...], rows: list[dict],
) -> None:
    """Require each ready atomic selector to match one finite result."""
    claimed = set()
    for rule in rules:
        target = rule.targets[0]
        metric = rule.metrics[0]
        matches = [
            row for row in rows
            if matching_rule((rule,), row, target, metric) is not None
        ]
        label = _rule_label(rule)
        if not matches:
            raise ValueError(
                "Unmatched plotting.outliers selector: " + label + "."
            )
        if len(matches) != 1:
            raise ValueError(
                "Ambiguous plotting.outliers selector: " + label + "."
            )
        value = matches[0][target][f"mean_{metric}"]
        if value is None or not np.isfinite(value):
            raise ValueError(
                "plotting.outliers requires a finite selected metric: "
                + label + "."
            )
        if label in claimed:
            raise ValueError(
                "Overlapping plotting.outliers selector: " + label + "."
            )
        claimed.add(label)


def _context_key(rule: OutlierRule, row: dict) -> tuple:
    """Return the session metric panel used for range validation."""
    return (
        row["session"],
        rule.horizon,
        rule.evaluation_set,
        rule.targets[0],
        rule.metrics[0],
    )


def _reference_range(
    rules: tuple[OutlierRule, ...], rows: list[dict], context: tuple,
) -> tuple[float, float] | None:
    """Calculate renderer-equivalent bounds from nonselected folds."""
    session, horizon, evaluation_set, target, metric = context
    excluded = {
        _result_id_from_rule(rule)
        for rule in rules
        if (
            rule.member.split("/")[1],
            rule.horizon,
            rule.evaluation_set,
            rule.targets[0],
            rule.metrics[0],
        ) == context
    }
    groups: dict[str, list[float]] = {}
    for row in rows:
        if (
            row["session"] != session
            or row["horizon"] != horizon
            or row["evaluation_set"] != evaluation_set
        ):
            continue
        if _result_id(row, target, metric) in excluded:
            continue
        value = row[target][f"mean_{metric}"]
        if value is not None and np.isfinite(value):
            groups.setdefault(row["configuration"], []).append(float(value))
    bounds = []
    for values in groups.values():
        array = np.asarray(values, dtype=float)
        mean = float(array.mean())
        spread = float(array.std(ddof=1)) if len(array) > 1 else 0.0
        bounds.extend((mean - spread, mean + spread))
    if not bounds:
        return None
    return min(bounds), max(bounds)


def _restore_normal_values(
    rules: tuple[OutlierRule, ...],
    rows: list[dict],
    warned: set[str] | None,
) -> tuple[OutlierRule, ...]:
    """Restore configured values that fall inside the evolving normal range."""
    active = list(rules)
    values = {
        _result_id(row, target, metric): row[target][f"mean_{metric}"]
        for row in rows
        for target in TARGETS
        for metric in METRICS
    }
    while active:
        restored = []
        contexts = {}
        for rule in active:
            member_rows = [
                row for row in rows if member_id(row) == rule.member
                and row["horizon"] == rule.horizon
                and row["evaluation_set"] == rule.evaluation_set
            ]
            context = _context_key(rule, member_rows[0])
            if context not in contexts:
                contexts[context] = _reference_range(
                    tuple(active), rows, context
                )
            reference_range = contexts[context]
            if reference_range is None:
                continue
            lower, upper = reference_range
            value = float(values[_result_id_from_rule(rule)])
            if lower <= value <= upper:
                warning_key = (
                    "outlier-restored/" + _result_id_from_rule(rule)
                )
                if warned is None or warning_key not in warned:
                    LOGGER.warning(
                        "Configured outlier is within the normal display "
                        "range and will be included and plotted normally; %s "
                        "value=%r reference_range=[%r, %r]",
                        _rule_label(rule), value, lower, upper,
                    )
                    if warned is not None:
                        warned.add(warning_key)
                restored.append(rule)
        if not restored:
            break
        restored_set = set(restored)
        active = [rule for rule in active if rule not in restored_set]
    return tuple(active)


def _rule_label(rule: OutlierRule) -> str:
    """Format one atomic selector for diagnostics."""
    return (
        f"{rule.member} horizon={rule.horizon} "
        f"evaluation_set={rule.evaluation_set} "
        f"target={rule.targets[0]} metric={rule.metrics[0]}"
    )


def _result_id_from_rule(rule: OutlierRule) -> str:
    """Return the selected raw-result identity for one atomic rule."""
    return (
        f"{rule.member} horizon={rule.horizon} "
        f"evaluation_set={rule.evaluation_set} "
        f"target={rule.targets[0]} metric={rule.metrics[0]}"
    )


def _result_id(row: dict, target: str, metric: str) -> str:
    """Format a selected raw result for validation and range filtering."""
    return (
        f"{member_id(row)} horizon={row['horizon']} "
        f"evaluation_set={row['evaluation_set']} target={target} "
        f"metric={metric}"
    )
