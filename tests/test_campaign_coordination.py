"""Verify model-independent campaign status and matched comparison output."""

from __future__ import annotations

import csv
import fcntl
import json
from pathlib import Path

import pytest

from experiments.campaign_comparison import ComparisonPlan, report
from experiments.campaign_status import FitJob, StatusPlan, refresh
from experiments.campaign_runner import run_phase
from experiments.multisession.store import finish


SESSIONS = ("session_a", "session_b")
CASES = ("baseline", "candidate")
HORIZONS = (1, 2)
METRICS = ("cc", "mse")


def _finish_or_fail(value: int) -> str:
    """Return one result or raise for a separate independent job."""
    if value == 2:
        raise ValueError("Synthetic fit failure")
    return f"job {value}"


def test_generic_phase_keeps_running_after_one_failure() -> None:
    """The process pool reports failures without skipping peer jobs."""
    failures = run_phase(
        _finish_or_fail, [(1,), (2,), (3,)], workers=2
    )
    assert len(failures) == 1
    assert failures[0][0] == (2,)
    assert isinstance(failures[0][1], ValueError)


def test_status_plan_accepts_unrelated_case_names(tmp_path: Path) -> None:
    """Count a reused baseline and a live candidate by their paths."""
    prep = tmp_path / "preprocessing" / "session_a"
    prep.mkdir(parents=True)
    (prep / "complete.json").write_text("{}")
    fit = tmp_path / "fits" / "candidate_session_a"
    fit.mkdir(parents=True)
    (fit / "status.json").write_text('{"state": "running"}')
    plan = StatusPlan(
        preprocessing=(prep,),
        fits=(
            FitJob(None, preprocessing_index=0, reused=True),
            FitJob(fit),
        ),
        report_marker=tmp_path / "report_complete.json",
        coordinator_module="experiments.unrelated_model",
    )
    (tmp_path / "launch.json").write_text(
        '{"pid": 999999999}'
    )
    with fit.with_suffix(".lock").open("w") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        record = refresh(tmp_path, plan)
    assert record["campaign_active"] is True
    assert record["fits"]["finished"] == 1
    assert record["fits"]["reused_finished"] == 1
    assert record["fits"]["running"] == 1


def _comparison_plan() -> ComparisonPlan:
    """Specify a two-session comparison without model names."""
    tasks = tuple(
        (session, 0, seed)
        for session in SESSIONS for seed in (0, 1)
    )
    return ComparisonPlan(
        tasks=tasks, sessions=SESSIONS,
        cases=CASES,
        case_labels={"baseline": "Baseline", "candidate": "Candidate"},
        candidate_cases=("candidate",),
        control_cases=("baseline",),
        candidate_column="candidate",
        targets=("neural",), horizons=HORIZONS,
        metrics=METRICS, smaller_is_better=("mse",),
        bin_ms=50, iqr_multiplier=3.0,
    )


def _write_fit(directory: Path, case: str) -> None:
    """Create one checksum-complete synthetic metric payload."""
    directory.mkdir(parents=True)
    rows = [
        dict(
            target="neural", horizon=horizon,
            mean_cc=0.6 if case == "candidate" else 0.4,
            mean_mse=1.0 if case == "candidate" else 2.0,
        )
        for horizon in HORIZONS
    ]
    (directory / "metrics.json").write_text(json.dumps(rows))
    finish(directory, {"identity_hash": case})


def test_comparison_reports_matched_oriented_gains(tmp_path: Path) -> None:
    """Produce complete tables and figures for arbitrary case names."""
    plan = _comparison_plan()

    def fit_path(
        session: str, fold: int, seed: int, case: str,
    ) -> Path:
        return tmp_path / "fits" / f"{session}_{fold}_{seed}_{case}"

    for session, fold, seed in plan.tasks:
        for case in CASES:
            _write_fit(fit_path(session, fold, seed, case), case)
    report(tmp_path, plan, fit_path)
    marker = tmp_path / "report_complete.json"
    assert marker.is_file()
    with (tmp_path / "paired_gains.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(plan.tasks) * len(HORIZONS) * len(METRICS)
    assert {row["candidate"] for row in rows} == {"candidate"}
    assert all(float(row["gain"]) > 0 for row in rows)
    original = marker.read_bytes()
    report(tmp_path, plan, fit_path)
    assert marker.read_bytes() == original


def test_comparison_rejects_incomplete_fits(tmp_path: Path) -> None:
    """Missing fits cannot produce partial comparison tables."""
    plan = _comparison_plan()

    def fit_path(
        session: str, fold: int, seed: int, case: str,
    ) -> Path:
        return tmp_path / "fits" / f"{session}_{fold}_{seed}_{case}"

    _write_fit(fit_path(SESSIONS[0], 0, 0, CASES[0]), CASES[0])
    with pytest.raises(RuntimeError, match="Comparison incomplete"):
        report(tmp_path, plan, fit_path)
    assert not (tmp_path / "raw_scores.csv").exists()
