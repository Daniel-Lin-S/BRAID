"""Atomic rolling and final report publication with synthetic fit metrics."""

import json
from pathlib import Path

import pytest

from experiments.multisession import reports
from experiments.multisession.plan import PRIMARY
from experiments.multisession.store import completed


def report_fixture(tmp_path: Path) -> tuple[dict, Path, Path]:
    """Return a one-analysis plan and its output directory."""
    plan = dict(
        analyses={PRIMARY: ["fit"]},
        settings=dict(paths=dict(artifact_root=str(tmp_path))),
    )
    path = tmp_path / "plans" / "plan-id" / "plan.json"
    path.parent.mkdir(parents=True)
    analysis = tmp_path / "analysis" / PRIMARY / "plan-id"
    return plan, path, analysis


def metric_row(value: float) -> dict:
    """Return one finite behavior CC result for aggregation."""
    return dict(
        objective="neural",
        latent=32,
        sharing="scratch",
        source_count=0,
        fraction=0.2,
        target="behavior",
        horizon=1,
        metric="cc",
        session="held_out",
        fold=0,
        seed=42,
        fit_id="fit",
        value=value,
        valid_dimensions=1,
        scored_samples=128,
    )


def test_rolling_report_replaces_only_validated_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unchanged fits skip rendering and a failed render keeps prior output."""
    plan, path, analysis = report_fixture(tmp_path)
    state = dict(value=0.4, digest="first", fail=False, renders=0)

    def collect(_plan: dict, _members: list[str]) -> tuple:
        row = metric_row(state["value"])
        return [row], [dict(row, coordinate="x")], [
            ("fit", state["digest"])
        ]

    def figures(_summary: list, _name: str, directory: Path,
                _settings: dict) -> None:
        state["renders"] += 1
        if state["fail"]:
            raise RuntimeError("synthetic figure failure")
        (directory / "figure.png").write_bytes(b"figure")

    monkeypatch.setattr(reports, "collect", collect)
    monkeypatch.setattr(reports, "cost_tables", lambda *args: None)
    monkeypatch.setattr(reports, "figures", figures)

    reports.report(plan, path)
    first = json.loads((analysis / "latest_report.json").read_text())
    first_path = analysis / first["snapshot"]
    assert first["mode"] == "live"
    assert first["completed_fits"] == 1
    assert reports.valid_live(first_path, first["fit_set_hash"])
    reports.report(plan, path)
    assert state["renders"] == 1

    state.update(value=0.5, digest="second")
    reports.report(plan, path)
    second = json.loads((analysis / "latest_report.json").read_text())
    second_path = analysis / second["snapshot"]
    assert second_path != first_path
    assert not first_path.exists()
    assert state["renders"] == 2

    state.update(value=0.6, digest="failed", fail=True)
    with pytest.raises(RuntimeError, match="synthetic figure failure"):
        reports.report(plan, path)
    assert json.loads((analysis / "latest_report.json").read_text()) == second
    assert second_path.exists()
    assert len(list((analysis / "reports").glob("live-*"))) == 1

    state.update(value=0.5, digest="second", fail=False)
    reports.report(plan, path, final=True)
    final = json.loads((analysis / "latest_report.json").read_text())
    assert final["mode"] == "final"
    assert completed(analysis / final["snapshot"]) is not None
    assert not list((analysis / "reports").glob("live-*"))
    renders = state["renders"]
    reports.report(plan, path, final=True)
    assert state["renders"] == renders
