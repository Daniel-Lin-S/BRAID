"""Check canonical/horizon history plots and cross-setting comparisons."""

import json

import matplotlib.pyplot as plt
import numpy as np
import pytest

from experiments.configuration import CONFIGURATION, read_yaml
from experiments.history import history_figure, render_component
from experiments.plots import curve_specs, metric_curve, plot_suite
from experiments.presentation import presentation

STYLE = presentation()


def history_rows(
    output: str | None = None,
    horizons: tuple[int, ...] = tuple(range(1, 9)),
) -> list[dict]:
    """Produce canonical totals and all eight step-wise metric families."""
    output_prefix = "" if output is None else f"{output}_"
    return [
        dict(
            epoch=epoch,
            attempt=1,
            metrics={
                **{prefix + "loss": 9 / epoch for prefix in ("", "val_")},
                **{
                    f"{prefix}rnn_{output_prefix}{horizon}step_{metric}": (
                        horizon / epoch
                    )
                    for prefix in ("", "val_")
                    for horizon in horizons
                    for metric in ("loss", "MSE", "R2", "CC")
                },
            },
        )
        for epoch in range(1, 5)
    ]


def test_horizon_layout_colors_and_axes(tmp_path):
    rows = history_rows()
    for metric in ("loss", "MSE", "R2", "CC"):
        figure = history_figure(rows, metric, list(range(1, 9)), metric, STYLE)
        try:
            left, right = figure.axes
            assert len(left.lines) == len(right.lines) == 8
            assert left.get_shared_x_axes().joined(left, right)
            assert left.get_shared_y_axes().joined(left, right)
            assert left.get_position().x0 < right.get_position().x0
            assert [line.get_color() for line in left.lines] == [
                line.get_color() for line in right.lines
            ]
            assert len({line.get_color() for line in left.lines}) == 8
            np.testing.assert_allclose(figure.get_size_inches(), [20, 7])
        finally:
            plt.close(figure)
    path = tmp_path / "components" / "main" / "rnn"
    path.mkdir(parents=True)
    history = path / "history.jsonl"
    history.write_text("\n".join(json.dumps(row) for row in rows))
    original = history.read_bytes()
    render_component(path, STYLE)
    assert sorted(
        p.name for p in (path / "plots" / "attempt_1").glob("*.png")
    ) == [
        "cc_by_horizon.png",
        "loss_by_horizon.png",
        "mse_by_horizon.png",
        "r2_by_horizon.png",
        "total_loss.png",
    ]
    assert history.read_bytes() == original
    assert not list(tmp_path.rglob("stage_loss_summary.json"))


def test_analysis_catalogue_and_undefined_gaps(tmp_path):
    specifications = []
    for name, count in (
        ("latent_dimension_sweep", 12),
        ("neural_population_sweep", 6),
    ):
        settings = read_yaml(CONFIGURATION / "plotting" / f"{name}.yaml")
        specs = curve_specs(settings)
        assert len(specs) == count
        specifications.extend(specs)
    assert len({spec["name"] for spec in specifications}) == 18
    spec = next(
        s
        for s in specifications
        if s["name"] == ("neural_r2_vs_nx_horizon4_full")
    )
    rows = [
        dict(spec["where"], nx=nx, mean=value, sem=None)
        for nx, value in ((1, -1), (2, None), (4, 0.5))
    ]
    figure = metric_curve(rows, spec, STYLE, partial=True)
    try:
        axis = figure.axes[0]
        assert axis.get_xscale() == "log"
        assert "R²" in axis.get_ylabel()
        assert np.isnan(axis.lines[0].get_ydata()[1])
        assert "partial" in figure._suptitle.get_text()
        assert not axis.containers
    finally:
        plt.close(figure)
    settings = dict(enabled=True, curves=[spec], metrics=["r2"])
    with pytest.raises(RuntimeError, match="No finite results"):
        plot_suite([], tmp_path, settings)
    assert not list(tmp_path.glob("*.png"))


def test_preview_error_does_not_invalidate_scientific_work():
    from experiments.diagnostics import render_safely

    failures = []

    def fail():
        raise ValueError("No valid preview window")

    render_safely(failures, "preview", fail)
    assert failures == ["preview: ValueError: No valid preview window"]


def test_component_monitor_observes_completion(tmp_path):
    """A spawned renderer reads only published components and reports errors."""
    from experiments.diagnostics import history_monitor

    component = tmp_path / "components" / "main" / "decoder"
    component.mkdir(parents=True)
    rows = [
        dict(
            epoch=i,
            attempt=1,
            metrics=dict(
                loss=1 / i,
                val_loss=2 / i,
            ),
        )
        for i in range(1, 4)
    ]
    history = component / "history.jsonl"
    history.write_text("\n".join(json.dumps(row) for row in rows))
    before = history.read_bytes()
    failures = []
    with history_monitor(tmp_path, STYLE, failures):
        assert not (component / "plots").exists()
        (component / "complete.json").write_text("{}")
    assert not failures
    assert (component / "plots" / "attempt_1" / "total_loss.png").exists()
    assert history.read_bytes() == before


def test_completed_indexed_component_repairs_only_missing_plots(tmp_path):
    """Resume an indexed-output component without redrawing existing PNGs."""
    from experiments.diagnostics import history_monitor

    component = tmp_path / "components" / "main" / "residual"
    component.mkdir(parents=True)
    rows = history_rows(output="1", horizons=(1, 2, 4, 8))
    history = component / "history.jsonl"
    history.write_text("\n".join(json.dumps(row) for row in rows))
    render_component(component, STYLE)
    plots = component / "plots" / "attempt_1"
    expected = {
        "cc_by_horizon.png",
        "loss_by_horizon.png",
        "mse_by_horizon.png",
        "r2_by_horizon.png",
        "total_loss.png",
    }
    assert {path.name for path in plots.glob("*.png")} == expected
    preserved = {
        path.name: path.read_bytes()
        for path in plots.glob("*.png")
        if path.name != "cc_by_horizon.png"
    }
    (plots / "cc_by_horizon.png").unlink()
    completion = component / "complete.json"
    completion.write_text("{}")
    history_before = history.read_bytes()
    completion_before = completion.read_bytes()
    changed_style = dict(STYLE, title_font=STYLE["title_font"] + 1)
    failures = []
    with history_monitor(tmp_path, changed_style, failures):
        pass
    assert not failures
    assert {path.name for path in plots.glob("*.png")} == expected
    assert all(
        (plots / name).read_bytes() == contents
        for name, contents in preserved.items()
    )
    assert history.read_bytes() == history_before
    assert completion.read_bytes() == completion_before
    render_component(component, changed_style, regenerate=True)
    assert (plots / "total_loss.png").read_bytes() != preserved[
        "total_loss.png"
    ]


def test_all_comparison_files_are_generated(tmp_path):
    """Publish all 18 comparison figures using real plot configurations."""
    from itertools import product

    for name, count in (
        ("latent_dimension_sweep", 12),
        ("neural_population_sweep", 6),
    ):
        settings = read_yaml(CONFIGURATION / "plotting" / f"{name}.yaml")
        rows = []
        for nx, scale, horizon, target, metric, scoring in product(
            [1, 2, 4, 8, 16, 32, 64],
            [0.25, 0.5, 1.0],
            [1, 2, 4, 8],
            ["neural", "behavior"],
            ["cc", "r2", "mse"],
            ["full", "common"],
        ):
            rows.append(
                dict(
                    nx=nx,
                    population_scale=scale,
                    horizon=horizon,
                    target=target,
                    metric=metric,
                    evaluation_set=scoring,
                    mean=0.5 + 0.01 * horizon,
                    sem=0.05,
                )
            )
        destination = tmp_path / name / "plots"
        plot_suite(rows, destination, settings)
        assert len(list(destination.glob("*.png"))) == count
        assert not (destination / "behavior_example.png").exists()


@pytest.fixture
def comparison_design():
    """Provide one comparison with two models and two contributing folds."""
    settings = dict(
        enabled=True, metrics=["cc"],
        curves=[dict(
            parameter="nx",
            where=dict(
                target="neural", metric="cc", horizon=4,
                evaluation_set="full",
            ),
        )],
    )
    members = {}
    for nx in (1, 2):
        for fold in (0, 1):
            key = f"nx{nx}/session/fold_{fold}"
            members[key] = dict(
                case=dict(
                    name=f"nx{nx}", dimensions={"nx": nx},
                    population_scale=1.0,
                ),
                state="complete", common_ids=[],
            )
    manifest = dict(
        specification=dict(settings=dict(evaluation=dict(horizons=[4]))),
        members=members,
    )
    rows = [
        dict(
            settings["curves"][0]["where"], nx=nx,
            configuration=f"nx{nx}", population_scale=1.0,
            mean=0.5, sem=None,
        )
        for nx in (1, 2)
    ]
    return settings, manifest, rows


@pytest.mark.parametrize("failed", [False, True])
def test_pending_contributors_silently_defer_figure(
    tmp_path, caplog, comparison_design, failed,
):
    """Existing means and historical failures cannot hide pending folds."""
    from experiments.reporting import expected_comparisons

    settings, manifest, rows = comparison_design
    keys = list(manifest["members"])
    if failed:
        manifest["members"][keys[0]]["state"] = "failed"
    attempted = set(keys[:-1])
    expected = expected_comparisons(manifest, attempted)
    rendered = set()
    plot_suite(rows, tmp_path, settings, expected, rendered=rendered)
    assert not rendered
    assert not list(tmp_path.glob("*.png"))
    assert not caplog.records


def test_failed_point_is_annotated_and_warned_once(
    tmp_path, caplog, comparison_design, monkeypatch,
):
    """A terminal failure preserves the x position without an invented y."""
    from experiments import plots
    from experiments.reporting import expected_comparisons

    settings, manifest, rows = comparison_design
    for key, member in manifest["members"].items():
        if key.startswith("nx2/"):
            member["state"] = "failed"
    saved = []

    def save(figure, path, style):
        saved.append(figure)

    monkeypatch.setattr(plots, "save_figure", save)
    expected = expected_comparisons(manifest, set(manifest["members"]))
    rendered = set()
    for _ in range(2):
        plot_suite(rows[:1], tmp_path, settings, expected, rendered=rendered)
    assert len(saved) == 1
    figure = saved[0]
    try:
        axis = figure.axes[0]
        assert np.isnan(axis.lines[0].get_ydata()[1])
        assert any(line.get_marker() == "x" for line in axis.lines)
        assert any("0/2 results" in text.get_text() for text in axis.texts)
        assert "failed experiments" in figure._suptitle.get_text()
    finally:
        plt.close(figure)
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1
    assert "nx2/session/fold_0" in warnings[0].message
    assert "nx2/session/fold_1" in warnings[0].message


def test_partial_aggregate_marks_reduced_contributions(comparison_design):
    """A finite mean still exposes a failed fold in its aggregate."""
    from experiments import plots
    from experiments.reporting import expected_comparisons

    settings, manifest, rows = comparison_design
    manifest["members"]["nx2/session/fold_1"]["state"] = "failed"
    expected = expected_comparisons(manifest)
    selected = [row for row in expected if row["target"] == "neural"
                and row["metric"] == "cc"]
    attached = plots.comparison_rows(rows, selected, "test")
    figure = metric_curve(attached, curve_specs(settings)[0], STYLE, True)
    try:
        assert any(
            "1/2 results" in text.get_text() for text in figure.axes[0].texts
        )
        assert np.isfinite(figure.axes[0].lines[0].get_ydata()).all()
    finally:
        plt.close(figure)


def test_all_failed_warns_without_empty_figure(
    tmp_path, caplog, comparison_design,
):
    from experiments.reporting import expected_comparisons

    settings, manifest, _ = comparison_design
    for member in manifest["members"].values():
        member["state"] = "failed"
    plot_suite([], tmp_path, settings, expected_comparisons(manifest))
    assert not list(tmp_path.glob("*.png"))
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"


def test_unattempted_historical_failure_defers_until_retry(
    tmp_path, caplog, comparison_design,
):
    from experiments.reporting import expected_comparisons

    settings, manifest, rows = comparison_design
    key = "nx2/session/fold_1"
    manifest["members"][key]["state"] = "failed"
    attempted = set(manifest["members"]) - {key}
    plot_suite(
        rows, tmp_path, settings, expected_comparisons(manifest, attempted)
    )
    assert not caplog.records
    assert not list(tmp_path.glob("*.png"))
    manifest["members"][key]["state"] = "complete"
    attempted.add(key)
    plot_suite(
        rows, tmp_path, settings, expected_comparisons(manifest, attempted)
    )
    assert len(list(tmp_path.glob("*.png"))) == 1
    assert not caplog.records


def test_ready_figure_does_not_wait_for_unrelated_models(
    tmp_path, caplog, comparison_design,
):
    from experiments.reporting import expected_comparisons

    settings, manifest, rows = comparison_design
    settings["curves"] = [dict(
        parameter="horizon", where=dict(
            target="neural", metric="cc", nx=1, evaluation_set="full",
        ),
    )]
    for key, member in manifest["members"].items():
        if key.startswith("nx2/"):
            member["state"] = "pending"
    plot_suite(rows, tmp_path, settings, expected_comparisons(manifest))
    assert len(list(tmp_path.glob("*.png"))) == 1
    assert not caplog.records


def test_unexplained_missing_metric_is_a_rendering_error(
    tmp_path, comparison_design,
):
    from experiments.reporting import expected_comparisons

    settings, manifest, rows = comparison_design
    with pytest.raises(RuntimeError, match="Missing completed result"):
        plot_suite(rows[:1], tmp_path, settings, expected_comparisons(manifest))
    assert not list(tmp_path.glob("*.png"))
