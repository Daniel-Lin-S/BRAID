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


def history_rows():
    """Produce canonical totals and all eight step-wise metric families."""
    return [
        dict(
            epoch=epoch,
            attempt=1,
            metrics={
                **{prefix + "loss": 9 / epoch for prefix in ("", "val_")},
                **{
                    f"{prefix}rnn_{horizon}step_{metric}": horizon / epoch
                    for prefix in ("", "val_")
                    for horizon in range(1, 9)
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
