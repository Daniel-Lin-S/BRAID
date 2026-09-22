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
        ("latent_dimension_sweep", 6),
        ("neural_population_sweep", 6),
    ):
        settings = read_yaml(CONFIGURATION / "plotting" / f"{name}.yaml")
        specs = curve_specs(settings)
        assert len(specs) == count
        specifications.extend(specs)
    assert len({spec["name"] for spec in specifications}) == 12
    spec = dict(
        name="neural_r2_vs_nx_horizon4_full",
        parameter="nx",
        group=None,
        where=dict(
            target="neural",
            metric="r2",
            horizon=4,
            evaluation_set="full",
        ),
    )
    rows = [
        dict(spec["where"], nx=nx, mean=value, sem=None)
        for nx, value in ((1, -1), (2, None), (4, 0.5))
    ]
    figure = metric_curve(rows, spec, STYLE)
    try:
        axis = figure.axes[0]
        assert axis.get_xscale() == "log"
        assert "R²" in axis.get_ylabel()
        assert np.isnan(axis.lines[0].get_ydata()[1])
        assert "partial" not in figure._suptitle.get_text()
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
    """Publish all 12 comparison figures using real plot configurations."""
    from itertools import product

    for name, count in (
        ("latent_dimension_sweep", 6),
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

def test_latent_comparisons_use_panels_and_redundant_group_styles():
    """Both latent views distinguish every series beyond color alone."""
    from itertools import product

    settings = read_yaml(
        CONFIGURATION / "plotting" / "latent_dimension_sweep.yaml"
    )
    specs = curve_specs(settings)
    assert {spec["name"] for spec in specs} == {
        f"{metric}_vs_{parameter}_by_{group}"
        for metric in ("cc", "r2", "mse")
        for parameter, group in (("nx", "horizon"), ("horizon", "nx"))
    }
    rows = [
        dict(
            nx=nx,
            horizon=horizon,
            target=target,
            metric="cc",
            population_scale=1.0,
            evaluation_set="full",
            mean=0.5,
            std=0.05,
        )
        for nx, horizon, target in product(
            [1, 2, 4, 16, 64],
            [1, 2, 4, 8, 16, 32],
            ["neural", "behavior"],
        )
    ]
    for name, count, labels in (
        (
            "cc_vs_nx_by_horizon",
            6,
            [
                "1 step", "2 steps", "4 steps", "8 steps", "16 steps",
                "32 steps",
            ],
        ),
        (
            "cc_vs_horizon_by_nx",
            5,
            ["nx=1", "nx=2", "nx=4", "nx=16", "nx=64"],
        ),
    ):
        spec = next(item for item in specs if item["name"] == name)
        figure = metric_curve(rows, spec, STYLE)
        try:
            assert len(figure.axes) == 2
            np.testing.assert_allclose(
                figure.get_size_inches(), STYLE["horizon_size"]
            )
            for axis, target in zip(figure.axes, ("Neural", "Behavior")):
                lines = [
                    line for line in axis.lines
                    if not line.get_label().startswith("_")
                ]
                assert axis.get_title() == target
                assert len(lines) == count
                assert [line.get_label() for line in lines] == labels
                assert len({line.get_color() for line in lines}) == count
                assert len({line.get_marker() for line in lines}) == count
                assert len({line.get_linestyle() for line in lines}) >= 4
            if spec["parameter"] == "nx":
                assert all(
                    axis.get_xscale() == "log" for axis in figure.axes
                )
        finally:
            plt.close(figure)


def _residual_plot_rows(
    horizons: tuple[int, ...] = (1, 2),
) -> list[dict]:
    """Build complete residual-grid summaries for both target panels."""
    splits = (
        (1, 1, 0),
        (2, 2, 0),
        (4, 4, 0),
        (8, 8, 0),
        (16, 16, 0),
        (32, 32, 0),
        (64, 64, 0),
        (8, 4, 4),
        (16, 8, 8),
        (32, 16, 16),
        (64, 16, 48),
    )
    return [
        dict(
            nx=nx,
            n1=n1,
            n2=n2,
            residual=n2 > 0,
            horizon=horizon,
            target=target,
            metric="cc",
            population_scale=1.0,
            evaluation_set="full",
            mean=0.5,
            std=0.05,
        )
        for nx, n1, n2 in splits
        for horizon in horizons
        for target in ("neural", "behavior")
    ]

def test_residual_comparisons_branch_from_resolved_dimensions():
    """Residual figure branches derive from n1/n2 without fixed thresholds."""
    settings = read_yaml(
        CONFIGURATION / "plotting" / "latent_dimension_residual_sweep.yaml"
    )
    specs = curve_specs(settings)
    rows = _residual_plot_rows()
    nx_spec = next(
        spec for spec in specs
        if spec["name"] == "cc_vs_nx_by_horizon"
    )
    horizon_spec = next(
        spec for spec in specs
        if spec["name"] == "cc_vs_horizon_by_nx"
    )
    nx_figure = metric_curve(rows, nx_spec, STYLE)
    horizon_figure = metric_curve(rows, horizon_spec, STYLE)
    try:
        nx_lines = [
            line for line in nx_figure.axes[0].lines
            if not line.get_label().startswith("_")
        ]
        assert [line.get_label() for line in nx_lines] == [
            "1 step, main only",
            "1 step, residual",
            "2 steps, main only",
            "2 steps, residual",
        ]
        np.testing.assert_allclose(
            nx_lines[1].get_xdata(), [4, 8, 16, 32, 64]
        )
        assert nx_lines[0].get_color() == nx_lines[1].get_color()
        assert nx_lines[0].get_linestyle() != nx_lines[1].get_linestyle()
        horizon_lines = [
            line for line in horizon_figure.axes[0].lines
            if not line.get_label().startswith("_")
        ]
        assert len(horizon_lines) == 11
        assert "nx=8, n1=4, n2=4" in {
            line.get_label() for line in horizon_lines
        }
    finally:
        plt.close(nx_figure)
        plt.close(horizon_figure)

def test_residual_branching_adapts_to_grid_and_rejects_ambiguity():
    """Branching follows rows and identifies conflicting residual members."""
    settings = read_yaml(
        CONFIGURATION / "plotting" / "latent_dimension_residual_sweep.yaml"
    )
    spec = next(
        spec for spec in curve_specs(settings)
        if spec["name"] == "cc_vs_nx_by_horizon"
    )
    rows = [
        dict(
            nx=nx,
            n1=n1,
            n2=n2,
            residual=n2 > 0,
            horizon=1,
            target="neural",
            metric="cc",
            population_scale=1.0,
            evaluation_set="full",
            mean=0.5,
            std=0.05,
        )
        for nx, n1, n2 in ((1, 1, 0), (2, 2, 0), (4, 2, 2))
    ]
    figure = metric_curve(rows, spec, STYLE)
    try:
        lines = [
            line for line in figure.axes[0].lines
            if not line.get_label().startswith("_")
        ]
        np.testing.assert_allclose(lines[1].get_xdata(), [2, 4])
    finally:
        plt.close(figure)
    no_residual = [row for row in rows if row["n2"] == 0]
    figure = metric_curve(no_residual, spec, STYLE)
    try:
        lines = [
            line for line in figure.axes[0].lines
            if not line.get_label().startswith("_")
        ]
        assert [line.get_label() for line in lines] == ["1 step, main only"]
    finally:
        plt.close(figure)
    ambiguous = rows + [dict(rows[-1], n1=1, n2=3)]
    with pytest.raises(ValueError, match="Ambiguous residual branch"):
        metric_curve(ambiguous, spec, STYLE)

def test_panel_curve_change_invalidates_existing_png(tmp_path):
    """Missing-only rendering replaces figures with changed semantics."""
    from copy import deepcopy
    from itertools import product

    settings = read_yaml(
        CONFIGURATION / "plotting" / "latent_dimension_sweep.yaml"
    )
    settings["metrics"] = ["cc"]
    settings["curves"] = settings["curves"][:1]
    rows = [
        dict(
            nx=nx,
            horizon=horizon,
            target=target,
            metric="cc",
            population_scale=1.0,
            evaluation_set="full",
            mean=0.5,
            sem=0.05,
        )
        for nx, horizon, target in product(
            [1, 2], [1, 2], ["neural", "behavior"]
        )
    ]
    plot_suite(rows, tmp_path, settings)
    path = tmp_path / "cc_vs_nx_by_horizon.png"
    original = path.read_bytes()
    changed = deepcopy(settings)
    changed["curves"][0]["where"]["target"].reverse()
    plot_suite(rows, tmp_path, changed)
    assert path.read_bytes() != original

def test_panel_style_change_requires_full_regeneration(tmp_path):
    """Presentation changes redraw comprehensive figures only when explicit."""
    from copy import deepcopy
    from itertools import product

    settings = read_yaml(
        CONFIGURATION / "plotting" / "latent_dimension_sweep.yaml"
    )
    settings["metrics"] = ["cc"]
    settings["curves"] = settings["curves"][:1]
    rows = [
        dict(
            nx=nx,
            horizon=horizon,
            target=target,
            metric="cc",
            population_scale=1.0,
            evaluation_set="full",
            mean=0.5,
            sem=0.05,
        )
        for nx, horizon, target in product(
            [1, 2], [1, 2], ["neural", "behavior"]
        )
    ]
    plot_suite(rows, tmp_path, settings)
    path = tmp_path / "cc_vs_nx_by_horizon.png"
    original = path.read_bytes()
    changed = deepcopy(settings)
    changed["presentation"]["title_font"] += 1
    plot_suite(rows, tmp_path, changed)
    assert path.read_bytes() == original
    plot_suite(rows, tmp_path, changed, regenerate=True)
    assert path.read_bytes() != original

def test_panel_metric_error_does_not_block_other_metrics(
    tmp_path, monkeypatch,
):
    """Each configured panel metric retains an independent render boundary."""
    from itertools import product

    from experiments import plots

    settings = read_yaml(
        CONFIGURATION / "plotting" / "latent_dimension_sweep.yaml"
    )
    rows = [
        dict(
            nx=nx,
            horizon=horizon,
            target=target,
            metric=metric,
            population_scale=1.0,
            evaluation_set="full",
            mean=0.5,
            sem=0.05,
        )
        for nx, horizon, target, metric in product(
            [1, 2],
            [1, 2],
            ["neural", "behavior"],
            ["cc", "r2", "mse"],
        )
    ]
    original = plots.metric_curve

    def fail_r2(selected, spec, style):
        if spec["where"]["metric"] == "r2":
            raise ValueError("synthetic R2 comparison failure")
        return original(selected, spec, style)

    monkeypatch.setattr(plots, "metric_curve", fail_r2)
    with pytest.raises(RuntimeError, match="synthetic R2 comparison failure"):
        plot_suite(rows, tmp_path, settings)
    assert {path.name for path in tmp_path.glob("*.png")} == {
        "cc_vs_nx_by_horizon.png",
        "mse_vs_nx_by_horizon.png",
        "cc_vs_horizon_by_nx.png",
        "mse_vs_horizon_by_nx.png",
    }


def evaluation_context() -> dict:
    """Return complete diagnostic identifiers for one synthetic evaluation."""
    return dict(
        session="session_a",
        fold=2,
        model="BRAID_nx4_p1",
        horizon=4,
        evaluation_set="full",
        target="neural",
    )

def test_flat_channels_are_ignored_with_context(caplog):
    """Flat truth is expected exclusion, distinct from invalid metric output."""
    from experiments.evaluation import score_channels

    truth = np.array([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
    predicted = np.array([[0.0, 2.0], [1.0, 2.0], [2.0, 2.0]])
    scores = score_channels(
        truth, predicted, ["unit_a", "unit_flat"], evaluation_context()
    )
    assert scores["valid_cc_channels"] == 1
    assert scores["valid_r2_channels"] == 1
    assert scores["per_dimension_cc"][1] is None
    assert scores["per_dimension_r2"][1] is None
    assert scores["mean_cc"] == pytest.approx(
        scores["per_dimension_cc"][0]
    )
    messages = [record.message for record in caplog.records]
    assert len(messages) == 1
    assert "observed range is zero" in messages[0]
    assert "session=session_a" in messages[0]
    assert "fold=2" in messages[0]
    assert "model=BRAID_nx4_p1" in messages[0]
    assert "horizon=4" in messages[0]
    assert "evaluation_set=full" in messages[0]
    assert "target=neural" in messages[0]
    assert "affected_count=1" in messages[0]
    assert "total_channels=2" in messages[0]
    assert "unit_flat" in messages[0]

def test_nonfinite_nonflat_metric_has_distinct_warning(caplog, monkeypatch):
    """Unexpected metric failures identify their metric, value and channel."""
    from experiments import evaluation

    def metric(_truth, _predicted, measure):
        return [np.nan, 0.5] if measure == "CC" else [0.25, 0.5]

    monkeypatch.setattr(evaluation, "evalPrediction", metric)
    truth = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 4.0]])
    scores = evaluation.score_channels(
        truth, truth, ["unit_bad", "unit_ok"], evaluation_context()
    )
    assert scores["mean_cc"] == pytest.approx(0.5)
    assert scores["valid_cc_channels"] == 1
    messages = [record.message for record in caplog.records]
    assert len(messages) == 1
    assert "Unexpected nonfinite CC" in messages[0]
    assert "truth range is nonzero" in messages[0]
    assert "affected_count=1" in messages[0]
    assert "total_channels=2" in messages[0]
    assert "unit_bad" in messages[0]
    assert "nan" in messages[0]
    assert "session=session_a" in messages[0]

def test_per_session_fold_statistics_use_sample_standard_deviation():
    """Per-session plotting values remain in memory and vary across folds."""
    from experiments.evaluation import aggregate_folds

    rows = []
    for fold, value in enumerate((1.0, 3.0, 5.0)):
        score = {
            f"mean_{metric}": value for metric in ("cc", "r2", "mse")
        }
        rows.append(dict(
            session="session_a",
            fold=fold,
            configuration="BRAID_nx4_p1",
            population_scale=1.0,
            nx=4,
            n1=4,
            horizon=4,
            evaluation_set="full",
            neural=score,
            behavior=score,
        ))
    summaries = aggregate_folds(rows)
    neural_cc = next(
        row for row in summaries
        if row["target"] == "neural" and row["metric"] == "cc"
    )
    assert neural_cc["mean"] == pytest.approx(3.0)
    assert neural_cc["std"] == pytest.approx(2.0)
    assert neural_cc["folds"] == 3


def test_per_session_fold_statistics_report_missing_metric_member():
    """Average defined folds while retaining the unavailable fold identity."""
    from experiments.evaluation import aggregate_folds

    rows = []
    for fold, value in enumerate((1.0, None, 5.0)):
        score = dict(mean_cc=value, mean_r2=0.5, mean_mse=0.25)
        rows.append(dict(
            session="session_a",
            fold=fold,
            configuration="BRAID_nx4_p1",
            population_scale=1.0,
            nx=4,
            n1=4,
            horizon=4,
            evaluation_set="full",
            neural=score,
            behavior=score,
        ))
    summaries = aggregate_folds(rows)
    neural_cc = next(
        row for row in summaries
        if row["target"] == "neural" and row["metric"] == "cc"
    )
    assert neural_cc["mean"] == pytest.approx(3.0)
    assert neural_cc["std"] == pytest.approx(np.sqrt(8.0))
    assert neural_cc["folds"] == 2
    assert neural_cc["missing_members"] == [
        "BRAID_nx4_p1/session_a/fold_1"
    ]


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
                state="complete", common_ids=[], session="session",
            )
    manifest = dict(
        specification=dict(settings=dict(evaluation=dict(horizons=[4]))),
        members=members,
    )
    rows = [
        dict(
            settings["curves"][0]["where"], nx=nx,
            configuration=f"nx{nx}", population_scale=1.0,
            session="session",
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


def test_session_readiness_does_not_wait_for_pending_sessions(
    tmp_path, comparison_design,
):
    """Terminal sessions render while later sessions remain silent."""
    from copy import deepcopy

    from experiments.reporting import expected_comparisons

    settings, manifest, rows = comparison_design
    pending = {}
    for key, member in manifest["members"].items():
        copied = deepcopy(member)
        copied["session"] = "later_session"
        copied["state"] = "pending"
        pending[key.replace("/session/", "/later_session/")] = copied
    manifest["members"].update(pending)
    expected = expected_comparisons(manifest, per_session=True)
    current = [row for row in expected if row["session"] == "session"]
    later = [row for row in expected if row["session"] == "later_session"]
    plot_suite(
        rows,
        tmp_path / "sessions" / "session",
        settings,
        current,
        namespace="session/session",
    )
    plot_suite(
        [],
        tmp_path / "sessions" / "later_session",
        settings,
        later,
        namespace="session/later_session",
    )
    assert len(list((tmp_path / "sessions" / "session").glob("*.png"))) == 1
    assert not (tmp_path / "sessions" / "later_session").exists()

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

    def save(figure, path, style, signature):
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
        assert not any(line.get_marker() == "x" for line in axis.lines)
        assert not axis.texts
        assert "partial" not in figure._suptitle.get_text()
    finally:
        plt.close(figure)
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1
    assert "nx2/session/fold_0" in warnings[0].message
    assert "nx2/session/fold_1" in warnings[0].message


def test_completed_reattempt_replaces_partial_comparison(
    tmp_path, comparison_design,
):
    """A changed dependency state invalidates an existing partial PNG."""
    from experiments.reporting import expected_comparisons

    settings, manifest, rows = comparison_design
    failed_key = "nx2/session/fold_1"
    manifest["members"][failed_key]["state"] = "failed"
    expected = expected_comparisons(manifest)
    plot_suite(rows, tmp_path, settings, expected)
    figure = next(tmp_path.glob("*.png"))
    partial = figure.read_bytes()

    manifest["members"][failed_key]["state"] = "complete"
    plot_suite(rows, tmp_path, settings, expected_comparisons(manifest))
    repaired = figure.read_bytes()
    assert repaired != partial

    plot_suite(rows, tmp_path, settings, expected_comparisons(manifest))
    assert figure.read_bytes() == repaired

def test_partial_aggregate_keeps_finite_contributions(comparison_design):
    """A finite mean remains available when one fold failed."""
    from experiments import plots
    from experiments.reporting import expected_comparisons

    settings, manifest, rows = comparison_design
    manifest["members"]["nx2/session/fold_1"]["state"] = "failed"
    expected = expected_comparisons(manifest)
    selected = [row for row in expected if row["target"] == "neural"
                and row["metric"] == "cc"]
    attached = plots.comparison_rows(rows, selected, "test")
    figure = metric_curve(attached, curve_specs(settings)[0], STYLE)
    try:
        assert not figure.axes[0].texts
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

def test_missing_completed_metric_is_annotated_and_warned(
    tmp_path, caplog, comparison_design, monkeypatch,
):
    """Keep finite points and mark a completed member's unavailable metric."""
    from experiments import plots
    from experiments.reporting import expected_comparisons

    settings, manifest, rows = comparison_design
    saved = []
    monkeypatch.setattr(
        plots,
        "save_figure",
        lambda figure, path, style, signature: saved.append(figure),
    )
    plot_suite(
        rows[:1], tmp_path, settings, expected_comparisons(manifest)
    )
    assert len(saved) == 1
    figure = saved[0]
    try:
        axis = figure.axes[0]
        assert np.isnan(axis.lines[0].get_ydata()[1])
        assert not any(line.get_marker() == "x" for line in axis.lines)
        assert not axis.texts
        assert "partial" not in figure._suptitle.get_text()
    finally:
        plt.close(figure)
    warnings = [
        record for record in caplog.records if record.levelname == "WARNING"
    ]
    assert len(warnings) == 1
    assert "nx2/session/fold_0" in warnings[0].message
    assert "nx2/session/fold_1" in warnings[0].message

def canonical_history_rows(r2_values: tuple[float | None, ...]) -> list[dict]:
    """Build direct component metrics with configurable R2 history values."""
    return [
        {
            "epoch": epoch,
            "attempt": 1,
            "metrics": {
                "loss": 1.0 / epoch,
                "val_loss": 2.0 / epoch,
                "MSE": 0.5 / epoch,
                "val_MSE": 0.75 / epoch,
                "R2": r2,
                "val_R2": r2,
                "CC": 0.25 * epoch,
                "val_CC": 0.2 * epoch,
            },
        }
        for epoch, r2 in enumerate(r2_values, 1)
    ]

def test_all_gap_r2_is_recorded_without_suppressing_cc(tmp_path, caplog):
    """An intentionally omitted R2 plot does not block independent metrics."""
    component = tmp_path / "components" / "main" / "decoder"
    component.mkdir(parents=True)
    history = component / "history.jsonl"
    rows = canonical_history_rows((None, None, None))
    history.write_text("\n".join(json.dumps(row) for row in rows))

    render_component(component, STYLE)

    plots = component / "plots" / "attempt_1"
    assert (plots / "total_loss.png").is_file()
    assert (plots / "mse.png").is_file()
    assert (plots / "cc.png").is_file()
    assert not (plots / "r2.png").exists()
    record = json.loads((plots / "rendering.json").read_text())
    assert record["skipped"]["r2"]["reason"] == "no_finite_epoch_values"
    warnings = [
        message
        for message in (item.message for item in caplog.records)
        if "Omitting all-gap history figure" in message
    ]
    assert len(warnings) == 1
    assert "metric=R2" in warnings[0]

    caplog.clear()
    render_component(component, STYLE)
    assert not [
        item
        for item in caplog.records
        if "Omitting all-gap history figure" in item.message
    ]

    rows = canonical_history_rows((0.1, 0.2, 0.3))
    history.write_text("\n".join(json.dumps(row) for row in rows))
    render_component(component, STYLE)
    assert (plots / "r2.png").is_file()

def test_metric_render_error_does_not_block_later_metrics(
    tmp_path,
    monkeypatch,
):
    """A rendering exception is combined after unaffected figures are saved."""
    from experiments import history as history_module

    component = tmp_path / "components" / "main" / "decoder"
    component.mkdir(parents=True)
    rows = canonical_history_rows((0.1, 0.2, 0.3))
    (component / "history.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows)
    )
    original = history_module.history_figure

    def fail_r2(*args, **kwargs):
        if args[1] == "R2":
            raise ValueError("synthetic R2 renderer failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(history_module, "history_figure", fail_r2)
    with pytest.raises(RuntimeError, match="r2: synthetic R2 renderer failure"):
        render_component(component, STYLE)

    plots = component / "plots" / "attempt_1"
    assert (plots / "total_loss.png").is_file()
    assert (plots / "mse.png").is_file()
    assert (plots / "cc.png").is_file()
    assert not (plots / "r2.png").exists()


def outlier_rows() -> list[dict]:
    """Build completed rows with one selected finite R2/MSE outlier."""
    rows = []
    for fold, r2, mse in ((0, -1000.0, 10000.0), (2, 2.0, 4.0), (4, 4.0, 8.0)):
        score = dict(mean_cc=0.5, mean_r2=r2, mean_mse=mse)
        rows.append(dict(
            session="indy_20160630_01",
            fold=fold,
            configuration="BRAID_nx4_p1",
            population_scale=1.0,
            nx=4,
            n1=4,
            horizon=32,
            evaluation_set="full",
            neural=dict(score),
            behavior=dict(score),
        ))
    return rows


def outlier_settings() -> dict:
    """Provide the approved generic finite-outlier selector shape."""
    return dict(
        outliers=[dict(
            member="BRAID_nx4_p1/indy_20160630_01/fold_0",
            horizon=32,
            evaluation_set="full",
            targets=["neural", "behavior"],
            metrics=["r2", "mse"],
            aggregate_exclude=True,
            session_zoom=True,
            reason="Synthetic finite numerical outlier",
        )]
    )


def test_outlier_policy_preserves_raw_rows_and_filters_only_selected_metrics(
    tmp_path, caplog,
):
    """Aggregate/session values exclude only selected finite R2/MSE."""
    from experiments.evaluation import aggregate, aggregate_folds
    from experiments.outliers import resolve_outliers

    rows = outlier_rows()
    original = json.loads(json.dumps(rows))
    rules = resolve_outliers(outlier_settings(), rows)
    caplog.set_level("WARNING", logger="experiments.evaluation")
    summaries = aggregate(rows, tmp_path, rules)
    r2 = next(
        row for row in summaries
        if row["target"] == "neural" and row["metric"] == "r2"
    )
    cc = next(
        row for row in summaries
        if row["target"] == "neural" and row["metric"] == "cc"
    )
    assert r2["mean"] == pytest.approx(3.0)
    assert r2["metric_counts"] == {"indy_20160630_01": 2}
    assert r2["excluded_members"] == [
        "BRAID_nx4_p1/indy_20160630_01/fold_0"
    ]
    assert r2["excluded_contribution_count"] == 1
    assert cc["mean"] == pytest.approx(0.5)
    assert not cc["outlier_exclusions"]
    assert rows == original
    assert json.loads((tmp_path / "raw_metrics.json").read_text()) == original
    assert any("Excluding 1 finite outlier" in record.message
               for record in caplog.records)
    session = next(
        row for row in aggregate_folds(rows, rules)
        if row["target"] == "neural" and row["metric"] == "r2"
    )
    assert session["mean"] == pytest.approx(3.0)
    assert session["std"] == pytest.approx(np.sqrt(2.0))
    assert session["folds"] == 2
    assert session["outliers"][0]["value"] == -1000.0


def test_outlier_arrow_uses_normal_range_and_three_significant_figures():
    """Session arrows retain the value while keeping normal values legible."""
    spec = dict(
        name="neural_r2_vs_horizon_nx4_full",
        parameter="horizon",
        where=dict(target="neural", metric="r2", nx=4,
                   evaluation_set="full"),
    )
    rows = [
        dict(horizon=4, nx=4, mean=1.0, std=0.1),
        dict(
            horizon=32,
            nx=4,
            mean=3.0,
            std=np.sqrt(2.0),
            outliers=[dict(
                member="BRAID_nx4_p1/indy_20160630_01/fold_0",
                value=-1000.0,
                reason="Synthetic finite numerical outlier",
            )],
        ),
    ]
    figure = metric_curve(rows, spec, STYLE)
    try:
        axis = figure.axes[0]
        assert axis.get_ylim()[0] > -1000.0
        assert [item.get_text() for item in axis.texts] == ["↓ -1e+03"]
    finally:
        plt.close(figure)


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        (dict(outliers=[outlier_settings()["outliers"][0]] * 2), "Duplicate"),
        (dict(outliers=[dict(outlier_settings()["outliers"][0], horizon=4)]),
         "Unmatched"),
    ],
)
def test_outlier_selectors_reject_duplicate_and_unmatched_rules(
    settings, message,
):
    """Explicit plotting selectors cannot silently miss or overlap raw rows."""
    from experiments.outliers import resolve_outliers

    with pytest.raises(ValueError, match=message):
        resolve_outliers(settings, outlier_rows())


def test_outlier_selectors_reject_malformed_and_overlapping_rules():
    """Rules need a complete schema and must select disjoint metric values."""
    from experiments.outliers import resolve_outliers

    malformed = dict(outlier_settings()["outliers"][0])
    malformed.pop("reason")
    with pytest.raises(ValueError, match="fields are invalid"):
        resolve_outliers(dict(outliers=[malformed]), outlier_rows())
    first = dict(outlier_settings()["outliers"][0], targets=["neural"])
    second = dict(
        outlier_settings()["outliers"][0],
        targets=["neural", "behavior"],
    )
    with pytest.raises(ValueError, match="Overlapping"):
        resolve_outliers(dict(outliers=[first, second]), outlier_rows())


def test_session_outlier_exclusion_requires_a_normal_fold_range():
    """A session renderer cannot invent a range after excluding all folds."""
    from experiments.evaluation import aggregate_folds
    from experiments.outliers import resolve_outliers

    rules = []
    for fold in (0, 2, 4):
        rules.append(dict(
            member=f"BRAID_nx4_p1/indy_20160630_01/fold_{fold}",
            horizon=32,
            evaluation_set="full",
            targets=["neural"],
            metrics=["r2"],
            aggregate_exclude=True,
            session_zoom=True,
            reason="Synthetic finite numerical outlier",
        ))
    rows = outlier_rows()
    with pytest.raises(ValueError, match="No finite normal fold metrics"):
        aggregate_folds(rows, resolve_outliers(dict(outliers=rules), rows))


def test_outlier_arrows_stack_same_edge_labels_by_nx():
    """Same-position finite outliers retain readable deterministic labels."""
    spec = dict(
        name="neural_r2_vs_horizon_by_nx",
        parameter="horizon",
        group="nx",
        panel="target",
        where=dict(target=["neural", "behavior"], metric="r2",
                   horizon=[8], evaluation_set="full"),
    )
    rows = []
    for nx, value in zip((1, 2, 4, 16, 64),
                         (-1000.0, -2000.0, -3000.0, -4000.0, -5000.0)):
        rows.append(dict(
            horizon=8,
            nx=nx,
            target="neural",
            mean=-0.02,
            std=0.01,
            outliers=[dict(
                member=f"BRAID_nx{nx}_p1/session/fold_2",
                value=value,
                reason="Synthetic finite numerical outlier",
            )],
        ))
    figure = metric_curve(rows, spec, STYLE)
    try:
        texts = figure.axes[0].texts
        assert [item.get_text() for item in texts] == [
            "↓ -1e+03",
            "↓ -2e+03",
            "↓ -3e+03",
            "↓ -4e+03",
            "↓ -5e+03",
        ]
        assert [item.xyann[1] for item in texts] == [8, 22, 36, 50, 64]
    finally:
        plt.close(figure)
