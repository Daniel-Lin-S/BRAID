"""Validate neural capacity cases, figure routing and panel series."""

from pathlib import Path

from experiments.artifacts import effective_model_configuration
from experiments.configuration import CONFIGURATION, read_yaml
from experiments.neural_capacity import build_capacity_cases
from experiments.plots import curve_specs, matches, metric_curve, plot_suite
from experiments.presentation import presentation


def capacity_settings() -> tuple[dict, dict]:
    """Load tracked capacity and plotting definitions."""
    experiment = read_yaml(
        CONFIGURATION / "experiments"
        / "neural_population_capacity_sweep.yaml"
    )
    plotting = read_yaml(
        CONFIGURATION / "plotting"
        / "neural_population_capacity_sweep.yaml"
    )
    return experiment, plotting


def test_capacity_cases_and_figure_destinations() -> None:
    """Expand seven model recipes and sixty disjoint figure paths."""
    experiment, plotting = capacity_settings()
    cases = build_capacity_cases(experiment["suite"])
    specs = curve_specs(plotting)
    assert len(cases) == 21
    assert len({(
        case["summary_parameters"]["n1"],
        case["summary_parameters"]["n2"],
        case["summary_parameters"]["neural_decoder_width"],
    ) for case in cases}) == 7
    assert len(specs) == 60
    assert len({
        (spec["subdirectory"], spec["name"]) for spec in specs
    }) == 60
    assert {spec["subdirectory"] for spec in specs} == {
        "nx64_splits", "nx128_splits", "cross_nx_equal_split",
        "cross_nx_three_to_one", "decoder_width",
    }
    for spec in specs:
        assert spec["where"]["horizon"] == [1, 2, 4, 8, 16, 32]
        assert spec["where"]["population_scale"] == [0.25, 0.5, 1.0]
    model = read_yaml(CONFIGURATION / "models" / "BRAID.yaml")
    for case in cases:
        effective = effective_model_configuration(
            {"configurations": {"model": model}, "case": case}
        )
        width = case["summary_parameters"]["neural_decoder_width"]
        assert [
            effective["model"][stage]["neural_decoder"]["hidden_size"]
            for stage in ("stage_1", "stage_2")
        ] == [width, width]


def test_both_orientations_render_all_series(tmp_path: Path) -> None:
    """Draw split panels with all horizons and populations in family folders."""
    experiment, plotting = capacity_settings()
    cases = build_capacity_cases(experiment["suite"])
    specs = [
        spec for spec in curve_specs(plotting)
        if spec["subdirectory"] == "nx64_splits"
        and spec["where"]["target"] == "neural"
        and spec["where"]["metric"] == "cc"
    ]
    assert len(specs) == 2
    rows = []
    for case in cases:
        factors = case["summary_parameters"]
        if factors["nx"] != 64 or factors["neural_decoder_width"] != 64:
            continue
        for horizon in (1, 2, 4, 8, 16, 32):
            rows.append(dict(
                configuration=case["name"],
                population_scale=case["population_scale"],
                **factors,
                horizon=horizon,
                evaluation_set="common",
                target="neural",
                metric="cc",
                mean=0.5 + horizon / 1000,
                sem=None,
                outliers=[],
            ))
    style = presentation(plotting["presentation"])
    for spec in specs:
        selected = [row for row in rows if matches(row, spec["where"])]
        figure = metric_curve(selected, spec, style)
        assert len(figure.axes) == 3
        expected_lines = 6 if spec["group"] == "horizon" else 3
        assert all(
            len(axis.get_lines()) == expected_lines
            for axis in figure.axes
        )
        figure.clear()
    selection = dict(plotting, curves=[
        curve for curve in plotting["curves"]
        if curve["subdirectory"] == "nx64_splits"
        and curve["where"]["target"] == "neural"
    ], metrics=["cc"])
    plot_suite(rows, tmp_path / "plots", selection)
    assert len(list((tmp_path / "plots" / "nx64_splits").glob(
        "*.png"
    ))) == 2
