"""Verify the native 12-case sweep and its lossless report views."""

from pathlib import Path

import matplotlib.pyplot as plt

from BRAID.config import resolve_braid_fit_arguments
from experiments.artifacts import (
    PRESENTATION_KEYS, effective_model_configuration, fit_identity,
    resolved_fit,
)
from experiments.braid_backend import BRAIDBackend
from experiments.configuration import read_yaml
from experiments.structure_plots import _figure, zoom_limits
from experiments.structure_report import _raw_rows, _summary_rows
from experiments.structure_sweep import build_cases

CONFIGURATION = Path(__file__).resolve().parents[1] / "assets/config/nhp"
HORIZONS = (1, 2, 4, 8, 16, 32)


def _configured_cases() -> tuple[dict, list[dict]]:
    """Load the tracked portable model and its structural suite."""
    experiment = read_yaml(
        CONFIGURATION / "experiments/nonlinearity_sweep.yaml"
    )
    model = read_yaml(CONFIGURATION / "models/BRAID.yaml")
    return model, build_cases(experiment["suite"])


def test_structure_cases_resolve_distinct_native_recipes() -> None:
    """All 12 settings reach both native BRAID stages."""
    model, cases = _configured_cases()
    assert len(cases) == 12
    assert len({case["name"] for case in cases}) == 12
    assert all(case["dimensions"]["nx"] == 32 for case in cases)
    assert all(case["dimensions"]["n1"] == 16 for case in cases)
    recipes = []
    for case in cases:
        identity = dict(
            configurations=dict(model=model), case=case,
        )
        effective = effective_model_configuration(identity)
        for stage_name in ("stage_1", "stage_2"):
            stage = effective["model"][stage_name]
            assert stage["state_transition"]["architecture"] == (
                "lstm" if case["structure"]["dynamics"] == "lstm"
                else "multilayer_perceptron"
            )
            nonlinear = case["structure"]["encoder"] == "nonlinear"
            assert bool(stage["input_mapping"]["depth"]) == nonlinear
            nonlinear = case["structure"]["decoder"] == "nonlinear"
            for name in ("neural_decoder", "behaviour_decoder"):
                assert bool(stage[name]["depth"]) == nonlinear
        recipes.append(resolve_braid_fit_arguments(effective))
    assert len({repr(recipe) for recipe in recipes}) == len(cases)


def test_structure_enters_canonical_fit_identity() -> None:
    """Each saved effective structure has a distinct numerical fit recipe."""
    model, cases = _configured_cases()
    snapshots = dict(
        model=model, data=dict(features="spike"),
        experiment=dict(
            seed=42,
            model_plugin="experiments.braid_backend:BRAIDBackend",
        ),
    )
    identities = []
    for case in cases:
        request = dict(
            configurations=snapshots, case=case, fold=0,
            source=dict(session="test"), selected_ids=["unit"],
        )
        effective = effective_model_configuration(request)
        canonical = fit_identity(request)
        assert canonical["model"]["model"] == resolved_fit(request)
        saved_arguments = BRAIDBackend.resolve_fit_configuration(
            effective, case["dimensions"]
        )
        for key in PRESENTATION_KEYS:
            saved_arguments["args_base"].pop(key, None)
        assert canonical["model"]["model"] == saved_arguments
        identities.append(canonical["settings_hash"])
    assert len(set(identities)) == len(cases)


def test_existing_decoder_case_keeps_its_numerical_recipe() -> None:
    """The matching historical fit remains eligible for reuse."""
    model, cases = _configured_cases()
    case = next(
        item for item in cases if item["structure"]["dynamics"]
        == "linear_mlp" and item["structure"]["encoder"] == "linear"
        and item["structure"]["decoder"] == "nonlinear"
    )
    current = effective_model_configuration(dict(
        configurations=dict(model=model), case=case,
    ))
    baseline = resolve_braid_fit_arguments(model)
    adapted = resolve_braid_fit_arguments(current)
    assert baseline == adapted


def test_status_tables_keep_missing_members_and_metrics() -> None:
    """Failed members remain rows while existing aggregate means are used."""
    _, cases = _configured_cases()
    complete = dict(
        case=cases[0], session="first", fold=0, state="complete",
    )
    failed = dict(
        case=cases[0], session="second", fold=0, state="failed",
    )
    manifest = dict(
        specification=dict(settings=dict(
            evaluation=dict(horizons=[4]),
        )),
        members=dict(first=complete, second=failed),
    )
    metric = dict(
        behavior=dict(mean_cc=0.5, mean_r2=0.1, mean_mse=0.2),
        neural=dict(mean_cc=0.4, mean_r2=-0.1, mean_mse=0.3),
    )
    completed = {(cases[0]["name"], "first", 0, 4): metric}
    raw = _raw_rows(manifest, completed)
    assert len(raw) == 12
    assert sum(row["status"] == "failed" for row in raw) == 6
    aggregates = [dict(
        configuration=cases[0]["name"], horizon=4, target="behavior",
        metric="cc", evaluation_set="full", mean=0.5, sem=None,
        contributing_sessions=1,
    )]
    summary = _summary_rows(raw, aggregates)
    cc = next(
        row for row in summary
        if row["target"] == "behavior" and row["metric"] == "cc"
    )
    assert cc["mean"] == 0.5
    assert cc["failed_members"] == 1
    assert cc["completed_members"] == 1


def test_zero_iqr_still_marks_values_outside_median() -> None:
    """A repeated central value cannot hide a finite outlier."""
    rows = [
        dict(mean=0.5, sem=0.1) for _ in range(9)
    ] + [dict(mean=-1.0, sem=0.1)]
    low, high, robust_low, robust_high = zoom_limits(rows, 2)
    assert robust_low == robust_high == 0.5
    assert low < 0.4 < 0.6 < high
    assert -1.0 < low


def test_robust_horizon_figure_labels_clipped_metric() -> None:
    """The figure marks an extreme mean while preserving its numeric row."""
    _, cases = _configured_cases()
    rows = []
    for case in cases:
        structure = case["structure"]
        for horizon in HORIZONS:
            mean = horizon / 100
            if (
                structure["dynamics"] == "lstm"
                and structure["encoder"] == "nonlinear"
                and structure["decoder"] == "nonlinear"
                and horizon == 32
            ):
                mean = 10.0
            rows.append(dict(
                **{
                    key: structure[key]
                    for key in ("dynamics", "encoder", "decoder")
                },
                target="behavior", metric="cc", horizon=horizon,
                mean=mean, sem=0.01,
            ))
    low, high, robust_low, robust_high = zoom_limits(rows, 2)
    assert low < high
    assert robust_low < robust_high < 10
    style = read_yaml(
        CONFIGURATION / "plotting/style.yaml"
    )["presentation"]
    figure = _figure(
        rows, "behavior", "cc", style, 20, 2, True,
    )
    try:
        assert len(figure.axes) == 3
        assert any(
            annotation.get_text() == "10"
            for axis in figure.axes
            for annotation in axis.texts
        )
        assert "50" in figure.axes[0].get_xticklabels()[0].get_text()
    finally:
        plt.close(figure)
