"""Contracts for filtering and residual sharing sensitivities.

Inputs are synthetic unequal-width source windows and fake reference jobs.
Outputs are assertions about shared weights, residual priors, fit identities,
and immutable sensitivity memberships. No real artifacts are changed.
"""

from copy import deepcopy
from pathlib import Path

import numpy as np
import tensorflow as tf

from BRAID.multisession import HORIZONS, LINEAR, MixedForecaster
from BRAID.RNNModel import RNNModel
from experiments.multisession.source import fit_source, residual_data
from experiments.multisession.store import load_pickle
from experiments.multisession.training import recurrent_data, select_batch

LENGTH = 128
SOURCE_UPDATES = 2


def _windows() -> list[dict]:
    """Create unequal-width sessions with optional irrelevant behavior."""
    rng = np.random.default_rng(7)
    return [
        dict(
            Y=rng.normal(size=(2, LENGTH, width)).astype("float32"),
            U=rng.normal(size=(2, LENGTH, 2)).astype("float32"),
            Z=rng.normal(size=(2, LENGTH, 4)).astype("float32"),
        )
        for width in (3, 5)
    ]


def test_shared_filter_and_native_freeze() -> None:
    """A and forecast maps share variables and remain frozen natively."""
    model = MixedForecaster([(3, 3), (5, 5)], 4, "AAK", 42)
    assert set(model.transferable()) == {"A", "Afw", "Kfw"}
    for name in ("A", "Afw", "Kfw"):
        assert model.sessions[0][name] is model.sessions[1][name]
    data = [recurrent_data(row) for row in _windows()]
    batch = select_batch(
        data, [np.array([0, 1]), np.array([], dtype=int)]
    )
    state, _ = model.states(batch[0])
    assert state.shape == (2, LENGTH, 4)
    absent = [v.numpy().copy() for v in model.private_variables[1]]
    loss, gradients = model.loss_and_gradients(*batch)
    assert np.isfinite(loss.numpy())
    assert all(np.isfinite(g.numpy()).all() for g in gradients)
    for before, variable in zip(absent, model.private_variables[1]):
        np.testing.assert_array_equal(before, variable.numpy())
    native = RNNModel(
        nx=4, ny=5, ny_out=3, block_samples=LENGTH,
        batch_size=2, n1_in=2, steps_ahead=HORIZONS,
        enable_forward_pred=True,
        cell_args=dict(
            ASettings=deepcopy(LINEAR),
            KSettings=deepcopy(LINEAR),
            CSettings=deepcopy(LINEAR),
        ),
    )
    native.set_frozen_modules(model.transferable())
    frozen = {
        variable.ref()
        for name in ("A", "Afw", "Kfw")
        for variable in getattr(native.rnn.cell, name).weights
    }
    assert not frozen & {
        variable.ref() for variable in native.model.trainable_weights
    }


def test_neural_residual_prior_and_source_serialization(
    tmp_path: Path,
) -> None:
    """Stage 2 consumes stage-1 predictions and exports both stages."""
    values = _windows()
    input_rows = [recurrent_data(row) for row in values]
    main = MixedForecaster([(3, 3), (5, 5)], 4, "AAK", 42)
    result = residual_data(main, None, input_rows)
    inputs, forward, _, _ = select_batch(
        input_rows, [np.arange(2), np.arange(2)]
    )
    expected, _ = main.forecast(inputs, forward)
    for index, row in enumerate(result):
        np.testing.assert_allclose(row["P"], expected[index].numpy())
        assert row["I"].shape == (2, LENGTH, values[index]["Y"].shape[-1] + 6)
        assert row["F"].shape == (2, LENGTH, 6, 8)
    directory = tmp_path / "fit"
    directory.mkdir()
    identity = dict(
        latent=4, sharing="AAK2", seed=42,
        objective="neural_residual",
    )
    fit_source(
        values, values, directory, tmp_path / "source_components",
        identity,
        dict(
            maximum_updates=SOURCE_UPDATES * 2,
            batch_size=2,
            learning_rate=0.001,
            validation_interval=1,
            patience=10,
        ),
    )
    exported = load_pickle(directory / "shared.pkl")
    assert set(exported) == {"stage1", "stage2"}
    for weights in exported.values():
        assert set(weights) == {"A", "Afw", "Kfw"}
    import json

    records = json.loads((directory / "components.json").read_text())
    assert set(records) == {"neural", "residual"}
    for record in records.values():
        assert record["training"]["settings"]["maximum_updates"] == (
            SOURCE_UPDATES
        )


def test_sensitivity_plan_reuses_completed_reference(
    tmp_path: Path, monkeypatch,
) -> None:
    """Each root references old IDs and schedules only two source fits."""
    from experiments.configuration import read_yaml
    from experiments.multisession.plan import build_plan
    from experiments.multisession import sensitivity

    settings = read_yaml(
        Path(__file__).resolve().parents[1]
        / "assets/config/nhp/experiments/multi_session_training.yaml"
    )
    for key in ("data", "model"):
        settings[key] = read_yaml(Path(settings["modules"][key]))
    settings["paths"] = dict(artifact_root=str(tmp_path))
    sessions = {
        name: dict(sha256=name)
        for name in settings["held_out_sessions"]
    }
    sessions.update({
        f"source_{i}": dict(sha256=str(i)) for i in range(33)
    })
    reference = build_plan(settings, sessions)
    reference["storage"] = dict(reference["storage"])
    current = dict(settings, name=sensitivity.CAMPAIGN)
    monkeypatch.setattr(
        sensitivity, "completed", lambda directory: dict(checksums={"x": "y"})
    )
    result = sensitivity.build_sensitivity_plan(
        current, sessions, reference
    )
    old = set(reference["jobs"])
    new_sources = {
        key for key, spec in result["jobs"].items()
        if key not in old and spec["kind"] == "source"
    }
    new_adaptations = {
        key for key, spec in result["jobs"].items()
        if key not in old and spec["kind"] == "adaptation"
    }
    assert len(new_sources) == 2
    assert len(new_adaptations) == 72
    assert all(
        len(members) == 108 for members in result["analyses"].values()
    )
    for analysis, profiles in (
        (sensitivity.FILTER, {"scratch", "AK", "AAK"}),
        (sensitivity.RESIDUAL, {"scratch", "AAK", "AAK2"}),
    ):
        assert {
            result["jobs"][key]["sharing"]
            for key in result["analyses"][analysis]
        } == profiles


def test_nested_sensitivity_figures(tmp_path: Path) -> None:
    """Only named comparisons and scratch appear in new figure trees."""
    from experiments.multisession.sensitivity import FILTER, RESIDUAL
    from experiments.visualization.sharing_sensitivity import (
        source_figures, target_figures,
    )

    summary = []
    for target in ("neural", "behavior"):
        for profile in ("AK", "AAK", "AAK2", "scratch"):
            for fraction in (0.05, 0.2, 0.8):
                summary.append(dict(
                    target=target, metric="cc", horizon=1,
                    sharing=profile, fraction=fraction,
                    mean=0.3, sem=0.01,
                ))
    settings = dict(
        target_fractions=[0.05, 0.2, 0.8],
        evaluation_horizons=[1],
    )
    for analysis in (FILTER, RESIDUAL):
        root = tmp_path / analysis
        target_figures(summary, analysis, root, settings)
        for target in ("neural", "behavior"):
            assert (
                root / "target_transfer" / target / "h1.png"
            ).stat().st_size > 0
    metrics = [
        dict(
            sharing=profile, source_count=count,
            horizon=horizon, cc=0.3,
        )
        for profile, count in (
            ("A", 4), ("A", 8), ("AK", 4), ("AK", 8),
            ("AAK", 4), ("AAK2", 4),
        )
        for horizon in HORIZONS
    ]
    losses = [
        dict(
            sharing=profile, source_count=count,
            macro_validation_loss=2.0,
        )
        for profile, count in (
            ("A", 4), ("A", 8), ("AK", 4), ("AK", 8),
            ("AAK", 4), ("AAK2", 4),
        )
    ]
    directory = tmp_path / "source_validation"
    source_figures(metrics, losses, directory)
    assert (directory / "macro_loss.png").stat().st_size > 0
    for horizon in HORIZONS:
        assert (
            directory / f"common_four_neural_h{horizon}.png"
        ).stat().st_size > 0


def test_nested_report_payload_and_validation(
    tmp_path: Path, monkeypatch,
) -> None:
    """New roots keep tables and figures structured and checksum-ready."""
    from experiments.multisession import reports, source_validation
    from experiments.multisession.sensitivity import FILTER
    from experiments.visualization.sharing_sensitivity import HORIZONS

    rows = []
    for target in ("neural", "behavior"):
        for sharing in ("AK", "AAK", "scratch"):
            for fraction in (0.05, 0.2, 0.8):
                rows.append(dict(
                    session="target", fold=0, seed=42, fraction=fraction,
                    latent=32, objective="scratch" if sharing == "scratch"
                    else "neural", sharing=sharing,
                    source_count=0 if sharing == "scratch" else 4,
                    target=target, horizon=1, metric="cc",
                    fit_id=f"{sharing}-{fraction}",
                    value=0.3, valid_dimensions=3, scored_samples=64,
                ))
    source_rows = [
        dict(
            sharing=sharing, source_count=count,
            horizon=horizon, cc=0.3,
        )
        for sharing, count in (("AK", 4), ("AAK", 4))
        for horizon in HORIZONS
    ]
    losses = [
        dict(
            sharing=sharing, source_count=4,
            macro_validation_loss=2.0,
        )
        for sharing in ("AK", "AAK")
    ]
    monkeypatch.setattr(reports, "cost_tables", lambda *args: None)
    monkeypatch.setattr(
        source_validation, "collect_source_validation",
        lambda plan: (source_rows, losses),
    )
    directory = tmp_path / "report"
    directory.mkdir()
    plan = dict(settings=dict(
        target_fractions=[0.05, 0.2, 0.8],
        evaluation_horizons=[1],
        source_training=dict(maximum_updates=10000),
    ))
    reports.write_sharing_payload(
        plan, FILTER, [], rows,
        [dict(row, coordinate="x") for row in rows],
        reports.summarize(rows), directory,
    )
    assert (directory / "tables" / "summary.csv").stat().st_size > 0
    assert (
        directory / "figures" / "target_transfer"
        / "neural" / "h1.png"
    ).stat().st_size > 0
    assert (
        directory / "figures" / "source_validation"
        / "macro_loss.png"
    ).stat().st_size > 0
    assert not list(directory.glob("*.csv"))
    reports.ready_checksums(directory, "synthetic")
    assert reports.valid_live(directory, "synthetic")
