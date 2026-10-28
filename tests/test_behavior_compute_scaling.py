"""Regression checks for behavioral source scaling arithmetic and ancestry.

Inputs are small synthetic session windows shaped like native I/F/T/P arrays.
Outputs are assertions on native stage loss, balanced Adam updates, restored
checkpoint state, immutable reuse, and microbatch-aware FLOP accounting.
Temporary checkpoint files are removed by pytest after each test.
"""

import json

import numpy as np
import pytest

from BRAID.multisession import MixedForecaster
from experiments.multisession.behavior_compute_scaling.flops import (
    update_flops,
)
from experiments.multisession.behavior_compute_scaling.training import (
    balanced_update, fit_segment, gradient_function,
    optimizer_groups, session_loss,
)
from experiments.multisession.behavior_compute_scaling.queue import (
    hold_source_slot, runnable,
)
from experiments.multisession.behavior_compute_scaling import (
    report as scaling_report,
)
from experiments.multisession.store import completed
from experiments.multisession.training import select_batch


def synthetic_data() -> list[dict]:
    """Create two heterogeneous-equivalent (4, 128, D) source arrays."""
    rng = np.random.default_rng(13)
    return [
        dict(
            I=rng.normal(size=(4, 128, 5)).astype("f"),
            F=rng.normal(size=(4, 128, 2, 8)).astype("f"),
            T=rng.normal(size=(4, 128, 3)).astype("f"),
            P=np.zeros((4, 4, 128, 3), dtype="f"),
        )
        for _ in range(2)
    ]


def source_settings() -> dict:
    """Return a two-update schedule with one-window source microbatches."""
    return dict(
        learning_rate=1e-3,
        warmup_updates=1,
        warmup_start_learning_rate=1e-7,
        branch_updates=1,
        session_batch_sizes={"2": 2},
    )


def source_model() -> MixedForecaster:
    """Construct the same two-session sharing model for paired checks."""
    return MixedForecaster(
        [(3, 3)] * 2, 2, "A", 42, decoder_width=0
    )


def test_session_loss_and_balanced_accumulation() -> None:
    """Each single-session loss and accumulated update match native loss."""
    data = synthetic_data()
    model = source_model()
    batch = select_batch(data, [np.arange(4)] * 2)
    native = model.loss(*batch)[0].numpy()
    local = [
        session_loss(
            model, index, tuple(group[index] for group in batch)
        ).numpy()
        for index in range(2)
    ]
    np.testing.assert_allclose(native, local, rtol=1e-6)
    initial = model.weights()
    updates = []
    for microbatch in (1, 2):
        model.restore(initial)
        optimizers = optimizer_groups(model, source_settings())
        functions = [
            gradient_function(model, index) for index in range(2)
        ]
        loss, selected = balanced_update(
            model, data, optimizers, np.random.default_rng(42),
            2, microbatch, ("/CPU:0", "/CPU:0"), functions,
        )
        updates.append((loss, selected, model.weights()))
    assert updates[0][1] == updates[1][1]
    np.testing.assert_allclose(
        updates[0][0], updates[1][0], rtol=1e-6
    )
    for key in initial:
        for left, right in zip(
            updates[0][2][key], updates[1][2][key]
        ):
            np.testing.assert_allclose(
                left, right, rtol=1e-4, atol=1e-6
            )


def test_checkpoint_restore_and_immutable_reuse(tmp_path) -> None:
    """Split segments reproduce one continuous fit bit for bit."""
    data = synthetic_data()
    direct = source_model()
    staged = source_model()
    settings = source_settings()
    devices = ("/CPU:0", "/CPU:0")
    _, first = fit_segment(
        direct, data, tmp_path, {"path": "continuous"},
        settings, 42, 0, 2, None, devices, 2,
    )
    parent, _ = fit_segment(
        staged, data, tmp_path, {"path": "staged"},
        settings, 42, 0, 1, None, devices, 2,
    )
    child, second = fit_segment(
        staged, data, tmp_path, {"path": "staged"},
        settings, 42, 1, 2, parent, devices, 2,
    )
    for key in first["weights"]:
        for left, right in zip(
            first["weights"][key], second["weights"][key]
        ):
            np.testing.assert_array_equal(left, right)
    assert first["sampling_state"] == second["sampling_state"]
    marker = completed(child)
    stamp = (child / "complete.json").stat().st_mtime_ns
    reused, _ = fit_segment(
        staged, data, tmp_path, {"path": "staged"},
        settings, 42, 1, 2, parent, devices, 1,
    )
    assert reused == child
    assert completed(child) == marker
    assert (child / "complete.json").stat().st_mtime_ns == stamp
    record = json.loads((child / "training.json").read_text())
    assert record["effective_batch"] == 4
    assert record["total_flops"] == (
        record["flops_per_update"]["total"]
    )


def test_flops_include_accumulated_microbatches() -> None:
    """Smaller microbatches add reductions without changing core work."""
    model = source_model()
    data = synthetic_data()
    whole = update_flops(model, data, 2, 2)
    split = update_flops(model, data, 2, 1)
    assert whole["forward"] == split["forward"]
    assert whole["backward"] == split["backward"]
    assert whole["optimizer"] == split["optimizer"]
    assert split["accumulation"] > whole["accumulation"]
    assert split["total"] > whole["total"]


def test_ready_adaptation_precedes_scratch() -> None:
    """A completed source count releases its targets independently."""
    plan = {"jobs": {
        "source4": dict(kind="source", source_count=4,
                        dependencies=[]),
        "source33": dict(kind="source", source_count=33,
                         dependencies=[]),
        "preprocessing": dict(kind="preprocessing", dependencies=[]),
        "adaptation4": dict(kind="adaptation", source_count=4,
                            dependencies=["source4", "preprocessing"]),
        "scratch": dict(kind="scratch", dependencies=["preprocessing"]),
    }}
    states = {
        "source4": "complete",
        "source33": "running",
        "preprocessing": "complete",
        "adaptation4": "pending",
        "scratch": "pending",
    }
    assert runnable(plan, states, {"adaptation4", "scratch"}) == [
        "adaptation4", "scratch",
    ]


def test_waiting_source_reserves_one_gpu() -> None:
    """Scratch cannot repeatedly consume a waiting source's second GPU."""
    scratch = {"kind": "scratch"}
    adaptation = {"kind": "adaptation", "source_count": 4}
    assert hold_source_slot(scratch, 1, True, set())
    assert not hold_source_slot(adaptation, 1, True, set())
    assert hold_source_slot(adaptation, 1, True, {4})
    assert not hold_source_slot(scratch, 2, True, set())
    assert not hold_source_slot(scratch, 1, False, set())


def test_campaign_report_waits_for_every_fit(
    tmp_path, monkeypatch,
) -> None:
    """Incomplete membership cannot publish a partial report snapshot."""
    plan = {
        "analyses": {
            "multi_session_behavior_compute_scaling": ["first", "second"]
        },
        "jobs": {
            "first": {"kind": "source", "source_count": 4},
            "second": {"kind": "source", "source_count": 33},
        },
    }
    monkeypatch.setattr(
        scaling_report, "fit_directory",
        lambda unused_plan, key: tmp_path / key,
    )
    monkeypatch.setattr(
        scaling_report, "source_tables",
        lambda unused_plan: pytest.fail(
            "Incomplete report unexpectedly collected source rows."
        ),
    )
    (tmp_path / "first").mkdir()
    (tmp_path / "first" / "complete.json").write_text("{}")
    assert not scaling_report.campaign_complete(plan, False)
    scaling_report.report(plan, tmp_path / "plan.json")
    assert not (tmp_path / "reports").exists()
    with pytest.raises(ValueError, match="1 of 2 fits are incomplete"):
        scaling_report.campaign_complete(plan, True)
    monkeypatch.setattr(
        scaling_report, "completed",
        lambda unused_path: {"checksums": {}},
    )
    assert scaling_report.campaign_complete(
        plan, True, scaling_report.INTERIM_COUNTS
    )
    (tmp_path / "second").mkdir()
    (tmp_path / "second" / "complete.json").write_text("{}")
    monkeypatch.setattr(
        scaling_report, "completed",
        lambda unused_path: {"checksums": {}},
    )
    assert scaling_report.campaign_complete(plan, True)


def test_report_requires_every_source_count_curve() -> None:
    """A completed report cannot draw a partial transfer comparison."""
    settings = {
        "target_fractions": [0.05],
        "evaluation_horizons": [1],
    }
    source = [
        dict(
            target=target, metric=metric, fraction=0.05,
            horizon=1, source_count=count, sharing="A1_fw_RNN2",
        )
        for target in ("neural", "behavior")
        for metric in ("cc", "r2", "mse")
        for count in (4, 8, 16, 33)
    ]
    scratch = [
        dict(
            target=target, metric=metric, fraction=0.05,
            horizon=1, source_count=0, sharing="scratch",
        )
        for target in ("neural", "behavior")
        for metric in ("cc", "r2", "mse")
    ]
    scaling_report.validate_target_coverage(
        source + scratch, source, settings
    )
    with pytest.raises(ValueError, match="requested source-count"):
        scaling_report.validate_target_coverage(
            source[:-1] + scratch, source[:-1], settings
        )
    interim = [row for row in source if row["source_count"] != 33]
    scaling_report.validate_target_coverage(
        interim + scratch, interim, settings,
        scaling_report.INTERIM_COUNTS,
    )


def test_nonuniform_prefixes_and_branch_ancestry(tmp_path) -> None:
    """A later endpoint extends the main prefix, never its decay branch."""
    from experiments.multisession.behavior_compute_scaling.pipeline import (
        fit_component_budget,
    )
    from experiments.multisession.store import load_pickle

    model = source_model()
    initial = model.weights()
    settings = dict(source_settings(), checkpoints=[1, 3], total_updates=[2, 5])
    identity = {"stage": "behavior_relevant"}
    devices = ("/CPU:0", "/CPU:0")
    first, _ = fit_component_budget(
        model, synthetic_data(), tmp_path, identity, settings, 42, 1,
        devices, 2,
    )
    model.restore(initial)
    last, flops = fit_component_budget(
        model, synthetic_data(), tmp_path, identity, settings, 42, 3,
        devices, 2,
    )
    meta = json.loads((last / "identity.json").read_text())
    parent = tmp_path / meta["parent"]
    assert parent != first
    parent_meta = json.loads((parent / "identity.json").read_text())
    assert (parent_meta["start"], parent_meta["stop"]) == (1, 3)
    assert not parent_meta["branch"]
    assert load_pickle(last / "checkpoint.pkl")["update"] == 5
    total = 0
    current = last
    while current is not None:
        total += json.loads((current / "training.json").read_text())[
            "total_flops"
        ]
        parent_id = json.loads(
            (current / "identity.json").read_text()
        )["parent"]
        current = tmp_path / parent_id if parent_id else None
    assert flops == total
    with pytest.raises(ValueError, match="Expected parent at update"):
        fit_segment(model, synthetic_data(), tmp_path, identity, settings,
                    42, 2, 3, first.parent / meta["parent"], devices, 2)


def test_fixed_stage_stopping_restores_best_and_counts_all_work(
    tmp_path, monkeypatch,
) -> None:
    """A known loss sequence selects the best state but charges later work."""
    from experiments.multisession.compute_scaling import training

    losses = iter([3.0, 2.0, 4.0, 5.0])
    monkeypatch.setattr(training, "evaluate_sessions", lambda model, data: (
        [next(losses)] * 2, 123,
    ))
    settings = dict(source_settings(), validation_exposure=4, fixed_patience=2)
    model = source_model()
    directory, checkpoint = fit_segment(
        model, synthetic_data(), tmp_path, {"stage": "neural_readout"},
        settings, 42, 0, 8, None, ("/CPU:0", "/CPU:0"), 2,
        validation=synthetic_data(),
    )
    record = json.loads((directory / "training.json").read_text())
    assert record["stop"] == 4
    assert checkpoint["update"] == record["selected_update"] == 2
    assert record["best_validation_loss"] == 2.0
    assert record["evaluation_flops"] == 4 * 123
    assert record["total_flops"] == 4 * record["flops_per_update"]["total"]
    for name, values in model.weights().items():
        for actual, expected in zip(values, checkpoint["weights"][name]):
            np.testing.assert_array_equal(actual, expected)
    marker = completed(directory)
    reused, _ = fit_segment(
        model, synthetic_data(), tmp_path, {"stage": "neural_readout"},
        settings, 42, 0, 8, None, ("/CPU:0", "/CPU:0"), 1,
        validation=synthetic_data(),
    )
    assert reused == directory and completed(directory) == marker


def test_exposure_cadence_and_resolved_schedule() -> None:
    """Caps round down; checks occur only on crossed exposure thresholds."""
    from experiments.multisession.behavior_compute_scaling.pipeline import (
        resolved_schedule,
    )
    from experiments.multisession.behavior_compute_scaling.training import (
        learning_rate, validation_due,
    )

    settings = dict(
        checkpoints=[40, 80, 160, 320, 640],
        total_updates=[50, 100, 200, 400, 800], warmup_updates=10,
        update_multipliers={"4": 1, "8": 1, "16": 1, "33": 2},
        session_batch_sizes={"4": 8, "8": 8, "16": 8, "33": 4},
        fixed_exposure=64000, learning_rate=1e-3,
        warmup_start_learning_rate=1e-7,
    )
    for count, cap in ((4, 2000), (8, 1000), (16, 500), (33, 484)):
        resolved = resolved_schedule(settings, count)
        assert resolved["fixed_update_cap"] == cap
        assert learning_rate(1, resolved, False) == 1e-7
        assert learning_rate(
            resolved["warmup_updates"], resolved, False
        ) == 1e-3
        for main, total in zip(
            resolved["checkpoints"], resolved["total_updates"]
        ):
            branch = dict(resolved, branch_updates=total - main)
            assert learning_rate(1, branch, True) == 1e-3
            assert learning_rate(total - main, branch, True) == 0.0
        batch = count * settings["session_batch_sizes"][str(count)]
        checks = [u for u in range(1, cap + 1)
                  if validation_due(u, batch, 800, cap)]
        assert checks[0] == int(np.ceil(800 / batch))
        assert checks[-1] == cap
        assert len(checks) == 80
    assert resolved_schedule(settings, 33)["total_updates"][-1] == 1600


def test_inner_split_preserves_outer_and_training_only_scales() -> None:
    """Guarded preprocessing cannot read inner or outer held-out samples."""
    from experiments.contracts import FeatureSet
    from experiments.multisession.data import source_features, session_windows

    rng = np.random.default_rng(7)
    n = 12000
    arrays = dict(
        Y=rng.poisson(2, size=(n, 3)).astype(float),
        Z=rng.normal(size=(n, 2)), U=rng.normal(size=(n, 2)),
        t=np.arange(n) / 20, ids=np.arange(3), units=np.arange(3),
    )
    settings = dict(
        cv={"guard_seconds": 1.8},
        spike_smoothing=dict(sigma_ms=50, boundary="reflect", truncate=4),
    )
    raw = FeatureSet(arrays, {})
    _, _, old = session_windows(source_features(raw, settings), True)
    train, inner, outer, record = session_windows(
        source_features(raw, settings, 0.1), True, include_test=True,
    )
    assert old["validation"] == record["validation"]
    roles = [record[k] for k in ("training", "inner_validation", "validation")]
    for i, left in enumerate(roles):
        for right in roles[i + 1:]:
            assert not np.intersect1d(left, right).size
    changed = {k: value.copy() for k, value in arrays.items()}
    cut = int(np.floor((int(n * 0.8) - 36) * 0.9))
    changed["Y"][cut:] += 1000
    changed["Z"][cut:] *= 100
    other_train, _, _, other_record = session_windows(
        source_features(FeatureSet(changed, {}), settings, 0.1), True,
        include_test=True,
    )
    assert record["scales"] == other_record["scales"]
    for name in train:
        np.testing.assert_array_equal(train[name], other_train[name])
    assert all(len(part["Y"]) > 0 for part in (train, inner, outer))


def test_new_protocol_rejects_partial_reports() -> None:
    """The previous interim exception never applies to revised fits."""
    plan = {"settings": {"source_training": {"protocol": "revision"}}}
    with pytest.raises(ValueError, match="not authorized"):
        scaling_report.report_members(plan, scaling_report.INTERIM_COUNTS)


def test_source_figure_axes_and_coverage(tmp_path, monkeypatch) -> None:
    """Only two recurrent panels have four complete five-point log curves."""
    captured = []
    monkeypatch.setattr(
        scaling_report.plt, "close", lambda fig: captured.append(fig)
    )
    rows = [dict(
        stage=stage, source_count=count, full_path_flops=(i + 1) * count * 100,
        chronological_source_test_loss=1.0 / (i + 1), transfer_selected=i == 0,
    ) for stage in scaling_report.RECURRENT_STAGES
      for count in scaling_report.COUNTS for i in range(5)]
    scaling_report.source_figures(
        rows, tmp_path, stages=scaling_report.RECURRENT_STAGES,
    )
    assert len(list(tmp_path.glob("*.png"))) == 2
    for fig in captured:
        axis = fig.axes[0]
        assert axis.get_xscale() == axis.get_yscale() == "log"
        assert len(axis.lines) == 4
        assert all(len(line.get_xdata()) == 5 for line in axis.lines)
    rows[0]["chronological_source_test_loss"] = float("nan")
    with pytest.raises(ValueError, match="finite positive"):
        scaling_report.source_figures(
            rows, tmp_path, stages=scaling_report.RECURRENT_STAGES,
        )


def test_full_pipeline_reuses_preprocessing_across_budgets(
    tmp_path, monkeypatch,
) -> None:
    """Five paths share fixed targets but retain matched readout ancestry."""
    from experiments.multisession.behavior_compute_scaling import pipeline
    from experiments.multisession.behavior_supervision import models

    monkeypatch.setattr(models, "PREPROCESSING_SIZE", 2)
    monkeypatch.setattr(models, "BEHAVIOR_WIDTH", 2)
    monkeypatch.setattr(models, "NEURAL_WIDTH", 2)
    rng = np.random.default_rng(45)

    def rows(sizes):
        return [dict(
            Y=rng.normal(size=(size, 128, 3)).astype("f"),
            U=rng.normal(size=(size, 128, 2)).astype("f"),
            Z=rng.normal(size=(size, 128, 4)).astype("f"),
        ) for size in sizes]

    names = [f"session_{i}" for i in range(4)]
    test_sizes = [11, 15, 48, 13]
    records = {name: {"validation": list(range(size))}
               for name, size in zip(names, test_sizes)}
    settings = dict(
        source_settings(), checkpoints=[1, 2, 3, 4, 5],
        total_updates=[2, 3, 4, 5, 6], update_multipliers={"4": 1},
        session_batch_sizes={"4": 1}, fixed_exposure=8,
        validation_exposure=4, fixed_patience=10,
    )
    spec = dict(
        sessions=names, source_count=4, sources={}, data={}, model={},
        latent=2, seed=42, source_training=settings, implementation="test",
    )
    fit = tmp_path / "fit"
    fit.mkdir()
    component_root = tmp_path / "components"
    pipeline.fit_source_campaign(
        rows([3] * 4), rows(test_sizes), records, fit, component_root,
        spec, ("/CPU:0", "/CPU:0"), 1, rows([2] * 4),
    )
    budgets = json.loads((fit / "budgets.json").read_text())
    assert len(budgets) == 5
    for stage in ("preprocessing_neural", "preprocessing_behavior"):
        assert len({
            row["stages"][stage]["component_id"] for row in budgets
        }) == 1
    assert len({row["stages"]["neural_readout"]["component_id"]
                for row in budgets}) == 5
    for budget in budgets:
        stages = budget["stages"]
        assert budget["full_path_flops"] == sum(
            row["component_flops"] for row in stages.values()
        )
        for downstream, upstream in (
            ("neural_readout", "behavior_relevant"),
            ("residual", "neural_readout"),
        ):
            path = component_root / stages[downstream]["component_id"]
            identity = json.loads((path / "identity.json").read_text())
            assert identity["upstream_checkpoint"] == (
                stages[upstream]["component_id"]
            )
        for row in stages.values():
            assert np.isfinite(row["score"]["mean_loss"])
            assert completed(component_root / row["component_id"])
    assert (fit / "shared.pkl").is_file()


def test_recorded_behavior_target_keeps_training_normalization() -> None:
    """Direct supervision uses observed Z without a generated scale."""
    from experiments.multisession.behavior_supervision.targets import (
        RECORDED, prepare_targets, target_mode,
    )

    partitions = tuple(
        [dict(Z=np.full((2, 128, 4), role, dtype="f"))]
        for role in (1.0, 2.0, 3.0)
    )
    assert target_mode({"behavior_supervision": {"target": RECORDED}}) == (
        RECORDED
    )
    prepared, scales = prepare_targets(partitions, RECORDED)
    assert scales is None
    for original, selected in zip(partitions, prepared):
        np.testing.assert_array_equal(
            original[0]["Z"], selected[0]["Z"]
        )
    with pytest.raises(ValueError, match="cannot use preprocessing"):
        prepare_targets(partitions, RECORDED, states=[])


def test_source_target_mode_and_stage_membership_are_separate() -> None:
    """The direct target changes components without changing the ladder."""
    from experiments.multisession.behavior_compute_scaling.plan import (
        DIRECT_CAMPAIGN, DIRECT_COMPONENTS, source_implementation,
    )
    from experiments.multisession.behavior_compute_scaling.report import (
        source_components,
    )

    assert source_components({"name": DIRECT_CAMPAIGN}) == (
        DIRECT_COMPONENTS
    )
    assert source_components(
        {"name": "multi_session_behavior_compute_scaling"}
    ) == (
        "preprocessing_neural", "preprocessing_behavior",
        "behavior_relevant", "neural_readout", "residual",
    )
    assert source_implementation() != source_implementation(direct=True)
