"""Regression checks for behavioral source scaling arithmetic and ancestry.

Inputs are small synthetic session windows shaped like native I/F/T/P arrays.
Outputs are assertions on native stage loss, balanced Adam updates, restored
checkpoint state, immutable reuse, and microbatch-aware FLOP accounting.
Temporary checkpoint files are removed by pytest after each test.
"""

import json

import numpy as np

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
