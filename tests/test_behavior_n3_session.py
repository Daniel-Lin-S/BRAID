"""Verify session-private n3 scheduling and saved-map extraction.

Inputs are synthetic recurrent arrays and fixed-stage settings. Outputs are
checks on per-session forecast identity, validation cadence, and log progress.
"""

import json
from pathlib import Path

import numpy as np

from BRAID.multisession import MixedForecaster
from experiments.multisession.behavior_compute_scaling.n3_session import (
    log_progress, publish_status, selected_session_weights,
    session_settings,
)
from experiments.multisession.log_paths import source_session_log_path
from experiments.multisession.behavior_compute_scaling.training import (
    validation_due,
)


def test_private_rnn1_map_matches_full_source_forecast() -> None:
    """An extracted source session reproduces the same saved-map forecast."""
    rng = np.random.default_rng(23)
    full = MixedForecaster([(3, 4)] * 3, 2, "A", 42, decoder_width=0)
    single = MixedForecaster([(3, 4)], 2, "A", 42, decoder_width=0)
    single.restore(selected_session_weights(full.weights(), 2))
    inputs = tuple(
        rng.normal(size=(2, 128, 5)).astype("f") for _ in range(3)
    )
    forward = tuple(
        rng.normal(size=(2, 128, 2, 8)).astype("f")
        for _ in range(3)
    )
    all_predictions, _ = full.forecast(inputs, forward)
    one_prediction, _ = single.forecast(
        (inputs[2],), (forward[2],)
    )
    np.testing.assert_allclose(
        all_predictions[2], one_prediction[0], rtol=1e-5, atol=1e-4
    )


def test_session_validation_cadence_matches_group_protocol() -> None:
    """Splitting fits does not move any fixed-stage validation update."""
    settings = dict(
        update_multipliers={"4": 1, "8": 1, "16": 1, "33": 2},
        checkpoints=[40], total_updates=[50], warmup_updates=10,
        session_batch_sizes={"4": 8, "8": 8, "16": 8, "33": 4},
        fixed_exposure=64000, validation_exposure=800,
    )
    for count in (4, 8, 16, 33):
        source = dict(count=count, spec=dict(source_training=settings))
        local = settings["session_batch_sizes"][str(count)]
        resolved = session_settings(source)
        cap = resolved["fixed_update_cap"]
        assert cap == 64000 // (count * local)
        assert resolved["session_batch_sizes"] == {"1": local}
        for update in range(1, cap + 1):
            assert validation_due(
                update, count * local, 800, cap
            ) == validation_due(
                update, local, resolved["validation_exposure"], cap
            )


def test_live_progress_reads_last_real_update(tmp_path: Path) -> None:
    """Per-session status reflects logged updates and inner checks."""
    log = tmp_path / "worker.log"
    log.write_text(
        "INFO behavior_post_n3 update=25/2000 loss=1.2 lr=0.001\n"
        "INFO behavior_post_n3 update=26 train=1.1 inner=0.9 "
        "best=26 stale=0\n"
    )
    assert log_progress(log) == dict(
        update=26, training_loss=1.1,
        inner_validation_loss=0.9, best_update=26,
        stale_checks=0,
    )


def test_worker_logs_are_nested_and_status_is_lifecycle_only(
    tmp_path: Path,
) -> None:
    """Model progress stays outside the campaign lifecycle file."""
    launch = tmp_path / "n3_sweep_20261028T000000Z_1"
    path = source_session_log_path(
        launch, 33, "indy_20160915_01", "component", 16
    )
    assert path.parent == (
        launch / "jobs" / "source_n3" / "indy_20160915_01"
    )
    members = {
        "33:0:n3_16": dict(
            state="running", update=100, training_loss=0.5,
            component="component",
        )
    }
    publish_status(launch, members, "active")
    status = json.loads((launch / "status.json").read_text())
    assert status["state"] == "active"
    assert status["counts"]["running"] == 1
    assert status["total"] == 1
    assert "members" not in status
    assert "training_loss" not in status
    progress = json.loads(
        (launch / "jobs" / "source_n3" / "progress.json")
        .read_text()
    )
    assert progress["members"]["33:0:n3_16"]["training_loss"] == 0.5
