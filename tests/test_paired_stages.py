"""Check paired neural topology, stopped checkpoints, and score aggregation.

Inputs are synthetic native windows and fixed session/seed score rows.
Outputs are finite forecasts, checksum-valid selected checkpoints, and
equal-session/equal-seed summary assertions under temporary storage.
"""

import json

import numpy as np

from BRAID.multisession import MixedForecaster
from experiments.multisession.behavior_compute_scaling import paired_stages
from experiments.multisession.behavior_supervision.models import (
    rnn2_direct_model, rnn2_model,
)
from experiments.multisession.compute_scaling import training
from experiments.multisession.store import completed
from experiments.storage.cache import fingerprint


def test_direct_rnn2_has_no_rnn1_state_input() -> None:
    """Direct neural inputs omit every latent-state coordinate."""
    spec = dict(latent=16, seed=42)
    dimensions = [(5, 4)]
    direct = rnn2_direct_model(dimensions, spec)
    residual = rnn2_model(dimensions, spec)
    assert direct.sessions[0]["K"].layers[0].kernel.shape[0] == 7
    assert direct.sessions[0]["Kfw"].layers[0].kernel.shape[0] == 2
    assert residual.sessions[0]["K"].layers[0].kernel.shape[0] == 23
    assert residual.sessions[0]["Kfw"].layers[0].kernel.shape[0] == 18
    assert paired_stages.DEPENDENCIES["neural_direct"] == ()
    assert (
        paired_stages.ARCHITECTURES["two_stage"][0]
        == paired_stages.ARCHITECTURES["three_stage"][0]
    )


def test_stopped_recurrent_rate_uses_native_warmup_and_decay() -> None:
    """The 2,500-update cap preserves the selected 80/20 rate phases."""
    settings = dict(
        learning_rate=1e-3, warmup_updates=10,
        warmup_start_learning_rate=1e-7,
    )
    rate = training.recurrent_stopping_rate
    assert rate(1, settings, 2000, 2500) == 1e-7
    assert rate(10, settings, 2000, 2500) == 1e-3
    assert rate(2000, settings, 2000, 2500) == 1e-3
    assert rate(2001, settings, 2000, 2500) == 1e-3
    assert rate(2500, settings, 2000, 2500) == 0.0


def test_stopped_fit_restores_best_and_reuses_completion(
    tmp_path, monkeypatch,
) -> None:
    """Patience restores the best inner checkpoint without rerunning it."""
    model = MixedForecaster(
        [(3, 3)], 2, "A", 42, decoder_width=0
    )
    row = dict(
        I=np.ones((2, 128, 5), dtype="f"),
        F=np.ones((2, 128, 2, 8), dtype="f"),
        T=np.ones((2, 128, 3), dtype="f"),
        P=np.zeros((4, 2, 128, 3), dtype="f"),
    )
    settings = dict(
        learning_rate=1e-3,
        warmup_updates=10,
        warmup_start_learning_rate=1e-7,
        session_batch_sizes={"1": 8},
        validation_exposure=8,
        fixed_patience=2,
    )
    updates = []
    validation = iter((2.0, 1.0, 3.0, 4.0))

    def fake_update(*args, **kwargs):
        updates.append(len(updates) + 1)
        return 0.5, [[0] * 8]

    def fake_validation(*args, **kwargs):
        return [next(validation)], 0

    monkeypatch.setattr(training, "balanced_update", fake_update)
    monkeypatch.setattr(
        training, "evaluate_sessions", fake_validation
    )
    arguments = (
        model, [row], tmp_path,
        dict(stage="paired_test", seed=42),
        settings, 42, 0, 2500, None,
        ("/CPU:0", "/CPU:0"), 4,
    )
    scientific = training.segment_identity(
        arguments[3], 0, 2500, False, None, True,
        (2000, 2500),
    )
    target = tmp_path / f"paired_test_{fingerprint(scientific)}"
    target.mkdir()
    (target / "partial.txt").write_text("interrupted")
    directory, payload = training.fit_segment(
        *arguments, validation=[row],
        recurrent_schedule=(2000, 2500),
        directory_prefix="paired_test_",
    )
    record = json.loads(
        (directory / "training.json").read_text()
    )
    assert completed(directory) is not None
    assert directory == target
    quarantine = list((tmp_path / "quarantine").iterdir())
    assert len(quarantine) == 1
    assert (quarantine[0] / "partial.txt").read_text() == "interrupted"
    assert directory.name.startswith("paired_test_")
    assert record["stop"] == 4
    assert record["selected_update"] == 2
    assert payload["update"] == 2
    assert len(updates) == 4
    again, restored = training.fit_segment(
        *arguments, validation=[row],
        recurrent_schedule=(2000, 2500),
        directory_prefix="paired_test_",
    )
    assert again == directory
    assert restored["update"] == 2
    assert len(updates) == 4


def test_summary_weights_sessions_then_seeds_equally() -> None:
    """A session with many samples still receives one of four weights."""
    rows = []
    horizons = (50, 100, 200, 400)
    for architecture in paired_stages.ARCHITECTURES:
        for seed in paired_stages.SEEDS:
            for session in range(4):
                for horizon in horizons:
                    value = seed - 42 + session
                    rows.append(dict(
                        architecture=architecture, seed=seed,
                        session=str(session), horizon_ms=horizon,
                        cc=value, r2=value / 2,
                        mse_ratio=4 - value,
                    ))
    by_seed, summary = paired_stages.summarize_scores(rows)
    assert len(by_seed) == 24
    assert len(summary) == 8
    selected = next(
        row for row in summary
        if row["architecture"] == "two_stage"
        and row["horizon_ms"] == 50
    )
    assert selected["cc"] == 2.5
    assert selected["r2"] == 1.25
    assert selected["mse_ratio"] == 1.5


def test_stage_ablation_has_dedicated_artifact_root(tmp_path) -> None:
    """New components remain outside the multi-session fit store."""
    plan = dict(settings=dict(
        paths=dict(artifact_root=str(tmp_path))
    ))
    assert paired_stages.component_root(plan) == (
        tmp_path / "stage_ablation" / "source_components"
    )
