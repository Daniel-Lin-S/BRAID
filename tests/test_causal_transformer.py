"""Check BRAID Transformer token causality and native reconstruction.

Inputs are finite synthetic neural, target, and stage-one arrays. Outputs are
assertions on forecast access, state shape, and batch-size reconstruction.
"""

import numpy as np

from BRAID.RNNModel import RNNModel
from BRAID.config import resolve_braid_fit_arguments


WIDTH = 32
MARKER = -1e20


def _model(
    stage_one_dims: int = 0, block_samples: int = 3,
    horizons: tuple[int, ...] = (1, 2),
) -> RNNModel:
    """Build a sequence Transformer with native BRAID outputs."""
    neural_dims = 4
    target_dims = 2
    return RNNModel(
        nx=WIDTH,
        ny=stage_one_dims + neural_dims + target_dims,
        block_samples=block_samples,
        batch_size=1,
        ny_out=2,
        nft=0,
        n1_in=stage_one_dims + target_dims,
        state_transition_architecture="causal_transformer",
        neural_dims=neural_dims,
        stage_one_dims=stage_one_dims,
        steps_ahead=list(horizons),
        enable_forward_pred=True,
        stateful=False,
        missing_marker=MARKER,
        has_prior_pred=bool(stage_one_dims),
        cell_args={"KSettings": {}, "CSettings": {}},
    )


def test_causal_forecast_uses_future_target_without_future_neural():
    """Horizon two can see target at t+1 but cannot see neural at t+1."""
    model = _model()
    rng = np.random.default_rng(23)
    observed = rng.normal(size=(1, 3, 6)).astype("float32")
    future = rng.normal(size=(1, 3, 2, 2)).astype("float32")
    baseline = model.model((observed, future), training=False)[1]
    later_neural = observed.copy()
    later_neural[0, 1:, :4] += 100.0
    unchanged = model.model((later_neural, future), training=False)[1]
    np.testing.assert_allclose(
        baseline.numpy()[:, 0], unchanged.numpy()[:, 0],
        rtol=0, atol=0,
    )
    later_target = future.copy()
    later_target[0, 0, :, 0] += 10.0
    changed = model.model((observed, later_target), training=False)[1]
    assert not np.allclose(
        baseline.numpy()[:, 0], changed.numpy()[:, 0]
    )


def test_all_direct_horizons_exclude_later_neural_tokens():
    """Future neural bins cannot affect an earlier forecast query."""
    horizons = (1, 2, 4, 8, 16, 32)
    model = _model(block_samples=32, horizons=horizons)
    rng = np.random.default_rng(91)
    observed = rng.normal(size=(1, 32, 6)).astype("float32")
    future = rng.normal(size=(1, 32, 2, 32)).astype("float32")
    baseline = model.model((observed, future), training=False)
    changed = observed.copy()
    changed[:, 6:, :4] += 50.0
    later = model.model((changed, future), training=False)

    for index in range(len(horizons)):
        np.testing.assert_allclose(
            baseline[index].numpy()[:, :6],
            later[index].numpy()[:, :6],
            rtol=0, atol=0,
        )


def test_stage_two_and_batch_reconstruction_are_finite():
    """Stage-two tokens and restored weights preserve native outputs."""
    model = _model(stage_one_dims=4)
    rng = np.random.default_rng(4)
    observed = rng.normal(size=(1, 3, 10)).astype("float32")
    future = rng.normal(size=(1, 3, 6, 2)).astype("float32")
    prior = rng.normal(size=(1, 3, 2, 2)).astype("float32")
    expected = model.model(
        (observed, future, prior), training=False
    )
    assert all(np.isfinite(value.numpy()).all() for value in expected)
    model.set_batch_size(2)
    doubled = tuple(np.repeat(value, 2, axis=0) for value in (
        observed, future, prior
    ))
    actual = model.model(doubled, training=False)
    for left, right in zip(expected, actual):
        np.testing.assert_allclose(
            left.numpy()[0], right.numpy()[0], rtol=1e-2, atol=1e-2
        )


def test_transformer_config_adds_time_embedding_option():
    """Both stages can independently select the Transformer."""
    from pathlib import Path
    import yaml

    path = Path(__file__).parents[1] / (
        "assets/config/nhp/models/BRAID.yaml"
    )
    configuration = yaml.safe_load(path.read_text())
    configuration["model"]["stage_1"]["state_transition"] = {
        "architecture": "causal_transformer"
    }
    arguments = resolve_braid_fit_arguments(configuration)["args_base"]
    assert arguments["stage_1_state_transition_architecture"] == (
        "causal_transformer"
    )
    assert arguments["stage_2_state_transition_architecture"] == (
        "multilayer_perceptron"
    )
    assert arguments["stage_1_transformer_arguments"] == {
        "include_stage_one_state": True
    }


def test_streaming_context_resets_and_uses_x0():
    """One-bin predictions retain context and explicit seeds restart it."""
    model = _model()
    observed = np.ones(6, dtype="float32")
    future = np.ones((2, 2), dtype="float32")
    model.reset_stream(np.zeros(WIDTH, dtype="float32"))
    first = model.stream_step(observed, future=future)[2]
    second = model.stream_step(observed, future=future)[2]
    assert not np.allclose(first, second)
    model.reset_stream(np.zeros(WIDTH, dtype="float32"))
    repeated = model.stream_step(observed, future=future)[2]
    np.testing.assert_allclose(first, repeated, rtol=0, atol=0)
    model.reset_stream(np.ones(WIDTH, dtype="float32"))
    changed = model.stream_step(observed, future=future)[2]
    assert not np.allclose(first, changed)


def test_forecast_horizons_can_extend_after_fitting():
    """Evaluation horizons reuse the fitted stack without changing h1."""
    model = _model()
    rng = np.random.default_rng(11)
    observed = rng.normal(size=(1, 3, 6)).astype("float32")
    future = rng.normal(size=(1, 3, 2, 2)).astype("float32")
    before = model.model((observed, future), training=False)[0]
    model.set_steps_ahead([1, 2, 4])
    extended = rng.normal(size=(1, 3, 2, 4)).astype("float32")
    extended[..., :2] = future
    outputs = model.model((observed, extended), training=False)
    assert len(outputs) == 9
    assert all(np.isfinite(value.numpy()).all() for value in outputs)
    np.testing.assert_allclose(
        before.numpy(), outputs[0].numpy(), rtol=0, atol=0
    )


def test_full_prediction_seeds_each_independent_window():
    """Full prediction handles thirty independent initial states."""
    model = _model()
    window_count = 30
    sample_count = window_count * model.block_samples
    rng = np.random.default_rng(19)
    observed = rng.normal(size=(model.ny, sample_count)).astype(
        "float32"
    )
    future = rng.normal(size=(model.n1_in, sample_count)).astype(
        "float32"
    )

    outputs = model.predict(observed, n1_in=[future, future])

    assert outputs[0].shape == (WIDTH, sample_count)
    assert all(np.isfinite(value).all() for value in outputs)
    initial_horizon_two = outputs[1][:, ::model.block_samples]
    assert np.max(np.ptp(initial_horizon_two, axis=1)) > 1e-5
