"""Regression tests for feedthrough-free forecast decoders.

The tests build RNN forecast models with a feedthrough input and verify
that Cfw is created only for horizons that require an input-free decoder.
"""

import unittest

import numpy as np
import tensorflow as tf

from BRAID.RNNModel import RNNModel

BATCH_SIZE = 1
BLOCK_SAMPLES = 3
DIMENSION = 1
INPUT_VALUE = 1.0
SINGLE_STEP_HORIZONS = (1,)
MULTI_STEP_HORIZONS = (1, 2)
STATE_TRANSITION_ARCHITECTURES = (
    "multilayer_perceptron",
    "lstm",
)


def _build_model(
    steps_ahead: tuple[int, ...],
    state_transition_architecture: str,
) -> RNNModel:
    """Build a one-dimensional forecast model with feedthrough."""
    return RNNModel(
        nx=DIMENSION,
        ny=DIMENSION,
        block_samples=BLOCK_SAMPLES,
        batch_size=BATCH_SIZE,
        ny_out=DIMENSION,
        nft=DIMENSION,
        steps_ahead=steps_ahead,
        enable_forward_pred=True,
        state_transition_architecture=state_transition_architecture,
    )


def _model_inputs() -> tuple[tf.Tensor, tf.Tensor]:
    """Return model inputs shaped (batch, time, dimension)."""
    shape = (BATCH_SIZE, BLOCK_SAMPLES, DIMENSION)
    return (
        tf.ones(shape, dtype=tf.float32) * INPUT_VALUE,
        tf.ones(shape, dtype=tf.float32) * INPUT_VALUE,
    )


class ForecastDecoderTest(unittest.TestCase):
    """Validate Cfw construction and loss connectivity."""

    def test_single_step_forecast_does_not_create_cfw(self) -> None:
        """Avoid an unconnected Cfw variable for one-step training."""
        for architecture in STATE_TRANSITION_ARCHITECTURES:
            with self.subTest(architecture=architecture):
                model = _build_model(SINGLE_STEP_HORIZONS, architecture)
                self.assertFalse(hasattr(model.rnn.cell, "Cfw"))

    def test_lstm_prediction_uses_output_state_dimensions(self) -> None:
        """Initial-state forecasting passes LSTM output to its decoder."""
        model = RNNModel(
            nx=2, ny=3, block_samples=8, batch_size=1, ny_out=1,
            has_prior_pred=True, steps_ahead=(1, 2),
            enable_forward_pred=True,
            state_transition_architecture="lstm",
            cell_args=dict(
                ASettings={}, KSettings={}, CSettings={},
            ),
        )
        observations = np.ones((3, 8), dtype=np.float32)
        prior = [
            np.zeros((1, 8), dtype=np.float32) for _ in range(2)
        ]
        forecasts = model.predict(
            observations, prior_pred=prior,
            prior_pred_shift_by_one=True,
        )
        self.assertTrue(model.LSTM_cell)
        self.assertEqual(forecasts[0].shape, (2, 8))
        self.assertEqual(forecasts[4].shape, (2,))

    def test_multi_step_forecast_cfw_has_gradients(self) -> None:
        """Ensure the multi-step loss is connected to Cfw variables."""
        for architecture in STATE_TRANSITION_ARCHITECTURES:
            with self.subTest(architecture=architecture):
                model = _build_model(MULTI_STEP_HORIZONS, architecture)
                with tf.GradientTape() as tape:
                    predictions = model.model(_model_inputs(), training=True)
                    loss = tf.reduce_sum(predictions[1])
                gradients = tape.gradient(
                    loss,
                    model.rnn.cell.Cfw.trainable_variables,
                )

                self.assertTrue(gradients)
                self.assertTrue(
                    all(gradient is not None for gradient in gradients)
                )


if __name__ == "__main__":
    unittest.main()
