"""Regression tests for multi-step training metric logging."""

import unittest

import numpy as np
import tensorflow as tf

from BRAID.MainModel import (
    getLossLogStr,
    shift_1s_to_ms_series,
    shift_ms_to_1s_series,
)
from BRAID.tools.tf_losses import (
    MaskedR2,
    compute_R2,
    masked_CC,
    masked_mse,
    masked_R2,
)

HORIZON = 4
SAMPLE_COUNT = 10
SIGNAL_TYPE = "cont"


def _loss_functions():
    """Return continuous-signal metrics used by the training logger."""
    return [masked_mse(), masked_R2(), masked_CC()]


def _targets() -> np.ndarray:
    """Return a non-flat scalar signal in time-first layout."""
    return np.arange(SAMPLE_COUNT, dtype=float).reshape(-1, 1)


def _assert_perfect_metrics(
    test_case: unittest.TestCase,
    log_message: str,
) -> None:
    """Assert that alignment padding does not contaminate logged metrics."""
    test_case.assertIn("{}-step".format(HORIZON), log_message)
    test_case.assertNotIn("nan", log_message.lower())
    test_case.assertIn("MSE=0", log_message)
    test_case.assertIn("R2=1", log_message)
    test_case.assertIn("CC=1", log_message)


class MultiStepMetricLoggingTest(unittest.TestCase):
    """Validate metrics after leading or trailing forecast alignment padding."""

    def test_leading_alignment_padding_is_excluded(self) -> None:
        """Score a shifted multi-step prediction at its target time points."""
        targets = _targets()
        raw_prediction = np.zeros((1, SAMPLE_COUNT))
        raw_prediction[:, : -(HORIZON - 1)] = targets[HORIZON - 1 :].T
        prediction = shift_ms_to_1s_series(
            raw_prediction,
            [HORIZON],
            time_first=False,
        )[0]

        log_message = getLossLogStr(
            targets,
            prediction,
            [HORIZON],
            SIGNAL_TYPE,
            _loss_functions(),
        )

        _assert_perfect_metrics(self, log_message)

    def test_trailing_alignment_padding_is_excluded(self) -> None:
        """Score a reverse-shifted prediction at its target time points."""
        targets = _targets()
        raw_prediction = np.zeros((1, SAMPLE_COUNT))
        raw_prediction[:, HORIZON - 1 :] = targets[: -(HORIZON - 1)].T
        prediction = shift_1s_to_ms_series(
            raw_prediction,
            [HORIZON],
            time_first=False,
        )[0]

        log_message = getLossLogStr(
            targets,
            prediction,
            [HORIZON],
            SIGNAL_TYPE,
            _loss_functions(),
        )

        _assert_perfect_metrics(self, log_message)


def test_r2_ignores_flat_dimensions_without_losing_valid_values() -> None:
    """A flat target dimension must not invalidate a varying dimension."""
    targets = tf.constant([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
    values = compute_R2(targets, targets).numpy()
    assert values[0] == 1.0
    assert np.isnan(values[1])
    assert masked_R2()(targets, targets).numpy() == 1.0


def test_stateful_r2_skips_all_flat_batches_and_epochs() -> None:
    """R2 remains undefined only when an entire epoch has no valid dimension."""
    flat = tf.constant([[1.0], [1.0]])
    varying = tf.constant([[0.0], [1.0], [2.0]])
    metric = MaskedR2()
    metric.update_state(flat, flat)
    assert np.isnan(metric.result().numpy())
    metric.reset_state()
    metric.update_state(flat, flat)
    metric.update_state(varying, varying)
    assert metric.result().numpy() == 1.0


if __name__ == "__main__":
    unittest.main()
