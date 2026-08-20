"""Regression tests for multi-step training metric logging."""

import unittest

import numpy as np

from BRAID.MainModel import (
    getLossLogStr,
    shift_1s_to_ms_series,
    shift_ms_to_1s_series,
)
from BRAID.tools.tf_losses import masked_CC, masked_mse, masked_R2

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
    test_case.assertIn("MSE_maskV_None=0", log_message)
    test_case.assertIn("R2_maskV_None=1", log_message)
    test_case.assertIn("CC_maskV_None=1", log_message)


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


if __name__ == "__main__":
    unittest.main()
