"""Compatibility exports for native source FLOP arithmetic."""

from experiments.multisession.compute_scaling.flops import (
    MAX_HORIZON, ADAM_OPERATIONS_PER_PARAMETER,
    LOSS_FORWARD_OPERATIONS, LOSS_BACKWARD_OPERATIONS,
    dense_ops, session_ops, update_flops,
)
