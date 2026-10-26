"""Count native dense/recurrent source arithmetic for every update.

Input is a fitted component, per-session window shapes, effective session
batch and actual microbatch. Output counts forward, reverse-mode and Adam
scalar arithmetic. Matrix multiply uses two operations per multiply-add;
kernel launches, memory traffic, communication and data loading are excluded.
"""

from math import ceil

from BRAID.multisession import HORIZONS, MixedForecaster
from experiments.multisession.training import ReadoutStage

MAX_HORIZON = max(HORIZONS)
ADAM_OPERATIONS_PER_PARAMETER = 14
LOSS_FORWARD_OPERATIONS = 4
LOSS_BACKWARD_OPERATIONS = 3


def dense_ops(mapping: object, rows: int) -> tuple[int, int]:
    """Count forward and adjoint operations for a mapping's Dense layers."""
    forward = 0
    backward = 0
    for layer in mapping.layers:
        width_in, width_out = map(int, layer.kernel.shape)
        matmul = 2 * rows * width_in * width_out
        bias = rows * width_out if layer.use_bias else 0
        activation = (
            rows * width_out
            if layer.activation.__name__ != "linear" else 0
        )
        forward += matmul + bias + activation
        backward += 2 * matmul + bias + activation
    return forward, backward


def session_ops(
    model: object, session: int, batch: int, length: int,
    output_width: int,
) -> tuple[int, int]:
    """Count one source session's actual stage forward and backward pass."""
    forward = 0
    backward = 0
    if isinstance(model, ReadoutStage):
        mapping = model.models[session]
        for horizon in HORIZONS:
            rows = batch * (length - horizon)
            fwd, bwd = dense_ops(mapping, rows)
            forward += fwd + LOSS_FORWARD_OPERATIONS * rows * output_width
            backward += bwd + LOSS_BACKWARD_OPERATIONS * rows * output_width
        return forward, backward
    if not isinstance(model, MixedForecaster):
        raise TypeError(f"Unknown component {type(model)}.")
    maps = model.sessions[session]
    rows = batch * length
    for name, repeats in (
        ("K", 1), ("Afw", MAX_HORIZON - 1),
        ("Kfw", MAX_HORIZON - 1),
        ("C", len(HORIZONS)),
    ):
        fwd, bwd = dense_ops(maps[name], rows)
        forward += repeats * fwd
        backward += repeats * bwd
    transition = 2 * rows * model.latent * model.latent
    forward += transition + rows * model.latent
    backward += 2 * transition + rows * model.latent
    forward += 2 * (MAX_HORIZON - 1) * rows * model.latent
    backward += 2 * (MAX_HORIZON - 1) * rows * model.latent
    for horizon in HORIZONS:
        elements = batch * (length - horizon) * output_width
        forward += LOSS_FORWARD_OPERATIONS * elements
        backward += LOSS_BACKWARD_OPERATIONS * elements
    return forward, backward


def update_flops(
    model: object, data: list[dict], local: int, microbatch: int,
) -> dict:
    """Count each session microbatch and one optimizer update."""
    if microbatch <= 0 or local <= 0:
        raise ValueError("Positive session batch and microbatch required.")
    sessions = []
    for index, row in enumerate(data):
        forward = backward = 0
        length = int(row["T"].shape[1])
        output_width = int(row["T"].shape[2])
        for start in range(0, local, microbatch):
            batch = min(microbatch, local - start)
            fwd, bwd = session_ops(
                model, index, batch, length, output_width
            )
            forward += fwd
            backward += bwd
        active = [*model.shared_variables]
        active.extend(model.private_variables[index])
        accumulation = ceil(local / microbatch) * (
            sum(int(variable.shape.num_elements()) for variable in active)
            + 1
        )
        sessions.append(dict(
            forward=forward, backward=backward,
            accumulation=accumulation,
            total=forward + backward + accumulation,
        ))
    active = [*model.shared_variables]
    for group in model.private_variables:
        active.extend(group)
    optimizer = ADAM_OPERATIONS_PER_PARAMETER * sum(
        int(variable.shape.num_elements()) for variable in active
    )
    reduction = 2 * sum(
        int(variable.shape.num_elements()) for variable in active
    ) + 1
    result = dict(
        forward=sum(item["forward"] for item in sessions),
        backward=sum(item["backward"] for item in sessions),
        accumulation=(
            sum(item["accumulation"] for item in sessions)
            + reduction
        ),
        optimizer=optimizer,
        sessions=sessions,
    )
    result["total"] = sum(
        result[key] for key in (
            "forward", "backward", "accumulation", "optimizer"
        )
    )
    if result["total"] <= 0:
        raise ValueError("Source component has no arithmetic FLOPs.")
    return result
