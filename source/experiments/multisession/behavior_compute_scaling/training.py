"""Balanced checkpointed training for behavioral source components.

Inputs are per-session I/F/T/P window arrays and a native five-stage model.
Every completed segment contains checkpoint.pkl with weights, Adam state and
sampling state, plus training.json with exact sampled windows and update rates.
The common component store supplies checksum-validated immutable completion.
"""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
import logging

import numpy as np
import tensorflow as tf

from BRAID.multisession import HORIZONS, MixedForecaster
from experiments.multisession.behavior_compute_scaling.flops import (
    session_ops, update_flops,
)
from experiments.multisession.store import (
    completed, dump_pickle, job, load_pickle,
)
from experiments.multisession.training import ReadoutStage, select_batch
from experiments.storage.cache import atomic_json, fingerprint

MAX_HORIZON = max(HORIZONS)
EVALUATION_BATCH = 16
LOGGER = logging.getLogger(__name__)


def session_loss(model: object, session: int, batch: tuple) -> tf.Tensor:
    """Sum native stage loss for one session's (B, 128, D) batch."""
    inputs, forward, target, prior = batch
    if isinstance(model, ReadoutStage):
        readout = model.models[session]
        loss = tf.zeros(tf.shape(target)[0], tf.float32)
        for index, horizon in enumerate(HORIZONS):
            predicted = readout.apply_func(
                inputs[:, :-horizon, :, index]
            )
            difference = predicted - target[:, horizon:]
            loss += tf.reduce_mean(tf.square(difference), axis=(1, 2))
        return tf.reduce_sum(loss)
    if not isinstance(model, MixedForecaster):
        raise TypeError(f"Unknown source component {type(model)}.")
    maps = model.sessions[session]
    innovation = maps["K"].apply_func(inputs)
    transition = maps["A"].layers[0].kernel
    states = tf.scan(
        lambda previous, current: (
            tf.matmul(previous, transition) + current
        ),
        tf.transpose(innovation, (1, 0, 2)),
        initializer=tf.zeros((tf.shape(inputs)[0], model.latent)),
    )
    state = tf.transpose(states, (1, 0, 2))
    loss = tf.zeros(tf.shape(target)[0], tf.float32)
    for horizon in range(1, MAX_HORIZON + 1):
        if horizon > 1:
            state = (
                maps["Afw"].apply_func(state)
                + maps["Kfw"].apply_func(forward[..., horizon - 2])
            )
        if horizon in HORIZONS:
            index = HORIZONS.index(horizon)
            predicted = maps["C"].apply_func(state)
            difference = (
                predicted[:, :-horizon]
                + prior[index, :, :-horizon]
                - target[:, horizon:]
            )
            loss += tf.reduce_mean(tf.square(difference), axis=(1, 2))
    return tf.reduce_sum(loss)


def gradient_function(model: object, session: int):
    """Compile gradients for one session's active variable groups."""
    variables = model.shared_variables + model.private_variables[session]

    @tf.function(reduce_retracing=True)
    def calculate(batch: tuple, denominator: tf.Tensor):
        with tf.GradientTape() as tape:
            loss = session_loss(model, session, batch) / denominator
        gradients = tape.gradient(loss, variables)
        for gradient in gradients:
            tf.debugging.assert_all_finite(gradient, "Source gradient")
        tf.debugging.assert_all_finite(loss, "Source training loss")
        return loss, gradients

    return calculate


def learning_rate(update: int, settings: dict, branch: bool) -> float:
    """Return linear warmup or branch-decay Adam rate."""
    base = settings["learning_rate"]
    if branch:
        end = settings["branch_updates"]
        if end == 1:
            return 0.0
        return base * (1 - (update - 1) / (end - 1))
    warm = settings["warmup_updates"]
    start = settings["warmup_start_learning_rate"]
    if warm == 1:
        return base
    return start + (base - start) * min(
        (update - 1) / (warm - 1), 1
    )


def optimizer_groups(model: object, settings: dict) -> list:
    """Initialize independent shared and session-private Adam states."""
    groups = [model.shared_variables, *model.private_variables]
    optimizers = [
        tf.keras.optimizers.legacy.Adam(settings["learning_rate"])
        for _ in groups
    ]
    for optimizer, variables in zip(optimizers, groups):
        if variables:
            optimizer._create_all_weights(variables)
    return optimizers


def restore_state(
    model: object, optimizers: list, payload: dict,
) -> np.random.Generator:
    """Restore model arrays, Adam slots, and sampling RNG exactly."""
    model.restore(payload["weights"])
    for optimizer, values in zip(optimizers, payload["optimizers"]):
        optimizer.set_weights(values)
    rng = np.random.default_rng()
    rng.bit_generator.state = deepcopy(payload["sampling_state"])
    return rng


def selected_batch(row: dict, indices: np.ndarray) -> tuple:
    """Gather one session with the native batch axis layout."""
    columns = select_batch([row], [indices])
    return tuple(group[0] for group in columns)


def balanced_update(
    model: object, data: list[dict], optimizers: list,
    rng: np.random.Generator, local: int, microbatch: int,
    devices: tuple[str, str], functions: list,
    pool: ThreadPoolExecutor | None = None,
) -> tuple[float, list[list[int]]]:
    """Sum session-local gradients before one effective-batch update."""
    if local <= 0 or microbatch <= 0:
        raise ValueError("Session batch and microbatch must be positive.")
    if any(len(row["T"]) == 0 for row in data):
        raise ValueError("Each source session needs training windows.")
    indices = [
        rng.integers(len(row["T"]), size=local) for row in data
    ]
    total = len(data) * local
    groups = [model.shared_variables, *model.private_variables]

    def replica(replica_index: int) -> tuple[float, list]:
        subtotal = 0.0
        gradients = [
            [tf.zeros_like(variable) for variable in group]
            for group in groups
        ]
        with tf.device(devices[replica_index]):
            for session in range(replica_index, len(data), 2):
                for start in range(0, local, microbatch):
                    chosen = indices[session][start:start + microbatch]
                    batch = selected_batch(data[session], chosen)
                    loss, values = functions[session](
                        batch, tf.constant(float(total), tf.float32)
                    )
                    subtotal += float(loss.numpy())
                    shared = len(groups[0])
                    for number, value in enumerate(values[:shared]):
                        gradients[0][number] += value
                    for number, value in enumerate(values[shared:]):
                        gradients[session + 1][number] += value
        return subtotal, gradients

    if pool is None:
        with ThreadPoolExecutor(max_workers=2) as temporary:
            results = list(temporary.map(replica, (0, 1)))
    else:
        results = list(pool.map(replica, (0, 1)))
    value = sum(loss for loss, _ in results)
    if not np.isfinite(value):
        raise FloatingPointError("Nonfinite balanced source loss.")
    sums = [
        [tf.zeros_like(variable) for variable in group]
        for group in groups
    ]
    for _, replica_sums in results:
        for index, gradients in enumerate(replica_sums):
            for number, gradient in enumerate(gradients):
                sums[index][number] += gradient
    for optimizer, variables, gradients in zip(optimizers, groups, sums):
        if variables:
            optimizer.apply_gradients(zip(gradients, variables))
    return value, [chosen.tolist() for chosen in indices]


def checkpoint_state(
    model: object, optimizers: list, rng: np.random.Generator,
    update: int,
) -> dict:
    """Capture weights, Adam slots and sampling state at one update."""
    return dict(
        weights=model.weights(),
        optimizers=[optimizer.get_weights() for optimizer in optimizers],
        sampling_state=deepcopy(rng.bit_generator.state),
        update=update,
    )


def evaluate_sessions(model: object, data: list[dict]) -> tuple[list, int]:
    """Return per-session mean loss and forward FLOPs for held-out windows.

    Parameters
    ----------
    model : object
        Native recurrent or readout model with one private map per session.
    data : list of dict
        I/F/T/P arrays; T has shape (windows, 128, output_channels).

    Returns
    -------
    losses : list of float
        Mean window loss per session, summed over native forecast horizons.
    flops : int
        Model and loss forward arithmetic; no gradients or optimizer work.
    """
    functions = getattr(model, "_evaluation_functions", None)
    if functions is None:
        functions = [
            tf.function(
                lambda batch, index=index: session_loss(model, index, batch),
                reduce_retracing=True,
            ) for index in range(len(model.private_variables))
        ]
        model._evaluation_functions = functions
    losses, flops = [], 0
    for index, row in enumerate(data):
        count, length, width = row["T"].shape
        if not count:
            raise ValueError(f"Empty evaluation windows for session {index}.")
        loss = 0.0
        for start in range(0, count, EVALUATION_BATCH):
            chosen = np.arange(start, min(start + EVALUATION_BATCH, count))
            loss += float(functions[index](selected_batch(row, chosen)))
            flops += session_ops(model, index, len(chosen), length, width)[0]
        losses.append(loss / count)
    if not losses or not np.isfinite(losses).all():
        raise FloatingPointError("Expected nonempty finite evaluation losses.")
    return losses, flops


def validation_due(update: int, batch: int, interval: int, cap: int) -> bool:
    """Check after crossing an exposure threshold and at the final update."""
    if min(update, batch, interval, cap) <= 0 or update > cap:
        raise ValueError("Invalid positive exposure cadence or update cap.")
    return update == cap or (
        update * batch // interval > (update - 1) * batch // interval
    )


def fit_segment(
    model: object, data: list[dict], root: Path, identity: dict,
    settings: dict, seed: int, start: int, stop: int,
    parent: Path | None, devices: tuple[str, str], microbatch: int,
    branch: bool = False,
    validation: list[dict] | None = None,
) -> tuple[Path, dict]:
    """Complete or reuse one immutable segment and its full state."""
    if validation is not None and (branch or parent is not None or start):
        raise ValueError("Fixed stages must start without a branch or parent.")
    if stop <= start:
        raise ValueError(f"Invalid update segment {start} to {stop}.")
    parent_record = completed(parent) if parent is not None else None
    if parent is not None and parent_record is None:
        raise ValueError(f"Incomplete source parent {parent.resolve()}.")
    scientific = dict(
        identity, start=start, stop=stop, branch=branch,
        parent=parent.name if parent is not None else None,
        parent_checksums=(
            parent_record["checksums"] if parent_record is not None else None
        ),
    )
    if validation is not None:
        scientific["fixed_stage"] = True
    if parent is not None:
        parent_payload = load_pickle(parent / "checkpoint.pkl")
        if parent_payload["update"] != start:
            raise ValueError(
                f"Expected parent at update {start}, "
                f"got {parent_payload['update']}."
            )
    directory = root / fingerprint(scientific)
    with job(directory, scientific) as needed:
        if needed:
            optimizers = optimizer_groups(model, settings)
            if parent is None:
                rng = np.random.default_rng(seed)
            else:
                payload = load_pickle(parent / "checkpoint.pkl")
                rng = restore_state(model, optimizers, payload)
            local = settings["session_batch_sizes"][str(len(data))]
            initial = checkpoint_state(model, optimizers, rng, start)
            attempt = min(local, microbatch)
            while True:
                rng = restore_state(model, optimizers, initial)
                functions = getattr(model, "_balanced_functions", None)
                if functions is None:
                    functions = [
                        gradient_function(model, index)
                        for index in range(len(data))
                    ]
                    model._balanced_functions = functions
                history = []
                best, stale = float("inf"), 0
                best_payload = None
                evaluation_flops = 0
                try:
                    with ThreadPoolExecutor(max_workers=2) as pool:
                        for update in range(start + 1, stop + 1):
                            relative = update - start if branch else update
                            rate = (
                                settings["learning_rate"]
                                if validation is not None else
                                learning_rate(relative, settings, branch)
                            )
                            for optimizer in optimizers:
                                optimizer.learning_rate.assign(rate)
                            loss, selected = balanced_update(
                                model, data, optimizers, rng, local,
                                attempt, devices, functions, pool,
                            )
                            row = dict(
                                update=update, loss=loss,
                                sampled_windows=selected,
                                learning_rate=rate,
                            )
                            history.append(row)
                            if validation is not None and validation_due(
                                update, local * len(data),
                                settings["validation_exposure"], stop,
                            ):
                                losses, work = evaluate_sessions(
                                    model, validation
                                )
                                evaluation_flops += work
                                value = float(np.mean(losses))
                                row["validation_loss"] = value
                                row["session_validation_losses"] = losses
                                if value < best:
                                    best, stale = value, 0
                                    best_payload = checkpoint_state(
                                        model, optimizers, rng, update
                                    )
                                else:
                                    stale += 1
                                LOGGER.info(
                                    "%s update=%d train=%.6g inner=%.6g "
                                    "best=%d stale=%d",
                                    identity["stage"], update, loss, value,
                                    best_payload["update"], stale,
                                )
                                if stale >= settings["fixed_patience"]:
                                    break
                            elif update == stop or update % 25 == 0:
                                LOGGER.info(
                                    "%s update=%d/%d loss=%.6g lr=%.4g",
                                    identity.get("stage", "component"),
                                    update, stop, loss, rate,
                                )
                except tf.errors.ResourceExhaustedError:
                    if attempt == 1:
                        raise
                    attempt = max(1, attempt // 2)
                    continue
                break
            arithmetic = update_flops(model, data, local, attempt)
            actual_stop = history[-1]["update"]
            payload = checkpoint_state(model, optimizers, rng, actual_stop)
            if validation is not None:
                if best_payload is None:
                    raise RuntimeError("Fixed stage has no valid checkpoint.")
                payload = best_payload
                restore_state(model, optimizers, payload)
            dump_pickle(directory / "checkpoint.pkl", payload)
            atomic_json(directory / "training.json", dict(
                start=start, stop=actual_stop, requested_stop=stop,
                selected_update=payload["update"], branch=branch,
                selection=("inner_validation" if validation is not None
                           else "scheduled_endpoint"),
                best_validation_loss=(best if validation is not None else None),
                evaluation_flops=evaluation_flops,
                effective_batch=local * len(data),
                session_batch=local, microbatch=attempt,
                devices=list(devices), history=history,
                flops_per_update=arithmetic,
                total_flops=arithmetic["total"] * len(history),
            ))
        else:
            model.restore(
                load_pickle(directory / "checkpoint.pkl")["weights"]
            )
    return directory, load_pickle(directory / "checkpoint.pkl")
