"""Train source components with mixed-session batches and bounded updates.

Stage data contain I (N,L,input), F (N,L,forward_input,8), T (N,L,output),
and P (4,N,L,output), the additive forecast prior. A stage directory records
history.jsonl, best.pkl, completed.pkl, and training.json. Completed payloads
hold native BRAID mapping arrays, constructor arguments, and update counts.
"""

import json
import logging
from pathlib import Path
from time import monotonic

import numpy as np
import tensorflow as tf

from BRAID.multisession import HORIZONS, MixedForecaster, mapping

from experiments.storage.cache import atomic_json
from experiments.multisession.store import dump_pickle, load_pickle

LOGGER = logging.getLogger(__name__)
MAX_HORIZON = max(HORIZONS)


def forward_inputs(u: np.ndarray) -> np.ndarray:
    """Prepare recorded future U using native RNNModel's time alignment."""
    length = u.shape[1]
    return np.stack(
        [
            u[:, np.minimum(np.arange(length) + h, length - 1)]
            for h in range(1, MAX_HORIZON + 1)
        ],
        axis=-1,
    )


def recurrent_data(values: dict, target: str = "Y") -> dict:
    """Build one independent-window recurrent objective from Y/U arrays."""
    truth = values[target]
    return dict(
        I=np.concatenate((values["Y"], values["U"]), axis=-1),
        F=forward_inputs(values["U"]),
        T=truth,
        P=np.zeros((len(HORIZONS), *truth.shape), dtype=np.float32),
    )


def select_batch(data: list[dict], indices: list[np.ndarray]) -> tuple:
    """Gather heterogeneous windows while retaining zero-sized sessions."""
    return tuple(
        tuple(
            tf.convert_to_tensor(
                row[key][:, chosen] if key == "P" else row[key][chosen],
                dtype=tf.float32,
            )
            for row, chosen in zip(data, indices)
        )
        for key in ("I", "F", "T", "P")
    )


class ReadoutStage:
    """Session-private nonlinear readouts of already fitted latent states."""

    def __init__(
        self,
        dimensions: list[tuple[int, int]],
        width: int,
        seed: int,
        behavior: bool = False,
    ) -> None:
        """Create native readouts for horizon-first latent input arrays."""
        self.models = []
        for i, (ni, no) in enumerate(dimensions):
            tf.keras.utils.set_random_seed(seed + i + 1)
            self.models.append(
                mapping(
                    ni,
                    no,
                    width,
                    f"readout_{i}",
                    "uniform" if behavior else "glorot_uniform",
                )
            )
        self.shared_variables = []
        self.private_variables = [m.trainable_weights for m in self.models]
        self.variables = [v for group in self.private_variables for v in group]
        self.loss_and_gradients = tf.function(
            self._loss_and_gradients,
            reduce_retracing=True,
        )
        self.validation_loss = tf.function(self.loss, reduce_retracing=True)

    def loss(
        self, inputs: tuple, forward: tuple, targets: tuple, priors: tuple
    ) -> tuple:
        """Sum valid horizon MSE per window; inputs are (B,L,nx,H)."""
        losses, counts = [], []
        for model, states, target in zip(self.models, inputs, targets):
            per_window = tf.zeros(tf.shape(target)[0])
            for i, h in enumerate(HORIZONS):
                predicted = model.apply_func(states[:, :-h, :, i])
                per_window += tf.reduce_mean(
                    tf.square(predicted - target[:, h:]),
                    axis=(1, 2),
                )
            losses.append(tf.reduce_sum(per_window))
            counts.append(tf.cast(tf.shape(target)[0], tf.float32))
        return tf.stack(losses), tf.stack(counts)

    def _loss_and_gradients(self, *batch: tuple) -> tuple:
        """Differentiate only the decoder objective."""
        with tf.GradientTape() as tape:
            losses, counts = self.loss(*batch)
            loss = tf.reduce_sum(losses) / tf.reduce_sum(counts)
        gradients = tape.gradient(loss, self.variables)
        for gradient in gradients:
            tf.debugging.assert_all_finite(gradient, "Readout gradient")
        return loss, gradients

    def weights(self) -> dict:
        """Export per-session readout arrays."""
        return {
            str(i): model.get_weights() for i, model in enumerate(self.models)
        }

    def restore(self, weights: dict) -> None:
        """Restore validated per-session readouts."""
        for i, model in enumerate(self.models):
            model.set_weights(weights[str(i)])


def validation_mean(model: object, data: list[dict], batch_size: int) -> float:
    """Calculate an equally weighted mean over all session validations."""
    sums = np.zeros(len(data))
    counts = np.zeros(len(data))
    for start in range(0, max(len(d["T"]) for d in data), batch_size):
        indices = [
            np.arange(start, min(start + batch_size, len(d["T"]))) for d in data
        ]
        losses, number = model.validation_loss(*select_batch(data, indices))
        sums += losses.numpy()
        counts += number.numpy()
    if np.any(counts == 0):
        raise ValueError("Every source session needs validation windows.")
    result = float(np.mean(sums / counts))
    if not np.isfinite(result):
        raise FloatingPointError("Nonfinite source validation loss.")
    return result


def train_stage(
    model: object,
    train: list[dict],
    validation: list[dict],
    directory: Path,
    settings: dict,
    seed: int,
) -> dict:
    """Fit a source component and restore its best macro-validation epoch."""
    directory.mkdir(parents=True, exist_ok=True)
    complete = directory / "completed.pkl"
    if complete.exists():
        payload = load_pickle(complete)
        model.restore(payload["weights"])
        return payload["training"]
    optimizers = [
        tf.keras.optimizers.legacy.Adam(settings["learning_rate"])
        for _ in range(len(train) + 1)
    ]
    groups = [model.shared_variables, *model.private_variables]
    rng = np.random.default_rng(seed)
    best, stale, best_step = float("inf"), 0, 0
    exposure = np.zeros(len(train), dtype=int)
    started = monotonic()
    for step in range(1, settings["maximum_updates"] + 1):
        draws = rng.integers(len(train), size=settings["batch_size"])
        counts = np.bincount(draws, minlength=len(train))
        indices = [
            rng.integers(len(d["T"]), size=int(count))
            for d, count in zip(train, counts)
        ]
        loss, gradients = model.loss_and_gradients(
            *select_batch(train, indices)
        )
        offset = 0
        for index, (optimizer, variables) in enumerate(zip(optimizers, groups)):
            current = gradients[offset : offset + len(variables)]
            offset += len(variables)
            if variables and (index == 0 or counts[index - 1]):
                optimizer.apply_gradients(zip(current, variables))
        exposure += counts
        value = float(loss.numpy())
        if not np.isfinite(value):
            raise FloatingPointError(f"Nonfinite training loss at {step}.")
        row = dict(
            update=step,
            loss=value,
            elapsed_seconds=monotonic() - started,
            sampled_windows=[chosen.tolist() for chosen in indices],
        )
        check = (
            step % settings["validation_interval"] == 0
            or step == settings["maximum_updates"]
        )
        if check:
            value = validation_mean(model, validation, settings["batch_size"])
            row["validation_loss"] = value
            if value < best:
                best, stale, best_step = value, 0, step
                dump_pickle(directory / "best.pkl", model.weights())
            else:
                stale += 1
            LOGGER.info(
                "%s update=%d loss=%.6g validation=%.6g best=%d",
                directory.resolve(),
                step,
                row["loss"],
                value,
                best_step,
            )
        with (directory / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
        if stale >= settings["patience"]:
            break
    dump_pickle(
        directory / "final_optimizer_states.pkl",
        [optimizer.get_weights() for optimizer in optimizers],
    )
    model.restore(load_pickle(directory / "best.pkl"))
    record = dict(
        updates=step,
        best_update=best_step,
        best_loss=best,
        window_exposures=exposure.tolist(),
        elapsed_seconds=monotonic() - started,
        settings=settings,
    )
    dump_pickle(complete, dict(weights=model.weights(), training=record))
    atomic_json(directory / "training.json", record)
    return record


def stage_predictions(
    model: MixedForecaster, data: list[dict], batch_size: int = 32
) -> tuple[list, list]:
    """Extract forecast/state arrays without retaining training graphs."""
    predictions = [[] for _ in data]
    states = [[] for _ in data]
    for start in range(0, max(len(d["T"]) for d in data), batch_size):
        indices = [
            np.arange(start, min(start + batch_size, len(d["T"]))) for d in data
        ]
        inputs, forward, _, _ = select_batch(data, indices)
        pred, state = model.forecast(inputs, forward)
        for i in range(len(data)):
            if len(indices[i]):
                predictions[i].append(pred[i].numpy())
                states[i].append(state[i].numpy())
    return (
        [np.concatenate(x, axis=1) for x in predictions],
        [np.concatenate(x, axis=1) for x in states],
    )
