"""Differentiable mixed-session BRAID forecasting primitives.

Inputs are lists of session windows Y/U, each (B_s, L, C_s). Linear
session encoders project heterogeneous inputs into a common (B, L, nx)
tensor. Filtering and forward recursion use native RegressionModel weights.
Outputs are per-session forecasts (H, B_s, L, output_s) and a scalar loss.
Checkpoints contain named mapping weight arrays, never channel substitutions.
"""

from copy import deepcopy

import numpy as np
import tensorflow as tf

from .RegressionModel import RegressionModel

HORIZONS = (1, 2, 4, 8)
NEURAL_WIDTH = 256
BEHAVIOR_WIDTH = 64
LINEAR = dict(
    units=(),
    use_bias=False,
    activation="linear",
    output_activation="linear",
    kernel_initializer="uniform",
)


def mapping(
    inputs: int,
    outputs: int,
    width: int,
    name: str,
    initializer: str = "glorot_uniform",
) -> RegressionModel:
    """Construct a native linear map or one-hidden-layer decoder.

    Parameters
    ----------
    inputs, outputs : int
        Input/output coordinate counts.
    width : int
        Hidden width; zero selects a linear map.
    name : str
        Stable layer identifier.
    initializer : str, optional
        Decoder kernel initializer; default 'glorot_uniform'.
    """
    settings = deepcopy(LINEAR)
    if width:
        settings.update(
            units=(width,),
            use_bias=True,
            activation="relu",
            kernel_initializer=initializer,
        )
    return RegressionModel(inputs, outputs, name=name, **settings)


def unique_variables(maps: list[RegressionModel]) -> list:
    """Collect each trainable TensorFlow variable exactly once."""
    result, seen = [], set()
    for item in maps:
        for variable in item.trainable_weights:
            if variable.ref() not in seen:
                result.append(variable)
                seen.add(variable.ref())
    return result


class MixedForecaster:
    """Linear filtering/forecast dynamics with session-specific readouts."""

    def __init__(
        self,
        dimensions: list[tuple[int, int]],
        latent: int,
        sharing: str,
        seed: int,
        decoder_width: int = NEURAL_WIDTH,
        input_dimensions: list[int] | None = None,
        forward_dimensions: int = 2,
    ) -> None:
        """Build one source stage with explicit shared module identity.

        Parameters
        ----------
        dimensions : list of (neural, output) pairs
            Session-specific input and prediction widths.
        latent : int
            Common recurrent state width.
        sharing : str
            none, A, AK, AKC, or AAK (A, Afw, and Kfw).
        seed : int
            Paired initialization seed.
        decoder_width : int, optional
            Decoder hidden size; default 256.
        input_dimensions : list of int, optional
            Filtering input widths; default neural + 2.
        forward_dimensions : int, optional
            Observable forward-input width; default 2.
        """
        if sharing not in ("none", "A", "AK", "AKC", "AAK"):
            raise ValueError(f"Unknown sharing profile {sharing}.")
        self.latent, self.sharing = latent, sharing
        self.dimensions = dimensions
        self.forward_dimensions = forward_dimensions
        self.shared = {}
        tf.keras.utils.set_random_seed(seed)
        if sharing != "none":
            self.shared["Afw"] = mapping(latent, latent, 0, "shared_Afw")
        if sharing == "AAK":
            self.shared["A"] = mapping(latent, latent, 0, "shared_A")
        if sharing in ("AK", "AKC", "AAK"):
            self.shared["Kfw"] = mapping(
                forward_dimensions, latent, 0, "shared_Kfw"
            )
        if sharing == "AKC":
            widths = {d[1] for d in dimensions}
            if len(widths) != 1:
                raise ValueError("Shared Cz requires identical coordinates.")
            self.shared["C"] = mapping(
                latent,
                dimensions[0][1],
                decoder_width,
                "shared_C",
                "uniform",
            )
        self.sessions = []
        for index, (neural, output) in enumerate(dimensions):
            tf.keras.utils.set_random_seed(seed + index + 1)
            inputs = (
                neural + 2
                if input_dimensions is None
                else input_dimensions[index]
            )
            widths = dict(
                A=(latent, latent, 0),
                K=(inputs, latent, 0),
                Afw=(latent, latent, 0),
                Kfw=(forward_dimensions, latent, 0),
                C=(latent, output, decoder_width),
            )
            maps = {}
            for key, (ni, no, hidden) in widths.items():
                maps[key] = (
                    self.shared[key]
                    if key in self.shared
                    else mapping(
                        ni,
                        no,
                        hidden,
                        f"s{index}_{key}",
                        "uniform"
                        if decoder_width == BEHAVIOR_WIDTH
                        else "glorot_uniform",
                    )
                )
            self.sessions.append(maps)
        self.shared_variables = unique_variables(list(self.shared.values()))
        refs = {v.ref() for v in self.shared_variables}
        self.private_variables = [
            [
                v
                for v in unique_variables(list(maps.values()))
                if v.ref() not in refs
            ]
            for maps in self.sessions
        ]
        self.variables = self.shared_variables + [
            v for group in self.private_variables for v in group
        ]
        self.loss_and_gradients = tf.function(
            self._loss_and_gradients,
            reduce_retracing=True,
        )
        self.validation_loss = tf.function(self.loss, reduce_retracing=True)
        self.forecast = tf.function(self.predict, reduce_retracing=True)

    def states(self, inputs: tuple) -> tuple:
        """Filter session inputs into a common (B, L, nx) state tensor."""
        sizes = [tf.shape(value)[0] for value in inputs]
        innovations = tf.concat(
            [
                maps["K"].apply_func(value)
                for maps, value in zip(self.sessions, inputs)
            ],
            axis=0,
        )
        transitions = tf.concat(
            [
                tf.repeat(maps["A"].layers[0].kernel[None], size, axis=0)
                for maps, size in zip(self.sessions, sizes)
            ],
            axis=0,
        )
        initial = tf.zeros((tf.shape(innovations)[0], self.latent))
        states = tf.scan(
            lambda previous, innovation: (
                tf.squeeze(tf.matmul(previous[:, None, :], transitions), axis=1)
                + innovation
            ),
            tf.transpose(innovations, (1, 0, 2)),
            initializer=initial,
        )
        return tf.transpose(states, (1, 0, 2)), sizes

    def predict(
        self, inputs: tuple, forward: tuple, horizons: tuple = HORIZONS
    ) -> tuple:
        """Forecast from observed inputs without crossing window edges.

        Forward arrays have shape (B_s, L, D, max_horizon), matching native
        RNNModel's prepared forward inputs. The last axis includes step 1.
        Returns session tuples of predictions and states, each horizon-first.
        """
        state, sizes = self.states(inputs)
        by_session = [[] for _ in inputs]
        latent_states = [[] for _ in inputs]
        for horizon in range(1, max(horizons) + 1):
            if horizon > 1:
                chunks = tf.split(state, sizes, axis=0)
                driven = tf.concat(
                    [
                        maps["Kfw"].apply_func(u[..., horizon - 2])
                        for maps, u in zip(self.sessions, forward)
                    ],
                    axis=0,
                )
                if "Afw" in self.shared:
                    state = self.shared["Afw"].apply_func(state) + driven
                else:
                    state = (
                        tf.concat(
                            [
                                maps["Afw"].apply_func(chunk)
                                for maps, chunk in zip(self.sessions, chunks)
                            ],
                            axis=0,
                        )
                        + driven
                    )
            if horizon in horizons:
                chunks = tf.split(state, sizes, axis=0)
                for i, (maps, chunk) in enumerate(zip(self.sessions, chunks)):
                    by_session[i].append(maps["C"].apply_func(chunk))
                    latent_states[i].append(chunk)
        return (
            tuple(tf.stack(x) for x in by_session),
            tuple(tf.stack(x) for x in latent_states),
        )

    def loss(
        self, inputs: tuple, forward: tuple, targets: tuple, priors: tuple
    ) -> tuple:
        """Return per-session loss sums and counts over valid forecasts."""
        predicted, _ = self.predict(inputs, forward)
        losses, counts = [], []
        for pred, target, prior in zip(predicted, targets, priors):
            per_window = tf.zeros(tf.shape(target)[0])
            for index, horizon in enumerate(HORIZONS):
                error = (
                    pred[index, :, :-horizon]
                    + prior[index, :, :-horizon]
                    - target[:, horizon:]
                )
                per_window += tf.reduce_mean(tf.square(error), axis=(1, 2))
            losses.append(tf.reduce_sum(per_window))
            counts.append(tf.cast(tf.shape(target)[0], tf.float32))
        return tf.stack(losses), tf.stack(counts)

    def _loss_and_gradients(
        self, inputs: tuple, forward: tuple, targets: tuple, priors: tuple
    ) -> tuple:
        """Differentiate a window-weighted heterogeneous mini-batch."""
        with tf.GradientTape() as tape:
            losses, counts = self.loss(inputs, forward, targets, priors)
            loss = tf.reduce_sum(losses) / tf.reduce_sum(counts)
        gradients = tape.gradient(loss, self.variables)
        for gradient in gradients:
            tf.debugging.assert_all_finite(gradient, "Nonfinite gradient")
        tf.debugging.assert_all_finite(loss, "Nonfinite source loss")
        return loss, gradients

    def weights(self) -> dict:
        """Export native mapping arrays under stable session/module keys."""
        return {
            f"{session}/{key}": model.get_weights()
            for session, maps in enumerate(self.sessions)
            for key, model in maps.items()
        }

    def restore(self, weights: dict) -> None:
        """Restore all mappings, checking shared copies for consistency."""
        seen = {}
        for session, maps in enumerate(self.sessions):
            for key, model in maps.items():
                values = weights[f"{session}/{key}"]
                if id(model) in seen:
                    for first, second in zip(seen[id(model)], values):
                        np.testing.assert_array_equal(first, second)
                model.set_weights(values)
                seen[id(model)] = values

    def transferable(self) -> dict:
        """Export only explicitly shared modules for frozen adaptation."""
        return {key: model.get_weights() for key, model in self.shared.items()}
