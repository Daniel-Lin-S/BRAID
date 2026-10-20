"""Sequence-level Neuroformer dynamics for BRAID fitting and prediction.

Inputs are 128-bin neural and two-coordinate target sequences, optional
stage-one state sequences, and native BRAID forecast/decoder arrays. Outputs
are observation predictions and latent states for each configured horizon.
Four 32-bin query chunks use up to 96 preceding bins as neural history.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import tensorflow as tf

from .RegressionModel import RegressionModel


HISTORY_BINS = 96
QUERY_BINS = 32
WINDOW_BINS = HISTORY_BINS + QUERY_BINS
TRANSFORMER_BLOCKS = 6
ATTENTION_HEADS = 8
FEEDFORWARD_MULTIPLIER = 4
ATTENTION_DROPOUT = 0.2
STIMULUS_DIMS = 2
MAX_HORIZON = 32
TOKEN_TYPES = 4
START_TYPES = 3


def _time_embedding(position: tf.Tensor, width: int) -> tf.Tensor:
    """Return sinusoidal embeddings with ``position.shape + (width,)``."""
    indices = tf.cast(tf.range(width), tf.float32)
    scales = tf.pow(
        10000.0, 2.0 * tf.floor(indices / 2.0) / width
    )
    angles = tf.cast(position[..., None], tf.float32) / scales
    even = tf.equal(tf.math.mod(tf.range(width), 2), 0)
    return tf.where(even, tf.sin(angles), tf.cos(angles))


class AttentionBlock(tf.keras.layers.Layer):
    """Apply one Neuroformer attention, residual MLP, and layer norms."""

    def __init__(self, width: int, name: str) -> None:
        super().__init__(name=name)
        self.query_norm = tf.keras.layers.LayerNormalization()
        self.mlp_norm = tf.keras.layers.LayerNormalization()
        self.output_norm = tf.keras.layers.LayerNormalization()
        self.attention = tf.keras.layers.MultiHeadAttention(
            ATTENTION_HEADS, width // ATTENTION_HEADS,
            dropout=ATTENTION_DROPOUT,
        )
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(width * FEEDFORWARD_MULTIPLIER),
            tf.keras.layers.Activation(tf.nn.gelu),
            tf.keras.layers.Dense(width),
            tf.keras.layers.Dropout(ATTENTION_DROPOUT),
        ])

    def call(
        self, query: tf.Tensor, memory: tf.Tensor,
        mask: Optional[tf.Tensor], training: Optional[bool] = None,
    ) -> tf.Tensor:
        """Return updated queries of shape ``(batch, queries, width)``."""
        attended = self.attention(
            self.query_norm(query), memory,
            attention_mask=mask, training=training,
        )
        updated = query + attended
        updated += self.mlp(self.mlp_norm(updated), training=training)
        return self.output_norm(updated)


@tf.keras.utils.register_keras_serializable(package="BRAID")
class CausalTransformerCell(tf.keras.layers.Layer):
    """Shared sequence attention stack and native BRAID observation heads."""

    def __init__(
        self,
        units: int,
        ny_out: int,
        neural_dims: int,
        stage_one_dims: int,
        nft: int,
        n1_in: int,
        steps_ahead: Optional[Sequence[int]],
        CSettings: dict,
        KSettings: dict,
        missing_marker: Optional[float],
        has_prior_pred: bool,
        observable_U_in_Cfw: bool,
        sequence_bins: int = WINDOW_BINS,
        include_stage_one_state: bool = True,
        name: str = "causal_transformer",
        **kwargs,
    ) -> None:
        super().__init__(name=name, **kwargs)
        if units <= 0 or units % ATTENTION_HEADS:
            raise ValueError(
                f"Transformer width must be positive and divisible by "
                f"{ATTENTION_HEADS}; got {units}."
            )
        if neural_dims <= 0 or stage_one_dims < 0:
            raise ValueError(
                "Transformer requires positive neural dimensions and "
                "nonnegative stage-one dimensions."
            )
        if sequence_bins < 1 or sequence_bins > WINDOW_BINS:
            raise ValueError(
                f"Expected 1 to {WINDOW_BINS} sequence bins, "
                f"got {sequence_bins}."
            )
        horizons = list(steps_ahead or [1])
        if not horizons or min(horizons) < 1 or max(horizons) > MAX_HORIZON:
            raise ValueError(
                f"Forecast horizons must be in 1 to {MAX_HORIZON}; "
                f"got {horizons}."
            )
        if sorted(set(horizons)) != horizons:
            raise ValueError(
                f"Forecast horizons must be sorted and unique; "
                f"got {horizons}."
            )
        self.units = units
        self.ny_out = ny_out
        self.CSettings = dict(CSettings)
        self.KSettings = dict(KSettings)
        self.neural_dims = neural_dims
        self.stage_one_dims = stage_one_dims
        self.nft = nft
        self.n1_in = n1_in
        self.steps_ahead = horizons
        self.sequence_bins = sequence_bins
        self.padded_bins = (
            (sequence_bins + QUERY_BINS - 1) // QUERY_BINS
        ) * QUERY_BINS
        self.chunk_count = self.padded_bins // QUERY_BINS
        self.missing_marker = missing_marker
        self.has_prior_pred = has_prior_pred
        self.observable_U_in_Cfw = observable_U_in_Cfw
        self.include_stage_one_state = include_stage_one_state
        neural_settings = dict(KSettings)
        neural_settings["name"] = f"{name}_K_"
        self.neural_encoder = RegressionModel(
            neural_dims, units, **neural_settings
        )
        self.stimulus_encoder = tf.keras.layers.Dense(
            units, name="target_K_fw"
        )
        self.stage_one_encoder = (
            tf.keras.layers.Dense(units, name="stage_one_token")
            if stage_one_dims and include_stage_one_state else None
        )
        self.forecast_encoder = tf.keras.layers.Dense(
            units, name="forecast_query"
        )
        self.position_encoder = tf.keras.layers.Embedding(
            WINDOW_BINS, units, name="learned_position"
        )
        self.token_types = self.add_weight(
            "token_types", shape=(TOKEN_TYPES, units),
            initializer="zeros",
        )
        self.start_tokens = self.add_weight(
            "start_tokens", shape=(START_TYPES, units),
            initializer="glorot_uniform",
        )
        self.horizon_embeddings = self.add_weight(
            "horizon_embeddings", shape=(MAX_HORIZON, units),
            initializer="glorot_uniform",
        )
        self.history_blocks = [
            AttentionBlock(units, f"history_block_{index}")
            for index in range(TRANSFORMER_BLOCKS)
        ]
        self.stimulus_blocks = [
            AttentionBlock(units, f"stimulus_block_{index}")
            for index in range(TRANSFORMER_BLOCKS)
        ]
        self.causal_blocks = [
            AttentionBlock(units, f"causal_block_{index}")
            for index in range(TRANSFORMER_BLOCKS)
        ]
        decoder_settings = dict(CSettings)
        decoder_settings["has_prior_pred"] = has_prior_pred
        decoder_settings["name"] = f"{name}_C_"
        self.C = RegressionModel(
            units + nft, ny_out, **decoder_settings
        )
        self.Cfw = None
        if nft and not observable_U_in_Cfw:
            forecast_settings = dict(decoder_settings)
            forecast_settings["name"] = f"{name}_Cfw_"
            self.Cfw = RegressionModel(
                units, ny_out, **forecast_settings
            )
        self.initial_state = self.add_weight(
            "initial_state", shape=(units,), initializer="zeros",
            trainable=False,
        )
        self.state_size = units
        self.output_size = [ny_out] * len(horizons)
        self.output_size += [units] * (2 * len(horizons))

    def get_config(self) -> dict:
        """Return constructor arguments for Keras model serialization."""
        config = super().get_config()
        config.update(
            units=self.units,
            ny_out=self.ny_out,
            neural_dims=self.neural_dims,
            stage_one_dims=self.stage_one_dims,
            nft=self.nft,
            n1_in=self.n1_in,
            steps_ahead=self.steps_ahead,
            CSettings=self.CSettings,
            KSettings=self.KSettings,
            missing_marker=self.missing_marker,
            has_prior_pred=self.has_prior_pred,
            observable_U_in_Cfw=self.observable_U_in_Cfw,
            sequence_bins=self.sequence_bins,
            include_stage_one_state=self.include_stage_one_state,
        )
        return config

    def get_initial_state(
        self, inputs: Optional[tf.Tensor] = None,
        batch_size: Optional[int] = None,
        dtype: Optional[tf.dtypes.DType] = None,
    ) -> tf.Tensor:
        """Return initial latents of shape ``(batch, units)``."""
        if batch_size is None:
            if inputs is None:
                raise ValueError("Initial state needs inputs or batch_size.")
            batch_size = tf.shape(inputs)[0]
        state = tf.tile(self.initial_state[None, :], [batch_size, 1])
        return tf.cast(state, dtype or self.initial_state.dtype)

    def _valid(self, values: tf.Tensor) -> tf.Tensor:
        """Return finite, present values with shape ``values.shape[:-1]``."""
        valid = tf.reduce_all(tf.math.is_finite(values), axis=-1)
        if self.missing_marker is not None:
            valid = tf.logical_and(
                valid,
                tf.reduce_all(
                    tf.not_equal(values, self.missing_marker),
                    axis=-1,
                ),
            )
        return valid

    def _pad_time(self, values: tf.Tensor) -> tf.Tensor:
        """Pad sequence time to a whole 32-bin chunk."""
        gap = self.padded_bins - self.sequence_bins
        if gap == 0:
            return values
        paddings = [[0, 0], [0, gap]]
        paddings.extend([[0, 0]] * (len(values.shape) - 2))
        return tf.pad(values, paddings)

    def _chunks(self, values: tf.Tensor) -> tf.Tensor:
        """Fold sequence into ``(batch * chunks, 32, features)``."""
        return tf.reshape(
            values,
            (-1, QUERY_BINS, tf.shape(values)[-1]),
        )

    def _history(self, values: tf.Tensor) -> tf.Tensor:
        """Fold preceding 96 bins into one history per query chunk."""
        padded = tf.pad(values, [
            [0, 0], [HISTORY_BINS, 0], [0, 0]
        ])
        chunks = [
            padded[:, index * QUERY_BINS:
                   index * QUERY_BINS + HISTORY_BINS, :]
            for index in range(self.chunk_count)
        ]
        return tf.reshape(
            tf.stack(chunks, axis=1),
            (-1, HISTORY_BINS, tf.shape(values)[-1]),
        )

    def _start(
        self, index: int, batch: tf.Tensor,
        seed: Optional[tf.Tensor] = None,
    ) -> tf.Tensor:
        """Return one always-valid attention key per batch row."""
        token = self.start_tokens[index]
        if seed is not None:
            token = token + seed
        return tf.broadcast_to(token, (batch, 1, self.units))

    def _encode_observed(
        self, observed: tf.Tensor,
    ) -> tuple[tf.Tensor, ...]:
        """Encode observed neural, stimulus, and optional stage-one tokens."""
        observed = self._pad_time(observed)
        neural = observed[..., self.stage_one_dims:
                          self.stage_one_dims + self.neural_dims]
        target = observed[..., -STIMULUS_DIMS:]
        neural_valid = self._valid(neural)
        target_valid = self._valid(target)
        safe_neural = tf.where(
            neural_valid[..., None], neural, 0.0
        )
        safe_target = tf.where(
            target_valid[..., None], target, 0.0
        )
        positions = tf.range(self.padded_bins)
        position_token = self.position_encoder(positions)
        position_token += _time_embedding(positions, self.units)
        neural_content = self.neural_encoder.apply_func(safe_neural)
        query = neural_content + position_token + self.token_types[0]
        history = neural_content + position_token + self.token_types[2]
        stimulus = (
            self.stimulus_encoder(safe_target) + position_token
            + self.token_types[1]
        )
        if self.stage_one_encoder is not None:
            stage_one = observed[..., :self.stage_one_dims]
            stage_valid = self._valid(stage_one)
            safe_stage = tf.where(
                stage_valid[..., None], stage_one, 0.0
            )
            query += self.stage_one_encoder(safe_stage)
            query += self.token_types[3]
        return query, history, stimulus, neural_valid, target_valid

    def _observed_latents(
        self, observed: tf.Tensor, training: Optional[bool],
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Compute all current latents in parallel with causal masks."""
        query, history, stimulus, neural_valid, target_valid = (
            self._encode_observed(observed)
        )
        current = self._chunks(query)
        current_stimulus = self._chunks(stimulus)
        current_valid = self._chunks(
            neural_valid[..., None]
        )[..., 0]
        stimulus_valid = self._chunks(
            target_valid[..., None]
        )[..., 0]
        history_tokens = self._history(history)
        history_valid = self._history(
            tf.cast(neural_valid[..., None], tf.float32)
        )[..., 0] > 0
        batch = tf.shape(current)[0]
        one = tf.ones((batch, 1), dtype=tf.bool)
        history_memory = tf.concat((
            self._start(0, batch), history_tokens
        ), axis=1)
        history_mask = tf.concat((one, history_valid), axis=1)
        history_mask = history_mask[:, None, :]
        stimulus_memory = tf.concat((
            self._start(1, batch), current_stimulus
        ), axis=1)
        causal = tf.linalg.band_part(
            tf.ones((QUERY_BINS, QUERY_BINS), dtype=tf.bool),
            -1, 0,
        )
        stimulus_mask = tf.concat((
            tf.ones((QUERY_BINS, 1), dtype=tf.bool),
            causal,
        ), axis=1)
        stimulus_mask = tf.logical_and(
            stimulus_mask[None, :, :],
            tf.concat((one, stimulus_valid), axis=1)[:, None, :],
        )
        for block in self.history_blocks:
            current = block(
                current, history_memory, history_mask, training
            )
        for block in self.stimulus_blocks:
            current = block(
                current, stimulus_memory, stimulus_mask, training
            )
        causal_mask = tf.logical_and(
            tf.concat((
                tf.ones((QUERY_BINS, 1), dtype=tf.bool),
                causal,
            ), axis=1)[None, :, :],
            tf.concat((one, current_valid), axis=1)[:, None, :],
        )
        for block in self.causal_blocks:
            causal_memory = tf.concat((
                self._start(2, batch, self.initial_state), current
            ), axis=1)
            current = block(
                current, causal_memory, causal_mask, training
            )
        return current, history_memory, history_mask, current_valid

    def _forecast_inputs(
        self, observed: tf.Tensor, future: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Encode known coordinates and stage-one states at each horizon."""
        observed = self._pad_time(observed)
        future = self._pad_time(future)
        target_tokens = []
        stage_tokens = []
        validities = []
        positions = tf.range(self.padded_bins)
        for horizon in self.steps_ahead:
            source = observed if horizon == 1 else future[..., horizon - 2]
            target = source[..., -STIMULUS_DIMS:]
            valid = self._valid(target)
            safe = tf.where(valid[..., None], target, 0.0)
            target_token = self.stimulus_encoder(safe)
            target_token += _time_embedding(
                positions + horizon - 1, self.units
            )
            target_token += self.token_types[1]
            target_tokens.append(target_token)
            validities.append(valid)
            if self.stage_one_encoder is not None:
                stage_one = source[..., :self.stage_one_dims]
                stage_valid = self._valid(stage_one)
                safe_stage = tf.where(
                    stage_valid[..., None], stage_one, 0.0
                )
                stage_tokens.append(
                    self.stage_one_encoder(safe_stage)
                )
        tokens = tf.stack(target_tokens, axis=2)
        if stage_tokens:
            tokens += tf.stack(stage_tokens, axis=2)
        valid = tf.stack(validities, axis=2)
        return tokens, tf.stack(target_tokens, axis=2), valid

    def _forecast_latents(
        self, observed: tf.Tensor, future: tf.Tensor,
        current: tf.Tensor, history: tf.Tensor,
        history_mask: tf.Tensor, current_valid: tf.Tensor,
        training: Optional[bool],
    ) -> tf.Tensor:
        """Predict every horizon from direct, masked sequence queries."""
        source, stimulus, valid = self._forecast_inputs(
            observed, future
        )
        count = len(self.steps_ahead)
        source = tf.reshape(
            source, (-1, QUERY_BINS, count, self.units)
        )
        stimulus = tf.reshape(
            stimulus, (-1, QUERY_BINS, count, self.units)
        )
        valid = tf.reshape(valid, (-1, QUERY_BINS, count))
        horizon = tf.gather(
            self.horizon_embeddings,
            [value - 1 for value in self.steps_ahead],
        )
        query = self.forecast_encoder(current)[:, :, None, :]
        query += source + horizon[None, None, :, :]
        query = tf.reshape(
            query, (-1, QUERY_BINS * count, self.units)
        )
        stimulus = tf.reshape(
            stimulus, (-1, QUERY_BINS * count, self.units)
        )
        valid = tf.reshape(valid, (-1, QUERY_BINS * count))
        for block in self.history_blocks:
            query = block(query, history, history_mask, training)
        batch = tf.shape(query)[0]
        stimulus = tf.where(
            valid[..., None], stimulus,
            self.start_tokens[1][None, None, :],
        )
        for block in self.stimulus_blocks:
            flat_query = tf.reshape(
                query, (-1, 1, self.units)
            )
            flat_stimulus = tf.reshape(
                stimulus, (-1, 1, self.units)
            )
            flat_query = block(
                flat_query, flat_stimulus, None, training
            )
            query = tf.reshape(
                flat_query,
                (batch, QUERY_BINS * count, self.units),
            )
        source_positions = tf.repeat(tf.range(QUERY_BINS), count)
        key_positions = tf.range(QUERY_BINS)
        causal = key_positions[None, :] <= source_positions[:, None]
        causal_mask = tf.concat((
            tf.ones((QUERY_BINS * count, 1), dtype=tf.bool),
            causal,
        ), axis=1)
        causal_mask = tf.logical_and(
            causal_mask[None, :, :],
            tf.concat((
                tf.ones((batch, 1), dtype=tf.bool),
                current_valid,
            ), axis=1)[:, None, :],
        )
        memory = tf.concat((
            self._start(2, batch, self.initial_state), current
        ), axis=1)
        for block in self.causal_blocks:
            query = block(query, memory, causal_mask, training)
        latents = tf.reshape(
            query, (-1, self.padded_bins, count, self.units)
        )
        return latents[:, :self.sequence_bins]

    def _decode(
        self, latent: tf.Tensor,
        feedthrough: Optional[tf.Tensor],
        prior: Optional[tf.Tensor],
    ) -> tf.Tensor:
        """Apply BRAID's observation decoder to one horizon."""
        if feedthrough is None:
            decoder = self.Cfw or self.C
            decoder_input = latent
        else:
            decoder = self.C
            decoder_input = tf.concat((latent, feedthrough), axis=-1)
        if self.has_prior_pred and prior is not None:
            return decoder.apply_func([decoder_input, prior])
        return decoder.apply_func(decoder_input)

    def forward_sequence(
        self, inputs: Sequence[tf.Tensor],
        training: Optional[bool] = None,
    ) -> list[tf.Tensor]:
        """Return BRAID predictions and latents for all sequence bins."""
        observed = inputs[0]
        offset = 1
        feedthrough = None
        future = None
        prior = None
        if self.nft:
            feedthrough = inputs[offset]
            offset += 1
        if self.n1_in:
            future = inputs[offset]
            offset += 1
        if self.has_prior_pred:
            prior = inputs[offset]
        current, history, history_mask, current_valid = (
            self._observed_latents(observed, training)
        )
        if future is None:
            if max(self.steps_ahead) > 1:
                raise ValueError(
                    "Future target coordinates are required for "
                    "multi-horizon Transformer prediction."
                )
            future = tf.zeros((
                tf.shape(observed)[0], self.sequence_bins,
                self.n1_in, 1,
            ), dtype=observed.dtype)
        latents = self._forecast_latents(
            observed, future, current, history,
            history_mask, current_valid, training,
        )
        predictions = []
        states = []
        for index, horizon in enumerate(self.steps_ahead):
            latent = latents[:, :, index, :]
            if self.nft == 0:
                ft_now = None
            elif self.observable_U_in_Cfw:
                ft_now = feedthrough[..., index]
            else:
                ft_now = feedthrough if horizon == 1 else None
            prior_now = prior[..., index] if prior is not None else None
            predictions.append(self._decode(
                latent, ft_now, prior_now
            ))
            states.append(latent)
        return predictions + states + states

    def propagate_steps_ahead(
        self,
        state: tf.Tensor,
        latent: tf.Tensor,
        future: Optional[tf.Tensor],
        feedthrough: Optional[tf.Tensor],
        prior: Optional[tf.Tensor],
        training: Optional[bool] = None,
    ) -> list[tf.Tensor]:
        """Predict initial-window horizons from a seeded latent."""
        del state
        if isinstance(prior, np.ndarray) and prior.ndim == 0:
            prior = None
        batch = tf.shape(latent)[0]
        history = self._start(0, batch)
        history_mask = tf.ones((batch, 1, 1), dtype=tf.bool)
        causal_memory = self._start(
            2, batch, self.initial_state
        )
        latents = []
        predictions = []
        for index, horizon in enumerate(self.steps_ahead):
            if horizon == 1:
                current = latent
            else:
                if future is None:
                    raise ValueError(
                        f"Horizon {horizon} requires future inputs."
                    )
                source = future[..., horizon - 2]
                target = source[..., -STIMULUS_DIMS:]
                target_valid = self._valid(target)
                safe_target = tf.where(
                    target_valid[..., None], target, 0.0
                )
                stimulus = self.stimulus_encoder(safe_target)
                stimulus += _time_embedding(
                    tf.fill((batch,), horizon - 1), self.units
                )
                stimulus += self.token_types[1]
                query = self.forecast_encoder(latent)
                query += self.horizon_embeddings[horizon - 1]
                query += stimulus
                if self.stage_one_encoder is not None:
                    stage_one = source[..., :self.stage_one_dims]
                    stage_valid = self._valid(stage_one)
                    safe_stage = tf.where(
                        stage_valid[..., None], stage_one, 0.0
                    )
                    query += self.stage_one_encoder(safe_stage)
                query = query[:, None, :]
                for block in self.history_blocks:
                    query = block(
                        query, history, history_mask, training
                    )
                stimulus = tf.where(
                    target_valid[..., None], stimulus,
                    self.start_tokens[1][None, :],
                )[:, None, :]
                for block in self.stimulus_blocks:
                    query = block(query, stimulus, None, training)
                for block in self.causal_blocks:
                    query = block(
                        query, causal_memory, None, training
                    )
                current = query[:, 0, :]
            if self.nft == 0:
                ft_now = None
            elif self.observable_U_in_Cfw:
                ft_now = feedthrough[..., index]
            else:
                ft_now = feedthrough if horizon == 1 else None
            prior_now = prior[..., index] if prior is not None else None
            predictions.append(self._decode(
                current, ft_now, prior_now
            ))
            latents.append(current)
        return predictions + latents + latents

    def setTrainableParameters(
        self,
        base: Optional[bool] = None,
        fw: Optional[bool] = None,
        initial_state: Optional[bool] = None,
    ) -> None:
        """Set native BRAID trainable groups for one shared stack."""
        if base is not None and fw is not None and base != fw:
            raise ValueError(
                "Transformer uses shared observed and forecast weights; "
                "base and fw trainability must match."
            )
        setting = base if base is not None else fw
        if setting is not None:
            self.trainable = setting
        if initial_state is not None:
            self.initial_state._trainable = initial_state


@tf.keras.utils.register_keras_serializable(package="BRAID")
class CausalTransformerSequence(tf.keras.layers.Layer):
    """Expose the Transformer stack as one Keras sequence layer."""

    def __init__(
        self, cell: CausalTransformerCell,
        name: str = "causal_transformer_sequence", **kwargs,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.cell = cell
        self.stateful = False

    def get_config(self) -> dict:
        """Serialize the nested Transformer core with the sequence layer."""
        config = super().get_config()
        config["cell"] = tf.keras.utils.serialize_keras_object(
            self.cell
        )
        return config

    @classmethod
    def from_config(cls, config: dict) -> "CausalTransformerSequence":
        """Recreate a sequence layer from its serialized core."""
        config = dict(config)
        config["cell"] = tf.keras.utils.deserialize_keras_object(
            config["cell"]
        )
        return cls(**config)

    def call(
        self, inputs: Sequence[tf.Tensor],
        training: Optional[bool] = None,
    ) -> list[tf.Tensor]:
        """Return time-major BRAID outputs for one independent window."""
        return self.cell.forward_sequence(inputs, training=training)
