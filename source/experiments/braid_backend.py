"""Adapt existing BRAID fitting to the experiment model protocol.

Input FeatureSet arrays are time-first; fitting uses dimension-first arrays.
Each checkpoint directory contains fit_indices.npz, fit_arguments.json,
component epoch artifacts, fitted-stage previews, and the native BRAID model.
Forecasts are returned as (horizon, time, output) arrays with explicit masks.
"""

import logging
import json
from pathlib import Path

import numpy as np
import yaml

from BRAID.BRAIDModel import BRAIDModel
from BRAID.MainModel import shift_ms_to_1s_series
from BRAID.config import resolve_braid_fit_arguments
from BRAID.sequence import window_shift
from BRAID.tools.tensorboard import event_scope

from .artifacts import artifact_path
from .cache import atomic_json, file_digest, cached
from .contracts import FeatureSet
from .nhp import TRAIN, VALIDATION
from .previews import preview_windows
from .windows import window_indices

LOGGER = logging.getLogger(__name__)

MISSING_MARKER = -1000000.0


class BRAIDBackend:
    """Preserve BRAID training internals behind a reusable experiment API."""

    model_name = "BRAID"

    def __init__(
        self,
        configuration: str,
        overrides: dict | None = None,
        seed: int = 42,
        previews: dict | None = None,
    ) -> None:
        self.seed = seed
        self.configuration = yaml.safe_load(Path(configuration).read_text())
        self.overrides = overrides or {}
        self.previews = previews
        self.arguments = self.resolve_fit_configuration(
            self.configuration, {}, self.overrides
        )
        self.length = self.arguments["args_base"]["sequence_length"]
        self.model = None

    @staticmethod
    def resolve_fit_configuration(
        configuration: dict,
        dimensions: dict,
        overrides: dict | None = None,
        features: FeatureSet | None = None,
    ) -> dict:
        """Resolve defaults, case overrides and data-dependent batch size.

        Features, when supplied, contain time-first arrays and split roles.
        The returned mapping is passed unchanged to BRAIDModel.fit.
        """
        arguments = resolve_braid_fit_arguments(configuration)
        arguments.update(dimensions)
        base = arguments["args_base"]
        base.update(overrides or {})
        if features is not None:
            length = base["sequence_length"]
            counts = [
                len(window_indices(features, role, length))
                for role in (TRAIN, VALIDATION)
            ]
            if min(counts) < 1:
                raise ValueError(
                    "Fitting requires training and validation windows."
                )
            base["batch_size"] = min(base["batch_size"], *counts)
        return arguments

    @event_scope()
    def fit(
        self,
        features: FeatureSet,
        columns: np.ndarray,
        dimensions: dict,
        directory: Path,
    ) -> None:
        """Fit BRAID using isolated training and validation windows.

        Parameters
        ----------
        features : FeatureSet
            Time-first Y (T, C), Z (T, D), U (T, 2) and segment indices.
        columns : ndarray, shape (K,)
            Ordered neural input/output columns for this population.
        dimensions : dict
            BRAID latent sizes, including nx, n1 and n_pre.
        directory : Path
            Destination for checkpoint-specific fitting artifacts.
        """
        import tensorflow as tf

        tf.keras.utils.set_random_seed(self.seed)
        self.feature_cache = features.path
        self.run_directory = directory
        self.preview_excerpts = []
        arguments = self.resolve_fit_configuration(
            self.configuration, dimensions, self.overrides, features
        )
        base = arguments["args_base"]
        training = window_indices(features, TRAIN, self.length)
        validation = window_indices(features, VALIDATION, self.length)
        batch = base["batch_size"]
        training = training[: len(training) // batch * batch]
        validation = validation[: len(validation) // batch * batch]
        configured_batch = self.arguments["args_base"]["batch_size"]
        if batch != configured_batch:
            LOGGER.warning(
                "Effective batch size %s (configured %s): limited by the "
                "number of complete training/validation windows.",
                batch,
                configured_batch,
            )
        atomic_json(
            artifact_path(directory, "training_deviations.json"),
            dict(
                configured_batch_size=configured_batch,
                effective_batch_size=batch,
                reason="bounded by available complete windows",
            ),
        )
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            artifact_path(directory, "fit_indices.npz"),
            training=features.arrays["indices"][training],
            validation=features.arrays["indices"][validation],
        )
        atomic_json(artifact_path(directory, "fit_arguments.json"), arguments)
        arrays = features.arrays
        self.model = BRAIDModel(
            log_dir=str(directory / "components"),
            missing_marker=MISSING_MARKER,
        )
        self.model.fit(
            arrays["Y"][training.ravel()][:, columns].T,
            arrays["Z"][training.ravel()].T,
            U=arrays["U"][training.ravel()].T,
            Y_validation=arrays["Y"][validation.ravel()][:, columns].T,
            Z_validation=arrays["Z"][validation.ravel()].T,
            U_validation=arrays["U"][validation.ravel()].T,
            YType="cont",
            ZType="cont",
            UType="cont",
            **arguments,
        )
        if self.previews and self.previews["enabled"]:
            for indices in preview_windows(features, self.previews):
                self._previews(features, columns, indices)

    def _previews(
        self,
        features: FeatureSet,
        columns: np.ndarray,
        indices: np.ndarray,
    ) -> None:
        """Save checkpoint-specific transforms on the raw-preview time grid."""
        arrays = features.arrays
        segment = np.flatnonzero(
            arrays["segment"] == arrays["segment"][indices[0]]
        )
        start = min(indices[0], segment[-1] + 1 - self.length)
        if start < segment[0] or indices[-1] >= start + self.length:
            raise ValueError("Preview must fit inside one model window.")
        context = np.arange(start, start + self.length)
        y, z, u = (
            arrays["Y"][indices][:, columns],
            arrays["Z"][indices],
            arrays["U"][indices],
        )
        pre = self.model.sId_pre
        normalized = dict(
            Y=pre.YPrepMap.apply(y.T.copy()).T,
            Z=pre.ZPrepMap.apply(z.T.copy()).T,
            U=pre.UPrepMap.apply(u.T.copy()).T,
        )
        learned = pre.predict(
            arrays["Y"][context][:, columns], arrays["U"][context]
        )[0][indices - start]
        for name, value in dict(normalized, learned_Z=learned).items():
            if not np.isfinite(value).all():
                raise FloatingPointError(f"Nonfinite fitted preview: {name}.")
        self.preview_excerpts.append(
            dict(
                t=arrays["t"][indices],
                channel_ids=arrays["ids"][columns],
                source_indices=arrays["indices"][indices],
                context_indices=arrays["indices"][context],
                learned_Z=learned,
                raw_Z=z,
                **normalized,
            )
        )

    def predict(
        self, y: np.ndarray, u: np.ndarray, horizons: list[int]
    ) -> dict[str, np.ndarray]:
        """Forecast complete independent windows in original output units."""
        if self.model is None:
            raise RuntimeError("Fit or load a model before prediction.")
        self.model.set_steps_ahead(horizons)
        outputs = self.model.predict(y, u)
        count = len(horizons)
        result = {}
        for name, start in (("Z", 0), ("Y", count), ("X", 2 * count)):
            result[name] = np.stack(
                window_shift(
                    shift_ms_to_1s_series,
                    outputs[start : start + count],
                    horizons,
                    self.length,
                    MISSING_MARKER,
                    time_first=True,
                )
            )
        positions = np.arange(len(y)) % self.length
        result["valid"] = np.array([positions >= h for h in horizons])
        for name in ("Y", "Z", "X"):
            if not np.isfinite(result[name][result["valid"]]).all():
                raise FloatingPointError(f"Nonfinite {name} forecasts.")
        return result

    def save(self, path: Path) -> None:
        """Save all BRAID stages using its native reconstruction format."""
        self.model.saveToFile(str(path))
        self.model.restoreModels()
        if self.preview_excerpts:
            base = self.feature_cache or (self.run_directory / "data")
            arrays = {
                f"window_{i}_{key}": value
                for i, excerpt in enumerate(self.preview_excerpts)
                for key, value in excerpt.items()
            }
            identity = dict(checkpoint_sha256=file_digest(path), version=1)
            entry = cached(
                base / "fitted_previews" / identity["checkpoint_sha256"],
                "excerpts",
                identity,
                "reuse",
                lambda: FeatureSet(
                    arrays, {"windows": len(self.preview_excerpts)}
                ),
            )
            atomic_json(
                artifact_path(self.run_directory, "fitted_excerpts.json"),
                dict(
                    directory=str(entry.path),
                    identity=json.loads(
                        (entry.path / "manifest.json").read_text()
                    )["identity"],
                    manifest_sha256=file_digest(entry.path / "manifest.json"),
                ),
            )
            self.preview_excerpts = []

    def load(self, path: Path) -> None:
        """Restore all BRAID stages without fitting or changing parameters."""
        self.model = BRAIDModel.loadFromFile(str(path))


def build_cases(settings: dict) -> list[dict]:
    """Build one manifest's configured latent/population Cartesian product.

    Parameters
    ----------
    settings : dict
        nx_values, n1_max, n_pre and population_scales for one experiment.

    Returns
    -------
    list of dict
        Unique named cases with dimensions and population selections.
    """
    sizes = settings["nx_values"]
    populations = settings["population_scales"]
    if not sizes or not populations or min(sizes) < 1:
        raise ValueError("Expected nonempty positive latent/population grids.")
    if len(set(sizes)) != len(sizes) or len(set(populations)) != len(
        populations
    ):
        raise ValueError("Sweep values must be unique.")
    if any(not 0 < value <= 1 for value in populations):
        raise ValueError("Population scales must be in (0, 1].")
    cases = []
    for population in populations:
        for nx in sizes:
            n1 = min(settings["n1_max"], nx)
            cases.append(
                dict(
                    name=f"BRAID_nx{nx}_p{population:g}",
                    population_scale=population,
                    dimensions=dict(
                        nx=nx, n1=n1, n3=None, n_pre=settings["n_pre"]
                    ),
                    summary_parameters=dict(nx=nx, n1=n1),
                )
            )
    return cases


def checkpoint_preview_arrays(
    source_run: Path, features: FeatureSet, windows: list[np.ndarray],
) -> list[dict[str, np.ndarray]]:
    """Infer fitted stages from a completed checkpoint without any fitting.

    Parameters
    ----------
    source_run : Path
        Completed fit containing checkpoint, arguments and selected channels.
    features : FeatureSet
        Cached fold arrays with shapes (T, C), (T, D), and split metadata.
    windows : list of ndarray
        Split-local indices, each shape (window_samples,).

    Returns
    -------
    list of dict of ndarray
        Learned Z and pre/main normalized Y/Z/U for each requested window.
    """
    from .artifacts import validate_completion

    validate_completion(source_run)
    model = BRAIDModel.loadFromFile(str(artifact_path(source_run, "model.p")))
    arguments = json.loads(
        artifact_path(source_run, "fit_arguments.json").read_text()
    )
    length = arguments["args_base"]["sequence_length"]
    with np.load(
        artifact_path(source_run, "selection.npz"), allow_pickle=False,
    ) as saved:
        columns = saved["selected_columns"]
        ids = saved["channel_ids"]
        units = saved["unit_dimensions"]
    arrays = features.arrays
    np.testing.assert_array_equal(arrays["ids"][columns], ids)
    np.testing.assert_array_equal(arrays["units"][columns], units)
    output = []
    for indices in windows:
        segment = np.flatnonzero(
            arrays["segment"] == arrays["segment"][indices[0]]
        )
        start = min(indices[0], segment[-1] + 1 - length)
        if (
            start < segment[0] or indices[-1] >= start + length
            or np.unique(arrays["segment"][indices]).size != 1
        ):
            raise ValueError("Preview must fit inside one model context.")
        context = np.arange(start, start + length)
        learned = model.sId_pre.predict(
            arrays["Y"][context][:, columns], arrays["U"][context],
        )[0][indices - start]
        raw = dict(
            Y=arrays["Y"][indices][:, columns],
            Z=arrays["Z"][indices], U=arrays["U"][indices],
        )
        result = dict(channel_ids=ids, learned_Z=learned)
        for component, fitted in (("pre", model.sId_pre), ("main", model.sId)):
            for name in ("Y", "Z", "U"):
                values = (
                    learned if component == "main" and name == "Z"
                    else raw[name]
                )
                normalized = getattr(fitted, f"{name}PrepMap").apply(
                    values.T.copy()
                ).T
                if normalized.shape != values.shape:
                    raise ValueError(
                        f"Expected {component} {name} shape {values.shape}, "
                        f"got {normalized.shape}."
                    )
                result[f"{component}_{name}"] = normalized
        for name, values in result.items():
            if name != "channel_ids" and not np.isfinite(values).all():
                raise ValueError(f"Nonfinite fitted preview values: {name}.")
        output.append(result)
    return output
