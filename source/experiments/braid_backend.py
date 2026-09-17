"""Adapt existing BRAID fitting to the experiment model protocol.

Input FeatureSet arrays are time-first; fitting uses dimension-first arrays.
Each checkpoint directory contains fit_indices.npz, fit_arguments.json,
component epoch artifacts, fitted-stage previews, and the native BRAID model.
Forecasts are returned as (horizon, time, output) arrays with explicit masks.
"""

import copy
import logging
import shutil
from pathlib import Path

import numpy as np

from BRAID.BRAIDModel import BRAIDModel
from BRAID.MainModel import shift_ms_to_1s_series
from BRAID.config import load_braid_fit_arguments
from BRAID.sequence import window_shift

from .cache import atomic_json, file_digest
from .contracts import FeatureSet
from .nhp import TRAIN, VALIDATION
from .previews import preview_windows
from .windows import window_indices

LOGGER = logging.getLogger(__name__)

MISSING_MARKER = -1000000.0


class BRAIDBackend:
    """Preserve BRAID training internals behind a reusable experiment API."""

    def __init__(
        self,
        configuration: str,
        overrides: dict | None = None,
        seed: int = 42,
        previews: dict | None = None,
    ) -> None:
        self.seed = seed
        self.previews = previews
        self.arguments = load_braid_fit_arguments(configuration)
        self.arguments["args_base"].update(overrides or {})
        self.length = self.arguments["args_base"]["sequence_length"]
        self.model = None

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
        arguments = copy.deepcopy(self.arguments)
        arguments.update(dimensions)
        base = arguments["args_base"]
        training = window_indices(features, TRAIN, self.length)
        validation = window_indices(features, VALIDATION, self.length)
        batch = min(base["batch_size"], len(training), len(validation))
        training = training[: len(training) // batch * batch]
        validation = validation[: len(validation) // batch * batch]
        configured_batch = base["batch_size"]
        if batch != configured_batch:
            LOGGER.warning(
                "Effective batch size %s (configured %s): limited by the "
                "number of complete training/validation windows.",
                batch,
                configured_batch,
            )
        base["batch_size"] = batch
        atomic_json(
            directory / "training_deviations.json",
            dict(
                configured_batch_size=configured_batch,
                effective_batch_size=batch,
                reason="bounded by available complete windows",
            ),
        )
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            directory / "fit_indices.npz",
            training=features.arrays["indices"][training],
            validation=features.arrays["indices"][validation],
        )
        atomic_json(directory / "fit_arguments.json", arguments)
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
            for number, indices in enumerate(
                preview_windows(features, self.previews)
            ):
                self._previews(features, columns, indices, directory, number)

    def _previews(
        self,
        features: FeatureSet,
        columns: np.ndarray,
        indices: np.ndarray,
        directory: Path,
        number: int,
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
        suffix = "" if number == 0 else f"_{number}"
        stem = directory / f"fitted_preprocessing{suffix}"
        np.savez_compressed(
            stem.with_suffix(".npz"),
            t=arrays["t"][indices],
            channel_ids=arrays["ids"][columns],
            source_indices=arrays["indices"][indices],
            context_indices=arrays["indices"][context],
            learned_Z=learned,
            raw_Z=z,
            **normalized,
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
        if (
            getattr(self, "feature_cache", None) is not None
            and self.previews
            and self.previews["enabled"]
        ):
            target = self.feature_cache / "fitted_previews" / file_digest(path)
            target.mkdir(parents=True, exist_ok=True)
            sources = list(path.parent.glob("fitted_preprocessing*"))
            sources.append(path.parent / "fit_indices.npz")
            for source in sources:
                shutil.copy2(source, target / source.name)
            atomic_json(
                target / "provenance.json",
                {
                    "checkpoint": str(path.resolve()),
                    "sha256": file_digest(path),
                    "scope": "this fitted checkpoint only",
                },
            )

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
    source_run: Path, features: FeatureSet, windows: list[np.ndarray]
) -> list[dict[str, np.ndarray]]:
    """Read saved excerpts and apply existing BRAID normalization maps.

    Parameters
    ----------
    source_run : Path
        Trusted run with model.p, selection.npz and fitted excerpt NPZ files.
    features : FeatureSet
        Validated cached fold arrays in time-first orientation.
    windows : list of ndarray, each shape (T,)
        Exact fold indices selected for each saved preview window.

    Returns
    -------
    list of dict of ndarray
        Pre/main normalized Y, Z, U and saved learned Z for each window.
        Y columns follow channel_ids; no model is reconstructed or fitted.
    """
    from BRAID.tools.file_tools import pickle_load

    model = pickle_load(str(source_run / "model.p"))["model"]
    with np.load(source_run / "selection.npz", allow_pickle=False) as saved:
        columns = saved["selected_columns"]
        ids = saved["channel_ids"]
        units = saved["unit_dimensions"]
    np.testing.assert_array_equal(features.arrays["ids"][columns], ids)
    np.testing.assert_array_equal(features.arrays["units"][columns], units)
    excerpts = []
    for path in sorted(source_run.glob("fitted_preprocessing*.npz")):
        with np.load(path, allow_pickle=False) as saved:
            excerpts.append(dict(saved))
    output = []
    for indices in windows:
        time = features.arrays["t"][indices]
        matches = [
            entry
            for entry in excerpts
            if entry["t"].shape == time.shape
            and np.allclose(entry["t"], time, rtol=0, atol=1e-8)
        ]
        if len(matches) != 1:
            raise ValueError(
                "Expected exactly one saved fitted excerpt "
                f"for window starting at {time[0]}."
            )
        saved = matches[0]
        np.testing.assert_array_equal(saved["channel_ids"], ids)
        np.testing.assert_array_equal(
            saved["source_indices"], features.arrays["indices"][indices]
        )
        raw = dict(
            Y=features.arrays["Y"][indices][:, columns],
            Z=features.arrays["Z"][indices],
            U=features.arrays["U"][indices],
        )
        np.testing.assert_allclose(raw["Z"], saved["raw_Z"], rtol=0, atol=1e-8)
        result = dict(channel_ids=ids, learned_Z=saved["learned_Z"])
        for component, fitted in (("pre", model.sId_pre), ("main", model.sId)):
            for name in ("Y", "Z", "U"):
                values = (
                    saved["learned_Z"]
                    if component == "main" and name == "Z"
                    else raw[name]
                )
                mapping = getattr(fitted, f"{name}PrepMap")
                normalized = mapping.apply(values.T.copy()).T
                if normalized.shape != values.shape:
                    raise ValueError(
                        f"{component} {name} normalization changed dimensions: "
                        f"expected {values.shape}, got {normalized.shape}."
                    )
                if not np.isfinite(normalized).all():
                    raise ValueError(f"Nonfinite {component} {name} preview.")
                result[f"{component}_{name}"] = normalized
                if component == "pre":
                    np.testing.assert_allclose(
                        normalized, saved[name], rtol=1e-7, atol=1e-8
                    )
        if not np.isfinite(result["learned_Z"]).all():
            raise ValueError(
                "Saved learned behavior contains nonfinite values."
            )
        output.append(result)
    return output
