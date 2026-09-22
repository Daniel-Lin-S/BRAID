"""Adapt existing BRAID fitting to the experiment model protocol.

Input FeatureSet arrays are time-first; fitting uses dimension-first arrays.
Each checkpoint directory contains fit_indices.npz, fit_arguments.json,
component epoch artifacts and the native BRAID model.
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
from .cache import atomic_json
from .contracts import FeatureSet
from .nhp import TRAIN, VALIDATION
from .windows import window_indices

LOGGER = logging.getLogger(__name__)

MISSING_MARKER = -1000000.0
LATENT_SPLIT_FIELDS = ("n1", "n2", "n3")
ZERO_STAGE_THREE = 0
CANONICAL_ZERO_STAGE_THREE = None
CASE_NAME_PREFIX = "BRAID"


class BRAIDBackend:
    """Preserve BRAID training internals behind a reusable experiment API."""

    model_name = "BRAID"

    def __init__(
        self,
        configuration: str,
        overrides: dict | None = None,
        seed: int = 42,
    ) -> None:
        self.seed = seed
        self.configuration = yaml.safe_load(Path(configuration).read_text())
        self.overrides = overrides or {}
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

    def load(self, path: Path) -> None:
        """Restore all BRAID stages without fitting or changing parameters."""
        self.model = BRAIDModel.loadFromFile(str(path))


def build_cases(settings: dict) -> list[dict]:
    """Build one manifest's configured latent/population Cartesian product.

    Parameters
    ----------
    settings : dict
        Legacy settings use nx_values and n1_max. Explicit settings use
        latent_splits with n1, n2 and n3. Both use n_pre and
        population_scales.

    Returns
    -------
    list of dict
        Unique named cases with dimensions and population selections.
    """
    populations = settings["population_scales"]
    if not populations:
        raise ValueError("Expected a nonempty population grid.")
    if len(set(populations)) != len(populations):
        raise ValueError("Population scales must be unique.")
    if any(not 0 < value <= 1 for value in populations):
        raise ValueError("Population scales must be in (0, 1].")
    n_pre = _positive_integer(settings.get("n_pre"), "n_pre")
    splits = settings.get("latent_splits")
    if splits is not None:
        if (
            settings.get("nx_values") is not None
            or settings.get("n1_max") is not None
        ):
            raise ValueError(
                "Explicit latent_splits cannot be combined with nx_values "
                "or n1_max."
            )
        dimensions = _explicit_latent_splits(splits)
    else:
        dimensions = _legacy_latent_splits(settings)
    cases = []
    for population in populations:
        for nx, n1, n2 in dimensions:
            cases.append(
                dict(
                    name=_case_name(
                        nx, n1, n2, population, splits is not None
                    ),
                    population_scale=population,
                    dimensions=dict(
                        nx=nx,
                        n1=n1,
                        n3=CANONICAL_ZERO_STAGE_THREE,
                        n_pre=n_pre,
                    ),
                    summary_parameters=(
                        dict(nx=nx, n1=n1, n2=n2, residual=n2 > 0)
                        if splits is not None
                        else dict(nx=nx, n1=n1)
                    ),
                )
            )
    return cases


def _positive_integer(value: object, name: str) -> int:
    """Return one strictly positive integer configuration value."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"Expected positive integer {name}, got {value!r}.")
    return value


def _nonnegative_integer(value: object, name: str) -> int:
    """Return one nonnegative integer configuration value."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            f"Expected nonnegative integer {name}, got {value!r}."
        )
    return value


def _legacy_latent_splits(settings: dict) -> list[tuple[int, int, int]]:
    """Resolve the established one-dimensional latent sweep definition."""
    sizes = settings.get("nx_values")
    if not isinstance(sizes, list) or not sizes:
        raise ValueError("Expected a nonempty nx_values list.")
    if len(set(sizes)) != len(sizes):
        raise ValueError("Latent dimensions must be unique.")
    n1_max = _positive_integer(settings.get("n1_max"), "n1_max")
    result = []
    for value in sizes:
        nx = _positive_integer(value, "nx_values entry")
        n1 = min(n1_max, nx)
        result.append((nx, n1, nx - n1))
    return result


def _explicit_latent_splits(
    settings: object,
) -> list[tuple[int, int, int]]:
    """Validate explicit dimensions and derive total state size."""
    if not isinstance(settings, list) or not settings:
        raise ValueError("Expected a nonempty latent_splits list.")
    result = []
    seen = set()
    for number, split in enumerate(settings):
        location = f"latent_splits[{number}]"
        if (
            not isinstance(split, dict)
            or set(split) != set(LATENT_SPLIT_FIELDS)
        ):
            received = sorted(split) if isinstance(split, dict) else split
            raise ValueError(
                f"Expected {location} keys {LATENT_SPLIT_FIELDS}, got "
                f"{received!r}."
            )
        n1 = _positive_integer(split["n1"], f"{location}.n1")
        n2 = _nonnegative_integer(split["n2"], f"{location}.n2")
        n3 = _nonnegative_integer(split["n3"], f"{location}.n3")
        if n3 != ZERO_STAGE_THREE:
            raise ValueError(
                f"Expected {location}.n3={ZERO_STAGE_THREE}, got {n3}."
            )
        dimensions = (n1 + n2 + n3, n1, n2)
        if dimensions in seen:
            raise ValueError(f"Duplicate latent split at {location}: {split}.")
        seen.add(dimensions)
        result.append(dimensions)
    return result


def _case_name(
    nx: int,
    n1: int,
    n2: int,
    population: float,
    explicit: bool,
) -> str:
    """Return a readable membership name without affecting fit identity."""
    if not explicit:
        return f"{CASE_NAME_PREFIX}_nx{nx}_p{population:g}"
    return f"{CASE_NAME_PREFIX}_nx{nx}_n1{n1}_n2{n2}_p{population:g}"


def _retained_neural_indices(
    mapping: object,
    channel_count: int,
    component: str,
) -> np.ndarray:
    """Return source neural dimensions retained by a saved linear mapping."""
    weight = mapping.get_overall_W()
    if weight is None:
        return np.arange(channel_count)
    weight = np.asarray(weight)
    if weight.ndim != 2 or weight.shape[1] != channel_count:
        raise ValueError(
            f"Expected {component} neural preprocessing weight shape "
            f"(*, {channel_count}), got {weight.shape}."
        )
    if weight.shape[0] == channel_count:
        return np.arange(channel_count)
    supports = [np.flatnonzero(row) for row in weight]
    if (
        not supports
        or any(len(indices) != 1 for indices in supports)
        or len({int(indices[0]) for indices in supports}) != len(supports)
    ):
        raise ValueError(
            f"{component} neural preprocessing does not retain a direct "
            "channel mapping for fitted previews."
        )
    return np.asarray([indices[0] for indices in supports], dtype=int)


def _fitted_neural_indices(
    model: BRAIDModel,
    channel_count: int,
) -> np.ndarray:
    """Validate the shared channel projection used by fitted stages."""
    retained = [
        _retained_neural_indices(
            component.YPrepMap,
            channel_count,
            name,
        )
        for name, component in (
            ("pre", model.sId_pre),
            ("main", model.sId),
        )
    ]
    if not np.array_equal(*retained):
        raise ValueError(
            "Pre and main neural preprocessing retain different channels."
        )
    return retained[0]


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
    retained = _fitted_neural_indices(model, len(columns))
    preview_ids = ids[retained]
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
        result = dict(channel_ids=preview_ids, learned_Z=learned)
        for component, fitted in (("pre", model.sId_pre), ("main", model.sId)):
            for name in ("Y", "Z", "U"):
                values = (
                    learned if component == "main" and name == "Z"
                    else raw[name]
                )
                normalized = getattr(fitted, f"{name}PrepMap").apply(
                    values.T.copy()
                ).T
                expected = (
                    (len(indices), len(retained))
                    if name == "Y"
                    else values.shape
                )
                if normalized.shape != expected:
                    raise ValueError(
                        f"Expected {component} {name} shape {expected}, "
                        f"got {normalized.shape}."
                    )
                result[f"{component}_{name}"] = normalized
        for name, values in result.items():
            if name != "channel_ids" and not np.isfinite(values).all():
                raise ValueError(f"Nonfinite fitted preview values: {name}.")
        output.append(result)
    return output
