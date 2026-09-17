"""Shared experiment interfaces.

Datasets return time-first arrays Y (T, ny), Z (T, nz), U (T, nu), and t
(T,). Adapters own model-specific orientation, state, and serialization.
Plugins are configured as module:class references and loaded lazily.
"""

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Protocol

import numpy as np


@dataclass
class FeatureSet:
    """Validated arrays and provenance from one cached transformation."""

    arrays: dict[str, np.ndarray]
    metadata: dict[str, Any]
    path: Path | None = None


class Dataset(Protocol):
    """Contract required by the model-independent experiment runner."""

    def sessions(self) -> list[str]: ...
    def inventory(self) -> dict[str, list[str]]: ...
    def load(self, session: str) -> FeatureSet: ...
    def fold(
        self, features: FeatureSet, fold: int, velocity: bool
    ) -> FeatureSet: ...


class Model(Protocol):
    """Contract for fitting and evaluating an experiment backend."""

    length: int

    def fit(
        self,
        features: FeatureSet,
        columns: np.ndarray,
        dimensions: dict,
        directory: Path,
    ) -> None: ...
    def predict(
        self, y: np.ndarray, u: np.ndarray, horizons: list[int]
    ) -> dict[str, np.ndarray]: ...
    def save(self, path: Path) -> None: ...
    def load(self, path: Path) -> None: ...


def plugin(reference: str, **kwargs: Any) -> Any:
    """Construct a configured plugin without a dataset/model dispatch chain.

    Parameters
    ----------
    reference : str
        Importable ``module:class`` implementation.
    kwargs : mapping
        Configuration arguments passed to the implementation.

    Returns
    -------
    object
        Constructed implementation.
    """
    module, separator, name = reference.partition(":")
    if not separator:
        raise ValueError("Plugin reference must have the form module:class.")
    return getattr(import_module(module), name)(**kwargs)
