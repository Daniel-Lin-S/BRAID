"""Select shared split windows and publish preprocessing or fitted previews.

Inputs are cached session/fold arrays, ordered neural populations, and saved
checkpoints. Preprocessing outputs belong to the preview cache; fitted outputs
remain under their owning fit. Preview artifacts never affect fit identity.
"""

from pathlib import Path

import numpy as np

from .artifacts import artifact_path
from .contracts import FeatureSet, plugin

SPLIT_NAMES = ("train", "validation", "test")
TIME_TOLERANCE = 1e-8


def preview_windows(features: FeatureSet, settings: dict) -> list[np.ndarray]:
    """Select one seeded contiguous window per split, in split-name order.

    Parameters
    ----------
    features : FeatureSet
        Fold arrays t, segment and role, each shape (T,).
    settings : dict
        Seed, seconds, windows=3, and optional context_samples/window_ranges.

    Returns
    -------
    list of ndarray
        Three index arrays, each shape (seconds * sampling_rate,).
    """
    arrays = features.arrays
    if settings["windows"] != len(SPLIT_NAMES):
        raise ValueError(
            "Expected three preview windows: train/validation/test."
        )
    time = arrays["t"]
    delta = np.diff(time)
    interval = float(np.min(delta))
    count = int(round(settings["seconds"] / interval))
    if count < 1 or not np.isclose(count * interval, settings["seconds"]):
        raise ValueError(
            "Preview duration must contain whole sample intervals."
        )
    context = max(count, settings.get("context_samples", count))
    requested = settings.get("window_ranges")
    if requested is not None and len(requested) != len(SPLIT_NAMES):
        raise ValueError("Expected one explicit window range per split.")
    rng = np.random.default_rng(settings["seed"])
    windows = []
    for role, split in enumerate(SPLIT_NAMES):
        candidates = []
        for segment in np.unique(arrays["segment"]):
            indices = np.flatnonzero(
                (arrays["segment"] == segment) & (arrays["role"] == role)
            )
            if len(indices) < context:
                continue
            if not np.allclose(np.diff(time[indices]), interval):
                raise ValueError(f"Non-contiguous time grid in {split}.")
            candidates.extend(indices[: len(indices) - count + 1])
        if not candidates:
            raise ValueError(
                f"No {settings['seconds']}-second {split} preview supports "
                f"the required {context}-sample context."
            )
        if requested is None:
            start = int(rng.choice(candidates))
        else:
            left, right = requested[role]
            matches = [
                index
                for index in candidates
                if np.isclose(
                    time[index], left, rtol=0, atol=TIME_TOLERANCE
                )
            ]
            if len(matches) != 1 or not np.isclose(
                right - left, settings["seconds"]
            ):
                raise ValueError(f"Invalid explicit {split} preview range.")
            start = matches[0]
        windows.append(np.arange(start, start + count))
    return windows


def preprocessing_previews(
    session: FeatureSet,
    fold: FeatureSet,
    settings: dict,
    destination: Path,
    columns: np.ndarray,
    windows: list[np.ndarray],
    regenerate: bool = False,
) -> Path:
    """Publish preprocessing panels for one ordered neural population.

    Parameters
    ----------
    session, fold : FeatureSet
        Native session arrays and split-local processed arrays.
    settings : dict
        Rendering and excerpt settings without invocation selection fields.
    destination : Path
        Stable cache-owned rendering directory.
    columns : ndarray, shape (C,)
        Ordered neural population columns.
    windows : list of ndarray
        Canonical train, validation, and test indices, each shape (N,).
    regenerate : bool, optional
        Force replacement of a valid matching rendering, by default False.

    Returns
    -------
    Path
        Completed preprocessing preview directory.
    """
    from .preview_publication import publish_previews

    return publish_previews(
        session,
        fold,
        settings,
        destination,
        windows,
        columns,
        regenerate=regenerate,
    )


def _fitted_preview_columns(
    features: FeatureSet,
    fitted: list[dict],
) -> np.ndarray:
    """Map checkpoint-retained channel IDs to fold source columns."""
    if not fitted or any("channel_ids" not in values for values in fitted):
        raise ValueError(
            "Fitted previews require channel IDs for every window."
        )
    expected = np.asarray(fitted[0]["channel_ids"])
    if (
        expected.ndim != 1
        or not len(expected)
        or len(set(expected)) != len(expected)
    ):
        raise ValueError(
            "Fitted preview channel IDs must be nonempty and unique."
        )
    if any(
        not np.array_equal(values["channel_ids"], expected)
        for values in fitted[1:]
    ):
        raise ValueError(
            "Fitted preview windows use inconsistent channel IDs."
        )
    identifiers = features.arrays["ids"]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Fold preview channel IDs must be unique.")
    locations = {
        identifier: number for number, identifier in enumerate(identifiers)
    }
    missing = [
        identifier for identifier in expected if identifier not in locations
    ]
    if missing:
        raise ValueError(
            f"Fitted preview channels are absent from the fold: {missing}."
        )
    return np.asarray(
        [locations[identifier] for identifier in expected], dtype=int
    )


def fitted_previews(
    session: FeatureSet,
    fold: FeatureSet,
    settings: dict,
    run: Path,
    model_plugin: str,
    windows: list[np.ndarray],
    regenerate: bool = False,
) -> Path:
    """Infer saved checkpoint stages and publish fitted-only panels.

    Parameters
    ----------
    session, fold : FeatureSet
        Native session arrays and split-local processed arrays.
    settings : dict
        Rendering and excerpt settings without invocation selection fields.
    run : Path
        Completed fit that owns the destination and checkpoint.
    model_plugin : str
        Checkpoint preview plugin in ``module:function`` form.
    windows : list of ndarray
        Canonical train, validation, and test indices, each shape (N,).
    regenerate : bool, optional
        Force replacement of a valid matching rendering, by default False.

    Returns
    -------
    Path
        Completed fitted preview directory.
    """
    from .preview_publication import publish_previews

    fitted = plugin(
        model_plugin,
        source_run=run,
        features=fold,
        windows=windows,
    )
    columns = _fitted_preview_columns(fold, fitted)
    return publish_previews(
        session,
        fold,
        settings,
        run / "data_preview" / "fitted",
        windows,
        columns,
        fitted,
        artifact_path(run, "model.p"),
        regenerate=regenerate,
    )
