"""Select split-local excerpts and publish fit-owned data previews.

Inputs are cached session/fold arrays and shared presentation settings.
Outputs belong to data_preview/preprocessing or data_preview/fitted.
Selection and rendering never contribute to numerical fitting identity.
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
                i
                for i in candidates
                if np.isclose(time[i], left, rtol=0, atol=TIME_TOLERANCE)
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
    run: Path,
    columns: np.ndarray,
) -> Path | None:
    """Publish preprocessing panels for the owning fit's ordered population."""
    if not settings["enabled"]:
        return None
    from .preview_publication import publish_previews

    return publish_previews(
        session,
        fold,
        settings,
        run / "data_preview" / "preprocessing",
        preview_windows(fold, settings),
        columns,
    )


def fitted_previews(
    session: FeatureSet,
    fold: FeatureSet,
    settings: dict,
    run: Path,
    model_plugin: str,
    columns: np.ndarray,
) -> Path | None:
    """Infer checkpoint-derived stages and publish fitted-only panels."""
    if not settings["enabled"]:
        return None
    from .preview_publication import publish_previews

    windows = preview_windows(fold, settings)
    fitted = plugin(
        model_plugin,
        source_run=run,
        features=fold,
        windows=windows,
    )
    return publish_previews(
        session,
        fold,
        settings,
        run / "data_preview" / "fitted",
        windows,
        columns,
        fitted,
        artifact_path(run, "model.p"),
    )
