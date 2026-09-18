"""Plot cached preprocessing stages on shared real-data time windows.

PNG panels and numeric excerpts are published under analysis previews.
Preview settings do not contribute to the numerical feature identity.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .artifacts import artifact_path
from .cache import file_digest, fingerprint
from .contracts import FeatureSet, plugin


def trace_plot(
    path: Path,
    time: np.ndarray,
    panels: dict[str, np.ndarray],
    title: str,
    xlabel: str = "Time (s)",
) -> None:
    """Save stacked time-series panels with matching time axes."""
    if not len(time) or not panels:
        raise ValueError("Cannot plot empty preprocessing excerpts.")
    figure, axes = plt.subplots(
        len(panels),
        1,
        sharex=True,
        figsize=(10, 2.2 * len(panels)),
        squeeze=False,
    )
    for axis, (label, values) in zip(axes[:, 0], panels.items()):
        axis.plot(time, values, linewidth=0.8)
        axis.set_ylabel(label)
    axes[-1, 0].set_xlabel(xlabel)
    figure.suptitle(title)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=130)
    plt.close(figure)


def preview_windows(features: FeatureSet, settings: dict) -> list[np.ndarray]:
    """Select deterministic excerpts confined to valid segment interiors.

    Parameters
    ----------
    features : FeatureSet
        Fold arrays including t and segment, each shape (T,).
    settings : dict
        Seed, window count, and duration in seconds.

    Returns
    -------
    list of ndarray
        Per-window sample indices, each shape (duration * sample_rate,).
    """
    arrays = features.arrays
    rng = np.random.default_rng(settings["seed"])
    duration = settings["seconds"]
    candidates = []
    for segment in np.unique(arrays["segment"]):
        indices = np.flatnonzero(arrays["segment"] == segment)
        time = arrays["t"][indices]
        safe = indices[(time > time[0]) & (time < time[-1] - duration)]
        if len(safe):
            candidates.append(safe)
    if not candidates:
        raise ValueError("No segment has a valid preview interior.")
    windows = []
    for number in range(settings["windows"]):
        start = arrays["t"][rng.choice(candidates[number % len(candidates)])]
        windows.append(
            np.flatnonzero(
                (arrays["t"] >= start) & (arrays["t"] < start + duration)
            )
        )
    return windows


def preprocessing_previews(
    session: FeatureSet, fold: FeatureSet, settings: dict, fallback: Path
) -> Path | None:
    """Persist raw and transformed excerpts for deterministic valid windows."""
    if not settings["enabled"]:
        return
    base = fallback / fingerprint(
        dict(stage="preprocess", source=fold.metadata, previews=settings)
    )
    from .preview_publication import publish_previews

    return publish_previews(
        session,
        fold,
        settings,
        base / "previews",
        preview_windows(fold, settings),
    )


def fitted_previews(
    session: FeatureSet,
    fold: FeatureSet,
    settings: dict,
    run: Path,
    model_plugin: str,
    columns: np.ndarray,
    destination: Path,
) -> Path | None:
    """Publish checkpoint-derived previews inside the analysis.

    Parameters
    ----------
    session, fold : FeatureSet
        Native and split-dependent cached arrays, time-first.
    settings : dict
        Enabled flag, window selection and presentation settings.
    run : Path
        Saved model and fitted excerpts; used as fallback without caching.
    model_plugin : str
        Adapter function extracting fitted arrays without refitting.
    columns : ndarray, shape (C,)
        Neural columns belonging to this fitted model's population.
    destination : Path
        Analysis-owned preview destination.

    Returns
    -------
    Path or None
        Absolute immutable revision directory, or None when disabled.
    """
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
    checkpoint = artifact_path(run, "model.p")
    base = destination
    return publish_previews(
        session,
        fold,
        settings,
        base / "fitted_previews" / file_digest(checkpoint),
        windows,
        columns,
        fitted,
        checkpoint,
    )
