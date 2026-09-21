"""Build signal-separated data previews from cached arrays and fitted stages.

Each split contains descriptively named PNGs and excerpts.npz.
Numerical excerpts retain native and aligned timestamps, channel/unit IDs,
source indices, spike events, and optional pre/main normalized arrays.
The renderer is shared by preprocessing, completed fits and offline review.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np

from .contracts import FeatureSet
from .preview_rendering import PreviewStage, save_signal

TIME_TOLERANCE = 1e-8
COORDINATES = ("x", "y")
NORMALIZED_UNIT = "Standardized\n(unitless)"


def preview_columns(
    features: FeatureSet,
    settings: dict,
    available: np.ndarray | None = None,
) -> np.ndarray:
    """Select stable channel identities within the model's neural population.

    Parameters
    ----------
    features : FeatureSet
        Fold arrays containing ids, shape (C,).
    settings : dict
        Channel count, seed, and optional explicit channel_ids.
    available : ndarray, optional
        Permitted source columns, shape (K,); default is every column.

    Returns
    -------
    ndarray, shape (N,)
        Selected source columns in stable order.
    """
    ids = features.arrays["ids"].tolist()
    allowed = np.arange(len(ids)) if available is None else available
    requested = settings.get("channel_ids")
    if requested is not None:
        if not requested or len(set(requested)) != len(requested):
            raise ValueError("Preview channel IDs must be nonempty and unique.")
        if set(requested) - {ids[i] for i in allowed}:
            raise ValueError("Requested preview channel is absent from model.")
        return np.array([ids.index(name) for name in requested])
    count = settings["channels"]
    if count < 1 or len(allowed) < count:
        raise ValueError(
            f"Expected {count} preview channels, got {len(allowed)}."
        )
    return np.asarray(allowed[:count], dtype=int)


def window_excerpts(
    session: FeatureSet,
    fold: FeatureSet,
    indices: np.ndarray,
    columns: np.ndarray,
    stop: float,
    fitted: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Collect matching native, processed and optional fitted window arrays."""
    arrays, native = fold.arrays, session.arrays
    time = arrays["t"][indices]
    if not len(time) or np.unique(arrays["segment"][indices]).size != 1:
        raise ValueError(
            "Preview window must stay inside one nonempty segment."
        )
    raw = (native["native_t"] >= time[0]) & (native["native_t"] < stop)
    if not raw.any():
        raise ValueError("Preview window contains no native behavior samples.")
    values = {
        "t": time,
        "channel_ids": arrays["ids"][columns],
        "unit_dimensions": arrays["units"][columns],
        "source_indices": arrays["indices"][indices],
        "counts": arrays["counts"][indices][:, columns],
        "smoothed": arrays["Y"][indices][:, columns],
        "position": session.arrays["Z"][arrays["indices"][indices]],
        "Z": arrays["Z"][indices],
        "U": arrays["U"][indices],
        "native_t": native["native_t"][raw],
        "native_Z": native["native_Z"][raw],
        "native_U": native["native_U"][raw],
    }
    np.testing.assert_array_equal(native["ids"][columns], values["channel_ids"])
    np.testing.assert_array_equal(
        native["units"][columns], values["unit_dimensions"]
    )
    for number, column in enumerate(columns):
        left, right = native["spike_offsets"][column : column + 2]
        spikes = native["spike_values"][left:right]
        values[f"spikes_{number}"] = spikes[
            (spikes >= time[0]) & (spikes < stop)
        ]
    if fitted is not None:
        ids = fitted["channel_ids"].tolist()
        selected = [ids.index(name) for name in values["channel_ids"]]
        for key, array in fitted.items():
            if key == "channel_ids":
                continue
            if key.endswith("_Y"):
                array = array[:, selected]
            if len(array) != len(time):
                raise ValueError(
                    f"Fitted {key} does not match preview time grid."
                )
            values[key] = array
    for key, array in values.items():
        if (
            np.issubdtype(array.dtype, np.number)
            and not np.isfinite(array).all()
        ):
            raise ValueError(f"Nonfinite excerpt values in {key}.")
    return values


def fitted_stages(
    excerpts: dict,
    name: str,
    dimension: int | slice,
) -> list[PreviewStage]:
    """Describe separately labeled learned stages of one coordinate."""
    if f"pre_{name}" not in excerpts:
        return []
    stages = [
        PreviewStage(
            f"pre_{name}",
            "Preprocessing-model normalization",
            NORMALIZED_UNIT,
            excerpts["t"],
            excerpts[f"pre_{name}"][:, dimension],
        )
    ]
    if name == "Z":
        stages.append(
            PreviewStage(
                "learned_Z",
                "Neural-related behavior\n(checkpoint-derived estimate)",
                "mm"
                if (
                    dimension.start
                    if isinstance(dimension, slice)
                    else dimension
                )
                < 2
                else "mm/s",
                excerpts["t"],
                excerpts["learned_Z"][:, dimension],
            )
        )
    stages.append(
        PreviewStage(
            f"main_{name}",
            "Main-model normalization"
            + ("\nof learned behavior" if name == "Z" else ""),
            NORMALIZED_UNIT,
            excerpts["t"],
            excerpts[f"main_{name}"][:, dimension],
        )
    )
    return stages


def coordinate_figures(
    excerpts: dict,
    fitted: bool,
    suffix: str,
) -> list[tuple[str, str, list[PreviewStage]]]:
    """Describe shared behavior and measured-input preview figures."""
    time = excerpts["t"]
    figures = []
    for name, group in (("Z", "behavior"), ("U", "input")):
        values = excerpts[f"pre_{name}" if fitted else name]
        if values.shape[1] not in ((2, 4) if name == "Z" else (2,)):
            raise ValueError(f"Unsupported coordinate count for {name}.")
        for dimension in range(0, values.shape[1], 2):
            pair = slice(dimension, dimension + 2)
            velocity = name == "Z" and dimension == 2
            quantity = (
                "velocity"
                if velocity
                else "position"
                if name == "Z"
                else "target"
            )
            if fitted:
                stages = fitted_stages(excerpts, name, pair)
            else:
                stages = [
                    PreviewStage(
                        f"native_{name}",
                        "Native position" if name == "Z" else "Native target",
                        "mm",
                        excerpts["native_t"],
                        excerpts[f"native_{name}"][:, :2],
                    ),
                    PreviewStage(
                        "aligned_position" if name == "Z" else "aligned_U",
                        "Aligned position"
                        if name == "Z"
                        else "Aligned target (held)",
                        "mm",
                        time,
                        excerpts["position"][:, :2]
                        if name == "Z"
                        else excerpts["U"],
                    ),
                ]
                if velocity:
                    stages.append(
                        PreviewStage(
                            "velocity",
                            "Inferred backward-difference velocity",
                            "mm/s",
                            time,
                            excerpts["Z"][:, pair],
                        )
                    )
            stages = [
                replace(
                    stage,
                    coordinates=tuple(
                        (
                            "Position"
                            if velocity and i < 2 and not fitted
                            else quantity.capitalize()
                        )
                        + f" {coordinate}"
                        for coordinate in COORDINATES
                    ),
                    kind="held" if name == "U" else stage.kind,
                )
                for i, stage in enumerate(stages)
            ]
            figures.append(
                (
                    f"{group}_{quantity}_{suffix}.png",
                    f"{quantity.capitalize()} x and y",
                    stages,
                )
            )
    return figures


def signal_stages(excerpts: dict) -> list[tuple[str, str, list[PreviewStage]]]:
    """Describe preprocessing-only or fitted-only figures from their arrays."""
    time = excerpts["t"]
    fitted = "pre_Y" in excerpts
    suffix = "fitted" if fitted else "preprocessing"
    figures = []
    for number, (channel, unit) in enumerate(
        zip(
            excerpts["channel_ids"],
            excerpts["unit_dimensions"],
        )
    ):
        if fitted:
            stages = fitted_stages(excerpts, "Y", number)
        else:
            stages = [
                PreviewStage(
                    "spikes",
                    "Original spike events",
                    "Events",
                    excerpts[f"spikes_{number}"],
                    None,
                    "spikes",
                ),
                PreviewStage(
                    "counts",
                    "Binned counts",
                    "Spikes/bin",
                    time,
                    excerpts["counts"][:, number],
                    "counts",
                ),
                PreviewStage(
                    "smoothed",
                    "Smoothed counts (offline Gaussian)",
                    "Spikes/bin",
                    time,
                    excerpts["smoothed"][:, number],
                ),
            ]
        safe = str(channel).replace(" ", "_")
        if Path(safe).name != safe or safe in (".", ".."):
            raise ValueError(f"Unsafe preview channel filename: {channel!r}")
        figures.append(
            (
                f"neural_{safe}_unit_{unit}_{suffix}.png",
                f"{channel}, unit {unit}",
                stages,
            )
        )
    figures.extend(coordinate_figures(excerpts, fitted, suffix))
    return figures


def render_figures(
    destination: Path,
    excerpts: dict,
    session: str,
    bounds: tuple[float, float],
    style: dict,
    figures: list[tuple[str, str, list[PreviewStage]]],
    fitted_metadata: tuple[str, ...] = ("unit_dimensions",),
) -> list[dict]:
    """Write supplied preview figures and their numerical excerpt archive."""
    records = []
    for filename, label, stages in figures:
        title = f"{session} — {label}\nWindow {bounds[0]:.3f}–{bounds[1]:.3f} s"
        save_signal(destination / filename, stages, label, title, bounds, style)
        records.append(
            {
                "file": filename,
                "signal": label,
                "stages": [
                    {
                        "key": s.key,
                        "label": s.label,
                        "unit": s.unit,
                        "kind": s.kind,
                        "coordinates": list(s.coordinates),
                    }
                    for s in stages
                ],
            }
        )
    payload = excerpts
    if "pre_Y" in excerpts:
        payload = {
            key: value
            for key, value in excerpts.items()
            if key
            in (
                "t",
                "channel_ids",
                "source_indices",
                "learned_Z",
            )
            or key in fitted_metadata
            or key.startswith(("pre_", "main_"))
        }
    np.savez_compressed(destination / "excerpts.npz", **payload)
    return records


def render_window(
    destination: Path,
    excerpts: dict,
    session: str,
    bounds: tuple[float, float],
    style: dict,
) -> list[dict]:
    """Write standard spike preview figures for one time window."""
    return render_figures(
        destination,
        excerpts,
        session,
        bounds,
        style,
        signal_stages(excerpts),
    )
