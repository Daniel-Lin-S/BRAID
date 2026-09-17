"""Build signal-separated data previews from cached arrays and fitted stages.

Each window contains neural/, behavior/, input/ PNGs and excerpts.npz.
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
    count = min(settings["channels"], len(allowed))
    if count < 1:
        raise ValueError("No neural channels available for previews.")
    rng = np.random.default_rng(settings["seed"])
    return np.sort(rng.choice(allowed, count, replace=False))


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
    values = dict(
        t=time,
        channel_ids=arrays["ids"][columns],
        unit_dimensions=arrays["units"][columns],
        source_indices=arrays["indices"][indices],
        counts=arrays["counts"][indices][:, columns],
        smoothed=arrays["Y"][indices][:, columns],
        position=session.arrays["Z"][arrays["indices"][indices]],
        Z=arrays["Z"][indices],
        U=arrays["U"][indices],
        native_t=native["native_t"][raw],
        native_Z=native["native_Z"][raw],
        native_U=native["native_U"][raw],
    )
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


def signal_stages(excerpts: dict) -> list[tuple[str, str, list[PreviewStage]]]:
    """Build one figure description per neural channel or coordinate pair."""
    time = excerpts["t"]
    figures = []
    for number, (channel, unit) in enumerate(
        zip(
            excerpts["channel_ids"],
            excerpts["unit_dimensions"],
        )
    ):
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
                "50-ms binned counts",
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
        ] + fitted_stages(excerpts, "Y", number)
        filename = f"neural/{channel.replace(' ', '_')}_unit_{unit}.png"
        figures.append((filename, f"{channel}, unit {unit}", stages))
    for name, group in (("Z", "behavior"), ("U", "input")):
        if excerpts[name].shape[1] not in ((2, 4) if name == "Z" else (2,)):
            raise ValueError(f"Unsupported coordinate count for {name}.")
        for dimension in range(0, excerpts[name].shape[1], 2):
            pair = slice(dimension, dimension + 2)
            velocity = name == "Z" and dimension >= len(COORDINATES)
            quantity = (
                "velocity"
                if velocity
                else ("position" if name == "Z" else "target")
            )
            label = f"{quantity.capitalize()} x and y"
            native_label = "Native position" if name == "Z" else "Native target"
            kind = "held" if name == "U" else "line"
            stages = [
                PreviewStage(
                    f"native_{name}",
                    native_label,
                    "mm",
                    excerpts["native_t"],
                    excerpts[f"native_{name}"][:, :2],
                    kind,
                )
            ]
            stages.append(
                PreviewStage(
                    "aligned_position" if velocity else f"aligned_{name}",
                    "Aligned position"
                    if name == "Z"
                    else "Aligned target (held)",
                    "mm",
                    time,
                    excerpts["position"][:, :2]
                    if name == "Z"
                    else excerpts[name][:, pair],
                    kind,
                )
            )
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
            stages += fitted_stages(excerpts, name, pair)
            stages = [
                replace(
                    stage,
                    coordinates=tuple(
                        (
                            "Position"
                            if velocity and i < 2
                            else quantity.capitalize()
                        )
                        + f" {c}"
                        for c in COORDINATES
                    ),
                    kind="held" if name == "U" else stage.kind,
                )
                for i, stage in enumerate(stages)
            ]
            figures.append((f"{group}/{quantity}_xy.png", label, stages))
    return figures


def render_window(
    destination: Path,
    excerpts: dict,
    session: str,
    bounds: tuple[float, float],
    style: dict,
) -> list[dict]:
    """Write separated PNGs and a numerical archive for one time window."""
    records = []
    for filename, label, stages in signal_stages(excerpts):
        title = f"{session} — {label}\nWindow {bounds[0]:.3f}–{bounds[1]:.3f} s"
        save_signal(destination / filename, stages, label, title, bounds, style)
        records.append(
            dict(
                file=filename,
                signal=label,
                stages=[
                    dict(
                        key=s.key,
                        label=s.label,
                        unit=s.unit,
                        kind=s.kind,
                        coordinates=list(s.coordinates),
                    )
                    for s in stages
                ],
            )
        )
    np.savez_compressed(destination / "excerpts.npz", **excerpts)
    return records
