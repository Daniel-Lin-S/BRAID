"""Build native-versus-aligned previews for cached NHP LFP features.

Preprocessing previews read only the selected native NWB time/channel window
and compare it with the final 20-Hz LFP. Fitted previews contain checkpoint-
derived stages and reuse the shared behavior and measured-input rendering.
"""

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .cache import file_digest
from .contracts import FeatureSet
from .nhp_lfp import _repair_timestamps
from .preview_rendering import PreviewStage
from .signal_previews import (
    coordinate_figures,
    fitted_stages,
    render_figures,
)

VOLT_UNIT = "V"


def _signature(path: Path) -> dict[str, Any]:
    """Return the source attributes protected during a preview read."""
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _source_window(
    metadata: dict,
    channel_ids: list[str],
    start: float,
    stop: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Read converted native broadband for one short preview interval."""
    source = metadata["broadband_source"]
    path = Path(source["path"]).resolve()
    expected = source["signature"]
    if _signature(path) != expected:
        raise ValueError(f"Broadband source signature changed: {path}")
    interval = source["clock_interval"]
    approximate = source["clock_start"]
    left = max(0, int(np.floor((start - approximate) / interval)) - 2)
    right = min(
        source["sample_count"],
        int(np.ceil((stop - approximate) / interval)) + 3,
    )
    if right - left < 2:
        raise ValueError("Native broadband preview window is empty.")
    with h5py.File(path, "r") as data:
        group = data[source["group"]]
        raw_ids = [
            value.decode() if isinstance(value, bytes) else str(value)
            for value in group["electrode_names"][()].ravel()
        ]
        missing = set(channel_ids) - set(raw_ids)
        if missing:
            raise ValueError(
                f"Preview channels are absent from broadband: {sorted(missing)}"
            )
        columns = np.array([raw_ids.index(value) for value in channel_ids])
        timestamps = np.asarray(group["timestamps"][left:right]).ravel()
        ticks = np.asarray(group["sync/ticks"][left:right]).ravel()
        timestamps, _ = _repair_timestamps(timestamps, ticks)
        order = np.argsort(columns)
        sorted_values = np.asarray(
            group["data"][slice(left, right), columns[order]],
            dtype=np.float32,
        )
        values = sorted_values[:, np.argsort(order)] * source["conversion"]
    selected = (timestamps >= start) & (timestamps < stop)
    if not selected.any() or not np.isfinite(values[selected]).all():
        raise ValueError("Native broadband preview contains no finite samples.")
    if _signature(path) != expected:
        raise ValueError(f"Broadband source changed during preview: {path}")
    return timestamps[selected], values[selected]


class LFPPreviewAdapter:
    """Extract and render LFP-specific preprocessing and fitted previews."""

    def __init__(self, settings: dict) -> None:
        self.settings = settings

    def implementation_identity(self) -> dict[str, str]:
        """Return hashes of the LFP-specific rendering implementation."""
        return {
            name: file_digest(Path(__file__).with_name(name))
            for name in (
                "lfp_previews.py",
                "signal_previews.py",
                "preview_rendering.py",
            )
        }

    def window_excerpts(
        self,
        session: FeatureSet,
        fold: FeatureSet,
        indices: np.ndarray,
        columns: np.ndarray,
        stop: float,
        fitted: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        """Collect native, final and optional fitted LFP window arrays."""
        arrays, native = fold.arrays, session.arrays
        time = arrays["t"][indices]
        if not len(time) or np.unique(arrays["segment"][indices]).size != 1:
            raise ValueError(
                "LFP preview window must stay inside one nonempty segment."
            )
        channel_ids = arrays["ids"][columns]
        behavior = (native["native_t"] >= time[0]) & (native["native_t"] < stop)
        if not behavior.any():
            raise ValueError("LFP preview contains no native behavior samples.")
        values = {
            "t": time,
            "channel_ids": channel_ids,
            "electrode_indices": arrays["units"][columns],
            "source_indices": arrays["indices"][indices],
            "position": session.arrays["Z"][arrays["indices"][indices]],
            "Z": arrays["Z"][indices],
            "U": arrays["U"][indices],
            "native_t": native["native_t"][behavior],
            "native_Z": native["native_Z"][behavior],
            "native_U": native["native_U"][behavior],
            "preprocessing_steps_json": np.asarray(
                json.dumps(
                    session.metadata["preprocessing_steps"],
                    sort_keys=True,
                )
            ),
        }
        if fitted is None:
            raw_t, raw_lfp = _source_window(
                session.metadata,
                channel_ids.tolist(),
                float(time[0]),
                stop,
            )
            values.update(
                raw_lfp_t=raw_t,
                raw_lfp=raw_lfp,
                raw_lfp_unit=np.asarray(VOLT_UNIT),
                lfp=arrays["Y"][indices][:, columns],
                lfp_unit=np.asarray(VOLT_UNIT),
            )
        np.testing.assert_array_equal(
            native["ids"][columns], values["channel_ids"]
        )
        if fitted is not None:
            fitted_ids = fitted["channel_ids"].tolist()
            selected = [fitted_ids.index(name) for name in channel_ids]
            for key, array in fitted.items():
                if key == "channel_ids":
                    continue
                if key.endswith("_Y"):
                    array = array[:, selected]
                if len(array) != len(time):
                    raise ValueError(
                        f"Fitted {key} does not match LFP preview time grid."
                    )
                values[key] = array
        for key, array in values.items():
            if (
                np.issubdtype(array.dtype, np.number)
                and not np.isfinite(array).all()
            ):
                raise ValueError(f"Nonfinite LFP preview values in {key}.")
        return values

    def figures(
        self, excerpts: dict
    ) -> list[tuple[str, str, list[PreviewStage]]]:
        """Describe native/final or checkpoint-derived LFP panels."""
        fitted = "pre_Y" in excerpts
        suffix = "fitted" if fitted else "preprocessing"
        figures = []
        for number, (channel, electrode) in enumerate(
            zip(excerpts["channel_ids"], excerpts["electrode_indices"])
        ):
            if fitted:
                stages = fitted_stages(excerpts, "Y", number)
            else:
                stages = [
                    PreviewStage(
                        "raw_lfp",
                        "Native broadband before preprocessing",
                        VOLT_UNIT,
                        excerpts["raw_lfp_t"],
                        excerpts["raw_lfp"][:, number],
                    ),
                    PreviewStage(
                        "lfp",
                        "Final aligned LFP after preprocessing",
                        VOLT_UNIT,
                        excerpts["t"],
                        excerpts["lfp"][:, number],
                    ),
                ]
            safe = str(channel).replace(" ", "_")
            if Path(safe).name != safe or safe in (".", ".."):
                raise ValueError(
                    f"Unsafe LFP preview channel filename: {channel!r}"
                )
            figures.append(
                (
                    f"neural_{safe}_electrode_{electrode}_{suffix}.png",
                    f"{channel}, electrode {electrode}",
                    stages,
                )
            )
        figures.extend(coordinate_figures(excerpts, fitted, suffix))
        return figures

    def render_window(
        self,
        destination: Path,
        excerpts: dict,
        session: str,
        bounds: tuple[float, float],
        style: dict,
    ) -> list[dict]:
        """Write LFP figures and the corresponding numerical excerpt."""
        return render_figures(
            destination,
            excerpts,
            session,
            bounds,
            style,
            self.figures(excerpts),
            fitted_metadata=("electrode_indices",),
        )

    def window_metadata(self, excerpts: dict) -> dict[str, Any]:
        """Return channel metadata stored in the preview manifest."""
        metadata = {"electrode_indices": excerpts["electrode_indices"].tolist()}
        if "raw_lfp_unit" in excerpts:
            metadata["units"] = {
                "raw_lfp": str(excerpts["raw_lfp_unit"]),
                "lfp": str(excerpts["lfp_unit"]),
            }
        return metadata
