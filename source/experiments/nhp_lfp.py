"""Read Indy NWB broadband recordings and cache aligned LFP features.

Each ``raw/indy_*.nwb`` recording is paired with the matching MATLAB task
file. The ordered preprocessing pipeline produces Y (T,C) in volts on the
20-Hz task grid; Z, U and t retain the existing spike experiment mapping.
Fold caches add indices, segment and role arrays without changing spike data.
"""

import json
import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .cache import atomic_json, cached, file_digest, fingerprint
from .contracts import FeatureSet
from .lfp_preprocessing import (
    PipelineContext,
    preprocessing_identity,
    run_preprocessing,
)
from .nhp import aligned, fold_features

LOGGER = logging.getLogger(__name__)
PREPROCESSING_VERSION = "nhp-lfp-v1"
BROADBAND_GROUP = "acquisition/timeseries/broadband"
EXPECTED_CHANNELS = 96
EXPECTED_BEHAVIOR_RATE_HZ = 250
FILTER_PADDING_SECONDS = 2.0
TIMESTAMP_RELATIVE_TOLERANCE = 0.1


class NWBBroadbandSource:
    """Read converted voltage channel blocks from one open NWB dataset."""

    def __init__(
        self,
        dataset: h5py.Dataset,
        start: int,
        stop: int,
        timestamps: np.ndarray,
        channel_ids: list[str],
        conversion: float,
    ) -> None:
        self.dataset = dataset
        self.start = start
        self.stop = stop
        self.timestamps = timestamps
        self.channel_ids = channel_ids
        self.conversion = conversion
        self.shape = (stop - start, len(channel_ids))

    def read(self, rows: slice, columns: np.ndarray) -> np.ndarray:
        """Read one time/channel block and convert integer counts to volts."""
        start, stop, step = rows.indices(self.shape[0])
        if step != 1:
            raise ValueError("Broadband source rows must use unit stride.")
        if (
            columns.ndim != 1
            or not len(columns)
            or not np.issubdtype(columns.dtype, np.integer)
            or len(np.unique(columns)) != len(columns)
            or np.any(columns < 0)
            or np.any(columns >= self.shape[1])
        ):
            raise ValueError("Broadband source requires selected channels.")
        absolute = slice(self.start + start, self.start + stop)
        order = np.argsort(columns)
        sorted_values = self.dataset[absolute, columns[order]]
        values = sorted_values[:, np.argsort(order)]
        result = np.asarray(values, dtype=np.float32) * self.conversion
        if result.shape != (stop - start, len(columns)):
            raise ValueError(
                "Expected broadband block shape "
                f"{(stop - start, len(columns))}, got {result.shape}."
            )
        if not np.isfinite(result).all():
            raise ValueError("Broadband source contains nonfinite voltages.")
        return result


def _decode(values: np.ndarray) -> list[str]:
    """Decode a one-dimensional NWB byte-string array."""
    result = [
        value.decode() if isinstance(value, bytes) else str(value)
        for value in values.ravel()
    ]
    if not result or len(result) != len(set(result)):
        raise ValueError("Expected nonempty unique broadband channel IDs.")
    return result


def _validate_channels(group: h5py.Group) -> tuple[list[str], np.ndarray]:
    """Validate the fixed M1 array and return IDs and electrode indices."""
    required = {
        "data",
        "timestamps",
        "sync",
        "electrode_names",
        "electrode_idx",
    }
    missing = required - set(group)
    if missing or "ticks" not in group.get("sync", {}):
        raise ValueError(f"Broadband NWB is missing fields: {sorted(missing)}.")
    names = _decode(group["electrode_names"][()])
    if len(names) != EXPECTED_CHANNELS or any(
        not name.startswith("M1 ") for name in names
    ):
        raise ValueError(
            f"Expected {EXPECTED_CHANNELS} M1 channels, got {names}."
        )
    data = group["data"]
    if data.ndim != 2 or data.shape[1] != len(names):
        raise ValueError(
            f"Expected broadband shape (T, {len(names)}), got {data.shape}."
        )
    if "electrode_idx" in group:
        indices = np.asarray(group["electrode_idx"][()]).ravel()
    else:
        indices = np.arange(len(names))
    if indices.shape != (len(names),) or len(np.unique(indices)) != len(names):
        raise ValueError("Expected one unique electrode index per channel.")
    return names, indices


def _repair_timestamps(
    timestamps: np.ndarray, ticks: np.ndarray
) -> tuple[np.ndarray, list[int]]:
    """Repair isolated timestamp holes proven continuous by sample ticks."""
    if timestamps.ndim != 1 or ticks.shape != timestamps.shape:
        raise ValueError(
            "Broadband timestamps and ticks must be matching vectors."
        )
    tick_delta = np.diff(ticks)
    if not len(tick_delta) or np.any(tick_delta != 1):
        where = np.flatnonzero(tick_delta != 1)
        raise ValueError(
            "Broadband sample ticks are discontinuous at indices "
            f"{where[:10].tolist()}."
        )
    result = np.asarray(timestamps, dtype=float).copy()
    invalid = np.flatnonzero(~np.isfinite(result) | (result <= 0))
    repaired = []
    for index in invalid:
        if (
            index == 0
            or index == len(result) - 1
            or index - 1 in invalid
            or index + 1 in invalid
        ):
            raise ValueError(f"Ambiguous broadband timestamp at index {index}.")
        result[index] = (result[index - 1] + result[index + 1]) / 2
        repaired.append(int(index))
    delta = np.diff(result)
    median = float(np.median(delta))
    if (
        not np.isfinite(result).all()
        or median <= 0
        or np.any(delta <= 0)
        or not np.allclose(
            delta,
            median,
            rtol=TIMESTAMP_RELATIVE_TOLERANCE,
            atol=0,
        )
    ):
        raise ValueError("Broadband timestamps are not a regular clock.")
    return result, repaired


def _task_arrays(
    path: Path, sampling_rate_hz: float
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Read task signals and construct the established model time grid."""
    with h5py.File(path, "r") as data:
        required = {"t", "cursor_pos", "target_pos"}
        missing = required - set(data)
        if missing:
            raise ValueError(
                f"Task file is missing fields {sorted(missing)}: {path}"
            )
        native_t = np.asarray(data["t"][()]).ravel()
        position = np.asarray(data["cursor_pos"][()]).T
        target = np.asarray(data["target_pos"][()]).T
    expected_interval = 1 / EXPECTED_BEHAVIOR_RATE_HZ
    if (
        len(native_t) < 2
        or not np.isfinite(native_t).all()
        or not np.allclose(
            np.diff(native_t), expected_interval, rtol=0, atol=1e-7
        )
    ):
        raise ValueError(f"Expected a regular 250-Hz task stream: {path}")
    expected_shape = (len(native_t), 2)
    for name, values in (("cursor_pos", position), ("target_pos", target)):
        if values.shape != expected_shape or not np.isfinite(values).all():
            raise ValueError(
                f"Expected {name} shape {expected_shape}, got {values.shape}."
            )
    interval = 1 / sampling_rate_hz
    count = int(np.floor((native_t[-1] - native_t[0]) / interval + 1e-7))
    if count < 2:
        raise ValueError(f"Task stream is too short for 20-Hz output: {path}")
    grid = native_t[0] + np.arange(1, count + 1) * interval
    behavior = np.column_stack(
        [np.interp(grid, native_t, position[:, axis]) for axis in range(2)]
    )
    measured_input = target[np.searchsorted(native_t, grid, side="right") - 1]
    return native_t, position, target, behavior, measured_input, grid


def _source_signature(path: Path) -> dict[str, Any]:
    """Return the storage signature used to memoize a source checksum."""
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


class NHPLFPDataset:
    """Dataset plugin for paired Indy broadband and task recordings."""

    def __init__(self, settings: dict[str, Any]) -> None:
        self.settings = settings
        self.root = Path(settings["root"]).expanduser().resolve()
        self.raw_root = self.root / settings["broadband_subdirectory"]
        self.cache_root = Path(settings["cache_root"]).expanduser().resolve()
        self.mode = settings["cache_mode"]
        if settings.get("features") != "lfp":
            raise ValueError("NHPLFPDataset requires features=lfp.")
        self.records = json.loads((self.root / "record.json").read_text())

    def sessions(self) -> list[str]:
        """Return sessions having both broadband and task recordings."""
        found = sorted(path.stem for path in self.raw_root.glob("indy_*.nwb"))
        if not found:
            raise FileNotFoundError(
                f"No Indy NWB recordings under {self.raw_root.resolve()}."
            )
        missing_tasks = [
            name for name in found if not (self.root / f"{name}.mat").is_file()
        ]
        if missing_tasks:
            raise FileNotFoundError(
                f"Missing paired task files: {missing_tasks}."
            )
        chosen = self.settings.get("sessions") or found
        if not chosen or len(chosen) != len(set(chosen)):
            raise ValueError("Expected a nonempty unique session list.")
        missing = set(chosen) - set(found)
        if missing:
            raise FileNotFoundError(
                f"Missing Indy broadband sessions: {sorted(missing)}"
            )
        return chosen

    def inventory(self) -> dict[str, list[str]]:
        """Return ordered M1 broadband channel IDs for every session."""
        sessions = self.sessions()
        signatures = {
            name: _source_signature(self.raw_root / f"{name}.nwb")
            for name in sessions
        }
        path = self.cache_root / "lfp_inventory.json"
        if path.exists() and self.mode == "reuse":
            previous = json.loads(path.read_text())
            if previous.get("signatures") == signatures:
                return previous["sessions"]
        inventory = {}
        for name in sessions:
            with h5py.File(self.raw_root / f"{name}.nwb", "r") as data:
                if BROADBAND_GROUP not in data:
                    raise ValueError(
                        f"Missing {BROADBAND_GROUP} in {name}.nwb."
                    )
                inventory[name], _ = _validate_channels(data[BROADBAND_GROUP])
        if self.mode != "off":
            atomic_json(path, {"signatures": signatures, "sessions": inventory})
        return inventory

    def _digest(self, path: Path) -> tuple[dict[str, Any], str]:
        """Reuse a checksum only while the exact source signature matches."""
        signature = _source_signature(path)
        index = (
            self.cache_root
            / "sources"
            / "lfp"
            / f"{fingerprint(str(path.resolve()))}.json"
        )
        previous = json.loads(index.read_text()) if index.exists() else {}
        if previous.get("signature") == signature:
            return signature, previous["sha256"]
        digest = file_digest(path)
        if self.mode != "off":
            atomic_json(index, {"signature": signature, "sha256": digest})
        return signature, digest

    def load(self, session: str) -> FeatureSet:
        """Load or build one provenance-addressed aligned LFP session."""
        if session not in self.sessions():
            raise ValueError(f"Session is not configured for LFP: {session}")
        broadband = (self.raw_root / f"{session}.nwb").resolve()
        task = (self.root / f"{session}.mat").resolve()
        broadband_signature, broadband_digest = self._digest(broadband)
        task_signature, task_digest = self._digest(task)
        implementation = {
            name: file_digest(Path(__file__).with_name(name))
            for name in ("nhp_lfp.py", "lfp_preprocessing.py", "nhp.py")
        }
        identity = {
            "features": "lfp",
            "version": PREPROCESSING_VERSION,
            "sources": {
                "broadband": {
                    "signature": broadband_signature,
                    "sha256": broadband_digest,
                },
                "task": {"signature": task_signature, "sha256": task_digest},
            },
            "sampling_rate_hz": self.settings["sampling_rate_hz"],
            "preprocessing": preprocessing_identity(
                self.settings["preprocessing"]
            ),
            "implementation": implementation,
            "alignment": "bin_end_linear_position_previous_target",
        }
        return cached(
            self.cache_root,
            "lfp_session",
            identity,
            self.mode,
            lambda: self._read(broadband, task, identity),
        )

    def fold(
        self, features: FeatureSet, fold: int, velocity: bool
    ) -> FeatureSet:
        """Cache split-local LFP and behavior features for one CV fold."""
        identity = {
            "source": features.metadata["identity"],
            "fold": fold,
            "velocity": velocity,
            "cv": self.settings["cv"],
            "version": PREPROCESSING_VERSION,
        }
        return cached(
            self.cache_root,
            "lfp_fold",
            identity,
            self.mode,
            lambda: fold_features(features, self.settings, fold, velocity),
        )

    def _read(
        self, broadband: Path, task: Path, identity: dict[str, Any]
    ) -> FeatureSet:
        """Validate, preprocess and align one paired broadband/task session."""
        rate = float(self.settings["sampling_rate_hz"])
        (
            native_t,
            native_position,
            native_target,
            behavior,
            measured_input,
            grid,
        ) = _task_arrays(task, rate)
        with h5py.File(broadband, "r") as data:
            if BROADBAND_GROUP not in data:
                raise ValueError(
                    f"Missing {BROADBAND_GROUP}: {broadband.resolve()}"
                )
            group = data[BROADBAND_GROUP]
            names, indices = _validate_channels(group)
            raw = group["data"]
            unit = group["data"].attrs.get("unit")
            if isinstance(unit, bytes):
                unit = unit.decode()
            conversion = group["data"].attrs.get("conversion")
            if (
                unit != "volt"
                or not isinstance(
                    conversion, (int, float, np.integer, np.floating)
                )
                or not np.isfinite(conversion)
                or conversion <= 0
            ):
                raise ValueError(
                    "Broadband data require a positive volts conversion."
                )
            timestamps = np.asarray(group["timestamps"][()]).ravel()
            ticks = np.asarray(group["sync/ticks"][()]).ravel()
            timestamps, repaired = _repair_timestamps(timestamps, ticks)
            source_rate = float(
                (ticks[-1] - ticks[0]) / (timestamps[-1] - timestamps[0])
            )
            left = int(
                np.searchsorted(
                    timestamps,
                    grid[0] - FILTER_PADDING_SECONDS,
                    side="left",
                )
            )
            right = int(
                np.searchsorted(
                    timestamps,
                    grid[-1] + FILTER_PADDING_SECONDS,
                    side="right",
                )
            )
            if left == 0 or right == len(timestamps) or right - left < 2:
                raise ValueError(
                    "Broadband recording lacks anti-alias filter padding."
                )
            source = NWBBroadbandSource(
                raw,
                left,
                right,
                timestamps[left:right],
                names,
                float(conversion),
            )
            result = run_preprocessing(
                source,
                self.settings["preprocessing"],
                PipelineContext(
                    target_timestamps=grid,
                    target_sampling_rate_hz=rate,
                    source_sampling_rate_hz=source_rate,
                ),
            )
        arrays = {
            "Y": result.values,
            "Z": behavior,
            "U": measured_input,
            "t": result.timestamps,
            "ids": np.asarray(names),
            "units": np.asarray(indices),
            "native_t": native_t,
            "native_Z": native_position,
            "native_U": native_target,
        }
        aligned(arrays, 1 / rate)
        broadband_signature = identity["sources"]["broadband"]["signature"]
        metadata = {
            "identity": identity,
            "session": broadband.stem,
            "source": str(broadband),
            "neural_modality": "lfp",
            "raw_channel_ids": names,
            "selected_channel_ids": names,
            "electrode_indices": indices.tolist(),
            "behavior_fields": ["cursor_pos.x", "cursor_pos.y"],
            "input_fields": ["target_pos.x", "target_pos.y"],
            "position_mapping": "documented task-plane fingertip in mm",
            "dataset_record": self.records["id"],
            "preprocessing_steps": result.steps,
            "resampling": result.resampling,
            "broadband_source": {
                "path": str(broadband),
                "signature": broadband_signature,
                "sha256": identity["sources"]["broadband"]["sha256"],
                "group": BROADBAND_GROUP,
                "conversion": float(conversion),
                "unit": "volt",
                "clock_start": float(timestamps[0]),
                "clock_interval": 1 / source_rate,
                "sample_count": len(timestamps),
                "repaired_timestamp_indices": repaired,
            },
            "deviations": [
                "velocity inferred by backward 20-Hz position differences",
                "provisional guarded temporal CV",
            ],
        }
        LOGGER.info(
            "Loaded %s: %d samples, %d M1 LFP channels",
            broadband,
            len(grid),
            len(names),
        )
        return FeatureSet(arrays, metadata)
