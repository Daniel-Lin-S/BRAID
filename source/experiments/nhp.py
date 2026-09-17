"""Read Indy MATLAB 7.3 sessions and cache aligned spike features.

Input files contain referenced chan_names/spikes cells, timestamps t,
cursor_pos (task-plane fingertip position, mm), and target_pos (mm).
MATLAB arrays are transposed to time-first output. Waveform snippets are
not continuous LFP and are never interpreted as such.

Session arrays: Y counts (T,C), Z positions (T,2), U targets (T,2), t (T,),
ids (C,), units (C,), native_t/Z/U, and flattened spike_values with
spike_offsets (C+1,). Fold arrays replace Y/Z with split-local features,
and add indices, segment, and role (0=train, 1=validation, 2=test), all (T,).
"""

import json
import logging
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter1d

from .cache import atomic_json, cached, file_digest, fingerprint
from .contracts import FeatureSet

LOGGER = logging.getLogger(__name__)
PREPROCESSING_VERSION = "nhp-v1"
TRAIN, VALIDATION, TEST = 0, 1, 2


def aligned(arrays: dict[str, np.ndarray], interval: float) -> None:
    """Validate finite, monotone, time-aligned observations on a fixed grid."""
    count = len(arrays["t"])
    if count < 2:
        raise ValueError(f"Expected at least two samples, got {count}.")
    for name in ("Y", "Z", "U"):
        value = arrays[name]
        if value.ndim != 2 or len(value) != count:
            raise ValueError(
                f"{name}: expected ({count}, D), got {value.shape}."
            )
        if not np.isfinite(value).all():
            raise ValueError(f"{name} contains nonfinite values.")
    delta = np.diff(arrays["t"])
    if not np.isfinite(delta).all() or np.any(delta <= 0):
        raise ValueError("Timestamps must be finite and strictly increasing.")
    if not np.allclose(delta, interval, rtol=0, atol=1e-8):
        raise ValueError(f"Expected timestamp spacing {interval} seconds.")


def temporal_segments(
    count: int, folds: int, fold: int, guard: int
) -> list[tuple[int, int, int]]:
    """Return contiguous role segments after trimming role-change guards."""
    if folds < 3 or not 0 <= fold < folds or guard < 0:
        raise ValueError("Invalid fold count, fold ID, or guard length.")
    blocks = np.array_split(np.arange(count), folds)
    if any(not len(block) for block in blocks):
        raise ValueError("Session is too short for the requested folds.")
    roles = np.full(count, TRAIN)
    roles[blocks[fold]] = TEST
    roles[blocks[(fold + 1) % folds]] = VALIDATION
    boundaries = np.r_[0, np.flatnonzero(np.diff(roles)) + 1, count]
    segments = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        start = int(left + (guard if left else 0))
        stop = int(right - (guard if right < count else 0))
        if stop - start < 2:
            raise ValueError("Guard trimming leaves an empty/short segment.")
        segments.append((start, stop, int(roles[left])))
    return segments


def fold_features(
    session: FeatureSet, settings: dict, fold: int, velocity: bool
) -> FeatureSet:
    """Apply smoothing and backward velocity within each guarded segment."""
    interval = 1 / settings["sampling_rate_hz"]
    guard = int(np.ceil(settings["cv"]["guard_seconds"] / interval))
    segments = temporal_segments(
        len(session.arrays["t"]), settings["cv"]["folds"], fold, guard
    )
    outputs = {
        key: []
        for key in ("Y", "Z", "U", "t", "indices", "segment", "role", "counts")
    }
    for segment, (start, stop, role) in enumerate(segments):
        source = session.arrays
        counts = source["Y"][start:stop]
        neural = counts.copy()
        smoothing = settings.get("spike_smoothing")
        if smoothing:
            neural = gaussian_filter1d(
                neural,
                smoothing["sigma_ms"] / (interval * 1000),
                axis=0,
                mode=smoothing["boundary"],
                truncate=smoothing["truncate"],
            )
        behavior = source["Z"][start:stop]
        first = 0
        if velocity:
            behavior = np.column_stack(
                (behavior[1:], np.diff(behavior, axis=0) / interval)
            )
            first = 1
        arrays = dict(
            Y=neural[first:],
            Z=behavior,
            U=source["U"][start + first : stop],
            t=source["t"][start + first : stop],
        )
        aligned(arrays, interval)
        size = len(arrays["t"])
        arrays.update(
            indices=np.arange(start + first, stop),
            segment=np.full(size, segment),
            role=np.full(size, role),
            counts=counts[first:],
        )
        for key, value in arrays.items():
            outputs[key].append(value)
    result = {key: np.concatenate(value) for key, value in outputs.items()}
    result["ids"] = session.arrays["ids"]
    result["units"] = session.arrays["units"]
    metadata = dict(
        session.metadata,
        fold=fold,
        segments=segments,
        velocity=velocity,
        settings=settings,
        session_cache=str(session.path) if session.path else None,
    )
    if velocity:
        metadata["behavior_fields"] = [
            "cursor_pos.x",
            "cursor_pos.y",
            "inferred_velocity.x",
            "inferred_velocity.y",
        ]
        metadata["deviations"] = session.metadata["deviations"] + [
            "velocity inferred by backward 20-Hz position differences; "
            "dataset-provided velocities are not used",
            "first unsupported velocity sample removed per segment",
        ]
    return FeatureSet(result, metadata)


class NHPDataset:
    """Dataset plugin for the real Indy reaching sessions."""

    def __init__(self, settings: dict) -> None:
        self.settings = settings
        self.root = Path(settings["root"]).expanduser().resolve()
        self.cache_root = Path(settings["cache_root"]).expanduser().resolve()
        self.mode = settings["cache_mode"]
        if settings["features"] != "spike":
            raise ValueError("LFP training is deferred; select features=spike.")
        self.records = json.loads((self.root / "record.json").read_text())

    def sessions(self) -> list[str]:
        """Return the configured ordered session IDs, validating existence."""
        found = sorted(p.stem for p in self.root.glob("indy_*.mat"))
        chosen = self.settings.get("sessions") or found
        if not chosen or len(chosen) != len(set(chosen)):
            raise ValueError("Expected a nonempty unique session list.")
        missing = set(chosen) - set(found)
        if missing:
            raise FileNotFoundError(f"Missing Indy sessions: {sorted(missing)}")
        return chosen

    def inventory(self) -> dict[str, list[str]]:
        """Cache all selected channel IDs using only MATLAB cell headers."""
        sessions = self.sessions()
        signatures = {
            session: [
                int((self.root / f"{session}.mat").stat().st_size),
                int((self.root / f"{session}.mat").stat().st_mtime_ns),
            ]
            for session in sessions
        }
        path = self.cache_root / "inventory.json"
        if path.exists() and self.mode == "reuse":
            previous = json.loads(path.read_text())
            if previous["signatures"] == signatures:
                return previous["sessions"]
        inventory = {}
        for session in sessions:
            selected = []
            with h5py.File(self.root / f"{session}.mat") as data:
                for index, ref in enumerate(data["chan_names"][()].ravel()):
                    name = "".join(chr(int(c)) for c in data[ref][()].ravel())
                    if not name.startswith("M1 "):
                        continue
                    for unit in range(data["spikes"].shape[0]):
                        cell = data[data["spikes"][unit, index]]
                        if cell.size and not cell.attrs.get(
                            "MATLAB_empty", False
                        ):
                            selected.append(name)
                            break
            if not selected:
                raise ValueError(f"No nonempty M1 dimensions in {session}.")
            inventory[session] = selected
        if self.mode != "off":
            atomic_json(path, dict(signatures=signatures, sessions=inventory))
        return inventory

    def load(self, session: str) -> FeatureSet:
        """Load or validate cached aligned arrays for one real session."""
        path = (self.root / f"{session}.mat").resolve()
        stat = path.stat()
        signature = dict(
            path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns
        )
        source_index = (
            self.cache_root / "sources" / (fingerprint(str(path)) + ".json")
        )
        previous = (
            json.loads(source_index.read_text())
            if source_index.exists()
            else {}
        )
        if previous.get("signature") == signature:
            digest = previous["sha256"]
        else:
            digest = file_digest(path)
            if self.mode != "off":
                atomic_json(
                    source_index, dict(signature=signature, sha256=digest)
                )
        identity = dict(
            source=signature,
            sha256=digest,
            version=PREPROCESSING_VERSION,
            implementation_sha256=file_digest(Path(__file__)),
            sample_rate=self.settings["sampling_rate_hz"],
            selection="first_nonempty_M1",
            alignment="bin_end_linear_position_previous_target",
        )
        return cached(
            self.cache_root,
            "session",
            identity,
            self.mode,
            lambda: self._read(path, identity),
        )

    def fold(
        self, features: FeatureSet, fold: int, velocity: bool
    ) -> FeatureSet:
        """Reuse split-local preprocessing independently of model settings."""
        identity = dict(
            source=features.metadata["identity"],
            fold=fold,
            velocity=velocity,
            cv=self.settings["cv"],
            smoothing=self.settings.get("spike_smoothing"),
            version=PREPROCESSING_VERSION,
        )
        return cached(
            self.cache_root,
            "fold",
            identity,
            self.mode,
            lambda: fold_features(features, self.settings, fold, velocity),
        )

    def _read(self, path: Path, identity: dict) -> FeatureSet:
        """Map MATLAB references and documented cursor/fingertip coordinates."""
        with h5py.File(path, "r") as data:
            names = [
                "".join(chr(int(c)) for c in data[ref][()].ravel())
                for ref in data["chan_names"][()].ravel()
            ]
            time = data["t"][()].ravel()
            position = data["cursor_pos"][()].T
            target = data["target_pos"][()].T
            if not np.allclose(np.diff(time), 0.004, atol=1e-7, rtol=0):
                raise ValueError(f"Expected a regular 250-Hz stream: {path}")
            if position.shape != (len(time), 2):
                raise ValueError(f"Unexpected position shape {position.shape}.")
            interval = 1 / self.settings["sampling_rate_hz"]
            bins = int(np.floor((time[-1] - time[0]) / interval + 1e-7))
            edges = time[0] + np.arange(bins + 1) * interval
            grid = edges[1:]
            ids, units, events, counts = [], [], [], []
            for channel, name in enumerate(names):
                if not name.startswith("M1 "):
                    continue
                for unit in range(data["spikes"].shape[0]):
                    cell = data[data["spikes"][unit, channel]]
                    if cell.attrs.get("MATLAB_empty", False) or not cell.size:
                        continue
                    spikes = cell[()].ravel()
                    if not np.isfinite(spikes).all():
                        raise ValueError(f"Nonfinite spike times: {name}.")
                    spikes = np.sort(spikes)
                    ids.append(name)
                    units.append(unit)
                    events.append(spikes)
                    counts.append(
                        np.diff(np.searchsorted(spikes, edges, side="left"))
                    )
                    break
            if not ids:
                raise ValueError(f"No nonempty M1 spike dimensions: {path}")
            arrays = dict(
                Y=np.array(counts, dtype=float).T,
                Z=np.column_stack(
                    [
                        np.interp(grid, time, position[:, axis])
                        for axis in range(2)
                    ]
                ),
                U=target[np.searchsorted(time, grid, side="right") - 1],
                t=grid,
                ids=np.array(ids),
                units=np.array(units),
                native_t=time,
                native_Z=position,
                native_U=target,
                spike_values=np.concatenate(events),
                spike_offsets=np.r_[0, np.cumsum([len(e) for e in events])],
            )
        aligned(arrays, interval)
        metadata = dict(
            identity=identity,
            session=path.stem,
            source=str(path),
            raw_channel_ids=names,
            selected_channel_ids=ids,
            spike_dimensions=units,
            behavior_fields=["cursor_pos.x", "cursor_pos.y"],
            input_fields=["target_pos.x", "target_pos.y"],
            position_mapping="documented task-plane fingertip in mm",
            dataset_record=self.records["id"],
            deviations=[
                "symmetric offline smoothing",
                "provisional guarded temporal CV",
            ],
        )
        LOGGER.info("Loaded %s: %d samples, %d M1 units", path, bins, len(ids))
        return FeatureSet(arrays, metadata)
