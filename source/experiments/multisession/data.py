"""Prepare session-isolated data for multi-session training.

Inputs are existing spike FeatureSets and explicit split/subset settings.
Outputs contain float32 window arrays (N, L, C), training-only scaling maps,
and original sample indices (N, L). Source neural-only records omit Z.
Selection records include requested and realized label fractions.
"""

from dataclasses import dataclass
import hashlib
import logging

import numpy as np

from experiments.contracts import FeatureSet
from experiments.datasets.nhp import TRAIN, VALIDATION, TEST
from experiments.datasets.windows import window_indices

LOGGER = logging.getLogger(__name__)
LENGTH = 128
SAMPLE_RATE = 20
NONTEST_FRACTION = 0.8


def stable_seed(*parts: object) -> int:
    """Return a reproducible seed independent of Python hash randomization."""
    value = ":".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:4], "little")


@dataclass
class Scale:
    """Training-only centering, scalar scaling, and active coordinates."""

    mean: np.ndarray
    scale: float
    keep: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray, name: str) -> "Scale":
        """Fit finite (N, L, C) values, rejecting empty/flat signals."""
        flat = values.reshape(-1, values.shape[-1]).astype(float)
        if not len(flat) or not np.isfinite(flat).all():
            raise ValueError(f"{name}: expected nonempty finite values.")
        keep = np.ptp(flat, axis=0) > 0
        if not keep.any():
            raise ValueError(f"{name}: all training coordinates are flat.")
        if not keep.all():
            LOGGER.warning(
                "%s: removing flat coordinates %s",
                name,
                np.flatnonzero(~keep).tolist(),
            )
        if name in ("U", "Z") and not keep.all():
            raise ValueError(
                f"{name}: flat coordinates prevent fixed shared dimensions."
            )
        mean = flat.mean(axis=0)
        # BRAID uses the mean of per-coordinate standard deviations.
        scale = float(np.std(flat[:, keep], axis=0, ddof=1).mean())
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"{name}: invalid training scale {scale}.")
        return cls(mean, scale, keep)

    def apply(self, values: np.ndarray) -> np.ndarray:
        """Transform (..., C) arrays to (..., retained_C) finite values."""
        result = (values - self.mean)[..., self.keep] / self.scale
        if not np.isfinite(result).all():
            raise ValueError("Normalization produced nonfinite values.")
        return result.astype(np.float32)

    def inverse(self, values: np.ndarray) -> np.ndarray:
        """Restore (..., retained_C) arrays to original coordinates."""
        result = np.broadcast_to(
            self.mean, (*values.shape[:-1], len(self.mean))
        ).copy()
        result[..., self.keep] += values * self.scale
        if not np.isfinite(result).all():
            raise ValueError("Inverse normalization is nonfinite.")
        return result.astype(np.float32)

    def record(self) -> dict:
        """Return JSON-compatible fitted statistics."""
        return dict(
            mean=self.mean.tolist(), scale=self.scale, keep=self.keep.tolist()
        )


def source_features(
    session: FeatureSet, settings: dict,
    inner_fraction: float | None = None,
) -> FeatureSet:
    """Create guarded source segments with isolated smoothing.

    Parameters
    ----------
    session : FeatureSet
        Raw time-first arrays, including original sample indices.
    settings : dict
        Guard duration and spike smoothing parameters.
    inner_fraction : float, optional
        Final fraction of outer training reserved for validation;
        default None retains the two-role 80/20 protocol.
    """
    # Five blocks: first four are training, final block is validation.
    from experiments.datasets import nhp

    count = len(session.arrays["t"])
    guard = int(np.ceil(settings["cv"]["guard_seconds"] * SAMPLE_RATE))
    boundary = int(np.floor(count * NONTEST_FRACTION))
    segments = [
        (0, boundary - guard, TRAIN),
        (boundary + guard, count, VALIDATION),
    ]
    if inner_fraction is not None:
        if not 0 < inner_fraction < 1:
            raise ValueError("Inner fraction must lie in (0, 1).")
        train_stop = boundary - guard
        inner = int(np.floor(train_stop * (1 - inner_fraction)))
        segments = [
            (0, inner - guard, TRAIN),
            (inner + guard, train_stop, VALIDATION),
            (boundary + guard, count, TEST),
        ]
    from scipy.ndimage import gaussian_filter1d

    pieces = {
        key: []
        for key in ("Y", "Z", "U", "t", "indices", "segment", "role", "counts")
    }
    smooth = settings["spike_smoothing"]
    for number, (start, stop, role) in enumerate(segments):
        if stop - start <= LENGTH:
            raise ValueError("Source split is shorter than one window.")
        y = session.arrays["Y"][start:stop]
        y = gaussian_filter1d(
            y,
            smooth["sigma_ms"] * SAMPLE_RATE / 1000,
            axis=0,
            mode=smooth["boundary"],
            truncate=smooth["truncate"],
        )
        z = session.arrays["Z"][start:stop]
        z = np.column_stack((z[1:], np.diff(z, axis=0) * SAMPLE_RATE))
        size = stop - start - 1
        values = dict(
            Y=y[1:],
            Z=z,
            U=session.arrays["U"][start + 1 : stop],
            t=session.arrays["t"][start + 1 : stop],
            indices=np.arange(start + 1, stop),
            segment=np.full(size, number),
            role=np.full(size, role),
            counts=session.arrays["Y"][start + 1 : stop],
        )
        nhp.aligned(values, 1 / SAMPLE_RATE)
        for key, value in values.items():
            pieces[key].append(value)
    arrays = {key: np.concatenate(value) for key, value in pieces.items()}
    arrays.update(ids=session.arrays["ids"], units=session.arrays["units"])
    return FeatureSet(arrays, dict(session.metadata, source_split="80/20"))


def target_selection(
    features: FeatureSet,
    sample_count: int,
    fraction: float,
    seed: int,
    length: int = LENGTH,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Select paired nested train/validation windows without test access.

    Parameters
    ----------
    features : FeatureSet
        Existing guarded target-fold arrays.
    sample_count : int
        Original aligned session label count, before guards.
    fraction : float
        Requested fraction of all labels, including validation.
    seed : int
        Repeat seed, paired across models and source counts.
    length : int, optional
        Independent sequence length; default 128.

    Returns
    -------
    training, validation : ndarray, shape (N_role, length)
        Selected indices into features, after native batch trimming.
    record : dict
        Requested/realized counts and effective batch size.
    """
    if not 0 < fraction <= NONTEST_FRACTION:
        raise ValueError(f"Invalid target fraction {fraction}.")
    windows = [window_indices(features, r, length) for r in (TRAIN, VALIDATION)]
    sizes = [len(w) for w in windows]
    total = (
        sum(sizes)
        if fraction == NONTEST_FRACTION
        else min(sum(sizes), int(sample_count * fraction) // length)
    )
    if total < 2:
        raise ValueError("Target budget needs train and validation windows.")
    train = min(
        sizes[0],
        total - 1,
        max(1, int(np.floor(total * sizes[0] / sum(sizes) + 0.5))),
    )
    validation = total - train
    if validation > sizes[1]:
        validation = sizes[1]
        train = total - validation
    rng = np.random.default_rng(
        stable_seed(
            features.metadata["session"],
            features.metadata["fold"],
            seed,
        )
    )
    selected = [
        w[rng.permutation(len(w))[:n]]
        for w, n in zip(windows, (train, validation))
    ]
    batch = min(32, train, validation)
    selected = [w[: len(w) // batch * batch] for w in selected]
    record = dict(
        fraction=fraction,
        denominator=sample_count,
        requested_labels=int(sample_count * fraction),
        training_labels=int(selected[0].size),
        validation_labels=int(selected[1].size),
        batch_size=batch,
        realized_fraction=sum(w.size for w in selected) / sample_count,
    )
    return selected[0], selected[1], record


def session_windows(
    features: FeatureSet,
    supervised: bool,
    include_test: bool = False,
) -> tuple[dict, ...]:
    """Return train-only normalized windows and provenance.

    Parameters
    ----------
    features : FeatureSet
        Time-first arrays with guarded segment and role labels.
    supervised : bool
        Include four-dimensional behavioral targets.
    include_test : bool, optional
        Default False returns train, validation, record. True
        returns train, inner validation, outer test, record.

    Returns
    -------
    tuple of dict
        Data arrays have shape (windows, 128, channels). The record
        contains scales and original sample indices for each role;
        its validation key always identifies the outer holdout.
    """
    roles = (TRAIN, VALIDATION, TEST) if include_test else (
        TRAIN, VALIDATION
    )
    indices = {
        role: window_indices(features, role, LENGTH) for role in roles
    }
    names = ("Y", "U", "Z") if supervised else ("Y", "U")
    for name, width in (("U", 2), ("Z", 4)):
        if name in names and features.arrays[name].shape[1] != width:
            raise ValueError(
                f"{name}: expected {width} task coordinates, got "
                f"{features.arrays[name].shape[1]}."
            )
    prepared = {role: {} for role in roles}
    maps = {}
    for name in names:
        raw = features.arrays[name][indices[TRAIN]]
        scaling = Scale.fit(raw, name)
        for role in roles:
            prepared[role][name] = scaling.apply(
                features.arrays[name][indices[role]]
            )
        maps[name] = scaling.record()
    record = dict(
        scales=maps,
        training=features.arrays["indices"][indices[TRAIN]].tolist(),
        validation=features.arrays["indices"][
            indices[TEST if include_test else VALIDATION]
        ].tolist(),
        channel_ids=features.arrays["ids"].tolist(),
    )
    if include_test:
        record["inner_validation"] = features.arrays["indices"][
            indices[VALIDATION]
        ].tolist()
    return (*[prepared[role] for role in roles], record)
