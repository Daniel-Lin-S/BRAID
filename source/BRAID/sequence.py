"""Index independent sequences without borrowing samples across boundaries.

Inputs are concatenated dimension-by-time arrays. Outputs are flat source
indices and masks, or aligned arrays preserving the caller's orientation.
"""

from typing import Callable, Sequence

import numpy as np


def independent_indices(
    samples: int,
    length: int,
    horizons: Sequence[int],
    batch_size: int,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Return input/target indices and boundary masks for full batches.

    Parameters
    ----------
    samples : int
        Length of concatenated equal-length sequences.
    length : int
        Samples per independently initialized sequence.
    horizons : sequence of int
        Positive forecast offsets smaller than the sequence length.
    batch_size : int
        Number of sequences per batch.

    Returns
    -------
    inputs : ndarray, shape (N,)
        Flattened input indices for complete batches.
    targets, masks : list of ndarray, each shape (N,)
        Safe target indices and invalid boundary masks for each horizon.
    """
    if samples % length or not horizons or min(horizons) < 1:
        raise ValueError("Expected complete windows and positive horizons.")
    if max(horizons) >= length or batch_size < 1:
        raise ValueError("Horizons must be shorter than a training window.")
    count = samples // length // batch_size * batch_size * length
    if not count:
        raise ValueError("No complete training batch remains.")
    inputs = np.arange(count)
    ends = (inputs // length + 1) * length
    targets = [np.minimum(inputs + h, ends - 1) for h in horizons]
    masks = [inputs + h >= ends for h in horizons]
    return inputs, targets, masks


def window_shift(
    shift: Callable,
    values: np.ndarray | Sequence[np.ndarray] | None,
    horizons: Sequence[int],
    length: int | None,
    *args,
    **kwargs,
) -> tuple[np.ndarray, ...] | None:
    """Apply an existing forecast alignment separately inside each window.

    Parameters
    ----------
    shift : callable
        Existing alignment implementation.
    values : ndarray or sequence of ndarray
        Forecast arrays with time first or last, selected by ``time_first``.
    horizons : sequence of int
        Forecast offsets.
    length : int or None
        Independent window length; None preserves legacy continuous behavior.

    Returns
    -------
    tuple of ndarray
        Aligned predictions with boundary padding confined to each window.
    """
    if length is None or values is None:
        return shift(values, horizons, *args, **kwargs)
    arrays = values if isinstance(values, (list, tuple)) else [values]
    axis = 0 if kwargs.get("time_first", True) else 1
    count = arrays[0].shape[axis]
    if count % length:
        raise ValueError(f"Expected a multiple of {length}, got {count}.")
    pieces = []
    for start in range(0, count, length):
        indices = np.arange(start, start + length)
        chunk = [np.take(a, indices, axis=axis) for a in arrays]
        value = chunk if isinstance(values, (list, tuple)) else chunk[0]
        pieces.append(shift(value, horizons, *args, **kwargs))
    return tuple(np.concatenate(parts, axis=axis) for parts in zip(*pieces))
