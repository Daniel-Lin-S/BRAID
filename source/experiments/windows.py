"""Build independent training windows from real contiguous feature segments.

Input is a FeatureSet with role and segment labels. Output indices have
shape (number_of_windows, sequence_length) and refer to its time-first arrays.
"""

import numpy as np

from .contracts import FeatureSet


def window_indices(features: FeatureSet, role: int, length: int) -> np.ndarray:
    """Retain complete windows within each segment of the requested role."""
    arrays = features.arrays
    windows = []
    for segment in np.unique(arrays["segment"][arrays["role"] == role]):
        indices = np.flatnonzero(
            (arrays["segment"] == segment) & (arrays["role"] == role)
        )
        count = len(indices) // length * length
        if count:
            windows.append(indices[:count].reshape(-1, length))
    if not windows:
        raise ValueError(
            f"No complete {length}-sample windows for role {role}."
        )
    return np.concatenate(windows)
