"""Resolve training populations and the experiment's shared scoring channels.

Inputs are ordered channel IDs and every configured case, including cases
not yet fitted. Outputs retain channel order and use actual set intersection.
"""

import numpy as np

POPULATION_SELECTION_POLICY = "seeded_shared_id_prefix_v1"


def population_columns(
    ids: list[str], shared_order: list[str], scale: float,
) -> np.ndarray:
    """Select an ordered seeded prefix, returning neural column indices."""
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Expected nonempty, unique session channel IDs.")
    if len(set(shared_order)) != len(shared_order):
        raise ValueError("Population ordering contains duplicate channel IDs.")
    if not 0 < scale <= 1:
        raise ValueError(f"Expected population scale in (0, 1], got {scale}.")
    mapping = {name: number for number, name in enumerate(ids)}
    ordered = [mapping[name] for name in shared_order if name in mapping]
    if len(ordered) != len(ids):
        raise ValueError("Population ordering is missing session channel IDs.")
    count = int(np.floor(len(ordered) * scale))
    if count < 1:
        raise ValueError(f"Population scale {scale} selects no channels.")
    return np.asarray(ordered[:count], dtype=int)


def common_channels(groups: list[list[str]]) -> list[str]:
    """Return the largest intersection in the first population's order."""
    if not groups or any(not group for group in groups):
        raise ValueError("Cannot intersect empty neural populations.")
    if any(len(set(group)) != len(group) for group in groups):
        raise ValueError("Neural populations contain duplicate channel IDs.")
    common = set.intersection(*(set(group) for group in groups))
    if not common:
        raise ValueError("Configured neural populations have no common IDs.")
    return [name for name in groups[0] if name in common]


def scoring_channels(
    ids: list[str], shared_order: list[str], cases: list[dict],
) -> list[str] | None:
    """Resolve common IDs from all configured scales, or full-only scoring."""
    scales = sorted({case["population_scale"] for case in cases})
    if not scales:
        raise ValueError("Expected at least one configured population.")
    groups = [
        [ids[index] for index in population_columns(ids, shared_order, scale)]
        for scale in scales
    ]
    return common_channels(groups) if len(scales) > 1 else None


def scoring_indices(
    selected_ids: list[str], common: list[str] | None,
) -> np.ndarray | None:
    """Map scoring channel IDs into a fit's output-column order."""
    if common is None:
        return None
    common_channels([selected_ids, common])
    missing = set(common) - set(selected_ids)
    if missing:
        raise ValueError(f"Fit outputs are missing common channels: {missing}")
    return np.asarray([selected_ids.index(name) for name in common])
