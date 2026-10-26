"""Audit fixed source holdout and matched target folds before planning.

Inputs are the active NHP adapter and immutable source/target reference plans.
The audit reads completed selection inventories and checks today's guarded
windows against their exact session/fold/fraction identities. No artifacts
are written.
"""

import json

import numpy as np

from experiments.datasets.nhp import TEST
from experiments.datasets.windows import window_indices
from experiments.multisession.behavior_compute_scaling.plan import (
    FOLDS, FRACTIONS, SEED, target_templates,
)
from experiments.multisession.data import (
    LENGTH, session_windows, source_features, target_selection,
)
from experiments.multisession.store import completed, fit_directory

COMMON_SESSION_COUNT = 4
COMMON_WINDOW_COUNT = 87


def audit_source_holdout(adapter: object, reference: dict) -> list[str]:
    """Check every source count against the same current 87 windows."""
    common = None
    inventories = {}
    for key in reference["source_references"]:
        spec = reference["jobs"][key]
        names = spec["sessions"][:COMMON_SESSION_COUNT]
        directory = fit_directory(reference, key)
        if completed(directory) is None:
            raise ValueError(
                f"Incomplete reference source: {directory.resolve()}."
            )
        saved = json.loads((directory / "data.json").read_text())
        current = {
            name: saved[name]["validation"] for name in names
        }
        if common is None:
            common = names
            inventories = current
        elif names != common or current != inventories:
            raise ValueError(
                "Reference source holdout differs between counts."
            )
    if common is None or sum(
        len(value) for value in inventories.values()
    ) != COMMON_WINDOW_COUNT:
        raise ValueError("Expected 87 common source test windows.")
    for name in common:
        features = source_features(
            adapter.load(name), reference["settings"]["data"]
        )
        _, _, record = session_windows(features, True)
        if record["validation"] != inventories[name]:
            raise ValueError(
                f"Current source test windows changed: {name}."
            )
    return common


def audit_target_folds(
    adapter: object, reference: dict, held_out: list[str],
) -> int:
    """Check all target train/validation selections and held-out tests."""
    templates = target_templates(reference, held_out)
    if set(fold for _, fold, _ in templates) != set(FOLDS):
        raise ValueError("Target fold coverage differs.")
    if set(part for _, _, part in templates) != set(FRACTIONS):
        raise ValueError("Target fraction coverage differs.")
    evaluated = {}
    for session, fold, fraction in sorted(templates):
        raw = adapter.load(session)
        key = session, fold
        if key not in evaluated:
            features = adapter.fold(raw, fold, True)
            tested = features.arrays["indices"][
                window_indices(features, TEST, LENGTH)
            ]
            if not len(tested):
                raise ValueError(
                    f"No held-out target test windows: {session} fold {fold}."
                )
            evaluated[key] = features, np.unique(tested)
        features, tested = evaluated[key]
        chosen_train, chosen_val, _ = target_selection(
            features, len(raw.arrays["t"]), fraction, SEED
        )
        actual = dict(
            training=features.arrays["indices"][chosen_train],
            validation=features.arrays["indices"][chosen_val],
        )
        old_key, _ = templates[session, fold, fraction]["preprocessing"]
        directory = fit_directory(reference, old_key)
        if completed(directory) is None:
            raise ValueError(
                f"Incomplete target selection: {directory.resolve()}."
            )
        with np.load(directory / "selection.npz") as saved:
            for role, values in actual.items():
                if not np.array_equal(values, saved[role]):
                    raise ValueError(
                        f"Target {role} changed: {session} fold {fold}, "
                        f"fraction {fraction}."
                    )
                if np.intersect1d(values, tested).size:
                    raise ValueError(
                        f"Target {role} overlaps test: {session} fold {fold}."
                    )
    return len(templates)
