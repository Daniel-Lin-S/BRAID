"""Construct recorded or neural-generated behavioral targets.

Inputs are three ordered lists of per-session normalized Y/U/Z windows for
training, inner validation, and chronological test. Generated mode also
requires fitted neural states and a behavioral readout. Output is the three
partitions with the selected Z target and optional generated-target scales.
Recorded mode retains the training-normalized observed Z arrays unchanged.
"""

import numpy as np

from experiments.multisession.data import Scale
from experiments.multisession.source import behavior_targets
from experiments.multisession.training import ReadoutStage

RECORDED = "recorded"
GENERATED = "generated"
TARGET_MODES = (RECORDED, GENERATED)


def target_mode(spec: dict) -> str:
    """Resolve the scientific target choice, preserving legacy plans."""
    mode = spec.get("behavior_supervision", {}).get(
        "target", GENERATED
    )
    if mode not in TARGET_MODES:
        raise ValueError(
            f"Expected behavior target in {TARGET_MODES}, got {mode}."
        )
    return mode


def prepare_targets(
    partitions: tuple[list[dict], list[dict], list[dict]],
    mode: str,
    states: list[list[np.ndarray]] | None = None,
    readout: ReadoutStage | None = None,
) -> tuple[list[list[dict]], list[dict] | None]:
    """Prepare Z targets with generated-target scaling fit on train only.

    Parameters
    ----------
    partitions : tuple of three lists of dict
        Train, inner-validation and test windows. Y/U/Z arrays have shape
        (windows, 128, channels) for each recording.
    mode : str
        ``recorded`` or ``generated``.
    states : list of lists of ndarray, optional
        Preprocessing neural states for generated mode, default None.
    readout : ReadoutStage, optional
        Fitted generated-behavior readout, default None.

    Returns
    -------
    prepared : list of lists of dict
        Three partitions with the chosen Z target.
    scales : list of dict or None
        Generated-target training scales, or None for recorded targets.
    """
    if mode not in TARGET_MODES or len(partitions) != 3:
        raise ValueError("Expected three partitions and a valid target mode.")
    if mode == RECORDED:
        if states is not None or readout is not None:
            raise ValueError("Recorded target cannot use preprocessing.")
        return [list(part) for part in partitions], None
    if states is None or readout is None or len(states) != 3:
        raise ValueError(
            "Generated target requires readout and three state partitions."
        )
    prepared = [
        behavior_targets(readout, state, part)
        for state, part in zip(states, partitions)
    ]
    scales = []
    for index, row in enumerate(prepared[0]):
        scale = Scale.fit(row["Z"], "Z")
        for part in prepared:
            part[index]["Z"] = scale.apply(part[index]["Z"])
        scales.append(scale.record())
    return prepared, scales
