"""Build behavioral source preprocessing, RNN1, readout, and RNN2.

Input is a list of (retained neural channels, behavioral channels) per source
recording and one scientific source specification. Outputs are unfitted model
objects; checkpoint identity and compute schedules belong to the caller.
"""

from BRAID.multisession import MixedForecaster
from experiments.multisession.training import ReadoutStage
from experiments.multisession.source import (
    BEHAVIOR_WIDTH, NEURAL_WIDTH, PREPROCESSING_SIZE,
)

STAGE1_SHARING = "A"
STAGE2_SHARING = "A2_A2_fw_K2_fw"
TASK_WIDTH = 2


def preprocessing_model(
    dimensions: list[tuple[int, int]], spec: dict,
    devices: tuple[str, str] | None = None,
) -> MixedForecaster:
    """Build the generated-target neural preprocessing recurrence."""
    neural = [(width, width) for width, _ in dimensions]
    return MixedForecaster(
        neural, PREPROCESSING_SIZE, "none", spec["seed"],
        session_devices=devices,
    )


def preprocessing_readout_model(
    dimensions: list[tuple[int, int]], spec: dict,
    devices: tuple[str, str] | None = None,
) -> ReadoutStage:
    """Build the generated-behavior readout of neural n0 states."""
    return ReadoutStage(
        [(PREPROCESSING_SIZE, behavior) for _, behavior in dimensions],
        BEHAVIOR_WIDTH, spec["seed"] + 1, behavior=True,
        session_devices=devices,
    )


def rnn1_model(
    dimensions: list[tuple[int, int]], spec: dict,
    devices: tuple[str, str] | None = None,
) -> MixedForecaster:
    """Build RNN1 with one private map set per source recording."""
    return MixedForecaster(
        dimensions, spec["latent"], STAGE1_SHARING,
        spec["seed"] + 2, decoder_width=BEHAVIOR_WIDTH,
        session_devices=devices,
    )


def neural_readout_model(
    dimensions: list[tuple[int, int]], spec: dict,
    devices: tuple[str, str] | None = None,
) -> ReadoutStage:
    """Build the session-private readout of RNN1 latent states."""
    return ReadoutStage(
        [(spec["latent"], neural) for neural, _ in dimensions],
        NEURAL_WIDTH, spec["seed"] + 3,
        session_devices=devices,
    )


def rnn2_model(
    dimensions: list[tuple[int, int]], spec: dict,
    devices: tuple[str, str] | None = None,
) -> MixedForecaster:
    """Build RNN2's shared dynamics and private neural maps."""
    latent = spec["latent"]
    return MixedForecaster(
        [(neural, neural) for neural, _ in dimensions],
        latent, STAGE2_SHARING, spec["seed"] + 4,
        input_dimensions=[
            neural + latent + TASK_WIDTH for neural, _ in dimensions
        ],
        forward_dimensions=latent + TASK_WIDTH,
        session_devices=devices,
    )
