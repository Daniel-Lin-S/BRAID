"""Expand BRAID structural cases and configure their native mappings.

Input is the experiment suite mapping and the base BRAID model YAML. Output is
one case per dynamics/encoder/decoder combination; each case carries its
effective structure in the analysis manifest and fit identity.
"""

from copy import deepcopy

from .braid_backend import build_cases as build_latent_cases

DYNAMICS = ("linear_mlp", "nonlinear_mlp", "lstm")
EXECUTION_ORDER = ("lstm", "linear_mlp", "nonlinear_mlp")
MAPPING_LEVELS = ("linear", "nonlinear")
STAGES = ("stage_1", "stage_2")
DECODERS = ("neural_decoder", "behaviour_decoder")
LINEAR_ACTIVATION = "linear"
NONLINEAR_ACTIVATION = "relu"
NONLINEAR_DEPTH = 1


def build_cases(settings: dict) -> list[dict]:
    """Return the configured Cartesian product of BRAID structures.

    Parameters
    ----------
    settings : dict
        Latent/population grid and positive nonlinear_hidden_size.

    Returns
    -------
    list of dict
        Named cases with dimensions and structural choices.
    """
    width = settings.get("nonlinear_hidden_size")
    if type(width) is not int or width < 1:
        raise ValueError(
            "Expected positive integer nonlinear_hidden_size, "
            f"got {width!r}."
        )
    latent_cases = build_latent_cases(settings)
    cases = []
    for latent in latent_cases:
        for dynamics in EXECUTION_ORDER:
            for encoder in MAPPING_LEVELS:
                for decoder in MAPPING_LEVELS:
                    case = deepcopy(latent)
                    case["name"] = (
                        f"{latent['name']}_{dynamics}_{encoder}_{decoder}"
                    )
                    case["structure"] = dict(
                        dynamics=dynamics, encoder=encoder,
                        decoder=decoder, nonlinear_hidden_size=width,
                    )
                    case["summary_parameters"].update(
                        dynamics=dynamics, encoder=encoder,
                        decoder=decoder,
                    )
                    cases.append(case)
    return cases


def _set_mlp(mapping: dict, nonlinear: bool, width: int) -> None:
    """Set a linear or one-hidden-layer ReLU native mapping in place."""
    mapping.update(
        hidden_size=width if nonlinear else None,
        depth=NONLINEAR_DEPTH if nonlinear else None,
        activation=(
            NONLINEAR_ACTIVATION if nonlinear else LINEAR_ACTIVATION
        ),
        output_activation=LINEAR_ACTIVATION,
        use_bias=nonlinear,
    )


def apply_structure(configuration: dict, structure: dict) -> None:
    """Apply one structural choice to both stages of model YAML in place.

    Parameters
    ----------
    configuration : dict
        Independent resolved BRAID model YAML mapping.
    structure : dict
        Dynamics, encoder, decoder and nonlinear hidden width.
    """
    expected = {
        "dynamics", "encoder", "decoder", "nonlinear_hidden_size",
    }
    if set(structure) != expected:
        raise ValueError(
            f"Expected structure keys {sorted(expected)}, "
            f"got {sorted(structure)}."
        )
    dynamics = structure["dynamics"]
    encoder = structure["encoder"]
    decoder = structure["decoder"]
    width = structure["nonlinear_hidden_size"]
    if (
        dynamics not in DYNAMICS
        or encoder not in MAPPING_LEVELS
        or decoder not in MAPPING_LEVELS
        or type(width) is not int or width < 1
    ):
        raise ValueError(f"Invalid BRAID structure: {structure!r}.")
    for stage_name in STAGES:
        stage = configuration["model"][stage_name]
        if dynamics == "lstm":
            stage["state_transition"] = {"architecture": "lstm"}
        else:
            transition = stage["state_transition"]
            transition["architecture"] = "multilayer_perceptron"
            _set_mlp(transition, dynamics == "nonlinear_mlp", width)
        _set_mlp(stage["input_mapping"], encoder == "nonlinear", width)
        for name in DECODERS:
            _set_mlp(stage[name], decoder == "nonlinear", width)
