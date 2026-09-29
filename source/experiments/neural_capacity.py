"""Expand neural capacity experiments into reusable fit cases.

Input is the suite mapping with latent_splits, population_scales, n_pre and
decoder_widths. Output is a list of named fit cases, with split and decoder
metadata for analysis tables and figures.
"""

import copy

from .braid_backend import build_cases

REFERENCE_DECODER_WIDTH = 64
DECODER_SPLIT = (32, 32)


def build_capacity_cases(settings: dict) -> list[dict]:
    """Expand latent splits and neural decoder widths.

    Parameters
    ----------
    settings : dict
        Explicit latent_splits, population_scales, n_pre and decoder_widths.

    Returns
    -------
    list of dict
        Cases with unique names, dimensions and summary parameters.
    """
    widths = settings["decoder_widths"]
    if (
        not isinstance(widths, list)
        or not widths
        or any(type(width) is not int or width < 1 for width in widths)
        or len(set(widths)) != len(widths)
        or REFERENCE_DECODER_WIDTH not in widths
    ):
        raise ValueError(
            "decoder_widths must be unique positive integers including 64."
        )
    base_settings = {
        key: settings[key]
        for key in ("latent_splits", "population_scales", "n_pre")
    }
    cases = []
    found_decoder_split = False
    for base in build_cases(base_settings):
        n1 = base["summary_parameters"]["n1"]
        n2 = base["summary_parameters"]["n2"]
        selected_widths = (
            widths if (n1, n2) == DECODER_SPLIT
            else [REFERENCE_DECODER_WIDTH]
        )
        found_decoder_split |= (n1, n2) == DECODER_SPLIT
        for width in selected_widths:
            case = copy.deepcopy(base)
            if width != REFERENCE_DECODER_WIDTH:
                case["name"] += f"_decoder{width}"
                case["neural_decoder_width"] = width
            case["summary_parameters"].update(
                split_label=f"n1={n1}, n2={n2}",
                neural_decoder_width=width,
            )
            cases.append(case)
    if not found_decoder_split:
        raise ValueError(
            "latent_splits must include n1=32, n2=32 for decoder widths."
        )
    if len({case["name"] for case in cases}) != len(cases):
        raise ValueError("Capacity cases have duplicate names.")
    return cases
