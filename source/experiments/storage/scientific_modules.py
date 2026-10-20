"""Version scientific modules used by BRAID neural fits.

Input is a resolved BRAID model configuration and a fit role. Output is a
mapping of active numerical modules to explicit implementation versions.
The mapping enters new fit identities; source-file checksums are audit data.
Changing numerical behavior requires a version bump of its affected module.
"""

from experiments.storage.cache import fingerprint


MODULE_VERSIONS = {
    "data.nhp": 1,
    "data.windows": 1,
    "braid.preprocessing": 1,
    "braid.neural_encoder": 1,
    "braid.observation_decoders": 1,
    "braid.dynamics.multilayer_perceptron": 1,
    "braid.dynamics.lstm": 1,
    "braid.dynamics.causal_transformer": 1,
    "braid.training": 1,
    "braid.inference": 1,
    "multisession.selection": 1,
    "multisession.source": 1,
    "multisession.target": 1,
    "multisession.transfer": 1,
}
FIT_ROLES = frozenset({
    "generic", "source", "preprocessing", "scratch", "adaptation",
})
DYNAMICS = frozenset({
    "multilayer_perceptron", "lstm", "causal_transformer",
})


def active_stages(arguments: dict) -> tuple[int, ...]:
    """Return stages with positive latent width in resolved fit arguments.

    Parameters
    ----------
    arguments : dict
        BRAID fit arguments containing nx, n1, and optional n3 dimensions.

    Returns
    -------
    tuple of int
        Stage indices that execute neural state dynamics.
    """
    nx, n1 = arguments.get("nx"), arguments.get("n1")
    n3 = arguments.get("n3")
    if type(nx) is not int or type(n1) is not int:
        raise ValueError(
            f"Expected integer nx and n1, got nx={nx!r}, n1={n1!r}."
        )
    if n3 is not None and type(n3) is not int:
        raise ValueError(f"Expected integer or None n3, got {n3!r}.")
    n3 = 0 if n3 is None else n3
    if nx < 1 or n1 < 0 or n1 > nx or n3 < 0:
        raise ValueError(
            f"Expected nx > 0, 0 <= n1 <= nx, and n3 >= 0; "
            f"got nx={nx}, n1={n1}, n3={n3}."
        )
    result = []
    if n1 or (nx > n1 and n3):
        result.append(1)
    if nx > n1 + n3:
        result.append(2)
    return tuple(result)


def active_dynamics(model: dict, stages: tuple[int, ...]) -> tuple[str, ...]:
    """Select dynamics architectures executed by active BRAID stages.

    Parameters
    ----------
    model : dict
        Resolved model YAML with stage_1 and stage_2 transition settings.
    stages : tuple of int
        Active stage numbers, each either 1 or 2.

    Returns
    -------
    tuple of str
        Unique dynamics module names in sorted order.
    """
    try:
        architectures = {
            model["model"][f"stage_{stage}"]["state_transition"][
                "architecture"
            ]
            for stage in stages
        }
    except (KeyError, TypeError) as error:
        raise ValueError(
            "Missing BRAID stage dynamics configuration."
        ) from error
    if architectures - DYNAMICS:
        raise ValueError(
            f"Unknown BRAID dynamics: {sorted(architectures - DYNAMICS)}."
        )
    return tuple(sorted(architectures))


def fit_module_versions(
    role: str,
    model: dict | None = None,
    stages: tuple[int, ...] = (1, 2),
) -> dict[str, int]:
    """Return only the module versions used by a neural fit.

    Parameters
    ----------
    role : str
        Generic, source, preprocessing, scratch, or adaptation fit.
    model : dict, optional
        Resolved BRAID model YAML; default None for source fits.
    stages : tuple of int, optional
        Active BRAID stages; default (1, 2).

    Returns
    -------
    dict of str to int
        Active module versions used in the canonical fit signature.
    """
    if role not in FIT_ROLES:
        raise ValueError(f"Unknown BRAID fit role: {role}.")
    names = {"data.nhp", "data.windows", "braid.training"}
    if role == "source":
        names.add("multisession.source")
    else:
        if model is None:
            raise ValueError(
                f"{role} fit requires a BRAID model configuration."
            )
        names.update({
            "braid.preprocessing", "braid.neural_encoder",
            "braid.observation_decoders",
        })
        names.update(
            f"braid.dynamics.{name}"
            for name in active_dynamics(model, stages)
        )
        if role != "generic":
            names.update({"multisession.selection", "multisession.target"})
        if role == "adaptation":
            names.add("multisession.transfer")
    return {name: MODULE_VERSIONS[name] for name in sorted(names)}


def fit_module_digest(
    role: str,
    model: dict | None = None,
    stages: tuple[int, ...] = (1, 2),
) -> str:
    """Hash the active module-version vector for a new neural fit."""
    return fingerprint(fit_module_versions(role, model, stages))
