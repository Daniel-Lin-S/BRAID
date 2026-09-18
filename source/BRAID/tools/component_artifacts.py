"""Name fitted BRAID components by role and execution order.

Input is a MainModel's resolved dimensions and enabled decoder settings.
Output maps internal component identifiers to numbered semantic directories.
Each directory has component.json describing the internal identifier, inputs,
training target and latent role. Numbers indicate fit order within the group;
TensorBoard timestamps and serialized mathematical attributes are unchanged.
"""

import json
from pathlib import Path


def component_paths(model: object, forward: bool) -> dict[str, str]:
    """Create named component directories in actual fitting order.

    Parameters
    ----------
    model : MainModel
        Resolved dimensions, decoder flags, log_dir and optional artifact_role.
    forward : bool
        Whether separate no-feedthrough forward decoders are required.

    Returns
    -------
    dict of str
        Internal identifiers mapped to absolute log directories, or empty
        strings when component artifacts are disabled.
    """
    role = getattr(model, "artifact_role", "main")
    entries = []
    if model.n1 > 0:
        label = (
            "non_neural_behaviour_dynamics"
            if role == "input_only"
            else "behaviour_relevant_neural_dynamics"
        )
        entries.append(
            (
                "RNN1",
                label,
                "u" if role == "input_only" else "y,u",
                "behaviour residual"
                if role == "input_only"
                else "learned neural-related behaviour"
                if role == "main"
                else "behaviour",
                "x3" if role == "input_only" else "x1",
            )
        )
    neural = model.n1 > 0 and (model.n2 > 0 or not model.skip_Cy)
    if neural and not model.model1_Cy_Full:
        entries.append(
            (
                "Cy1",
                "neural_decoder",
                "x1,u" if model.has_UFT_reg else "x1",
                "y",
                "fixed x1",
            )
        )
        if forward:
            entries.append(
                ("Cy1_fw", "neural_decoder_forward", "x1", "y", "fixed x1")
            )
    if model.n2 > 0:
        pre = role == "behaviour_preprocess"
        entries.append(
            (
                "RNN2",
                "neural_dynamics" if pre else "residual_neural_dynamics",
                "y,u" if model.n1 == 0 else "x1,y,u",
                "y" if model.n1 == 0 else "y residual after neural_decoder",
                "x0" if pre else "x2",
            )
        )
    behaviour = model.nz > 0 and (
        model.model2_Cz_Full
        or model.n1 == 0
        or (model.n1 > 0 and model.n2 > 0 and model.allow_nonzero_Cz2)
    )
    if behaviour:
        label = (
            "full_state_behaviour_decoder"
            if model.model2_Cz_Full
            else "behaviour_decoder"
            if model.n1 == 0
            else "residual_behaviour_decoder"
        )
        inputs = (
            "x1,x2"
            if model.model2_Cz_Full
            else "x0"
            if role == "behaviour_preprocess"
            else "x2"
        )
        if model.has_Dyz and (model.n1 == 0 or model.model2_Cz_Full):
            inputs += ",y"
        if model.has_UFT_reg:
            inputs += ",u"
        target = (
            "behaviour"
            if model.n1 == 0 or model.model2_Cz_Full
            else "behaviour residual after RNN1"
        )
        entries.append(("Cz2", label, inputs, target, "fixed latent state"))
        if forward:
            entries.append(
                (
                    "Cz2_fw",
                    label + "_forward",
                    inputs.replace(",y", "").replace(",u", ""),
                    target,
                    "fixed latent state",
                )
            )
    if model.model1_Cy_Full:
        entries.append(
            (
                "Cy1",
                "full_state_neural_decoder",
                "x1,x2,u" if model.has_UFT else "x1,x2",
                "y",
                "fixed x1,x2",
            )
        )
        if forward:
            entries.append(
                (
                    "Cy1_fw",
                    "full_state_neural_decoder_forward",
                    "x1,x2",
                    "y",
                    "fixed x1,x2",
                )
            )
    paths = {}
    for order, (internal, name, inputs, target, latent) in enumerate(
        entries, 1
    ):
        if not model.log_dir:
            paths[internal] = ""
            continue
        directory = Path(model.log_dir).resolve() / f"{order:02d}_{name}"
        directory.mkdir(parents=True, exist_ok=True)
        metadata = dict(
            order=order,
            role=role,
            name=name,
            internal=internal,
            inputs=inputs,
            target=target,
            latent=latent,
        )
        path = directory / "component.json"
        if path.exists():
            if json.loads(path.read_text()) != metadata:
                raise ValueError(f"Incompatible component metadata: {path}")
        else:
            path.write_text(json.dumps(metadata, indent=2) + "\n")
        paths[internal] = str(directory)
    return paths
