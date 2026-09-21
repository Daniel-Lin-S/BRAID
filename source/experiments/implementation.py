"""Fingerprint fitting, inference and analysis implementations independently.

Function-level adapter signatures prevent scoring/report changes from
invalidating trained checkpoints. Scientific model and preprocessing source
versions still participate in fit identity.
"""

import ast
from importlib.metadata import version
from pathlib import Path
import platform

from .cache import file_digest, fingerprint

SOURCE = Path(__file__).resolve().parents[1]
FIT_ADAPTER_METHODS = {"resolve_fit_configuration", "fit", "save"}
# Only certified numerical-equivalence pairs permit immutable bundle reuse.
FITTING_REUSE = frozenset({
    (
        "e457f919edcfc3a49358b5ca1d617bc5ee3aae5fbb3330c3382f338954446f4f",
        "338ef3f4781672973338c3fa4ef934d2400a43b99d85d98d81b8be0ee23530b9",
    ),
    (
        "338ef3f4781672973338c3fa4ef934d2400a43b99d85d98d81b8be0ee23530b9",
        "042c67c49a52ec50c90bca7dee26b046cd996ef50b04f27a4403bb852467624d",
    ),
})
INFERENCE_REUSE = frozenset({
    (
        "5de680b8a69abb410057b53c555c1601eeb91feb83b5abfba03da1c450cb88a5",
        "22ddcba8335dd9dae6106b7757cd21559ef7862a6e7e0397c101a909b20f91af",
    ),
    (
        "22ddcba8335dd9dae6106b7757cd21559ef7862a6e7e0397c101a909b20f91af",
        "93dae4befe7f5e019312b38e868224c3dd3c219a808c585e25e5e2635cf22c0f",
    ),
    (
        "93dae4befe7f5e019312b38e868224c3dd3c219a808c585e25e5e2635cf22c0f",
        "d19d145d1d9596783bda90d19a3cd57d7037e9d019c322c94efdb5bfaac34e1f",
    ),
})


def scientific_versions() -> dict:
    """Read numerical dependency versions without initializing a model."""
    return dict(
        python=platform.python_version(),
        packages={
            name: version(name)
            for name in (
                "tensorflow", "tf-keras", "numpy", "scipy", "h5py",
                "scikit-learn", "PyYAML",
            )
        },
    )


def implementation_signatures() -> dict[str, str]:
    """Return scientific implementation signatures for each artifact owner."""
    adapter = SOURCE / "experiments" / "braid_backend.py"
    tree = ast.parse(adapter.read_text())
    methods = {
        node.name: ast.dump(node, include_attributes=False)
        for cls in tree.body if isinstance(cls, ast.ClassDef)
        for node in cls.body if isinstance(node, ast.FunctionDef)
    }
    model = {
        str(path.relative_to(SOURCE)): file_digest(path)
        for path in sorted((SOURCE / "BRAID").rglob("*.py"))
    }
    fitting = dict(
        model=model,
        adapter={name: methods[name] for name in FIT_ADAPTER_METHODS},
        window_selection=file_digest(SOURCE / "experiments" / "windows.py"),
        constants=[
            ast.dump(node, include_attributes=False)
            for node in tree.body if isinstance(node, ast.Assign)
        ],
    )
    prediction_tree = ast.parse(
        (SOURCE / "experiments" / "fitting.py").read_text()
    )
    inference = dict(
        model=model, predict=methods["predict"],
        inputs=[
            ast.dump(node, include_attributes=False)
            for node in prediction_tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "prediction_arrays"
        ],
        window_selection=fitting["window_selection"],
    )
    analysis = {
        name: file_digest(SOURCE / "experiments" / name)
        for name in (
            "evaluation.py", "populations.py",
        )
    }
    return dict(
        fitting_implementation=fingerprint(fitting),
        inference_implementation=fingerprint(inference),
        analysis_implementation=fingerprint(analysis),
    )


def _compatible_transition(
    recorded: str | None,
    expected: str | None,
    transitions: frozenset[tuple[str, str]],
) -> bool:
    """Return whether certified forward transitions connect fingerprints."""
    if recorded == expected:
        return True
    if recorded is None or expected is None:
        return False
    pending = [recorded]
    visited = {recorded}
    while pending:
        current = pending.pop()
        for predecessor, successor in transitions:
            if predecessor != current or successor in visited:
                continue
            if successor == expected:
                return True
            visited.add(successor)
            pending.append(successor)
    return False


def compatible_fitting(recorded: str | None, expected: str | None) -> bool:
    """Return whether a completed fit is numerically reusable."""
    return _compatible_transition(recorded, expected, FITTING_REUSE)


def fitting_predecessors(expected: str | None) -> tuple[str, ...]:
    """Return certified predecessor fingerprints for one implementation."""
    return tuple(sorted(
        recorded for recorded, _ in FITTING_REUSE
        if recorded != expected
        and _compatible_transition(recorded, expected, FITTING_REUSE)
    ))


def compatible_inference(recorded: str | None, expected: str | None) -> bool:
    """Accept only exact or explicitly certified numerical implementations.

    Parameters
    ----------
    recorded, expected : str or None
        Inference fingerprints from the saved bundle and current request.

    Returns
    -------
    bool
        Whether numerical provenance permits reuse without rewriting.
    """
    return _compatible_transition(recorded, expected, INFERENCE_REUSE)
