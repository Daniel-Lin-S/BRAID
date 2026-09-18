"""Fingerprint fitting, inference and analysis implementations independently.

Function-level adapter signatures prevent scoring/report changes from
invalidating trained checkpoints. Scientific model and preprocessing source
versions still participate in fit identity.
"""

import ast
from pathlib import Path

from .cache import file_digest, fingerprint

SOURCE = Path(__file__).resolve().parents[1]
FIT_ADAPTER_METHODS = {"resolve_fit_configuration", "fit", "save"}


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
            "evaluation.py", "populations.py", "reporting.py", "plots.py",
        )
    }
    return dict(
        fitting_implementation=fingerprint(fitting),
        inference_implementation=fingerprint(inference),
        analysis_implementation=fingerprint(analysis),
    )
