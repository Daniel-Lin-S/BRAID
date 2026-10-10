"""Lazy aliases for saved plugin paths and historical command lines.

Canonical implementations live in responsibility-specific packages. Aliases
return the same module object, so monkeypatches and pickle imports retain their
original behavior. The loader also supports historical ``python -m`` commands.
"""

import importlib
import importlib.abc
import importlib.util
from pathlib import Path
import sys

MODULES = {
    "nhp": "datasets.nhp",
    "nhp_lfp": "datasets.nhp_lfp",
    "lfp_preprocessing": "datasets.lfp_preprocessing",
    "windows": "datasets.windows",
    "populations": "datasets.populations",
    "braid_backend": "models.braid_backend",
    "structure_sweep": "models.structure_sweep",
    "neural_capacity": "models.neural_capacity",
    "fitting": "execution.fitting",
    "parallel": "execution.parallel",
    "runtime": "execution.runtime",
    "runner": "execution.runner",
    "launcher": "execution.launcher",
    "cache": "storage.cache",
    "artifacts": "storage.artifacts",
    "analysis": "storage.analysis",
    "implementation": "storage.implementation",
    "coordination": "storage.coordination",
    "restart": "storage.restart",
    "reporting": "reports.reporting",
    "structure_report": "reports.structure_report",
    "model_summary": "reports.model_summary",
    "evaluation": "reports.evaluation",
    "outliers": "reports.outliers",
    "plots": "visualization.plots",
    "structure_plots": "visualization.structure_plots",
    "presentation": "visualization.presentation",
    "history": "visualization.history",
    "diagnostics": "visualization.diagnostics",
    "previews": "visualization.previews",
    "lfp_previews": "visualization.lfp_previews",
    "preview_rendering": "visualization.preview_rendering",
    "preview_regeneration": "visualization.preview_regeneration",
    "preview_publication": "visualization.preview_publication",
    "signal_previews": "visualization.signal_previews",
}


class LegacyLoader(importlib.abc.Loader):
    """Resolve one historical name without eager experiment imports."""

    def __init__(self, target: str) -> None:
        self.target = target

    def create_module(self, spec):
        """Return the canonical module so aliases share mutable state."""
        return importlib.import_module(self.target)

    def exec_module(self, module) -> None:
        """Canonical import has already executed the module."""

    def get_code(self, fullname: str):
        """Compile canonical source for a historical module command."""
        spec = importlib.util.find_spec(self.target)
        return compile(Path(spec.origin).read_text(), spec.origin, "exec")


class LegacyFinder(importlib.abc.MetaPathFinder):
    """Recognize only documented flat experiment module aliases."""

    def find_spec(self, fullname: str, path=None, target=None):
        """Return a lazy alias spec, leaving all other imports untouched."""
        prefix, _, name = fullname.rpartition(".")
        if prefix == "experiments" and name in MODULES:
            loader = LegacyLoader("experiments." + MODULES[name])
            spec = importlib.util.find_spec(loader.target)
            return importlib.util.spec_from_loader(
                fullname,
                loader,
                origin=spec.origin,
            )
        return None


def install() -> None:
    """Register the compatibility finder once per interpreter."""
    if not any(isinstance(item, LegacyFinder) for item in sys.meta_path):
        sys.meta_path.append(LegacyFinder())
