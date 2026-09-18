"""Resolve experiment manifests, stage configurations and local settings.

An experiment selects four module YAML files and its own sweep/selection.
Module files may extend another YAML mapping; relative references resolve
against their declaring file. Filename suffixes have no loading semantics.
No loading operation starts a job or creates an output directory.
"""

import argparse
import copy
from pathlib import Path
import os

import yaml

REPOSITORY = Path(__file__).resolve().parents[2]
CONFIGURATION = REPOSITORY / "assets" / "config" / "nhp"
MODULES = ("data", "model", "evaluation", "plotting")
PATH_KEYS = ("dataset_root", "cache_root", "artifact_root", "log_root")
STAGES = ("fit", "preprocess", "preview", "evaluate", "plot")


def merge_settings(base: dict, override: dict) -> dict:
    """Deep-copy mappings with explicit leaf overrides.

    Parameters
    ----------
    base, override : dict
        Defaults and overriding settings; lists replace existing lists.

    Returns
    -------
    dict
        Independent merged configuration.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_settings(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def read_yaml(path: Path, ancestors: tuple[Path, ...] = ()) -> dict:
    """Read a mapping and resolve inheritance without changing source files.

    Parameters
    ----------
    path : Path
        YAML file; inherited paths are relative to this file.
    ancestors : tuple of Path, optional
        Internal cycle detection stack; default is empty.

    Returns
    -------
    dict
        Merged settings with environment variables expanded.
    """
    path = path.expanduser().resolve()
    if path in ancestors:
        raise ValueError(f"Configuration inheritance cycle at {path}")
    text = os.path.expandvars(path.read_text())
    if "${" in text:
        raise ValueError(f"Unresolved environment variable in {path}")
    settings = yaml.safe_load(text)
    if not isinstance(settings, dict) or not settings:
        raise ValueError(f"Expected a nonempty YAML mapping: {path}")
    parents = settings.pop("extends", [])
    if isinstance(parents, str):
        parents = [parents]
    if not isinstance(parents, list) or any(
        not isinstance(parent, str) or not parent for parent in parents
    ):
        raise ValueError(
            f"Expected extends to be a path or list of paths: {path}"
        )
    # Bind module references before merging so inherited paths keep their base.
    if "modules" in settings:
        settings["modules"] = {
            key: str((path.parent / value).expanduser().resolve())
            for key, value in settings["modules"].items()
        }
    base = {}
    for parent in parents:
        base = merge_settings(
            base, read_yaml(path.parent / parent, (*ancestors, path))
        )
    settings = merge_settings(base, settings)
    return settings


def argument_parser() -> argparse.ArgumentParser:
    """Build the shared launcher/runner CLI with explicit experiment choice."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--stage", choices=STAGES, default="fit")
    parser.add_argument("--session", action="append")
    parser.add_argument("--fold", type=int, action="append")
    parser.add_argument(
        "--device", help="auto (most free GPU memory), cpu, GPU index or UUID"
    )
    parser.add_argument("--cpu-threads", type=int)
    parser.add_argument("--cpu-interop-threads", type=int)
    parser.add_argument(
        "--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    parser.add_argument("--cache-mode", choices=["reuse", "rebuild", "off"])
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-previews", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--analysis-id", help="Saved analysis revision for the plot stage"
    )
    return parser


def resolve_configuration(arguments: argparse.Namespace) -> dict:
    """Resolve one experiment independently from machine-specific paths.

    Parameters
    ----------
    arguments : Namespace
        Shared CLI arguments from argument_parser.

    Returns
    -------
    dict
        Experiment, four modules, paths and runtime settings.
    """
    path = arguments.experiment.expanduser().resolve()
    experiment = read_yaml(path)
    if not experiment.get("enabled", True):
        raise ValueError(f"Experiment is deferred: {experiment['reason']}")
    name = experiment["name"]
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise ValueError("Experiment name must be one directory component.")
    paths = experiment.pop("paths", {})
    runtime = experiment.pop("runtime", {})
    for key in PATH_KEYS:
        value = paths.get(key)
        if (
            not isinstance(value, str)
            or not Path(value).expanduser().is_absolute()
        ):
            raise ValueError(
                f"Set an absolute paths.{key} in the experiment configuration."
            )
        paths[key] = str(Path(value).expanduser().resolve())
    resolved = dict(experiment=experiment, paths=paths, runtime=runtime)
    for module in MODULES:
        source = (path.parent / experiment["modules"][module]).resolve()
        resolved[module] = read_yaml(source)
    if "neural_scoring_fraction" in resolved["evaluation"]:
        raise ValueError(
            "evaluation.neural_scoring_fraction is unsupported; common "
            "channels are the intersection of configured populations."
        )
    from .fitting import canonical_horizons

    resolved["evaluation"]["horizons"] = canonical_horizons(
        resolved["evaluation"]["horizons"]
    )
    for curve in resolved["plotting"].get("curves", []):
        scoring_set = curve.get("where", {}).get("evaluation_set")
        if scoring_set is not None and scoring_set not in ("full", "common"):
            raise ValueError(
                f"Expected evaluation_set full or common, got {scoring_set!r}."
            )
    data = resolved["data"]
    if "presentation" in data.get("previews", {}):
        raise ValueError("Configure figure style under plotting.presentation.")
    data.update(
        root=paths["dataset_root"],
        cache_root=paths["cache_root"],
    )
    if arguments.cache_mode:
        data["cache_mode"] = arguments.cache_mode
    if arguments.no_previews:
        data["previews"]["enabled"] = False
    if arguments.no_plots:
        resolved["plotting"]["enabled"] = False
    for key in ("device", "log_level", "cpu_threads", "cpu_interop_threads"):
        value = getattr(arguments, key)
        if value is not None:
            runtime[key] = value
    if runtime["log_level"] not in ("DEBUG", "INFO", "WARNING", "ERROR"):
        raise ValueError("Unsupported runtime.log_level.")
    runtime.setdefault("cpu_interop_threads", 1)
    for key in ("cpu_threads", "cpu_interop_threads"):
        if type(runtime[key]) is not int or runtime[key] < 1:
            raise ValueError(f"runtime.{key} must be a positive integer.")
    device = runtime["device"]
    if not isinstance(device, str) or not (
        device in ("auto", "cpu")
        or device.isdecimal()
        or device.startswith("GPU-")
    ):
        raise ValueError("device must be auto, cpu, a GPU index or GPU UUID.")
    return resolved
