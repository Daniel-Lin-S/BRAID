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
STAGES = ("fit", "preprocess", "evaluate", "plot")


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
    parser.add_argument(
        "--parallel-workers", type=int,
        help="Concurrent fitting sessions on distinct GPUs (default: 1)",
    )
    parser.add_argument("--cpu-threads", type=int)
    parser.add_argument("--cpu-interop-threads", type=int)
    parser.add_argument(
        "--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    parser.add_argument("--cache-mode", choices=["reuse", "rebuild", "off"])
    parser.add_argument("--no-plots", action="store_true")
    previews = parser.add_mutually_exclusive_group()
    previews.add_argument(
        "--previews", dest="previews", action="store_true",
        help="Render selected data previews instead of analysis figures",
    )
    previews.add_argument(
        "--no-previews", dest="previews", action="store_false",
        help="Disable configured preview rendering for this invocation",
    )
    parser.set_defaults(previews=None)
    parser.add_argument("--preview-session", action="append")
    parser.add_argument("--preview-fold", type=int, action="append")
    parser.add_argument("--preview-case", action="append")
    parser.add_argument("--tensorboard", action="store_true", default=None)
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
    plotting_overrides = experiment.pop("plotting_overrides", {})
    if not isinstance(plotting_overrides, dict):
        raise ValueError("plotting_overrides must be a mapping.")
    for module in MODULES:
        source = (path.parent / experiment["modules"][module]).resolve()
        resolved[module] = read_yaml(source)
        if module == "plotting":
            resolved[module] = merge_settings(
                resolved[module], plotting_overrides
            )
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
    previews = resolved["plotting"].get("previews")
    if not isinstance(previews, dict):
        raise ValueError("plotting.previews must be a mapping.")
    selection = previews.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("plotting.previews.selection must be a mapping.")
    expected = {"sessions", "folds", "cases"}
    if set(selection) != expected:
        raise ValueError(
            "plotting.previews.selection must contain sessions, folds, "
            "and cases."
        )
    for key, values in selection.items():
        if values is None:
            continue
        expected_type = int if key == "folds" else str
        if (
            not isinstance(values, list)
            or any(type(value) is not expected_type for value in values)
            or len(set(values)) != len(values)
        ):
            raise ValueError(
                f"plotting.previews.selection.{key} must be a unique list "
                f"of {expected_type.__name__} values or null."
            )
    for key in ("windows", "channels", "seed"):
        value = previews.get(key)
        minimum = 0 if key == "seed" else 1
        if type(value) is not int or value < minimum:
            raise ValueError(
                f"plotting.previews.{key} must be an integer >= {minimum}."
            )
    seconds = previews.get("seconds")
    if (
        isinstance(seconds, bool)
        or not isinstance(seconds, (int, float))
        or seconds <= 0
    ):
        raise ValueError("plotting.previews.seconds must be positive.")
    if type(previews.get("enabled")) is not bool:
        raise ValueError("plotting.previews.enabled must be Boolean.")
    if arguments.previews and arguments.stage != "plot":
        raise ValueError("--previews is only valid with --stage plot.")
    overrides = {
        "sessions": arguments.preview_session,
        "folds": arguments.preview_fold,
        "cases": arguments.preview_case,
    }
    for key, values in overrides.items():
        if values is not None:
            selection[key] = values
    if arguments.previews is not None:
        previews["enabled"] = arguments.previews
    selectors = any(values is not None for values in overrides.values())
    if selectors and arguments.stage != "plot":
        raise ValueError(
            "Preview selectors are only valid with --stage plot."
        )
    if selectors and not previews["enabled"]:
        raise ValueError(
            "Preview selectors require --previews or "
            "plotting.previews.enabled=true."
        )
    if previews["enabled"] and not selection["sessions"]:
        raise ValueError(
            "Preview rendering requires at least one selected session."
        )
    data.update(
        root=paths["dataset_root"],
        cache_root=paths["cache_root"],
    )
    if arguments.cache_mode:
        data["cache_mode"] = arguments.cache_mode
    if arguments.no_plots:
        resolved["plotting"]["enabled"] = False
    for key in ("device", "log_level", "cpu_threads", "cpu_interop_threads"):
        value = getattr(arguments, key)
        if value is not None:
            runtime[key] = value
    if runtime["log_level"] not in ("DEBUG", "INFO", "WARNING", "ERROR"):
        raise ValueError("Unsupported runtime.log_level.")
    workers = getattr(arguments, "parallel_workers", None)
    if workers is not None:
        runtime["parallel_workers"] = workers
    runtime.setdefault("parallel_workers", 1)
    if (type(runtime["parallel_workers"]) is not int
            or runtime["parallel_workers"] < 1):
        raise ValueError("runtime.parallel_workers must be positive integer.")
    if runtime["parallel_workers"] > 1 and runtime["device"] != "auto":
        raise ValueError("Multiple workers require device=auto.")
    runtime.setdefault("cpu_interop_threads", 1)
    runtime.setdefault("figure_regeneration", "incomplete")
    if runtime["figure_regeneration"] not in ("incomplete", "all"):
        raise ValueError(
            "runtime.figure_regeneration must be incomplete or all."
        )
    runtime.setdefault("preview_regeneration", "incomplete")
    if runtime["preview_regeneration"] not in ("incomplete", "all"):
        raise ValueError(
            "runtime.preview_regeneration must be incomplete or all."
        )
    if arguments.tensorboard:
        runtime["tensorboard"] = True
    runtime.setdefault("tensorboard", False)
    if type(runtime["tensorboard"]) is not bool:
        raise ValueError("runtime.tensorboard must be Boolean.")
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
