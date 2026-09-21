"""Check portable experiment isolation, private overrides and log routing."""

import logging
from pathlib import Path
import subprocess

import pytest
import yaml

from experiments.configuration import (
    CONFIGURATION,
    REPOSITORY,
    argument_parser,
    read_yaml,
    resolve_configuration,
)
from experiments.runtime import session_logging, launch_directory


def local_settings(tmp_path: Path) -> Path:
    """Write test-owned local paths without embedding machine locations."""
    path = tmp_path / "local.yaml"
    path.write_text(
        yaml.safe_dump(
            dict(
                paths={
                    name: str(tmp_path / name)
                    for name in (
                        "dataset_root",
                        "cache_root",
                        "artifact_root",
                        "log_root",
                    )
                },
                runtime=dict(
                    python=None, device="auto", cpu_threads=1, log_level="INFO"
                ),
            )
        )
    )
    return path


def resolve(name: str, local: Path) -> dict:
    """Resolve one public experiment with test-owned machine settings."""
    experiment = local.parent / name
    experiment.write_text(
        yaml.safe_dump(
            dict(
                extends=[
                    str(CONFIGURATION / "experiments" / name),
                    str(local),
                ]
            )
        )
    )
    args = argument_parser().parse_args(["--experiment", str(experiment)])
    return resolve_configuration(args)


def test_sweeps_have_separate_grids_and_plotting(tmp_path):
    """Sweep definitions cannot leak into evaluation or another experiment."""
    from experiments.braid_backend import build_cases

    local = local_settings(tmp_path)
    latent = resolve("latent_dimension_sweep.yaml", local)
    population = resolve("neural_population_sweep.yaml", local)
    assert len(build_cases(latent["experiment"]["suite"])) == 7
    assert len(build_cases(population["experiment"]["suite"])) == 6
    assert latent["evaluation"] == population["evaluation"]
    assert set(latent["evaluation"]) == {
        "horizons",
    }
    assert latent["data"]["infer_velocity"]
    assert latent["model"]["training"]["training_batch_size"] == 32
    assert latent["model"]["training"]["maximum_epochs"] == 2500
    assert latent["plotting"] != population["plotting"]
    assert latent["runtime"]["figure_regeneration"] == "incomplete"


def test_figure_regeneration_policy_is_validated(tmp_path):
    """Require an explicit missing-only or full component redraw policy."""
    local = local_settings(tmp_path)
    settings = read_yaml(local)
    settings["runtime"]["figure_regeneration"] = "automatic"
    local.write_text(yaml.safe_dump(settings))
    with pytest.raises(ValueError, match="figure_regeneration"):
        resolve("latent_dimension_sweep.yaml", local)


def test_module_inheritance_does_not_mutate_defaults(tmp_path):
    """A private cap and velocity option leave public defaults unchanged."""
    model = tmp_path / "model.yaml"
    parent = CONFIGURATION / "models" / "BRAID.yaml"
    model.write_text(
        yaml.safe_dump(
            dict(
                extends=str(parent),
                training=dict(maximum_epochs=3),
            )
        )
    )
    assert read_yaml(model)["training"]["maximum_epochs"] == 3
    assert read_yaml(parent)["training"]["maximum_epochs"] == 2500
    cyclic = tmp_path / "cyclic.yaml"
    cyclic.write_text("extends: cyclic.yaml\n")
    with pytest.raises(ValueError, match="cycle"):
        read_yaml(cyclic)


def test_obsolete_scoring_fraction_is_rejected(tmp_path):
    """Fail before training when a private configuration uses an old field."""
    local = local_settings(tmp_path)
    evaluation = tmp_path / "evaluation.yaml"
    evaluation.write_text(
        "horizons: [1, 2]\nneural_scoring_fraction: 0.25\n"
    )
    settings = yaml.safe_load(local.read_text())
    settings["modules"] = {"evaluation": str(evaluation)}
    local.write_text(yaml.safe_dump(settings))
    with pytest.raises(ValueError, match="neural_scoring_fraction"):
        resolve("latent_dimension_sweep.yaml", local)


def test_private_configuration_is_ignored():
    """Machine settings and private experiments cannot be added by accident."""
    candidates = [
        CONFIGURATION / "paths.local.yaml",
        CONFIGURATION / "experiments" / "quick_validation.local.yaml",
        CONFIGURATION / "data" / "quick_validation.local.yaml",
        CONFIGURATION / "models" / "quick_validation.local.yaml",
        CONFIGURATION / "plotting" / "quick_validation.local.yaml",
        REPOSITORY / "another_module" / "experiment.local.yaml",
    ]
    result = subprocess.run(
        ["git", "check-ignore", *map(str, candidates)],
        cwd=REPOSITORY,
        text=True,
        capture_output=True,
        check=True,
    )
    assert len(result.stdout.splitlines()) == len(candidates)


def test_deferred_experiment_fails_before_machine_setup(tmp_path):
    """An unspecified nonlinearity design cannot accidentally start fitting."""
    with pytest.raises(ValueError, match="deferred"):
        args = argument_parser().parse_args(
            [
                "--experiment",
                str(CONFIGURATION / "experiments/nonlinearity_sweep.yaml"),
            ]
        )
        resolve_configuration(args)


def test_missing_local_path_has_actionable_error(tmp_path):
    """Unconfigured roots must fail instead of using a developer's machine."""
    path = local_settings(tmp_path)
    local = read_yaml(path)
    local["paths"]["dataset_root"] = None
    path.write_text(yaml.safe_dump(local))
    with pytest.raises(ValueError, match="paths.dataset_root"):
        resolve("latent_dimension_sweep.yaml", path)


def test_launch_and_case_logs_are_isolated(tmp_path):
    """Repeated launches and model fits cannot append to each other's logs."""
    settings = resolve("latent_dimension_sweep.yaml", local_settings(tmp_path))
    first = launch_directory(settings, "fit")
    second = launch_directory(settings, "fit")
    assert first != second
    assert first.parent.name == "fit"
    assert first.parent.parent.name == "latent_dimension_sweep"
    for directory, token in ((first, "first-fit"), (second, "second-fit")):
        with session_logging(directory, "session"):
            logging.getLogger(__name__).warning(token)
    first_log = (first / "sessions" / "session.log").read_text()
    second_log = (second / "sessions" / "session.log").read_text()
    assert "first-fit" in first_log and "second-fit" not in first_log
    assert "second-fit" in second_log and "first-fit" not in second_log


def test_yaml_suffix_does_not_change_semantics(tmp_path):
    """The same configuration resolves identically under either filename."""
    paths = local_settings(tmp_path)
    contents = yaml.safe_dump(
        dict(
            extends=[
                str(CONFIGURATION / "experiments/latent_dimension_sweep.yaml"),
                str(paths),
            ],
            seed=17,
        )
    )
    resolved = []
    for name in ("experiment.yaml", "experiment.local.yaml"):
        path = tmp_path / name
        path.write_text(contents)
        args = argument_parser().parse_args(["--experiment", str(path)])
        resolved.append(resolve_configuration(args))
    assert resolved[0] == resolved[1]
    assert resolved[0]["experiment"]["seed"] == 17
    assert resolved[0]["model"]["training"]["training_batch_size"] == 32
