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
                    python=None, device="auto", cpu_threads=1,
                    log_level="INFO",
                    shared_fit_wait_timeout_seconds=7200,
                    shared_fit_poll_interval_seconds=120,
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
    assert latent["runtime"]["preview_regeneration"] == "incomplete"
    assert not latent["runtime"]["tensorboard"]
    assert not latent["plotting"]["previews"]["enabled"]
    assert "previews" not in latent["data"]
    assert latent["data"]["preview_adapter"] is None


def test_residual_sweep_uses_explicit_zero_stage_three_splits(tmp_path):
    """The residual sweep has a separate membership and canonical fit inputs."""
    from experiments.artifacts import model_settings
    from experiments.braid_backend import build_cases

    local = local_settings(tmp_path)
    legacy = resolve("latent_dimension_sweep.yaml", local)
    residual = resolve("latent_dimension_residual_sweep.yaml", local)
    cases = build_cases(residual["experiment"]["suite"])
    assert residual["experiment"]["name"] == "latent_dimension_residual_sweep"
    assert residual["plotting"]["curves"] != legacy["plotting"]["curves"]
    assert len(cases) == 11
    observed = {
        (
            case["dimensions"]["nx"],
            case["dimensions"]["n1"],
            case["summary_parameters"]["n2"],
        )
        for case in cases
    }
    assert observed == {
        (1, 1, 0),
        (2, 2, 0),
        (4, 4, 0),
        (8, 4, 4),
        (8, 8, 0),
        (16, 8, 8),
        (16, 16, 0),
        (32, 16, 16),
        (32, 32, 0),
        (64, 16, 48),
        (64, 64, 0),
    }
    assert all(case["dimensions"]["n3"] is None for case in cases)
    assert all(case["dimensions"]["n_pre"] == 150 for case in cases)
    old_case = next(
        case for case in build_cases(legacy["experiment"]["suite"])
        if case["dimensions"]["nx"] == 32
    )
    reused = next(
        case for case in cases
        if case["dimensions"]["nx"] == 32
        and case["summary_parameters"]["n2"] == 16
    )
    assert model_settings(residual, reused) == model_settings(legacy, old_case)


@pytest.mark.parametrize(
    ("split", "message"),
    [
        ({"n1": 0, "n2": 1, "n3": 0}, "positive integer"),
        ({"n1": 1, "n2": -1, "n3": 0}, "nonnegative integer"),
        ({"n1": 1, "n2": 0, "n3": 1}, "n3=0"),
        ({"n1": 1, "n2": 0}, "keys"),
    ],
)
def test_explicit_latent_splits_reject_invalid_dimensions(split, message):
    """Explicit residual grids reject invalid dimensions before scheduling."""
    from experiments.braid_backend import build_cases

    settings = dict(
        latent_splits=[split],
        nx_values=None,
        n1_max=None,
        n_pre=150,
        population_scales=[1.0],
    )
    with pytest.raises(ValueError, match=message):
        build_cases(settings)

def test_figure_regeneration_policy_is_validated(tmp_path):
    """Require an explicit missing-only or full component redraw policy."""
    local = local_settings(tmp_path)
    settings = read_yaml(local)
    settings["runtime"]["figure_regeneration"] = "automatic"
    local.write_text(yaml.safe_dump(settings))
    with pytest.raises(ValueError, match="figure_regeneration"):
        resolve("latent_dimension_sweep.yaml", local)

def test_preview_cli_requires_enablement_and_replaces_selection(tmp_path):
    """Preview selectors are explicit plot-stage overrides."""
    local = local_settings(tmp_path)
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text(
        yaml.safe_dump(
            dict(
                extends=[
                    str(
                        CONFIGURATION
                        / "experiments"
                        / "latent_dimension_sweep.yaml"
                    ),
                    str(local),
                ]
            )
        )
    )
    parser = argument_parser()
    disabled = parser.parse_args(
        [
            "--experiment",
            str(experiment),
            "--preview-session",
            "session_a",
        ]
    )
    with pytest.raises(ValueError, match="only valid with --stage plot"):
        resolve_configuration(disabled)
    disabled = parser.parse_args(
        [
            "--experiment",
            str(experiment),
            "--stage",
            "plot",
            "--preview-session",
            "session_a",
        ]
    )
    with pytest.raises(ValueError, match="selectors require"):
        resolve_configuration(disabled)
    enabled = parser.parse_args(
        [
            "--experiment",
            str(experiment),
            "--stage",
            "plot",
            "--previews",
            "--preview-session",
            "session_a",
            "--preview-fold",
            "2",
            "--preview-case",
            "BRAID_nx16_p1",
        ]
    )
    resolved = resolve_configuration(enabled)
    previews = resolved["plotting"]["previews"]
    assert previews["enabled"]
    assert previews["selection"] == {
        "sessions": ["session_a"],
        "folds": [2],
        "cases": ["BRAID_nx16_p1"],
    }


def test_removed_preview_stage_and_preview_regeneration_validation(tmp_path):
    """Only plot owns previews and regeneration has explicit policies."""
    parser = argument_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--experiment", "fixture.yaml", "--stage", "preview"]
        )
    local = local_settings(tmp_path)
    settings = read_yaml(local)
    settings["runtime"]["preview_regeneration"] = "automatic"
    local.write_text(yaml.safe_dump(settings))
    with pytest.raises(ValueError, match="preview_regeneration"):
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

def test_structure_sweep_requires_explicit_local_paths(tmp_path):
    """The enabled portable sweep resolves only with a local overlay."""
    with pytest.raises(ValueError, match="paths.dataset_root"):
        args = argument_parser().parse_args(
            [
                "--experiment",
                str(CONFIGURATION / "experiments/nonlinearity_sweep.yaml"),
            ]
        )
        resolve_configuration(args)
    settings = resolve(
        "nonlinearity_sweep.yaml", local_settings(tmp_path)
    )
    assert settings["data"]["features"] == "spike"
    assert settings["experiment"]["suite_plugin"] == (
        "experiments.structure_sweep:build_cases"
    )

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


@pytest.mark.parametrize("workers", [0, -1, True, 1.5, "4"])
def test_invalid_parallel_workers(tmp_path, workers):
    """Reject ambiguous or nonpositive worker allocations."""
    local = local_settings(tmp_path)
    settings = read_yaml(local)
    settings["runtime"]["parallel_workers"] = workers
    local.write_text(yaml.safe_dump(settings))
    with pytest.raises(ValueError, match="parallel_workers"):
        resolve("latent_dimension_sweep.yaml", local)

def test_parallel_workers_default_override_and_explicit_device(tmp_path):
    """Serial is default; a CLI count overrides local GPU allocation."""
    local = local_settings(tmp_path)
    assert resolve("latent_dimension_sweep.yaml", local)["runtime"][
        "parallel_workers"
    ] == 1
    experiment = local.parent / "latent_dimension_sweep.yaml"
    args = argument_parser().parse_args([
        "--experiment", str(experiment), "--parallel-workers", "4",
    ])
    assert resolve_configuration(args)["runtime"]["parallel_workers"] == 4
    for device in ("cpu", "0", "GPU-example"):
        args.device = device
        with pytest.raises(ValueError, match="device=auto"):
            resolve_configuration(args)
        args.parallel_workers = 1
        assert resolve_configuration(args)["runtime"]["device"] == device
        args.parallel_workers = 4


def test_plotting_overrides_are_resolved_without_experiment_identity(tmp_path):
    """Local display policy stays outside experiment identity."""
    local = local_settings(tmp_path)
    settings = read_yaml(local)
    settings["plotting_overrides"] = dict(
        outliers=[dict(member="case/session/fold_0")]
    )
    local.write_text(yaml.safe_dump(settings))
    resolved = resolve("latent_dimension_sweep.yaml", local)
    assert resolved["plotting"]["outliers"] == [
        dict(member="case/session/fold_0")
    ]
    assert "plotting_overrides" not in resolved["experiment"]

@pytest.mark.parametrize(
    "key",
    [
        "shared_fit_wait_timeout_seconds",
        "shared_fit_poll_interval_seconds",
    ],
)
def test_fit_requires_shared_wait_settings(tmp_path, key):
    """Fit orchestration requires explicit positive invocation controls."""
    local = local_settings(tmp_path)
    settings = read_yaml(local)
    settings["runtime"].pop(key)
    local.write_text(yaml.safe_dump(settings))
    with pytest.raises(ValueError, match=key):
        resolve("latent_dimension_sweep.yaml", local)
