"""Check exclusive text output, numbered roles and structured artifact reuse."""

import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.artifacts import (
    artifact_path,
    fit_identity,
    model_directory,
    prepare_run,
)
from experiments.cache import file_digest, fingerprint
from BRAID.tools.component_artifacts import component_paths

LOG_SCRIPT = r"""
import logging, os, sys
from pathlib import Path
from experiments.runtime import (
    configure_logging, lifecycle_scope, session_logging, stage_scope,
)
root = Path(sys.argv[1])
configure_logging(root, 'INFO')
child = logging.getLogger('test.child')
child.addHandler(logging.StreamHandler())
print('launch-diagnostic', flush=True)
for session in ('session_a', 'session_b'):
    with lifecycle_scope(root, session):
        with session_logging(root, session):
            for fold in (0, 1):
                with lifecycle_scope(root, session, fold):
                    for case in ('small', 'large'):
                        with lifecycle_scope(root, session, fold, case):
                            token = f'{session}-{fold}-{case}'
                            logging.getLogger('test.child').info(
                                'python-' + token
                            )
                            print('stdout-' + token)
                            print('stderr-' + token, file=sys.stderr)
                            os.write(1, ('native1-' + token + '\n').encode())
                            os.write(2, ('native2-' + token + '\n').encode())
with stage_scope(root, 'preprocess', session='preprocess_a', fold=0) as state:
    state['completed'] = 2
with stage_scope(root, 'plot', analysis_id='analysis_a') as state:
    state['completed'] = 1
if len(sys.argv) > 2:
    with lifecycle_scope(root, 'session_bad'):
        with session_logging(root, 'session_bad'):
            with lifecycle_scope(root, 'session_bad', 0):
                raise ValueError('unique-failure-marker')
"""


@pytest.mark.parametrize("failure", [False, True])
def test_session_output_has_one_owner(tmp_path, failure):
    """Capture Python and native output once in foreground/detached workers."""
    for detached in (False, True):
        root = tmp_path / str(detached)
        root.mkdir()
        with (root / "console.log").open("w") as console:
            result = subprocess.run(
                [sys.executable, "-c", LOG_SCRIPT, str(root)]
                + (["fail"] if failure else []),
                stdout=console,
                stderr=subprocess.STDOUT,
                start_new_session=detached,
            )
        assert result.returncode == int(failure)
        paths = list(root.rglob("*.log"))
        text = {str(p.relative_to(root)): p.read_text() for p in paths}
        all_text = "".join(text.values())
        assert text["console.log"].strip() == "launch-diagnostic"
        for session in ("session_a", "session_b"):
            for fold in (0, 1):
                for case in ("small", "large"):
                    token = f"{session}-{fold}-{case}"
                    for kind in (
                        "python",
                        "stdout",
                        "stderr",
                        "native1",
                        "native2",
                    ):
                        message = f"{kind}-{token}"
                        assert all_text.count(message) == 1
                        assert message in text[f"sessions/{session}.log"]
        assert "Started session=session_a fold=0" in text["experiment.log"]
        assert "Finished session=session_b fold=1" in text["experiment.log"]
        assert (
            "Finished stage=preprocess session=preprocess_a fold=0; "
            "previews=2 failed=0"
        ) in text["experiment.log"]
        assert (
            "Finished stage=plot analysis=analysis_a; reports=1 failed=0"
        ) in text["experiment.log"]
        for session in ("session_a", "session_b"):
            detail = str((root / "sessions" / f"{session}.log").resolve())
            assert text["experiment.log"].count(detail) == 1
            for fold in (0, 1):
                for model in ("small", "large"):
                    label = f"session={session} fold={fold} model={model}"
                    assert f"Started {label}" in text["experiment.log"]
                    assert f"Finished {label}" in text["experiment.log"]
        if failure:
            assert all_text.count("ValueError: unique-failure-marker") == 1
            assert "Failed session=session_bad" in text["experiment.log"]
            assert "unique-failure-marker" not in text["experiment.log"]


def model_options(tmp_path, **overrides):
    """Build resolved component flags without initializing TensorFlow models."""
    values = dict(
        log_dir=str(tmp_path),
        artifact_role="main",
        n1=2,
        n2=2,
        nz=2,
        skip_Cy=False,
        model1_Cy_Full=False,
        model2_Cz_Full=False,
        allow_nonzero_Cz2=False,
        has_UFT_reg=False,
        has_UFT=False,
        has_Dyz=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_component_names_and_fit_order(tmp_path):
    """Number real fits, retaining distinct dynamics and decoder names."""
    paths = component_paths(model_options(tmp_path / "main"), False)
    assert [Path(p).name for p in paths.values()] == [
        "01_behaviour_relevant_neural_dynamics",
        "02_neural_decoder",
        "03_residual_neural_dynamics",
    ]
    assert "Cz2" not in paths
    pre = component_paths(
        model_options(
            tmp_path / "pre", artifact_role="behaviour_preprocess", n1=0
        ),
        False,
    )
    assert [Path(p).name for p in pre.values()] == [
        "01_neural_dynamics",
        "02_behaviour_decoder",
    ]
    full = component_paths(
        model_options(
            tmp_path / "full", model1_Cy_Full=True, allow_nonzero_Cz2=True
        ),
        True,
    )
    assert list(full) == ["RNN1", "RNN2", "Cz2", "Cz2_fw", "Cy1", "Cy1_fw"]
    assert "full_state_neural_decoder" in full["Cy1"]
    post = component_paths(
        model_options(
            tmp_path / "post", artifact_role="input_only", n2=0, skip_Cy=True
        ),
        False,
    )
    assert [Path(p).name for p in post.values()] == [
        "01_non_neural_behaviour_dynamics"
    ]


def settings():
    """Minimal scientific identity with separate invocation preferences."""
    return dict(
        model={"epochs": 2, "training": {"verbose": True}},
        data={"previews": {"enabled": False}},
        experiment={
            "seed": 42,
            "model_plugin": "test:Model",
            "selection": {"sessions": ["a"], "folds": [0]},
        },
        evaluation={"horizons": [1]},
        paths={"root": "local"},
        runtime={"cpu_threads": 1},
        plotting={"enabled": False},
        implementation="presentation-only-code-hash",
    )


def test_component_completion_restores_without_fit(tmp_path, monkeypatch):
    """Reuse selected component weights without modifying epoch evidence."""
    import tensorflow as tf
    from BRAID.tools.training_artifacts import (
        EpochArtifacts,
        complete_component,
        previous_attempts,
    )
    from BRAID.tools.model_base_classes import ModelWithFitWithRetry

    model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(2,))])
    callback = EpochArtifacts(str(tmp_path), 1)
    callback.set_model(model)
    callback.on_epoch_end(0, {"loss": 1.0, "val_loss": 2.0})
    reference = model.get_weights()
    history = SimpleNamespace(params={"artifact_attempt": 1, "picked_epoch": 0})
    complete_component(model, str(tmp_path), history)
    before = {p: file_digest(p) for p in tmp_path.iterdir() if p.is_file()}
    model.set_weights([np.zeros_like(value) for value in reference])

    def forbidden(*args, **kwargs):
        raise AssertionError("Completed component was fitted again")

    monkeypatch.setattr(model, "fit", forbidden)
    holder = SimpleNamespace(model=model, log_dir=str(tmp_path))
    restored = ModelWithFitWithRetry.fit_with_retry(
        holder, epoch_artifacts=True
    )
    assert restored.history["loss"] == [1.0]
    assert previous_attempts(str(tmp_path)) == 1
    for actual, expected in zip(model.get_weights(), reference):
        np.testing.assert_array_equal(actual, expected)
    assert all(file_digest(p) == digest for p, digest in before.items())


def test_fitted_excerpts_have_one_canonical_payload(tmp_path):
    """Publish fitted arrays once and keep fitting indices only in the run."""
    from experiments.braid_backend import BRAIDBackend
    from experiments.cache import load_entry

    class SavedModel:
        def saveToFile(self, path):
            Path(path).write_bytes(b"checkpoint")

        def restoreModels(self):
            pass

    run = tmp_path / "run"
    prepare_run(run)
    backend = object.__new__(BRAIDBackend)
    backend.model = SavedModel()
    backend.run_directory = run
    backend.feature_cache = tmp_path / "fold-cache"
    backend.preview_excerpts = [dict(t=np.arange(5), Y=np.ones((5, 2)))]
    np.savez_compressed(artifact_path(run, "fit_indices.npz"), training=[1, 2])
    backend.save(artifact_path(run, "model.p"))
    reference = json.loads(
        artifact_path(run, "fitted_excerpts.json").read_text()
    )
    entry = load_entry(Path(reference["directory"]), reference["identity"])
    assert entry.arrays["window_0_Y"].shape == (5, 2)
    assert not list(run.glob("fitted_preprocessing*"))
    assert len(list(tmp_path.rglob("fit_indices.npz"))) == 1
    assert len(list(tmp_path.rglob("arrays.npz"))) == 1


def test_run_hash_excludes_invocation_settings():
    """Hash the same scientific identity used for completion reuse."""
    identity = dict(
        configurations=settings(),
        case={
            "name": "small", "dimensions": {"nx": 2},
            "population_scale": 1.0,
        },
        fold=0,
        source={
            "session": "a",
            "identity": {"sha256": "contents", "source": "old-path"},
            "source": "old-path",
            "session_cache": "old-cache",
            "settings": {"root": "old-root", "sampling_rate_hz": 20},
        },
        selected_ids=["M1_008_unit_1"],
        model_overrides={"epoch_artifacts": True, "verbose": True},
    )
    expected = fingerprint(fit_identity(identity))
    changed = copy.deepcopy(identity)
    snapshots = changed["configurations"]
    snapshots["run_time"] = "a different launch time"
    snapshots["paths"] = {"log_root": "new-log", "cache_root": "new-cache"}
    snapshots["runtime"] = {"device": "cpu", "cpu_threads": 8}
    snapshots["plotting"] = {"enabled": True}
    snapshots["evaluation"] = {"horizons": [2, 4]}
    snapshots["experiment"]["name"] = "different-sweep"
    snapshots["experiment"]["suite"] = {"population_scales": [0.25, 1.0]}
    snapshots["experiment"].update(
        modules={"model": "renamed.local.yaml"},
        selection={"sessions": ["a", "b"], "folds": [0, 1]},
    )
    snapshots["data"].update(
        root="new-root",
        cache_root="new-cache",
        cache_mode="rebuild",
        previews={"enabled": True, "dpi": 300},
    )
    snapshots["model"]["training"].update(
        verbose=False,
        save_training_logs=False,
        epoch_artifacts=False,
    )
    changed["source"].update(source="new-path", session_cache="new-cache")
    changed["source"]["identity"]["source"] = "new-path"
    changed["source"]["settings"]["root"] = "new-root"
    changed["model_overrides"] = {"verbose": False, "epoch_artifacts": False}
    assert fingerprint(fit_identity(changed)) == expected
    assert model_directory(Path("models"), snapshots, changed["case"]) == (
        model_directory(
            Path("models"), identity["configurations"], identity["case"]
        )
    )
    for section, key, value in (
        ("source", "identity", {"sha256": "different-data"}),
        ("configurations", "model", {"epochs": 3}),
        ("case", "dimensions", {"nx": 4}),
    ):
        semantic = copy.deepcopy(identity)
        semantic[section][key] = value
        assert fingerprint(fit_identity(semantic)) != expected
    for key, value in (("fold", 1), ("selected_ids", ["M1_039_unit_1"])):
        semantic = dict(identity, **{key: value})
        assert fingerprint(fit_identity(semantic)) != expected


@pytest.fixture(autouse=True)
def test_model_resolver(monkeypatch):
    """Provide the test adapter's semantic resolution contract."""
    import types

    module = types.ModuleType("test")

    class Model:
        model_name = "TestModel"

        @staticmethod
        def resolve_fit_configuration(
            configuration, dimensions, overrides=None, features=None
        ):
            result = copy.deepcopy(configuration)
            result.pop("training", None)
            result.update(dimensions)
            return result

    module.Model = Model
    monkeypatch.setitem(sys.modules, "test", module)


def test_defaulted_braid_hash_and_effective_batch(tmp_path, monkeypatch):
    """Omitted defaults and overridden values identify the same fit."""
    import yaml
    from experiments.braid_backend import BRAIDBackend
    from experiments.artifacts import resolved_fit, model_identity
    from experiments.contracts import FeatureSet

    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (root / "assets/config/nhp/models/BRAID.yaml").read_text()
    )
    snapshots = settings()
    snapshots["model"] = config
    snapshots["experiment"]["model_plugin"] = (
        "experiments.braid_backend:BRAIDBackend"
    )
    case = {
        "name": "small", "dimensions": {"n1": 2, "n2": 2, "nx": 4},
        "population_scale": 1.0,
    }
    omitted = copy.deepcopy(snapshots)
    omitted["model"]["training"].pop("optimiser")
    assert model_identity(snapshots, case) == model_identity(omitted, case)
    omitted["model"]["model"]["dimensions"][
        "behaviour_relevant_neural_state_size"
    ] = 999
    assert model_identity(snapshots, case) == model_identity(omitted, case)
    arrays = dict(
        role=np.repeat([0, 1], 256),
        segment=np.repeat([0, 1], 256),
        indices=np.arange(512),
    )
    features = FeatureSet(arrays, {})
    identity = dict(configurations=snapshots, case=case)
    resolved = resolved_fit(identity, features)
    assert resolved["args_base"]["batch_size"] == 2
    changed = copy.deepcopy(identity)
    changed["configurations"]["model"]["training"]["training_batch_size"] = 8
    assert resolved_fit(changed, features) == resolved
    fit = BRAIDBackend.resolve_fit_configuration(
        config,
        case["dimensions"],
        features=features,
    )
    for key in ("verbose", "save_logs", "epoch_artifacts", "clear_graph"):
        fit["args_base"].pop(key, None)
    assert fit == resolved

    from experiments import braid_backend

    captured = {}

    class FitRecorder:
        def __init__(self, **kwargs):
            pass

        def fit(self, *args, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(braid_backend, "BRAIDModel", FitRecorder)
    configuration = tmp_path / "model.yaml"
    configuration.write_text(yaml.safe_dump(config))
    backend = BRAIDBackend(str(configuration))
    for key in ("Y", "Z", "U"):
        arrays[key] = np.ones((512, 2))
    directory = tmp_path / "run"
    prepare_run(directory)
    backend.fit(features, np.array([0, 1]), case["dimensions"], directory)
    actual = {key: captured[key] for key in resolved}
    for key in ("verbose", "save_logs", "epoch_artifacts", "clear_graph"):
        actual["args_base"].pop(key, None)
    assert actual == resolved
    identity.update(
        source={"session": "a"},
        fold=0,
        selected_ids=["a", "b"],
        resolved_fit=resolved,
    )
    assert (
        json.loads(artifact_path(directory, "fit_arguments.json").read_text())[
            "args_base"
        ]["batch_size"]
        == 2
    )
