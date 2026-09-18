"""Exercise real sweep definitions against shared fit/analysis publication.

A deterministic adapter replaces expensive neural optimization only; case
resolution, fit IDs, persistence, prediction reuse and scoring use production
code. Every test publishes artifacts solely inside pytest temporary folders.
"""

import copy
import csv
import json
import re
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from experiments.analysis import read_manifest
from experiments.artifacts import (
    artifact_path, fit_directory, fit_identity, fit_inputs, fit_seed,
    model_directory, model_settings, prepare_model_settings, settings_record,
    validate_model_settings,
)
from experiments.braid_backend import build_cases
from experiments.cache import file_digest, fingerprint
from experiments.configuration import CONFIGURATION, read_yaml
from experiments.contracts import FeatureSet
from experiments.evaluation import collect_results
from experiments.populations import common_channels, scoring_indices
from experiments.runner import case_identity, plan_analysis, run_case
from experiments.windows import window_indices

WINDOW_LENGTH = 128
CHANNELS = 8


def sweep(name: str) -> dict:
    """Load the tracked experiment and all its actual module definitions."""
    experiment = read_yaml(CONFIGURATION / "experiments" / f"{name}.yaml")
    result = dict(experiment=experiment)
    for key, path in experiment["modules"].items():
        result[key] = read_yaml(Path(path))
    result["data"]["previews"]["enabled"] = False
    result["plotting"]["enabled"] = False
    return result


@pytest.fixture
def workflow(tmp_path, monkeypatch):
    """Provide deterministic arrays with three isolated split roles."""
    from experiments import fitting

    samples = WINDOW_LENGTH * 12
    time = np.arange(samples, dtype=float)
    neural = np.stack([
        np.sin(time / (5 + channel)) + channel
        for channel in range(CHANNELS)
    ], axis=1)
    behavior = np.stack([np.sin(time / 7), np.cos(time / 11)], axis=1)
    ids = [f"channel_{i}" for i in range(CHANNELS)]
    features = FeatureSet(
        dict(
            Y=neural, Z=behavior, U=behavior, t=time,
            indices=np.arange(samples), ids=np.array(ids),
            units=np.zeros(CHANNELS, dtype=int),
            role=np.repeat([0, 1, 2], samples // 3),
            segment=np.repeat([0, 1, 2], samples // 3),
        ),
        dict(session="session_a", fold=0, identity={"sha256": "fixture"}),
    )
    dataset = SimpleNamespace(
        load=lambda name: features,
        fold=lambda *args: features,
        sessions=lambda: ["session_a"],
        inventory=lambda: {"session_a": ids},
    )
    calls = dict(fit=0, predict=0, load=0)

    class Backend:
        length = WINDOW_LENGTH

        def fit(self, features, columns, dimensions, directory):
            calls["fit"] += 1
            np.savez_compressed(
                artifact_path(directory, "fit_indices.npz"),
                training=window_indices(features, 0, self.length),
            )

        def save(self, path):
            path.write_bytes(b"deterministic-checkpoint")

        def load(self, path):
            assert path.read_bytes() == b"deterministic-checkpoint"
            calls["load"] += 1

        def predict(self, y, u, horizons):
            calls["predict"] += 1
            return dict(
                Y=np.stack([y + 0.1 for _ in horizons]),
                Z=np.stack([u + 0.1 for _ in horizons]),
                X=np.stack([y for _ in horizons]),
                valid=np.stack([
                    np.arange(len(y)) % self.length >= h for h in horizons
                ]),
            )

    monkeypatch.setattr(fitting, "plugin", lambda *a, **kw: Backend())
    arguments = SimpleNamespace(
        no_previews=True, no_plots=True, stage="fit",
        log_level="INFO", log_directory=tmp_path / "logs",
    )

    def prepare(config):
        cases = build_cases(config["experiment"]["suite"])
        directory = plan_analysis(
            dataset, ["session_a"], [0], cases, config, ids, tmp_path
        )
        return cases, directory

    def run(config, case, directory):
        settings = dict(
            config["evaluation"], velocity=config["data"]["infer_velocity"],
            previews=config["data"]["previews"],
        )
        return run_case(
            dataset, features, 0, case, settings, arguments, ids,
            config, None, tmp_path, directory,
        )

    return SimpleNamespace(
        root=tmp_path, features=features, ids=ids, dataset=dataset,
        calls=calls, arguments=arguments, prepare=prepare, run=run,
    )


def hashes(directory: Path) -> dict[Path, str]:
    """Capture every existing payload to detect accidental rewriting."""
    return {
        path: file_digest(path)
        for path in directory.rglob("*") if path.is_file()
    }


@pytest.mark.parametrize("nx", [16, 64])
def test_population_common_reuses_latent_fit_and_predictions(workflow, nx):
    """Full-population overlap must not invoke fitting or inference again."""
    latent = sweep("latent_dimension_sweep")
    population = sweep("neural_population_sweep")
    latent_cases, latent_analysis = workflow.prepare(latent)
    pop_cases, pop_analysis = workflow.prepare(population)
    left = next(c for c in latent_cases if c["dimensions"]["nx"] == nx)
    right = next(
        c for c in pop_cases
        if c["dimensions"]["nx"] == nx and c["population_scale"] == 1.0
    )
    assert left == right
    assert workflow.run(latent, left, latent_analysis)
    assert workflow.calls["fit"] == 1
    latent_rows = collect_results(latent_analysis)
    assert {row["evaluation_set"] for row in latent_rows} == {"full"}
    identity, _ = case_identity(
        workflow.features, 0, left, latent, workflow.ids
    )
    run = fit_directory(workflow.root, identity)
    assert run.is_relative_to(workflow.root / "experiments")
    assert run.parent.name == "fold_0"
    assert re.fullmatch(r"BRAID_[0-9a-f]{64}", run.parents[2].name)
    record = validate_model_settings(run.parents[2])
    assert record == settings_record(identity)
    assert fit_identity(identity)["settings_hash"] == record["settings_hash"]
    population_identity, _ = case_identity(
        workflow.features, 0, right, population, workflow.ids
    )
    assert fit_directory(workflow.root, population_identity) == run
    settings_before = hashes(run.parents[2])
    before = hashes(run)
    counts = dict(workflow.calls)
    assert workflow.run(population, right, pop_analysis)
    assert workflow.calls == counts
    assert all(file_digest(path) == digest for path, digest in before.items())
    assert all(
        file_digest(path) == digest
        for path, digest in settings_before.items()
    )
    rows = collect_results(pop_analysis)
    assert {row["evaluation_set"] for row in rows} == {"full", "common"}
    assert {row["fit_id"] for row in rows} == {run.name}
    common = [row for row in rows if row["evaluation_set"] == "common"]
    assert all(
        row["scored_channel_ids"] == workflow.ids[:2] for row in common
    )
    full = [row for row in rows if row["evaluation_set"] == "full"]
    assert [row["neural"] for row in full] == [
        row["neural"] for row in latent_rows
    ]
    assert all(row["neural"]["valid_mse_channels"] == 2 for row in common)
    assert latent_analysis != pop_analysis
    assert not workflow.run(population, right, pop_analysis)
    assert workflow.calls == counts

    # A different common intersection is solely an analysis change.
    changed = copy.deepcopy(population)
    changed["experiment"]["suite"]["population_scales"] = [0.5, 1.0]
    _, changed_analysis = workflow.prepare(changed)
    assert workflow.run(changed, right, changed_analysis)
    changed_rows = collect_results(changed_analysis)
    assert all(
        row["scored_channel_ids"] == workflow.ids[:4]
        for row in changed_rows if row["evaluation_set"] == "common"
    )
    assert workflow.calls == counts
    assert all(file_digest(path) == digest for path, digest in before.items())


def test_rendering_changes_reuse_analysis_and_fit(workflow):
    """Presentation changes neither evaluation ownership nor model identity."""
    config = sweep("latent_dimension_sweep")
    cases, analysis = workflow.prepare(config)
    workflow.run(config, cases[0], analysis)
    counts = dict(workflow.calls)
    scientific = hashes(workflow.root / "experiments")
    changed = copy.deepcopy(config)
    changed["plotting"]["presentation"]["title_font"] = 40
    changed["plotting"]["metrics"] = ["mse"]
    changed["data"]["previews"]["seed"] = 99
    _, refreshed = workflow.prepare(changed)
    assert refreshed == analysis
    workflow.run(changed, cases[0], refreshed)
    assert workflow.calls == counts
    assert scientific == hashes(workflow.root / "experiments")
    assert read_manifest(analysis)["rendering"]["plotting"] == changed["plotting"]


def test_rendering_failure_preserves_fit_and_evaluation(workflow, monkeypatch):
    """A failed preview cannot invalidate a successfully completed fit."""
    from experiments import runner

    config = sweep("latent_dimension_sweep")
    cases, analysis = workflow.prepare(config)

    def fail(*args, **kwargs):
        raise ValueError("injected renderer failure")

    monkeypatch.setattr(runner, "preprocessing_previews", fail)
    with pytest.raises(RuntimeError, match="Rendering failed"):
        workflow.run(config, cases[0], analysis)
    assert workflow.calls["fit"] == 1
    assert any(
        member["state"] == "complete"
        for member in read_manifest(analysis)["members"].values()
    )
    counts = dict(workflow.calls)
    monkeypatch.setattr(runner, "preprocessing_previews", lambda *args: None)
    assert not workflow.run(config, cases[0], analysis)
    assert workflow.calls == counts


def test_new_horizons_reuse_fit(workflow):
    config = sweep("latent_dimension_sweep")
    cases, directory = workflow.prepare(config)
    case = cases[0]
    workflow.run(config, case, directory)
    identity, _ = case_identity(
        workflow.features, 0, case, config, workflow.ids
    )
    run = fit_directory(workflow.root, identity)
    before = hashes(run)
    changed = copy.deepcopy(config)
    changed["evaluation"]["horizons"] = [1, 3]
    _, other = workflow.prepare(changed)
    workflow.run(changed, case, other)
    assert workflow.calls["fit"] == 1
    assert len(list((run / "predictions" / "test").glob("horizons_*/"))) == 2
    assert all(file_digest(path) == digest for path, digest in before.items())


def test_failed_scoring_does_not_invalidate_fit(workflow, monkeypatch):
    from experiments import evaluation

    config = sweep("latent_dimension_sweep")
    cases, directory = workflow.prepare(config)
    original = evaluation.evaluate_forecasts

    def fail(*args, **kwargs):
        raise RuntimeError("injected scoring failure")

    monkeypatch.setattr(evaluation, "evaluate_forecasts", fail)
    with pytest.raises(RuntimeError, match="injected scoring"):
        workflow.run(config, cases[0], directory)
    counts = dict(workflow.calls)
    monkeypatch.setattr(evaluation, "evaluate_forecasts", original)
    workflow.arguments.stage = "evaluate"
    workflow.run(config, cases[0], directory)
    assert workflow.calls == counts
    assert collect_results(directory)


def test_evaluate_missing_fit_never_trains(workflow):
    config = sweep("latent_dimension_sweep")
    cases, directory = workflow.prepare(config)
    workflow.arguments.stage = "evaluate"
    with pytest.raises(ValueError, match="requires a completed fit"):
        workflow.run(config, cases[0], directory)
    assert workflow.calls["fit"] == 0


def test_concurrent_analysis_requests_share_one_fit(workflow):
    first = sweep("latent_dimension_sweep")
    second = copy.deepcopy(first)
    second["experiment"]["name"] = "other_question"
    cases, left = workflow.prepare(first)
    _, right = workflow.prepare(second)
    with ThreadPoolExecutor(max_workers=2) as executor:
        jobs = [
            executor.submit(workflow.run, conf, cases[0], directory)
            for conf, directory in ((first, left), (second, right))
        ]
        assert all(job.result() for job in jobs)
    assert workflow.calls["fit"] == 1
    assert workflow.calls["predict"] == 2  # Forecast plus reload check.


def test_fit_identity_uses_training_not_analysis(workflow):
    config = sweep("latent_dimension_sweep")
    cases = build_cases(config["experiment"]["suite"])
    identity, _ = case_identity(
        workflow.features, 0, cases[0], config, workflow.ids
    )
    expected = fit_identity(identity)
    changed = copy.deepcopy(identity)
    changed["case"]["name"] = "different-label"
    changed["case"]["summary_parameters"]["description"] = "another-label"
    changed["configurations"]["evaluation"]["horizons"] = [1, 3]
    changed["configurations"]["analysis_implementation"] = "new-common-code"
    changed["configurations"]["experiment"]["name"] = "other_question"
    assert fit_identity(changed) == expected
    assert fit_seed(changed) == fit_seed(identity)
    assert fit_directory(workflow.root, changed) == fit_directory(
        workflow.root, identity
    )
    for key, value in (("fold", 1), ("selected_ids", ["different-channel"])):
        changed = dict(identity, **{key: value})
        assert fit_identity(changed) != expected
    changed = copy.deepcopy(identity)
    changed["configurations"]["experiment"]["seed"] += 1
    assert fit_identity(changed) != expected
    changed = copy.deepcopy(identity)
    changed["resolved_fit"]["args_base"]["epochs"] += 1
    assert fit_identity(changed) != expected


def test_common_intersection_and_output_mapping():
    groups = [["a", "c", "b"], ["d", "b", "c"], ["c", "e", "b"]]
    assert common_channels(groups) == ["c", "b"]
    np.testing.assert_array_equal(
        scoring_indices(["b", "a", "c"], ["c", "b"]), [2, 0]
    )
    for groups in ([], [[]], [["a"], ["b"]], [["a", "a"]]):
        with pytest.raises(ValueError):
            common_channels(groups)
    with pytest.raises(ValueError, match="missing"):
        scoring_indices(["a", "b"], ["b", "c"])


def test_completed_checksum_failure_never_retrains(workflow):
    config = sweep("latent_dimension_sweep")
    cases, directory = workflow.prepare(config)
    workflow.run(config, cases[0], directory)
    member = next(
        m for m in read_manifest(directory)["members"].values()
        if m["state"] == "complete"
    )
    path = artifact_path(workflow.root / member["fit"], "model.p")
    path.write_bytes(b"injected corruption")
    with pytest.raises(ValueError, match="checksum"):
        workflow.run(config, cases[0], directory)
    assert workflow.calls["fit"] == 1


def test_reports_follow_manifest_membership_only(workflow):
    from experiments.reporting import braid_report
    from experiments.analysis import find_analysis

    config = sweep("latent_dimension_sweep")
    cases, directory = workflow.prepare(config)
    workflow.run(config, cases[0], directory)
    orphan = directory / "metrics" / "unregistered" / "metrics.json"
    orphan.parent.mkdir(parents=True)
    orphan.write_text('[{"session": "unrelated"}]')
    counts = dict(workflow.calls)
    braid_report(directory, config["plotting"], 20)
    assert find_analysis(workflow.root, config) == directory
    with (directory / "summaries" / "model_summary.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert sum(row["status"] == "pending" for row in rows) == len(cases) - 1
    assert {row["session"] for row in rows} == {"session_a"}
    assert all(
        row["configuration"] == cases[0]["name"]
        for row in collect_results(directory)
    )
    assert workflow.calls == counts


def test_incomplete_prediction_bundle_is_quarantined(workflow):
    from experiments.fitting import prediction_directory

    config = sweep("latent_dimension_sweep")
    cases, directory = workflow.prepare(config)
    identity, _ = case_identity(
        workflow.features, 0, cases[0], config, workflow.ids
    )
    run = fit_directory(workflow.root, identity)
    bundle = prediction_directory(run, config["evaluation"]["horizons"])
    prepare_model_settings(run.parents[2], identity)
    bundle.mkdir(parents=True)
    (bundle / "predictions.npz").write_bytes(b"incomplete")
    workflow.run(config, cases[0], directory)
    quarantined = list(
        (bundle.parent / "quarantine").glob("*/predictions.npz")
    )
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == b"incomplete"
    assert workflow.calls["fit"] == 1


def test_obsolete_layout_is_not_discovered(workflow):
    old = workflow.root / "latent_dimension_sweep" / "model" / "fold_0"
    old.mkdir(parents=True)
    completed = old / "fit_complete.json"
    completed.write_text('{"old": true}')
    before = hashes(old)
    config = sweep("latent_dimension_sweep")
    cases, directory = workflow.prepare(config)
    workflow.run(config, cases[0], directory)
    assert workflow.calls["fit"] == 1
    assert all(file_digest(path) == digest for path, digest in before.items())


def test_interrupted_fit_retries_without_publishing_completion(
    workflow, monkeypatch,
):
    from experiments import fitting

    config = sweep("latent_dimension_sweep")
    cases, directory = workflow.prepare(config)
    factory = fitting.plugin
    failed = []

    def interrupted(*args, **kwargs):
        backend = factory(*args, **kwargs)
        original = backend.save

        def save(path):
            if not failed:
                failed.append(True)
                raise RuntimeError("injected interrupted save")
            original(path)

        backend.save = save
        return backend

    monkeypatch.setattr(fitting, "plugin", interrupted)
    with pytest.raises(RuntimeError, match="interrupted save"):
        workflow.run(config, cases[0], directory)
    assert not list(
        (workflow.root / "experiments").rglob("fit_complete.json")
    )
    workflow.run(config, cases[0], directory)
    assert workflow.calls["fit"] == 2
    assert collect_results(directory)


def test_main_sweeps_share_two_fits_without_common_retraining(
    workflow, monkeypatch,
):
    """Run both entire sweep grids through the real stage orchestration."""
    from experiments import runner

    original_plugin = runner.plugin

    def dispatch(reference, **kwargs):
        if reference == "experiments.nhp:NHPDataset":
            return workflow.dataset
        return original_plugin(reference, **kwargs)

    monkeypatch.setattr(runner, "plugin", dispatch)
    monkeypatch.setattr(runner, "configure_device", lambda *a: None)
    monkeypatch.setattr(runner, "configure_logging", lambda *a: None)
    monkeypatch.setattr(runner, "session_logging", lambda *a: nullcontext())
    monkeypatch.setattr(
        runner, "lifecycle_scope", lambda *a: nullcontext({})
    )
    monkeypatch.setattr(
        runner, "launch_directory", lambda *a: workflow.root / "logs"
    )
    artifacts = workflow.root / "artifacts"
    saved_latent = None
    for name in ("latent_dimension_sweep", "neural_population_sweep"):
        config = sweep(name)
        config.update(
            paths={"artifact_root": str(artifacts)},
            runtime=dict(
                log_level="INFO", device="cpu",
                cpu_threads=1, cpu_interop_threads=1,
            ),
        )
        monkeypatch.setattr(
            runner, "resolve_configuration", lambda args: copy.deepcopy(config)
        )
        monkeypatch.setattr(
            sys, "argv",
            ["runner", "--experiment", str(CONFIGURATION / "experiments"
                                          / f"{name}.yaml"),
             "--fold", "0", "--no-plots", "--no-previews"],
        )
        runner.main()
        manifests = list(
            (artifacts / "analysis" / name).glob("*/manifest.json")
        )
        assert len(manifests) == 1
        assert (manifests[0].parent / "summaries" / "summary.csv").exists()
        if name == "latent_dimension_sweep":
            assert workflow.calls["fit"] == 7
            assert workflow.calls["predict"] == 14
            saved_latent = hashes(artifacts / "experiments")
        else:
            assert workflow.calls["fit"] == 11
            assert workflow.calls["predict"] == 22
            assert all(
                file_digest(path) == digest
                for path, digest in saved_latent.items()
            )
            manifest = read_manifest(manifests[0].parent)
            assert all(
                member["state"] == "complete"
                for member in manifest["members"].values()
            )
            rows = collect_results(manifests[0].parent)
            assert {r["evaluation_set"] for r in rows} == {"full", "common"}
            before = dict(workflow.calls)
            runner.main()
            assert workflow.calls == before


@pytest.mark.parametrize(
    "change",
    [
        "optimizer", "regularization", "training_horizons",
        "preprocessing", "population", "batch_limit", "seed",
        "implementation", "dependencies", "numerical_override",
    ],
)
def test_settings_hash_covers_complete_recipe(workflow, change):
    """Every scientific recipe change must leave the old settings group."""
    config = sweep("latent_dimension_sweep")
    case = build_cases(config["experiment"]["suite"])[0]
    expected = model_directory(workflow.root, config, case)
    changed = copy.deepcopy(config)
    changed_case = copy.deepcopy(case)
    overrides = None
    if change == "optimizer":
        changed["model"]["training"]["optimiser"] = "SGD"
    elif change == "regularization":
        transition = changed["model"]["model"]["stage_1"]["state_transition"]
        transition["kernel_regulariser"] = "l2"
        transition["kernel_regulariser_arguments"] = {"l2": 0.01}
    elif change == "training_horizons":
        changed["model"]["forecast"]["steps_ahead"] = [1, 2]
        changed["model"]["forecast"]["steps_ahead_loss_weights"] = [1, 1]
    elif change == "preprocessing":
        changed["data"]["sampling_rate_hz"] = 40
    elif change == "population":
        changed_case["population_scale"] = 0.5
    elif change == "batch_limit":
        changed["model"]["training"]["training_batch_size"] = 16
    elif change == "seed":
        changed["experiment"]["seed"] = 100
    elif change == "implementation":
        changed["fitting_implementation"] = "different-numerical-code"
    elif change == "dependencies":
        changed["versions"] = {"python": "different-version"}
    elif change == "numerical_override":
        overrides = {"epochs": 12}
    actual = model_directory(
        workflow.root, changed, changed_case, overrides
    )
    assert actual != expected
    assert re.fullmatch(r"BRAID_[0-9a-f]{64}", actual.name)


def test_recipe_defaults_and_configured_batch_limit(workflow):
    """Session-specific capping must not replace the configured recipe."""
    config = sweep("latent_dimension_sweep")
    case = build_cases(config["experiment"]["suite"])[0]
    expected = model_directory(workflow.root, config, case)
    explicit = model_settings(config, case)
    omitted = copy.deepcopy(config)
    omitted["model"]["training"].pop("optimiser")
    assert model_settings(omitted, case) == explicit
    assert model_directory(workflow.root, omitted, case) == expected
    identity, _ = case_identity(
        workflow.features, 0, case, config, workflow.ids
    )
    assert identity["resolved_fit"]["args_base"]["batch_size"] == 4
    assert explicit["model"]["args_base"]["batch_size"] == 32
    changed = copy.deepcopy(config)
    changed["model"]["training"]["training_batch_size"] = 8
    other, _ = case_identity(
        workflow.features, 0, case, changed, workflow.ids
    )
    assert identity["resolved_fit"] == other["resolved_fit"]
    assert fit_directory(workflow.root, identity) != fit_directory(
        workflow.root, other
    )
    assert fit_seed(identity) == fit_seed(other)
    old_seed = int(fingerprint(fit_inputs(identity))[:8], 16) % (2**31 - 1)
    assert fit_seed(identity) == old_seed


@pytest.mark.parametrize(
    "name", [None, "", "../BRAID", "BRAID/name", "BRAID name"],
)
def test_invalid_public_model_name_is_rejected(workflow, monkeypatch, name):
    from experiments.braid_backend import BRAIDBackend

    config = sweep("latent_dimension_sweep")
    case = build_cases(config["experiment"]["suite"])[0]
    monkeypatch.setattr(BRAIDBackend, "model_name", name)
    with pytest.raises(ValueError, match="model_name"):
        model_directory(workflow.root, config, case)


def test_public_name_is_metadata_not_seed_input(workflow, monkeypatch):
    from experiments.braid_backend import BRAIDBackend

    config = sweep("latent_dimension_sweep")
    case = build_cases(config["experiment"]["suite"])[0]
    identity, _ = case_identity(
        workflow.features, 0, case, config, workflow.ids
    )
    seed, recipe = fit_seed(identity), settings_record(identity)
    monkeypatch.setattr(BRAIDBackend, "model_name", "DeclaredName")
    assert model_directory(workflow.root, config, case).name == (
        f"DeclaredName_{recipe['settings_hash']}"
    )
    assert settings_record(identity) == recipe
    assert fit_seed(identity) == seed


def test_settings_conflict_is_not_overwritten(workflow):
    config = sweep("latent_dimension_sweep")
    cases, directory = workflow.prepare(config)
    case = cases[0]
    identity, _ = case_identity(
        workflow.features, 0, case, config, workflow.ids
    )
    run = fit_directory(workflow.root, identity)
    prepare_model_settings(run.parents[2], identity)
    path = run.parents[2] / "model_settings.json"
    record = json.loads(path.read_text())
    record["settings"]["seed"] += 1
    path.write_text(json.dumps(record))
    digest = file_digest(path)
    with pytest.raises(ValueError, match="settings hash mismatch"):
        workflow.run(config, case, directory)
    assert file_digest(path) == digest
    assert workflow.calls["fit"] == 0


def test_missing_settings_is_not_recreated_for_completed_fit(workflow):
    config = sweep("latent_dimension_sweep")
    cases, directory = workflow.prepare(config)
    workflow.run(config, cases[0], directory)
    identity, _ = case_identity(
        workflow.features, 0, cases[0], config, workflow.ids
    )
    run = fit_directory(workflow.root, identity)
    path = run.parents[2] / "model_settings.json"
    path.rename(path.with_suffix(".missing"))
    before = hashes(run.parents[2])
    with pytest.raises(ValueError, match="Missing model settings"):
        workflow.run(config, cases[0], directory)
    assert not path.exists()
    assert workflow.calls["fit"] == 1
    assert all(file_digest(path) == digest for path, digest in before.items())


def test_old_model_settings_name_is_not_reused(workflow):
    config = sweep("latent_dimension_sweep")
    cases, directory = workflow.prepare(config)
    workflow.run(config, cases[0], directory)
    identity, _ = case_identity(
        workflow.features, 0, cases[0], config, workflow.ids
    )
    run = fit_directory(workflow.root, identity)
    wrong = run.parents[2].with_name("BRAIDBackend_nx1_n11_n_pre150")
    run.parents[2].rename(wrong)
    before = hashes(wrong)
    workflow.run(config, cases[0], directory)
    assert run.exists()
    assert workflow.calls["fit"] == 2
    assert all(file_digest(path) == digest for path, digest in before.items())
