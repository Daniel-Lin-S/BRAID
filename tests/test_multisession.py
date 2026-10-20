"""Scientific and artifact contracts for multi-session training.

Synthetic fixtures use independent 128-step windows with unequal neural
widths. No real data or completed campaign artifacts are modified.
"""

from copy import deepcopy
from pathlib import Path
import json

import numpy as np
import tensorflow as tf

from BRAID.multisession import HORIZONS, LINEAR, MixedForecaster
from BRAID.RNNModel import RNNModel
from experiments.configuration import read_yaml
from experiments.multisession.plan import (
    ANALYSES,
    CAPACITY,
    PRIMARY,
    SUPERVISION,
    build_plan,
)
from experiments.multisession.store import completed, dump_pickle, job
from experiments.multisession.training import recurrent_data, select_batch

ROOT = Path(__file__).resolve().parents[1]
LENGTH = 128


def fixture_data(widths=(3, 5)):
    """Return two finite session datasets with different channel widths."""
    rng = np.random.default_rng(7)
    return [
        recurrent_data(
            dict(
                Y=rng.normal(size=(2, LENGTH, width)).astype("float32"),
                U=rng.normal(size=(2, LENGTH, 2)).astype("float32"),
            )
        )
        for width in widths
    ]


def test_grid_and_membership():
    """Count unique fits and enforce held-out, paired source membership."""
    settings = read_yaml(
        ROOT / "assets/config/nhp/experiments/multi_session_training.yaml"
    )
    for key in ("data", "model"):
        settings[key] = read_yaml(Path(settings["modules"][key]))
    sessions = {s: dict(sha256=s) for s in settings["held_out_sessions"]}
    sessions.update({f"source_{i}": dict(sha256=str(i)) for i in range(33)})
    plan = build_plan(settings, sessions)
    jobs = plan["jobs"]
    assert {
        kind: sum(j["kind"] == kind for j in jobs.values())
        for kind in ("source", "adaptation", "scratch", "preprocessing")
    } == {
        "source": 13,
        "adaptation": 468,
        "scratch": 72,
        "preprocessing": 36,
    }
    assert set(plan["analyses"]) == set(ANALYSES)
    sources = [j for j in jobs.values() if j["kind"] == "source"]
    for seed in settings["seeds"]:
        paired = [
            j
            for j in sources
            if j["seed"] == seed
            and j["objective"] == "neural"
            and j["latent"] == 32
        ]
        largest = next(j["sessions"] for j in paired if j["source_count"] == 33)
        for item in paired:
            assert item["sessions"] == largest[: item["source_count"]]
            assert not set(item["sessions"]) & set(
                settings["held_out_sessions"]
            )
    references = set(plan["analyses"][PRIMARY])
    assert references & set(plan["analyses"][CAPACITY])
    assert references & set(plan["analyses"][SUPERVISION])


def test_mixed_gradients_optimizer_isolation_and_serialization(tmp_path):
    """Absent sessions keep their weights and Adam state exactly unchanged."""
    model = MixedForecaster([(3, 3), (5, 5)], 4, "AK", 42)
    data = fixture_data()
    batch = select_batch(data, [np.array([0, 1]), np.array([], dtype=int)])
    states, _ = model.states(batch[0])
    assert states.shape == (2, LENGTH, 4)
    absent = [v.numpy().copy() for v in model.private_variables[1]]
    optimizers = [tf.keras.optimizers.legacy.Adam(0.001) for _ in range(3)]
    groups = [model.shared_variables, *model.private_variables]
    loss, gradients = model.loss_and_gradients(*batch)
    assert np.isfinite(loss.numpy())
    offset = 0
    for i, (optimizer, variables) in enumerate(zip(optimizers, groups)):
        selected = gradients[offset : offset + len(variables)]
        offset += len(variables)
        if i != 2:
            optimizer.apply_gradients(zip(selected, variables))
    assert optimizers[2].get_weights() == []
    for before, after in zip(absent, model.private_variables[1]):
        np.testing.assert_array_equal(before, after.numpy())
    weights = model.weights()
    restored = MixedForecaster([(3, 3), (5, 5)], 4, "AK", 93)
    restored.restore(weights)
    assert restored.sessions[0]["Afw"] is restored.sessions[1]["Afw"]
    first, _ = model.predict(batch[0], batch[1])
    second, _ = restored.predict(batch[0], batch[1])
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)


def test_native_forecast_and_frozen_maps():
    """Mixed recursion matches native RNN outputs for identical inputs."""
    model = MixedForecaster([(3, 3)], 4, "AK", 42)
    decoder = dict(
        LINEAR,
        units=(256,),
        use_bias=True,
        activation="relu",
        kernel_initializer="glorot_uniform",
    )
    native = RNNModel(
        nx=4,
        ny=5,
        ny_out=3,
        block_samples=LENGTH,
        batch_size=2,
        n1_in=2,
        steps_ahead=HORIZONS,
        enable_forward_pred=True,
        cell_args=dict(
            ASettings=deepcopy(LINEAR),
            KSettings=deepcopy(LINEAR),
            CSettings=decoder,
        ),
    )
    for name, layer in model.sessions[0].items():
        getattr(native.rnn.cell, name).set_weights(layer.get_weights())
    data = fixture_data((3,))
    batch = select_batch(data, [np.array([0, 1])])
    expected, _ = model.predict(batch[0], batch[1])
    actual = native.model((batch[0][0], batch[1][0]), training=False)
    for i in range(len(HORIZONS)):
        np.testing.assert_allclose(expected[0][i], actual[i], atol=1e-7)
    native.set_frozen_modules(model.transferable())
    shared_refs = {v.ref() for v in model.shared_variables}
    assert shared_refs
    frozen_refs = {
        v.ref()
        for name in ("Afw", "Kfw")
        for v in getattr(native.rnn.cell, name).weights
    }
    assert not frozen_refs & {v.ref() for v in native.model.trainable_weights}
    before = [
        v.numpy().copy()
        for name in ("Afw", "Kfw")
        for v in getattr(native.rnn.cell, name).weights
    ]
    with tf.GradientTape() as tape:
        output = native.model((batch[0][0], batch[1][0]), training=True)
        loss = sum(tf.reduce_mean(tf.square(x)) for x in output[:4])
    variables = native.model.trainable_weights
    gradients = tape.gradient(loss, variables)
    tf.keras.optimizers.Adam(0.001).apply_gradients(zip(gradients, variables))
    after = [
        v.numpy()
        for name in ("Afw", "Kfw")
        for v in getattr(native.rnn.cell, name).weights
    ]
    for a, b in zip(before, after):
        np.testing.assert_array_equal(a, b)


def test_completion_reuse_and_quarantine(tmp_path):
    """Completed payloads are reused; incomplete attempts are isolated."""
    directory = tmp_path / "fit"
    directory.mkdir()
    (directory / "partial").write_text("interrupted")
    with job(directory, dict(seed=42)) as needed:
        assert needed
        dump_pickle(directory / "weights.pkl", [np.ones(2)])
    assert completed(directory)
    assert list((tmp_path / "quarantine").glob("*/partial"))
    before = (directory / "complete.json").read_bytes()
    with job(directory, dict(seed=42)) as needed:
        assert not needed
    assert (directory / "complete.json").read_bytes() == before


def test_neural_objective_ignores_behavior_labels(tmp_path):
    """Changing only source Z cannot change scaling, selection or weights."""
    from experiments.contracts import FeatureSet
    from experiments.multisession.data import session_windows
    from experiments.multisession.training import train_stage

    rng = np.random.default_rng(42)
    count = LENGTH * 4
    arrays = dict(
        Y=rng.normal(size=(count, 3)),
        U=rng.normal(size=(count, 2)),
        Z=rng.normal(size=(count, 4)),
        indices=np.arange(count),
        role=np.repeat([0, 1], count // 2),
        segment=np.repeat([0, 1], count // 2),
        ids=np.arange(3),
    )
    first = session_windows(FeatureSet(arrays, {}), False)
    changed = dict(arrays, Z=arrays["Z"][::-1] * 100)
    second = session_windows(FeatureSet(changed, {}), False)
    assert first[2] == second[2]
    outputs = []
    for index, (train, validation, _) in enumerate((first, second)):
        model = MixedForecaster([(3, 3)], 4, "AK", 42)
        train_stage(
            model,
            [recurrent_data(train)],
            [recurrent_data(validation)],
            tmp_path / str(index),
            dict(
                maximum_updates=2,
                learning_rate=0.001,
                batch_size=2,
                validation_interval=1,
                patience=10,
            ),
            42,
        )
        outputs.append(model.weights())
    for key in outputs[0]:
        for a, b in zip(outputs[0][key], outputs[1][key]):
            np.testing.assert_array_equal(a, b)


def test_nested_subsets_and_fixed_test():
    """Guarded role subsets remain nested after complete-batch trimming."""
    from experiments.contracts import FeatureSet
    from experiments.datasets.nhp import temporal_segments
    from experiments.datasets.windows import window_indices
    from experiments.multisession.data import target_selection

    for count in (7202, 67258, 11526, 16319):
        for fold in (0, 2, 4):
            segments = temporal_segments(count, 5, fold, 36)
            indices, roles, labels = [], [], []
            for number, (start, stop, role) in enumerate(segments):
                indices.extend(range(start + 1, stop))
                roles.extend([role] * (stop - start - 1))
                labels.extend([number] * (stop - start - 1))
            features = FeatureSet(
                dict(
                    indices=np.asarray(indices),
                    role=np.asarray(roles),
                    segment=np.asarray(labels),
                ),
                dict(session="target", fold=fold),
            )
            test = window_indices(features, 2, LENGTH)
            for seed in (42, 43, 44):
                previous = (set(), set())
                for fraction in (0.05, 0.1, 0.2, 0.4, 0.8):
                    tr, va, record = target_selection(
                        features,
                        count,
                        fraction,
                        seed,
                    )
                    assert len(tr) and len(va)
                    assert previous[0] <= set(tr.ravel())
                    assert previous[1] <= set(va.ravel())
                    assert not set(test.ravel()) & set(tr.ravel())
                    assert not set(test.ravel()) & set(va.ravel())
                    assert record["realized_fraction"] <= fraction
                    previous = set(tr.ravel()), set(va.ravel())
                    np.testing.assert_array_equal(
                        test,
                        window_indices(features, 2, LENGTH),
                    )


def test_reports_use_session_means_and_scratch_references(
    tmp_path, monkeypatch
):
    """Sensitivity legends keep paired conditions and scratch references."""
    from experiments.multisession.reports import figures, summarize
    import matplotlib.figure

    base = dict(
        objective="neural",
        latent=32,
        sharing="A",
        source_count=33,
        fraction=0.05,
        target="neural",
        horizon=1,
        metric="cc",
    )
    rows = [dict(base, session="one", value=v) for v in (0.1, 0.2, 0.3)]
    rows.append(dict(base, session="two", value=0.8))
    assert np.isclose(summarize(rows)[0]["mean"], 0.5)
    summary = []
    for objective in ("neural", "behavior"):
        for latent in (32, 64):
            for sharing in ("A", "AK", "AKC"):
                summary.append(
                    dict(
                        base,
                        objective=objective,
                        latent=latent,
                        sharing=sharing,
                        mean=0.5,
                        sem=0.1,
                    )
                )
    for latent in (32, 64):
        summary.append(
            dict(
                base,
                objective="scratch",
                latent=latent,
                sharing="scratch",
                source_count=0,
                mean=0.4,
                sem=0.05,
            )
        )
    saved = []

    def capture(figure, path, **kwargs):
        saved.append(
            (
                path.name,
                [
                    t.get_text()
                    for legend in figure.legends
                    for t in legend.get_texts()
                ],
                {
                    line.get_label(): line.get_linestyle()
                    for axis in figure.axes
                    for line in axis.lines
                },
            )
        )

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", capture)
    settings = dict(
        target_fractions=[0.05], source_counts=[33], evaluation_horizons=[1]
    )
    for analysis in (CAPACITY, SUPERVISION):
        figures(summary, analysis, tmp_path, settings)
    assert saved
    for name, labels, styles in saved:
        for label in labels:
            expected = "--" if label.startswith("Scratch") else "-"
            assert styles[label] == expected, (name, label)
        if name.startswith("capacity"):
            assert set(labels) == {
                "32 → 32+32",
                "64 → 64+64",
                "Scratch 32+32",
                "Scratch 64+64",
            }
        elif name.startswith("supervision"):
            assert set(labels) == {
                "Neural-only",
                "Behavior-supervised",
                "Scratch BRAID",
            }
        else:
            assert set(labels) == {
                "A1_fw + K1_fw",
                "A1_fw + K1_fw + Cz",
                "Scratch BRAID",
            }


def test_queue_isolates_failure_and_reuses_completion(tmp_path, monkeypatch):
    """A failed dependency blocks descendants while independent jobs finish."""
    import json
    from experiments.multisession import runner

    specs = {
        'bad': dict(kind='source', dependencies=[], seed=42),
        'good': dict(kind='source', dependencies=[], seed=43),
        'reused': dict(kind='source', dependencies=[], seed=44),
        'dependent': dict(kind='adaptation', dependencies=['bad'], seed=42),
        'independent': dict(kind='adaptation', dependencies=['good'], seed=43),
    }
    for spec in specs.values():
        if spec['kind'] == 'source':
            spec.update(
                source_count=4, latent=32, objective='neural', sharing='A',
            )
        else:
            spec.update(
                session='held_out', fold=0, fraction=0.2,
                latent=32, source_count=4, objective='neural', sharing='A',
            )
    settings = dict(paths=dict(artifact_root=str(tmp_path)), runtime=dict(
        device='auto', parallel_workers=4, cpu_threads=4,
    ))
    plan = dict(jobs=specs, settings=settings,
                analyses={PRIMARY: list(specs)})
    path = tmp_path / 'plans' / 'test' / 'plan.json'
    path.parent.mkdir(parents=True)
    launch = tmp_path / 'logs'
    launch.mkdir()
    with job(runner.fit_directory(plan, 'reused'), specs['reused']):
        dump_pickle(runner.fit_directory(plan, 'reused') / 'value.pkl', 1)
    started = []

    class Process:
        def __init__(self, command, **kwargs):
            self.key = command[command.index('--job') + 1]
            self.pid = 100 + len(started)
            started.append(self.key)
            assert self.key != 'reused'
            if self.key != 'bad':
                directory = runner.fit_directory(plan, self.key)
                with job(directory, specs[self.key]):
                    dump_pickle(directory / 'value.pkl', 1)

        def poll(self):
            return 1 if self.key == 'bad' else 0

    monkeypatch.setattr(runner.subprocess, 'Popen', Process)
    monkeypatch.setattr(runner, 'select_gpus', lambda *args: [
        dict(uuid=f'GPU-{index}', index=index) for index in range(4)
    ])
    monkeypatch.setattr(runner, 'publish_reports', lambda *args: None)
    monkeypatch.setattr(runner, 'start_report', lambda *args: None)
    monkeypatch.setattr(runner.time, 'sleep', lambda seconds: None)
    runner.run_queue(plan, path, launch, settings)
    status = json.loads((launch / 'status.json').read_text())
    assert status['counts'] == dict(
        complete=3, failed=1, blocked=1, finished=5,
        running=0, remaining=0, total=5,
    )
    assert set(started) == {'bad', 'good', 'independent'}
    for key in started:
        matches = list((launch / 'jobs').rglob(f'*__{key}.log'))
        assert len(matches) == 1
    assert not list(launch.glob('*.log'))


def test_log_paths_and_analysis_manifest_are_compact(tmp_path):
    """Keep readable worker logs and store job specifications only in plan."""
    import json
    from experiments.multisession.log_paths import job_log_path
    from experiments.multisession.runner import reconcile

    source = dict(
        kind='source', seed=42, source_count=4, latent=32,
        objective='neural', sharing='AK',
    )
    target = dict(
        kind='adaptation', seed=42, session='held_out', fold=2,
        fraction=0.2, latent=32, source_count=4,
        objective='neural', sharing='AK',
    )
    jobs = {'source-id': source, 'target-id': target}
    launch = tmp_path / 'logs'
    source_log = job_log_path(launch, 'source-id', source)
    target_log = job_log_path(launch, 'target-id', target)
    assert source_log.parent == launch / 'jobs' / 'source'
    assert target_log.parent == launch / 'jobs' / 'adaptation' / 'held_out'
    assert 'fold2_labels20pct' in target_log.name
    assert 'sources4_neural_AK' in target_log.name

    plan = dict(
        jobs=jobs,
        settings=dict(paths=dict(artifact_root=str(tmp_path))),
        analyses={PRIMARY: list(jobs)},
    )
    path = tmp_path / 'plans' / 'plan-id' / 'plan.json'
    path.parent.mkdir(parents=True)
    reconcile(plan, path, {key: 'pending' for key in jobs})
    manifest = json.loads((
        tmp_path / 'analysis' / PRIMARY / 'plan-id' / 'manifest.json'
    ).read_text())
    assert manifest['members'] == {
        key: {'state': 'pending'} for key in jobs
    }
    assert 'source_states' not in manifest
    assert 'preprocessing_states' not in manifest


def test_reduced_plan_reuses_source_and_target_identities(tmp_path):
    """Grid membership and storage locations never change retained IDs."""
    from experiments.multisession.store import FIT_STORES, fit_directory

    settings = read_yaml(
        ROOT / "assets/config/nhp/experiments/multi_session_training.yaml"
    )
    for key in ("data", "model"):
        settings[key] = read_yaml(Path(settings["modules"][key]))
    settings["paths"] = dict(artifact_root=str(tmp_path))
    sessions = {s: dict(sha256=s) for s in settings["held_out_sessions"]}
    sessions.update({f"source_{i}": dict(sha256=str(i)) for i in range(33)})
    full = build_plan(
        dict(settings, seeds=[42, 43, 44],
             target_fractions=[0.05, 0.1, 0.2, 0.4, 0.8]),
        sessions,
    )
    reduced = build_plan(settings, sessions)
    for key, spec in reduced["jobs"].items():
        assert full["jobs"][key] == spec
        directory = fit_directory(reduced, key)
        if spec["kind"] in FIT_STORES:
            assert directory.is_relative_to(tmp_path / "experiments")
        else:
            assert directory.is_relative_to(tmp_path / PRIMARY)
    legacy = dict(reduced)
    legacy.pop("storage")
    key = next(k for k, s in reduced["jobs"].items() if s["kind"] == "scratch")
    assert fit_directory(legacy, key) == tmp_path / PRIMARY / "fits" / key


def test_queue_shared_source_resource():
    """Waiting for common preprocessing must not occupy another GPU."""
    from experiments.multisession.queue import resource_available

    spec = dict(kind="source", objective="behavior", seed=42,
                sessions=["a", "b"])
    jobs = dict(first=spec, second=dict(spec, sharing="AK"),
                independent=dict(kind="scratch"))
    active = {0: ("first", object())}
    assert not resource_available("second", jobs, active)
    assert resource_available("independent", jobs, active)
    assert resource_available("second", jobs, {})


def test_adopted_worker_pid_reuse(tmp_path, monkeypatch):
    """An unrelated process reusing a worker PID is never adopted as live."""
    from experiments.multisession import queue

    plan = dict(jobs=dict(fit=dict(kind="source")),
                settings=dict(paths=dict(artifact_root=str(tmp_path))))
    monkeypatch.setattr(queue, "process_state", lambda pid: ("S", "old"))
    worker = queue.AdoptedWorker(123, plan, "fit")
    assert worker.poll() is None
    monkeypatch.setattr(queue, "process_state", lambda pid: ("S", "new"))
    assert worker.poll() == 1


def test_preprocessing_cache_identity_and_corruption(tmp_path, monkeypatch):
    """Only identical checkpoint/data bytes reuse precomputed predictions."""
    import pytest
    from experiments.multisession import preprocessing

    calls = []

    class Preprocessor:
        missing_marker = -1000000

        def restoreModels(self):
            calls.append("restore")

    def predict(model, **kwargs):
        calls.append("predict")
        return ([kwargs["Z"].copy()],)

    monkeypatch.setattr(preprocessing, "runPredict", predict)
    pre = tmp_path / "preprocessing" / "fit"
    pre.mkdir(parents=True)
    (pre / "model.p").write_bytes(b"checkpoint-one")
    arrays = {
        name + suffix: np.ones((width, LENGTH))
        for name, width in (("Y", 3), ("U", 2), ("Z", 4))
        for suffix in ("", "_validation")
    }
    first = preprocessing.preprocessing_predictions(
        Preprocessor(), arrays, pre, "implementation"
    )
    assert calls == ["restore", "predict", "predict"]
    second = preprocessing.preprocessing_predictions(
        Preprocessor(), arrays, pre, "implementation"
    )
    assert len(calls) == 3
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left, right)
    changed = dict(arrays, Y=arrays["Y"] * 2)
    preprocessing.preprocessing_predictions(
        Preprocessor(), changed, pre, "implementation"
    )
    assert len(calls) == 6
    caches = list(
        tmp_path.glob("BRAID_preprocessing_predictions/*/behavior.npz")
    )
    assert len(caches) == 2
    for path in caches:
        path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        preprocessing.preprocessing_predictions(
            Preprocessor(), arrays, pre, "implementation"
        )


def test_snapshot_preserves_live_graphs_and_aliases(tmp_path):
    """Saving native weight containers must not replace live model objects."""
    from BRAID.MainModel import MainModel
    from BRAID.BRAIDModel import BRAIDModel
    from BRAID.tools.model_base_classes import ReconstructionInfo
    from experiments.multisession.preprocessing import save_snapshot
    from experiments.multisession.store import load_pickle

    pre = MainModel()
    model = BRAIDModel(preprocessing_model=pre)
    model.sId_pre = pre
    model.sId = MainModel()
    model.sId.model1_Cy = MixedForecaster([(3, 3)], 4, "A", 42).sessions[0]["C"]
    live = model.sId.model1_Cy
    path = tmp_path / "snapshot.p"
    save_snapshot(model, path)
    assert model.sId.model1_Cy is live
    stored = load_pickle(path)["model"]
    assert stored.sId_pre is stored.preprocessing_model
    assert isinstance(stored.sId.model1_Cy, ReconstructionInfo)
    for left, right in zip(live.get_weights(), stored.sId.model1_Cy.weights):
        np.testing.assert_array_equal(left, right)


def test_fit_identity_uses_active_module_versions(monkeypatch):
    """Irrelevant source edits preserve IDs; used module versions change IDs."""
    from experiments.multisession import compatibility, plan
    from experiments.storage.scientific_modules import MODULE_VERSIONS

    settings = read_yaml(
        ROOT / "assets/config/nhp/experiments/multi_session_training.yaml"
    )
    model = read_yaml(Path(settings["modules"]["model"]))
    source = plan.implementation_digest("source")
    target = plan.implementation_digest("adaptation", model)
    monkeypatch.setattr(
        compatibility, "source_code_digest", lambda: "changed"
    )
    assert plan.implementation_digest("source") == source
    assert plan.implementation_digest("adaptation", model) == target
    monkeypatch.setitem(
        MODULE_VERSIONS, "braid.dynamics.causal_transformer", 2
    )
    assert plan.implementation_digest("adaptation", model) == target
    monkeypatch.setitem(MODULE_VERSIONS, "multisession.source", 2)
    assert plan.implementation_digest("source") != source
    assert plan.implementation_digest("adaptation", model) == target
    monkeypatch.setitem(
        MODULE_VERSIONS, "braid.dynamics.multilayer_perceptron", 2
    )
    assert plan.implementation_digest("adaptation", model) != target


def test_non_neural_tail_does_not_activate_second_dynamics():
    """A post-stage width does not contribute stage-two dynamics."""
    from experiments.storage.scientific_modules import active_stages

    assert active_stages({"nx": 64, "n1": 32, "n3": 32}) == (1,)
    assert active_stages({"nx": 64, "n1": 32, "n3": 16}) == (1, 2)
    assert active_stages({"nx": 64, "n1": 0, "n3": 32}) == (1, 2)


def test_quarantine_incomplete_components(tmp_path):
    """A stage with events but no history is isolated on target retry."""
    from experiments.multisession.store import (
        quarantine_incomplete_components,
    )

    fit = tmp_path / "fits" / "fit-id"
    root = tmp_path / "target_components" / "fit-id" / "main"
    complete = root / "01_complete"
    interrupted = root / "02_interrupted"
    complete.mkdir(parents=True)
    interrupted.mkdir()
    (complete / "complete.json").write_text("{}")
    event = interrupted / "train" / "events.out.tfevents.test"
    event.parent.mkdir()
    event.write_text("event")

    moved = quarantine_incomplete_components(fit)

    assert len(moved) == 1
    assert (moved[0] / "train" / event.name).read_text() == "event"
    assert (complete / "complete.json").exists()
    assert not interrupted.exists()
    assert quarantine_incomplete_components(fit) == []


def test_status_does_not_report_exited_workers_as_running(
    tmp_path, monkeypatch,
):
    """Status derives current running counts from PID start ticks."""
    from experiments.multisession import queue_status

    status = dict(
        coordinator_pid=100,
        coordinator_start_ticks="old",
        updated_at_utc="2026-10-20T00:00:00+00:00",
        phase="active",
        counts=dict(
            total=4, complete=1, failed=0, blocked=0,
            running=2, remaining=1,
        ),
        active={
            "0": dict(pid=200, start_ticks="same"),
            "1": dict(pid=201, start_ticks="reused"),
        },
    )
    (tmp_path / "status.json").write_text(json.dumps(status))
    states = {
        100: ("Z", "old"),
        200: ("S", "same"),
        201: ("S", "different"),
    }
    monkeypatch.setattr(
        queue_status, "process_state", lambda pid: states.get(pid)
    )
    actual = queue_status.inspect_status(tmp_path)
    assert actual["phase"] == "stopped"
    assert actual["counts"]["running"] == 1
    assert actual["counts"]["remaining"] == 2
    assert set(actual["active"]) == {"0"}


def test_frozen_launch_source_survives_worktree_edits(
    tmp_path, monkeypatch,
):
    """A later repository edit cannot change a pinned launch's code."""
    from experiments import campaign_source as compatibility

    repository = tmp_path / "repository"
    source = repository / "source"
    config = repository / "assets" / "config"
    source.mkdir(parents=True)
    config.mkdir(parents=True)
    original = source / "numerical.py"
    original.write_text("VERSION = 1\n")
    (config / "default.yaml").write_text("model: {}\n")
    monkeypatch.setattr(compatibility, "SOURCE_ROOT", source)
    launch = tmp_path / "launch"
    launch.mkdir()
    digest = compatibility.freeze_source(launch)
    original.write_text("VERSION = 2\n")
    assert compatibility.source_code_digest() != digest
    assert compatibility.source_code_digest(launch / "source") == digest
    monkeypatch.setattr(
        compatibility, "SOURCE_ROOT", launch / "source"
    )
    compatibility.verify_source_code(digest)
    snapshot_config = launch / "assets" / "config" / "default.yaml"
    snapshot_config.write_text("model: {changed: true}\n")
    assert compatibility.source_code_digest() != digest
