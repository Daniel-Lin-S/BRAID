"""Verify source-only continuation, immutable ancestry, and monitor budgets.

Synthetic arrays use native (windows, 128, channels) layouts. Temporary fit
stores exercise completed preprocessing reuse, exact main-prefix restoration,
and full-path arithmetic without retaining experiment artifacts.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.multisession.behavior_compute_scaling import pipeline
from experiments.multisession.behavior_compute_scaling.ancestry import (
    inherited_paths, main_prefix, prefix_flops, validate_continuation,
)
from experiments.multisession.behavior_compute_scaling.report import report
from experiments.multisession.store import completed, job
from experiments.storage.cache import atomic_json


def test_extended_pipeline_reuses_only_matching_inputs(
    tmp_path, monkeypatch,
) -> None:
    """Continuation keeps fixed targets and excludes old decay FLOPs."""
    from experiments.multisession.behavior_supervision import models

    for name in ("PREPROCESSING_SIZE", "BEHAVIOR_WIDTH", "NEURAL_WIDTH"):
        monkeypatch.setattr(models, name, 2)
    rng = np.random.default_rng(45)

    def rows(sizes: list[int]) -> list[dict]:
        return [dict(
            Y=rng.normal(size=(size, 128, 3)).astype("f"),
            U=rng.normal(size=(size, 128, 2)).astype("f"),
            Z=rng.normal(size=(size, 128, 4)).astype("f"),
        ) for size in sizes]

    train, test, inner = rows([3] * 4), rows([11, 15, 48, 13]), rows([2] * 4)
    names = [f"session_{index}" for index in range(4)]
    records = {name: {"validation": list(range(len(row["Y"])))}
               for name, row in zip(names, test)}
    settings = dict(
        learning_rate=1e-3, warmup_updates=1,
        warmup_start_learning_rate=1e-7,
        checkpoints=[1, 2], total_updates=[2, 3],
        update_multipliers={"4": 1}, session_batch_sizes={"4": 1},
        fixed_exposure=8, validation_exposure=4, fixed_patience=10,
    )
    spec = dict(
        sessions=names, source_count=4, sources={}, data={}, model={},
        latent=2, seed=42, source_training=settings, implementation="test",
        sharing="A1_fw_RNN2", objective="behavior",
    )
    original = tmp_path / "fits" / "original"
    root = tmp_path / "source_components"
    devices = ("/CPU:0", "/CPU:0")
    with job(original, spec):
        atomic_json(original / "data.json", records)
        pipeline.fit_source_campaign(
            train, test, records, original, root, spec, devices, 1, inner,
        )
    marker = completed(original)
    old_paths = inherited_paths(original)
    old_components = {p.name: completed(p) for p in root.iterdir()
                      if p.is_dir()}
    resumed = main_prefix(
        root / old_paths[-1]["stages"]["behavior_relevant"]["component_id"]
    )
    extended = dict(
        spec, source_training=dict(
            settings, checkpoints=[1, 2, 4, 6], total_updates=[2, 3, 5, 8],
        ), continuation=dict(
            source_fit=original.name, checksums=marker["checksums"],
            completed_total_updates=[2, 3],
        ),
    )
    destination = tmp_path / "fits" / "extended"
    with job(destination, extended):
        atomic_json(destination / "data.json", records)
        pipeline.fit_source_campaign(
            train, test, records, destination, root, extended,
            devices, 1, inner,
        )
    new_paths = json.loads((destination / "budgets.json").read_text())
    assert [p["total_updates"] for p in new_paths] == [5, 8]
    assert len(inherited_paths(destination)) == 4
    assert completed(original) == marker
    assert all(completed(root / key) == value
               for key, value in old_components.items())
    assert json.loads((original / "preprocessing.json").read_text()) == (
        json.loads((destination / "preprocessing.json").read_text())
    )
    for path in new_paths:
        stages = path["stages"]
        for stage in ("preprocessing_neural", "preprocessing_behavior"):
            assert stages[stage] == old_paths[-1]["stages"][stage]
        rnn1 = root / stages["behavior_relevant"]["component_id"]
        main = main_prefix(rnn1)
        branch_work = pipeline.load_flops(rnn1)
        assert stages["behavior_relevant"]["component_flops"] == (
            prefix_flops(main) + branch_work
        )
        chain = []
        while main is not None:
            chain.append(main)
            identity = json.loads((main / "identity.json").read_text())
            assert not identity["branch"]
            main = root / identity["parent"] if identity["parent"] else None
        assert resumed in chain
        assert path["full_path_flops"] == sum(
            value["component_flops"] for value in stages.values()
        )
        for downstream, upstream in (
            ("neural_readout", "behavior_relevant"),
            ("residual", "neural_readout"),
        ):
            component = root / stages[downstream]["component_id"]
            identity = json.loads((component / "identity.json").read_text())
            assert identity["upstream_checkpoint"] == (
                stages[upstream]["component_id"]
            )
            assert component.name not in old_components
    with pytest.raises(ValueError, match="splits or scales"):
        validate_continuation(extended, {}, original.parent)


def test_source_only_report_never_collects_target_jobs(monkeypatch) -> None:
    """Source-only queue publication cannot require or enqueue targets."""
    import importlib

    module = importlib.import_module(report.__module__)
    snapshots = []
    monkeypatch.setattr(module, "source_report", lambda *args: snapshots)
    monkeypatch.setattr(module, "campaign_complete", lambda *args: (
        pytest.fail("Source-only report accessed target jobs.")
    ))
    report(dict(source_only=True, settings=dict(
        source_training=dict(protocol="continuation"),
    )), Path("/unused/plan.json"), final=True)


def test_monitor_uses_extension_budgets(tmp_path, monkeypatch) -> None:
    """Live training tags distinguish 2x and 3x endpoints for 33 sources."""
    from experiments.multisession.behavior_compute_scaling import monitor

    class Writer:
        def flush(self) -> None:
            pass

    points = []
    monkeypatch.setattr(monitor, "write_scalar", lambda *args: (
        points.append(args[1:4])
    ))
    log = tmp_path / "worker.log"
    log.write_text(
        "2026-10-27 12:00:00,000 INFO behavior_relevant "
        "update=1300/2560 loss=1.2 lr=0.001\n"
        "2026-10-27 12:01:00,000 INFO residual "
        "update=3200/3200 loss=3.0 lr=0\n"
        "2026-10-27 12:02:00,000 INFO behavior_relevant "
        "update=2600/3840 loss=1.1 lr=0.001\n"
    )
    reader = monitor.SourceLogReader(log, 33, Writer(), (1600, 2400))
    reader.scan()
    assert reader.last["budget"] == 4800
    assert points[0][0] == "budget_3200/behavior_relevant/training_loss"
    previous = list(points)
    reader.scan()
    assert points == previous


def test_continuation_plan_excludes_targets_and_runtime_identity(
    tmp_path, monkeypatch,
) -> None:
    """Source-only plans pin ancestry but ignore GPU/thread assignments."""
    from copy import deepcopy
    from experiments.multisession.behavior_compute_scaling import plan

    monkeypatch.setattr(plan, "source_implementation", lambda: "science")
    monkeypatch.setattr(plan, "fit_directory", lambda *args: tmp_path)
    monkeypatch.setattr(plan, "completed", lambda *args: {
        "checksums": {"checkpoint": "checksum"},
    })
    settings = dict(
        data={}, model={}, latent=16, seeds=[42],
        source_counts=list(plan.COUNTS), held_out_sessions=[],
        evaluation_horizons=[1, 2, 4, 8, 16, 32],
        source_training=dict(
            checkpoints=list(plan.CHECKPOINTS),
            total_updates=list(plan.TOTAL_UPDATES), maximum_updates=640,
        ), runtime=dict(cpu_threads=4, device="auto"),
    )
    sessions = {str(index): dict(checksum=index) for index in range(33)}
    jobs = {f"old{count}": dict(
        kind="source", source_count=count,
        sessions=list(sessions)[:count],
        sources={name: sessions[name] for name in list(sessions)[:count]},
    ) for count in plan.COUNTS}
    reference = dict(settings=deepcopy(settings), jobs=jobs,
                     source_references=list(jobs))
    settings["source_training"].update(
        checkpoints=list(plan.CHECKPOINTS + plan.EXTENSION_CHECKPOINTS),
        total_updates=list(plan.TOTAL_UPDATES + plan.EXTENSION_TOTAL_UPDATES),
        maximum_updates=1920,
    )
    first = plan.build_extension_plan(settings, sessions, reference)
    settings["runtime"] = dict(cpu_threads=2, device="different GPUs")
    second = plan.build_extension_plan(settings, sessions, reference)
    assert first["jobs"] == second["jobs"]
    assert first["source_only"]
    assert len(first["jobs"]) == 4
    assert {spec["kind"] for spec in first["jobs"].values()} == {"source"}
    assert all(not spec["dependencies"] for spec in first["jobs"].values())
    assert {spec["continuation"]["source_fit"]
            for spec in first["jobs"].values()} == set(jobs)
