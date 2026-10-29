"""Verify neural-only scheduling and fixed common-session scoring.

Inputs are synthetic source plans and evaluation arrays. Outputs are only
assertions; no experiment artifacts or fits are created.
"""

from pathlib import Path

import numpy as np

from experiments.multisession.behavior_compute_scaling import budget_queue
from experiments.multisession.behavior_compute_scaling import pipeline
from experiments.multisession.behavior_compute_scaling.budget_parallel import (
    budget_identity, validate_plan,
)

COUNTS = (4, 8, 16, 33)
CHECKPOINTS = (40, 80, 160, 320, 640, 1280, 1920)


def source_plan(root: Path) -> dict:
    """Construct a four-count neural-only source plan."""
    jobs = {}
    for count in COUNTS:
        jobs[str(count)] = dict(
            kind="source", source_count=count,
            objective="neural", implementation="stable",
            source_training=dict(
                checkpoints=list(CHECKPOINTS),
                update_multipliers={
                    "4": 1, "8": 1, "16": 1, "33": 2,
                },
            ),
        )
    return dict(
        name="multi_session_self_supervised_scaling",
        source_only=True, jobs=jobs,
        source_references=list(jobs),
        settings=dict(paths=dict(artifact_root=str(root))),
    )


def test_extended_budget_uses_saved_path_without_refitting(
    tmp_path: Path,
) -> None:
    """An overlapping neural budget resolves to its completed old fit."""
    from experiments.multisession.behavior_compute_scaling import (
        budget_parallel,
    )

    plan = source_plan(tmp_path)
    plan["jobs"]["4"]["source_training"]["checkpoints"] = [
        160, 320, 640, 1280, 1920, 2560, 3840,
    ]
    saved = tmp_path / "previous" / "budget160"
    plan["continuation"] = {
        "4": {"budgets": {"160": str(saved)}},
    }
    assert budget_parallel.budget_directory(plan, "4", 160) == saved
    assert budget_parallel.budget_directory(
        plan, "4", 2560
    ) != saved


def test_extended_schedule_changes_only_budget_fields() -> None:
    """The main and decay ladder may extend without changing sampling."""
    from experiments.multisession.compute_scaling.self_supervised import (
        schedule_without_ladder,
    )

    original = dict(
        checkpoints=[40, 80], total_updates=[50, 100],
        maximum_updates=80, learning_rate=0.001,
        session_batch_sizes={"4": 8},
    )
    extended = dict(
        original, checkpoints=[160, 320],
        total_updates=[200, 400], maximum_updates=320,
    )
    assert schedule_without_ladder(original) == (
        schedule_without_ladder(extended)
    )


def test_seen_session_paths_wait_for_own_count(
    tmp_path: Path, monkeypatch,
) -> None:
    """Four seen-session paths release after their own source fit."""
    plan = source_plan(tmp_path)
    validate_plan(plan)
    tasks = budget_queue.task_plan(plan)["jobs"]
    assert len(tasks) == 56
    sessions = [
        row for row in tasks.values()
        if row["task_kind"] == "session"
    ]
    assert len(sessions) == 16
    assert sum(
        row["task_kind"] == "full" for row in tasks.values()
    ) == 4
    assert {
        (row["source_count"], row["session_index"]) for row in sessions
    } == {
        (count, index) for count in COUNTS for index in range(4)
    }
    assert budget_identity(plan, "33", 80)["budget"] == 80
    states = {key: "pending" for key in tasks}
    pending = set(tasks)
    monkeypatch.setattr(
        budget_queue, "prefix_component",
        lambda unused_plan, unused_launch, key, budget: None,
    )
    monkeypatch.setattr(
        budget_queue, "completed",
        lambda path: {} if "fits/4" in str(path) else None,
    )
    ready = budget_queue.ready_tasks(
        plan, tmp_path, tasks, states, pending, {},
    )
    assert {
        (tasks[key]["source_count"], tasks[key]["task_kind"])
        for key in ready if tasks[key]["task_kind"] == "session"
    } == {(4, "session")}


def test_score_component_uses_only_named_common_sessions(
    monkeypatch,
) -> None:
    """Four common inner losses determine a larger source fit score."""
    training = [dict(T=np.zeros((2, 9, 3))) for _ in range(8)]
    holdout = [dict(T=np.ones((2, 9, 3))) for _ in range(8)]
    names = [f"common_{index}" for index in range(4)]
    seen = []

    def evaluate(unused_model: object, selected: list[dict]) -> tuple:
        """Record the evaluated window subset and return stage losses."""
        seen.append(len(selected))
        return [1.0] * len(selected), 1

    monkeypatch.setattr(pipeline, "evaluate_sessions", evaluate)
    score = pipeline.score_component(
        object(), training, holdout, names,
    )
    assert seen == [4]
    assert len(score["per_session"]) == 4
    assert score["mean_loss"] == 1.0

def test_transplanted_core_stays_frozen_during_behavior_update(
    monkeypatch, tmp_path,
) -> None:
    """The adapted RNN1 updates private K/C without moving A/Afw/Kfw."""
    from experiments.multisession.behavior_supervision import (
        source_adaptation as adaptation,
    )
    from experiments.multisession.compute_scaling.self_supervised import (
        neural_model,
    )
    from experiments.multisession.compute_scaling.training import (
        balanced_update, gradient_function, optimizer_groups,
    )
    from experiments.multisession.compute_scaling.flops import update_flops
    from experiments.multisession.training import recurrent_data

    plan = {"jobs": {"4": {"latent": 16, "seed": 42}}}
    pretrained = neural_model([(3, 3)] * 4, plan["jobs"]["4"])
    payload = {"weights": pretrained.weights()}
    monkeypatch.setattr(
        adaptation, "selected_backbone",
        lambda unused_plan, unused_key: ({}, tmp_path),
    )
    monkeypatch.setattr(
        adaptation, "load_pickle", lambda unused_path: payload,
    )
    model, core = adaptation.initialized_rnn1(
        plan, "4", 0, 3, 4,
    )
    assert not model.shared_variables
    before = model.weights()
    rng = np.random.default_rng(42)
    values = dict(
        Y=rng.normal(size=(10, 16, 3)).astype("float32"),
        U=rng.normal(size=(10, 16, 2)).astype("float32"),
        Z=rng.normal(size=(10, 16, 4)).astype("float32"),
    )
    data = [recurrent_data(values, "Z")]
    optimizers = optimizer_groups(model, adaptation.recurrent_settings())
    functions = [gradient_function(model, 0)]
    loss, sampled = balanced_update(
        model, data, optimizers, rng, 8, 4,
        ("/CPU:0", "/CPU:0"), functions,
    )
    assert np.isfinite(loss) and len(sampled[0]) == 8
    adaptation.verify_frozen_core(model, core)
    arithmetic = update_flops(model, data, 8, 4)
    trainable = sum(
        int(value.shape.num_elements())
        for group in model.private_variables for value in group
    )
    assert arithmetic["optimizer"] == 14 * trainable
    after = model.weights()
    assert any(
        not np.array_equal(old, new)
        for old, new in zip(before["0/C"], after["0/C"])
    )

def test_full_path_waits_for_four_completed_session_paths(
    tmp_path: Path, monkeypatch,
) -> None:
    """A finished source fit alone cannot complete its full BRAID task."""
    from experiments.multisession.behavior_supervision import (
        source_adaptation as adaptation,
    )

    plan = source_plan(tmp_path)
    tasks = budget_queue.task_plan(plan)["jobs"]
    full = next(
        row for row in tasks.values()
        if row["source_count"] == 4
        and row["task_kind"] == "full"
    )
    monkeypatch.setattr(
        budget_queue, "completed",
        lambda path: {} if "fits/4" in str(path) else None,
    )
    monkeypatch.setattr(
        adaptation, "session_directory",
        lambda unused_plan, unused_key, index: (
            tmp_path / f"session{index}"
        ),
    )
    monkeypatch.setattr(
        adaptation, "full_directory",
        lambda unused_plan, unused_key: (_ for _ in ()).throw(
            AssertionError("Full identity read before sessions finish")
        ),
    )
    assert not budget_queue.task_complete(plan, tmp_path, full)

def test_seen_session_report_rows_keep_one_session_field(
    tmp_path: Path, monkeypatch,
) -> None:
    """The report records four session paths without duplicate CSV keys."""
    import json

    from experiments.multisession.compute_scaling import (
        self_supervised_report as report,
    )

    names = [f"common_{index}" for index in range(4)]
    for index, name in enumerate(names):
        directory = tmp_path / f"session{index}"
        directory.mkdir()
        (directory / "path.json").write_text(json.dumps(dict(
            session=name, session_index=index,
            behavior_rnn1_flops=100,
            neural_readout_flops=200,
        )))
    monkeypatch.setattr(
        report, "session_directory",
        lambda unused_plan, unused_key, index: (
            tmp_path / f"session{index}"
        ),
    )
    source = dict(
        count=4, key="fit", spec=dict(sessions=names),
    )
    rows = report.session_training_rows({}, [source])
    assert len(rows) == 4
    assert [row["session"] for row in rows] == names
    assert all(row["source_count"] == 4 for row in rows)
