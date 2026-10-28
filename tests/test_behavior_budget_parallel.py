"""Check budget task identity, release, and runtime-only scheduling.

Inputs are small synthetic source plans. Outputs are assertions that the
four-count DAG has independent budget work and stable scientific identities.
"""

from pathlib import Path

import pytest

from experiments.multisession.behavior_compute_scaling import budget_queue
from experiments.multisession.behavior_compute_scaling.budget_parallel import (
    budget_identity, validate_plan,
)


COUNTS = (4, 8, 16, 33)
CHECKPOINTS = (40, 80, 160, 320, 640, 1280, 1920)


def source_plan(tmp_path: Path) -> dict:
    """Create a four-count direct plan with a single nested design."""
    jobs = {}
    for count in COUNTS:
        jobs[str(count)] = dict(
            kind="source", source_count=count, seed=42,
            latent=16, objective="behavior", sharing="A1_fw_RNN2",
            behavior_supervision=dict(target="recorded"),
            source_training=dict(
                checkpoints=list(CHECKPOINTS),
                update_multipliers={
                    "4": 1, "8": 1, "16": 1, "33": 2,
                },
            ),
            implementation="unchanged_scientific_version",
        )
    return dict(
        name="multi_session_behavior_scaling_no_preprocess",
        source_only=True, jobs=jobs,
        source_references=list(jobs),
        settings=dict(paths=dict(artifact_root=str(tmp_path))),
    )


def test_four_gpu_dag_has_all_independent_budget_tasks(tmp_path) -> None:
    """Each source count has one prefix, seven paths and a finalizer."""
    plan = source_plan(tmp_path)
    validate_plan(plan)
    tasks = budget_queue.task_plan(plan)["jobs"]
    assert len(tasks) == 36
    assert sum(row["task_kind"] == "prefix" for row in tasks.values()) == 4
    assert sum(row["task_kind"] == "budget" for row in tasks.values()) == 28
    assert sum(row["task_kind"] == "final" for row in tasks.values()) == 4
    assert {
        row["budget"] for row in tasks.values()
        if row["task_kind"] == "budget" and row["source_count"] == 33
    } == {2 * value for value in CHECKPOINTS}


def test_budget_releases_when_its_prefix_is_complete(
    tmp_path, monkeypatch,
) -> None:
    """A ready budget need not wait for all seven main checkpoints."""
    plan = source_plan(tmp_path)
    tasks = budget_queue.task_plan(plan)["jobs"]
    pending = set(tasks)
    states = {key: "pending" for key in tasks}
    before = budget_queue.ready_tasks(
        plan, tmp_path, tasks, states, pending, {},
    )
    assert {tasks[key]["task_kind"] for key in before} == {"prefix"}
    monkeypatch.setattr(
        budget_queue, "prefix_component",
        lambda unused_plan, unused_launch, key, budget: (
            tmp_path if key == "4" and budget == 40 else None
        ),
    )
    after = budget_queue.ready_tasks(
        plan, tmp_path, tasks, states, pending, {},
    )
    first = tasks[after[0]]
    assert first["task_kind"] == "budget"
    assert first["source_count"] == 4
    assert first["budget"] == 40


def test_budget_identity_excludes_runtime_device(tmp_path) -> None:
    """A one-GPU relaunch keeps the original scientific source fit ID."""
    plan = source_plan(tmp_path)
    identity = budget_identity(plan, "33", 80)
    assert identity["source_fit"] == "33"
    assert "device" not in identity
    assert "microbatch" not in identity
    with pytest.raises(ValueError, match="Unknown source budget"):
        budget_identity(plan, "33", 40)
