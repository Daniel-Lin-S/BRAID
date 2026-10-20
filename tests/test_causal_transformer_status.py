"""Verify live campaign counts from preprocessing and fit markers."""

from __future__ import annotations

import fcntl
import json
import os
import time

import pytest

from experiments.nhp_dynamics_study import (
    CURRENT_CONTROL_TARGET_DIGEST, LEGACY_CONTROL_TARGET_DIGEST,
    LEGACY_TRANSFORMER_DIGEST, _fit_directory,
    _preprocessing_directory, _reusable_completed_fit, target_key,
)
from experiments.storage.cache import file_digest, fingerprint
from experiments.campaign_status import (
    refresh as refresh_plan, relocate_events as relocate_component_events,
    snapshot as snapshot_plan,
)
from experiments.nhp_dynamics_study import status_plan



def snapshot(root, reused_keys=None, active=None, active_since_ns=None):
    return snapshot_plan(
        root, status_plan(root, reused_keys), active, active_since_ns
    )

def refresh(root):
    return refresh_plan(root, status_plan(root, None))

def relocate_events(root, log_directory):
    return relocate_component_events(
        root / "target_components", log_directory
    )

def test_snapshot_counts_reused_and_local_states(tmp_path):
    """Count reused completion, running jobs, and failed jobs separately."""
    reused = target_key("indy_20160921_01", 0, 42)
    active = target_key("indy_20160921_01", 0, 43)
    reused_directory = _preprocessing_directory(tmp_path, reused)
    reused_directory.mkdir(parents=True)
    (reused_directory / "complete.json").write_text("{}")
    active_directory = _preprocessing_directory(tmp_path, active)
    active_directory.mkdir(parents=True)
    (active_directory / "status.json").write_text(
        json.dumps({"state": "running"})
    )
    running_fit = _fit_directory(tmp_path, "lstm", active)
    running_fit.mkdir(parents=True)
    (running_fit / "status.json").write_text(
        json.dumps({"state": "running"})
    )
    failed_fit = _fit_directory(tmp_path, "mlp", active)
    failed_fit.mkdir(parents=True)
    (failed_fit / "status.json").write_text(
        json.dumps({"state": "failed"})
    )

    record = snapshot(tmp_path)

    assert record["preprocessing"] == {
        "finished": 1, "running": 1, "interrupted": 0,
        "failed": 0, "pending": 34, "total": 36, "remaining": 35,
    }
    assert record["fits"] == {
        "finished": 1, "running": 1, "interrupted": 0,
        "failed": 1, "pending": 177, "total": 180, "remaining": 179,
        "reused_finished": 1, "local_finished": 0,
    }


def test_snapshot_counts_linked_completed_control_as_reused(tmp_path):
    """A validated control linked from a prior campaign is reused."""
    key = target_key("indy_20160921_01", 0, 43)
    source = tmp_path / "archive" / "mlp_control"
    source.mkdir(parents=True)
    (source / "complete.json").write_text("{}")
    linked = _fit_directory(tmp_path, "mlp", key)
    linked.parent.mkdir()
    linked.symlink_to(source, target_is_directory=True)

    record = snapshot(tmp_path)

    assert record["fits"]["finished"] == 1
    assert record["fits"]["reused_finished"] == 1
    assert record["fits"]["local_finished"] == 0


def test_refresh_preserves_launch_metadata(tmp_path):
    """Keep process and log metadata when publishing progress."""
    launch = tmp_path / "launch.json"
    launch.write_text(json.dumps({"pid": 123, "log": "campaign-log"}))

    record = refresh(tmp_path)

    assert record["pid"] == 123
    assert record["log"] == "campaign-log"
    assert record["fits"]["remaining"] == 180
    assert json.loads(launch.read_text()) == record


def test_snapshot_excludes_old_running_markers(tmp_path):
    """A new coordinator does not inherit stale running counts."""
    key = target_key("indy_20160921_01", 0, 43)
    old_fit = _fit_directory(tmp_path, "transformer_both", key)
    new_fit = _fit_directory(tmp_path, "transformer_stage1", key)
    old_fit.mkdir(parents=True)
    new_fit.mkdir(parents=True)
    for directory in (old_fit, new_fit):
        (directory / "status.json").write_text(
            json.dumps({"state": "running"})
        )
    start = time.time_ns()
    os.utime(old_fit / "status.json",
             ns=(start - 1_000_000_000, start - 1_000_000_000))
    os.utime(new_fit / "status.json",
             ns=(start + 1_000_000_000, start + 1_000_000_000))

    lock = new_fit.with_suffix(".lock")
    with lock.open("w") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        record = snapshot(
            tmp_path, active=True, active_since_ns=start
        )

    assert record["fits"]["running"] == 1
    assert record["fits"]["interrupted"] == 1


def test_relocate_events_preserves_component_artifacts(tmp_path):
    """Move only TensorBoard streams while keeping checkpoint files."""
    root = tmp_path / "outputs"
    component = root / "target_components" / "fit" / "01_neural_dynamics"
    component.mkdir(parents=True)
    (component / "component.json").write_text("{}")
    (component / "history.jsonl").write_text('{"epoch": 1}\n')
    train = component / "train"
    train.mkdir()
    (train / "events.out.tfevents.test").write_bytes(b"training")
    validation = component / "validation"
    validation.mkdir()
    (validation / "events.out.tfevents.test").write_bytes(b"validation")
    logs = tmp_path / "logs"

    assert relocate_events(root, logs) == 2
    assert relocate_events(root, logs) == 0
    assert train.is_symlink()
    assert validation.is_symlink()
    assert train.resolve().parent == validation.resolve().parent
    assert (train / "events.out.tfevents.test").read_bytes() == b"training"
    assert (validation / "events.out.tfevents.test").read_bytes() == (
        b"validation"
    )
    assert (component / "history.jsonl").read_text() == (
        '{"epoch": 1}\n'
    )


def test_verified_reference_counts_before_local_preprocessing(tmp_path):
    """Count validated reused fits even when local prep is pending."""
    key = target_key("indy_20160921_01", 0, 42)

    record = snapshot(tmp_path, {key})

    assert record["preprocessing"]["finished"] == 0
    assert record["fits"]["finished"] == 1
    assert record["fits"]["reused_finished"] == 1
    assert record["fits"]["remaining"] == 179


def test_refresh_reports_stranded_jobs_as_interrupted(tmp_path):
    """Dead coordinators cannot leave false live-fit counts."""
    key = target_key("indy_20160921_01", 0, 43)
    directory = _fit_directory(tmp_path, "transformer_both", key)
    directory.mkdir(parents=True)
    (directory / "status.json").write_text(
        json.dumps({"state": "running"})
    )
    (tmp_path / "launch.json").write_text(
        json.dumps({"pid": 999999999})
    )

    record = refresh(tmp_path)

    assert record["campaign_active"] is False
    assert record["fits"]["running"] == 0
    assert record["fits"]["interrupted"] == 1


def test_refresh_detects_fit_lock_when_pid_is_hidden(tmp_path):
    """A held fit lock keeps an inaccessible coordinator active."""
    key = target_key("indy_20160921_01", 0, 43)
    directory = _fit_directory(tmp_path, "transformer_both", key)
    directory.mkdir(parents=True)
    (directory / "status.json").write_text(
        json.dumps({"state": "running"})
    )
    (tmp_path / "launch.json").write_text(
        json.dumps({"pid": 999999999})
    )
    with directory.with_suffix(".lock").open("w") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        record = refresh(tmp_path)
    assert record["campaign_active"] is True
    assert record["fits"]["running"] == 1


def test_legacy_control_reuse_validates_scientific_identity(tmp_path):
    """A Transformer-only fix preserves matching completed controls."""
    directory = tmp_path / "fit"
    directory.mkdir()
    payload = directory / "payload.txt"
    payload.write_text("saved control")
    current = {
        "case": "mlp", "selection": {"batch_size": 10},
        "implementation": {
            "target": CURRENT_CONTROL_TARGET_DIGEST,
            "transformer": LEGACY_TRANSFORMER_DIGEST,
        },
    }
    legacy = json.loads(json.dumps(current))
    legacy["implementation"]["target"] = (
        LEGACY_CONTROL_TARGET_DIGEST
    )
    identity_path = directory / "identity.json"
    identity_path.write_text(json.dumps(legacy))
    (directory / "complete.json").write_text(json.dumps({
        "identity_hash": fingerprint(legacy),
        "checksums": {
            "payload.txt": file_digest(payload),
            "identity.json": file_digest(identity_path),
        },
    }))

    assert _reusable_completed_fit(directory, current)
    changed = json.loads(json.dumps(current))
    changed["selection"]["batch_size"] = 11
    with pytest.raises(ValueError, match="different scientific"):
        _reusable_completed_fit(directory, changed)


def test_unreviewed_implementation_cannot_reuse_completed_fit(tmp_path):
    """Unknown raw-code hashes require an explicit compatibility review."""
    directory = tmp_path / "fit"
    directory.mkdir()
    saved = {
        "case": "transformer_both",
        "selection": {"batch_size": 10},
        "implementation": {
            "target": "unknown-source",
            "transformer": "unknown-transformer",
        },
    }
    identity_path = directory / "identity.json"
    identity_path.write_text(json.dumps(saved))
    payload = directory / "payload.txt"
    payload.write_text("completed")
    (directory / "complete.json").write_text(json.dumps({
        "identity_hash": fingerprint(saved),
        "checksums": {
            "identity.json": file_digest(identity_path),
            "payload.txt": file_digest(payload),
        },
    }))
    current = json.loads(json.dumps(saved))
    current["implementation"] = {"modules": {"braid.training": 1}}

    with pytest.raises(ValueError, match="Unreviewed"):
        _reusable_completed_fit(directory, current)
