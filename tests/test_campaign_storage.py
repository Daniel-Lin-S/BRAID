"""Check shared storage separation and completed-campaign reuse."""

from pathlib import Path
import json
from threading import Event

import pytest

from experiments.campaign_runner import run_with_status
from experiments.storage.analysis import initialize_specification
from experiments.storage.artifacts import (
    publish_model_settings, recipe_directory, validate_model_settings,
)
from experiments.storage.campaign import (
    bind_analysis, campaign_analysis, campaign_member, validate_roots,
)


def test_recipe_and_analysis_use_existing_shared_layout(tmp_path):
    """Repeated launches reference the same scientific model and fit."""
    recipe = dict(
        model_plugin="experiments.models.braid_backend:BRAIDBackend",
        model=dict(nx=64, n1=32),
    )
    artifact = tmp_path / "artifacts"
    model = recipe_directory(artifact, recipe)
    publish_model_settings(model, recipe)
    assert validate_model_settings(model)["settings"] == recipe
    original = (model / "model_settings.json").read_bytes()
    publish_model_settings(model, recipe)
    assert (model / "model_settings.json").read_bytes() == original
    fit = model / "session" / "fold_0" / "existing-fit-id"
    members = {"case/session": dict(
        fit=str(fit.relative_to(artifact)), fit_id=fit.name,
        role="fit", reused=True,
    )}
    analysis = initialize_specification(
        artifact, dict(name="dynamics_model_comparison"), members, {},
    )
    paths = dict(
        artifact_root=str(artifact), log_root=str(tmp_path / "logs"),
        cache_root=str(tmp_path / "cache"),
    )
    for name in ("first", "second"):
        logs = tmp_path / "logs" / name
        bind_analysis(logs, paths, analysis)
        assert campaign_analysis(logs) == analysis
        assert campaign_member(logs, "case/session")[0] == fit
        assert {p.name for p in logs.iterdir()} == {"launch.json"}
    assert model.parent == artifact / "experiments"
    assert analysis.parent == artifact / "analysis/dynamics_model_comparison"


def test_overlapping_storage_roots_fail_before_writing(tmp_path):
    """Misconfigured artifact roots cannot put scientific output in logs."""
    with pytest.raises(ValueError, match="separate artifact_root"):
        validate_roots(dict(
            log_root=tmp_path / "logs", artifact_root=tmp_path / "logs/fits",
            cache_root=tmp_path / "cache",
        ))
    with pytest.raises(ValueError, match="Missing shared artifact references"):
        campaign_analysis(tmp_path / "missing")
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("fail", [False, True])
def test_coordinator_publishes_terminal_liveness(tmp_path, fail):
    """Success and failure cannot leave the final heartbeat marked active."""
    def watch(root: Path, stop: Event) -> None:
        stop.wait()

    def work() -> None:
        if fail:
            raise RuntimeError("Test worker failure")

    if fail:
        with pytest.raises(RuntimeError, match="Test worker failure"):
            run_with_status(tmp_path, {}, watch, work)
    else:
        run_with_status(tmp_path, {}, watch, work)
    record = json.loads((tmp_path / "launch.json").read_text())
    assert record["campaign_active"] is False
    assert record["ended_at_utc"]
