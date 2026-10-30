"""Check complete budget pools and independent checkpoint selection.

Synthetic source paths exercise report eligibility without training or files.
"""

from copy import deepcopy
from pathlib import Path

import pytest

from experiments.multisession.behavior_compute_scaling import (
    preprocessing_comparison as comparison,
)


def test_full_pool_and_independent_best_selection(monkeypatch) -> None:
    """Both models need seven budgets, but their winners may differ."""
    sources = []
    paths = {}
    for variant in range(2):
        group = []
        for count in comparison.COUNTS:
            fit = Path(f"/synthetic/{variant}/{count}")
            totals = [
                value * (2 if count == 33 else 1)
                for value in comparison.BUDGET_POOL
            ]
            winner = len(totals) - 1 - variant
            paths[fit] = [
                dict(
                    budget=total * 4 // 5, total_updates=total,
                    selection_score=abs(index - winner),
                    full_path_flops=total * 100,
                )
                for index, total in enumerate(totals)
            ]
            group.append(dict(
                count=count, fit=fit,
                spec=dict(sessions=["same"], latent=16,
                          sharing="same", seed=42),
                selection=dict(evaluation_set_id="identical"),
                path=paths[fit][winner],
            ))
        sources.append(group)
    monkeypatch.setattr(comparison, "inherited_paths", paths.__getitem__)
    comparison.validate_selected_sources(*sources)
    wrong = deepcopy(sources)
    wrong[0][0]["path"] = paths[sources[0][0]["fit"]][0]
    with pytest.raises(ValueError, match="not the best"):
        comparison.validate_selected_sources(*wrong)
    paths[sources[0][0]["fit"]] = paths[sources[0][0]["fit"]][:-1]
    with pytest.raises(ValueError, match="complete budget pool"):
        comparison.validate_selected_sources(*sources)


def test_session_sd_is_sample_sd_and_ratios_are_not_clipped() -> None:
    """Values above dummy remain valid; error bars describe recordings."""
    import numpy as np

    rows = [dict(
        variant=comparison.VARIANTS[0], source_count=4,
        group="position", horizon=1, session=str(index),
        cc=value, r2=value, mse_ratio=value, mse=value, dummy_mse=1.0,
    ) for index, value in enumerate((0.5, 1.0, 1.5, 2.0))]
    summary = comparison.session_summary(rows)[0]
    assert summary["mse_ratio"] == 1.25
    assert summary["mse_ratio_sd"] == pytest.approx(
        np.std([0.5, 1.0, 1.5, 2.0], ddof=1)
    )
    rows[0]["mse_ratio"] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        comparison.session_summary(rows)


def test_matched_spec_caps_and_private_initialization(monkeypatch) -> None:
    """Both architectures preserve the four-source per-recording protocol."""
    from experiments.multisession.behavior_compute_scaling import (
        single_session_baseline as baseline,
    )

    names = [f"session_{index}" for index in range(4)]
    inventory = {name: dict(scales={
        "Y": dict(keep=[True] * 5), "Z": dict(keep=[True] * 4),
    }) for name in names}
    plan = dict(
        settings=dict(paths=dict(artifact_root="/synthetic")),
        jobs=dict(reference=dict(
            sessions=names, sources={name: {} for name in names},
            source_count=4, seed=42, latent=16,
            source_training=dict(warmup_updates=10),
        )),
    )
    monkeypatch.setattr(
        baseline, "reference", lambda _: ("reference", inventory)
    )
    for mode in ("recorded", "generated"):
        plan["jobs"]["reference"]["behavior_supervision"] = dict(target=mode)
        spec, _, directory = baseline.saved_spec(plan, 2, matched=True)
        settings = spec["source_training"]
        assert settings["checkpoints"] == [1920]
        assert settings["total_updates"] == [2400]
        assert settings["fixed_update_cap"] == 2000
        assert settings["validation_exposure"] == 200
        assert settings["session_batch_sizes"] == {"1": 8}
        assert spec["initialization_session_index"] == 2
        assert spec["behavior_supervision"]["target"] == mode
        assert directory.is_absolute()
    assert plan["jobs"]["reference"]["source_count"] == 4


def test_inner_selection_never_reads_outer_test(monkeypatch) -> None:
    """Inner ranking wins even when saved outer scores prefer another path."""
    import numpy as np
    from experiments.multisession.behavior_compute_scaling import pipeline
    from experiments.multisession.behavior_supervision import models

    class Model:
        component = ""

    names = [str(index) for index in range(4)]
    common = {}
    for name in names:
        partition = dict(
            Y=np.ones((2, 128, 3), dtype="f"),
            U=np.ones((2, 128, 2), dtype="f"),
            Z=np.ones((2, 128, 4), dtype="f"),
        )
        common[name] = dict(training=partition, inner=partition)
        # Deliberately omit test: selection must never access it.
    paths = [dict(
        budget=budget, total_updates=budget, full_path_flops=budget,
        selection_score=-budget,
        stages={stage: dict(component_id=str(budget)) for stage in (
            "behavior_relevant", "neural_readout", "residual"
        )},
    ) for budget in (200, 400)]
    source = dict(
        spec=dict(sessions=names, behavior_supervision=dict(target="recorded")),
        inventory={name: dict(
            scales={"Y": dict(keep=[True] * 3),
                    "Z": dict(keep=[True] * 4)},
            inner_validation=[[1]],
        ) for name in names},
        paths=paths, fit=Path("/synthetic/fits/source"), count=4,
        selection=dict(selected_budget=400),
    )
    for factory in ("rnn1_model", "neural_readout_model", "rnn2_model"):
        monkeypatch.setattr(models, factory, lambda *args: Model())
    monkeypatch.setattr(
        pipeline, "restore_component",
        lambda model, path: setattr(model, "component", path.name),
    )
    monkeypatch.setattr(pipeline, "residual_data", lambda _, __, data: data)
    monkeypatch.setattr(pipeline, "score_component", lambda model, *args: dict(
        mean_ratio=float(model.component), per_session=[],
    ))
    chosen, rows = pipeline.inner_source_choices([source], common)
    assert chosen[0]["path"]["budget"] == 200
    assert chosen[0]["selection"]["selection_role"] == (
        "four_common_inner_validation"
    )
    assert len(rows) == 2


def test_worker_command_uses_importable_module() -> None:
    """Detached workers must not attempt to execute the __main__ module."""
    inputs = {key: "/synthetic/input" for key in (
        "with_plan", "with_source_report", "direct_plan", "direct_report"
    )}
    command = comparison.matched_command(inputs, "matched-worker")
    assert command[2] == comparison.MODULE
    assert "__main__" not in command


def test_paired_views_match_recordings_and_orient_all_metrics() -> None:
    """Both references use the same recording and positive means better."""
    scores, controls = [], []
    sessions = [f"session_{index}" for index in range(4)]
    for variant in comparison.VARIANTS:
        for group in comparison.GROUPS:
            for horizon in comparison.HORIZONS:
                for index, session in enumerate(sessions):
                    common = dict(
                        variant=variant, group=group, horizon=horizon,
                        session=session, samples=100,
                        dummy_mse=1.0, evaluation_set_id="fixed",
                    )
                    controls.append(dict(
                        common, source_count=1, cc=0.4, r2=0.2,
                        mse=0.8, mse_ratio=0.8,
                    ))
                    for count in comparison.COUNTS:
                        gain = (count - 4) * 0.001 * (index + 1)
                        scores.append(dict(
                            common, source_count=count,
                            cc=0.5 + gain, r2=0.3 + gain,
                            mse=0.7 - gain,
                            mse_ratio=0.7 - gain,
                        ))
    single = comparison.paired_gain_rows(
        scores, controls, "single_session"
    )
    four = comparison.paired_gain_rows(scores, controls, "four_source")
    selected = [row for row in four if (
        row["variant"] == comparison.VARIANTS[0]
        and row["source_count"] == 33
        and row["group"] == "position"
        and row["horizon"] == 1
        and row["metric"] == "mse_ratio"
    )]
    assert [row["gain"] for row in selected] == pytest.approx(
        [0.029, 0.058, 0.087, 0.116]
    )
    summary = comparison.gain_summary(four)
    point = next(row for row in summary if (
        row["reference"] == "four_source"
        and row["variant"] == comparison.VARIANTS[0]
        and row["source_count"] == 33
        and row["group"] == "position"
        and row["horizon"] == 1
        and row["metric"] == "mse_ratio"
    ))
    assert point["mean_gain"] == pytest.approx(0.0725)
    assert point["improved_sessions"] == 4
    assert point["worsened_sessions"] == 0
    assert len(single) == len(four) == 1152
    bad = [dict(row) for row in controls]
    bad[0]["dummy_mse"] = 2.0
    with pytest.raises(ValueError, match="different samples"):
        comparison.paired_gain_rows(scores, bad, "single_session")
