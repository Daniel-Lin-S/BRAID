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
