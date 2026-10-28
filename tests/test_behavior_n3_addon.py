"""Check input-only n3 semantics on independent synthetic windows.

Inputs are synthetic task target and recorded-behavior windows. Outputs are
assertions that n3 has no neural input and its loss scores additive forecasts.
"""

import numpy as np

from BRAID.multisession import HORIZONS
from experiments.multisession.behavior_compute_scaling.n3_addon import (
    n3_model, post_data,
)
from experiments.multisession.behavior_compute_scaling.training import (
    session_loss,
)
from experiments.multisession.training import select_batch


def test_n3_uses_u_only_and_scores_additive_behavior() -> None:
    """The one-state residual adds to RNN1 in recorded-Z units."""
    rng = np.random.default_rng(17)
    partitions = [
        dict(
            U=rng.normal(size=(2, 128, 2)).astype("f"),
            Z=rng.normal(size=(2, 128, 4)).astype("f"),
        ) for _ in range(2)
    ]
    prior = [
        rng.normal(size=(len(HORIZONS), 2, 128, 4)).astype("f")
        for _ in partitions
    ]
    rows = post_data(partitions, prior)
    model = n3_model(
        dict(count=2, spec=dict(seed=42)),
        ("/CPU:0", "/CPU:0"),
    )
    assert model.latent == 1
    assert rows[0]["I"].shape[-1] == 2
    assert rows[0]["P"].shape == prior[0].shape
    batch = select_batch(rows, [np.arange(2)] * 2)
    forecasts, _ = model.forecast(batch[0], batch[1])
    for index, forecast in enumerate(forecasts):
        expected = 0.0
        for step, horizon in enumerate(HORIZONS):
            error = (
                forecast[step, :, :-horizon].numpy()
                + prior[index][step, :, :-horizon]
                - partitions[index]["Z"][:, horizon:]
            )
            expected += float(np.square(error).mean(axis=(1, 2)).sum())
        local = tuple(group[index] for group in batch)
        np.testing.assert_allclose(
            float(session_loss(model, index, local)), expected,
            rtol=1e-5,
        )
