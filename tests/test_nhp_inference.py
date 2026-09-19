"""Validate prediction contracts and opt-in real checkpoint inference.

Set NHP_INFERENCE_RUN to a completed fit directory to check native checkpoint
reconstruction and independent windows. Production artifacts are read-only;
no fitting, prediction publication or evaluation publication is performed.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from experiments.fitting import validate_forecast_arrays

SAMPLES = 8
WINDOW = 4
HORIZONS = [1, 2]
DIMENSIONS = {"Y": 3, "Z": 2, "X": 4}
RTOL = 1e-5
ATOL = 1e-5


@pytest.mark.parametrize(
    "fault", ["shape", "nan", "infinity", "missing", "mask", "mask_type"],
)
def test_invalid_prediction_contract_is_rejected(fault):
    """Malformed forecasts must never reach prediction publication."""
    arrays = {
        name: np.ones((len(HORIZONS), SAMPLES, channels))
        for name, channels in DIMENSIONS.items()
    }
    arrays["valid"] = np.array([
        np.arange(SAMPLES) % WINDOW >= h for h in HORIZONS
    ])
    if fault == "shape":
        arrays["X"] = arrays["X"][:, :, :1]
    elif fault == "nan":
        arrays["Y"][0, 2, 0] = np.nan
    elif fault == "infinity":
        arrays["Z"][0, 2, 0] = np.inf
    elif fault == "missing":
        del arrays["X"]
    elif fault == "mask":
        arrays["valid"][0, 0] = True
    else:
        arrays["valid"] = arrays["valid"].astype(int)
    with pytest.raises(ValueError, match="shape|nonfinite|valid mask"):
        validate_forecast_arrays(arrays, SAMPLES, 3, 2, 4, HORIZONS, WINDOW)


@pytest.fixture(scope="module")
def real_checkpoint():
    """Read recorded held-out inputs under a filesystem write guard."""
    from test_nhp_readonly_identification import forbid_root_writes
    from experiments.braid_backend import BRAIDBackend
    from experiments.contracts import FeatureSet
    from experiments.windows import window_indices

    reference = os.environ.get("NHP_INFERENCE_RUN")
    if reference is None:
        pytest.skip("Set NHP_INFERENCE_RUN for read-only checkpoint probes.")
    run = Path(reference).resolve()
    runtime = json.loads((run / "provenance" / "runtime.json").read_text())
    cache = Path(runtime["cache"])
    with forbid_root_writes([run, cache]):
        with np.load(cache / "arrays.npz", allow_pickle=False) as saved:
            features = FeatureSet(dict(saved), {})
        with np.load(run / "data" / "selection.npz") as saved:
            columns = saved["selected_columns"]
        config = run / "configuration" / "model_configuration.yaml"
        backend = BRAIDBackend(str(config))
        backend.load(run / "checkpoints" / "model.p")
        indices = window_indices(features, 2, backend.length).ravel()
        y = features.arrays["Y"][indices][:, columns]
        u = features.arrays["U"][indices]
        arguments = json.loads(
            (run / "configuration" / "fit_arguments.json").read_text()
        )
        snapshots = json.loads(
            (run / "configuration" / "resolved_configurations.json")
            .read_text()
        )
        yield dict(
            run=run, backend=backend, features=features, columns=columns,
            y=y, u=u, horizons=snapshots["evaluation"]["horizons"],
            identity={"case": {"dimensions": {"nx": arguments["nx"]}}},
        )


def test_checkpoint_reconstruction_uses_identical_inputs(real_checkpoint):
    """Two independently restored instances agree at matching input shapes."""
    from experiments.braid_backend import BRAIDBackend

    probe = real_checkpoint
    run, backend = probe["run"], probe["backend"]
    other = BRAIDBackend(str(run / "configuration/model_configuration.yaml"))
    other.load(run / "checkpoints" / "model.p")
    count = backend.length
    args = (probe["y"][:count], probe["u"][:count], probe["horizons"])
    left, right = backend.predict(*args), other.predict(*args)
    np.testing.assert_array_equal(left["valid"], right["valid"])
    for name in DIMENSIONS:
        np.testing.assert_allclose(
            left[name], right[name], rtol=RTOL, atol=ATOL
        )


def test_windows_are_independent_of_neighbor_inputs(real_checkpoint):
    """Reordering complete windows preserves each window's own forecasts."""
    probe = real_checkpoint
    backend = probe["backend"]
    length = backend.length
    y, u = probe["y"][:2 * length], probe["u"][:2 * length]
    order = np.r_[np.arange(length, 2 * length), np.arange(length)]
    original = backend.predict(y, u, probe["horizons"])
    swapped = backend.predict(y[order], u[order], probe["horizons"])
    for name in DIMENSIONS:
        np.testing.assert_allclose(
            original[name][:, order], swapped[name], rtol=RTOL, atol=ATOL,
        )


def test_completed_checkpoint_prediction_payload(real_checkpoint):
    """The production inference path accepts the checkpoint without writes."""
    from experiments.artifacts import validate_completion
    from experiments.fitting import prediction_arrays

    probe = real_checkpoint
    validate_completion(probe["run"])
    payload = prediction_arrays(
        probe["backend"], probe["run"], probe["identity"],
        probe["features"], probe["columns"], probe["horizons"],
    )
    assert payload["Y"].shape[1] == len(probe["y"])
    assert all(values.size and np.isfinite(values).all()
               for values in payload.values())
