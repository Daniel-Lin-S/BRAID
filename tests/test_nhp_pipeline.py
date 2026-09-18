"""Real-data regressions for alignment, leakage, caching and plugins."""

import os
from pathlib import Path

import numpy as np
import pytest
import yaml

from experiments.cache import cached
from experiments.contracts import FeatureSet, plugin
from experiments.nhp import NHPDataset, fold_features, temporal_segments
from experiments.windows import window_indices
from BRAID.MainModel import shift_ms_to_1s_series
from BRAID.sequence import independent_indices, window_shift

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = (
    Path(os.environ["NHP_DATA_ROOT"])
    if "NHP_DATA_ROOT" in os.environ
    else None
)


@pytest.fixture(scope="module")
def real_session(tmp_path_factory):
    """Load the actual first Indy session without waveform snippets."""
    if DATA_ROOT is None or not DATA_ROOT.is_dir():
        pytest.skip("Real NHP dataset is unavailable on this host.")
    settings = yaml.safe_load(
        (ROOT / "assets/config/nhp/data/spikes.yaml").read_text()
    )
    settings.update(
        root=str(DATA_ROOT),
        cache_mode="reuse",
        cache_root=str(tmp_path_factory.mktemp("nhp-cache")),
    )
    dataset = NHPDataset(settings)
    return dataset, dataset.load("indy_20160407_02")


def test_alignment_and_unit_mapping(real_session):
    """Verify timestamps, channel IDs, and first nonempty dimensions."""
    dataset, session = real_session
    arrays = session.arrays
    np.testing.assert_allclose(np.diff(arrays["t"]), 0.05, atol=1e-8)
    assert arrays["Z"].shape == arrays["U"].shape == (len(arrays["t"]), 2)
    assert arrays["Y"].shape[1] == len(arrays["ids"])
    assert all(name.startswith("M1 ") for name in arrays["ids"])
    assert np.all(np.diff(arrays["spike_offsets"]) > 0)
    assert (
        dataset.inventory()[session.metadata["session"]]
        == arrays["ids"].tolist()
    )


def test_backward_velocity_and_segment_endpoints(real_session):
    """Differentiate within splits and drop unsupported samples."""
    dataset, session = real_session
    position = dataset.fold(session, 0, False)
    velocity = dataset.fold(session, 0, True)
    for segment in np.unique(position.arrays["segment"]):
        old = position.arrays["segment"] == segment
        new = velocity.arrays["segment"] == segment
        np.testing.assert_allclose(
            velocity.arrays["Z"][new, 2:],
            np.diff(position.arrays["Z"][old], axis=0) / 0.05,
        )
        np.testing.assert_array_equal(
            velocity.arrays["indices"][new],
            position.arrays["indices"][old][1:],
        )


def test_no_preprocessing_leakage_from_test(real_session):
    """Changing heldout observations cannot change any training feature."""
    dataset, session = real_session
    reference = fold_features(session, dataset.settings, 0, True)
    changed = FeatureSet(
        {k: v.copy() for k, v in session.arrays.items()}, session.metadata
    )
    test_block = np.array_split(np.arange(len(changed.arrays["t"])), 5)[0]
    changed.arrays["Y"][test_block] *= 10
    changed.arrays["Z"][test_block] += 100
    actual = fold_features(changed, dataset.settings, 0, True)
    train = reference.arrays["role"] == 0
    for key in ("Y", "Z", "U", "t"):
        np.testing.assert_array_equal(
            actual.arrays[key][train], reference.arrays[key][train]
        )


def test_windows_never_cross_segments(real_session):
    """Every independent input sequence stays within one role and segment."""
    dataset, session = real_session
    fold = dataset.fold(session, 2, True)
    for role in (0, 1, 2):
        indices = window_indices(fold, role, 128)
        assert np.all(fold.arrays["role"][indices] == role)
        assert np.all(np.ptp(fold.arrays["segment"][indices], axis=1) == 0)
        assert np.all(np.diff(fold.arrays["indices"][indices], axis=1) == 1)


def test_independent_target_indices(real_session):
    """No valid multistep target reaches into a different sequence."""
    inputs, targets, masks = independent_indices(128 * 4, 128, [1, 2, 4, 8], 2)
    for target, mask in zip(targets, masks):
        np.testing.assert_array_equal(
            inputs[~mask] // 128, target[~mask] // 128
        )
    _, session = real_session
    values = session.arrays["Z"][:256]
    shifted = window_shift(
        shift_ms_to_1s_series, values, [1, 4], 128, -1000000.0, time_first=True
    )
    assert np.all(shifted[1][128:131] == -1000000.0)
    np.testing.assert_array_equal(shifted[1][131:256], values[128:253])


def test_cache_hits_and_corruption_are_explicit(real_session, tmp_path):
    """Verify reuse avoids extraction and corruption cannot become a hit."""
    _, session = real_session
    builds = []

    def build():
        builds.append(True)
        return FeatureSet(
            {"position": session.arrays["Z"][:128]}, {"real": True}
        )

    first = cached(tmp_path, "test", {"sigma": 50}, "reuse", build)
    second = cached(tmp_path, "test", {"sigma": 50}, "reuse", build)
    assert len(builds) == 1
    np.testing.assert_array_equal(
        first.arrays["position"], second.arrays["position"]
    )
    with (first.path / "arrays.npz").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        cached(tmp_path, "test", {"sigma": 50}, "reuse", build)
    cached(tmp_path, "test", {"sigma": 50}, "rebuild", build)
    third = cached(tmp_path, "test", {"sigma": 25}, "reuse", build)
    assert third.path != first.path


def test_dataset_plugin_extension(real_session):
    """Resolve dataset implementations without adding branches to the runner."""
    dataset, _ = real_session
    loaded = plugin("experiments.nhp:NHPDataset", settings=dataset.settings)
    assert loaded.sessions() == dataset.sessions()


def test_empty_segment_rejected():
    """Reject a split whose guard destroys a usable segment."""
    with pytest.raises(ValueError, match="empty/short"):
        temporal_segments(100, 5, 0, 36)


def test_window_padding_is_not_a_logged_error(real_session):
    """Mask declared per-window padding while retaining every real sample."""
    from BRAID.MainModel import getLossLogStr
    from BRAID.tools.tf_losses import masked_mse

    _, session = real_session
    truth = session.arrays["Z"][:256]
    prediction = truth.copy()
    marker = -1000000.0
    prediction[np.arange(256) % 128 < 3] = marker
    text = getLossLogStr(
        truth,
        prediction.T,
        [4],
        "cont",
        [masked_mse(marker)],
        window_length=128,
        missing_marker=marker,
    )
    assert "MSE=0" in text


def test_restore_best_weights_at_epoch_cap(real_session):
    """Restore best validation weights when the epoch limit is reached."""
    from BRAID.tools.model_base_classes import EarlyStoppingWithMinEpochs

    _, session = real_session

    class WeightHolder:
        stop_training = False

        def __init__(self):
            self.weights = [session.arrays["Z"][:1].copy()]

        def get_weights(self):
            return [value.copy() for value in self.weights]

        def set_weights(self, values):
            self.weights = values

    model = WeightHolder()
    callback = EarlyStoppingWithMinEpochs(
        monitor="val_loss", patience=10, restore_best_weights=True
    )
    callback.set_model(model)
    callback.on_train_begin()
    reference = model.get_weights()
    callback.on_epoch_end(0, {"val_loss": 1.0})
    model.weights[0] += 1
    callback.on_epoch_end(1, {"val_loss": 2.0})
    callback.on_train_end()
    np.testing.assert_array_equal(model.get_weights()[0], reference[0])


def test_model_plugin_configuration_has_independent_stages():
    """Instantiate the configured adapter without changing runner code."""
    backend = plugin(
        "experiments.braid_backend:BRAIDBackend",
        configuration=str(ROOT / "assets/config/nhp/models/BRAID.yaml"),
        seed=123,
    )
    assert backend.seed == 123
    assert backend.arguments["args_base"]["independent_windows"]
    assert backend.arguments["args_base"]["honor_explicit_validation"]
    assert backend.arguments["args_base"]["restore_best_weights"]


def test_preview_windows_are_deterministic_and_isolated(real_session):
    """Matching plots select reproducible five-second segment interiors."""
    from experiments.previews import preview_windows

    dataset, session = real_session
    features = dataset.fold(session, 0, False)
    settings = dict(seed=42, seconds=5, windows=3)
    first = preview_windows(features, settings)
    second = preview_windows(features, settings)
    assert len(first) == 3
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left, right)
        assert len(left) == 100
        assert len(np.unique(features.arrays["segment"][left])) == 1
