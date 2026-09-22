"""Validate component-owned TensorBoard streams and restart isolation."""

import json

import numpy as np
import pytest
import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
)
from tensorboard.util.tensor_util import make_ndarray

from BRAID.tools.tensorboard import (
    ComponentTensorBoard, event_scope, tensorboard_enabled,
    tensorboard_scope,
)
from experiments.cache import file_digest, writer_lock
from experiments.restart import prepare_components


def events(path):
    """Read the event file in one component split directory."""
    files = list(path.rglob("*tfevents*"))
    assert len(files) == 1
    assert files[0].parent == path
    return EventAccumulator(
        str(files[0]), size_guidance={"tensors": 0}
    ).Reload()


def test_tensorboard_scope_disables_events_but_keeps_artifacts(tmp_path):
    """Opt-out suppresses events without disabling component recovery data."""
    from BRAID.RegressionModel import RegressionModel

    directory = tmp_path / "component"
    values = np.arange(8, dtype=float)[None, :] / 8
    assert tensorboard_enabled()
    with tensorboard_scope(False):
        assert not tensorboard_enabled()
        model = RegressionModel(1, 1, log_dir=str(directory))
        model.fit(
            values, values, epochs=1, batch_size=4, verbose=0,
            epoch_artifacts=True,
        )
    assert tensorboard_enabled()
    assert not list(directory.rglob("*tfevents*"))
    assert (directory / "history.jsonl").is_file()
    assert (directory / "completed.weights.h5").is_file()
    assert (directory / "complete.json").is_file()


def test_callback_has_no_per_batch_work():
    """Keep TensorBoard serialization outside the training batch path."""
    assert "on_train_batch_end" not in ComponentTensorBoard.__dict__


def test_single_writer_synthetic_fit_and_retries(tmp_path):
    """Two internal fits share one file and cumulative epoch coordinates."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(1,)),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="sgd", loss="mse")
    data = np.arange(8, dtype=np.float32)[:, None] / 8
    expected = []
    with event_scope():
        for _ in range(2):
            callback = ComponentTensorBoard(str(tmp_path))
            history = model.fit(
                data,
                data,
                validation_data=(data, data),
                batch_size=4,
                epochs=2,
                callbacks=[callback],
                verbose=0,
            )
            expected.extend(history.history["val_loss"])
    train = events(tmp_path / "train")
    saved = events(tmp_path / "validation")
    assert {path.parent.name for path in tmp_path.rglob("*tfevents*")} == {
        "train", "validation"
    }
    assert "epoch_loss" in train.Tags()["tensors"]
    tags = saved.Tags()["tensors"]
    assert "epoch_loss" in tags
    assert not any(
        "evaluation_" in tag or "vs_iterations" in tag
        for tag in tags
    )
    values = saved.Tensors("epoch_loss")
    assert [value.step for value in values] == [0, 1, 2, 3]
    np.testing.assert_allclose(
        [make_ndarray(value.tensor_proto) for value in values],
        expected,
    )
    assert [v.step for v in train.Tensors("fitting/attempt")] == [0, 2]
    assert any(tag.endswith("/histogram") for tag in train.Tags()["tensors"])


def test_clean_names_collision_and_exception_flush(tmp_path):
    """Sanitize presentation only, reject collisions and close on failure."""
    model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))])
    logs = {"MSE_maskV_-1000000.0": 2.0, "val_CC_maskV_None": 0.5}
    with pytest.raises(RuntimeError, match="interrupted"):
        with event_scope():
            callback = ComponentTensorBoard(str(tmp_path))
            callback.set_model(model)
            callback.on_train_begin()
            callback.on_epoch_end(0, logs)
            assert "MSE_maskV_-1000000.0" in logs
            with pytest.raises(ValueError, match="collision"):
                callback.on_epoch_end(1, {"MSE": 1, "MSE_maskV_None": 2})
            raise RuntimeError("interrupted")
    train = events(tmp_path / "train")
    saved = events(tmp_path / "validation")
    assert "epoch_MSE" in train.Tags()["tensors"]
    assert "epoch_CC" in saved.Tags()["tensors"]
    with event_scope(), pytest.raises(ValueError, match="Uncleaned"):
        ComponentTensorBoard(str(tmp_path))


def component(root, name):
    """Create one registered incomplete component."""
    directory = root / "main" / name
    directory.mkdir(parents=True)
    (directory / "component.json").write_text(json.dumps({"name": name}))
    return directory


def test_cleanup_preserves_complete_and_restarts_partial(tmp_path):
    """Validate every completed component before touching partial outputs."""
    complete = component(tmp_path, "01_complete")
    for name in ("completed.weights.h5", "history.jsonl"):
        (complete / name).write_text("saved data")
    (complete / "complete.json").write_text(
        json.dumps(
            {
                "checksums": {
                    name: file_digest(complete / name)
                    for name in ("completed.weights.h5", "history.jsonl")
                },
            }
        )
    )
    before = {p: file_digest(p) for p in complete.iterdir()}
    partial = component(tmp_path, "02_partial")
    for name in (
        "history.jsonl",
        "latest.weights.h5",
        "events.out.tfevents.old",
    ):
        (partial / name).write_text("partial")
    unknown = partial / "unknown.data"
    unknown.write_text("must inspect")
    with pytest.raises(ValueError, match="Unrecognized"):
        prepare_components(tmp_path)
    assert (partial / "history.jsonl").exists()
    unknown.unlink()
    prepare_components(tmp_path)
    prepare_components(tmp_path)
    assert [p.name for p in partial.iterdir()] == ["component.json"]
    assert all(file_digest(p) == digest for p, digest in before.items())
    with event_scope():
        callback = ComponentTensorBoard(str(partial))
        callback.set_model(
            tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(1, input_shape=(1,)),
                ]
            )
        )
        callback.on_train_begin()
        callback.on_epoch_end(0, {"loss": 1})
    assert len(events(partial / "train").Tensors("epoch_loss")) == 1
    (complete / "history.jsonl").write_text("corrupt")
    with pytest.raises(ValueError, match="checksum"):
        prepare_components(tmp_path)
    assert {path.parent.name for path in partial.rglob("*tfevents*")} == {
        "train", "validation"
    }


def test_nonblocking_run_lock(tmp_path):
    """An active owner prevents a second invocation from preparing a run."""
    lock = tmp_path / "run.lock"
    with writer_lock(lock):
        with pytest.raises(RuntimeError, match="active writer"):
            with writer_lock(lock, blocking=False):
                pytest.fail("Contended lock entered")
