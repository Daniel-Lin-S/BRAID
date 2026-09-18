"""Persist per-component epoch histories and recoverable weight snapshots.

Each component directory contains history.jsonl (epoch, attempt, metrics),
initial.weights.h5, latest.weights.h5, and architecture.json. Nonfinite
metrics are recorded as null with a warning; nonfinite loss aborts fitting.
"""

import json
from artifact_io import file_digest as checkpoint_digest
import logging
import os
from pathlib import Path

import numpy as np
import tensorflow as tf

LOGGER = logging.getLogger(__name__)


class EpochArtifacts(tf.keras.callbacks.Callback):
    """Flush training evidence independently of text-log verbosity."""

    def __init__(self, directory: str, attempt: int) -> None:
        super().__init__()
        if not directory:
            raise ValueError("Epoch artifacts require an absolute log_dir.")
        self.directory = Path(directory).resolve()
        self.attempt = attempt

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        """Publish metrics and weights after each completed epoch."""
        self.directory.mkdir(parents=True, exist_ok=True)
        metrics = {}
        for key, value in (logs or {}).items():
            value = float(value)
            if not np.isfinite(value):
                LOGGER.warning(
                    "Undefined training metric %s at %s", key, self.directory
                )
                value = None
            metrics[key] = value
        if metrics.get("loss") is None:
            raise FloatingPointError("Training loss is nonfinite.")
        if "val_loss" in metrics and metrics["val_loss"] is None:
            raise FloatingPointError("Validation loss is nonfinite.")
        row = dict(epoch=epoch + 1, attempt=self.attempt, metrics=metrics)
        with (self.directory / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary = self.directory / "pending.weights.h5"
        self.model.save_weights(str(temporary))
        os.replace(temporary, self.directory / "latest.weights.h5")
        if epoch == 0:
            self.model.save_weights(str(self.directory / "initial.weights.h5"))
            (self.directory / "architecture.json").write_text(
                self.model.to_json(), encoding="utf-8"
            )


def restore_completed_component(
    model: object,
    directory: str,
) -> tf.keras.callbacks.History | None:
    """Restore a completed component without invoking another fit.

    Parameters
    ----------
    model : keras.Model
        Constructed component whose weights are restored in place.
    directory : str
        Component directory inside a verified compatible run.

    Returns
    -------
    keras.callbacks.History or None
        Recorded selected history, or None if no completion record exists.
    """
    root = Path(directory)
    path = root / "complete.json"
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    for name, expected in record["checksums"].items():
        if checkpoint_digest(root / name) != expected:
            raise ValueError(
                f"Completed component checksum mismatch: {root / name}"
            )
    model.load_weights(str(root / "completed.weights.h5"))
    if any(not np.isfinite(value).all() for value in model.get_weights()):
        raise ValueError(f"Completed component has nonfinite weights: {root}")
    rows = [
        json.loads(line)
        for line in (root / "history.jsonl").read_text().splitlines()
    ]
    rows = [row for row in rows if row["attempt"] == record["attempt"]]
    if not rows:
        raise ValueError(f"Completed component has no selected history: {root}")
    history = tf.keras.callbacks.History()
    history.epoch = [row["epoch"] - 1 for row in rows]
    history.history = {
        key: [row["metrics"].get(key) for row in rows]
        for key in rows[0]["metrics"]
    }
    history.params = record["params"]
    LOGGER.info("Reusing completed component: %s", root.resolve())
    return history


def complete_component(model: object, directory: str, history: object) -> None:
    """Atomically mark selected, restored component weights as reusable."""
    root = Path(directory)
    if (root / "complete.json").exists():
        restore_completed_component(model, directory)
        return
    if any(not np.isfinite(value).all() for value in model.get_weights()):
        raise ValueError(f"Cannot complete nonfinite component: {root}")
    temporary = root / "completed.pending.weights.h5"
    model.save_weights(str(temporary))
    os.replace(temporary, root / "completed.weights.h5")
    params = {
        key: value
        for key, value in history.params.items()
        if key != "history_all"
    }
    record = dict(
        attempt=int(params["artifact_attempt"]),
        params=params,
        checksums={
            name: checkpoint_digest(root / name)
            for name in ("completed.weights.h5", "history.jsonl")
        },
    )
    temporary = root / "complete.pending.json"
    with temporary.open("w") as stream:
        json.dump(
            record,
            stream,
            allow_nan=False,
            default=lambda value: value.tolist(),
        )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, root / "complete.json")


def previous_attempts(directory: str) -> int:
    """Return the last recorded attempt without modifying partial history."""
    path = Path(directory) / "history.jsonl"
    if not path.exists():
        return 0
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if not rows:
        raise ValueError(f"Existing component history is empty: {path}")
    return max(row["attempt"] for row in rows)
