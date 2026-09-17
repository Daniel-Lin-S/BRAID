"""Persist per-component epoch histories and recoverable weight snapshots.

Each component directory contains history.jsonl (epoch, attempt, metrics),
initial.weights.h5, latest.weights.h5, and architecture.json. Nonfinite
metrics are recorded as null with a warning; nonfinite loss aborts fitting.
"""

import json
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
