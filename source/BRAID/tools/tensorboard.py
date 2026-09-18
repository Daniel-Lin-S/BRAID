"""Write split-specific TensorBoard event files for each fitted component.

Each component contains ``train/`` and ``validation/`` event directories. Both
use matching epoch-indexed metric tags so TensorBoard overlays the split runs.
A fitting scope shares writers across retries and closes them on every exit.
Metric names are normalized by the Keras loss factories before they are logged.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
import re
from typing import Iterator

import tensorflow as tf

_WRITERS = ContextVar("component_event_writers", default=None)
_MASK_SUFFIX = re.compile(
    r"_maskV_(?:None|[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)$"
)


@contextmanager
def event_scope() -> Iterator[None]:
    """Share component writers across a fit, closing them even on failure."""
    if _WRITERS.get() is not None:
        yield
        return
    writers = {}
    token = _WRITERS.set(writers)
    try:
        yield
    finally:
        try:
            for state in writers.values():
                for writer in state["writers"].values():
                    writer.flush()
                    writer.close()
        finally:
            _WRITERS.reset(token)


class ComponentTensorBoard(tf.keras.callbacks.Callback):
    """Record scalar metrics and weight distributions once per epoch.

    Parameters
    ----------
    directory : str
        Absolute component directory, owned by the active fitting scope.
    """

    def __init__(self, directory: str) -> None:
        super().__init__()
        path = Path(directory).resolve()
        states = _WRITERS.get()
        if states is None:
            raise RuntimeError("TensorBoard requires an active fitting scope.")
        if path not in states:
            if list(path.rglob("*tfevents*")):
                raise ValueError(f"Uncleaned TensorBoard events at {path}")
            states[path] = dict(
                writers={
                    "train": tf.summary.create_file_writer(str(path / "train")),
                    "validation": tf.summary.create_file_writer(
                        str(path / "validation")
                    ),
                },
                next_epoch=0,
                attempts=0,
            )
        self.state = states[path]
        self.train_writer = self.state["writers"]["train"]
        self.validation_writer = self.state["writers"]["validation"]
        self.step = self.state["next_epoch"]

    def on_train_begin(self, logs: dict | None = None) -> None:
        """Record the boundary before a new internal fitting attempt."""
        self.offset = self.state["next_epoch"]
        self.state["attempts"] += 1
        with self.train_writer.as_default():
            tf.summary.text(
                "fitting/attempt",
                str(self.state["attempts"]),
                step=self.offset,
            )

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        """Write metrics without modifying the original Keras log mapping."""
        self.step = self.offset + epoch
        tags = {"train": {}, "validation": {}}
        for name, value in (logs or {}).items():
            split = "validation" if name.startswith("val_") else "train"
            metric = name[4:] if split == "validation" else name
            tag = f"epoch_{_MASK_SUFFIX.sub('', metric)}"
            if tag in tags[split]:
                raise ValueError(f"TensorBoard metric name collision: {tag}")
            tags[split][tag] = value
        for split, writer in self.state["writers"].items():
            with writer.as_default():
                for tag, value in tags[split].items():
                    tf.summary.scalar(tag, value, step=self.step)
        with self.train_writer.as_default():
            for weight in self.model.weights:
                tag = weight.name.replace(":", "_") + "/histogram"
                tf.summary.histogram(tag, weight, step=self.step)
        self.state["next_epoch"] = self.step + 1
        for writer in self.state["writers"].values():
            writer.flush()

    def on_train_end(self, logs: dict | None = None) -> None:
        """Flush the attempt while retaining writers for subsequent fits."""
        for writer in self.state["writers"].values():
            writer.flush()
