"""Configure logging and require a working, explicitly selected GPU.

GPU identity and the result of a real device operation are returned for
run metadata. CPU fallback is prohibited for training launch modes.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
import logging
from typing import Iterator
from uuid import uuid4
import os
from pathlib import Path
import subprocess


def launch_directory(settings: dict, stage: str) -> Path:
    """Create a unique log directory for one experiment-stage invocation.

    Parameters
    ----------
    settings : dict
        Resolved experiment and local path settings.
    stage : str
        Execution stage, such as fit, preprocess or plot.

    Returns
    -------
    Path
        Absolute, newly created log directory.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    directory = (
        Path(settings["paths"]["log_root"])
        / settings["experiment"]["name"]
        / stage
        / f"{stamp}-{uuid4().hex[:8]}"
    )
    directory.mkdir(parents=True, exist_ok=False)
    return directory.resolve()


@contextmanager
def case_logging(directory: Path) -> Iterator[None]:
    """Route one model fit's text to its session/fold/case directory."""
    directory.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(directory / "experiment.log")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logger = logging.getLogger()
    logger.addHandler(handler)
    try:
        yield
    finally:
        logger.removeHandler(handler)
        handler.close()


def configure_logging(directory: Path, level: str) -> None:
    """Route application text logs to an absolute directory and stderr."""
    directory.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(directory / "experiment.log"),
            logging.StreamHandler(),
        ],
        force=True,
    )


def configure_gpu(device: str, threads: int) -> dict:
    """Select a GPU before importing TensorFlow and verify device execution."""
    if device == "auto":
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        candidates = []
        for line in output.splitlines():
            index, free, busy = map(int, line.split(","))
            if free >= 12000 and busy <= 5:
                candidates.append((free, index))
        if not candidates:
            raise RuntimeError("No idle GPU with at least 12 GB free.")
        device = str(max(candidates)[1])
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    os.environ["TF_USE_LEGACY_KERAS"] = "1"
    os.environ["TF_NUM_INTRAOP_THREADS"] = str(threads)
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError(
            "TensorFlow found no usable GPU; CPU fallback refused."
        )
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.set_soft_device_placement(False)
    with tf.device("/GPU:0"):
        variable = tf.Variable(tf.ones((16, 16)))
        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(tf.matmul(variable, variable))
        gradient = tape.gradient(loss, variable)
        result = float(tf.reduce_sum(gradient).numpy())
    return dict(
        device=device,
        tensorflow=tf.__version__,
        gradient_sum=result,
        build=tf.sysconfig.get_build_info(),
    )
