"""Configure logging, CPU threads and verified TensorFlow execution.

Device identity and a real gradient probe are returned for run metadata.
Automatic selection ranks visible GPUs by free memory, then utilization.
CPU execution must be explicitly requested; GPU failures are not hidden.
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


def thread_environment(threads: int, inter_threads: int) -> dict[str, str]:
    """Return TensorFlow and numerical-library CPU thread limits.

    Parameters
    ----------
    threads : int
        Positive intra-operation and BLAS/OpenMP thread count.
    inter_threads : int
        Positive TensorFlow inter-operation thread count.

    Returns
    -------
    dict of str
        Environment settings to install before importing numerical libraries.
    """
    if min(threads, inter_threads) < 1:
        raise ValueError("CPU thread counts must be positive integers.")
    result = {
        name: str(threads)
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "TF_NUM_INTRAOP_THREADS",
        )
    }
    result["TF_NUM_INTEROP_THREADS"] = str(inter_threads)
    return result


def select_gpu(request: str) -> dict:
    """Resolve a physical GPU within CUDA_VISIBLE_DEVICES restrictions.

    Parameters
    ----------
    request : str
        auto, a physical GPU index, or a full GPU UUID.

    Returns
    -------
    dict
        Physical index, UUID, free memory in MiB and utilization percentage.

    Raises
    ------
    RuntimeError
        GPU discovery fails or the requested visibility set has no device.
    """
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            "Cannot query CUDA GPUs; use --device cpu for CPU execution."
        ) from error
    devices = []
    try:
        for line in output.splitlines():
            index, uuid, free, busy = [v.strip() for v in line.split(",")]
            devices.append(
                dict(
                    index=int(index),
                    uuid=uuid,
                    free_memory_mib=int(free),
                    utilization_percent=int(busy),
                )
            )
    except ValueError as error:
        raise RuntimeError(
            "Invalid GPU metrics returned by nvidia-smi."
        ) from error
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        tokens = {token.strip() for token in visible.split(",")}
        devices = [
            d
            for d in devices
            if str(d["index"]) in tokens or d["uuid"] in tokens
        ]
    if request != "auto":
        devices = [
            d for d in devices if request in (str(d["index"]), d["uuid"])
        ]
    if not devices:
        raise RuntimeError(
            "No requested GPU is visible. Check CUDA_VISIBLE_DEVICES "
            "or use --device cpu for explicit CPU execution."
        )
    return max(
        devices,
        key=lambda d: (
            d["free_memory_mib"],
            -d["utilization_percent"],
            -d["index"],
        ),
    )


def configure_device(
    device: str,
    threads: int,
    inter_threads: int = 1,
) -> dict:
    """Configure CPU/CUDA placement and verify a forward/backward operation.

    Parameters
    ----------
    device : str
        auto, cpu, physical GPU index or full GPU UUID.
    threads : int
        Positive CPU intra-operation thread count.
    inter_threads : int, optional
        TensorFlow inter-operation threads; default is 1.

    Returns
    -------
    dict
        Selected device, thread limits, TensorFlow build and gradient probe.
    """
    selected = None if device == "cpu" else select_gpu(device)
    os.environ["CUDA_VISIBLE_DEVICES"] = (
        "-1" if selected is None else selected["uuid"]
    )
    os.environ["TF_USE_LEGACY_KERAS"] = "1"
    os.environ.update(thread_environment(threads, inter_threads))
    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(threads)
    tf.config.threading.set_inter_op_parallelism_threads(inter_threads)
    gpus = tf.config.list_physical_devices("GPU")
    if selected is None:
        tf.config.set_visible_devices([], "GPU")
    elif not gpus:
        raise RuntimeError(
            "TensorFlow found no usable GPU; check its CUDA installation "
            "or explicitly select --device cpu."
        )
    else:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    target = "/CPU:0" if selected is None else "/GPU:0"
    tf.config.set_soft_device_placement(False)
    with tf.device(target):
        variable = tf.Variable(tf.ones((16, 16)))
        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(tf.matmul(variable, variable))
        gradient = tape.gradient(loss, variable)
        tf.debugging.assert_all_finite(gradient, "Nonfinite device gradient")
        result = float(tf.reduce_sum(gradient).numpy())
    metadata = dict(
        device="cpu" if selected is None else selected["uuid"],
        selected_gpu=selected,
        execution_device=variable.device,
        cpu_threads=threads,
        cpu_interop_threads=inter_threads,
        tensorflow=tf.__version__,
        gradient_sum=result,
        build=tf.sysconfig.get_build_info(),
    )
    logging.getLogger(__name__).info("Execution device: %s", metadata)
    return metadata
