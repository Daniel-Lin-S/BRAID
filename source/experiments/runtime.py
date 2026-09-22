"""Configure logging, CPU threads and verified TensorFlow execution.

Device identity and a real gradient probe are returned for run metadata.
Automatic selection ranks visible GPUs by free memory, then utilization.
CPU execution must be explicitly requested; GPU failures are not hidden.
"""

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import logging
from time import monotonic
from typing import Iterator
from uuid import uuid4
import os
from pathlib import Path
import subprocess
import sys


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


LIFECYCLE_LOGGER = "experiments.lifecycle"
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
STAGE_ACTIVITY = {
    "plot": "reports",
    "preprocess": "folds",
}


@contextmanager
def session_logging(directory: Path, session: str) -> Iterator[None]:
    """Exclusively capture Python and native session output in one flat file.

    Parameters
    ----------
    directory : Path
        Launch log root, containing sessions/<session>.log.
    session : str
        Single safe filename component identifying the dataset session.

    Yields
    ------
    None
        Session scope. An exception is logged here once and exits with code 1.
    """
    if not session or Path(session).name != session:
        raise ValueError("Session log name must be one filename component.")
    path = directory / "sessions" / f"{session}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    previous = root.handlers[:]
    child_routes = []
    for name, logger in logging.Logger.manager.loggerDict.items():
        if isinstance(logger, logging.Logger) and name != LIFECYCLE_LOGGER:
            child_routes.append((logger, logger.handlers[:], logger.propagate))
            logger.handlers = []
            logger.propagate = True
    sys.stdout.flush()
    sys.stderr.flush()
    saved = [os.dup(fd) for fd in (1, 2)]
    with path.open("a", buffering=1) as stream:
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        try:
            for fd in (1, 2):
                os.dup2(stream.fileno(), fd)
            root.handlers = [handler]
            with redirect_stdout(stream), redirect_stderr(stream):
                try:
                    yield
                except BaseException:
                    root.exception("Session failed: %s", session)
                    raise SystemExit(1) from None
        finally:
            stream.flush()
            root.handlers = previous
            for logger, handlers, propagate in child_routes:
                logger.handlers = handlers
                logger.propagate = propagate
            handler.close()
            for fd, original in zip((1, 2), saved):
                os.dup2(original, fd)
                os.close(original)


@contextmanager
def lifecycle_scope(
    directory: Path,
    session: str,
    fold: int | None = None,
    model: str | None = None,
) -> Iterator[dict]:
    """Record session, fold and model boundaries with terminal outcomes.

    Parameters
    ----------
    directory : Path
        Launch log directory.
    session : str
        Session identifier.
    fold : int, optional
        Fold identifier; default None denotes the entire session.
    model : str, optional
        Model setting name; default None denotes a session or fold.

    Yields
    ------
    dict
        Outcome counts (completed/reused/failed) and active model phase.
        Set skipped=True for a reused model.
    """
    logger = logging.getLogger(LIFECYCLE_LOGGER)
    label = f"session={session}" + (f" fold={fold}" if fold is not None else "")
    if model is not None:
        label += f" model={model}"
    state = dict(skipped=False, completed=0, reused=0, failed=0, phase="setup")
    if fold is None and model is None:
        path = (directory / "sessions" / f"{session}.log").resolve()
        logger.info("Started %s; details: %s", label, path)
    else:
        logger.info("Started %s", label)
    started = monotonic()
    written = process_write_bytes()
    try:
        yield state
    except BaseException:
        logger.error("Failed %s; phase=%s", label, state["phase"])
        raise
    else:
        if state["failed"]:
            event = "Failed" if model else "Finished with failures"
        else:
            event = "Reused" if state["skipped"] else "Finished"
        counts = " ".join(
            f"{key}={state[key]}" for key in ("completed", "reused", "failed")
        )
        detail = counts
        if model and state["failed"]:
            detail += f" phase={state['phase']}"
        level = logging.ERROR if state["failed"] else logging.INFO
        logger.log(level, "%s %s; %s", event, label, detail)
    finally:
        after = process_write_bytes()
        delta = None if written is None or after is None else after - written
        logger.info(
            "Timing %s; elapsed_seconds=%.3f write_bytes=%s",
            label, monotonic() - started, delta,
        )


@contextmanager
def stage_scope(
    directory: Path,
    stage: str,
    session: str | None = None,
    fold: int | None = None,
    analysis_id: str | None = None,
) -> Iterator[dict[str, int]]:
    """Record a non-fitting stage using its own work counters.

    Parameters
    ----------
    directory : Path
        Launch log directory.
    stage : str
        One of preprocess or plot.
    session : str, optional
        Session identifier for preprocessing; default None.
    fold : int, optional
        Fold identifier for preprocessing; default None.
    analysis_id : str, optional
        Requested analysis identifier for plotting; default None.

    Yields
    ------
    dict of int
        Completed and failed stage work counts.
    """
    if stage not in STAGE_ACTIVITY:
        raise ValueError(f"Unsupported lifecycle stage: {stage}.")
    if fold is not None and session is None:
        raise ValueError("A preprocessing fold requires a session.")
    if analysis_id is not None and stage != "plot":
        raise ValueError("An analysis ID is only valid for the plot stage.")
    logger = logging.getLogger(LIFECYCLE_LOGGER)
    parts = [f"stage={stage}"]
    if session is not None:
        parts.append(f"session={session}")
    if fold is not None:
        parts.append(f"fold={fold}")
    if analysis_id is not None:
        parts.append(f"analysis={analysis_id}")
    label = " ".join(parts)
    state = dict(completed=0, failed=0)
    if session is not None and fold is None:
        path = (directory / "sessions" / f"{session}.log").resolve()
        logger.info("Started %s; details: %s", label, path)
    else:
        logger.info("Started %s", label)
    try:
        yield state
    except BaseException:
        logger.error("Failed %s", label)
        raise
    else:
        event = "Finished with failures" if state["failed"] else "Finished"
        level = logging.ERROR if state["failed"] else logging.INFO
        logger.log(
            level,
            "%s %s; %s=%d failed=%d",
            event,
            label,
            STAGE_ACTIVITY[stage],
            state["completed"],
            state["failed"],
        )


def configure_logging(directory: Path, level: str) -> None:
    """Separate lifecycle notifications from launch/session diagnostic text."""
    directory.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, level),
        format=LOG_FORMAT,
        handlers=[logging.StreamHandler()],
        force=True,
    )
    lifecycle = logging.getLogger(LIFECYCLE_LOGGER)
    for handler in lifecycle.handlers:
        handler.close()
    handler = logging.FileHandler(directory / "experiment.log")
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    lifecycle.handlers = [handler]
    lifecycle.setLevel(logging.INFO)
    lifecycle.propagate = False


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


def select_gpus(request: str, count: int = 1) -> list[dict]:
    """Rank physical GPUs within CUDA_VISIBLE_DEVICES restrictions.

    Parameters
    ----------
    request : str
        auto, a physical GPU index, or a full GPU UUID.

    count : int, optional
        Number of distinct GPUs to return; default 1.

    Returns
    -------
    list of dict
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
    if count < 1 or len(devices) < count:
        raise RuntimeError(
            f"Requested {count} GPUs, but only {len(devices)} are visible."
        )
    return sorted(
        devices,
        key=lambda d: (
            -d["free_memory_mib"],
            d["utilization_percent"],
            d["index"],
        ),
    )[:count]


def select_gpu(request: str) -> dict:
    """Select one visible GPU by free memory, utilization, then index."""
    return select_gpus(request)[0]


def configure_device(
    device: str,
    threads: int,
    inter_threads: int = 1,
    selected_gpu: dict | None = None,
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

    selected_gpu : dict, optional
        Preselected physical GPU metadata; default None queries discovery.

    Returns
    -------
    dict
        Selected device, thread limits, TensorFlow build and gradient probe.
    """
    selected = selected_gpu
    if selected is None and device != "cpu":
        selected = select_gpu(device)
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


def process_write_bytes() -> int | None:
    """Read process storage writes, or report unavailable OS accounting."""
    try:
        for line in Path("/proc/self/io").read_text().splitlines():
            if line.startswith("write_bytes:"):
                return int(line.split()[1])
    except (OSError, ValueError):
        pass
    return None
