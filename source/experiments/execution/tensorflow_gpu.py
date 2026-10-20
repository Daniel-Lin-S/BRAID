"""Configure one bounded TensorFlow GPU for each spawned fit worker.

Inputs are GPU IDs, a positive memory cap in MiB, and spawn-context shared
counter/lock objects. Each worker sees exactly one CUDA device. Outputs are
TensorFlow logical-device configuration and a printed assignment record.
"""

from __future__ import annotations

import multiprocessing
import os



def initialize_worker(
    gpu_ids: tuple[str, ...], memory_limit_mib: int, counter, lock,
    source_sha256: str | None = None,
) -> None:
    """Verify frozen code, then assign one bounded GPU."""
    if source_sha256 is not None:
        from experiments.campaign_source import verify_source_code

        verify_source_code(source_sha256)
    with lock:
        index = counter.value
        counter.value += 1
    device = gpu_ids[index % len(gpu_ids)]
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    import tensorflow as tf

    visible = tf.config.list_physical_devices("GPU")
    if len(visible) != 1:
        raise RuntimeError(
            f"Worker assigned GPU {device}, but TensorFlow sees "
            f"{len(visible)} GPUs."
        )
    tf.config.set_logical_device_configuration(
        visible[0], [tf.config.LogicalDeviceConfiguration(
            memory_limit=memory_limit_mib
        )],
    )
    print(
        f"Worker {os.getpid()} assigned GPU {device} with "
        f"{memory_limit_mib} MiB limit.",
        flush=True,
    )


def initializer_arguments(
    gpu_ids: tuple[str, ...], memory_limit_mib: int,
    source_sha256: str | None = None,
) -> tuple:
    """Create spawn-compatible round-robin GPU assignment arguments."""
    if not gpu_ids:
        raise ValueError("At least one GPU ID is required.")
    if memory_limit_mib < 1:
        raise ValueError(
            f"GPU memory limit must be positive; got {memory_limit_mib}."
        )
    context = multiprocessing.get_context("spawn")
    return (
        gpu_ids, memory_limit_mib,
        context.Value("i", 0), context.Lock(), source_sha256,
    )
