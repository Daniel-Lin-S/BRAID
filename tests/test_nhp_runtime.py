"""Check deterministic GPU selection, visibility and CPU thread controls."""

import subprocess

import pytest

from experiments.runtime import select_gpu, thread_environment

METRICS = "0, GPU-a, 8000, 0\n1, GPU-b, 16000, 80\n2, GPU-c, 16000, 20\n"


@pytest.fixture
def gpu_query(monkeypatch):
    """Provide a stable physical GPU inventory without requiring hardware."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: METRICS)


def test_free_memory_then_utilization(gpu_query):
    """Choose maximum free memory even when all large GPUs are busy."""
    assert select_gpu("auto")["uuid"] == "GPU-c"
    assert select_gpu("1")["uuid"] == "GPU-b"


def test_visibility_restricts_physical_selection(gpu_query, monkeypatch):
    """Respect allocated IDs and UUIDs without reinterpreting GPU indices."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,1")
    assert select_gpu("auto")["uuid"] == "GPU-b"
    with pytest.raises(RuntimeError, match="No requested GPU"):
        select_gpu("2")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    with pytest.raises(RuntimeError, match="No requested GPU"):
        select_gpu("auto")


def test_discovery_failure_is_explicit(monkeypatch):
    """Offer explicit CPU mode instead of silently changing execution."""
    def missing(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(subprocess, "check_output", missing)
    with pytest.raises(RuntimeError, match="--device cpu"):
        select_gpu("auto")


def test_thread_limits():
    """Apply consistent BLAS/OpenMP and independent TensorFlow limits."""
    values = thread_environment(8, 2)
    assert values["OMP_NUM_THREADS"] == "8"
    assert values["OPENBLAS_NUM_THREADS"] == "8"
    assert values["TF_NUM_INTRAOP_THREADS"] == "8"
    assert values["TF_NUM_INTEROP_THREADS"] == "2"
    with pytest.raises(ValueError, match="positive"):
        thread_environment(0, 1)
