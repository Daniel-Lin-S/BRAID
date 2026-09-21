"""Exercise real spawned scheduling and logging without GPU training."""

import argparse
import json
import multiprocessing
from contextlib import nullcontext
import os
import signal
import sys
import time

import pytest

from experiments import parallel
from experiments.runtime import configure_logging, select_gpus


def _fake_worker(*args):
    """Use the production worker with deterministic, observable session work."""
    from experiments import runner
    from experiments.runtime import lifecycle_scope, session_logging

    root = args[4]
    mode = args[2].get("test_mode")

    def plan(*unused):
        if mode == "plan_fail":
            raise ValueError("injected planning failure")
        return dict(
            sessions=["a", "b", "c", "d", "e"], folds=[0], cases=[],
            shared=[], settings={}, analysis_root=root,
        )

    def configure(*unused, **kwargs):
        assert os.environ["CUDA_VISIBLE_DEVICES"] == kwargs["selected_gpu"][
            "uuid"
        ]
        return kwargs["selected_gpu"]

    def session(name, dataset, folds, cases, settings, arguments, shared,
                snapshots, gpu, artifact_root, analysis_root, stop):
        with (root / f"{name}.owner").open("x") as stream:
            json.dump(dict(pid=os.getpid(), gpu=gpu["uuid"]), stream)
        with lifecycle_scope(arguments.log_directory, name):
            with session_logging(arguments.log_directory, name):
                os.write(1, f"native-{name}\n".encode())
                if mode in ("crash", "crash_zero") and name == "a":
                    os._exit(17 if mode == "crash" else 0)
                if mode == "fatal" and name == "a":
                    raise RuntimeError("injected setup failure")
                if mode == "interrupt" and name == "a":
                    os.kill(os.getppid(), signal.SIGINT)
                    time.sleep(30)
                time.sleep(0.05 if name == "a" else 0.15)
        return dict(
            attempted={name}, rendering_errors=[],
            model_errors=["contained model failure"]
            if mode == "model_fail" and name == "a" else [],
        )

    parallel._plan = plan
    parallel.configure_device = configure
    def plugin(*unused, **kwargs):
        if "attempted" in kwargs:
            with (root / "reports.jsonl").open("a") as stream:
                stream.write(json.dumps(dict(
                    pid=os.getpid(), attempted=sorted(kwargs["attempted"]),
                )) + "\n")
        return object()

    parallel.plugin = plugin
    runner.run_session = session
    parallel._worker(*args)


@pytest.mark.parametrize("mode", [None, "model_fail", "fatal", "crash",
                                  "crash_zero", "plan_fail", "interrupt"])
def test_spawned_scheduler(tmp_path, monkeypatch, mode):
    """Use real processes to check assignment, collection and shutdown."""
    worker = parallel._worker
    # The spawn target restores the production worker in its fresh module.
    monkeypatch.setattr(parallel, "_worker", _fake_worker)
    monkeypatch.setattr(
        parallel, "select_gpus",
        lambda *args: [dict(uuid="GPU-a"), dict(uuid="GPU-b")],
    )
    configure_logging(tmp_path, "INFO")
    settings = dict(
        runtime=dict(parallel_workers=2, cpu_threads=1,
                     cpu_interop_threads=1, log_level="INFO"),
        experiment=dict(report_plugin="fixture:report"),
        data=dict(plugin="fixture:data", sampling_rate_hz=20),
        plotting={}, test_mode=mode,
    )
    before = set(multiprocessing.active_children())
    expectation = (
        pytest.raises(KeyboardInterrupt) if mode == "interrupt"
        else pytest.raises(RuntimeError) if mode else nullcontext()
    )
    with expectation:
        parallel.run_parallel(
            settings, argparse.Namespace(log_directory=tmp_path), tmp_path,
        )
    assert set(multiprocessing.active_children()) == before
    if mode in (None, "model_fail"):
        owners = [json.loads(p.read_text()) for p in tmp_path.glob("*.owner")]
        assert len(owners) == 5
        assert len({x["pid"] for x in owners}) == 2
        assert {x["gpu"] for x in owners} == {"GPU-a", "GPU-b"}
        reports = [json.loads(line) for line in
                   (tmp_path / "reports.jsonl").read_text().splitlines()]
        assert len({r["pid"] for r in reports}) == 1
        assert reports[0]["pid"] not in {x["pid"] for x in owners}
        assert set(reports[-1]["attempted"]) == set("abcde")
        lifecycle = (tmp_path / "experiment.log").read_text()
        for name in "abcde":
            path = tmp_path / "sessions" / f"{name}.log"
            assert lifecycle.count(str(path)) == 1
            assert path.read_text().count(f"native-{name}") == 1
    else:
        assert not (tmp_path / "e.owner").exists()
    assert worker is not _fake_worker


def test_four_gpu_order_and_single_query(monkeypatch):
    calls = []

    def query(*args, **kwargs):
        calls.append(True)
        return ("0, GPU-a, 8, 0\n1, GPU-b, 16, 80\n"
                "2, GPU-c, 16, 20\n3, GPU-d, 16, 20\n4, GPU-e, 1, 0\n")

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr("subprocess.check_output", query)
    assert [d["uuid"] for d in select_gpus("auto", 4)] == [
        "GPU-c", "GPU-d", "GPU-b", "GPU-a",
    ]
    assert len(calls) == 1
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-c")
    with pytest.raises(RuntimeError, match="Requested 4 GPUs.*only 2"):
        select_gpus("auto", 4)


def test_coordinator_import_does_not_import_tensorflow():
    import subprocess

    subprocess.run(
        [sys.executable, "-c", "import sys; import experiments.runner; "
         "import experiments.parallel; assert 'tensorflow' not in sys.modules"],
        check=True,
    )
