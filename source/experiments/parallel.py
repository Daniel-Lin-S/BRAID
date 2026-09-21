"""Schedule independent sessions on distinct GPUs within one analysis.

A CPU-only report child plans membership and serializes comparison reports.
Persistent GPU children own session logs and return terminal attempt records.
The coordinator owns lifecycle logging and dispatches comparison reports.
Runtime
worker assignments and IO timings belong to launch logs, not fit identities.
"""

import logging
from logging.handlers import QueueHandler, QueueListener
import multiprocessing
import os
from pathlib import Path
from queue import Empty
import signal

from .contracts import plugin
from .runtime import (
    LIFECYCLE_LOGGER, configure_device, select_gpus, thread_environment,
)

POLL_SECONDS = 0.2
SHUTDOWN_SECONDS = 10
LOGGER = logging.getLogger(__name__)


def _plan(snapshots: dict, arguments: object, root: Path) -> dict:
    """Resolve full membership and prioritize unresolved, longer sessions."""
    from .analysis import read_manifest
    from .model_summary import write_model_summary
    from .runner import plan_analysis, population_order

    experiment, data = snapshots["experiment"], snapshots["data"]
    dataset = plugin(data["plugin"], settings=data)
    available = dataset.sessions()
    sessions = arguments.session or experiment["selection"]["sessions"]
    sessions = sessions or available
    if len(set(sessions)) != len(sessions) or set(sessions) - set(available):
        raise ValueError("Expected unique sessions from the dataset inventory.")
    shared = population_order(dataset.inventory(), experiment["seed"])
    folds = (arguments.fold or experiment["selection"]["folds"]
             or list(range(data["cv"]["folds"])))
    cases = plugin(experiment["suite_plugin"], settings=experiment["suite"])
    analysis = plan_analysis(
        dataset, sessions, folds, cases, snapshots, shared, root,
    )
    write_model_summary(analysis)
    members = read_manifest(analysis)["members"].values()
    priorities = {}
    for session in sessions:
        unresolved = sum(
            m["session"] == session and m["state"] != "complete"
            for m in members
        )
        samples = len(dataset.load(session).arrays["t"])
        priorities[session] = (unresolved, samples)
    sessions = sorted(sessions, key=priorities.get, reverse=True)
    return dict(
        sessions=sessions, shared=shared, folds=folds, cases=cases,
        analysis_root=analysis,
        settings=dict(
            snapshots["evaluation"], seed=experiment["seed"],
            model_plugin=experiment["model_plugin"],
            velocity=data["infer_velocity"], previews=data["previews"],
        ),
    )


def _worker(
    index: int, device: dict | None, snapshots: dict, arguments: object,
    root: Path, tasks: object, events: object, logs: object, stop: object,
) -> None:
    """Own a GPU and successive sessions; report fatal failures explicitly."""
    # A process group also owns the component rendering subprocesses.
    os.setsid()
    runtime = snapshots["runtime"]
    os.environ["CUDA_VISIBLE_DEVICES"] = (
        "-1" if device is None else device["uuid"]
    )
    os.environ["TF_USE_LEGACY_KERAS"] = "1"
    os.environ.update(thread_environment(
        runtime["cpu_threads"], runtime["cpu_interop_threads"],
    ))
    logging.basicConfig(level=runtime["log_level"], force=True)
    lifecycle = logging.getLogger(LIFECYCLE_LOGGER)
    lifecycle.handlers = [QueueHandler(logs)]
    lifecycle.setLevel(logging.INFO)
    lifecycle.propagate = False
    try:
        if device is None:
            from .diagnostics import render_safely

            plan = _plan(snapshots, arguments, root)
            events.put((index, "plan", plan))
            rendered = set()
            while not stop.is_set():
                attempted = tasks.get()
                if attempted is None or stop.is_set():
                    return
                failures = []
                render_safely(
                    failures, "Analysis report", plugin,
                    snapshots["experiment"]["report_plugin"],
                    root=plan["analysis_root"],
                    settings=snapshots["plotting"],
                    sample_rate=snapshots["data"]["sampling_rate_hz"],
                    attempted=attempted, rendered=rendered,
                    regenerate=runtime.get("figure_regeneration") == "all",
                )
                events.put((index, "report", failures))
            return
        gpu = configure_device(
            device["uuid"], runtime["cpu_threads"],
            runtime["cpu_interop_threads"], selected_gpu=device,
        )
        from .runner import run_session

        dataset = plugin(
            snapshots["data"]["plugin"], settings=snapshots["data"],
        )
        events.put((index, "ready", None))
        while not stop.is_set():
            task = tasks.get()
            if task is None or stop.is_set():
                return
            session, plan = task
            result = run_session(
                session, dataset, plan["folds"], plan["cases"],
                plan["settings"], arguments, plan["shared"], snapshots,
                gpu, root, plan["analysis_root"], stop,
            )
            events.put((index, "complete", result))
    except BaseException as error:
        stop.set()
        events.put((index, "fatal", f"{type(error).__name__}: {error}"))
        raise


def _event(events: object, processes: list) -> tuple:
    """Wait for a result while detecting children that exit without one."""
    while True:
        try:
            event = events.get(timeout=POLL_SECONDS)
            failed = [p for p in processes if p.exitcode is not None]
            if failed:
                raise RuntimeError(
                    f"Worker {failed[0].pid} exited {failed[0].exitcode}."
                )
            return event
        except Empty:
            failed = [p for p in processes if p.exitcode is not None]
            if failed:
                raise RuntimeError(
                    "Worker exited before its result: "
                    + ", ".join(f"{p.pid} (exit {p.exitcode})" for p in failed)
                )


def _shutdown(processes: list, stop: object) -> None:
    """Stop workers and their renderer groups, escalating after a grace time."""
    stop.set()
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
        for process in processes:
            if process.pid is None:
                continue
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                if process.is_alive():
                    os.kill(process.pid, sig)
        for process in processes:
            if process.pid is not None:
                process.join(SHUTDOWN_SECONDS if sig != signal.SIGKILL else 1)
        if all(not p.is_alive() for p in processes):
            # Reap remaining descendants even if their group leader exited.
            for process in processes:
                if process.pid is not None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            return


def run_parallel(snapshots: dict, arguments: object, root: Path) -> None:
    """Run a complete analysis with one persistent process per chosen GPU.

    Parameters
    ----------
    snapshots : dict
        Resolved configuration, signatures and numerical package versions.
    arguments : Namespace
        Invocation flags and absolute launch log directory.
    root : Path
        Shared artifact root. Completed artifacts remain immutable.
    """
    devices = select_gpus("auto", snapshots["runtime"]["parallel_workers"])
    context = multiprocessing.get_context("spawn")
    events, logs, stop = context.Queue(), context.Queue(), context.Event()
    listener = QueueListener(
        logs, *logging.getLogger(LIFECYCLE_LOGGER).handlers,
    )
    processes, queues = [], []
    report_tasks, report_events = context.Queue(), context.Queue()
    planner = None
    previous_term = signal.getsignal(signal.SIGTERM)

    def interrupt(signum: int, frame: object) -> None:
        stop.set()
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    listener.start()
    try:
        planner = context.Process(
            target=_worker,
            args=(-1, None, snapshots, arguments, root, report_tasks,
                  report_events, logs, stop),
        )
        planner.start()
        _, kind, plan = _event(report_events, [planner])
        if kind != "plan":
            raise RuntimeError(f"Analysis planning failed: {plan}")
        LOGGER.info("Analysis artifacts: %s", plan["analysis_root"])
        sessions = plan.pop("sessions")
        pending = iter(sessions)
        # Do not allocate more devices than session tasks.
        count = min(len(devices), len(sessions))
        for index, device in enumerate(devices[:count]):
            tasks = context.Queue()
            queues.append(tasks)
            process = context.Process(
                target=_worker,
                args=(index, device, snapshots, arguments, root, tasks,
                      events, logs, stop),
            )
            process.start()
            processes.append(process)
            LOGGER.info("Worker %d pid=%d GPU=%s", index, process.pid,
                        device["uuid"])
        active = set(range(count))
        attempted, failures = set(), []

        def report() -> None:
            report_tasks.put(attempted.copy())
            _, kind, result = _event(
                report_events, [planner, *processes],
            )
            if kind != "report":
                raise RuntimeError(f"Reporting process failed: {result}")
            failures.extend(result)

        while active:
            index, kind, result = _event(
                events, [planner, *processes],
            )
            if kind == "fatal" or stop.is_set():
                raise RuntimeError(f"Worker {index} failed: {result}")
            if kind == "complete":
                attempted.update(result["attempted"])
                failures.extend(result["model_errors"])
                failures.extend(result["rendering_errors"])
            session = next(pending, None)
            if session is None:
                active.remove(index)
            else:
                queues[index].put((session, plan))
            if kind == "complete":
                report()
        report()
        report_tasks.put(None)
        planner.join()
        if planner.exitcode:
            raise RuntimeError(f"Reporting process exited {planner.exitcode}.")
        for queue in queues:
            queue.put(None)
        for process in processes:
            process.join()
            if process.exitcode:
                raise RuntimeError(f"Worker exited {process.exitcode}.")
        if failures:
            raise RuntimeError(
                "Scientific results remain saved; experiment failures:\n"
                + "\n".join(failures)
            )
    finally:
        _shutdown(processes + ([planner] if planner is not None else []),
                  stop)
        listener.stop()
        signal.signal(signal.SIGTERM, previous_term)
        for queue in [*queues, events, logs, report_tasks, report_events]:
            queue.cancel_join_thread()
            queue.close()
