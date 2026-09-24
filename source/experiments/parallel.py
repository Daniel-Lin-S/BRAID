"""Schedule independent sessions on distinct GPUs within one analysis.

A CPU-only report child plans membership and serializes comparison reports.
Persistent GPU children own session logs and return terminal attempt records.
The coordinator owns lifecycle logging and dispatches comparison reports.
Runtime
worker assignments and IO timings belong to launch logs, not fit identities.
"""

from collections import deque
import logging
from logging.handlers import QueueHandler, QueueListener
import multiprocessing
import os
from pathlib import Path
from queue import Empty
from time import monotonic
import signal

from .contracts import plugin
from .coordination import shared_writer_available
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
    from .runner import reconcile_analysis

    reconcile_analysis(analysis, snapshots, root)
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
            velocity=data["infer_velocity"],
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
                from .runner import reconcile_analysis

                reconciled, member_errors = reconcile_analysis(
                    plan["analysis_root"], snapshots, root,
                )
                attempted = set(attempted) | reconciled
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
                events.put((
                    index, "report",
                    dict(
                        rendering_errors=failures,
                        member_errors=member_errors,
                        reconciled=reconciled,
                    ),
                ))
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


def _release_worker(index: int, workers: dict) -> None:
    """Stop one idle worker normally so its GPU context is released."""
    process, queue = workers.pop(index)
    queue.put(None)
    process.join()
    if process.exitcode:
        raise RuntimeError(
            f"Worker {process.pid} exited {process.exitcode}."
        )


def run_parallel(snapshots: dict, arguments: object, root: Path) -> None:
    """Run session tasks while parking GPUs during shared-fit waits."""
    from .runner import record_wait_timeout

    runtime = snapshots["runtime"]
    devices = select_gpus("auto", runtime["parallel_workers"])
    context = multiprocessing.get_context("spawn")
    events, logs, stop = context.Queue(), context.Queue(), context.Event()
    listener = QueueListener(
        logs, *logging.getLogger(LIFECYCLE_LOGGER).handlers,
    )
    report_tasks, report_events = context.Queue(), context.Queue()
    workers = {}
    all_processes, all_queues = [], []
    planner = None
    previous_term = signal.getsignal(signal.SIGTERM)

    def interrupt(signum: int, frame: object) -> None:
        stop.set()
        raise KeyboardInterrupt

    def start_worker(index: int) -> None:
        tasks = context.Queue()
        process = context.Process(
            target=_worker,
            args=(
                index, devices[index], snapshots, arguments, root, tasks,
                events, logs, stop,
            ),
        )
        process.start()
        workers[index] = (process, tasks)
        all_processes.append(process)
        all_queues.append(tasks)
        LOGGER.info(
            "Worker %d pid=%d GPU=%s", index, process.pid,
            devices[index]["uuid"],
        )

    signal.signal(signal.SIGTERM, interrupt)
    listener.start()
    try:
        planner = context.Process(
            target=_worker,
            args=(
                -1, None, snapshots, arguments, root, report_tasks,
                report_events, logs, stop,
            ),
        )
        planner.start()
        _, kind, plan = _event(report_events, [planner])
        if kind != "plan":
            raise RuntimeError(f"Analysis planning failed: {plan}")
        LOGGER.info("Analysis artifacts: %s", plan["analysis_root"])
        ready = deque(plan.pop("sessions"))
        deferred = {}
        deadlines = {}
        attempted = set()
        member_errors = {}
        rendering_errors = []

        def report() -> None:
            report_tasks.put(attempted.copy())
            processes = [planner, *(p for p, _ in workers.values())]
            _, report_kind, result = _event(report_events, processes)
            if report_kind != "report":
                raise RuntimeError(
                    f"Reporting process failed: {result}"
                )
            rendering_errors.extend(result["rendering_errors"])
            member_errors.update(result["member_errors"])
            attempted.update(result["reconciled"])
            for key in result["reconciled"]:
                member_errors.pop(key, None)

        def launch_ready_workers() -> None:
            needed = min(len(devices), len(ready))
            available = [
                index for index in range(len(devices))
                if index not in workers
            ]
            for index in available[:max(0, needed - len(workers))]:
                start_worker(index)

        launch_ready_workers()
        while ready or workers or deferred:
            if not workers:
                now = monotonic()
                for session, item in list(deferred.items()):
                    deadline = deadlines[item["fit_id"]]
                    if now >= deadline:

                        member_errors[item["key"]] = record_wait_timeout(
                            plan["analysis_root"], item
                        )
                        deferred.pop(session)
                    elif shared_writer_available(
                        Path(item["lock_path"])
                    ):
                        LOGGER.info(
                            "Resuming session=%s after shared %s writer",
                            session, item["phase"],
                        )
                        deferred.pop(session)
                        ready.append(session)
                if ready:
                    launch_ready_workers()
                    continue
                if not deferred:
                    break
                next_deadline = min(
                    deadlines[item["fit_id"]]
                    for item in deferred.values()
                )
                delay = min(
                    runtime["shared_fit_poll_interval_seconds"],
                    max(0.0, next_deadline - monotonic()),
                )
                stop.wait(delay)
                if stop.is_set():
                    raise KeyboardInterrupt
                continue

            processes = [planner, *(p for p, _ in workers.values())]
            index, kind, result = _event(events, processes)
            if kind == "fatal" or stop.is_set():
                raise RuntimeError(f"Worker {index} failed: {result}")
            if kind == "complete":
                attempted.update(result["attempted"])
                member_errors.update(result["model_errors"])
                rendering_errors.extend(result["rendering_errors"])
                item = result.get("deferred")
                if item is not None:
                    deferred[item["session"]] = item
                    deadlines.setdefault(
                        item["fit_id"],
                        monotonic()
                        + runtime["shared_fit_wait_timeout_seconds"],
                    )
                report()
            if kind not in ("ready", "complete"):
                raise RuntimeError(f"Unexpected worker event: {kind}")
            if ready:
                workers[index][1].put((ready.popleft(), plan))
            else:
                _release_worker(index, workers)
            launch_ready_workers()

        report()
        report_tasks.put(None)
        planner.join()
        if planner.exitcode:
            raise RuntimeError(
                f"Reporting process exited {planner.exitcode}."
            )
        if member_errors or rendering_errors:
            details = [
                *member_errors.values(), *rendering_errors,
            ]
            raise RuntimeError(
                "Scientific results remain saved; experiment failures:\n"
                + "\n".join(details)
            )
    finally:
        live = [
            process for process in all_processes
            if process.is_alive()
        ]
        if planner is not None and planner.is_alive():
            live.append(planner)
        _shutdown(live, stop)
        listener.stop()
        signal.signal(signal.SIGTERM, previous_term)
        queues = [
            *all_queues, events, logs, report_tasks, report_events,
        ]
        for queue in queues:
            queue.cancel_join_thread()
            queue.close()
