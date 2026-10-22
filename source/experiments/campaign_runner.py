"""Coordinate independent fits and live status for any experiment.

Inputs are picklable job functions, argument tuples, an optional worker
initializer, and a campaign root. Outputs are reported job failures and an
atomic launch.json; detached stdout/stderr are saved at the supplied log path.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from importlib import import_module
from datetime import datetime, timezone
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
from threading import Event, Thread
from typing import Callable

from experiments.storage.cache import atomic_json
from experiments.campaign_source import (
    SOURCE_ROOT, freeze_source, source_code_digest,
    verify_source_code,
)


def run_phase(
    function: Callable[..., str], arguments: list[tuple],
    workers: int,
    initializer: Callable[..., None] | None = None,
    initargs: tuple = (),
) -> list[tuple]:
    """Run independent jobs and return all failed arguments and errors."""
    if workers < 1:
        raise ValueError(f"Worker count must be positive; got {workers}.")
    if initializer is None and initargs:
        raise ValueError("Initializer arguments need an initializer.")
    if not arguments:
        raise ValueError("Experiment phase has no jobs.")
    failures = []
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=context,
        initializer=initializer, initargs=initargs,
    ) as executor:
        futures = {
            executor.submit(function, *item): item
            for item in arguments
        }
        for future in as_completed(futures):
            item = futures[future]
            try:
                print(f"Completed {future.result()}", flush=True)
            except Exception as error:
                failures.append((item, error))
                print(
                    f"Failed {item}: {type(error).__name__}: {error}",
                    file=sys.stderr, flush=True,
                )
    return failures


def run_with_status(
    root: Path, launch_metadata: dict,
    watch: Callable[[Path, Event], None],
    work: Callable[[], None],
) -> None:
    """Run a campaign while a monitor atomically refreshes status."""
    root.mkdir(parents=True, exist_ok=True)
    launch = root / "launch.json"
    record = json.loads(launch.read_text()) if launch.is_file() else {}
    record.update(pid=os.getpid(), **launch_metadata)
    record.pop("ended_at_utc", None)
    atomic_json(launch, record)
    stop = Event()
    monitor = Thread(target=watch, args=(root, stop), daemon=True)
    monitor.start()
    try:
        work()
    finally:
        stop.set()
        monitor.join()
        record = json.loads(launch.read_text())
        record.update(
            campaign_active=False,
            ended_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        atomic_json(launch, record)


def detach(
    command: list[str], root: Path, log_path: Path,
    launch_metadata: dict, environment: dict | None = None,
) -> int:
    """Start a detached campaign and publish its PID and log path."""
    root.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as stream:
        process = subprocess.Popen(
            command, stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT,
            start_new_session=True, env=environment,
        )
    path = root / "launch.json"
    record = json.loads(path.read_text()) if path.exists() else {}
    record.update(
        pid=process.pid, log=str(log_path),
        log_directory=str(log_path.parent), **launch_metadata,
    )
    record.pop("ended_at_utc", None)
    atomic_json(path, record)
    return process.pid


def main() -> None:
    """Run, audit, or report a study through its generic campaign CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--audit", action="store_true")
    action.add_argument("--run", action="store_true")
    action.add_argument("--report", action="store_true")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--gpus", nargs="+", default=[])
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--source-sha256")
    parser.add_argument("--redraw-figures", action="store_true")
    args = parser.parse_args()
    if args.source_sha256 is not None:
        verify_source_code(args.source_sha256)
    study = import_module(args.study)
    experiment = args.experiment.resolve()
    root = args.root.resolve()
    if (root / "quarantine.json").is_file():
        parser.error(f"Campaign {root} is quarantined.")
    log_directory = study.log_directory(experiment, root)
    if root.parent != log_directory.parent:
        parser.error(
            f"Campaign root must be a direct child of "
            f"{log_directory.parent}; got {root}."
        )
    if args.detach and not args.run:
        parser.error("Detached launch requires --run.")
    if args.redraw_figures and not args.report:
        parser.error("Figure redraw requires --report.")
    if args.audit:
        study.audit(experiment, root)
        return
    analysis = study.initialize_storage(experiment, root)
    if args.report:
        study.report(experiment, root, redraw_figures=args.redraw_figures)
    else:
        workers = (
            study.DEFAULT_WORKERS if args.workers is None
            else args.workers
        )
        if workers < 1 or workers > len(args.gpus):
            parser.error(
                f"Expected 1 to {len(args.gpus)} workers for "
                f"{len(args.gpus)} GPUs; got {workers}."
            )
        if args.detach:
            assets = SOURCE_ROOT.parent / "assets"
            if not experiment.is_relative_to(assets):
                parser.error(
                    f"Experiment YAML must be beneath {assets}; "
                    f"got {experiment}."
                )
            source_sha256 = source_code_digest()
            provenance = analysis / "provenance" / source_sha256
            snapshot = provenance / "source"
            if snapshot.exists():
                if source_sha256 != source_code_digest(snapshot):
                    parser.error("Frozen campaign source checksum changed.")
            elif freeze_source(provenance) != source_sha256:
                parser.error("Campaign source changed while freezing it.")
            frozen_experiment = (
                provenance / "assets" / experiment.relative_to(assets)
            )
            if not frozen_experiment.is_file():
                parser.error(
                    f"Missing frozen experiment: {frozen_experiment}."
                )
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(snapshot.resolve())
            environment["BRAID_CAMPAIGN_SOURCE_SHA256"] = (
                source_sha256
            )
            command = [
                sys.executable, "-m", "experiments.campaign_runner",
                "--study", args.study, "--experiment",
                str(frozen_experiment),
                "--root", str(root), "--run", "--workers",
                str(workers), "--gpus", *args.gpus,
                "--source-sha256", source_sha256,
            ]
            pid = detach(
                command, root, log_directory / "campaign.log",
                dict(
                    workers=workers, gpu_ids=args.gpus,
                    source_sha256=source_sha256,
                    source_snapshot=str(provenance),
                ), environment,
            )
            print(f"Launched PID {pid}; root {root}")
        else:
            if args.source_sha256 is None:
                parser.error("Run requires a frozen source checksum.")
            from experiments.campaign_status import (
                watch as watch_status,
            )

            metadata = dict(
                workers=workers, gpu_ids=args.gpus,
                log_directory=str(log_directory),
                reused_completed_keys=study.verified_reused_keys(
                    experiment
                ),
            )

            def watch(root: Path, stop: Event) -> None:
                """Watch this study using the generic status service."""
                plan = study.status_plan(
                    root, set(metadata["reused_completed_keys"])
                )
                watch_status(root, plan, stop)

            run_with_status(
                root, metadata, watch,
                lambda: study.run_jobs(
                    experiment, root, workers, tuple(args.gpus)
                ),
            )


if __name__ == "__main__":
    main()
