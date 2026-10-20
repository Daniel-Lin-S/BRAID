"""Launch and resume Multi-session training without touching other campaigns.

Usage: python -m experiments.multisession.runner --experiment YAML --detach
Figure revision:
  python -m experiments.multisession.runner --plan PLAN --stage figures
Stages: plan, run, report, figures; worker is an internal subprocess interface.
The artifact root contains multi_session_training/fits and immutable plans.
Three separate analysis roots reference shared fit IDs. status.json and
coordinator.log stay at the launch root; jobs/ and diagnostics/ hold detail.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.storage.cache import atomic_json, fingerprint, writer_lock
from experiments.configuration import read_yaml
from experiments.datasets.nhp import NHPDataset
from experiments.execution.runtime import select_gpus, thread_environment
from experiments.multisession.log_paths import job_label, job_log_path
from experiments.multisession.compatibility import (
    freeze_source, load_compatibility, validate_resume,
    validate_worker, verify_source_code,
)
from experiments.multisession.queue import adopt_workers, resource_available
from experiments.multisession.queue_status import (
    inspect_status, publish_status,
)
from experiments.multisession.plan import (
    PRIMARY,
    analysis_directory,
    build_plan,
    implementation_digest,
)
from experiments.multisession.store import (
    completed, fit_directory, job, load_pickle,
    quarantine_incomplete_components,
)

LOGGER = logging.getLogger(__name__)
REPORT_INTERVAL_SECONDS = 300


def resolve(path: Path) -> dict:
    """Read portable scientific settings plus explicit machine paths."""
    settings = read_yaml(path)
    for key in ("data", "model"):
        settings[key] = read_yaml(Path(settings["modules"][key]))
    if "paths" not in settings:
        raise ValueError("Supply a local overlay with dataset/artifact paths.")
    return settings


def dataset(settings: dict) -> NHPDataset:
    """Construct the configured spike adapter with machine-local paths."""
    subjects = (
        ("indy", "loco")
        if settings["name"] in (
            "multi_session_sharing_sensitivity",
            "multi_session_source_scaling",
            "multi_session_source_capacity_scaling",
        ) and settings.get("loco_targets")
        else ("indy",)
    )
    return NHPDataset(
        dict(
            settings["data"],
            subjects=subjects,
            root=settings["paths"]["dataset_root"],
            cache_root=settings["paths"]["cache_root"],
            cache_mode="reuse",
        )
    )


def plan_campaign(settings: dict) -> tuple[dict, Path]:
    """Publish a deterministic plan and independent analysis manifests."""
    adapter = dataset(settings)
    sessions = {}
    for name in adapter.sessions():
        session = adapter.load(name)
        sessions[name] = dict(
            sha256=session.metadata["identity"]["sha256"],
            samples=len(session.arrays["t"]),
            channel_ids=session.arrays["ids"].tolist(),
        )
    if settings["name"] == "multi_session_sharing_sensitivity":
        from experiments.multisession.sensitivity import (
            build_sensitivity_plan,
        )

        reference_path = Path(settings["reference_plan"]).resolve()
        reference = json.loads(reference_path.read_text())
        plan = build_sensitivity_plan(settings, sessions, reference)
    elif settings["name"] == "multi_session_source_scaling":
        from experiments.multisession.scaling import build_scaling_plan

        reference = json.loads(
            Path(settings["reference_plan"]).resolve().read_text()
        )
        source_reference = json.loads(
            Path(settings["source_reference_plan"]).resolve().read_text()
        )
        plan = build_scaling_plan(
            settings, sessions, reference, source_reference
        )
    elif settings["name"] == "multi_session_source_capacity_scaling":
        from experiments.multisession.capacity import build_capacity_plan

        reference = json.loads(
            Path(settings["reference_plan"]).resolve().read_text()
        )
        prior = json.loads(
            Path(settings["capacity_reference_plan"]).resolve()
            .read_text()
        )
        plan = build_capacity_plan(settings, sessions, reference, prior)
    else:
        plan = build_plan(settings, sessions)
    identifier = fingerprint(dict(
        jobs=plan["jobs"], analyses=plan["analyses"],
        storage_overrides=plan.get("storage_overrides", {}),
        analysis_paths=plan.get("analysis_paths", {}),
    ))
    root = Path(settings["paths"]["artifact_root"]).resolve()
    path = root / PRIMARY / "plans" / identifier / "plan.json"
    if path.exists():
        stored = json.loads(path.read_text())
        if stored != plan:
            raise ValueError(f"Immutable plan mismatch: {path}")
    else:
        atomic_json(path, plan)
    for name, members in plan["analyses"].items():
        analysis = analysis_directory(plan, name, identifier)
        manifest = analysis / "manifest.json"
        if not manifest.exists():
            atomic_json(
                manifest,
                dict(
                    name=name,
                    plan_id=identifier,
                    members={
                        key: dict(state="pending")
                        for key in members
                    },
                ),
            )
    LOGGER.info(
        "Plan %s: %s",
        path,
        dict(Counter(x["kind"] for x in plan["jobs"].values())),
    )
    return plan, path


def run_job(
    plan: dict, key: str, compatibility: dict | None = None,
) -> None:
    """Execute one ready source, preprocessing, or target job."""
    import numpy as np
    from experiments.multisession.data import (
        session_windows,
        source_features,
        stable_seed,
        target_selection,
    )
    from experiments.multisession.source import fit_source
    from experiments.multisession.target import (
        evaluate_target,
        fit_arguments,
        fit_preprocessing,
        fit_target,
        training_arrays,
    )

    spec = plan["jobs"][key]
    current = implementation_digest(spec["kind"], spec.get("model"))
    validate_worker(key, spec, current, compatibility)
    directory = fit_directory(plan, key)
    identity = dict(spec)
    identity["parent_checksums"] = {}
    for parent in spec["dependencies"]:
        record = completed(fit_directory(plan, parent))
        if record is None:
            raise RuntimeError(f"Incomplete dependency {parent}.")
        identity["parent_checksums"][parent] = record["checksums"]
    with job(directory, identity) as needed:
        if not needed:
            LOGGER.info("Reused %s", directory)
            return
        adapter = dataset(plan["settings"])
        if spec["kind"] == "source":
            train, validation, records = [], [], {}
            for session in spec["sessions"]:
                features = source_features(adapter.load(session), spec["data"])
                tr, va, record = session_windows(
                    features,
                    spec["objective"] == "behavior",
                )
                train.append(tr)
                validation.append(va)
                records[session] = record
            atomic_json(directory / "data.json", records)
            fit_source(
                train,
                validation,
                directory,
                directory.parent.parent / "source_components",
                spec,
                spec["source_training"],
            )
            return
        session = adapter.load(spec["session"])
        features = adapter.fold(session, spec["fold"], True)
        training, validation, selection = target_selection(
            features,
            len(session.arrays["t"]),
            spec["fraction"],
            spec["seed"],
        )
        np.savez_compressed(
            directory / "selection.npz",
            training=features.arrays["indices"][training],
            validation=features.arrays["indices"][validation],
        )
        atomic_json(directory / "selection.json", selection)
        arrays = training_arrays(features, training, validation)
        arguments = fit_arguments(
            spec["model"],
            selection["batch_size"],
            spec.get("latent", 32),
            spec["target_training"]["maximum_epochs"],
        )
        atomic_json(directory / "arguments.json", arguments)
        moved = quarantine_incomplete_components(directory)
        for path in moved:
            LOGGER.warning("Quarantined interrupted component: %s", path)
        seed = stable_seed(
            spec["session"],
            spec["fold"],
            spec["seed"],
            "preprocessing" if spec["kind"] == "preprocessing" else "main",
        )
        if spec["kind"] == "preprocessing":
            fit_preprocessing(arrays, arguments, directory, seed)
            return
        weights = (
            load_pickle(fit_directory(plan, spec["source_fit"]) / "shared.pkl")
            if spec["kind"] == "adaptation"
            else {}
        )
        model = fit_target(
            arrays,
            arguments,
            directory,
            fit_directory(plan, spec["preprocessing"]),
            weights,
            seed,
            spec["implementation"],
        )
        evaluate_target(model, features, directory, spec, spec["horizons"])


def reconcile(
    plan: dict, path: Path, states: dict[str, str] | None = None
) -> dict[str, str]:
    """Refresh membership, validating saved completions on initial resume."""
    if states is None:
        states = {}
        for key in plan["jobs"]:
            directory = fit_directory(plan, key)
            try:
                record = completed(directory)
            except (ValueError, OSError):
                LOGGER.exception("Invalid completion: %s", directory)
                states[key] = "invalid"
                continue
            if record is not None:
                states[key] = "complete"
            elif (directory / "status.json").exists():
                status = json.loads((directory / "status.json").read_text())
                states[key] = status["state"]
            else:
                states[key] = "pending"
    for name, members in plan["analyses"].items():
        manifest = (
            analysis_directory(plan, name, path.parent.name)
            / "manifest.json"
        )
        atomic_json(
            manifest,
            dict(
                name=name,
                plan_id=path.parent.name,
                counts=dict(Counter(states[key] for key in members)),
                members={
                    key: dict(state=states[key])
                    for key in members
                },
            ),
        )
    return states


def run_queue(
    plan: dict, path: Path, launch: Path, settings: dict,
    previous_launch: Path | None = None,
    compatibility: dict | None = None,
    compatibility_path: Path | None = None,
    source_sha256: str | None = None,
) -> None:
    """Schedule dependency-ready jobs on four memory-ranked GPU workers."""
    if previous_launch is None:
        devices = select_gpus(
            settings["runtime"]["device"],
            settings["runtime"]["parallel_workers"],
        )
        active = {}
    else:
        devices, active = adopt_workers(plan, previous_launch)
        live_slots = {
            slot for slot, (_, process) in active.items()
            if process.poll() is None
        }
        occupied = {devices[slot]["uuid"] for slot in live_slots}
        ranked = select_gpus(settings["runtime"]["device"], len(devices))
        for slot in range(len(devices)):
            if slot in live_slots:
                continue
            devices[slot] = next(
                candidate for candidate in ranked
                if candidate["uuid"] not in occupied
            )
            occupied.add(devices[slot]["uuid"])
        atomic_json(
            launch / "adopted.json",
            dict(
                previous_launch=str(previous_launch.resolve()),
                workers={
                    str(slot): dict(job=key, pid=process.pid)
                    for slot, (key, process) in active.items()
                },
            ),
        )
        LOGGER.info("Adopted %d existing workers", len(active))
    atomic_json(launch / "devices.json", devices)
    launch_record = launch / "launch.json"
    recorded_source = (
        json.loads(launch_record.read_text()).get("source_sha256")
        if launch_record.is_file() else None
    )
    if (
        source_sha256 is not None and recorded_source is not None
        and source_sha256 != recorded_source
    ):
        raise ValueError("Launch and command source checksums differ.")
    expected_source = source_sha256 or recorded_source
    if expected_source is not None:
        verify_source_code(expected_source)
    states = reconcile(plan, path)
    if compatibility is not None:
        validate_resume(plan, states, compatibility)
    pending = {key for key, state in states.items() if state != "complete"}
    for key in pending:
        states[key] = "pending"
    failed = set()
    for key, _ in active.values():
        pending.discard(key)
        states[key] = "running"
    LOGGER.info("Initial queue counts: %s", dict(Counter(states.values())))
    priorities = {'source': 0, 'preprocessing': 1,
                  'scratch': 2, 'adaptation': 3}
    ordered = sorted(plan['jobs'], key=lambda key: (
        priorities[plan['jobs'][key]['kind']],
        plan['jobs'][key].get('source_count', 0),
        plan['jobs'][key].get('objective') != 'neural',
        plan['jobs'][key].get('latent', 0),
        plan['jobs'][key]['seed'], key,
    ))
    from experiments.multisession.capacity import (
        CAMPAIGN as CAPACITY_SCALING, PHASE_WIDTHS,
    )

    last_report = 0.0
    reporting = None
    with writer_lock(path.parent / "coordinator.lock", blocking=False):
        while pending or active:
            for slot, device in enumerate(devices):
                if slot in active:
                    continue
                eligible = [
                    key for key in ordered
                    if key in pending
                    and resource_available(key, plan["jobs"], active)
                    and all(
                        states.get(parent) == "complete"
                        for parent in plan["jobs"][key]["dependencies"]
                    )
                ]
                if plan.get("name") == CAPACITY_SCALING:
                    phase_pending = any(
                        states[key] in ("pending", "running")
                        and plan["jobs"][key].get("latent", 0)
                        in PHASE_WIDTHS
                        for key in ordered
                    )
                    if phase_pending:
                        eligible = [
                            key for key in eligible
                            if plan["jobs"][key].get("latent", 0)
                            in PHASE_WIDTHS
                        ]
                    kinds = (
                        "scratch", "source",
                        "preprocessing", "adaptation",
                    )
                    ready = next(
                        (key for kind in kinds for key in eligible
                         if plan["jobs"][key]["kind"] == kind),
                        None,
                    )
                elif plan.get("name") in (
                    "multi_session_sharing_sensitivity",
                    "multi_session_source_scaling",
                ):
                    active_sources = sum(
                        plan["jobs"][key]["kind"] == "source"
                        for key, _ in active.values()
                    )
                    kinds = (
                        ("source",) if active_sources < 2 else ()
                    ) + (
                        "adaptation", "preprocessing", "scratch", "source"
                    )
                    ready = next(
                        (
                            key for kind in kinds for key in eligible
                            if plan["jobs"][key]["kind"] == kind
                        ),
                        None,
                    )
                else:
                    ready = eligible[0] if eligible else None
                if ready is None:
                    continue
                pending.remove(ready)
                spec = plan["jobs"][ready]
                log = job_log_path(launch, ready, spec)
                log.parent.mkdir(parents=True, exist_ok=True)
                env = dict(os.environ)
                env.update(
                    thread_environment(settings["runtime"]["cpu_threads"], 1)
                )
                env["CUDA_VISIBLE_DEVICES"] = device["uuid"]
                env["TF_USE_LEGACY_KERAS"] = "1"
                command = [
                    sys.executable,
                    "-m",
                    "experiments.multisession.runner",
                    "--stage",
                    "worker",
                    "--plan",
                    str(path),
                    "--job",
                    ready,
                    "--threads",
                    str(settings["runtime"]["cpu_threads"]),
                ]
                if compatibility_path is not None:
                    command.extend([
                        "--compatibility", str(compatibility_path.resolve())
                    ])
                if expected_source is not None:
                    command.extend(["--source-sha256", expected_source])
                with log.open("a") as stream:
                    process = subprocess.Popen(
                        command,
                        env=env,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                    )
                process.log = log
                active[slot] = (ready, process)
                states[ready] = "running"
                LOGGER.info(
                    "Started %s pid=%d gpu=%d log=%s",
                    job_label(ready, spec),
                    process.pid,
                    device["index"],
                    log.resolve(),
                )
            changed = False
            for slot, (key, process) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                del active[slot]
                try:
                    valid = completed(fit_directory(plan, key))
                except (ValueError, OSError):
                    LOGGER.exception("Invalid result for %s", key)
                    valid = None
                states[key] = "complete" if code == 0 and valid else "failed"
                if states[key] == "failed":
                    failed.add(key)
                LOGGER.info(
                    "Job %s %s exit=%d",
                    job_label(key, plan["jobs"][key]),
                    states[key], code,
                )
                changed = True
            while True:
                blocked = {
                    key
                    for key in pending
                    if any(
                        parent in failed
                        for parent in plan["jobs"][key]["dependencies"]
                    )
                }
                if not blocked:
                    break
                for key in blocked:
                    states[key] = "blocked"
                    pending.remove(key)
                    failed.add(key)
            publish_status(
                launch, plan, states, pending, active, "active"
            )
            if changed:
                reconcile(plan, path, states)
                report_due = (
                    time.monotonic() - last_report >= REPORT_INTERVAL_SECONDS
                )
                report_idle = (
                    reporting is None or reporting.poll() is not None
                )
                if report_due and report_idle:
                    reporting = start_report(path, launch, settings)
                    last_report = time.monotonic()
            ready_after_completion = any(
                all(states.get(parent) == 'complete'
                    for parent in plan['jobs'][key]['dependencies'])
                for key in pending
            )
            if pending and not active and not ready_after_completion:
                raise RuntimeError(
                    "No runnable jobs remain; inspect dependencies."
                )
            if active:
                time.sleep(5)
        if reporting is not None:
            reporting.wait()
        reconcile(plan, path, states)
        publish_reports(plan, path)
        publish_status(
            launch, plan, states, pending, active, "finished"
        )
    if failed:
        LOGGER.error(
            "Campaign finished with %d failed/blocked jobs.", len(failed)
        )


def start_report(path: Path, launch: Path, settings: dict) -> subprocess.Popen:
    """Render periodic reports on CPU without blocking GPU job dispatch."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    env.update(thread_environment(settings["runtime"]["cpu_threads"], 1))
    command = [
        sys.executable, "-m", "experiments.multisession.runner",
        "--plan", str(path.resolve()), "--stage", "report",
    ]
    log = launch / "diagnostics" / "reporter.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as stream:
        return subprocess.Popen(
            command, env=env, stdout=stream, stderr=subprocess.STDOUT,
        )


def publish_reports(plan: dict, path: Path) -> None:
    """Keep reporting failures from interrupting independent fitting jobs."""
    from experiments.multisession.reports import report

    try:
        report(plan, path, final=True)
    except Exception:
        LOGGER.exception("Report generation failed; fitting will continue.")


def main() -> None:
    """Resolve a campaign or execute one explicitly assigned worker job."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path)
    parser.add_argument(
        "--stage", choices=(
            "plan", "run", "worker", "report", "figures", "status"
        ),
        default="run"
    )
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--job")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--launch", type=Path)
    parser.add_argument("--adopt-launch", type=Path)
    parser.add_argument("--compatibility", type=Path)
    parser.add_argument("--source-sha256")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    if args.stage == "status":
        if args.launch is None:
            parser.error("--stage status requires --launch")
        print(json.dumps(inspect_status(args.launch.resolve()), indent=2))
        return
    if args.stage == "worker":
        from experiments.execution.runtime import configure_device

        if args.source_sha256 is not None:
            verify_source_code(args.source_sha256)
        configure_device("auto", args.threads)
        path = args.plan.resolve()
        plan = json.loads(path.read_text())
        compatibility = (
            load_compatibility(args.compatibility, path)
            if args.compatibility is not None else None
        )
        run_job(plan, args.job, compatibility)
        return
    if args.plan is not None:
        path = args.plan.resolve()
        plan = json.loads(path.read_text())
        settings = plan["settings"]
    else:
        if args.experiment is None:
            parser.error("--experiment or --plan is required")
        settings = resolve(args.experiment)
        plan, path = plan_campaign(settings)
    compatibility = (
        load_compatibility(args.compatibility, path)
        if args.compatibility is not None else None
    )
    if args.stage == "plan":
        print(path.resolve())
        return
    if args.stage == "report":
        from experiments.multisession.reports import report

        reconcile(plan, path)
        report(plan, path)
        return
    if args.stage == "figures":
        if settings["name"] == "multi_session_source_scaling":
            from experiments.multisession.scaling_report import (
                refresh_scaling_figures,
            )

            refresh_scaling_figures(plan, path)
            return
        from experiments.multisession.reports import (
            refresh_sensitivity_figures,
        )

        refresh_sensitivity_figures(plan, path)
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    launch = args.launch or (
        Path(settings["paths"]["log_root"]).resolve()
        / PRIMARY
        / f"{stamp}-{os.getpid()}"
    )
    launch.mkdir(parents=True, exist_ok=True)
    if args.detach:
        source_sha256 = freeze_source(launch.resolve())
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str((launch / "source").resolve())
        command = [
            sys.executable,
            "-m",
            "experiments.multisession.runner",
            "--plan",
            str(path.resolve()),
            "--stage",
            "run",
            "--launch",
            str(launch),
            "--source-sha256",
            source_sha256,
        ]
        if args.adopt_launch is not None:
            command.extend(["--adopt-launch", str(args.adopt_launch.resolve())])
        if args.compatibility is not None:
            command.extend([
                "--compatibility", str(args.compatibility.resolve())
            ])
        with (launch / "coordinator.log").open("a") as stream:
            process = subprocess.Popen(
                command,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        atomic_json(
            launch / "launch.json",
            dict(
                pid=process.pid,
                plan=str(path),
                command=command,
                source_sha256=source_sha256,
            ),
        )
        print(f"Launched pid={process.pid}; logs: {launch.resolve()}")
        return
    run_queue(
        plan, path, launch, settings, args.adopt_launch,
        compatibility, args.compatibility, args.source_sha256,
    )


if __name__ == "__main__":
    main()
