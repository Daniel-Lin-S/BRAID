"""Launch one configured stage with an isolated console log and metadata.

Each launch creates console.log for launch diagnostics, experiment.log for
session/fold lifecycle notices, and flat sessions/<session>.log detail files.
launch.json records the invocation. Detached workers start a new OS session.
Dry runs print resolved settings and do not create logs or artifacts.
"""

import json
import os
from pathlib import Path
import subprocess
import sys

from .configuration import argument_parser, resolve_configuration
from .runtime import launch_directory, thread_environment


def main() -> None:
    """Resolve machine settings and execute one foreground/detached worker."""
    parser = argument_parser()
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    settings = resolve_configuration(args)
    if args.dry_run:
        print(json.dumps(settings, indent=2))
        return
    directory = launch_directory(settings, args.stage)
    python = settings["runtime"].get("python") or sys.executable
    command = [python, "-u", "-m", "experiments.runner"]
    command.extend(value for value in sys.argv[1:] if value != "--detach")
    environment = dict(os.environ, NHP_LAUNCH_LOG_DIR=str(directory))
    environment.update(
        thread_environment(
            settings["runtime"]["cpu_threads"],
            settings["runtime"]["cpu_interop_threads"],
        )
    )
    if settings["runtime"]["device"] == "cpu":
        environment["CUDA_VISIBLE_DEVICES"] = "-1"
    environment["TF_USE_LEGACY_KERAS"] = "1"
    environment["MPLBACKEND"] = "Agg"
    environment.setdefault(
        "MPLCONFIGDIR",
        str(Path(settings["paths"]["cache_root"]).parent / "matplotlib"),
    )
    log = directory / "console.log"
    metadata = dict(
        command=command,
        text_log=str(log),
        stage=args.stage,
        experiment=settings["experiment"]["name"],
    )
    with log.open("x") as stream:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL if args.detach else None,
            stdout=stream if args.detach else subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=args.detach,
            text=True,
        )
        metadata["pid"] = process.pid
        (directory / "launch.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
        print(f"PID: {process.pid}\nText log: {log}", flush=True)
        if not args.detach:
            try:
                for line in process.stdout:
                    stream.write(line)
                    stream.flush()
                    print(line, end="", flush=True)
                returncode = process.wait()
            except KeyboardInterrupt:
                process.terminate()
                process.wait()
                raise
            if returncode:
                raise SystemExit(returncode)


if __name__ == "__main__":
    main()
