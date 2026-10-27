"""Observe running behavioral source fits and publish complete source figures.

Usage: python -m experiments.multisession.behavior_compute_scaling.monitor \
  --plan ABSOLUTE_PLAN --source-launch ABSOLUTE_LAUNCH \
  --output ABSOLUTE_LOG_DIRECTORY

Inputs are immutable plan.json, the original launch's status.json and source
worker logs, and checksum-validated source fits. Output status.json records
per-count completion and last log values; TensorBoard scalar events under the
campaign analysis root contain training loss, inner validation, learning rate,
source test loss and full-path FLOPs. An interim source-only snapshot is
published after the 4/8/16 fits; the complete figure replaces its latest
pointer after all four source fits complete. This observer never launches
model or adaptation workers.
"""

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
import time

import numpy as np
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter

from experiments.multisession.behavior_compute_scaling.plan import (
    CAMPAIGN, COUNTS,
)
from experiments.multisession.behavior_compute_scaling.report import (
    INTERIM_COUNTS, source_report,
)
from experiments.multisession.plan import analysis_directory
from experiments.multisession.store import completed, fit_directory
from experiments.storage.cache import atomic_json

LOGGER = logging.getLogger(__name__)
LOG_PATTERN = re.compile(
    r"^(?P<time>\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+) INFO "
    r"(?P<stage>\w+) update=(?P<update>\d+)"
    r"(?:/(?P<limit>\d+))? (?P<values>.*)$"
)
VALUE_PATTERN = re.compile(r"([a-z_]+)=([\deE.+-]+)")
FIXED_STAGES = ("preprocessing_neural", "preprocessing_behavior")
BUDGET_STAGES = ("behavior_relevant", "neural_readout", "residual")
RECURRENT_LABELS = {
    "behavior_relevant": "RNN1", "residual": "RNN2",
}
DEFAULT_INTERVAL_SECONDS = 10


def source_workers(plan: dict, launch: Path) -> dict[int, dict]:
    """Match frozen source worker logs to the plan's four source counts."""
    saved = json.loads((launch / "status.json").read_text())
    workers = {}
    for worker in saved["active"].values():
        spec = plan["jobs"][worker["job"]]
        if spec["kind"] != "source":
            continue
        count = spec["source_count"]
        if count in workers:
            raise ValueError(f"Duplicate source worker for {count} sessions.")
        workers[count] = worker
    if set(workers) != set(COUNTS):
        raise ValueError(
            f"Expected source workers {COUNTS}, got {sorted(workers)}."
        )
    return workers


def write_scalar(
    writer: EventFileWriter, tag: str, value: float,
    step: int, wall_time: float,
) -> None:
    """Append one finite scalar to a separate TensorBoard event stream."""
    if not np.isfinite(value):
        raise FloatingPointError(f"Nonfinite monitored scalar {tag}.")
    writer.add_event(Event(
        wall_time=wall_time, step=step,
        summary=Summary(value=[Summary.Value(
            tag=tag, simple_value=float(value),
        )]),
    ))


class SourceLogReader:
    """Read appended worker log lines without accessing training state."""

    def __init__(self, path: Path, count: int, writer: EventFileWriter):
        self.path = path
        self.count = count
        self.writer = writer
        self.offset = 0
        self.budget_index = 0
        self.previous_stage = None
        self.last = {}
        self.budgets = (50, 100, 200, 400, 800)
        if count == 33:
            self.budgets = tuple(value * 2 for value in self.budgets)

    def scan(self) -> None:
        """Publish every newly completed metric line exactly once."""
        if not self.path.is_file():
            raise FileNotFoundError(
                f"Source worker log is missing: {self.path.resolve()}."
            )
        with self.path.open() as stream:
            stream.seek(self.offset)
            while True:
                position = stream.tell()
                line = stream.readline()
                if not line:
                    break
                if not line.endswith("\n"):
                    stream.seek(position)
                    break
                self.offset = stream.tell()
                match = LOG_PATTERN.match(line)
                if match is None:
                    continue
                self._record(match)
        self.writer.flush()

    def _record(self, match: re.Match) -> None:
        """Map one native source log row to stage and budget scalars."""
        stage = match["stage"]
        if stage not in (*FIXED_STAGES, *BUDGET_STAGES):
            return
        if stage == "behavior_relevant" and (
            self.previous_stage == "residual"
        ):
            self.budget_index += 1
        if self.budget_index >= len(self.budgets):
            raise ValueError(
                f"More than {len(self.budgets)} budgets in "
                f"{self.path.resolve()}."
            )
        self.previous_stage = stage
        update = int(match["update"])
        stamp = datetime.strptime(
            match["time"], "%Y-%m-%d %H:%M:%S,%f"
        ).replace(tzinfo=timezone.utc).timestamp()
        values = {
            key: float(value)
            for key, value in VALUE_PATTERN.findall(match["values"])
        }
        prefix = stage
        if stage in BUDGET_STAGES:
            prefix = f"budget_{self.budgets[self.budget_index]}/{stage}"
        tags = {
            "train": "training_loss", "loss": "training_loss",
            "inner": "inner_validation_loss",
            "lr": "learning_rate", "best": "best_update",
            "stale": "checks_without_improvement",
        }
        for key, tag in tags.items():
            if key in values:
                write_scalar(
                    self.writer, f"{prefix}/{tag}", values[key],
                    update, stamp,
                )
        self.last = dict(
            stage=stage, budget=(self.budgets[self.budget_index]
                                 if stage in BUDGET_STAGES else None),
            update=update, values=values,
            time_utc=datetime.fromtimestamp(
                stamp, timezone.utc
            ).isoformat(),
        )


def publish_source_scores(
    plan: dict, writers: dict[int, EventFileWriter],
    published: set[int],
) -> dict[int, bool]:
    """Validate completed source fits and publish their test scores once."""
    complete = {}
    for key in plan["source_references"]:
        count = plan["jobs"][key]["source_count"]
        fit = fit_directory(plan, key)
        marker = fit / "complete.json"
        if not marker.is_file():
            complete[count] = False
            continue
        if count not in published:
            if completed(fit) is None:
                raise ValueError(
                    f"Source completion disappeared: {fit.resolve()}."
                )
            budgets = json.loads((fit / "budgets.json").read_text())
            if len(budgets) != 5:
                raise ValueError(
                    f"Expected five completed source budgets for {count}."
                )
            writer = writers[count]
            now = datetime.now(timezone.utc).timestamp()
            for budget in budgets:
                step = budget["total_updates"]
                for stage, label in RECURRENT_LABELS.items():
                    loss = budget["stages"][stage]["score"]["mean_loss"]
                    if loss <= 0:
                        raise ValueError(
                            f"Nonpositive source test loss for {count}."
                        )
                    write_scalar(
                        writer, f"source_test/{label}", loss, step, now,
                    )
                write_scalar(
                    writer, "source_test/full_pipeline_flops",
                    budget["full_path_flops"], step, now,
                )
            writer.flush()
            published.add(count)
            LOGGER.info("Published source test scores for %d sessions", count)
        complete[count] = True
    return complete


def worker_running(worker: dict) -> bool:
    """Check saved PID and Linux start ticks before declaring an exit."""
    path = Path("/proc") / str(worker["pid"]) / "stat"
    try:
        fields = path.read_text().rsplit(")", 1)[1].split()
    except FileNotFoundError:
        return False
    return fields[0] not in ("Z", "X") and fields[19] == worker[
        "start_ticks"
    ]


def observe(
    plan: dict, plan_path: Path, launch: Path, output: Path,
    interval: int,
) -> None:
    """Stream source metrics, then publish the complete source figure group."""
    if interval <= 0:
        raise ValueError(f"Expected positive polling interval, got {interval}.")
    workers = source_workers(plan, launch)
    events = analysis_directory(
        plan, CAMPAIGN, plan_path.parent.name
    ) / "monitor" / launch.name
    events.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    writers = {
        count: EventFileWriter(str(events / f"sources_{count}"))
        for count in COUNTS
    }
    readers = {
        count: SourceLogReader(
            Path(workers[count]["log"]), count, writers[count]
        ) for count in COUNTS
    }
    published = set()
    partial_snapshot = None
    try:
        while True:
            for reader in readers.values():
                reader.scan()
            complete = publish_source_scores(plan, writers, published)
            if (
                not all(complete.values())
                and all(complete[count] for count in INTERIM_COUNTS)
                and partial_snapshot is None
            ):
                partial_snapshot = source_report(
                    plan, plan_path, INTERIM_COUNTS
                )
            stopped = [
                count for count in COUNTS
                if not complete[count] and not worker_running(workers[count])
            ]
            phase = (
                "complete" if all(complete.values()) else
                "failed" if stopped else "active"
            )
            status = dict(
                phase=phase,
                updated_at_utc=datetime.now(timezone.utc).isoformat(),
                complete=complete,
                last={count: readers[count].last for count in COUNTS},
                tensorboard_events=str(events.resolve()),
                failed_counts=stopped,
                partial_source_figures=(
                    str((partial_snapshot / "figures").resolve())
                    if partial_snapshot is not None else None
                ),
                adaptations_queued=False,
            )
            atomic_json(output / "status.json", status)
            if stopped:
                raise RuntimeError(
                    f"Source workers exited without completion: {stopped}."
                )
            if phase == "complete":
                snapshot = source_report(plan, plan_path)
                if snapshot is None:
                    raise RuntimeError(
                        "Complete sources did not produce source figures."
                    )
                status["source_figures"] = str(
                    (snapshot / "figures").resolve()
                )
                atomic_json(output / "status.json", status)
                print(f"Source figures: {status['source_figures']}")
                return
            time.sleep(interval)
    finally:
        for writer in writers.values():
            writer.close()


def main() -> None:
    """Run a low-priority source-only observer from absolute paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--source-launch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--interval-seconds", type=int, default=DEFAULT_INTERVAL_SECONDS
    )
    args = parser.parse_args()
    for path in (args.plan, args.source_launch, args.output):
        if not path.is_absolute():
            parser.error(f"Expected absolute path, got {path}.")
    logging.basicConfig(level=logging.INFO)
    plan = json.loads(args.plan.read_text())
    observe(
        plan, args.plan, args.source_launch, args.output,
        args.interval_seconds,
    )


if __name__ == "__main__":
    main()
