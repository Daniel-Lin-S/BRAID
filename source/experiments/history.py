"""Summarize persisted stage histories without inferring missing epochs.

Outputs include stage_loss_summary.json with first/last losses and trend
status, plus one PNG per trained component using its saved history.jsonl.
"""

import json
from pathlib import Path

import numpy as np

from .cache import atomic_json
from .previews import trace_plot

TREND_WINDOW = 3


def summarize_histories(
    directory: Path, destination: Path, plots: bool = True,
) -> None:
    """Save measured loss trends for each independently trained component."""
    summaries = []
    for path in sorted((directory / "components").rglob("history.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        attempts = sorted({row["attempt"] for row in rows})
        for attempt in attempts:
            history = [r for r in rows if r["attempt"] == attempt]
            epochs = np.array([r["epoch"] for r in history])
            panels = {}
            summary = dict(
                component=str(path.parent.resolve()),
                attempt=attempt,
                epochs=len(history),
            )
            for name in ("loss", "val_loss"):
                values = [row["metrics"].get(name) for row in history]
                if any(value is None for value in values):
                    summary[name] = dict(status="undefined")
                    continue
                values = np.array(values)
                first = float(np.median(values[:TREND_WINDOW]))
                last = float(np.median(values[-TREND_WINDOW:]))
                status = (
                    "insufficient_epochs"
                    if len(values) < 2 * TREND_WINDOW
                    else "decreasing"
                    if last < first
                    else "not_decreasing"
                )
                summary[name] = dict(
                    first=float(values[0]),
                    last=float(values[-1]),
                    early_median=first,
                    late_median=last,
                    status=status,
                )
                panels[name] = values
            summaries.append(summary)
            if panels and plots:
                trace_plot(
                    destination / path.parent.relative_to(directory)
                    / f"loss_attempt_{attempt}.png",
                    epochs,
                    panels,
                    path.parent.name + " training history",
                    xlabel="Epoch",
                )
    if not summaries:
        raise ValueError("No persisted component history to summarize.")
    atomic_json(destination / "stage_loss_summary.json", summaries)
