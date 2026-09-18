"""BRAID-specific artifact reports, selected through the reporting plugin.

Inputs are completed metrics.json/predictions.npz artifacts. Outputs include
raw_metrics.csv/json, summary.csv/json, population_summary.csv, and the
configured figure suite. Incomplete folds are reported, never filled in.
"""

import csv
from pathlib import Path

from .analysis import read_manifest
from .evaluation import aggregate, collect_results
from .plots import forecast_example, plot_suite


def braid_report(root: Path, settings: dict, sample_rate: float) -> None:
    """Produce the required BRAID tables and optional figures from artifacts."""
    from .model_summary import write_model_summary

    write_model_summary(root)
    destination = root / "summaries"
    summaries = aggregate(collect_results(root), destination)
    table = {}
    for row in summaries:
        if (
            row["nx"] not in (16, 64)
            or row["horizon"] != 4
            or row["evaluation_set"] != "common"
            or row["metric"] != "cc"
        ):
            continue
        key = (row["population_scale"], row["nx"], row["n1"])
        item = table.setdefault(
            key, dict(population_scale=key[0], nx=key[1], n1=key[2])
        )
        for field in ("mean", "sem"):
            item[f"{row['target']}_cc_4step_{field}"] = row[field]
    if table:
        with (destination / "population_summary.csv").open(
            "w", newline=""
        ) as stream:
            writer = csv.DictWriter(
                stream, fieldnames=list(next(iter(table.values())))
            )
            writer.writeheader()
            writer.writerows(table.values())
    plot_suite(summaries, root / "plots", settings)
    if not settings["enabled"]:
        return
    manifest = read_manifest(root)
    horizons = manifest["specification"]["settings"]["evaluation"]["horizons"]
    for member in manifest["members"].values():
        if member.get("state") != "complete":
            continue
        if member["case"]["dimensions"]["nx"] == 64 and 4 in horizons:
            (root / "plots").mkdir(exist_ok=True)
            forecast_example(
                root.parents[2] / member["predictions"],
                root / "plots" / "behavior_example.png",
                5, sample_rate, 4,
            )
            break
