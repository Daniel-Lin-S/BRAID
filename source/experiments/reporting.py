"""BRAID-specific artifact reports, selected through the reporting plugin.

Inputs are completed metrics.json/predictions.npz artifacts. Outputs include
raw_metrics.csv/json, summary.csv/json, population_summary.csv, and the
configured figure suite. Incomplete folds are reported, never filled in.
"""

import csv
from pathlib import Path

from .analysis import read_manifest, record_plotting
from .evaluation import aggregate, collect_results
from .plots import plot_suite


def braid_report(root: Path, settings: dict, sample_rate: float) -> None:
    """Produce the required BRAID tables and optional figures from artifacts."""
    from .model_summary import write_model_summary

    record_plotting(root, settings)
    write_model_summary(root)
    manifest = read_manifest(root)
    members = list(manifest["members"].values())
    if not any(member.get("state") == "complete" for member in members):
        return
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
    horizons = manifest["specification"]["settings"]["evaluation"]["horizons"]
    expected = []
    for member in members:
        case = member["case"]
        scoring = ["full", "common"] if member["common_ids"] else ["full"]
        for horizon in horizons:
            for evaluation_set in scoring:
                for target in ("neural", "behavior"):
                    for metric in ("cc", "r2", "mse"):
                        candidate = dict(
                            configuration=case["name"],
                            nx=case["dimensions"]["nx"],
                            population_scale=case["population_scale"],
                            horizon=horizon, evaluation_set=evaluation_set,
                            target=target, metric=metric,
                        )
                        existing = next((
                            row for row in expected if all(
                                row[key] == value
                                for key, value in candidate.items()
                            )
                        ), None)
                        incomplete = member.get("state") != "complete"
                        if existing is None:
                            expected.append(dict(candidate, pending=incomplete))
                        else:
                            existing["pending"] |= incomplete
    plot_suite(
        summaries, root / "plots", settings, expected,
        pending=any(m.get("state") != "complete" for m in members),
    )
