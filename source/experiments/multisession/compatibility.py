"""Validate reviewed legacy-fit compatibility for a pinned campaign.

Input is an immutable plan and a launch-local compatibility.json. The record
lists exact completed and retry job IDs, legacy and current implementation
signatures, and a frozen source snapshot checksum. Output is validation only;
completed artifacts and plan identities are never modified.
"""

import json
from pathlib import Path

from experiments.storage.cache import file_digest
from experiments.campaign_source import (
    freeze_source, source_code_digest, verify_source_code,
)


RECORD_KEYS = frozenset({
    "plan_sha256", "source_sha256", "completed", "retry", "current",
})
TARGET_KINDS = frozenset({"scratch", "adaptation"})
MLP = "multilayer_perceptron"


def load_compatibility(path: Path, plan_path: Path) -> dict:
    """Read a plan-bound record and verify the executing source snapshot.

    Parameters
    ----------
    path : Path
        Absolute launch-local compatibility record.
    plan_path : Path
        Immutable plan JSON file used by this coordinator or worker.

    Returns
    -------
    dict
        Validated compatibility mapping with exact job membership.
    """
    record = json.loads(path.read_text())
    if set(record) != RECORD_KEYS:
        raise ValueError(
            f"Expected compatibility keys {sorted(RECORD_KEYS)}, "
            f"got {sorted(record)}."
        )
    actual_plan = file_digest(plan_path)
    if record["plan_sha256"] != actual_plan:
        raise ValueError(
            f"Compatibility plan checksum mismatch: expected "
            f"{record['plan_sha256']}, got {actual_plan}."
        )
    verify_source_code(record["source_sha256"])
    if not isinstance(record["retry"], dict):
        raise ValueError("Compatibility retry must map job IDs to hashes.")
    if not isinstance(record["completed"], list):
        raise ValueError("Compatibility completed must list job IDs.")
    if not isinstance(record["current"], dict):
        raise ValueError("Compatibility current must map fit kinds to hashes.")
    return record


def validate_resume(plan: dict, states: dict, record: dict) -> None:
    """Require exact validated reuse membership before retry dispatch."""
    completed = set(record["completed"])
    retries = set(record["retry"])
    planned = set(plan["jobs"])
    if completed & retries or completed | retries != planned:
        raise ValueError(
            "Compatibility membership must partition every plan job."
        )
    actual = {key for key, state in states.items() if state == "complete"}
    if actual != completed:
        raise ValueError(
            f"Expected {len(completed)} reusable completions, "
            f"got {len(actual)}; missing={sorted(completed - actual)[:5]}, "
            f"unexpected={sorted(actual - completed)[:5]}."
        )
    for key in retries:
        spec = plan["jobs"][key]
        if spec["kind"] not in TARGET_KINDS:
            raise ValueError(
                f"Compatibility cannot retry {spec['kind']} {key}."
            )
        if record["retry"][key] != spec["implementation"]:
            raise ValueError(f"Legacy implementation mismatch for {key}.")
        architectures = {
            spec["model"]["model"][f"stage_{stage}"]["state_transition"]
            ["architecture"]
            for stage in (1, 2)
        }
        if architectures != {MLP}:
            raise ValueError(
                f"Compatibility requires MLP dynamics for {key}; "
                f"got {sorted(architectures)}."
            )


def validate_worker(
    key: str, spec: dict, current: str, record: dict | None,
) -> None:
    """Accept exact identity or one reviewed plan-bound legacy mapping."""
    if current == spec["implementation"]:
        return
    if record is None:
        raise RuntimeError(
            f"Numerical implementation changed for {key}: expected "
            f"{spec['implementation']}, got {current}."
        )
    if (
        record["retry"].get(key) != spec["implementation"]
        or record["current"].get(spec["kind"]) != current
    ):
        raise RuntimeError(
            f"No reviewed implementation compatibility for {key}: "
            f"legacy={spec['implementation']}, current={current}."
        )
