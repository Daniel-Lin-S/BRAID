"""Build source-session scaling from immutable Indy and Loco fits.

Inputs are two reference plan.json files and current session fingerprints.
Output is a plan.json with reused controls and four-source fits, plus six new
source fits and their target adaptations in two direction-specific analyses.
"""

from copy import deepcopy
import json

from experiments.storage.cache import fingerprint
from experiments.multisession.plan import implementation_digest
from experiments.multisession.store import completed, fit_directory

CAMPAIGN = "multi_session_source_scaling"
INDY = "Indy → Indy (held-out)"
LOCO = "Indy → Loco"
COUNTS = (4, 8, 16, 33)
STAGE_UPDATES = (5000, 10000, 20000, 41250)
VALIDATION_INTERVALS = (25, 50, 100, 206)
SHARING = "A2_A2_fw_K2_fw"
TARGET_FOLDS = (0, 2, 4)
TARGET_FRACTIONS = (0.05, 0.20, 0.80)
TARGETS_PER_DIRECTION = 36


def target_key(spec: dict) -> tuple:
    """Return the matched recording, fold, and label fraction."""
    return spec["session"], spec["fold"], spec["fraction"]


def reference_jobs(reference: dict) -> tuple[dict, dict, dict]:
    """Index completed four-source fits and matched scratch controls."""
    sources, adaptations, controls = {}, {}, {}
    for key, spec in reference["jobs"].items():
        if spec["kind"] == "source":
            if spec["source_count"] == COUNTS[0] and spec["sharing"] in (
                "A", SHARING
            ):
                sources[spec["sharing"]] = key
        elif spec["kind"] in ("scratch", "adaptation"):
            target = target_key(spec)
            if spec["kind"] == "scratch":
                controls[target] = key
            elif spec["source_count"] == COUNTS[0] and (
                spec["sharing"] == SHARING
            ):
                adaptations[target] = key
    expected = 2 * TARGETS_PER_DIRECTION
    if set(sources) != {"A", SHARING}:
        raise ValueError("Missing four-source A or full-RNN2 source fit.")
    if len(controls) != expected or set(controls) != set(adaptations):
        raise ValueError(
            f"Expected {expected} matched adaptations and scratch fits."
        )
    return sources, adaptations, controls


def source_sets(reference: dict, sessions: dict) -> dict[int, list[str]]:
    """Read and validate the original nested Indy source sets."""
    selected = {}
    for spec in reference["jobs"].values():
        if (
            spec["kind"] == "source"
            and spec["objective"] == "neural"
            and spec["sharing"] == "A"
            and spec["latent"] == 32
            and spec["seed"] == 42
        ):
            count = spec["source_count"]
            selected[count] = spec["sessions"]
            for name, recorded in spec["sources"].items():
                if sessions.get(name) != recorded:
                    raise ValueError(f"Source fingerprint changed: {name}.")
    if set(selected) != set(COUNTS):
        raise ValueError("Reference lacks one A source for each count.")
    for previous, current in zip(COUNTS, COUNTS[1:]):
        if selected[current][:previous] != selected[previous]:
            raise ValueError("Indy source sets are not nested.")
    return selected


def check_reference(
    settings: dict, sessions: dict, reference: dict,
    source_reference: dict, retained: set[str]
) -> None:
    """Require compatible settings and checksum-valid reused fits."""
    previous = reference["settings"]
    fields = (
        "data", "model", "seeds", "target_folds",
        "target_fractions", "held_out_sessions", "loco_targets",
        "source_training", "target_training", "evaluation_horizons",
    )
    for field in fields:
        if settings[field] != previous[field]:
            raise ValueError(f"Reference setting changed: {field}.")
    if settings["paths"]["artifact_root"] != (
        previous["paths"]["artifact_root"]
    ):
        raise ValueError("Artifact root differs from completed fits.")
    if tuple(settings["source_counts"]) != COUNTS:
        raise ValueError(f"Source counts must be {COUNTS}.")
    if tuple(settings["target_folds"]) != TARGET_FOLDS:
        raise ValueError(f"Target folds must be {TARGET_FOLDS}.")
    if tuple(settings["target_fractions"]) != TARGET_FRACTIONS:
        raise ValueError(f"Target fractions must be {TARGET_FRACTIONS}.")
    if reference["held_out_sessions"] != (
        source_reference["held_out_sessions"]
    ):
        raise ValueError("Held-out Indy sessions differ between plans.")
    for key in retained:
        spec = reference["jobs"][key]
        if spec["kind"] == "source":
            for name, recorded in spec["sources"].items():
                if sessions.get(name) != recorded:
                    raise ValueError(f"Source fingerprint changed: {name}.")
        elif sessions.get(spec["session"]) != spec["source"]:
            raise ValueError(
                f"Target fingerprint changed: {spec['session']}."
            )
        if completed(fit_directory(reference, key)) is None:
            raise ValueError(f"Incomplete reference fit: {key}.")


def add_job(jobs: dict, spec: dict) -> str:
    """Address one fit by its complete scientific specification."""
    key = fingerprint(spec)
    if key in jobs and jobs[key] != spec:
        raise ValueError(f"Conflicting fit identity: {key}.")
    jobs[key] = spec
    return key


def source_jobs(
    jobs: dict, reference: dict, sets: dict,
    existing: dict, sessions: dict
) -> dict[int, str]:
    """Create A and full-RNN2 sources at the three larger counts."""
    old = reference["jobs"]
    residual = {COUNTS[0]: existing[SHARING]}
    for count, cap, interval in zip(
        COUNTS[1:], STAGE_UPDATES[1:], VALIDATION_INTERVALS[1:]
    ):
        training = deepcopy(old[existing["A"]]["source_training"])
        training.update(
            maximum_updates=cap,
            validation_interval=interval,
        )
        first = deepcopy(old[existing["A"]])
        first.update(
            sessions=sets[count],
            sources={name: sessions[name] for name in sets[count]},
            source_count=count,
            source_training=training,
            implementation=implementation_digest(),
        )
        first_id = add_job(jobs, first)
        second = deepcopy(old[existing[SHARING]])
        second.update(
            sessions=sets[count],
            sources=first["sources"],
            source_count=count,
            source_training=dict(training, maximum_updates=2 * cap),
            stage1_source=first_id,
            dependencies=[first_id],
            implementation=implementation_digest(),
        )
        residual[count] = add_job(jobs, second)
    return residual


def build_scaling_plan(
    settings: dict, sessions: dict, reference: dict,
    source_reference: dict
) -> dict:
    """Build two direction-specific analyses and one shared fit queue."""
    existing, old_adaptations, controls = reference_jobs(reference)
    sets = source_sets(source_reference, sessions)
    old = reference["jobs"]
    prerequisites = {
        old[key]["preprocessing"] for key in controls.values()
    }
    retained = set(existing.values()) | set(old_adaptations.values())
    retained |= set(controls.values()) | prerequisites
    check_reference(
        settings, sessions, reference, source_reference, retained
    )
    if sets[COUNTS[0]] != old[existing["A"]]["sessions"]:
        raise ValueError("Four-source Indy session set changed.")
    first_directory = fit_directory(reference, existing["A"])
    first_records = json.loads(
        (first_directory / "components.json").read_text()
    )
    first_training = first_records["neural"]["training"]
    stage_settings = deepcopy(old[existing[SHARING]]["source_training"])
    stage_settings["maximum_updates"] //= 2
    comparable = dict(
        first_training["settings"],
        maximum_updates=stage_settings["maximum_updates"],
    )
    if (
        first_training["updates"] > STAGE_UPDATES[0]
        or comparable != stage_settings
    ):
        raise ValueError("Four-source A checkpoint exceeds stage budget.")
    jobs = {key: deepcopy(old[key]) for key in sorted(retained)}
    overrides = {
        key: value
        for key, value in reference.get("storage_overrides", {}).items()
        if key in retained
    }
    sources = source_jobs(jobs, reference, sets, existing, sessions)
    analyses = {INDY: [], LOCO: []}
    held_out = set(settings["held_out_sessions"])
    loco = set(settings["loco_targets"])
    for target, control_id in sorted(controls.items()):
        name = target[0]
        direction = INDY if name in held_out else LOCO
        if name not in held_out | loco:
            raise ValueError(f"Unexpected target recording: {name}.")
        analyses[direction].append(control_id)
        four_id = old_adaptations[target]
        analyses[direction].append(four_id)
        template = old[four_id]
        for count in COUNTS[1:]:
            spec = deepcopy(template)
            spec.update(
                source_count=count,
                source_fit=sources[count],
                source_training=jobs[sources[count]]["source_training"],
                dependencies=[sources[count], spec["preprocessing"]],
                implementation=implementation_digest("adaptation"),
            )
            key = add_job(jobs, spec)
            if direction == LOCO:
                overrides[key] = reference["storage_overrides"][four_id]
            analyses[direction].append(key)
    expected = TARGETS_PER_DIRECTION * (len(COUNTS) + 1)
    for direction, members in analyses.items():
        if len(members) != expected or len(set(members)) != expected:
            raise ValueError(
                f"{direction} has {len(members)} members; "
                f"expected {expected}."
            )
    return dict(
        name=CAMPAIGN,
        settings=deepcopy(settings),
        storage=deepcopy(reference.get("storage", {})),
        storage_overrides=overrides,
        jobs=jobs,
        analyses=analyses,
        analysis_paths={
            INDY: f"multi_session_source_scaling/{INDY}",
            LOCO: f"multi_session_source_scaling/{LOCO}",
        },
        analysis_kinds={INDY: CAMPAIGN, LOCO: CAMPAIGN},
        held_out_sessions=list(settings["held_out_sessions"]),
        loco_targets=list(settings["loco_targets"]),
        source_references=[sources[count] for count in COUNTS],
    )
