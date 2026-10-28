"""Build the behavioral source compute-scaling campaign.

Inputs are resolved settings, current session fingerprints, and immutable
source/target reference plans. The output plan contains four source fits,
matched n1=n2=16 target fits, and reusable preprocessing dependencies.
"""

from copy import deepcopy

from experiments.multisession.plan import implementation_digest
from experiments.multisession.behavior_supervision.models import (
    STAGE1_SHARING, STAGE2_SHARING,
)
from experiments.multisession.store import completed, fit_directory
from experiments.storage.cache import fingerprint

CAMPAIGN = "multi_session_behavior_compute_scaling"
DIRECT_CAMPAIGN = "multi_session_behavior_scaling_no_preprocess"
DIRECT_PROTOCOL = "recurrent_compute_branching_v1"
DIRECT_COMPONENTS = (
    "behavior_relevant", "neural_readout", "residual",
)
COUNTS = (4, 8, 16, 33)
FOLDS = (0, 2, 4)
FRACTIONS = (0.05, 0.20, 0.80)
LATENT = 16
SEED = 42
SHARING = "A1_fw_RNN2"
COMPONENTS = (
    "preprocessing_neural",
    "preprocessing_behavior",
    "behavior_relevant",
    "neural_readout",
    "residual",
)
CHECKPOINTS = (40, 80, 160, 320, 640)
TOTAL_UPDATES = (50, 100, 200, 400, 800)
WARMUP_UPDATES = 10
PROTOCOL = "recurrent_compute_fixed_preprocessing_v1"
FIXED_EXPOSURE = 64000
VALIDATION_EXPOSURE = 800
FIXED_PATIENCE = 10
INNER_FRACTION = 0.1
RECURRENT_STAGES = ("behavior_relevant", "residual")
WARMUP_START = 1e-7
LEARNING_RATE = 1e-3
LOCAL_WINDOWS = {4: 8, 8: 8, 16: 8, 33: 4}
EXTENSION_CHECKPOINTS = (1280, 1920)
EXTENSION_TOTAL_UPDATES = (1600, 2400)


LEGACY_SOURCE_IMPLEMENTATION = (
    "279b1e0d1595eee86aba72fe4c52c7c5c27c514ac3d1b220a7e360d3e6ff76cc"
)
DIRECT_SOURCE_IMPLEMENTATION = fingerprint(dict(
    target="recorded_behavior_v1",
    pipeline="matched_rnn1_readout_rnn2_v1",
    compute="branching_flops_v1",
))


def source_implementation(direct: bool = False) -> str:
    """Return a versioned scientific implementation identity."""
    return (
        DIRECT_SOURCE_IMPLEMENTATION
        if direct else LEGACY_SOURCE_IMPLEMENTATION
    )


def source_sets(reference: dict, sessions: dict) -> dict[int, list[str]]:
    """Read nested source sets from the completed source scaling plan."""
    selected = {}
    for fit_id in reference["source_references"]:
        spec = reference["jobs"][fit_id]
        count = spec["source_count"]
        selected[count] = spec["sessions"]
        for name in selected[count]:
            if sessions.get(name) != spec["sources"][name]:
                raise ValueError(f"Source fingerprint changed: {name}.")
    if set(selected) != set(COUNTS):
        raise ValueError(f"Expected source counts {COUNTS}.")
    for left, right in zip(COUNTS, COUNTS[1:]):
        if selected[right][:left] != selected[left]:
            raise ValueError("Source session sets are not nested.")
    return selected


def target_templates(reference: dict, held_out: list[str]) -> dict:
    """Index one preprocessing, scratch, and adaptation per target split."""
    result = {}
    for key, spec in reference["jobs"].items():
        if (
            spec["kind"] not in ("scratch", "adaptation")
            or spec["session"] not in held_out
            or spec.get("latent") != 32
            or spec.get("seed") != SEED
        ):
            continue
        if spec["kind"] == "adaptation" and not (
            spec["objective"] == "behavior"
            and spec["sharing"] == "AKC"
            and spec["source_count"] == 33
        ):
            continue
        target = (spec["session"], spec["fold"], spec["fraction"])
        result.setdefault(target, {})[spec["kind"]] = (key, spec)
        prep_id = spec["preprocessing"]
        result[target]["preprocessing"] = (
            prep_id, reference["jobs"][prep_id]
        )
    expected = {
        (name, fold, fraction)
        for name in held_out
        for fold in FOLDS
        for fraction in FRACTIONS
    }
    if set(result) != expected:
        raise ValueError("Reference target split coverage differs.")
    if any(set(row) != {
        "scratch", "adaptation", "preprocessing"
    } for row in result.values()):
        raise ValueError("Missing matched target template.")
    return result


def build_plan(
    settings: dict, sessions: dict, source_reference: dict,
    target_reference: dict,
) -> dict:
    """Create four source jobs and matched held-out target dependencies."""
    if (
        tuple(settings["source_counts"]) != COUNTS
        or settings["seeds"] != [SEED]
        or tuple(settings["target_folds"]) != FOLDS
        or tuple(settings["target_fractions"]) != FRACTIONS
        or settings["latent"] != LATENT
    ):
        raise ValueError("Compute-scaling design settings differ.")
    source = settings["source_training"]
    expected = dict(
        maximum_updates=CHECKPOINTS[-1],
        checkpoints=list(CHECKPOINTS),
        total_updates=list(TOTAL_UPDATES),
        protocol=PROTOCOL,
        fixed_exposure=FIXED_EXPOSURE,
        validation_exposure=VALIDATION_EXPOSURE,
        fixed_patience=FIXED_PATIENCE,
        inner_validation_fraction=INNER_FRACTION,
        update_multipliers={4: 1, 8: 1, 16: 1, 33: 2},
        warmup_updates=WARMUP_UPDATES,
        warmup_start_learning_rate=WARMUP_START,
        learning_rate=LEARNING_RATE,
        session_batch_sizes=LOCAL_WINDOWS,
    )
    for key, value in expected.items():
        if source[key] != value:
            raise ValueError(
                f"Source schedule {key} differs from the agreed design."
            )
    for name in ("data", "model", "held_out_sessions",
                 "evaluation_horizons", "target_training"):
        if settings[name] != target_reference["settings"][name]:
            raise ValueError(f"Reference setting changed: {name}.")
    if settings["paths"]["artifact_root"] != (
        target_reference["settings"]["paths"]["artifact_root"]
    ):
        raise ValueError("Artifact root differs from reference fits.")
    held_out = settings["held_out_sessions"]
    sets = source_sets(source_reference, sessions)
    templates = target_templates(target_reference, held_out)
    jobs, members, sources = {}, [], {}
    schedule_keys = (
        "maximum_updates", "checkpoints", "total_updates",
        "protocol", "fixed_exposure", "validation_exposure",
        "fixed_patience", "inner_validation_fraction",
        "update_multipliers",
        "warmup_updates", "warmup_start_learning_rate",
        "learning_rate", "session_batch_sizes",
    )
    schedule = {key: settings["source_training"][key]
                for key in schedule_keys}
    implementation = source_implementation()
    for count in COUNTS:
        template = next(
            spec for spec in target_reference["jobs"].values()
            if spec["kind"] == "source"
            and spec["objective"] == "behavior"
            and spec["source_count"] == 33
            and spec["latent"] == 32
        )
        spec = deepcopy(template)
        spec.update(
            implementation=implementation,
            latent=LATENT,
            sharing=SHARING,
            stage1_sharing=STAGE1_SHARING,
            stage2_sharing=STAGE2_SHARING,
            sessions=sets[count],
            sources={name: sessions[name] for name in sets[count]},
            source_count=count,
            source_training=schedule,
            dependencies=[],
        )
        key = fingerprint(spec)
        jobs[key] = spec
        sources[count] = key
        members.append(key)
    for target, records in sorted(templates.items()):
        for kind in ("preprocessing", "scratch"):
            old_key, old_spec = records[kind]
            if kind == "preprocessing":
                if completed(fit_directory(target_reference, old_key)) is None:
                    raise ValueError(
                        f"Incomplete target preprocessing: {old_key}."
                    )
                jobs[old_key] = deepcopy(old_spec)
                members.append(old_key)
                continue
            spec = deepcopy(old_spec)
            spec["latent"] = LATENT
            spec.pop("source_training", None)
            spec["implementation"] = implementation_digest(
                "scratch", spec["model"]
            )
            key = fingerprint(spec)
            jobs[key] = spec
            members.append(key)
        _, template = records["adaptation"]
        for count in COUNTS:
            spec = deepcopy(template)
            spec.update(
                latent=LATENT,
                sharing=SHARING,
                source_count=count,
                source_fit=sources[count],
                dependencies=[
                    sources[count], records["preprocessing"][0]
                ],
                implementation=implementation_digest(
                    "adaptation", template["model"]
                ),
            )
            spec.pop("source_training", None)
            key = fingerprint(spec)
            jobs[key] = spec
            members.append(key)
    if len(set(members)) != len(members):
        raise ValueError("Duplicate compute-scaling job membership.")
    return dict(
        name=CAMPAIGN,
        settings=deepcopy(settings),
        storage=deepcopy(target_reference.get("storage", {})),
        jobs=jobs,
        analyses={CAMPAIGN: members},
        analysis_paths={CAMPAIGN: CAMPAIGN},
        held_out_sessions=list(held_out),
        source_references=[sources[count] for count in COUNTS],
    )


def build_extension_plan(
    settings: dict, sessions: dict, reference: dict,
) -> dict:
    """Plan only four source continuations with pinned completed ancestors.

    Parameters
    ----------
    settings : dict
        Original experiment settings with the two agreed larger budgets.
    sessions : dict
        Current input data fingerprints indexed by recording name.
    reference : dict
        Completed five-budget campaign; target jobs are not queued.

    Returns
    -------
    dict
        Standard source-only plan referencing immutable parent fits.
    """
    previous = reference["settings"]
    for name in ("data", "model", "latent", "seeds", "source_counts",
                 "held_out_sessions", "evaluation_horizons"):
        if settings[name] != previous[name]:
            raise ValueError(f"Continuation setting changed: {name}.")
    old_schedule = previous["source_training"]
    schedule = settings["source_training"]
    expected = dict(
        old_schedule,
        maximum_updates=EXTENSION_CHECKPOINTS[-1],
        checkpoints=list(CHECKPOINTS + EXTENSION_CHECKPOINTS),
        total_updates=list(TOTAL_UPDATES + EXTENSION_TOTAL_UPDATES),
    )
    if schedule != expected:
        raise ValueError("Continuation differs from the agreed 2x/3x ladder.")
    if old_schedule["total_updates"] != list(TOTAL_UPDATES):
        raise ValueError("Expected the completed five-budget source campaign.")
    jobs, sources = {}, []
    for old_key in reference["source_references"]:
        original = reference["jobs"][old_key]
        for name in original["sessions"]:
            if sessions[name] != original["sources"][name]:
                raise ValueError(f"Source input changed: {name}.")
        marker = completed(fit_directory(reference, old_key))
        if marker is None:
            raise ValueError(f"Incomplete source ancestor: {old_key}.")
        spec = deepcopy(original)
        spec.update(
            source_training=deepcopy(schedule),
            implementation=source_implementation(), dependencies=[],
            continuation=dict(
                source_fit=old_key, checksums=marker["checksums"],
                completed_total_updates=list(TOTAL_UPDATES),
                mode="matched_pipeline_rnn1_main_prefix",
            ),
        )
        key = fingerprint(spec)
        jobs[key] = spec
        sources.append(key)
    if sorted(job["source_count"] for job in jobs.values()) != list(COUNTS):
        raise ValueError("Continuation requires all four source counts.")
    return dict(
        name=CAMPAIGN, settings=deepcopy(settings), source_only=True,
        storage=deepcopy(reference.get("storage", {})), jobs=jobs,
        analyses={CAMPAIGN: sources}, analysis_paths={CAMPAIGN: CAMPAIGN},
        held_out_sessions=list(settings["held_out_sessions"]),
        source_references=sources,
    )


def check_direct_settings(settings: dict, reference: dict) -> None:
    """Reject changes outside the agreed recorded-behavior target."""
    previous = reference["settings"]
    for key in ("data", "source_counts", "seeds", "latent",
                "evaluation_horizons", "held_out_sessions"):
        if settings[key] != previous[key]:
            raise ValueError(f"Direct behavior setting changed: {key}.")
    if tuple(settings["source_counts"]) != COUNTS or (
        settings["seeds"] != [SEED] or settings["latent"] != LATENT
    ):
        raise ValueError("Expected four source counts, seed 42, n1=n2=16.")
    model = deepcopy(previous["model"])
    model["model"]["dimensions"]["preprocessing_state_size"] = 0
    if settings["model"] != model:
        raise ValueError(
            "Direct model must differ only by disabling n0 preprocessing."
        )
    old = deepcopy(previous["source_training"])
    for key in ("session_batch_sizes", "update_multipliers"):
        old[key] = {int(count): value for count, value in old[key].items()}
    if settings.get("behavior_supervision") != {"target": "recorded"}:
        raise ValueError("Expected recorded behavior supervision mode.")
    schedule = dict(
        old, protocol=DIRECT_PROTOCOL,
        maximum_updates=EXTENSION_CHECKPOINTS[-1],
        checkpoints=list(CHECKPOINTS + EXTENSION_CHECKPOINTS),
        total_updates=list(TOTAL_UPDATES + EXTENSION_TOTAL_UPDATES),
    )
    if settings["source_training"] != schedule:
        raise ValueError("Direct source schedule differs from agreed ladder.")


def build_direct_plan(
    settings: dict, sessions: dict[str, dict], reference: dict,
) -> dict:
    """Create four source-only fits from verified nested source sets."""
    check_direct_settings(settings, reference)
    jobs = {}
    sources = []
    for old_key in reference["source_references"]:
        original = reference["jobs"][old_key]
        if original["source_count"] not in COUNTS:
            raise ValueError("Unexpected source count in reference plan.")
        if completed(fit_directory(reference, old_key)) is None:
            raise ValueError(f"Incomplete reference source fit: {old_key}.")
        for name in original["sessions"]:
            if sessions.get(name) != original["sources"][name]:
                raise ValueError(f"Source input changed: {name}.")
        spec = deepcopy(original)
        spec.update(
            model=deepcopy(settings["model"]),
            source_training=deepcopy(settings["source_training"]),
            behavior_supervision=deepcopy(settings[
                "behavior_supervision"
            ]),
            implementation=source_implementation(direct=True),
            dependencies=[],
        )
        spec.pop("continuation", None)
        key = fingerprint(spec)
        if key in jobs:
            raise ValueError("Duplicate direct-behavior source identity.")
        jobs[key] = spec
        sources.append(key)
    if sorted(job["source_count"] for job in jobs.values()) != (
        list(COUNTS)
    ):
        raise ValueError("Expected exactly four nested source fits.")
    return dict(
        name=DIRECT_CAMPAIGN, settings=deepcopy(settings),
        source_only=True,
        storage=deepcopy(reference.get("storage", {})), jobs=jobs,
        analyses={DIRECT_CAMPAIGN: sources},
        analysis_paths={DIRECT_CAMPAIGN: DIRECT_CAMPAIGN},
        held_out_sessions=list(settings["held_out_sessions"]),
        source_references=sources,
    )
