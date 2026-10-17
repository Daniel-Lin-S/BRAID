"""Plan matched Indy and Indy-to-Loco sharing comparisons.

Inputs are a completed four-source sensitivity plan, its source and target
fingerprints, and four explicitly selected Loco targets. The output plan has
four separate analysis memberships and fit-store overrides for Loco jobs.
Completed source and Indy target identities remain unchanged.
"""

from copy import deepcopy

from experiments.storage.cache import fingerprint
from experiments.multisession.plan import implementation_digest
from experiments.multisession.store import completed, fit_directory

FILTER = "multi_session_filter_sensitivity"
RESIDUAL = "multi_session_residual_sharing_sensitivity"
CAMPAIGN = "multi_session_sharing_sensitivity"
DIRECTION = "indy_to_loco"
FILTER_PROFILES = ("scratch", "A", "AA", "AK", "AAK")
RESIDUAL_PROFILES = (
    "A", "A2", "A2_fw", "K2_fw", "A2_A2_fw_K2_fw",
)
STAGE2_PROFILES = RESIDUAL_PROFILES[1:]
SOURCE_COUNT = 4
LATENT_SIZE = 32
SEED = 42
TARGET_COUNT = 36
LOCO_STORES = {
    "adaptation": "multi_session_training/indy_to_loco/fits",
    "scratch": "experiments/BRAID_indy_to_loco_matched_scratch",
    "preprocessing": (
        "experiments/BRAID_indy_to_loco_matched_preprocessing"
    ),
}


def target_key(spec: dict) -> tuple:
    """Identify one matched session, fold, and target fraction."""
    return spec["session"], spec["fold"], spec["fraction"]


def reference_jobs(reference: dict) -> tuple[dict, dict, dict]:
    """Find the completed four-source fits and matched target controls."""
    jobs = reference["jobs"]
    source = {}
    targets = {}
    scratch = {}
    for key, spec in jobs.items():
        if spec["kind"] == "source":
            if (
                spec["objective"] == "neural"
                and spec["source_count"] == SOURCE_COUNT
                and spec["latent"] == LATENT_SIZE
                and spec["seed"] == SEED
                and spec["sharing"] in ("A", "AK", "AAK")
            ):
                profile = spec["sharing"]
                if profile in source:
                    raise ValueError(
                        f"Duplicate four-source {profile} reference."
                    )
                source[profile] = key
        elif (
            spec["kind"] in ("adaptation", "scratch")
            and spec["latent"] == LATENT_SIZE
            and spec["seed"] == SEED
            and spec["session"] in reference["held_out_sessions"]
        ):
            matched = target_key(spec)
            if spec["kind"] == "scratch":
                scratch[matched] = key
            elif (
                spec["source_count"] == SOURCE_COUNT
                and spec["objective"] == "neural"
                and spec["sharing"] in ("A", "AK", "AAK")
            ):
                targets[spec["sharing"], matched] = key
    expected = set(scratch)
    if set(source) != {"A", "AK", "AAK"}:
        raise ValueError("Missing completed A, AK, or AAK source fit.")
    if len(expected) != TARGET_COUNT:
        raise ValueError(
            f"Expected {TARGET_COUNT} scratch targets, got {len(expected)}."
        )
    for profile in ("A", "AK", "AAK"):
        found = {
            matched for name, matched in targets if name == profile
        }
        if found != expected:
            raise ValueError(
                f"{profile} target set differs from matched scratch."
            )
    return source, targets, scratch


def validate_reference(
    settings: dict, sessions: dict, reference: dict, keys: set
) -> None:
    """Check completed dependencies and unchanged scientific settings."""
    previous = reference["settings"]
    fields = (
        "data", "model", "seeds", "target_folds",
        "target_fractions", "held_out_sessions",
        "source_training", "target_training",
        "evaluation_horizons",
    )
    if settings["paths"]["artifact_root"] != (
        previous["paths"]["artifact_root"]
    ):
        raise ValueError("Reference artifact root differs.")
    for field in fields:
        if settings[field] != previous[field]:
            raise ValueError(f"Reference setting changed: {field}.")
    for key in keys:
        spec = reference["jobs"][key]
        if spec["kind"] == "source":
            for name, recorded in spec["sources"].items():
                if sessions.get(name) != recorded:
                    raise ValueError(
                        f"Source fingerprint changed: {name}."
                    )
        elif spec["kind"] != "preprocessing":
            if sessions.get(spec["session"]) != spec["source"]:
                raise ValueError(
                    f"Target fingerprint changed: {spec['session']}."
                )
        if completed(fit_directory(reference, key)) is None:
            raise ValueError(f"Incomplete reference fit: {key}.")


def add_source(
    jobs: dict, template: dict, sharing: str,
    objective: str, first_stage: str | None = None,
) -> str:
    """Add one source identity without changing referenced completions."""
    spec = deepcopy(template)
    spec["sharing"] = sharing
    spec["objective"] = objective
    spec["implementation"] = implementation_digest()
    if first_stage is not None:
        spec["stage1_sharing"] = "A"
        spec["stage2_sharing"] = sharing
        spec["stage1_source"] = first_stage
        spec["dependencies"] = [first_stage]
    key = fingerprint(spec)
    jobs[key] = spec
    return key


def add_target(
    jobs: dict, template: dict, sharing: str,
    source_id: str, objective: str,
    target: dict | None = None,
    preprocessing: str | None = None,
) -> str:
    """Add a dependency-linked target adaptation."""
    spec = deepcopy(template)
    if target is not None:
        spec.update(target)
        spec["transfer_direction"] = DIRECTION
    if preprocessing is not None:
        spec["preprocessing"] = preprocessing
    spec.update(
        sharing=sharing,
        objective=objective,
        source_fit=source_id,
        dependencies=[source_id, spec["preprocessing"]],
        implementation=implementation_digest("adaptation"),
    )
    key = fingerprint(spec)
    jobs[key] = spec
    return key


def add_loco_control(
    jobs: dict, template: dict, target: dict,
    preprocessing: str | None = None,
) -> str:
    """Add a Loco preprocessing or scratch job."""
    spec = deepcopy(template)
    spec.update(target)
    spec["transfer_direction"] = DIRECTION
    spec["implementation"] = implementation_digest(
        spec["kind"]
    )
    if preprocessing is not None:
        spec["preprocessing"] = preprocessing
        spec["dependencies"] = [preprocessing]
    key = fingerprint(spec)
    jobs[key] = spec
    return key


def build_sensitivity_plan(
    settings: dict, sessions: dict, reference: dict
) -> dict:
    """Build four separate matched analyses in one dependency graph."""
    source, targets, scratch = reference_jobs(reference)
    old = reference["jobs"]
    target_keys = sorted(scratch)
    prerequisites = {
        old[key]["preprocessing"] for key in targets.values()
    }
    retained = set(source.values()) | set(targets.values())
    retained |= set(scratch.values()) | prerequisites
    validate_reference(settings, sessions, reference, retained)
    loco = settings["loco_targets"]
    if (
        len(loco) != SOURCE_COUNT
        or len(set(loco)) != len(loco)
        or any(not name.startswith("loco_") for name in loco)
        or set(loco) - set(sessions)
    ):
        raise ValueError("Expected four distinct available Loco targets.")
    jobs = {key: deepcopy(old[key]) for key in sorted(retained)}
    source_ids = dict(source)
    source_ids["AA"] = add_source(
        jobs, old[source["A"]], "AA", "neural"
    )
    for profile in STAGE2_PROFILES:
        source_ids[profile] = add_source(
            jobs, old[source["A"]], profile,
            "neural_residual", source["A"],
        )
    loco_filter = f"{FILTER}_{DIRECTION}"
    loco_residual = f"{RESIDUAL}_{DIRECTION}"
    analyses = {
        FILTER: [],
        RESIDUAL: [],
        loco_filter: [],
        loco_residual: [],
    }
    overrides = {}
    for matched in target_keys:
        analyses[FILTER].append(scratch[matched])
        for profile in ("A", "AK", "AAK"):
            key = targets[profile, matched]
            analyses[FILTER].append(key)
            if profile == "A":
                analyses[RESIDUAL].append(key)
        template = old[targets["A", matched]]
        for profile in ("AA", *STAGE2_PROFILES):
            objective = (
                "neural" if profile == "AA" else "neural_residual"
            )
            key = add_target(
                jobs, template, profile, source_ids[profile],
                objective,
            )
            group = FILTER if profile == "AA" else RESIDUAL
            analyses[group].append(key)
    prep_template = old[next(iter(prerequisites))]
    scratch_template = old[scratch[target_keys[0]]]
    adaptation_template = old[targets["A", target_keys[0]]]
    for session in loco:
        for fold in settings["target_folds"]:
            for fraction in settings["target_fractions"]:
                target = dict(
                    session=session,
                    fold=fold,
                    fraction=fraction,
                    source=sessions[session],
                )
                prep = add_loco_control(
                    jobs, prep_template, target
                )
                overrides[prep] = LOCO_STORES["preprocessing"]
                control = add_loco_control(
                    jobs, scratch_template, target, prep
                )
                overrides[control] = LOCO_STORES["scratch"]
                analyses[loco_filter].append(control)
                for profile in ("A", "AA", "AK", "AAK"):
                    key = add_target(
                        jobs, adaptation_template, profile,
                        source_ids[profile], "neural",
                        target, prep,
                    )
                    overrides[key] = LOCO_STORES["adaptation"]
                    analyses[loco_filter].append(key)
                    if profile == "A":
                        analyses[loco_residual].append(key)
                for profile in STAGE2_PROFILES:
                    key = add_target(
                        jobs, adaptation_template, profile,
                        source_ids[profile], "neural_residual",
                        target, prep,
                    )
                    overrides[key] = LOCO_STORES["adaptation"]
                    analyses[loco_residual].append(key)
    for name, members in analyses.items():
        if len(members) != TARGET_COUNT * len(FILTER_PROFILES):
            raise ValueError(
                f"{name} has {len(members)} members; expected 180."
            )
        if len(members) != len(set(members)):
            raise ValueError(f"Duplicate target membership: {name}.")
    return dict(
        name=CAMPAIGN,
        settings=deepcopy(settings),
        storage=deepcopy(reference.get("storage", {})),
        storage_overrides=overrides,
        jobs=jobs,
        analyses=analyses,
        analysis_paths={
            FILTER: FILTER,
            RESIDUAL: RESIDUAL,
            loco_filter: f"{FILTER}/{DIRECTION}",
            loco_residual: f"{RESIDUAL}/{DIRECTION}",
        },
        analysis_kinds={
            FILTER: FILTER,
            RESIDUAL: RESIDUAL,
            loco_filter: FILTER,
            loco_residual: RESIDUAL,
        },
        source_references=[
            source_ids[name] for name in ("A", "AA", "AK", "AAK")
        ],
        held_out_sessions=reference["held_out_sessions"],
        loco_targets=list(loco),
    )
