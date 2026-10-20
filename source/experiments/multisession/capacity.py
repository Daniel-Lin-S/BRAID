"""Build the linear multi-session source-count by latent-width campaign.

Inputs are completed scaling and capacity plan.json files and current session
fingerprints. Output is one immutable plan.json referencing completed fits and
new 64/128-width source, scratch, and adaptation jobs in two directions.
"""

from copy import deepcopy

from experiments.storage.cache import fingerprint
from experiments.multisession.plan import implementation_digest
from experiments.multisession.scaling import INDY, LOCO, SHARING
from experiments.multisession.store import completed, fit_directory

CAMPAIGN = "multi_session_source_capacity_scaling"
ANALYSIS_ROOT = "multi_session_training_capacity_sensitivity"
COUNTS = (4, 33)
LATENTS = (32, 64, 128)
PHASE_WIDTHS = LATENTS[:2]
TARGETS_PER_DIRECTION = 36


def target_key(spec: dict) -> tuple:
    """Identify one target selection independent of model capacity."""
    return spec["session"], spec["fold"], spec["fraction"]


def add_job(jobs: dict, spec: dict) -> str:
    """Register one complete scientific fit identity."""
    key = fingerprint(spec)
    if key in jobs and jobs[key] != spec:
        raise ValueError(f"Conflicting fit identity: {key}.")
    jobs[key] = spec
    return key


def reference_jobs(plan: dict) -> tuple[dict, dict, dict, dict]:
    """Index completed 32-width sources, adaptations, and controls."""
    sources, adaptations, controls, preprocessing = {}, {}, {}, {}
    for key, spec in plan["jobs"].items():
        kind = spec["kind"]
        if kind == "source" and spec["source_count"] in COUNTS:
            stage = (spec["source_count"], spec["latent"],
                     spec["objective"])
            sources[stage] = key
        elif kind == "preprocessing":
            preprocessing[target_key(spec)] = key
        elif kind == "scratch":
            controls[target_key(spec), spec["latent"]] = key
        elif kind == "adaptation" and spec["source_count"] in COUNTS:
            adaptations[target_key(spec), spec["source_count"],
                        spec["latent"]] = key
    for count in COUNTS:
        for objective in ("neural", "neural_residual"):
            if (count, LATENTS[0], objective) not in sources:
                raise ValueError(
                    f"Missing {count}-source 32-width {objective} fit."
                )
    expected = 2 * TARGETS_PER_DIRECTION
    if (len(preprocessing) != expected or len(controls) != expected
            or len(adaptations) != expected * len(COUNTS)):
        raise ValueError("Scaling reference lacks matched target jobs.")
    return sources, adaptations, controls, preprocessing


def prior_controls(plan: dict) -> dict:
    """Index the previous 64-width Indy scratch specifications."""
    result = {}
    for key, spec in plan["jobs"].items():
        if spec["kind"] == "scratch" and spec["latent"] == 64:
            target = target_key(spec)
            if target in result:
                raise ValueError(f"Duplicate 64-width control: {target}.")
            result[target] = key
    if len(result) != TARGETS_PER_DIRECTION:
        raise ValueError("Expected 36 prior 64-width Indy controls.")
    return result


def validate_reused(
    settings: dict, sessions: dict, reference: dict,
    prior: dict, retained: set[str], prior_ids: set[str]
) -> None:
    """Check configuration, fingerprints, and immutable completions."""
    fields = (
        "data", "model", "seeds", "target_folds", "target_fractions",
        "held_out_sessions", "loco_targets", "source_training",
        "target_training", "evaluation_horizons",
    )
    for field in fields:
        if settings[field] != reference["settings"][field]:
            raise ValueError(f"Scaling setting changed: {field}.")
    if settings["paths"] != reference["settings"]["paths"]:
        raise ValueError("Scaling paths changed.")
    if tuple(settings["source_counts"]) != COUNTS:
        raise ValueError(f"Source counts must be {COUNTS}.")
    if tuple(settings["latent_widths"]) != LATENTS:
        raise ValueError(f"Latent widths must be {LATENTS}.")
    for plan, keys in ((reference, retained), (prior, prior_ids)):
        for key in keys:
            spec = plan["jobs"][key]
            if spec["kind"] == "source":
                for name, recorded in spec["sources"].items():
                    if sessions.get(name) != recorded:
                        raise ValueError(
                            f"Source fingerprint changed: {name}."
                        )
            elif sessions.get(spec["session"]) != spec["source"]:
                raise ValueError(
                    f"Target fingerprint changed: {spec['session']}."
                )
            if completed(fit_directory(plan, key)) is None:
                raise ValueError(f"Incomplete reused fit: {key}.")


def capacity_sources(jobs: dict, reference: dict, indexed: dict) -> dict:
    """Create first and residual stages at each requested capacity."""
    sources = {}
    for count in COUNTS:
        first_id = indexed[count, LATENTS[0], "neural"]
        residual_id = indexed[count, LATENTS[0], "neural_residual"]
        first = reference["jobs"][first_id]
        residual = reference["jobs"][residual_id]
        if first["sharing"] != "A" or residual["sharing"] != SHARING:
            raise ValueError("Reference source sharing differs.")
        if residual["stage1_source"] != first_id:
            raise ValueError("Reference source stages are not paired.")
        sources[count, LATENTS[0]] = residual_id
        for latent in LATENTS[1:]:
            first_spec = deepcopy(first)
            first_spec["latent"] = latent
            first_spec["implementation"] = implementation_digest("source")
            new_first = add_job(jobs, first_spec)
            residual_spec = deepcopy(residual)
            residual_spec.update(
                latent=latent,
                stage1_source=new_first,
                dependencies=[new_first],
                implementation=implementation_digest("source"),
            )
            sources[count, latent] = add_job(jobs, residual_spec)
    return sources


def build_capacity_plan(
    settings: dict, sessions: dict, reference: dict, prior: dict
) -> dict:
    """Build one queue with separate Indy and Loco capacity analyses."""
    indexed, adaptations, controls, preprocessing = reference_jobs(
        reference
    )
    previous = prior_controls(prior)
    indy = set(settings["held_out_sessions"])
    loco = set(settings["loco_targets"])
    if len(indy) != 4 or len(loco) != 4 or indy & loco:
        raise ValueError("Expected four disjoint Indy and Loco targets.")
    for target, old_id in previous.items():
        if target[0] not in indy:
            raise ValueError(f"Non-Indy prior control: {target}.")
        expected = deepcopy(reference["jobs"][controls[target, 32]])
        expected["latent"] = 64
        if prior["jobs"][old_id] != expected:
            raise ValueError(f"64-width control changed: {target}.")
    retained = set(indexed.values()) | set(adaptations.values())
    retained |= set(controls.values()) | set(preprocessing.values())
    validate_reused(
        settings, sessions, reference, prior,
        retained, set(previous.values()),
    )
    jobs = {key: deepcopy(reference["jobs"][key]) for key in retained}
    overrides = {
        key: value
        for key, value in reference.get("storage_overrides", {}).items()
        if key in retained
    }
    sources = capacity_sources(jobs, reference, indexed)
    analyses = {INDY: [], LOCO: []}
    for target in sorted(preprocessing):
        direction = INDY if target[0] in indy else LOCO
        if target[0] not in indy | loco:
            raise ValueError(f"Unexpected target session: {target[0]}.")
        scratch_template = reference["jobs"][controls[target, 32]]
        for latent in LATENTS:
            if latent == 32:
                scratch_id = controls[target, 32]
            elif latent == 64 and direction == INDY:
                scratch_id = previous[target]
                jobs[scratch_id] = deepcopy(prior["jobs"][scratch_id])
                old_override = prior.get(
                    "storage_overrides", {}
                ).get(scratch_id)
                if old_override is not None:
                    overrides[scratch_id] = old_override
            else:
                scratch_spec = deepcopy(scratch_template)
                scratch_spec["latent"] = latent
                scratch_id = add_job(jobs, scratch_spec)
                old_override = reference.get(
                    "storage_overrides", {}
                ).get(controls[target, 32])
                if old_override is not None:
                    overrides[scratch_id] = old_override
            analyses[direction].append(scratch_id)
            for count in COUNTS:
                if latent == 32:
                    adaptation_id = adaptations[target, count, 32]
                else:
                    template_id = adaptations[target, count, 32]
                    spec = deepcopy(reference["jobs"][template_id])
                    spec.update(
                        latent=latent,
                        source_fit=sources[count, latent],
                        source_training=jobs[
                            sources[count, latent]
                        ]["source_training"],
                        dependencies=[
                            sources[count, latent],
                            preprocessing[target],
                        ],
                        implementation=implementation_digest(
                            "adaptation", spec["model"]
                        ),
                    )
                    adaptation_id = add_job(jobs, spec)
                    old_override = reference.get(
                        "storage_overrides", {}
                    ).get(template_id)
                    if old_override is not None:
                        overrides[adaptation_id] = old_override
                analyses[direction].append(adaptation_id)
    expected = TARGETS_PER_DIRECTION * len(LATENTS) * (
        len(COUNTS) + 1
    )
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
            direction: f"{ANALYSIS_ROOT}/{direction}"
            for direction in analyses
        },
        analysis_kinds={direction: CAMPAIGN for direction in analyses},
        held_out_sessions=list(settings["held_out_sessions"]),
        loco_targets=list(settings["loco_targets"]),
        source_references=[
            sources[count, latent]
            for count in COUNTS for latent in LATENTS
        ],
    )
