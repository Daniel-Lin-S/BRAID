"""Build two four-source sensitivity roots from completed reference jobs.

Inputs are resolved scientific settings, current source fingerprints, and an
immutable completed reference plan. Output is a dependency-linked plan whose
new jobs coexist with unchanged source, preprocessing, scratch, and adaptation
identities. No completed fit payload is copied or rewritten.
"""

from copy import deepcopy

from experiments.storage.cache import fingerprint
from experiments.multisession.plan import implementation_digest
from experiments.multisession.store import completed, fit_directory

FILTER = "multi_session_filter_sensitivity"
RESIDUAL = "multi_session_residual_sharing_sensitivity"
CAMPAIGN = "multi_session_sharing_sensitivity"
REFERENCE_PROFILE = "AK"
FILTER_PROFILE = "AAK"
RESIDUAL_PROFILE = "AAK2"
SOURCE_COUNT = 4
LATENT_SIZE = 32
SEED = 42


def _reference_jobs(reference: dict) -> tuple[str, list[str], list[str]]:
    """Locate the paired old source, target, and scratch job identifiers."""
    jobs = reference["jobs"]
    sources = [
        key for key, spec in jobs.items()
        if spec["kind"] == "source"
        and spec["objective"] == "neural"
        and spec["latent"] == LATENT_SIZE
        and spec["seed"] == SEED
        and spec["sharing"] in ("A", "AK")
    ]
    baseline = [
        key for key, spec in jobs.items()
        if spec["kind"] == "adaptation"
        and spec["source_count"] == SOURCE_COUNT
        and spec["objective"] == "neural"
        and spec["latent"] == LATENT_SIZE
        and spec["sharing"] == REFERENCE_PROFILE
        and spec["seed"] == SEED
    ]
    scratch = [
        key for key, spec in jobs.items()
        if spec["kind"] == "scratch"
        and spec["latent"] == LATENT_SIZE
        and spec["seed"] == SEED
    ]
    matching = [
        key for key in sources
        if jobs[key]["source_count"] == SOURCE_COUNT
        and jobs[key]["sharing"] == REFERENCE_PROFILE
    ]
    if (
        len(matching) != 1
        or len(baseline) != 36
        or len(scratch) != 36
    ):
        raise ValueError(
            "Reference requires one four-source AK fit, "
            "36 AK adaptations, and 36 matched scratch controls."
        )
    if len(sources) != 8:
        raise ValueError(
            f"Expected eight completed source-scaling fits, got {len(sources)}."
        )
    return matching[0], sources, baseline + scratch


def _validate_reference(
    settings: dict, sessions: dict, reference: dict, keys: list[str]
) -> None:
    """Reject changed data or missing reference completions."""
    prior = reference["settings"]
    if (
        settings["paths"]["artifact_root"]
        != prior["paths"]["artifact_root"]
    ):
        raise ValueError(
            "Sensitivity artifact root must match the reference fit store."
        )
    for name in (
        "data", "model", "seeds", "target_folds", "target_fractions",
        "held_out_sessions", "source_training", "target_training",
        "evaluation_horizons",
    ):
        if settings[name] != prior[name]:
            raise ValueError(
                f"Sensitivity setting {name} differs from reference."
            )
    for key in keys:
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
            raise ValueError(f"Reference fit is incomplete: {key}.")


def build_sensitivity_plan(
    settings: dict, sessions: dict, reference: dict
) -> dict:
    """Build exactly two new source fits and 72 new adaptations."""
    old = reference["jobs"]
    source_id, source_refs, target_refs = _reference_jobs(reference)
    source_digest = implementation_digest()
    target_digest = implementation_digest("adaptation")
    dependencies = {
        old[key]["preprocessing"]
        for key in target_refs
        if "preprocessing" in old[key]
    }
    retained = set(source_refs + target_refs) | dependencies
    _validate_reference(settings, sessions, reference, sorted(retained))
    jobs = {key: deepcopy(old[key]) for key in sorted(retained)}
    source_spec = deepcopy(old[source_id])
    source_spec["implementation"] = source_digest
    source_spec["sharing"] = FILTER_PROFILE
    filter_source = fingerprint(source_spec)
    jobs[filter_source] = source_spec
    residual_spec = deepcopy(source_spec)
    residual_spec["objective"] = "neural_residual"
    residual_spec["sharing"] = RESIDUAL_PROFILE
    residual_source = fingerprint(residual_spec)
    jobs[residual_source] = residual_spec

    scratch = [
        key for key in target_refs if old[key]["kind"] == "scratch"
    ]
    baseline = [
        key for key in target_refs if old[key]["kind"] == "adaptation"
    ]
    members = {
        FILTER: scratch + baseline,
        RESIDUAL: scratch[:],
    }
    for key in baseline:
        for source_key, profile, objective, analysis in (
            (filter_source, FILTER_PROFILE, "neural", FILTER),
            (
                residual_source, RESIDUAL_PROFILE,
                "neural_residual", RESIDUAL,
            ),
        ):
            spec = deepcopy(old[key])
            spec.update(
                sharing=profile,
                objective=objective,
                source_fit=source_key,
                dependencies=[source_key, spec["preprocessing"]],
                implementation=target_digest,
            )
            fit_id = fingerprint(spec)
            jobs[fit_id] = spec
            members[analysis].append(fit_id)
    members[RESIDUAL].extend(
        key for key in members[FILTER]
        if jobs[key]["kind"] == "adaptation"
        and jobs[key]["sharing"] == FILTER_PROFILE
    )
    if len(members[FILTER]) != 108 or len(members[RESIDUAL]) != 108:
        raise ValueError("Each sensitivity requires 108 target members.")
    return dict(
        name=CAMPAIGN,
        storage=deepcopy(reference["storage"]),
        settings=deepcopy(settings),
        jobs=jobs,
        analyses=members,
        source_references=source_refs + [filter_source, residual_source],
        held_out_sessions=reference["held_out_sessions"],
    )
