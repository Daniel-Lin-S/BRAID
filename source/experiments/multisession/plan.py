"""Build immutable multi-session jobs and three independent analysis roots.

Inputs are the resolved campaign YAML and dataset source fingerprints.
plan.json contains jobs keyed by deterministic IDs, dependency IDs, numerical
settings, and analysis memberships. No runtime allocation enters fit identity.
"""

from copy import deepcopy
from pathlib import Path

import numpy as np

from experiments.storage.cache import fingerprint, file_digest
from experiments.storage.implementation import scientific_versions
from experiments.multisession.store import FIT_STORES

PRIMARY = "multi_session_training"
CAPACITY = PRIMARY + "_capacity_sensitivity"
SUPERVISION = PRIMARY + "_supervision_sensitivity"
ANALYSES = (PRIMARY, CAPACITY, SUPERVISION)
# Historical source IDs contain target-only file slots. Keep those slots
# stable: source fitting never executes either target integration module.
SOURCE_TARGET_SLOTS = {
    "experiments/multisession/target.py":
        "6594517aa20012e2b094f8eb03a5428b492f026987a2012aca31e8486d0f0381",
    "BRAID/BRAIDModel.py":
        "f81ef97b61577af6b67404cc1e4c6623b3b990fa6cd2848d1df0c62698399d6f",
}


def implementation_digest(kind: str = "source") -> str:
    """Fingerprint numerical transfer code and native BRAID dependencies."""
    here = Path(__file__).resolve().parent
    paths = [
        here / name
        for name in (
            "data.py",
            "source.py",
            "target.py",
            "training.py",
        )
    ]
    paths += list((here.parents[1] / "BRAID").rglob("*.py"))
    paths += [
        here.parent / name
        for name in (
            "datasets/nhp.py",
            "datasets/windows.py",
            "contracts.py",
            "configuration.py",
            "models/braid_backend.py",
            "reports/evaluation.py",
        )
    ]
    if kind != "source":
        paths.append(here / "preprocessing.py")
    hashes = {
        str(p.relative_to(here.parents[1])): file_digest(p)
        for p in sorted(paths)
    }
    if kind == "source":
        hashes.update(SOURCE_TARGET_SLOTS)
    return fingerprint(hashes)


def build_plan(settings: dict, sessions: dict[str, dict]) -> dict:
    """Expand paired source, preprocessing, adaptation, and scratch jobs."""
    targets = settings["held_out_sessions"]
    if len(targets) != len(set(targets)) or set(targets) - set(sessions):
        raise ValueError("Held-out sessions must be unique available sessions.")
    sources = sorted(set(sessions) - set(targets))
    if max(settings["source_counts"]) != len(sources):
        raise ValueError(
            f"Expected {max(settings['source_counts'])} sources, "
            f"got {len(sources)}."
        )
    common = dict(
        implementation=implementation_digest(),
        numerical_versions=scientific_versions(),
        data=settings["data"],
        model=settings["model"],
        source_training=settings["source_training"],
        target_training=settings["target_training"],
        horizons=settings["evaluation_horizons"],
    )
    target_implementation = implementation_digest("adaptation")
    jobs, memberships = {}, {name: [] for name in ANALYSES}

    def add(spec: dict) -> str:
        identity = dict(common, **spec)
        if spec["kind"] != "source":
            identity["implementation"] = target_implementation
        key = fingerprint(identity)
        if key not in jobs:
            jobs[key] = identity
        return key

    source_jobs = []
    for count in settings["source_counts"]:
        for seed in settings["seeds"]:
            order = np.random.default_rng(seed).permutation(sources).tolist()
            families = [("neural", 32, ("A", "AK"))]
            if count == len(sources):
                families += [
                    ("neural", 64, ("A", "AK")),
                    ("behavior", 32, ("A", "AK", "AKC")),
                ]
            for objective, latent, profiles in families:
                for sharing in profiles:
                    chosen = order[:count]
                    spec = dict(
                        kind="source",
                        objective=objective,
                        latent=latent,
                        sharing=sharing,
                        seed=seed,
                        sessions=chosen,
                        source_count=count,
                        sources={s: sessions[s] for s in chosen},
                        dependencies=[],
                    )
                    source_jobs.append((add(spec), spec))
    for seed in settings["seeds"]:
        for session in targets:
            for fold in settings["target_folds"]:
                for fraction in settings["target_fractions"]:
                    target = dict(
                        seed=seed,
                        session=session,
                        fold=fold,
                        fraction=fraction,
                        source=sessions[session],
                    )
                    prep = add(
                        dict(target, kind="preprocessing", dependencies=[])
                    )
                    for latent in (32, 64):
                        scratch = add(
                            dict(
                                target,
                                kind="scratch",
                                latent=latent,
                                objective="scratch",
                                sharing="scratch",
                                source_count=0,
                                preprocessing=prep,
                                dependencies=[prep],
                            )
                        )
                        memberships[CAPACITY].append(scratch)
                        if latent == 32:
                            memberships[PRIMARY].append(scratch)
                            memberships[SUPERVISION].append(scratch)
                    for source_id, source in source_jobs:
                        if source["seed"] != seed:
                            continue
                        spec = dict(
                            target,
                            kind="adaptation",
                            latent=source["latent"],
                            sharing=source["sharing"],
                            objective=source["objective"],
                            source_count=source["source_count"],
                            source_fit=source_id,
                            preprocessing=prep,
                            dependencies=[source_id, prep],
                        )
                        adaptation = add(spec)
                        if source["objective"] == "neural":
                            if source["latent"] == 32:
                                memberships[PRIMARY].append(adaptation)
                            if source["source_count"] == len(sources):
                                memberships[CAPACITY].append(adaptation)
                                if source["latent"] == 32:
                                    memberships[SUPERVISION].append(adaptation)
                        else:
                            memberships[SUPERVISION].append(adaptation)
    return dict(
        name=PRIMARY,
        storage=dict(FIT_STORES),
        settings=deepcopy(settings),
        jobs=jobs,
        analyses=memberships,
        held_out_sessions=targets,
    )
