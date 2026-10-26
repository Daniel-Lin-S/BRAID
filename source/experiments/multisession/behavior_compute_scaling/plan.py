"""Build the behavioral source compute-scaling campaign.

Inputs are resolved settings, current session fingerprints, and immutable
source/target reference plans. The output plan contains four source fits,
matched n1=n2=16 target fits, and reusable preprocessing dependencies.
"""

from copy import deepcopy
from pathlib import Path

from experiments.multisession.plan import implementation_digest
from experiments.multisession.store import completed, fit_directory
from experiments.storage.cache import file_digest, fingerprint

CAMPAIGN = "multi_session_behavior_compute_scaling"
COUNTS = (4, 8, 16, 33)
FOLDS = (0, 2, 4)
FRACTIONS = (0.05, 0.20, 0.80)
LATENT = 16
SEED = 42
SHARING = "A1_fw_RNN2"
STAGE1_SHARING = "A"
STAGE2_SHARING = "A2_A2_fw_K2_fw"
COMPONENTS = (
    "preprocessing_neural",
    "preprocessing_behavior",
    "behavior_relevant",
    "neural_readout",
    "residual",
)
CHECKPOINTS = (1000, 2000, 3000, 4000, 5000)
BRANCH_UPDATES = 500
WARMUP_UPDATES = 200
WARMUP_START = 1e-7
LEARNING_RATE = 1e-3
LOCAL_WINDOWS = {4: 8, 8: 8, 16: 8, 33: 4}


def source_implementation() -> str:
    """Hash the scientific modules used by new source fits."""
    root = Path(__file__).resolve().parents[3]
    paths = (
        Path(__file__),
        root / "BRAID" / "multisession.py",
        root / "BRAID" / "RegressionModel.py",
        root / "experiments" / "multisession" / "source.py",
        root / "experiments" / "multisession" / "training.py",
        root / "experiments" / "multisession" / "data.py",
        root / "experiments" / "datasets" / "nhp.py",
        root / "experiments" / "datasets" / "windows.py",
        root / "experiments" / "multisession"
        / "behavior_compute_scaling" / "training.py",
        root / "experiments" / "multisession"
        / "behavior_compute_scaling" / "pipeline.py",
        root / "experiments" / "multisession"
        / "behavior_compute_scaling" / "flops.py",
    )
    return fingerprint({str(p.relative_to(root)): file_digest(p)
                        for p in paths})


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
        branch_updates=BRANCH_UPDATES,
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
        "maximum_updates", "checkpoints", "branch_updates",
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
