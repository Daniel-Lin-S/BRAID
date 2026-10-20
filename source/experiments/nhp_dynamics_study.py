"""Define the matched single-session NHP dynamics study.

Inputs are local NHP experiment settings and a campaign root. Outputs are
checksum-validated fit directories and matched score tables and figures.
The generic campaign runner calls this study for audit, fitting, and report.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Callable

for name in (
    "TF_USE_LEGACY_KERAS", "TF_NUM_INTEROP_THREADS",
    "TF_NUM_INTRAOP_THREADS", "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
):
    os.environ[name] = "1"
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import numpy as np

from experiments.campaign_status import FitJob, StatusPlan
from experiments.campaign_comparison import (
    ComparisonPlan, report as comparison_report,
)
from experiments.campaign_runner import (
    run_phase,
)
from experiments.execution.tensorflow_gpu import (
    initialize_worker, initializer_arguments,
)
from experiments.multisession.data import stable_seed, target_selection
from experiments.multisession.plan import implementation_digest
from experiments.storage.scientific_modules import fit_module_versions
from experiments.multisession.preprocessing import validate_predictions
from experiments.multisession.runner import dataset, resolve
from experiments.multisession.store import completed, job, load_pickle
from experiments.nhp_dynamics_design import (
    BIN_MS, CANDIDATE_CASES, CASES, CASE_LABELS,
    CONTROL_CASES, CPU_NICE, CURRENT_CONTROL_TARGET_DIGEST,
    DEFAULT_WORKERS, FOLDS, FRACTION, GPU_MEMORY_LIMIT_MIB,
    HORIZONS, IQR_MULTIPLIER, LATENT,
    LEGACY_CONTROL_TARGET_DIGEST, LEGACY_TRANSFORMER_DIGEST,
    METRICS, MLP_WIDTH, REFERENCE_CASE, REFERENCE_SEED,
    SEEDS, SESSIONS, TARGETS,
    fit_directory as _fit_directory,
    preprocessing_directory as _preprocessing_directory,
    target_key, tasks as _tasks,
)
from experiments.storage.cache import (
    atomic_json, file_digest, fingerprint,
)


# Reviewed numerical-equivalence signatures from this campaign's records.
PRIOR_TARGET_DIGESTS = frozenset({
    CURRENT_CONTROL_TARGET_DIGEST,
    "fe054ab10878e41f8c1654478697c5d2cf5e4ab460eb8a3bc258732a5266dab0",
})
PRIOR_TRANSFORMER_DIGEST = (
    "ce453765419b0a4d95ee39a6ff00bab91a18f2623b1ca0bd4d319c8dfa48efc6"
)


def _reusable_completed_fit(directory: Path, identity: dict) -> bool:
    """Accept a checksum-complete prior fit with identical science.

    Only reviewed implementation signatures from this campaign and its
    already validated controls can bridge the old raw-source identity.
    """
    record = completed(directory)
    if record is None:
        return False
    if record["identity_hash"] == fingerprint(identity):
        return True
    saved = json.loads((directory / "identity.json").read_text())
    if record["identity_hash"] != fingerprint(saved):
        raise ValueError(
            f"Completion identity checksum mismatch: {directory.resolve()}"
        )
    saved_impl = saved.pop("implementation", None)
    current = deepcopy(identity)
    current.pop("implementation", None)
    if saved != current:
        raise ValueError(
            f"Completed fit has different scientific parameters: "
            f"{directory.resolve()}"
        )
    if not isinstance(saved_impl, dict):
        raise ValueError(
            f"Completed fit lacks implementation: {directory.resolve()}"
        )
    prior = (saved_impl.get("target"), saved_impl.get("transformer"))
    reviewed = {
        (target, PRIOR_TRANSFORMER_DIGEST)
        for target in PRIOR_TARGET_DIGESTS
    }
    if identity["case"] in CONTROL_CASES:
        reviewed.add((
            LEGACY_CONTROL_TARGET_DIGEST, LEGACY_TRANSFORMER_DIGEST
        ))
    if prior not in reviewed:
        raise ValueError(
            f"Unreviewed completed fit implementation {prior}: "
            f"{directory.resolve()}"
        )
    return True


def case_configuration(base: dict, case: str) -> dict:
    """Return a native model configuration for one dynamics case."""
    if case not in CASES:
        raise ValueError(f"Unknown dynamics case: {case}.")
    result = deepcopy(base)
    for name in ("stage_1", "stage_2"):
        stage = result["model"][name]
        transition = stage["state_transition"]
        if case == "lstm":
            stage["state_transition"] = {"architecture": "lstm"}
        elif case == "mlp":
            transition.update(
                architecture="multilayer_perceptron",
                hidden_size=MLP_WIDTH,
                depth=1,
                activation="relu",
                output_activation="linear",
                use_bias=True,
            )
        elif case == "transformer_both" or (
            case == "transformer_stage1" and name == "stage_1"
        ):
            stage["state_transition"] = dict(
                architecture="causal_transformer",
                include_stage_one_state=True,
            )
        else:
            transition.update(architecture="multilayer_perceptron")
    result["forecast"]["use_observed_input_in_dynamics"] = True
    return result


def _settings(experiment: Path, root: Path) -> dict:
    """Load local settings with a writable campaign-local data cache."""
    settings = resolve(experiment)
    source = Path(settings["paths"]["cache_root"])
    cache = (root / "cache").resolve()
    for namespace in ("session", "fold"):
        source_group = source / namespace
        destination = cache / namespace
        destination.mkdir(parents=True, exist_ok=True)
        if not source_group.is_dir():
            continue
        for entry in source_group.iterdir():
            if not (entry.is_dir() and (entry / "manifest.json").is_file()):
                continue
            link = destination / entry.name
            if not link.exists():
                try:
                    link.symlink_to(
                        entry.resolve(), target_is_directory=True
                    )
                except FileExistsError:
                    if link.resolve() != entry.resolve():
                        raise
    settings["paths"]["cache_root"] = str(cache)
    return settings


def log_directory(experiment: Path, root: Path) -> Path:
    """Return the configured log directory for this comparison."""
    settings = resolve(experiment)
    log_root = Path(settings["paths"]["log_root"]).resolve()
    return log_root / root.name


def _selected(settings: dict, session: str, fold: int, seed: int):
    """Load one guarded target fold and deterministic 80-percent selection."""
    from experiments.multisession.target import training_arrays

    adapter = dataset(settings)
    original = adapter.load(session)
    features = adapter.fold(original, fold, True)
    training, validation, selection = target_selection(
        features, len(original.arrays["t"]), FRACTION, seed
    )
    arrays = training_arrays(features, training, validation)
    indices = {
        "training": features.arrays["indices"][training],
        "validation": features.arrays["indices"][validation],
    }
    return features, arrays, indices, selection


def _existing_scratch(
    settings: dict, session: str, fold: int, indices: dict,
    arguments: dict,
) -> tuple[Path, Path]:
    """Validate the completed seed-42 scratch fit and its preprocessing."""
    artifact = Path(settings["paths"]["artifact_root"])
    root = artifact / "experiments" / "BRAID_matched_scratch"
    matches = []
    for identity_path in root.glob("*/identity.json"):
        identity = json.loads(identity_path.read_text())
        if (
            identity.get("session") == session
            and identity.get("fold") == fold
            and identity.get("seed") == 42
            and identity.get("fraction") == FRACTION
            and identity.get("latent") == LATENT
            and identity.get("kind") == "scratch"
        ):
            matches.append((identity_path.parent, identity))
    if len(matches) != 1:
        raise ValueError(
            f"Expected one completed scratch fit for {session}, "
            f"fold {fold}, seed 42; got {len(matches)}."
        )
    directory, identity = matches[0]
    if completed(directory) is None:
        raise ValueError(f"Scratch fit is incomplete: {directory.resolve()}")
    with np.load(directory / "selection.npz") as saved:
        for name, selected in indices.items():
            np.testing.assert_array_equal(saved[name], selected)
    actual = json.loads((directory / "arguments.json").read_text())
    expected = deepcopy(arguments)
    for value in (actual["args_base"], expected["args_base"]):
        value.pop("stage_1_transformer_arguments", None)
        value.pop("stage_2_transformer_arguments", None)
    if actual != expected:
        raise ValueError(
            f"Scratch training arguments differ: {directory.resolve()}"
        )
    dependencies = identity.get("dependencies", [])
    if len(dependencies) != 1:
        raise ValueError(
            f"Expected one preprocessing dependency, got {dependencies}."
        )
    preprocessing = (
        artifact / "experiments" / "BRAID_matched_preprocessing"
        / dependencies[0]
    )
    if completed(preprocessing) is None:
        raise ValueError(
            f"Preprocessing is incomplete: {preprocessing.resolve()}"
        )
    return directory, preprocessing


def _prepare_one(
    experiment: str, root_name: str, session: str,
    fold: int, seed: int,
) -> str:
    """Create or reuse one checksum-validated preprocessing checkpoint."""
    from experiments.multisession.target import (
        fit_arguments, fit_preprocessing,
    )

    os.nice(CPU_NICE)
    root = Path(root_name)
    settings = _settings(Path(experiment), root)
    _, arrays, indices, selection = _selected(
        settings, session, fold, seed
    )
    key = target_key(session, fold, seed)
    directory = _preprocessing_directory(root, key)
    arguments = fit_arguments(
        case_configuration(settings["model"], "linear"),
        selection["batch_size"], LATENT,
        settings["target_training"]["maximum_epochs"],
    )
    source = None
    if seed == REFERENCE_SEED:
        _, source = _existing_scratch(
            settings, session, fold, indices, arguments
        )
    identity = dict(
        session=session, fold=fold, seed=seed,
        fraction=FRACTION, selection=selection,
        arguments=arguments,
        source_checkpoint=(
            file_digest(source / "model.p") if source else None
        ),
    )
    with job(directory, identity) as needed:
        if needed:
            np.savez_compressed(directory / "selection.npz", **indices)
            atomic_json(directory / "selection.json", selection)
            if source is not None:
                shutil.copyfile(source / "model.p", directory / "model.p")
                atomic_json(
                    directory / "source.json",
                    dict(checksum=file_digest(source / "model.p")),
                )
            else:
                prep_seed = stable_seed(
                    session, fold, seed, "preprocessing"
                )
                fit_preprocessing(
                    arrays, arguments, directory, prep_seed
                )
    return str(directory.resolve())


def _fit_one(
    experiment: str, root_name: str, session: str,
    fold: int, seed: int, case: str,
) -> str:
    """Fit, evaluate, and commit one independent scratch model."""
    from experiments.multisession.target import (
        evaluate_target, fit_arguments, fit_target,
    )

    os.nice(CPU_NICE)
    root = Path(root_name)
    settings = _settings(Path(experiment), root)
    features, arrays, indices, selection = _selected(
        settings, session, fold, seed
    )
    configuration = case_configuration(settings["model"], case)
    arguments = fit_arguments(
        configuration, selection["batch_size"], LATENT,
        settings["target_training"]["maximum_epochs"],
    )
    if case == REFERENCE_CASE and seed == REFERENCE_SEED:
        directory, _ = _existing_scratch(
            settings, session, fold, indices, arguments
        )
        return str(directory.resolve())
    key = target_key(session, fold, seed)
    preprocessing = _preprocessing_directory(root, key)
    if completed(preprocessing) is None:
        raise RuntimeError(
            f"Preprocessing is incomplete: {preprocessing.resolve()}"
        )
    directory = _fit_directory(root, case, key)
    identity = dict(
        session=session, fold=fold, seed=seed, case=case,
        fraction=FRACTION, latent=LATENT,
        selection=selection, arguments=arguments,
        preprocessing_checksum=file_digest(
            preprocessing / "model.p"
        ),
        implementation=dict(
            target=implementation_digest("adaptation", configuration),
            modules=fit_module_versions("adaptation", configuration),
        ),
    )
    if _reusable_completed_fit(directory, identity):
        return str(directory.resolve())
    with job(directory, identity) as needed:
        if needed:
            np.savez_compressed(directory / "selection.npz", **indices)
            atomic_json(directory / "selection.json", selection)
            atomic_json(directory / "arguments.json", arguments)
            from experiments.multisession.store import (
                quarantine_incomplete_components,
            )
            quarantine_incomplete_components(directory)
            fit_seed = stable_seed(session, fold, seed, "main")
            model = fit_target(
                arrays, arguments, directory, preprocessing, {},
                fit_seed, identity["implementation"]["target"],
            )
            evaluation = dict(
                session=session, fold=fold, kind=case,
            )
            evaluate_target(
                model, features, directory, evaluation,
                list(HORIZONS),
            )
    return str(directory.resolve())


def _run_gpu_phase(
    function: Callable[..., str], arguments: list[tuple],
    workers: int,
    gpu_ids: tuple[str, ...],
) -> list[tuple]:
    """Run this experiment's jobs with bounded TensorFlow GPU workers."""
    initargs = initializer_arguments(
        gpu_ids, GPU_MEMORY_LIMIT_MIB,
        os.environ.get("BRAID_CAMPAIGN_SOURCE_SHA256"),
    )
    return run_phase(
        function, arguments, workers, initialize_worker, initargs
    )


def run_jobs(
    experiment: Path, root: Path, workers: int,
    gpu_ids: tuple[str, ...],
) -> None:
    """Complete preprocessing before parallel independent model fits."""
    if workers < 1:
        raise ValueError("Workers must be positive.")
    if not gpu_ids:
        raise ValueError("At least one GPU ID is required for campaign fits.")
    if workers > len(gpu_ids):
        raise ValueError("Workers must not exceed assigned GPU count.")
    root.mkdir(parents=True, exist_ok=True)
    common = (str(experiment.resolve()), str(root.resolve()))
    preparations = [(*common, *task) for task in _tasks()]
    prep_failures = _run_gpu_phase(
        _prepare_one, preparations, workers, gpu_ids
    )
    ready = [
        task for task in _tasks()
        if completed(
            _preprocessing_directory(root, target_key(*task))
        ) is not None
    ]
    fits = [
        (*common, *task, case)
        for task in ready for case in CASES
    ]
    fit_failures = _run_gpu_phase(
        _fit_one, fits, workers, gpu_ids
    )
    if prep_failures or fit_failures:
        raise RuntimeError(
            f"Campaign has {len(prep_failures)} preprocessing and "
            f"{len(fit_failures)} fit failures; inspect launch.json "
            "for the campaign log."
        )
    report(experiment, root)


def audit(experiment: Path, root: Path) -> None:
    """Run a nonsaving single-target Transformer optimization smoke test."""
    with tempfile.TemporaryDirectory(
        prefix="braid-transformer-audit-"
    ) as scratch:
        settings = _settings(experiment, Path(scratch))
        _audit_selected(settings)
        _audit_io_shapes(settings)


def _audit_io_shapes(settings: dict) -> None:
    """Exercise every production batch/width pair and all six horizons.

    Parameters
    ----------
    settings : dict
        Local campaign settings with a temporary data cache.
    """
    from BRAID.causal_transformer import (
        CausalTransformerCell, STIMULUS_DIMS, WINDOW_BINS,
    )
    from experiments.models.braid_backend import MISSING_MARKER

    checked = set()
    for session, fold, seed in _tasks():
        _, arrays, _, selection = _selected(
            settings, session, fold, seed
        )
        batch = selection["batch_size"]
        neural_dims = arrays["Y"].shape[0]
        signature = (batch, neural_dims)
        if signature in checked:
            continue
        if arrays["U"].shape[0] != STIMULUS_DIMS:
            raise ValueError(
                f"Audit expects {STIMULUS_DIMS} target coordinates, "
                f"got {arrays['U'].shape[0]}."
            )
        required = batch * WINDOW_BINS
        if min(arrays[name].shape[1] for name in ("Y", "U")) < required:
            raise ValueError(
                f"Audit needs {required} samples for {signature}."
            )
        neural = arrays["Y"][:, :required].T.astype(
            "float32"
        ).reshape(batch, WINDOW_BINS, neural_dims)
        target = arrays["U"][:, :required].T.astype(
            "float32"
        ).reshape(batch, WINDOW_BINS, STIMULUS_DIMS)
        for stage_one_dims in (0, LATENT):
            stage_one = np.zeros(
                (batch, WINDOW_BINS, stage_one_dims),
                dtype="float32",
            )
            observed = np.concatenate(
                (stage_one, neural, target), axis=-1
            )
            future_bin = np.concatenate(
                (stage_one, target), axis=-1
            )
            future = np.repeat(
                future_bin[..., None], max(HORIZONS), axis=-1
            )
            prior = (
                np.zeros(
                    (batch, WINDOW_BINS, neural_dims,
                     len(HORIZONS)),
                    dtype="float32",
                )
                if stage_one_dims else None
            )
            cell = CausalTransformerCell(
                units=LATENT,
                ny_out=(
                    neural_dims if stage_one_dims else STIMULUS_DIMS
                ),
                neural_dims=neural_dims,
                stage_one_dims=stage_one_dims,
                nft=0,
                n1_in=stage_one_dims + STIMULUS_DIMS,
                steps_ahead=HORIZONS,
                CSettings={}, KSettings={},
                missing_marker=MISSING_MARKER,
                has_prior_pred=bool(stage_one_dims),
                observable_U_in_Cfw=False,
                sequence_bins=WINDOW_BINS,
            )
            inputs = (
                (observed, future, prior)
                if prior is not None else (observed, future)
            )
            outputs = cell.forward_sequence(inputs, training=False)
            expected_shape = (batch, WINDOW_BINS)
            if any(
                output.shape[:2] != expected_shape
                or not np.isfinite(output.numpy()).all()
                for output in outputs
            ):
                raise FloatingPointError(
                    f"Nonfinite or mismatched audit output for "
                    f"batch {batch}, neural width {neural_dims}, "
                    f"stage-one width {stage_one_dims}."
                )
        checked.add(signature)
    if not checked:
        raise ValueError("Audit found no selected input shapes.")
    print(f"Nonsaving shape audit covered {sorted(checked)}.", flush=True)


def _audit_selected(settings: dict) -> None:
    """Fit one full-batch update and exercise production prediction I/O."""
    from experiments.datasets.nhp import TEST
    from experiments.datasets.windows import window_indices
    from experiments.models.braid_backend import BRAIDBackend
    from experiments.multisession.data import LENGTH
    from experiments.multisession.target import fit_arguments

    session, fold, seed = SESSIONS[0], FOLDS[0], SEEDS[0]
    features, arrays, indices, selection = _selected(
        settings, session, fold, seed
    )
    configuration = case_configuration(
        settings["model"], "transformer_both"
    )
    arguments = fit_arguments(
        configuration, selection["batch_size"], LATENT, 1
    )
    linear_arguments = fit_arguments(
        case_configuration(settings["model"], "linear"),
        selection["batch_size"], LATENT,
        settings["target_training"]["maximum_epochs"],
    )
    _, preprocessing = _existing_scratch(
        settings, session, fold, indices, linear_arguments
    )
    from BRAID.BRAIDModel import BRAIDModel
    from BRAID.BRAIDModel import runPredict
    from experiments.models.braid_backend import MISSING_MARKER
    import tensorflow as tf

    pre = load_pickle(preprocessing / "model.p")["model"]
    pre.restoreModels()
    print("Audit preprocessing restored.", flush=True)
    base = arguments["args_base"]
    required = base["sequence_length"] * (
        selection["batch_size"]
        + math.ceil(
            max(base["steps_ahead"]) / base["sequence_length"]
        )
    )
    small = {}
    for name in ("Y", "Z", "U"):
        for suffix in ("", "_validation"):
            values = arrays[name + suffix]
            minimum = (
                selection["batch_size"] * base["sequence_length"]
                if suffix else required
            )
            if values.shape[1] < minimum:
                raise ValueError(
                    f"Audit needs {minimum} {name}{suffix} bins, "
                    f"got {values.shape[1]}."
                )
            small[name + suffix] = values[:, :minimum]
    predicted = []
    for suffix in ("", "_validation"):
        output = runPredict(
            pre, Y=small["Y" + suffix].T,
            Z=small["Z" + suffix].T,
            U=small["U" + suffix].T,
            YType="cont", ZType="cont", useXFilt=False,
            missing_marker=MISSING_MARKER,
        )[0]
        predicted.append(output[0] if isinstance(output, list) else output)
    validate_predictions(
        {"training": predicted[0], "validation": predicted[1]},
        small,
    )
    tf.keras.utils.set_random_seed(
        stable_seed(session, fold, seed, "main")
    )
    arguments["args_base"]["epoch_artifacts"] = False
    model = BRAIDModel(
        log_dir="", missing_marker=MISSING_MARKER,
        preprocessing_model=pre,
        preprocessing_predictions=tuple(predicted),
    )
    print("Audit stage fitting started.", flush=True)
    model.fit(
        **small, YType="cont", ZType="cont", UType="cont",
        **arguments,
    )
    if model.hasBlownUp():
        raise FloatingPointError("Transformer audit produced nonfinite fit.")
    from BRAID.causal_transformer import CausalTransformerCell

    fitted = getattr(model, "sId", None)
    if fitted is None:
        raise RuntimeError("Audit found no fitted BRAID stages.")
    for name in ("model1", "model2"):
        stage = getattr(fitted, name, None)
        if stage is None or not isinstance(
            stage.rnn.cell, CausalTransformerCell
        ):
            raise RuntimeError(
                f"Audit expected Transformer in {name}."
            )
        if not stage.model.to_json():
            raise RuntimeError(
                f"Audit could not serialize {name}."
            )
        with tempfile.TemporaryDirectory(
            prefix="braid-transformer-save-audit-"
        ) as scratch:
            stage.model.save(Path(scratch) / f"{name}.h5")
    test = window_indices(features, TEST, LENGTH).ravel()
    backend = object.__new__(BRAIDBackend)
    backend.model, backend.length = model, LENGTH
    predicted = backend.predict(
        features.arrays["Y"][test],
        features.arrays["U"][test],
        list(HORIZONS),
    )
    if predicted["valid"].shape != (len(HORIZONS), len(test)):
        raise ValueError(
            "Audit prediction validity shape differs from the "
            "production evaluation shape."
        )
    print("Nonsaving full-I/O Transformer audit completed.", flush=True)


def report(experiment: Path, root: Path) -> None:
    """Report the selected NHP dynamics cases using shared comparisons."""
    from experiments.multisession.target import fit_arguments

    settings = None

    def fit_path(
        session: str, fold: int, seed: int, case: str,
    ) -> Path:
        """Resolve one validated local or scratch fit directory."""
        nonlocal settings
        if case != REFERENCE_CASE or seed != REFERENCE_SEED:
            return _fit_directory(
                root, case, target_key(session, fold, seed)
            )
        if settings is None:
            settings = _settings(experiment, root)
        _, _, indices, selection = _selected(
            settings, session, fold, seed
        )
        arguments = fit_arguments(
            case_configuration(settings["model"], case),
            selection["batch_size"], LATENT,
            settings["target_training"]["maximum_epochs"],
        )
        directory, _ = _existing_scratch(
            settings, session, fold, indices, arguments
        )
        return directory

    plan = ComparisonPlan(
        tasks=tuple(_tasks()), sessions=SESSIONS,
        cases=CASES, case_labels=CASE_LABELS,
        candidate_cases=CANDIDATE_CASES,
        control_cases=CONTROL_CASES,
        candidate_column="transformer", targets=TARGETS,
        horizons=HORIZONS, metrics=METRICS,
        smaller_is_better=("mse",), bin_ms=BIN_MS,
        iqr_multiplier=IQR_MULTIPLIER,
    )
    comparison_report(root, plan, fit_path)



def verified_reused_keys(experiment: Path) -> list[str]:
    """Validate the twelve completed seed-42 reference fits.

    Parameters
    ----------
    experiment : Path
        Local experiment YAML with an artifact root.

    Returns
    -------
    list[str]
        Sorted target keys whose source fits are checksum complete.
    """
    settings = resolve(experiment)
    artifact = Path(settings["paths"]["artifact_root"])
    reference = artifact / "experiments" / "BRAID_matched_scratch"
    expected = {
        target_key(session, fold, REFERENCE_SEED)
        for session in SESSIONS for fold in FOLDS
    }
    found = set()
    for identity_path in reference.glob("*/identity.json"):
        identity = json.loads(identity_path.read_text())
        if (
            identity.get("kind") != "scratch"
            or identity.get("seed") != REFERENCE_SEED
            or identity.get("fraction") != FRACTION
            or identity.get("latent") != LATENT
        ):
            continue
        key = target_key(
            identity["session"], identity["fold"], REFERENCE_SEED
        )
        if key not in expected:
            continue
        if key in found:
            raise ValueError(f"Duplicate reference fit for {key}.")
        if completed(identity_path.parent) is None:
            raise ValueError(
                f"Incomplete reference fit: "
                f"{identity_path.parent.resolve()}"
            )
        found.add(key)
    if found != expected:
        missing = sorted(expected - found)
        raise ValueError(
            f"Expected {len(expected)} completed reference fits, "
            f"found {len(found)}; missing {missing}."
        )
    return sorted(found)

def status_plan(root: Path, reused_keys: set[str] | None) -> StatusPlan:
    """Map this comparison's jobs into a model-independent status plan."""
    preprocessing = []
    fits = []
    for session, fold, seed in _tasks():
        key = target_key(session, fold, seed)
        index = len(preprocessing)
        preprocessing.append(_preprocessing_directory(root, key))
        for case in CASES:
            if case == REFERENCE_CASE and seed == REFERENCE_SEED:
                fits.append(FitJob(
                    directory=None, preprocessing_index=index,
                    verified_reuse=(
                        reused_keys is not None and key in reused_keys
                    ),
                    reused=True,
                ))
            else:
                fits.append(FitJob(_fit_directory(root, case, key)))
    return StatusPlan(
        preprocessing=tuple(preprocessing), fits=tuple(fits),
        report_marker=root / "report_complete.json",
        coordinator_module="experiments.campaign_runner",
        component_root=root / "target_components",
    )
