"""Fit full no-preprocessing BRAID on four seen source recordings.

Inputs are a completed single-stage neural source fit and its guarded source
windows. Four independent session paths save behavioral RNN1 and neural
readout checkpoints; one dependent path saves a shared residual RNN2 fit.
Each path writes its component IDs, training FLOPs, and split inventory.
Outer-test windows are never used for fitting or checkpoint selection.
"""

import json
from pathlib import Path

import numpy as np

from BRAID.multisession import MixedForecaster
from experiments.multisession.behavior_supervision.models import (
    NEURAL_WIDTH, STAGE2_SHARING, rnn2_model,
)
from experiments.multisession.compute_scaling.ancestry import (
    main_prefix, restore_component,
)
from experiments.multisession.compute_scaling.branching import load_flops
from experiments.multisession.compute_scaling.self_supervised import (
    COMMON, LOCAL_DEVICE,
)
from experiments.multisession.compute_scaling.training import fit_segment
from experiments.multisession.data import session_windows, source_features
from experiments.multisession.plan import PRIMARY
from experiments.multisession.runner import dataset
from experiments.multisession.source import readout_data, residual_data
from experiments.multisession.store import (
    completed, dump_pickle, fit_directory, job, load_pickle,
)
from experiments.multisession.training import (
    ReadoutStage, recurrent_data, stage_predictions,
)
from experiments.storage.cache import atomic_json, fingerprint

PROTOCOL = "seen_full_braid_from_frozen_neural_core_v1"
MAIN_UPDATES = 640
TOTAL_UPDATES = 800
DECAY_UPDATES = TOTAL_UPDATES - MAIN_UPDATES
WARMUP_UPDATES = 10
WARMUP_START = 1e-7
LEARNING_RATE = 1e-3
HEAD_BATCH = 8
HEAD_CAP = 2000
HEAD_INTERVAL = 200
PREVIOUS_HEAD_INTERVAL = 800
HEAD_PATIENCE = 10
HEAD_WIDTH = 64
CORE_KEYS = ("A", "Afw", "Kfw")
TRANSFER_KEYS = (*CORE_KEYS, "K")


def component_root(plan: dict) -> Path:
    """Return the existing reusable source-component store."""
    return (
        Path(plan["settings"]["paths"]["artifact_root"])
        / PRIMARY / "source_components"
    )


def selected_backbone(plan: dict, key: str) -> tuple[dict, Path]:
    """Resolve the checksum-valid selected neural source component."""
    source = fit_directory(plan, key)
    if completed(source) is None:
        raise ValueError("Seen-session fitting needs a completed source fit.")
    selection = json.loads((source / "selection.json").read_text())
    paths = json.loads((source / "budgets.json").read_text())
    chosen = [
        row for row in paths
        if row["budget"] == selection["selected_budget"]
    ]
    if len(chosen) != 1:
        raise ValueError("Expected one selected neural source budget.")
    component = (
        component_root(plan)
        / chosen[0]["stages"]["neural"]["component_id"]
    )
    if completed(component) is None:
        raise ValueError("Selected neural source checkpoint is incomplete.")
    return chosen[0], component


def session_identity(plan: dict, key: str, index: int) -> dict:
    """Bind one seen-session full BRAID path to its neural checkpoint."""
    if index not in range(COMMON):
        raise ValueError(f"Expected common session index 0-3, got {index}.")
    source = fit_directory(plan, key)
    path, component = selected_backbone(plan, key)
    name = plan["jobs"][key]["sessions"][index]
    record = json.loads((source / "data.json").read_text())[name]
    return dict(
        protocol=PROTOCOL, source_fit=key,
        source_count=plan["jobs"][key]["source_count"],
        session=name, session_index=index,
        selected_budget=path["budget"],
        backbone_component=component.name,
        backbone_checksums=completed(component)["checksums"],
        data_identity=fingerprint(record),
        latent=plan["jobs"][key]["latent"],
        stage1_sharing=STAGE2_SHARING,
        frozen_core=CORE_KEYS,
        fine_tuned_private_map="K",
        decoder_width=HEAD_WIDTH,
        main_updates=MAIN_UPDATES,
        decay_updates=DECAY_UPDATES,
        readout_cap=HEAD_CAP,
        readout_batch=HEAD_BATCH,
        readout_interval=HEAD_INTERVAL,
        readout_patience=HEAD_PATIENCE,
        learning_rate=LEARNING_RATE,
        warmup_updates=WARMUP_UPDATES,
        warmup_start=WARMUP_START,
    )


def session_directory(
    plan: dict, key: str, index: int,
    readout_interval: int = HEAD_INTERVAL,
) -> Path:
    """Locate a seen-session path for one readout check interval."""
    identity = session_identity(plan, key, index)
    identity["readout_interval"] = readout_interval
    root = (
        Path(plan["settings"]["paths"]["artifact_root"])
        / PRIMARY / "source_session_paths" / key
    )
    return root / f"session{index}-{fingerprint(identity)}"


def session_windows_checked(
    plan: dict, key: str, index: int,
) -> tuple[dict, dict, dict]:
    """Load behavior only after pretraining and check source split identity."""
    spec = plan["jobs"][key]
    name = spec["sessions"][index]
    adapter = dataset(plan["settings"])
    adapter.mode = "off"
    features = source_features(
        adapter.load(name), spec["data"],
        spec["source_training"]["inner_validation_fraction"],
    )
    train, inner, _, record = session_windows(
        features, True, include_test=True,
    )
    saved = json.loads(
        (fit_directory(plan, key) / "data.json").read_text()
    )[name]
    for role in ("training", "inner_validation", "validation"):
        if record[role] != saved[role]:
            raise ValueError(f"Seen-session {role} windows changed: {name}.")
    for role in ("Y", "U"):
        if record["scales"][role] != saved["scales"][role]:
            raise ValueError(f"Seen-session {role} scale changed: {name}.")
    return train, inner, record


def recurrent_settings() -> dict:
    """Define 640 warm/constant steps and a 160-step decay branch."""
    return dict(
        learning_rate=LEARNING_RATE,
        warmup_updates=WARMUP_UPDATES,
        warmup_start_learning_rate=WARMUP_START,
        branch_updates=DECAY_UPDATES,
        session_batch_sizes={"1": HEAD_BATCH, "4": HEAD_BATCH},
    )


def readout_settings() -> dict:
    """Define the fixed-exposure neural readout protocol."""
    return dict(
        learning_rate=LEARNING_RATE,
        session_batch_sizes={"1": HEAD_BATCH},
        fixed_update_cap=HEAD_CAP,
        validation_exposure=HEAD_INTERVAL,
        fixed_patience=HEAD_PATIENCE,
    )


def initialized_rnn1(
    plan: dict, key: str, index: int,
    neural_width: int, behavior_width: int,
) -> tuple[MixedForecaster, dict]:
    """Copy core and private neural K; leave new behavior C initialized."""
    spec = plan["jobs"][key]
    _, component = selected_backbone(plan, key)
    from experiments.multisession.behavior_compute_scaling.n3_session import (
        selected_session_weights,
    )
    pretrained = selected_session_weights(
        load_pickle(component / "checkpoint.pkl")["weights"], index
    )
    model = MixedForecaster(
        [(neural_width, behavior_width)], spec["latent"],
        STAGE2_SHARING, spec["seed"] + 2 + index,
        decoder_width=HEAD_WIDTH,
        session_devices=LOCAL_DEVICE,
    )
    weights = model.weights()
    for name in TRANSFER_KEYS:
        target = f"0/{name}"
        original = pretrained[target]
        if any(
            left.shape != right.shape
            for left, right in zip(weights[target], original)
        ) or len(weights[target]) != len(original):
            raise ValueError(f"Incompatible pretrained {name} map.")
        weights[target] = original
    model.restore(weights)
    if not model.shared_variables:
        raise ValueError("Expected a shared pretrained recurrent core.")
    model.shared_variables = []
    model.frozen_modules = CORE_KEYS
    return model, {name: pretrained[f"0/{name}"] for name in CORE_KEYS}


def verify_frozen_core(model: MixedForecaster, core: dict) -> None:
    """Reject any behavioral fit that changed transferred core arrays."""
    for name, values in core.items():
        current = model.weights()[f"0/{name}"]
        for expected, actual in zip(values, current):
            np.testing.assert_array_equal(expected, actual)


def reused_behavior_branch(
    plan: dict, key: str, index: int,
    model: MixedForecaster, core: dict, record: dict,
) -> tuple[Path, int, str]:
    """Restore the completed RNN1 branch with identical scientific inputs.

    Parameters
    ----------
    plan : dict
        Neural source campaign plan.
    key : str
        Source fit identity.
    index : int
        Common recording index, from zero to three.
    model : MixedForecaster
        Initialized session-local behavioral model.
    core : dict
        Frozen pretrained core arrays indexed by module name.
    record : dict
        Verified training, inner-validation, and outer-test inventory.

    Returns
    -------
    tuple of Path, int, str
        Completed decay branch, full RNN1 training FLOPs, and previous
        session path name.
    """
    previous = session_directory(
        plan, key, index, PREVIOUS_HEAD_INTERVAL
    )
    if completed(previous) is None:
        raise ValueError(
            f"Missing completed RNN1 reuse path: {previous.resolve()}."
        )
    expected = session_identity(plan, key, index)
    expected["readout_interval"] = PREVIOUS_HEAD_INTERVAL
    saved = json.loads((previous / "identity.json").read_text())
    inventory = json.loads((previous / "data.json").read_text())
    if fingerprint(saved) != fingerprint(expected) or inventory != record:
        raise ValueError(
            f"RNN1 reuse inputs changed: {previous.resolve()}."
        )
    path = json.loads((previous / "path.json").read_text())
    if path["session"] != expected["session"] or (
        path["session_index"] != index
    ):
        raise ValueError("RNN1 reuse recording identity changed.")
    branch = component_root(plan) / path["stages"]["behavior_relevant"]
    prefix = main_prefix(branch)
    flops = load_flops(prefix) + load_flops(branch)
    if path["behavior_rnn1_flops"] != flops:
        raise ValueError("Saved RNN1 path FLOPs differ from its components.")
    restore_component(model, branch)
    verify_frozen_core(model, core)
    return branch, flops, previous.name


def fit_session(plan: dict, key: str, index: int) -> None:
    """Reuse behavioral RNN1 and fit one private neural readout."""
    identity = session_identity(plan, key, index)
    destination = session_directory(plan, key, index)
    with job(destination, identity) as needed:
        if not needed:
            return
        train, inner, record = session_windows_checked(
            plan, key, index
        )
        spec = plan["jobs"][key]
        width = train["Y"].shape[-1]
        model, core = initialized_rnn1(
            plan, key, index, width, train["Z"].shape[-1]
        )
        training = [recurrent_data(train, "Z")]
        validation = [recurrent_data(inner, "Z")]
        root = component_root(plan)
        branch, behavior_flops, previous = reused_behavior_branch(
            plan, key, index, model, core, record
        )
        train_states = stage_predictions(model, training)[1]
        inner_states = stage_predictions(model, validation)[1]
        readout_train = readout_data(
            [recurrent_data(train)], train_states
        )
        readout_inner = readout_data(
            [recurrent_data(inner)], inner_states
        )
        readout = ReadoutStage(
            [(spec["latent"], width)], NEURAL_WIDTH,
            spec["seed"] + 3 + index,
            session_devices=LOCAL_DEVICE,
        )
        fixed = readout_settings()
        readout_stage = dict(
            identity, stage="neural_readout",
            upstream=branch.name, settings=fixed,
        )
        readout_path, _ = fit_segment(
            readout, readout_train, root,
            readout_stage, fixed, spec["seed"] + 3 + index,
            0, HEAD_CAP, None, LOCAL_DEVICE, HEAD_BATCH,
            validation=readout_inner,
        )
        atomic_json(destination / "data.json", record)
        atomic_json(destination / "path.json", dict(
            session=identity["session"], session_index=index,
            stages=dict(
                behavior_relevant=branch.name,
                neural_readout=readout_path.name,
            ),
            behavior_rnn1_flops=behavior_flops,
            behavior_checkpoint_reused_from=previous,
            neural_readout_flops=load_flops(readout_path),
            readout_selected_update=json.loads((
                readout_path / "training.json"
            ).read_text())["selected_update"],
        ))


def assembled_identity(plan: dict, key: str, stage: str) -> dict:
    """Identify a four-session model assembled from private completions."""
    sessions = []
    for index in range(COMMON):
        path = session_directory(plan, key, index)
        marker = completed(path)
        if marker is None:
            raise ValueError("Cannot assemble incomplete seen-session path.")
        sessions.append(dict(path=path.name, checksums=marker["checksums"]))
    return dict(
        protocol=PROTOCOL, source_fit=key,
        stage=f"assembled_{stage}", sessions=sessions,
    )


def assembled_directory(plan: dict, key: str, stage: str) -> Path:
    """Locate a checksum-pinned assembled four-session component."""
    identity = assembled_identity(plan, key, stage)
    return component_root(plan) / fingerprint(identity)


def assemble_models(plan: dict, key: str) -> tuple:
    """Restore four private paths into full RNN1 and readout models."""
    spec = plan["jobs"][key]
    records = []
    paths = []
    for index in range(COMMON):
        directory = session_directory(plan, key, index)
        records.append(json.loads((directory / "data.json").read_text()))
        paths.append(json.loads((directory / "path.json").read_text()))
    dimensions = [
        (sum(row["scales"]["Y"]["keep"]),
         sum(row["scales"]["Z"]["keep"]))
        for row in records
    ]
    main = MixedForecaster(
        dimensions, spec["latent"], STAGE2_SHARING,
        spec["seed"] + 2, decoder_width=HEAD_WIDTH,
        session_devices=LOCAL_DEVICE,
    )
    readout = ReadoutStage(
        [(spec["latent"], neural) for neural, _ in dimensions],
        NEURAL_WIDTH, spec["seed"] + 3,
        session_devices=LOCAL_DEVICE,
    )
    weights, readouts = {}, {}
    for index, path in enumerate(paths):
        branch = component_root(plan) / path["stages"][
            "behavior_relevant"
        ]
        one = load_pickle(branch / "checkpoint.pkl")["weights"]
        for key_name, values in one.items():
            weights[f"{index}/{key_name.split('/', 1)[1]}"] = values
        fixed = component_root(plan) / path["stages"][
            "neural_readout"
        ]
        readouts[str(index)] = load_pickle(
            fixed / "checkpoint.pkl"
        )["weights"]["0"]
    for name in CORE_KEYS:
        first = weights[f"0/{name}"]
        for index in range(1, COMMON):
            for expected, actual in zip(
                first, weights[f"{index}/{name}"]
            ):
                np.testing.assert_array_equal(expected, actual)
    main.restore(weights)
    readout.restore(readouts)
    for stage, model in (
        ("behavior_relevant", main),
        ("neural_readout", readout),
    ):
        identity = assembled_identity(plan, key, stage)
        directory = assembled_directory(plan, key, stage)
        with job(directory, identity) as needed:
            if needed:
                dump_pickle(directory / "checkpoint.pkl", dict(
                    weights=model.weights()
                ))
    return main, readout, dimensions, records, paths


def full_identity(plan: dict, key: str) -> dict:
    """Bind shared RNN2 to all four seen-session path completions."""
    neural, component = selected_backbone(plan, key)
    return dict(
        protocol=PROTOCOL, source_fit=key,
        selected_budget=neural["budget"],
        backbone_checksums=completed(component)["checksums"],
        behavior=assembled_identity(plan, key, "behavior_relevant"),
        readout=assembled_identity(plan, key, "neural_readout"),
        rnn2_sharing=STAGE2_SHARING,
        main_updates=MAIN_UPDATES,
        decay_updates=DECAY_UPDATES,
        learning_rate=LEARNING_RATE,
        warmup_updates=WARMUP_UPDATES,
        warmup_start=WARMUP_START,
    )


def full_directory(plan: dict, key: str) -> Path:
    """Locate one full seen-session BRAID fit per selected backbone."""
    identity = full_identity(plan, key)
    root = (
        Path(plan["settings"]["paths"]["artifact_root"])
        / PRIMARY / "source_full_paths" / key
    )
    return root / f"seen-full-braid-{fingerprint(identity)}"


def fit_full(plan: dict, key: str) -> None:
    """Fit one shared RNN2 after all four private paths complete."""
    identity = full_identity(plan, key)
    destination = full_directory(plan, key)
    with job(destination, identity) as needed:
        if not needed:
            return
        main, readout, dimensions, records, sessions = (
            assemble_models(plan, key)
        )
        spec = plan["jobs"][key]
        adapter = dataset(plan["settings"])
        adapter.mode = "off"
        train, inner = [], []
        for index in range(COMMON):
            name = spec["sessions"][index]
            features = source_features(
                adapter.load(name), spec["data"],
                spec["source_training"]["inner_validation_fraction"],
            )
            training, validation, _, record = session_windows(
                features, True, include_test=True,
            )
            if record != records[index]:
                raise ValueError("Assembled seen-session split changed.")
            train.append(recurrent_data(training))
            inner.append(recurrent_data(validation))
        train_residual = residual_data(main, readout, train)
        inner_residual = residual_data(main, readout, inner)
        residual = rnn2_model(dimensions, spec, LOCAL_DEVICE)
        settings = recurrent_settings()
        stage = dict(
            identity, stage="residual", settings=settings,
        )
        root = component_root(plan)
        prefix, _ = fit_segment(
            residual, train_residual, root, stage, settings,
            spec["seed"] + 4, 0, MAIN_UPDATES, None,
            LOCAL_DEVICE, HEAD_BATCH,
        )
        branch, _ = fit_segment(
            residual, train_residual, root, stage, settings,
            spec["seed"] + 4,
            MAIN_UPDATES, TOTAL_UPDATES, prefix,
            LOCAL_DEVICE, HEAD_BATCH, branch=True,
        )
        source, _ = selected_backbone(plan, key)
        adaptation_flops = (
            sum(
                row["behavior_rnn1_flops"]
                + row["neural_readout_flops"]
                for row in sessions
            ) + load_flops(prefix) + load_flops(branch)
        )
        atomic_json(destination / "data.json", {
            spec["sessions"][index]: record
            for index, record in enumerate(records)
        })
        atomic_json(destination / "path.json", dict(
            stages=dict(
                behavior_relevant=dict(
                    component_id=assembled_directory(
                        plan, key, "behavior_relevant"
                    ).name,
                ),
                neural_readout=dict(
                    component_id=assembled_directory(
                        plan, key, "neural_readout"
                    ).name,
                ),
                residual=dict(component_id=branch.name),
            ),
            backbone_training_flops=source["full_path_flops"],
            behavior_adaptation_flops=adaptation_flops,
            full_path_flops=(
                source["full_path_flops"] + adaptation_flops
            ),
            inner_validation_windows=sum(
                len(row["T"]) for row in inner
            ),
        ))
