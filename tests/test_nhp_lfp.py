"""Validate LFP ingestion, extensible preprocessing and dedicated previews."""

import copy
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml
from experiments.analysis import analysis_settings
from experiments.artifacts import model_directory, model_settings
from experiments.cache import fingerprint
from experiments.configuration import (
    CONFIGURATION,
    argument_parser,
    read_yaml,
    resolve_configuration,
)
from experiments.fitting import prediction_directory
from experiments.lfp_preprocessing import (
    ArraySource,
    PipelineContext,
    preprocessing_identity,
    preprocessing_steps,
    run_preprocessing,
)
from experiments.nhp_lfp import NHPLFPDataset, _repair_timestamps
from experiments.presentation import presentation
from experiments.preview_publication import publish_previews
from experiments.previews import preprocessing_previews, preview_windows

CHANNEL_COUNT = 96
SOURCE_RATE_HZ = 100
TASK_RATE_HZ = 250
TARGET_RATE_HZ = 20
SESSION = "indy_20160624_03"


class MemorySource:
    """Provide finite in-memory time-first values to pipeline tests."""

    def __init__(
        self,
        values: np.ndarray,
        timestamps: np.ndarray,
        channel_ids: list[str],
    ) -> None:
        self.values = values
        self.timestamps = timestamps
        self.channel_ids = channel_ids
        self.shape = values.shape

    def read(self, rows: slice, columns: np.ndarray) -> np.ndarray:
        """Read a selected signal block."""
        return self.values[rows][:, columns]


class OffsetSource:
    """Add an offset lazily while preserving source addressing."""

    def __init__(self, source: MemorySource, offset: float) -> None:
        self.source = source
        self.offset = offset
        self.timestamps = source.timestamps
        self.channel_ids = source.channel_ids
        self.shape = source.shape

    def read(self, rows: slice, columns: np.ndarray) -> np.ndarray:
        """Read a selected signal block with its configured offset."""
        return self.source.read(rows, columns) + self.offset


class OffsetStep:
    """Test-only native-domain plugin proving external extensibility."""

    name = "offset"
    input_domain = "native"
    output_domain = "native"

    def __init__(self, parameters: dict) -> None:
        if set(parameters) != {"value"}:
            raise ValueError("OffsetStep requires only value.")
        self.parameters = {"value": float(parameters["value"])}

    def apply(
        self, source: MemorySource, context: PipelineContext
    ) -> OffsetSource:
        """Return a lazy offset signal source."""
        return OffsetSource(source, self.parameters["value"])

    def provenance(self) -> dict:
        """Return the resolved test-step settings."""
        return {"name": self.name, "parameters": self.parameters}


class NonfiniteAlignedStep:
    """Test-only aligned plugin returning an invalid numerical payload."""

    name = "nonfinite_aligned"
    input_domain = "native"
    output_domain = "aligned"

    def __init__(self, parameters: dict) -> None:
        if parameters:
            raise ValueError("NonfiniteAlignedStep takes no parameters.")
        self.parameters = {}

    def apply(
        self, source: MemorySource, context: PipelineContext
    ) -> ArraySource:
        """Return NaNs to prove the pipeline validates plugin output."""
        values = np.full(
            (len(context.target_timestamps), source.shape[1]),
            np.nan,
        )
        return ArraySource(
            values,
            context.target_timestamps.copy(),
            source.channel_ids,
            {"test": True},
        )

    def provenance(self) -> dict:
        """Return this test step's empty parameter mapping."""
        return {"name": self.name, "parameters": self.parameters}


def resampling_step() -> dict:
    """Return the tracked polyphase step configuration."""
    return {
        "plugin": "experiments.lfp_preprocessing:PolyphaseResample",
        "parameters": {
            "anti_alias_lowpass_hz": 10,
            "window": "kaiser",
            "beta": 5.0,
        },
    }


def pipeline(*steps: dict) -> dict:
    """Build an ordered preprocessing mapping."""
    return {"steps": list(steps)}


def create_dataset(root: Path) -> None:
    """Create one compact paired task/NWB recording for integration tests."""
    (root / "raw").mkdir()
    (root / "record.json").write_text(json.dumps({"id": 3854034}))
    task_t = 105 + np.arange(100 * TASK_RATE_HZ) / TASK_RATE_HZ
    position = np.column_stack([np.sin(task_t / 5), np.cos(task_t / 5)])
    target = np.column_stack(
        [np.floor(position[:, 0] * 2), np.floor(position[:, 1] * 2)]
    )
    with h5py.File(root / f"{SESSION}.mat", "w") as data:
        data.create_dataset("t", data=task_t[None, :])
        data.create_dataset("cursor_pos", data=position.T)
        data.create_dataset("target_pos", data=target.T)
    raw_t = 100 + np.arange(110 * SOURCE_RATE_HZ) / SOURCE_RATE_HZ
    low = np.sin(2 * np.pi * 2 * raw_t)
    high = np.sin(2 * np.pi * 30 * raw_t)
    channels = np.column_stack(
        [1000 * low + 500 * high + number for number in range(CHANNEL_COUNT)]
    ).astype(np.int16)
    timestamps = raw_t.copy()
    timestamps[5000] = 0
    names = np.asarray([f"M1 {number:03d}".encode() for number in range(1, 97)])
    with h5py.File(root / "raw" / f"{SESSION}.nwb", "w") as data:
        group = data.create_group("acquisition/timeseries/broadband")
        signal = group.create_dataset("data", data=channels)
        signal.attrs["conversion"] = 1e-4
        signal.attrs["unit"] = np.bytes_("volt")
        group.create_dataset("timestamps", data=timestamps)
        group.create_dataset("electrode_names", data=names)
        group.create_dataset(
            "electrode_idx", data=np.arange(CHANNEL_COUNT, dtype=np.int32)
        )
        sync = group.create_group("sync")
        sync.create_dataset("ticks", data=np.arange(len(raw_t)))


def preview_settings(dataset: NHPLFPDataset) -> dict:
    """Resolve LFP preview settings from plotting and data modules."""
    settings = copy.deepcopy(
        read_yaml(CONFIGURATION / "plotting" / "style.yaml")["previews"]
    )
    settings.pop("enabled")
    settings.pop("selection")
    settings["adapter"] = dataset.settings["preview_adapter"]
    settings["presentation"] = presentation()
    settings["context_samples"] = 128
    return settings


def dataset_settings(root: Path, cache: Path) -> dict:
    """Resolve tracked LFP defaults against test-owned paths."""
    settings = read_yaml(CONFIGURATION / "data" / "lfp.yaml")
    settings.update(root=str(root), cache_root=str(cache), cache_mode="off")
    return settings


@pytest.fixture(scope="module")
def lfp_session(tmp_path_factory):
    """Build and preprocess one synthetic 96-channel LFP session."""
    root = tmp_path_factory.mktemp("lfp-data")
    create_dataset(root)
    settings = dataset_settings(
        root,
        tmp_path_factory.mktemp("lfp-cache"),
    )
    settings["cache_mode"] = "reuse"
    dataset = NHPLFPDataset(settings)
    session = dataset.load(SESSION)
    return dataset, session


def test_lfp_alignment_conversion_and_clock_repair(lfp_session):
    """Produce finite 20-Hz volts with source and timing provenance."""
    dataset, session = lfp_session
    arrays = session.arrays
    assert dataset.sessions() == [SESSION]
    assert dataset.inventory()[SESSION] == arrays["ids"].tolist()
    assert arrays["Y"].shape[1] == CHANNEL_COUNT
    assert arrays["Y"].dtype == np.float32
    assert arrays["Z"].shape == arrays["U"].shape
    assert len(arrays["Y"]) == len(arrays["Z"]) == len(arrays["t"])
    np.testing.assert_allclose(np.diff(arrays["t"]), 0.05, atol=1e-8)
    assert session.metadata["neural_modality"] == "lfp"
    assert session.metadata["broadband_source"][
        "repaired_timestamp_indices"
    ] == [5000]
    assert session.metadata["resampling"]["library"] == (
        "scipy.signal.resample_poly"
    )
    assert session.path is not None
    assert session.path.parent.name == "lfp_session"
    assert dataset.load(SESSION).path == session.path


def test_pipeline_order_parameters_car_and_extension():
    """Execute ordered plugins and keep each resolved parameter set."""
    timestamps = 10 + np.arange(1000) / SOURCE_RATE_HZ
    common = np.sin(2 * np.pi * 2 * timestamps)
    values = np.column_stack([common + 1, common + 2, common + 4])
    source = MemorySource(values, timestamps, ["A", "B", "C"])
    target = np.arange(10.5, 19.5, 1 / TARGET_RATE_HZ)
    context = PipelineContext(target, TARGET_RATE_HZ, SOURCE_RATE_HZ)
    car = {
        "plugin": "experiments.lfp_preprocessing:CommonAverageReference",
        "parameters": {"channel_ids": None},
    }
    offset = {
        "plugin": f"{__name__}:OffsetStep",
        "parameters": {"value": 0.25},
    }
    result = run_preprocessing(
        source, pipeline(offset, car, resampling_step()), context
    )
    assert [step["name"] for step in result.steps] == [
        "offset",
        "common_average_reference",
        "polyphase_resample",
    ]
    np.testing.assert_allclose(result.values.mean(axis=1), 0, atol=1e-5)
    assert result.resampling["up"] == 1
    assert result.resampling["down"] == 5
    identities = [
        preprocessing_identity(pipeline(resampling_step())),
        preprocessing_identity(pipeline(offset, car, resampling_step())),
    ]
    changed = resampling_step()
    changed["parameters"]["beta"] = 8.0
    identities.append(preprocessing_identity(pipeline(changed)))
    assert len({fingerprint(value) for value in identities}) == 3


def test_polyphase_step_attenuates_above_nyquist():
    """Suppress a 30-Hz component while retaining a 2-Hz LFP component."""
    timestamps = 10 + np.arange(2000) / SOURCE_RATE_HZ
    target = np.arange(11, 29, 1 / TARGET_RATE_HZ)
    low = np.sin(2 * np.pi * 2 * timestamps)
    high = np.sin(2 * np.pi * 30 * timestamps)
    context = PipelineContext(target, TARGET_RATE_HZ, SOURCE_RATE_HZ)
    clean = run_preprocessing(
        MemorySource(low[:, None], timestamps, ["A"]),
        pipeline(resampling_step()),
        context,
    )
    mixed = run_preprocessing(
        MemorySource((low + high)[:, None], timestamps, ["A"]),
        pipeline(resampling_step()),
        context,
    )
    np.testing.assert_allclose(mixed.values, clean.values, atol=0.03)


def test_polyphase_step_bounds_each_source_read():
    """Limit native resampling reads to configured channel blocks."""

    class TrackingSource(MemorySource):
        """Record channel counts requested by preprocessing."""

        def __init__(self, values, timestamps, channel_ids):
            super().__init__(values, timestamps, channel_ids)
            self.read_widths = []

        def read(self, rows: slice, columns: np.ndarray) -> np.ndarray:
            """Record and perform one selected signal read."""
            self.read_widths.append(len(columns))
            return super().read(rows, columns)

    timestamps = 10 + np.arange(1000) / SOURCE_RATE_HZ
    source = TrackingSource(
        np.ones((len(timestamps), 19)),
        timestamps,
        [f"C{number}" for number in range(19)],
    )
    target = np.arange(10.5, 19.5, 1 / TARGET_RATE_HZ)
    context = PipelineContext(
        target,
        TARGET_RATE_HZ,
        SOURCE_RATE_HZ,
        channel_block_size=4,
    )
    run_preprocessing(source, pipeline(resampling_step()), context)
    assert source.read_widths == [4, 4, 4, 4, 3]


def test_pipeline_rejects_invalid_shapes_and_parameters():
    """Fail before reading data for malformed or incompatible pipelines."""
    with pytest.raises(ValueError, match="at least one"):
        preprocessing_steps(pipeline())
    duplicate = pipeline(resampling_step(), resampling_step())
    with pytest.raises(ValueError, match="Duplicate"):
        preprocessing_steps(duplicate)
    unknown = resampling_step()
    unknown["parameters"]["order"] = 8
    with pytest.raises(ValueError, match="Unsupported"):
        preprocessing_steps(pipeline(unknown))
    car = {
        "plugin": "experiments.lfp_preprocessing:CommonAverageReference",
        "parameters": {"channel_ids": []},
    }
    with pytest.raises(ValueError, match="nonempty"):
        preprocessing_steps(pipeline(car, resampling_step()))
    timestamps = np.arange(10, dtype=float) + 1
    timestamps[4] = 0
    ticks = np.arange(10)
    repaired, indices = _repair_timestamps(timestamps, ticks)
    assert indices == [4]
    assert repaired[4] == 5
    ticks[7:] += 1
    with pytest.raises(ValueError, match="ticks are discontinuous"):
        _repair_timestamps(np.arange(10, dtype=float) + 1, ticks)
    ticks = np.arange(10)
    timestamps = np.arange(10, dtype=float) + 1
    timestamps[5:] += 0.5
    with pytest.raises(ValueError, match="not a regular clock"):
        _repair_timestamps(timestamps, ticks)
    source = MemorySource(
        np.ones((10, 1)),
        np.arange(10, dtype=float) + 1,
        ["A"],
    )
    context = PipelineContext(
        np.arange(2, 10, dtype=float),
        1,
        1,
    )
    nonfinite = {
        "plugin": f"{__name__}:NonfiniteAlignedStep",
        "parameters": {},
    }
    with pytest.raises(ValueError, match="nonfinite"):
        run_preprocessing(source, pipeline(nonfinite), context)


def test_lfp_session_requires_paired_task_file(tmp_path):
    """Reject an NWB inventory that cannot supply aligned task signals."""
    root = tmp_path / "data"
    raw = root / "raw"
    raw.mkdir(parents=True)
    (root / "record.json").write_text(json.dumps({"id": 3854034}))
    with h5py.File(raw / f"{SESSION}.nwb", "w"):
        pass
    dataset = NHPLFPDataset(dataset_settings(root, tmp_path / "cache"))
    with pytest.raises(FileNotFoundError, match="paired task"):
        dataset.sessions()


def test_lfp_preprocessing_preview_is_cache_owned(lfp_session, tmp_path):
    """Publish native-versus-final LFP and task panels for every split."""
    dataset, session = lfp_session
    fold = dataset.fold(session, 0, True)
    assert fold.path is not None
    assert fold.path.parent.name == "lfp_fold"
    settings = preview_settings(dataset)
    windows = preview_windows(fold, settings)
    output = preprocessing_previews(
        session,
        fold,
        settings,
        tmp_path / "cache_preview",
        np.arange(CHANNEL_COUNT),
        windows,
    )
    assert output is not None
    assert len(list(output.rglob("*.png"))) == 18
    manifest = json.loads((output / "manifest.json").read_text())
    assert all("electrode_indices" in row for row in manifest["windows"])
    assert all(
        row["units"] == {"raw_lfp": "V", "lfp": "V"}
        for row in manifest["windows"]
    )
    with np.load(output / "train" / "excerpts.npz") as excerpt:
        assert excerpt["raw_lfp"].shape[1] == 3
        assert excerpt["lfp"].shape == (100, 3)
        assert len(excerpt["raw_lfp_t"]) > len(excerpt["t"])
        assert str(excerpt["raw_lfp_unit"]) == "V"
        assert str(excerpt["lfp_unit"]) == "V"
        steps = json.loads(str(excerpt["preprocessing_steps_json"]))
        assert steps[-1]["name"] == "polyphase_resample"


def test_lfp_preview_rejects_changed_source(tmp_path):
    """Refuse native preview reads after the recorded NWB source changes."""
    root = tmp_path / "data"
    root.mkdir()
    create_dataset(root)
    dataset = NHPLFPDataset(dataset_settings(root, tmp_path / "cache"))
    session = dataset.load(SESSION)
    fold = dataset.fold(session, 0, True)
    source = root / "raw" / f"{SESSION}.nwb"
    stat = source.stat()
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))
    settings = preview_settings(dataset)
    windows = preview_windows(fold, settings)
    with pytest.raises(ValueError, match="signature changed"):
        preprocessing_previews(
            session,
            fold,
            settings,
            tmp_path / "cache_preview",
            np.arange(CHANNEL_COUNT),
            windows,
        )
    fitted = [
        {
            "channel_ids": fold.arrays["ids"],
            "learned_Z": fold.arrays["Z"][indices],
            **{
                f"{prefix}_{key}": fold.arrays[key][indices]
                for prefix in ("pre", "main")
                for key in ("Y", "Z", "U")
            },
        }
        for indices in windows
    ]
    output = publish_previews(
        session,
        fold,
        settings,
        tmp_path / "run" / "data_preview" / "fitted",
        windows,
        np.arange(CHANNEL_COUNT),
        fitted,
    )
    with np.load(output / "train" / "excerpts.npz") as excerpt:
        assert "raw_lfp" not in excerpt
        assert "pre_Y" in excerpt and "main_Y" in excerpt


def _local_experiment(
    tmp_path: Path,
    name: str,
    selection: dict | None = None,
) -> dict:
    """Resolve a tracked experiment with temporary machine paths."""
    local = tmp_path / f"{name}.yaml"
    settings = {
        "extends": [
            str(CONFIGURATION / "experiments" / name),
            str(CONFIGURATION / "paths.example.yaml"),
        ],
        "paths": {
            key: str(tmp_path / key)
            for key in (
                "dataset_root",
                "cache_root",
                "artifact_root",
                "log_root",
            )
        },
    }
    if selection is not None:
        settings["selection"] = selection
    local.write_text(yaml.safe_dump(settings))
    arguments = argument_parser().parse_args(["--experiment", str(local)])
    return resolve_configuration(arguments)


def test_lfp_manifest_and_scientific_identity_are_isolated(tmp_path):
    """Resolve the approved grid and separate every LFP model recipe."""
    tracked = read_yaml(
        CONFIGURATION
        / "experiments"
        / "lfp_latent_dimension_sweep.yaml"
    )
    assert tracked["selection"] == {"sessions": None, "folds": None}
    assert tracked["suite"]["n1_max"] == 16
    assert tracked["suite"]["n_pre"] == 150
    assert tracked["suite"]["population_scales"] == [1.0]
    selected_sessions = [
        "indy_20160624_03",
        "indy_20160627_01",
        "indy_20160630_01",
        "indy_20160921_01",
        "indy_20161024_03",
        "indy_20161206_02",
        "indy_20161220_02",
        "indy_20170124_01",
        "indy_20170131_02",
    ]
    lfp = _local_experiment(
        tmp_path,
        "lfp_latent_dimension_sweep.yaml",
        {"sessions": selected_sessions, "folds": [0, 2, 4]},
    )
    spike = _local_experiment(tmp_path, "latent_dimension_sweep.yaml")
    experiment = lfp["experiment"]
    assert experiment["name"] == "latent_dimension_sweep_lfp"
    assert experiment["selection"]["folds"] == [0, 2, 4]
    assert experiment["selection"]["sessions"] == selected_sessions
    assert experiment["suite"]["nx_values"] == [1, 2, 4, 16, 32, 64]
    assert not lfp["plotting"]["previews"]["enabled"]
    assert lfp["data"]["preview_adapter"].endswith("LFPPreviewAdapter")
    assert lfp["data"]["features"] == "lfp"
    case = {
        "name": "nx_16_population_1",
        "dimensions": {"nx": 16, "n1": 16, "n2": 0, "n_pre": 150},
        "population_scale": 1.0,
    }
    assert model_settings(lfp, case) != model_settings(spike, case)
    lfp_model = model_directory(tmp_path, lfp, case)
    spike_model = model_directory(tmp_path, spike, case)
    assert lfp_model != spike_model
    lfp_fit = lfp_model / SESSION / "fold_0" / "fit"
    spike_fit = spike_model / SESSION / "fold_0" / "fit"
    horizons = lfp["evaluation"]["horizons"]
    assert prediction_directory(lfp_fit, horizons) != prediction_directory(
        spike_fit,
        horizons,
    )
    assert analysis_settings(lfp) != analysis_settings(spike)
    assert (
        tmp_path / "analysis" / lfp["experiment"]["name"]
        != tmp_path / "analysis" / spike["experiment"]["name"]
    )


@pytest.mark.skipif(
    "BRAID_NHP_LFP_DATASET_ROOT" not in os.environ,
    reason="set BRAID_NHP_LFP_DATASET_ROOT for the real-data LFP smoke test",
)
def test_real_lfp_preprocessing_and_preview_smoke(tmp_path):
    """Preprocess and preview the smallest paired real LFP session."""
    root = Path(os.environ["BRAID_NHP_LFP_DATASET_ROOT"]).resolve()
    candidates = [
        path.stem
        for path in (root / "raw").glob("indy_*.nwb")
        if (root / f"{path.stem}.mat").is_file()
    ]
    if not candidates:
        raise FileNotFoundError(
            "Expected paired Indy NWB and MATLAB task files for LFP smoke."
        )
    session = min(
        candidates,
        key=lambda value: (root / "raw" / f"{value}.nwb").stat().st_size,
    )
    settings = read_yaml(CONFIGURATION / "data" / "lfp.yaml")
    settings.update(
        root=str(root),
        cache_root=str(tmp_path / "cache"),
        cache_mode="off",
        sessions=[session],
    )
    dataset = NHPLFPDataset(settings)
    features = dataset.load(session)
    fold = dataset.fold(features, 0, True)
    previews = preview_settings(dataset)
    windows = preview_windows(fold, previews)
    output = preprocessing_previews(
        features,
        fold,
        previews,
        tmp_path / "cache_preview",
        np.arange(CHANNEL_COUNT),
        windows,
    )
    assert output is not None
    assert (output / "manifest.json").is_file()
