"""Validate fit-owned preview layout, selection, stage isolation and style."""

import copy
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pytest

from experiments.contracts import FeatureSet
from experiments.presentation import presentation
from experiments.preview_publication import publish_previews
from experiments.preview_rendering import PreviewStage, signal_figure
from experiments.previews import fitted_previews, preview_windows
from experiments.signal_previews import (
    preview_columns,
    signal_stages,
    window_excerpts,
)

STYLE = presentation()


@pytest.fixture
def signals():
    """Provide three contiguous split segments and four identified channels."""
    time = np.arange(600) / 10
    position = np.column_stack([np.sin(time), np.cos(time)])
    behavior = np.column_stack([position, np.cos(time), -np.sin(time)])
    neural = np.column_stack([np.sin(time + i) + 2 for i in range(4)])
    spikes = time[::10]
    arrays = dict(
        t=time,
        native_t=time,
        Y=neural,
        Z=behavior,
        U=position,
        native_Z=position,
        native_U=position,
        counts=neural,
        indices=np.arange(len(time)),
        ids=np.array(["A", "B", "C", "D"]),
        units=np.arange(4),
        role=np.repeat(np.arange(3), 200),
        segment=np.repeat(np.arange(3), 200),
        spike_values=np.tile(spikes, 4),
        spike_offsets=np.arange(5) * len(spikes),
    )
    metadata = dict(session="fixture", fold=0)
    return FeatureSet(arrays, metadata)


def preview_settings():
    """Return the approved defaults without writing a configuration file."""
    return dict(
        enabled=True,
        seconds=5,
        windows=3,
        channels=3,
        seed=42,
        context_samples=128,
        presentation=STYLE,
    )


def test_selection_is_split_local_and_population_invariant(signals):
    settings = preview_settings()
    first = preview_windows(signals, settings)
    second = preview_windows(signals, settings)
    for role, (left, right) in enumerate(zip(first, second)):
        np.testing.assert_array_equal(left, right)
        assert len(left) == 50
        assert np.all(signals.arrays["role"][left] == role)
    order = np.array([2, 0, 3, 1])
    np.testing.assert_array_equal(
        preview_columns(signals, settings, order),
        preview_columns(signals, settings, order[:3]),
    )
    with pytest.raises(ValueError, match="Expected 3 preview channels"):
        preview_columns(signals, settings, order[:2])
    with pytest.raises(ValueError, match="No .* preview"):
        preview_windows(signals, dict(settings, seconds=100))


def test_preprocessing_and_fitted_stage_isolation(signals):
    settings = preview_settings()
    indices = preview_windows(signals, settings)[0]
    time = signals.arrays["t"][indices]
    values = window_excerpts(
        signals,
        signals,
        indices,
        np.arange(3),
        time[0] + 5,
    )
    preprocessing = signal_stages(values)
    assert len(preprocessing) == 6
    assert [stage.key for stage in preprocessing[0][2]] == [
        "spikes",
        "counts",
        "smoothed",
    ]
    for key in ("Y", "Z", "U"):
        raw = signals.arrays[key][indices]
        values["pre_" + key] = raw + 10
        values["main_" + key] = raw + 20
    values["learned_Z"] = values["Z"] + 30
    for filename, _, stages in signal_stages(values):
        assert filename.endswith("_fitted.png")
        if filename.startswith("neural"):
            assert [s.key for s in stages] == ["pre_Y", "main_Y"]
        elif filename.startswith("behavior"):
            assert [s.key for s in stages] == ["pre_Z", "learned_Z", "main_Z"]
            dimension = 2 if "velocity" in filename else 0
            np.testing.assert_array_equal(
                stages[-1].values,
                values["main_Z"][:, dimension : dimension + 2],
            )
            assert (
                all("Velocity" in c for s in stages for c in s.coordinates)
                if dimension
                else True
            )
        else:
            assert [s.key for s in stages] == ["pre_U", "main_U"]


def test_readable_single_signal_panels(tmp_path):
    time = np.arange(4, dtype=float)
    stages = [
        PreviewStage("raw", "Original", "mm", time, time),
        PreviewStage("main", "Normalization", "unitless", time, -time),
    ]
    figure = signal_figure(stages, "Position x", "Session", (0, 4), STYLE)
    try:
        figure.canvas.draw()
        assert figure._suptitle.get_fontsize() == 32
        np.testing.assert_allclose(figure.get_size_inches(), [16, 8])
        for axis, stage in zip(figure.axes, stages):
            np.testing.assert_array_equal(
                axis.lines[0].get_ydata(), stage.values
            )
            assert axis._left_title.get_fontsize() == 28
            assert axis.get_legend().get_texts()[0].get_fontsize() == 24
    finally:
        plt.close(figure)


def test_invalid_values_and_silent_spikes():
    stage = PreviewStage(
        "raw",
        "Spike events",
        "Events",
        np.array([]),
        None,
        "spikes",
    )
    figure = signal_figure([stage], "Unit", "Session", (0, 1), STYLE)
    plt.close(figure)
    stage = PreviewStage(
        "raw",
        "Position",
        "mm",
        np.array([0.0]),
        np.array([np.nan]),
    )
    with pytest.raises(ValueError, match="Nonfinite"):
        signal_figure([stage], "Position", "Session", (0, 1), STYLE)


def test_file_tree_and_in_place_refresh(signals, tmp_path):
    """Exercise actual PNG/NPZ writers exclusively inside temporary storage."""
    settings = preview_settings()
    windows = preview_windows(signals, settings)
    root = tmp_path / "fit" / "data_preview"
    columns = np.arange(4)
    destination = publish_previews(
        signals,
        signals,
        settings,
        root / "preprocessing",
        windows,
        columns,
    )
    assert destination == root / "preprocessing"
    assert len(list(destination.rglob("*.png"))) == 18
    assert not list(destination.rglob("index.md"))
    for split in ("train", "validation", "test"):
        assert (destination / split / "excerpts.npz").exists()
    manifest = json.loads((destination / "manifest.json").read_text())
    assert [row["directory"] for row in manifest["windows"]] == [
        "train",
        "validation",
        "test",
    ]
    image = destination / "train" / "neural_A_unit_0_preprocessing.png"
    with Image.open(image) as png:
        np.testing.assert_allclose(png.info["dpi"], [200, 200], atol=0.01)
    before = image.stat().st_mtime_ns
    publish_previews(
        signals,
        signals,
        settings,
        destination,
        windows,
        columns,
    )
    assert image.stat().st_mtime_ns == before
    time.sleep(0.01)
    publish_previews(
        signals,
        signals,
        settings,
        destination,
        windows,
        columns,
        regenerate=True,
    )
    refreshed = image.stat().st_mtime_ns
    assert refreshed != before
    changed = copy.deepcopy(settings)
    changed["presentation"]["title_font"] = 34
    publish_previews(
        signals,
        signals,
        changed,
        destination,
        windows,
        columns,
    )
    assert image.stat().st_mtime_ns != refreshed
    fitted = [
        dict(
            channel_ids=signals.arrays["ids"],
            learned_Z=signals.arrays["Z"][indices] + 2,
            **{
                f"{prefix}_{key}": signals.arrays[key][indices] + offset
                for prefix, offset in (("pre", 1), ("main", 2))
                for key in ("Y", "Z", "U")
            },
        )
        for indices in windows
    ]
    output = publish_previews(
        signals,
        signals,
        settings,
        root / "fitted",
        windows,
        columns,
        fitted,
    )
    assert len(list(output.rglob("*.png"))) == 18
    with np.load(output / "train" / "excerpts.npz") as excerpt:
        assert "counts" not in excerpt and "native_Z" not in excerpt
        assert "learned_Z" in excerpt and "main_Z" in excerpt
        assert len(signal_stages(dict(excerpt))) == 6
    assert sorted(p.name for p in root.iterdir() if p.is_dir()) == [
        "fitted",
        "preprocessing",
    ]


def test_checkpoint_inference_never_fits(signals, tmp_path, monkeypatch):
    """Use saved mappings and split-local inference, without saved excerpts."""
    from types import SimpleNamespace

    from experiments import artifacts, braid_backend

    (tmp_path / "configuration").mkdir()
    (tmp_path / "data").mkdir()
    (tmp_path / "configuration" / "fit_arguments.json").write_text(
        json.dumps(dict(args_base=dict(sequence_length=128)))
    )
    np.savez_compressed(
        tmp_path / "data" / "selection.npz",
        selected_columns=np.arange(4),
        channel_ids=signals.arrays["ids"],
        unit_dimensions=signals.arrays["units"],
    )

    class Mapping:
        def apply(self, values):
            return values + 3

        def get_overall_W(self):
            return None

    calls = []

    def predict(neural, inputs):
        calls.append(len(neural))
        return (np.column_stack([inputs, inputs]),)

    component = SimpleNamespace(
        YPrepMap=Mapping(),
        ZPrepMap=Mapping(),
        UPrepMap=Mapping(),
        predict=predict,
    )
    model = SimpleNamespace(sId_pre=component, sId=component)

    def forbidden(*args, **kwargs):
        raise AssertionError("Fitted preview attempted training")

    monkeypatch.setattr(braid_backend.BRAIDModel, "fit", forbidden)
    monkeypatch.setattr(artifacts, "validate_completion", lambda run: {})
    monkeypatch.setattr(
        braid_backend.BRAIDModel,
        "loadFromFile",
        lambda path: model,
    )
    windows = preview_windows(signals, preview_settings())
    output = braid_backend.checkpoint_preview_arrays(tmp_path, signals, windows)
    assert calls == [128, 128, 128]
    for values in output:
        assert values["learned_Z"].shape == (50, 4)
        np.testing.assert_array_equal(
            values["main_Z"],
            values["learned_Z"] + 3,
        )


def test_checkpoint_previews_follow_retained_neural_channels(
    signals,
    tmp_path,
    monkeypatch,
):
    """Render only channels retained by a checkpoint preprocessing map."""
    from types import SimpleNamespace

    from BRAID.tools.LinearMapping import LinearMapping
    from experiments import artifacts, braid_backend

    (tmp_path / "configuration").mkdir()
    (tmp_path / "data").mkdir()
    (tmp_path / "configuration" / "fit_arguments.json").write_text(
        json.dumps(dict(args_base=dict(sequence_length=128)))
    )
    np.savez_compressed(
        tmp_path / "data" / "selection.npz",
        selected_columns=np.arange(4),
        channel_ids=signals.arrays["ids"],
        unit_dimensions=signals.arrays["units"],
    )

    class Mapping:
        def apply(self, values):
            return values + 3

        def get_overall_W(self):
            return None

    neural = LinearMapping()
    neural.set_to_dimension_remover(np.array([True, False, True, True]))
    component = SimpleNamespace(
        YPrepMap=neural,
        ZPrepMap=Mapping(),
        UPrepMap=Mapping(),
        predict=lambda y, u: (np.column_stack([u, u]),),
    )
    model = SimpleNamespace(sId_pre=component, sId=component)
    monkeypatch.setattr(artifacts, "validate_completion", lambda run: {})
    monkeypatch.setattr(
        braid_backend.BRAIDModel,
        "loadFromFile",
        lambda path: model,
    )
    windows = preview_windows(signals, preview_settings())
    fitted = braid_backend.checkpoint_preview_arrays(tmp_path, signals, windows)
    for values in fitted:
        np.testing.assert_array_equal(
            values["channel_ids"],
            np.array(["A", "C", "D"]),
        )
        assert values["pre_Y"].shape == (50, 3)
        assert values["main_Y"].shape == (50, 3)

    run = tmp_path / "fit"
    checkpoint = run / "checkpoints" / "model.p"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")

    def preview_plugin(reference, **kwargs):
        return fitted

    monkeypatch.setattr("experiments.previews.plugin", preview_plugin)
    output = fitted_previews(
        signals,
        signals,
        preview_settings(),
        run,
        "test:checkpoint_preview_arrays",
        windows,
    )
    with np.load(output / "train" / "excerpts.npz") as excerpt:
        np.testing.assert_array_equal(
            excerpt["channel_ids"],
            np.array(["A", "C", "D"]),
        )
        assert excerpt["pre_Y"].shape == (50, 3)


def preview_manifest(cases, states):
    """Build one analysis manifest for preview orchestration tests."""
    members = {}
    for case, state in zip(cases, states):
        key = f"{case['name']}/session_a/fold_0"
        members[key] = dict(
            session="session_a",
            fold=0,
            case=case,
            state=state,
            fit=(
                "experiments/model/session_a/fold_0/"
                f"{case['name']}"
            ),
        )
    return {"members": members}


def preview_orchestration_fixture(tmp_path):
    """Return cache-backed arrays and a deterministic in-memory dataset."""
    arrays = dict(
        t=np.arange(600) / 10,
        Y=np.ones((600, 4)),
        Z=np.ones((600, 2)),
        U=np.ones((600, 2)),
        indices=np.arange(600),
        ids=np.array(["A", "B", "C", "D"]),
        units=np.arange(4),
        role=np.repeat(np.arange(3), 200),
        segment=np.repeat(np.arange(3), 200),
    )
    session = FeatureSet(arrays, dict(session="session_a"))
    fold_path = tmp_path / "cache" / "fold" / "entry"
    fold_path.mkdir(parents=True)
    (fold_path / "manifest.json").write_text(
        json.dumps(
            {
                "identity": {"fold": 0, "source": "fixture"},
                "sha256": "fold-payload",
            }
        )
    )
    fold = FeatureSet(
        arrays,
        dict(session="session_a", fold=0),
        fold_path,
    )
    observed = {}

    class Dataset:
        def __init__(self, settings):
            observed["settings"] = settings

        def sessions(self):
            return ["session_a"]

        def inventory(self):
            return {"session_a": arrays["ids"].tolist()}

        def load(self, name):
            observed.setdefault("loads", []).append(name)
            return session

        def fold(self, source, number, velocity):
            observed.setdefault("folds", []).append((number, velocity))
            return fold

    return Dataset, observed


def preview_snapshots(tmp_path, cases):
    """Build minimal resolved settings for plot-stage preview tests."""
    return dict(
        experiment=dict(
            seed=42,
            preview_model_plugin="fixture:preview",
        ),
        data=dict(
            plugin="fixture:dataset",
            cache_mode="rebuild",
            infer_velocity=True,
            preview_adapter=None,
        ),
        plotting=dict(
            presentation=STYLE,
            previews=dict(
                enabled=True,
                windows=3,
                seconds=5,
                channels=3,
                seed=42,
                selection=dict(
                    sessions=["session_a"],
                    folds=None,
                    cases=[case["name"] for case in cases],
                ),
            ),
        ),
        runtime=dict(preview_regeneration="incomplete"),
        paths=dict(
            cache_root=str(tmp_path / "cache"),
            artifact_root=str(tmp_path / "artifacts"),
        ),
    )


def test_render_settings_uses_whole_analysis_suite(monkeypatch):
    """The largest model context defines every canonical preview window."""
    from experiments import preview_regeneration

    cases = [{"name": "short"}, {"name": "long"}]
    snapshots = dict(
        plotting=dict(
            presentation=STYLE,
            previews=dict(
                enabled=True,
                selection={},
                windows=3,
                seconds=5,
                channels=3,
                seed=42,
            ),
        ),
        data=dict(preview_adapter="fixture:adapter"),
    )

    def resolve(identity):
        length = 128 if identity["case"]["name"] == "short" else 256
        return {"args_base": {"sequence_length": length}}

    monkeypatch.setattr(preview_regeneration, "resolved_fit", resolve)
    settings = preview_regeneration._render_settings(snapshots, cases)
    assert settings["context_samples"] == 256
    assert settings["adapter"] == "fixture:adapter"
    assert "enabled" not in settings and "selection" not in settings


def test_population_destination_excludes_source_location(tmp_path):
    """Shared preview ownership follows scientific fold provenance."""
    from experiments.preview_regeneration import _population_destination

    arrays = dict(ids=np.array(["A", "B"]))
    destinations = []
    for number, source in enumerate(("/machine/a", "/machine/b")):
        path = tmp_path / f"fold_{number}"
        path.mkdir()
        (path / "manifest.json").write_text(json.dumps({
            "identity": {
                "fold": 0,
                "source": {"sha256": "source-digest", "source": source},
            },
            "sha256": "fold-digest",
        }))
        fold = FeatureSet(arrays, {}, path)
        destinations.append(_population_destination(
            tmp_path,
            "session_a",
            0,
            fold,
            np.array([1, 0]),
        ))
    assert destinations[0] == destinations[1]


def test_plot_previews_share_population_windows_and_warn_unavailable(
    tmp_path,
    monkeypatch,
    caplog,
):
    """One population rendering serves models and unavailable fits warn once."""
    from experiments import preview_regeneration

    cases = [
        dict(name="case_a", population_scale=1.0),
        dict(name="case_b", population_scale=1.0),
    ]
    manifest = preview_manifest(cases, ["complete", "failed"])
    snapshots = preview_snapshots(tmp_path, cases)
    Dataset, observed = preview_orchestration_fixture(tmp_path)
    calls = {"preprocessing": [], "fitted": []}

    monkeypatch.setattr(
        preview_regeneration, "read_manifest", lambda root: manifest
    )
    monkeypatch.setattr(
        preview_regeneration,
        "_render_settings",
        lambda settings, values: dict(
            windows=3,
            seconds=5,
            channels=3,
            seed=42,
            context_samples=128,
            presentation=STYLE,
            adapter=None,
        ),
    )
    monkeypatch.setattr(
        preview_regeneration,
        "plugin",
        lambda reference, **kwargs: Dataset(kwargs["settings"]),
    )

    def preprocessing(*args):
        calls["preprocessing"].append(args)

    def fitted(*args):
        calls["fitted"].append(args)

    monkeypatch.setattr(
        preview_regeneration, "preprocessing_previews", preprocessing
    )
    monkeypatch.setattr(preview_regeneration, "fitted_previews", fitted)
    caplog.set_level("WARNING")
    result = preview_regeneration.render_previews(snapshots, tmp_path)
    assert result == {"preprocessing": 1, "fitted": 1}
    assert observed["settings"]["cache_mode"] == "reuse"
    assert observed["loads"] == ["session_a"]
    assert observed["folds"] == [(0, True)]
    assert len(calls["preprocessing"]) == 1
    assert len(calls["fitted"]) == 1
    preprocessing_windows = calls["preprocessing"][0][5]
    fitted_windows = calls["fitted"][0][5]
    assert preprocessing_windows is fitted_windows
    destination = calls["preprocessing"][0][3]
    assert destination.is_relative_to(
        tmp_path / "cache" / "previews" / "preprocessing"
    )
    assert calls["preprocessing"][0][-1] is False
    assert calls["fitted"][0][-1] is False
    assert "case_b" in caplog.text
    assert caplog.text.count("unavailable fitted preview") == 1
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "-1"


def test_preview_rendering_continues_then_fails(tmp_path, monkeypatch):
    """Independent population failures are collected after every attempt."""
    from experiments import preview_regeneration

    cases = [
        dict(name="half", population_scale=0.5),
        dict(name="full", population_scale=1.0),
    ]
    snapshots = preview_snapshots(tmp_path, [])
    snapshots["plotting"]["previews"]["selection"]["cases"] = []
    manifest = preview_manifest(cases, ["pending", "pending"])
    Dataset, _ = preview_orchestration_fixture(tmp_path)
    attempts = []

    monkeypatch.setattr(
        preview_regeneration, "read_manifest", lambda root: manifest
    )
    monkeypatch.setattr(
        preview_regeneration,
        "_render_settings",
        lambda settings, values: dict(
            windows=3,
            seconds=5,
            channels=2,
            seed=42,
            context_samples=128,
            presentation=STYLE,
            adapter=None,
        ),
    )
    monkeypatch.setattr(
        preview_regeneration,
        "plugin",
        lambda reference, **kwargs: Dataset(kwargs["settings"]),
    )

    def fail(*args):
        attempts.append(args[3])
        raise ValueError("injected preview failure")

    monkeypatch.setattr(preview_regeneration, "preprocessing_previews", fail)
    with pytest.raises(RuntimeError, match="independent selections"):
        preview_regeneration.render_previews(snapshots, tmp_path)
    assert len(attempts) == 2
    assert attempts[0] != attempts[1]


def test_preview_cache_failure_does_not_skip_later_fold(
    tmp_path,
    monkeypatch,
):
    """Aggregate one cache failure after independent folds are attempted."""
    from experiments import preview_regeneration

    case = dict(name="case_a", population_scale=1.0)
    manifest = preview_manifest([case], ["pending"])
    second = copy.deepcopy(next(iter(manifest["members"].values())))
    second["fold"] = 1
    manifest["members"]["case_a/session_a/fold_1"] = second
    snapshots = preview_snapshots(tmp_path, [])
    Dataset, _ = preview_orchestration_fixture(tmp_path)
    attempts = []

    class FailingDataset(Dataset):
        def fold(self, source, number, velocity):
            if number == 0:
                raise ValueError("injected cache failure")
            return super().fold(source, number, velocity)

    monkeypatch.setattr(
        preview_regeneration, "read_manifest", lambda root: manifest
    )
    monkeypatch.setattr(
        preview_regeneration,
        "_render_settings",
        lambda settings, values: dict(
            windows=3,
            seconds=5,
            channels=2,
            seed=42,
            context_samples=128,
            presentation=STYLE,
            adapter=None,
        ),
    )
    monkeypatch.setattr(
        preview_regeneration,
        "plugin",
        lambda reference, **kwargs: FailingDataset(kwargs["settings"]),
    )
    monkeypatch.setattr(
        preview_regeneration,
        "preprocessing_previews",
        lambda *args: attempts.append(args[3]),
    )
    with pytest.raises(RuntimeError, match="injected cache failure"):
        preview_regeneration.render_previews(snapshots, tmp_path)
    assert len(attempts) == 1
    assert "fold_1" in str(attempts[0])
