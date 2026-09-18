"""Validate fit-owned preview layout, selection, stage isolation and style."""

import copy
import json

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pytest

from experiments.contracts import FeatureSet
from experiments.presentation import presentation
from experiments.preview_publication import publish_previews
from experiments.preview_rendering import PreviewStage, signal_figure
from experiments.previews import preview_windows
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
    assert image.stat().st_mtime_ns != before
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
