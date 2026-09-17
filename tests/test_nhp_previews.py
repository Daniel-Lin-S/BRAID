"""Validate signal isolation, readable labels and fitted-stage semantics."""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import yaml

from experiments.preview_rendering import PreviewStage, signal_figure
from experiments.signal_previews import signal_stages

ROOT = Path(__file__).resolve().parents[1]
STYLE = yaml.safe_load(
    (ROOT / "assets/config/nhp/data/spikes.yaml").read_text()
)["previews"]["presentation"]


def test_readable_single_signal_panels():
    """Keep each stage separate with readable outside legends."""
    time = np.arange(4, dtype=float)
    stages = [
        PreviewStage("raw", "Original", "mm", time, time),
        PreviewStage("main", "Main normalization", "unitless", time, -time),
    ]
    figure = signal_figure(stages, "Position x", "Session", (0, 4), STYLE)
    try:
        figure.canvas.draw()
        assert figure._suptitle.get_fontsize() == 22
        for axis, stage in zip(figure.axes, stages):
            assert len(axis.lines) == 1
            np.testing.assert_array_equal(
                axis.lines[0].get_ydata(), stage.values
            )
            assert axis._left_title.get_fontsize() == 18
            legend = axis.get_legend()
            assert legend.get_texts()[0].get_fontsize() >= 16
            assert legend.get_window_extent().x0 > axis.get_window_extent().x1
            assert all(t.get_fontsize() >= 16 for t in axis.get_yticklabels())
    finally:
        plt.close(figure)


def test_invalid_values_and_silent_spikes():
    """Reject nonfinite signals while accepting a genuinely silent raster."""
    stage = PreviewStage(
        "raw", "Spike events", "Events", np.array([]), None, "spikes"
    )
    figure = signal_figure([stage], "Unit", "Session", (0, 1), STYLE)
    plt.close(figure)
    stage = PreviewStage(
        "raw", "Position", "mm", np.array([0.0]), np.array([np.nan])
    )
    with pytest.raises(ValueError, match="Nonfinite"):
        signal_figure([stage], "Position", "Session", (0, 1), STYLE)


def test_position_velocity_and_input_stage_isolation():
    """Full behavior uses separate coordinates and learned main-model Z."""
    time = np.arange(5, dtype=float)
    values = dict(
        t=time,
        native_t=time,
        channel_ids=np.array(["unit A"]),
        unit_dimensions=np.array([2]),
        spikes_0=time,
        counts=time[:, None],
        smoothed=time[:, None],
        position=np.column_stack([time, -time]),
    )
    for key, width in (("Z", 4), ("U", 2), ("Y", 1)):
        array = np.column_stack([time + c for c in range(width)])
        values[key] = array
        values["native_" + key] = array[:, :2]
        values["pre_" + key] = array + 10
        values["main_" + key] = array + 20
    values["learned_Z"] = values["Z"] + 30
    figures = signal_stages(values)
    assert len(figures) == 4
    for filename, label, stages in figures:
        assert all(
            s.values is None or s.values.ndim == (2 if s.coordinates else 1)
            for s in stages
        )
        if filename.startswith("neural/"):
            assert [s.key for s in stages] == [
                "spikes",
                "counts",
                "smoothed",
                "pre_Y",
                "main_Y",
            ]
        elif filename.startswith("behavior/"):
            assert [s.key for s in stages][-3:] == [
                "pre_Z",
                "learned_Z",
                "main_Z",
            ]
            dim = 0
            if "velocity" in filename:
                dim += 2
                assert stages[2].key == "velocity"
            np.testing.assert_array_equal(
                stages[-1].values, values["main_Z"][:, dim : dim + 2]
            )
        else:
            assert [s.key for s in stages] == [
                "native_U",
                "aligned_U",
                "pre_U",
                "main_U",
            ]


def test_saved_preview_regeneration_without_training(monkeypatch):
    """Check actual saved outputs and block fitting or source extraction."""
    reference = os.environ.get("NHP_PREVIEW_EXPERIMENT")
    if reference is None:
        pytest.skip("Set NHP_PREVIEW_EXPERIMENT for saved-artifact validation.")
    from BRAID.BRAIDModel import BRAIDModel
    from experiments.braid_backend import BRAIDBackend
    from experiments.cache import file_digest
    from experiments.configuration import argument_parser, resolve_configuration
    from experiments.nhp import NHPDataset
    from experiments.preview_regeneration import regenerate_previews

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Preview regeneration attempted extraction or fit."
        )

    for owner, names in (
        (NHPDataset, ("load", "fold")),
        (BRAIDBackend, ("fit",)),
        (BRAIDModel, ("fit", "restoreModels")),
    ):
        for name in names:
            monkeypatch.setattr(owner, name, forbidden)
    arguments = argument_parser().parse_args(
        ["--experiment", reference, "--stage", "preview"]
    )
    settings = resolve_configuration(arguments)
    directory = regenerate_previews(settings)
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["complete"] and manifest["source_files_unchanged"]
    for name, digest in manifest["identity"]["source_checksums"].items():
        assert file_digest(Path(name)) == digest
    for name, digest in manifest["checksums"].items():
        assert file_digest(directory / name) == digest
    config = settings["data"]["previews"]
    assert config["presentation"] == STYLE
    expected_count = config["windows"] * (len(config["channel_ids"]) + 2)
    assert len(list(directory.rglob("*.png"))) == expected_count
    for record, bounds in zip(manifest["windows"], config["window_ranges"]):
        np.testing.assert_allclose([record["start"], record["stop"]], bounds)
        assert record["channel_ids"] == config["channel_ids"]
        with np.load(directory / record["directory"] / "excerpts.npz") as data:
            assert data["channel_ids"].tolist() == config["channel_ids"]
            assert data["unit_dimensions"].tolist() == record["unit_dimensions"]
            assert data["source_indices"].tolist() == record["source_indices"]
            for value in data.values():
                if np.issubdtype(value.dtype, np.number):
                    assert np.isfinite(value).all()
            descriptions = signal_stages(dict(data))
            for description, saved in zip(descriptions, record["figures"]):
                filename, label, stages = description
                assert filename == saved["file"]
                assert [s.key for s in stages] == [
                    s["key"] for s in saved["stages"]
                ]
                figure = signal_figure(
                    stages, label, "Saved window", tuple(bounds), STYLE
                )
                try:
                    assert len(figure.axes) == len(stages)
                    for axis, stage in zip(figure.axes, stages):
                        assert axis.get_legend() is not None
                        assert axis._left_title.get_fontsize() == 18
                        count = (
                            2
                            if stage.coordinates
                            else int(stage.kind != "spikes")
                        )
                        assert len(axis.lines) == count
                        for i, line in enumerate(axis.lines):
                            expected = (
                                stage.values[:, i]
                                if stage.coordinates
                                else stage.values
                            )
                            np.testing.assert_array_equal(
                                line.get_ydata(), expected
                            )
                        if stage.coordinates:
                            assert axis.lines[0].get_color() != (
                                axis.lines[1].get_color()
                            )
                            assert axis.lines[0].get_linestyle() != (
                                axis.lines[1].get_linestyle()
                            )
                            assert [
                                t.get_text()
                                for t in axis.get_legend().get_texts()
                            ] == list(stage.coordinates)
                finally:
                    plt.close(figure)

    from experiments.cache import load_entry
    from experiments.previews import fitted_previews, preprocessing_previews

    def saved_cache(path):
        metadata = json.loads((path / "manifest.json").read_text())
        return load_entry(path, metadata["identity"])

    fold = saved_cache(Path(manifest["fold_cache"]))
    session = saved_cache(Path(manifest["session_cache"]))
    preprocessing = preprocessing_previews(
        session, fold, config, directory / "unused_fallback"
    )
    assert preprocessing.is_relative_to(fold.path / "previews")
    with np.load(Path(config["source_run"]) / "selection.npz") as saved:
        columns = saved["selected_columns"]
    fitted = fitted_previews(
        session,
        fold,
        config,
        Path(config["source_run"]),
        settings["experiment"]["preview_model_plugin"],
        columns,
    )
    assert fitted.is_relative_to(fold.path / "fitted_previews")
    for revision in (preprocessing, fitted):
        assert len(list(revision.rglob("*.png"))) == expected_count
    for name, digest in manifest["identity"]["source_checksums"].items():
        assert file_digest(Path(name)) == digest
    print(f"Independent previews: {directory}")
    print(f"Preprocess previews: {preprocessing}")
    print(f"Fitted previews: {fitted}")
