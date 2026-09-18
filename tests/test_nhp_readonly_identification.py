"""Opt-in real-fit identification; never launch a script or write artifacts.

NHP_IDENTIFICATION_CONFIG selects an ignored/local experiment YAML.
NHP_IDENTIFICATION_BASELINE optionally selects a pre-edit JSON snapshot in
temporary storage. All real data/artifact/cache/log roots are write-guarded.
No concrete machine paths or real-validation configurations belong in Git.
"""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys

import pytest


@contextmanager
def forbid_root_writes(roots):
    """Reject Python filesystem mutations beneath protected absolute roots."""
    active = True

    def audit(event, arguments):
        if not active:
            return
        if event == "open":
            path, mode, flags = arguments
            if not (
                flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)
                or mode
                and any(char in mode for char in "wax+")
            ):
                return
            paths = [path]
        elif event in {
            "os.mkdir",
            "os.remove",
            "os.rmdir",
            "os.rename",
            "os.link",
            "os.symlink",
            "os.chmod",
            "os.utime",
            "os.truncate",
        }:
            paths = (
                arguments[:2]
                if event
                in {
                    "os.rename",
                    "os.link",
                    "os.symlink",
                }
                else arguments[:1]
            )
        else:
            return
        for path in paths:
            if isinstance(path, (str, bytes, os.PathLike)):
                resolved = Path(os.fsdecode(path)).resolve()
                if any(resolved.is_relative_to(root) for root in roots):
                    raise AssertionError(
                        f"Forbidden production write: {resolved}"
                    )

    sys.addaudithook(audit)
    try:
        yield
    finally:
        active = False


def test_real_completed_fit_identification(monkeypatch):
    """Recompute actual fit IDs with current code/config and validate reuse."""
    reference = os.environ.get("NHP_IDENTIFICATION_CONFIG")
    if reference is None:
        pytest.skip("Set NHP_IDENTIFICATION_CONFIG for read-only validation.")
    from experiments.analysis import analysis_settings
    from experiments.artifacts import (
        artifact_path,
        completed_fit,
        fit_directory,
    )
    from experiments.braid_backend import BRAIDBackend, build_cases
    from experiments.cache import file_digest, fingerprint, load_entry
    from experiments.configuration import argument_parser, resolve_configuration
    from experiments.implementation import (
        implementation_signatures,
        scientific_versions,
    )
    from experiments.nhp import NHPDataset
    from experiments import nhp
    from experiments.runner import case_identity, population_order

    def forbidden(*args, **kwargs):
        raise AssertionError("Identification attempted fitting or inference.")

    for name in ("fit", "load", "predict", "save"):
        monkeypatch.setattr(BRAIDBackend, name, forbidden)
    for name in ("load", "fold"):
        monkeypatch.setattr(NHPDataset, name, forbidden)
    arguments = argument_parser().parse_args(["--experiment", reference])
    config = resolve_configuration(arguments)
    config.update(implementation_signatures())
    config["versions"] = scientific_versions()
    root = Path(config["paths"]["artifact_root"])
    roots = [Path(value).resolve() for value in config["paths"].values()]
    baseline_path = os.environ.get("NHP_IDENTIFICATION_BASELINE")
    baseline = (
        json.loads(Path(baseline_path).read_text()) if baseline_path else {}
    )
    if baseline:
        for key in ("fitting_implementation", "inference_implementation"):
            assert config[key] == baseline["signatures"][key]
        runs = [Path(path) for path in baseline["records"]]
    else:
        runs = [
            path.parent.parent
            for path in (root / "experiments").glob(
                "*/indy_20160407_02/fold_0/*/provenance/fit_complete.json"
            )
        ]
    assert runs, "Expected at least one real completed fit."
    with forbid_root_writes(roots):
        dataset = NHPDataset(dict(config["data"], cache_mode="off"))
        order = population_order(
            dataset.inventory(), config["experiment"]["seed"]
        )
        cases = build_cases(config["experiment"]["suite"])
        checked = []
        for run in runs:
            recorded = json.loads(
                artifact_path(run, "identity.json").read_text()
            )
            old = recorded["identity"]
            case = next(
                case
                for case in cases
                if case["dimensions"]["nx"] == old["model"]["model"]["nx"]
            )
            runtime = json.loads(artifact_path(run, "runtime.json").read_text())
            cache = Path(runtime["cache"])
            manifest = json.loads((cache / "manifest.json").read_text())
            features = load_entry(cache, manifest["identity"])
            assert features.metadata["identity"]["implementation_sha256"] == (
                file_digest(Path(nhp.__file__))
            )
            current, _ = case_identity(
                features,
                old["fold"],
                case,
                config,
                order,
            )
            assert fit_directory(root, current) == run
            assert completed_fit(run, current)
            if baseline:
                previous = baseline["records"][str(run)]
                assert (
                    file_digest(artifact_path(run, "fit_complete.json"))
                    == (previous["completion_sha256"])
                )
                assert (
                    file_digest(artifact_path(run, "identity.json"))
                    == (previous["identity_sha256"])
                )
            checked.append(case["dimensions"]["nx"])
        manifests = list(
            (root / "analysis" / config["experiment"]["name"]).glob(
                "*/manifest.json"
            )
        )
        assert manifests, "Expected an existing analysis to compare identities."
        for path in manifests:
            original = json.loads(path.read_text())["specification"]
            if "plotting" not in original["settings"]:
                continue
            updated = dict(
                settings=analysis_settings(config),
                members=original["members"],
            )
            assert fingerprint(updated) != path.parent.name
        print(f"Read-only completed fits verified: nx={sorted(checked)}")


def test_real_analysis_publication_only_in_temporary_storage(tmp_path):
    """Generate a new analysis view of real membership without starting jobs."""
    reference = os.environ.get("NHP_IDENTIFICATION_CONFIG")
    if reference is None:
        pytest.skip("Set NHP_IDENTIFICATION_CONFIG for temporary publication.")
    from experiments.analysis import initialize_analysis, read_manifest
    from experiments.configuration import argument_parser, resolve_configuration
    from experiments.implementation import (
        implementation_signatures,
        scientific_versions,
    )

    args = argument_parser().parse_args(["--experiment", reference])
    config = resolve_configuration(args)
    config.update(implementation_signatures())
    config["versions"] = scientific_versions()
    production = Path(config["paths"]["artifact_root"])
    roots = [Path(value).resolve() for value in config["paths"].values()]
    with forbid_root_writes(roots):
        sources = list(
            (production / "analysis" / config["experiment"]["name"]).glob(
                "*/manifest.json"
            )
        )
        assert sources
        source = next(
            path
            for path in sources
            if "plotting"
            in json.loads(path.read_text())["specification"]["settings"]
        )
        members = json.loads(source.read_text())["specification"]["members"]
        output = initialize_analysis(tmp_path, config, members)
        assert output.is_relative_to(tmp_path)
        assert output.name != source.parent.name
        actual = read_manifest(output)
        assert actual["members"] == members
        assert all(
            actual["members"][key]["fit_id"] == member["fit_id"]
            for key, member in members.items()
        )
        assert not (tmp_path / "experiments").exists()
        assert initialize_analysis(tmp_path, config, members) == output
        config["plotting"]["presentation"]["title_font"] += 2
        assert initialize_analysis(tmp_path, config, members) == output
        assert len(list((tmp_path / "analysis").glob("*/*/manifest.json"))) == 1
