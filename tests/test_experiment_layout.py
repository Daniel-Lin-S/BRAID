"""Compatibility contracts for responsibility-grouped experiment packages."""

import importlib


def test_legacy_modules_are_canonical_objects():
    """Saved plugin paths retain module identity and mutable state."""
    for old, new in (
        ('nhp', 'datasets.nhp'), ('runner', 'execution.runner'),
        ('braid_backend', 'models.braid_backend'),
        ('cache', 'storage.cache'), ('reporting', 'reports.reporting'),
    ):
        assert importlib.import_module('experiments.' + old) is (
            importlib.import_module('experiments.' + new)
        )
