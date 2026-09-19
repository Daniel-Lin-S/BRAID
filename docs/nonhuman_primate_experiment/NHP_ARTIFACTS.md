# NHP fitting and artifact internals

This reference defines immutable fitting and analysis artifacts. The public
runbook is [NHP experiments](QUICK_START.md).

## Fit identity and layout

`source/experiments/artifacts.py` addresses a fit as:

```text
experiments/<model-name>_<settings-hash>/<session>/fold_<number>/<fit-id>/
```

The settings hash represents the session-independent fitting recipe. The fit
ID adds source provenance, fold, selected ordered channels, effective fitting
arguments, and the derived seed. Analysis membership, output paths, logging,
previews, rendering, scoring settings, and resource allocation do not alter a
fit identity.

Each fit separates configuration, provenance, data, and checkpoint artifacts.
`fit_complete.json` records checksums for published fit artifacts. The model
checkpoint is the native BRAID serialization in `checkpoints/model.p`.
`model_settings.json` validates the recipe at the settings-directory level.

## Completion and recovery

`fitting.py` publishes fit completion immediately after checkpoint save.
Prediction bundles are stored below the fit by requested horizon set and are
validated by checkpoint/source provenance and payload checksum. Prediction
payloads must have expected Y/Z/X dimensions, finite values, and valid masks
that respect independent-window boundaries.

Interrupted or invalid incomplete component, prediction, or metric outputs are
quarantined or rebuilt under their owning lock. Completed fits, completed
components, completed predictions, and completed metrics are never overwritten.
`restart.py` validates and reuses completed components before continuing an
unfinished fit.

## Analysis membership

`analysis.py` creates an analysis manifest before fitting. A member is keyed
by case, session, and fold and references its shared fit. Completed members add
checksummed prediction and metrics references. Analysis settings are separate
from fit identity, so multiple analyses can reuse a completed fit.

An evaluation failure marks only the analysis member as failed; it cannot
invalidate a completed checkpoint. Explicitly certified inference-only
implementation changes may reuse existing validated prediction bundles.
