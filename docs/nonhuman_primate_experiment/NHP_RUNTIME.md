# NHP launch, lifecycle, and preview internals

This reference covers process execution and rendering helpers. The public
runbook is [NHP experiments](QUICK_START.md).

## Launch and lifecycle

`launcher.py` resolves configuration, creates one launch directory, writes
`launch.json`, and starts `runner.py` in the foreground or a detached OS
session. It sets the Python environment needed by the configured runtime.

`runtime.py` selects an allowed GPU by free memory then utilization when the
device is `auto`; explicit CPU, GPU index, and UUID selection are supported.
It configures TensorFlow threading and verifies a forward/backward operation
before fitting. GPU selection is a point-in-time choice, not a reservation.

`session_logging()` captures Python and native output in one session log.
`lifecycle_scope()` writes fitting session, fold, and model events to
`experiment.log`. `stage_scope()` records preprocessing preview counts and
preview/plot completion separately. A model failure records its active phase,
preserves valid artifacts, and allows independent models, folds, and sessions
to continue. Interrupts and shared setup errors remain terminal.

## Previews and diagnostics

`previews.py` selects one contiguous preview window per train, validation,
and test split. Preprocessing previews are created before fitting; fitted
previews run checkpoint inference only. Preview publication records checksums
and selection metadata under the owning fit.

`diagnostics.py` observes completed components and invokes rendering without
adding training callbacks. Rendering failures are collected separately from
scientific fit completion. `preview_regeneration.py`,
`preview_publication.py`, `preview_rendering.py`, and
`signal_previews.py` own saved-preview regeneration and display details.
