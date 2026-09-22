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
`experiment.log`. `stage_scope()` records preprocessing fold counts and plot
completion separately. A model failure records its active phase,
preserves valid artifacts, and allows independent models, folds, and sessions
to continue. Interrupts and shared setup errors remain terminal.

## Previews and diagnostics

Preview rendering is a preview-only mode of the `plot` stage. It requires
selected sessions; folds default to the saved analysis folds, and exact
analysis case names select fitted previews. With no cases, only preprocessing
previews are produced. CLI selectors replace their corresponding YAML lists.

`previews.py` selects one train, validation, and test window using the largest
context required by the saved analysis suite. Every distinct ordered neural
population shares one preprocessing rendering under the cache root. Fitted
previews remain under selected completed fits and load checkpoints on CPU.
Missing scientific caches are built with reuse semantics; plotting never
rebuilds an existing scientific cache.

`runtime.preview_regeneration` accepts `incomplete` or `all`. Publication
records source checksums, window indices, channel IDs, rendering settings, and
implementation signatures. Independent rendering failures are collected and
reported after the remaining selections finish.

`diagnostics.py` observes completed components and invokes history rendering
without adding training callbacks. Component PNGs and recovery artifacts are
always retained. `runtime.tensorboard` defaults to false; `--tensorboard`
enables event serialization only for newly fitted components and does not
change fit identity or backfill completed fits.

## Parallel fitting

`runtime.parallel_workers` defaults to 1 and accepts a positive integer;
`--parallel-workers` overrides it. Multiple workers require `device: auto`.
Only fitting uses the pool; other stages execute serially. Each worker has
the configured CPU thread limits, so total CPU demand scales with workers.

The coordinator ranks visible GPUs once by descending free memory, ascending
utilization, then physical index, and requires enough distinct devices.
Selection respects `CUDA_VISIBLE_DEVICES` but does not reserve resources
against unrelated jobs. Assignments remain fixed for the invocation.

A CPU-only reporting child resolves the complete analysis membership. The
coordinator never initializes TensorFlow. Sessions with more unresolved models
run first, with sample count as the tie-breaker. Completed sessions still pass
through validation and reuse. Each persistent spawned GPU process receives
one session at a time and executes its folds and models sequentially.

Workers own their session logs, including native output. A logging queue sends
lifecycle events to the coordinator's sole `experiment.log` writer. Session
completion triggers serialized reports in that CPU child; pending sessions
remain deferred. Final reporting runs after all session attempts finish.

Model failures are collected while independent work continues. Shared setup
errors, worker crashes, and interrupts cancel dispatch and stop worker process
groups, including component renderers. Completed artifacts remain available
for normal validated reuse. Worker count and GPU allocation do not enter
scientific identity.

Launch diagnostics record assignments, elapsed session/model time, and Linux
process write-byte deltas (unavailable on unsupported systems). These counters
include all process writes and cannot isolate checkpoint callback time.
