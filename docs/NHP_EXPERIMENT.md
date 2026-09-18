# NHP experiments

Experiment manifests select separate data, model/training, evaluation and plotting configurations. Machine paths and private validation settings belong in ignored local YAML files.

## Dataset and local setup

Obtain the Indy `.mat` spike/task files from the [public reaching dataset](https://zenodo.org/records/3854034). Save the record's [API metadata](https://zenodo.org/api/records/3854034) as `record.json` alongside the session files.

Copy `assets/config/nhp/paths.example.yaml` to `assets/config/nhp/paths.local.yaml`. Set its absolute dataset, cache, artifact and log directories. Set the optional Python interpreter and runtime preferences there. These machine settings are ignored by Git.

Use Python 3.11 with the dependencies in `requirements.txt`; the exact validated package lock is `assets/environment/requirements-gpu.lock.txt`. The launcher selects legacy Keras and defaults to automatic GPU selection; explicit CPU execution is also supported.

## Experiment selection

| Manifest under `assets/config/nhp/experiments` | Scope |
| --- | --- |
| `latent_dimension_sweep.yaml` | Full M1 population; latent dimensions 1–64 |
| `neural_population_sweep.yaml` | Nested 25%, 50%, 100% populations at dimensions 16 and 64 |
| `nonlinearity_sweep.yaml` | Deferred; architecture variants and comparison controls require specification |

The manifests reference reusable YAML files under `data`, `models`, `evaluation` and `plotting`. Evaluation files contain scoring settings, not experiment grids or training profiles. Model files own training settings. Data files own velocity inference and inspection settings.

All YAML files use the same semantics; `.local.yaml` only controls Git ignoring. Pass either filename with `--experiment`. An experiment can define `paths` and `runtime` directly or inherit them alongside an experiment definition. For example, `experiments/latent_dimension_sweep.local.yaml` can contain:

```yaml
extends:
  - latent_dimension_sweep.yaml
  - ../paths.local.yaml
```

Parents merge in order, followed by the current file's overrides. Inherited module references remain relative to the file that declares them.

From the repository root, run the stages of experiment in the following order:

```bash
bash scripts/train_nhp.sh --experiment assets/config/nhp/experiments/latent_dimension_sweep.local.yaml --dry-run
bash scripts/train_nhp.sh --experiment assets/config/nhp/experiments/latent_dimension_sweep.local.yaml --stage preprocess
bash scripts/train_nhp.sh --experiment assets/config/nhp/experiments/latent_dimension_sweep.local.yaml --stage fit --detach
bash scripts/train_nhp.sh --experiment assets/config/nhp/experiments/latent_dimension_sweep.local.yaml --stage plot
```

`--session` and `--fold` restrict execution scope. `--log-level` changes verbosity; `--no-plots` and `--no-previews` independently disable experiment and preprocessing figures.

Dry runs resolve configuration without creating logs, extracting data or starting training.

## Logs, artifacts and cache

Text logs use this structure beneath the configured log root:

```text
<experiment>/<stage>/<UTC-launch-id>/
  launch.json
  console.log
  experiment.log
  sessions/<session>.log
```

Every invocation has a unique directory. `experiment.log` contains session/fold lifecycle notifications and absolute detail-log paths. `console.log` contains launch-level diagnostics. All Python logs, progress output and native library output within a session go only to its session log, including all folds and model settings. Detached launches print the worker PID and absolute log location.

Fitting artifacts are shared across experiment definitions. Analysis and figures remain experiment-specific:

```text
<artifact_root>/
  experiments/BRAID_<64-character-settings-hash>/
    model_settings.json
    <session>/fold_<number>/<fit-id>/
      configuration/  # Resolved settings and fitting arguments
      provenance/     # Identity, seed, runtime, fit completion and references
      data/           # Ordered channels and fitting indices
      checkpoints/    # Native fitted model
      components/
        behaviour_preprocess/
          01_neural_dynamics/
          02_behaviour_decoder/
        main/
          01_behaviour_relevant_neural_dynamics/
          02_neural_decoder/
          03_residual_neural_dynamics/
      predictions/test/horizons_<steps>/
        predictions.npz
        completion.json
  analysis/<experiment-name>/<analysis-id>/
    manifest.json
    metrics/<case>/<session>/fold_<number>/metrics.json
    summaries/
    plots/
    previews/
```

The adapter declares its public `model_name` (`BRAID`) independently of its Python class name. The settings directory combines that name with the full SHA-256 of the complete fitting recipe, serialized as canonical JSON with sorted keys and nonfinite values rejected. Recipe resolution includes YAML inheritance, defaults, dimensions and fitting overrides.

The recipe includes all architecture and numerical training settings (optimizer, stopping rules, configured batch limit, training horizons and loss weights), numerical preprocessing and split policy, this case's population scale and selection policy, configured base seed, adapter identifier, and fitting implementation/dependency versions. It excludes other cases in the sweep, session identity, fold index, source-file contents, resource allocation, paths, logging, previews, plotting and evaluation settings. A session-specific batch cap does not change the settings directory; different configured batch limits are distinct recipes even when their effective sizes coincide.

`model_settings.json` contains `settings_hash` and its complete canonical `settings` payload. Publication is atomic and locked; existing metadata is validated against the directory name and never overwritten. Missing or conflicting metadata for an occupied settings directory is an error.

The fit ID references the settings hash and additionally identifies effective numerical training settings, ordered channel IDs, source content, preprocessing, fold and derived random seed. Model adapters resolve data-dependent batch size before computing these fit-specific inputs. Seeds derive from effective fitting inputs, independently of directory names and sweep labels. Experiment membership, scoring sets and inference horizons do not enter either fitting hash. Only this naming scheme is supported; artifact lookup does not use aliases, fallback searches or migration.

For the same session/fold, `neural_population_sweep` at `nx=16` or `nx=64` and population scale `1.0` references the same fits as `latent_dimension_sweep` at the corresponding latent sizes. Computing `common` scores reads those fits' saved full-output predictions; it does not retrain or rerun inference.

The analysis manifest records all requested fits and their common-channel IDs before training. Completed entries reference checksummed metric files and shared prediction bundles. Reports consume only that membership. Analysis IDs distinguish specifications and participating fits; session/fold invocation subsets produce their own analysis views while still sharing fits. The `plot` stage reads saved results without loading a dataset; use `--analysis-id` when multiple saved revisions match.

Prediction bundles belong to their fit and have no independent model ID. Their readable horizon directory records checkpoint and test provenance, inference implementation signature and payload checksum. A different horizon request can add a bundle without fitting. Provenance or checksum conflicts raise an error; completed payloads are never silently replaced.

Component numbers follow fitting order within their group. Optional components appear only when enabled, including separate forward decoders, `residual_behaviour_decoder`, and the input-only `non_neural_behaviour_dynamics`. Each component records its role, inputs, target and original internal identifier in `component.json`. Each component has TensorBoard event files in `train/` and `validation/`; matching `epoch_<metric>` tags make shared metrics overlay in one TensorBoard panel. Persisted metric names omit masking-sentinel suffixes. Internal fitting attempts use cumulative epoch indices. The main residual behaviour decoder is disabled for NHP.

Completed fits and components are checksum-validated and restored without retraining. Fit completion is published immediately after saving the checkpoint, independently of prediction, scoring or rendering. An evaluation failure cannot invalidate a completed fit. The `evaluate` stage requires an existing completed fit and never trains. Only the documented artifact schema is read.

Fit indices have one canonical copy in the fit. Fitted preview excerpts have one canonical payload under the checkpoint-specific fold cache (or fit data when caching is disabled), referenced with checksums from provenance. Figures belong to analysis directories. Completed components are validated and reused after interruption; recognized incomplete component outputs are removed before fresh fitting under an exclusive fit lock. Incomplete prediction bundles are quarantined before regeneration.

The cache root contains reusable `session` and split-specific `fold` entries. Manifests and payload checksums validate reuse. Choose `--cache-mode reuse`, `rebuild` or `off` explicitly. Inspection figures and numeric excerpts remain separate from numerical feature identity. Normalization and learned behavior previews belong to their fitted checkpoint.

## Data and training conventions

M1 channels use their first nonempty spike dimension. Counts use half-open 50-ms bins labeled at their right edge. Task-plane fingertip position is interpolated there; target values use the latest native sample. Symmetric Gaussian smoothing uses sigma 50 ms, support ±200 ms and reflected segment boundaries. This is offline preprocessing that uses future spikes.

Five temporal folds use test block f, validation block f+1 modulo five and the other three blocks for training. Different split roles are separated by 1.8-second guards on both sides. Training and validation use independent 128-sample windows. Full experiments infer backward-difference velocities within segments and remove the unsupported first sample from every aligned signal.

Batch size is selected separately for each session and fold as `min(32, complete training windows, complete validation windows)`. The model configuration sets the maximum of 32; the effective value is recorded in `training_deviations.json`. This uses batch 32 whenever sufficient windows are available and smaller batches for shorter recordings. Model dimensions do not change the cached neural features.

Scoring preserves channelwise and aggregate CC, R² and MSE, with explicit undefined values. Fold means are computed within sessions before the cross-session mean and sample SEM. Saved artifacts drive plots and stage loss summaries; partial studies retain their actual fold/session counts. Successful startup does not establish convergence.


Each horizon has a `full` neural score using every channel fitted by that population. Experiments with multiple population scales also have a `common` score: the largest channel-ID intersection across all configured population groups, mapped into each fit's output-column order. Membership includes unfinished groups and is independent of execution order. Training populations retain their seeded ordering. Empty intersections, duplicate IDs and missing mappings are errors. Single-population experiments, including the latent-dimension sweep, emit only `full`.

Behaviour metrics are identical in `full` and `common` rows because behaviour dimensions do not change with neural scoring channels. `valid_<metric>_channels` counts channels with finite scores (or behaviour dimensions for behaviour metrics). Flat targets invalidate CC and R²; MSE normally remains defined. An aggregate metric is undefined if any included channel has an undefined score; no undefined channels are silently dropped.

### Saved-data previews

Use the public `preview` stage with the same `--experiment` YAML interface.
In an ignored `*.local.yaml`, configure `previews.source_run` with the absolute
path to a trusted saved run, `channel_ids`, and `window_ranges` in the data
module. The source must retain its checkpoint, fitted excerpts and validated
session/fold caches. This stage applies saved normalization maps without
refitting or training.

Shared data configuration controls presentation for all experiments: three
neural channels, 22-point titles, 18-point labels, and 16-point ticks and
outside legends. Each window has separate `neural`, `behavior`, and `input`
folders, one PNG per neural channel and paired x/y behavior or input panels,
ordered processing panels, and numerical excerpts. Coordinates use blue solid
x traces and orange dashed y traces with outside legends.
Fitted previews include both model normalization stages and explicitly label
learned behavior as a checkpoint-derived estimate. Revisions are published
atomically under the analysis's `previews/` directory. Independent
`preview` and preprocessing-only stages use their own source/presentation-
addressed analysis directories. Each revision
has an index and checksum
manifest; source artifacts are preserved. Keep regeneration settings and
outputs outside version control.


### Execution device

`train_nhp.sh` defaults to `--device auto`: among GPUs permitted by
`CUDA_VISIBLE_DEVICES`, select the most free memory, then lowest utilization.
Use `--device INDEX` or a full GPU UUID to select a physical device explicitly.
BRAID supports CUDA fitting through TensorFlow; the runner verifies a real
forward/backward operation and records the selected device in run metadata.
GPU selection is a snapshot, not a reservation against other processes.

Use `--device cpu --cpu-threads 8 --cpu-interop-threads 1` for CPU execution.
The corresponding YAML keys are `runtime.device`, `runtime.cpu_threads` and
`runtime.cpu_interop_threads`. Compute threads control TensorFlow intra-op
and numerical-library thread pools; inter-op threads control TensorFlow's
concurrent operations. GPU failures raise an error without a silent CPU
fallback. Preprocessing and preview stages do not require GPU selection.
