# NHP experiments

Experiment manifests select separate data, model/training, evaluation and plotting configurations. Machine paths and private validation settings belong in ignored local YAML files.

## Dataset and local setup

Obtain the Indy `.mat` spike/task files from the [public reaching dataset](https://zenodo.org/records/3854034). Save the record's [API metadata](https://zenodo.org/api/records/3854034) as `record.json` alongside the session files.

Copy `assets/config/nhp/paths.example.yaml` to `assets/config/nhp/paths.local.yaml`. Set its absolute dataset, cache, artifact and log directories. Set the optional Python interpreter and runtime preferences there. These machine settings are ignored by Git.

Use Python 3.11 with the dependencies in `requirements.txt`; the exact validated package lock is `assets/environment/requirements-gpu.lock.txt`. The launcher selects legacy Keras and requires a working GPU for fitting.

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
  sessions/<session>/fold_<number>/<case>-<identity>/experiment.log
```

Every invocation has a unique directory, including repeated invocations of the same experiment. Detached launches print the worker PID and absolute log location. Existing running jobs retain their original log destinations.

Model artifacts use `<artifact_root>/<experiment>/<session>/fold_<number>/<identity>/`. They include resolved settings, source/cache references, selected channels, fit indices, component histories/checkpoints, predictions, channelwise metrics and completion/failure status. Completed compatible results are validated before reuse; completed artifacts are not overwritten.

The cache root contains reusable `session` and split-specific `fold` entries. Manifests and payload checksums validate reuse. Choose `--cache-mode reuse`, `rebuild` or `off` explicitly. Inspection figures and numeric excerpts remain separate from numerical feature identity. Normalization and learned behavior previews belong to their fitted checkpoint.

## Data and training conventions

M1 channels use their first nonempty spike dimension. Counts use half-open 50-ms bins labeled at their right edge. Task-plane fingertip position is interpolated there; target values use the latest native sample. Symmetric Gaussian smoothing uses sigma 50 ms, support ±200 ms and reflected segment boundaries. This is offline preprocessing that uses future spikes.

Five temporal folds use test block f, validation block f+1 modulo five and the other three blocks for training. Different split roles are separated by 1.8-second guards on both sides. Training and validation use independent 128-sample windows. Full experiments infer backward-difference velocities within segments and remove the unsupported first sample from every aligned signal.

Batch size is selected separately for each session and fold as `min(32, complete training windows, complete validation windows)`. The model configuration sets the maximum of 32; the effective value is recorded in `training_deviations.json`. This uses batch 32 whenever sufficient windows are available and smaller batches for shorter recordings. Model dimensions do not change the cached neural features.

Scoring preserves channelwise and aggregate CC, R² and MSE, with explicit undefined values. Fold means are computed within sessions before the cross-session mean and sample SEM. Saved artifacts drive plots and stage loss summaries; partial studies retain their actual fold/session counts. Successful startup does not establish convergence.


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
atomically alongside their fold cache: preprocessing uses `previews/`, and
fitting uses `fitted_previews/<checkpoint checksum>/`. The independent
`preview` stage uses the separate cache-root `previews/` tree. Each revision
has an index and checksum
manifest; source artifacts are preserved. Keep regeneration settings and
outputs outside version control.
