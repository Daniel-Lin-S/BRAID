# NHP experiments

Use this guide to configure and run an Indy reaching experiment. Internal
implementation contracts are linked below.

## Set up a machine

Obtain the Indy `.mat` spike/task files from the
[public reaching dataset](https://zenodo.org/records/3854034). Save the
record's [API metadata](https://zenodo.org/api/records/3854034) as
`record.json` beside the session files. LFP experiments additionally expect
matching broadband files at `raw/<session>.nwb` beneath the same dataset root.

Copy `assets/config/nhp/paths.example.yaml` to
`assets/config/nhp/paths.local.yaml`, then set the absolute dataset, cache,
artifact, and log roots. This local file is ignored by Git. Use Python 3.11
with `requirements.txt`; the validated lock file is
`assets/environment/requirements-gpu.lock.txt`.

## Select and run an experiment

| Manifest | Runs |
| --- | --- |
| `latent_dimension_sweep.yaml` | Full M1 population over latent dimensions 1–64 |
| `neural_population_sweep.yaml` | 25%, 50%, and 100% nested populations at dimensions 16 and 64 |
| `lfp_latent_dimension_sweep.yaml` | Broadband LFP latent sweep; selection is local |

Create a local experiment YAML that inherits both the tracked manifest and
`paths.local.yaml`:

```yaml
extends:
  - latent_dimension_sweep.yaml
  - ../paths.local.yaml
```

For LFP, use `lfp_latent_dimension_sweep.local.yaml` and keep its sessions
and folds in that local file:

```yaml
extends:
  - lfp_latent_dimension_sweep.yaml
  - ../paths.lfp.local.yaml
selection:
  sessions: [indy_20160624_03]
  folds: [0, 2, 4]
```

Run from the repository root:

```bash
bash scripts/train_nhp.sh --experiment assets/config/nhp/experiments/latent_dimension_sweep.local.yaml --dry-run
bash scripts/train_nhp.sh --experiment assets/config/nhp/experiments/latent_dimension_sweep.local.yaml --stage preprocess
bash scripts/train_nhp.sh --experiment assets/config/nhp/experiments/latent_dimension_sweep.local.yaml --stage fit --detach
bash scripts/train_nhp.sh --experiment assets/config/nhp/experiments/latent_dimension_sweep.local.yaml --stage plot
bash scripts/train_nhp.sh --experiment assets/config/nhp/experiments/latent_dimension_sweep.local.yaml --stage plot --previews --preview-session indy_20160407_02
```

`--session` and `--fold` restrict the run. `--device` selects `auto`,
`cpu`, or a GPU index/UUID; `--cache-mode` selects `reuse`, `rebuild`,
or `off`. Preview runs require selected sessions; `--preview-fold` and
`--preview-case` narrow them. `--tensorboard` enables component event
files for new fits. `--parallel-workers N` runs N sessions on
distinct GPUs with `device: auto`; the default is one worker.
`--dry-run` only resolves configuration.

Experiments with overlapping model fits can be run concurrently, but runtime profiles must set `shared_fit_wait_timeout_seconds` and
`shared_fit_poll_interval_seconds`. A session that encounters an active shared fit or prediction writer is deferred.

On resume, incomplete figures are repaired automatically, including
comparisons whose failed contributions later complete. Set
`runtime.figure_regeneration: all` to redraw comparison and training-history
figures after a style change. Use `runtime.preview_regeneration: all`
to redraw selected data previews. Both default to `incomplete`.

## Find results and recover from interruption

Each launch has a unique log directory containing `console.log`,
`experiment.log`, and one `sessions/<session>.log` per session.
`experiment.log` reports preprocessing, plotting, and fitting outcomes.
Fitting includes session, fold, and model events; the session log contains
detailed output and tracebacks.

Artifacts are rooted as follows:

```text
<artifact_root>/
  experiments/<model-settings>/<session>/fold_<number>/<fit-id>/
    checkpoints/  predictions/  components/  data_preview/fitted/
  analysis/<experiment>/<analysis-id>/
    metrics/  summaries/  plots/
<cache_root>/previews/preprocessing/<session>/fold_<number>/...
```

Completed fits are validated and reused. A model-specific fitting, prediction,
or evaluation failure preserves its completed work, allows independent models,
folds, and sessions to continue, and makes the invocation exit unsuccessfully
after reporting all failures. Re-run the same command to resume unfinished
work; do not delete completed artifacts.

## Internal implementation reference

These documents define internal behavior and artifact contracts; they are not
required to launch an experiment.

| Area | Internal reference | Owning modules |
| --- | --- | --- |
| Configuration | [NHP configuration](NHP_CONFIGURATION.md) | `configuration.py`, `contracts.py`, `braid_backend.py` |
| Dataset and cache | [NHP data](NHP_DATA.md) | `nhp.py`, `cache.py`, `windows.py`, `populations.py` |
| Fits and artifacts | [NHP artifacts](NHP_ARTIFACTS.md) | `artifacts.py`, `fitting.py`, `restart.py`, `analysis.py` |
| Scores and figures | [NHP analysis](NHP_ANALYSIS.md) | `evaluation.py`, `reporting.py`, `plots.py`, `history.py` |
| Launch, logs, and previews | [NHP runtime](NHP_RUNTIME.md) | `launcher.py`, `runner.py`, `runtime.py`, preview modules |
