# Multi-session training

Use the fixed campaign configuration `assets/config/nhp/experiments/multi_session_training.yaml`. Create an ignored `multi_session_training.local.yaml` beside it extending that file and `../paths.local.yaml`. The local paths file supplies `dataset_root`, `cache_root`, `artifact_root`, and `log_root`; select the environment through `BRAID_PYTHON`. The current local overlay follows this arrangement. No dataset or machine paths belong in tracked configuration.

```bash
bash scripts/train_multi_session.sh --experiment assets/config/nhp/experiments/multi_session_training.local.yaml --stage plan
bash scripts/train_multi_session.sh --experiment assets/config/nhp/experiments/multi_session_training.local.yaml --detach
bash scripts/train_multi_session.sh --experiment assets/config/nhp/experiments/multi_session_training.local.yaml --stage report
```

For the four-source sharing sensitivities, create an ignored
multi_session_sharing_sensitivity.local.yaml extending
multi_session_sharing_sensitivity.yaml and ../paths.local.yaml, and set
reference_plan to the completed campaign plan.json. Run the same script
with that local YAML. It validates completed reference fits before scheduling
two source fits and 72 adaptations.

The detached queue selects the four visible GPUs with the most free memory, including GPUs already in use, and starts one worker per selected GPU. Each worker has four CPU threads and TensorFlow memory growth. Unrelated processes are untouched. To hand over an authorized stopped coordinator, pass `--adopt-launch` with its launch directory; verified active workers finish without restart, and unoccupied slots are assigned by current free GPU memory. Repeating the launch resumes incomplete work and validates completed fits before reuse. A coordinator lock prevents duplicate queues for the same plan.

## Scientific design

Primary source training uses a single 32-state neural predictor, nonlinear neural decoders of width 256, and linear filtering and forward maps. It shares either `A1_fw` or `A1_fw + K1_fw`. Source counts are 4, 8, 16, and 33, with nested subsets paired across profiles and seed 42. Recorded future target positions drive the forecast input maps. Source behavior labels are excluded from the neural objective, scaling, validation, and checkpoint selection.

Each update samples 32 independent 128-step windows, choosing sessions uniformly and then windows uniformly within each chosen session. Private linear encoders map differing channel counts into one differentiable latent tensor. Forecasts return through the corresponding private neural decoder. Only represented sessions update private parameters and Adam state. Source validation averages session means equally. Chronological 80/20 splits retain the existing guards and split-local spike smoothing. Training uses Adam 0.001, at most 10,000 updates, validation every 25 updates, patience 10 checks, and restored best weights.

The following sessions never enter source training or source validation: `indy_20160921_01`, `indy_20160627_01`, `indy_20161220_02`, and `indy_20170131_02`. The remaining 33 Indy spike sessions are sources. Coordinate widths and finite training-only scales are checked explicitly; M1 selection, 20 Hz sampling, and native forecasting alignment are preserved.

Target BRAID retains preprocessing (`n_pre=150`), `n1=n2=32`, neural decoder width 256, behavior decoder width 64, and linear dynamics/encoders. Exactly the imported maps are frozen. Folds 0, 2, and 4 use nested randomized training/validation windows at 5%, 20%, and 80% of all session labels. Complete-window and native batch trimming determine the saved realized percentage; 80% uses the full eligible original roles before that trimming. Test windows are identical across conditions. Normalization and preprocessing see only selected training data.

Scratch controls share preprocessing, subset indices, initialization seeds, architecture, future-input policy, batch rules, and optimization limits with adaptations. Each target component receives at most 2,500 epochs with Adam 0.001, patience 10, and restored best weights. Historical single-session results are contextual references because their forecast-input policy differs. Source costs and target allowances/consumption are tabulated separately; there is no campaign-specific HPO.

## Analyses and outputs

| Analysis identifier | Main comparison |
|---|---|
| multi_session_filter_sensitivity | Four-source AK versus A+AK single-stage neural training |
| multi_session_residual_sharing_sensitivity | Four-source single-stage A+AK versus two-stage A+AK with shared residual dynamics |
| `multi_session_training` | Source scaling and target-label scaling for the two sharing profiles |
| `multi_session_training_capacity_sensitivity` | 32 → 32+32 versus 64 → 64+64 at 33 sources |
| `multi_session_training_supervision_sensitivity` | Neural-only versus behavior-supervised source training at 33 sources |

Behavior-supervised sources use the five-component BRAID procedure: neural preprocessing, preprocessing behavior readout, behavior-relevant dynamics, neural readout, and residual neural dynamics. Each component receives at most 2,000 updates; unused updates are not redistributed. The supplementary supervision comparison is exactly `A1_fw + K1_fw` versus `A1_fw + K1_fw + Cz` within behavior-supervised training.

The plan contains 13 source fits, 468 adaptations, 72 scratch controls, and 36 reusable target preprocessing fits. Analyses reference a common fit store, with independent memberships and completion counts:

```text
<artifact_root>/multi_session_training/
  plans/<plan_id>/plan.json
  fits/<fit_id>/
  source_components/<component_id>/
  target_components/<fit_id>/
<artifact_root>/experiments/
  BRAID_matched_scratch/<fit_id>/
  BRAID_matched_preprocessing/<fit_id>/
  BRAID_preprocessing_predictions/<cache_id>/
  target_components/<fit_id>/
<artifact_root>/analysis/<analysis_id>/<plan_id>/
  manifest.json
  latest_report.json
  reports/<snapshot_id>/
<log_root>/multi_session_training/<launch_id>/
  launch.json, devices.json, status.json, coordinator.log
  jobs/<kind>/<session>/<readable_label>__<fit_id>.log
```

Scratch fits and their preprocessing dependencies live in the shared experiments store. Each analysis manifest records the resolved fit directory. Other analyses can load `model.p`, `identity.json`, `selection.npz`, `metrics.json`, and the checksum-validated `complete.json` directly; storage location and analysis membership do not change fit identity.

Fit identities include scientific settings, source and target selections, seeds, implementation/dependency versions, shared-map names, and parent-checkpoint checksums. GPU and worker allocations do not enter scientific identities. Source implementation identities exclude target-only integration changes, preserving source checkpoints when target execution is optimized. Completed fits have checksum-validated `complete.json` records. Interrupted attempts are quarantined; separately completed components survive fit retries. Nonfatal worker/reporting failures do not stop independent fits.

Source records retain training-only normalization, original window indices, sampled windows per update, exposure counts, histories, best weights, final optimizer state, and component completion checksums. Target records retain selected indices, realized fractions, normalization, native checkpoints, component dependencies, allowed/consumed budgets, test predictions, masks, and per-dimension metrics.

Reports retain CC, R², and MSE at horizons 1, 2, 4, 8, 16, and 32. `raw_metrics.csv`, `per_dimension.csv`, and `session_metrics.csv` retain fit-level, coordinate-level, and session-level results. `summary.csv` averages folds/repeats within each session before computing the four-session mean and SEM, with validity counts. `paired_gain.csv` and `paired_gain_summary.csv` compare matched scratch controls. `source_costs.csv` and `target_costs.csv` expose budgets and reusable component IDs. Manifests retain failure/completion accounting even before enough results exist to draw a figure.

The two four-source sensitivity roots retain separate manifests and nested
tables/, figures/target_transfer/{neural,behavior}/, and
figures/source_validation/ output. Source-validation figures show CC on the
same four source sessions and macro-validation loss over each fit's full
source set. The residual comparison holds the total source-update ceiling at
10,000 but allocates 5,000 per stage; it compares training packages because
stage-1 exposure and architecture differ.

Primary CC figures use logarithmic source-count and target-label axes, the prescribed ticks, profile colors, and dashed scratch references. Sensitivity figures encode exactly two conditions in color with separate matching-sharing panels. Neural/behavior outcomes and horizons have separate figures. Capacity-matched scratch results remain in tables. Report snapshots are immutable and the latest-report pointer advances as work completes.

## Execution efficiency

Frozen target preprocessing predictions are cached once per exact checkpoint and input-array content, shape, and dtype. Warm-cache fits load weight containers without reconstructing the preprocessing graph. Main-stage initialization is seeded immediately before fitting, so cache hits and misses remain paired. Native checkpoints are saved from shallow model-container snapshots, preserving live graphs and avoiding save/reconstruct cycles; current forecast horizons and batch sizes survive serialization.

The scheduler keeps source jobs sharing private preprocessing from competing for GPU slots. Periodic reporting runs in a separate CPU process. Detached coordinators read the already published plan instead of rescanning all sessions. Training losses, horizons, optimization limits, split indices, and stopping rules are unchanged.

## Implementation layout

`experiments.multisession` separates data selection, job planning, source fitting, target fitting, training, immutable storage, reporting, and queue execution. It uses native BRAID regression maps and forecast alignment. Optional frozen-module and preprocessing-injection interfaces preserve ordinary single-session behavior when absent.

The broader experiment tree groups adapters under `datasets`, model integration under `models`, orchestration under `execution`, artifact logic under `storage`, metrics/reporting under `reports`, and rendering under `visualization`. Root-level configuration/contracts remain lightweight shared interfaces. A lazy compatibility registry preserves historical Python imports, saved plugin paths, and module launch commands without duplicating implementations.
