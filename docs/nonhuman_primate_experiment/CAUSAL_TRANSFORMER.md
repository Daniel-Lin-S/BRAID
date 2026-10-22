# Single-session Causal Transformer comparison

BRAID accepts `causal_transformer` as either stage's `state_transition.architecture`. It uses 32-wide latent tokens, six neural-history blocks, six stimulus blocks, six causal self-attention blocks, eight heads, 0.2 attention/MLP dropout, learned position embeddings, and sinusoidal time embeddings. The two target coordinates use one linear token projection. BRAID's neural encoder and observation decoders remain in use.

A 128-bin independent window has four 32-bin query chunks, each with up to 96 preceding neural-history bins. Missing early history uses learned start tokens and attention masks. The model computes each chunk's observed latents in parallel, then makes direct horizon queries with learned horizon embeddings and the known future target coordinates. Forecast queries attend causally to observed latents and never read future neural observations. In stage two, `include_stage_one_state: true` (the default) projects stage-one states separately into query tokens. The comparison always sets `forecast.use_observed_input_in_dynamics: true`.

Use a local experiment YAML specifying separate `artifact_root`, `cache_root`, and `log_root` paths. Set `--root` to a direct child of `log_root`. It contains compact launch/status metadata, console logs, and failure records. Fits and analyses use the same shared storage modules and outer layout as the other NHP experiments. The run validates and reuses matching completed seed-42 linear fits. Set `PYTHONPATH=source` from the repository root.

```bash
export BRAID_COMPARISON_ROOT="$PWD/logs/causal_transformer_sequence"
python -m experiments.campaign_runner \
  --study experiments.nhp_dynamics_study \
  --experiment "$BRAID_EXPERIMENT" --root "$BRAID_COMPARISON_ROOT" --audit
python -m experiments.campaign_runner \
  --study experiments.nhp_dynamics_study \
  --experiment "$BRAID_EXPERIMENT" --root "$BRAID_COMPARISON_ROOT" \
  --run --workers 2 --gpus 0 1 --detach
python -m experiments.campaign_runner \
  --study experiments.nhp_dynamics_study \
  --experiment "$BRAID_EXPERIMENT" --root "$BRAID_COMPARISON_ROOT" --report
python -m experiments.campaign_status \
  --study experiments.nhp_dynamics_study --root "$BRAID_COMPARISON_ROOT"
```

A detached launch freezes Python source and YAML defaults under the analysis's `provenance/<source-checksum>/` directory. `launch.json` records the snapshot, artifact root, and analysis directory. Workers verify the snapshot before fitting; source checksums do not enter scientific fit IDs.

The audit fits one full training batch for one epoch and runs the production prediction path with all six evaluation horizons. It uses a temporary directory and saves no persistent model or fit artifacts. Set `CUDA_VISIBLE_DEVICES` to select its GPU. For a run, supply one GPU ID per worker; each worker sees only its assigned GPU and has a 10 GiB TensorFlow memory limit. The detached run validates reusable preprocessing and control fits, then runs missing fits for five dynamics cases across four Indy sessions, folds 0/2/4, and seeds 42/43/44. Each independent job commits a completion marker after its model, metrics, and predictions are durable. Failed jobs remain labeled and can be retried with the same command. The comparison expects 180 fits total, including the 12 reused linear fits.

Paths below are relative to the configured artifact root. `launch.json` links to the analysis manifest, whose members reference the shared fit locations.

| Output | Contents |
| --- | --- |
| `experiments/BRAID_<settings-hash>/<session>/fold_<k>/<fit-id>/` | Reusable fit identity, selection, resolved arguments, native checkpoint, predictions, metrics, and completion checksums. Existing completed fit IDs and payloads are retained. |
| `<fit>/components/` | Native stage checkpoints, histories, and completion records. TensorBoard streams stay beside the fit; historical streams use `<fit>/diagnostics/`. |
| `experiments/BRAID_preprocessing_predictions/<cache-id>/` | Reusable frozen preprocessing predictions. |
| `analysis/dynamics_model_comparison/<analysis-id>/manifest.json` | Scientific comparison specification and shared fit membership. |
| `<analysis>/tables/raw_scores.csv` | Session, fold, seed, case, target, horizon, metric, and score. |
| `<analysis>/tables/session_scores.csv` | Mean score over nine fold/seed fits within each session. |
| `<analysis>/tables/paired_gains.csv` | Matched per-fit Transformer minus control gains; MSE uses control minus Transformer. |
| `<analysis>/tables/session_paired_gains.csv` | Mean paired gain over nine fold/seed pairs within each session. |
| `<analysis>/figures/<target>_<metric>.png` | Six log-horizon curves with mean ± SEM across the four sessions, shared plot fonts, and legends outside the axes. |
| `<analysis>/report_complete.json` | Checksums for completed tables and figures. |

Detached runs write console output to `<launch-root>/campaign.log`. `launch.json` updates every 15 seconds with finished, running, interrupted, failed, pending, and remaining counts. Remaining includes running, interrupted, failed, and pending jobs; `reused_finished` includes validated references from other launches. The coordinator publishes an inactive final status on exit.

To redraw figures from validated saved tables, add `--redraw-figures` to the `--report` command. This updates only derived figure renderings and their report checksums; fits and numeric tables are unchanged. Audits use temporary storage and publish no persistent fit artifacts.

The cases are linear dynamics, a one-hidden-layer 64-unit ReLU MLP, native LSTM defaults, stage-one Transformer with linear stage two, and Transformer in both stages. All use `n1=n2=32`, `n_pre=150`, the same BRAID fitting schedule, and evaluation horizons 1, 2, 4, 8, 16, and 32 bins. The report is descriptive at the session level.
