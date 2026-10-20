# Single-session Causal Transformer comparison

BRAID accepts `causal_transformer` as either stage's `state_transition.architecture`. It uses 32-wide latent tokens, six neural-history blocks, six stimulus blocks, six causal self-attention blocks, eight heads, 0.2 attention/MLP dropout, learned position embeddings, and sinusoidal time embeddings. The two target coordinates use one linear token projection. BRAID's neural encoder and observation decoders remain in use.

A 128-bin independent window has four 32-bin query chunks, each with up to 96 preceding neural-history bins. Missing early history uses learned start tokens and attention masks. The model computes each chunk's observed latents in parallel, then makes direct horizon queries with learned horizon embeddings and the known future target coordinates. Forecast queries attend causally to observed latents and never read future neural observations. In stage two, `include_stage_one_state: true` (the default) projects stage-one states separately into query tokens. The comparison always sets `forecast.use_observed_input_in_dynamics: true`.

Use a local experiment YAML with dataset paths. Set the campaign root to a direct child of the configured `log_root`; the campaign stores checkpoints, metrics, TensorBoard events, console output, and status together there. The run validates and reuses matching completed seed-42 linear fits. Set `PYTHONPATH=source` from the repository root.

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

A detached launch freezes Python source and YAML defaults under the campaign
root and records their checksum in `launch.json`. Its workers verify that
snapshot before fitting; later source edits do not change the live campaign.

The audit fits one full training batch for one epoch and runs the production prediction path with all six evaluation horizons. It uses a temporary directory and saves no persistent model or fit artifacts. Set `CUDA_VISIBLE_DEVICES` to select its GPU. For a run, supply one GPU ID per worker; each worker sees only its assigned GPU and has a 10 GiB TensorFlow memory limit. The detached run validates reusable preprocessing and control fits, then runs missing fits for five dynamics cases across four Indy sessions, folds 0/2/4, and seeds 42/43/44. Each independent job commits a completion marker after its model, metrics, and predictions are durable. Failed jobs remain labeled and can be retried with the same command. The comparison expects 180 fits total, including the 12 reused linear fits.

| Output | Contents |
| --- | --- |
| `preprocessing/<target>/` | Selected indices, a validated preprocessing checkpoint, and completion checksums. |
| `fits/<case>__<target>/` | Identity, selection, resolved arguments, native checkpoint, component records, test predictions, per-horizon metrics, and completion checksums. |
| `raw_scores.csv` | Session, fold, seed, case, target, horizon, metric, and score. |
| `session_scores.csv` | Mean score over nine fold/seed fits within each session. |
| `paired_gains.csv` | Matched per-fit Transformer minus control gains; MSE uses control minus Transformer. |
| `session_paired_gains.csv` | Mean paired gain over nine fold/seed pairs within each session. |
| `<target>_<metric>.png` | Six log-horizon curves with mean ± SEM across the four sessions. |
| `report_complete.json` | Checksums for completed tables and figures. |

Detached runs write console output to `<campaign-root>/campaign.log`. The root's `launch.json` updates every 15 seconds with finished, running, interrupted, failed, pending, and remaining counts for preprocessing and fits. TensorBoard events live under `<campaign-root>/events/`. Remaining includes running, interrupted, failed, and pending jobs; `reused_finished` counts validated seed-42 linear fits.

The cases are linear dynamics, a one-hidden-layer 64-unit ReLU MLP, native LSTM defaults, stage-one Transformer with linear stage two, and Transformer in both stages. All use `n1=n2=32`, `n_pre=150`, the same BRAID fitting schedule, and evaluation horizons 1, 2, 4, 8, 16, and 32 bins. The report is descriptive at the session level.
