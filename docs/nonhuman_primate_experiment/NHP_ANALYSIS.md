# NHP evaluation and figure internals

This reference covers saved-result evaluation and rendering. The public
runbook is [NHP experiments](QUICK_START.md).

## Evaluation

`source/experiments/evaluation.py` scores held-out forecasts by horizon for
neural and behavior targets. It stores channelwise and aggregate CC, R², MSE,
and constant-predictor baseline MSE. CC and R² are undefined for flat targets;
an aggregate remains undefined when an included channel is undefined.

`full` neural scoring uses all channels fitted by a population. `common`
scoring uses the configured populations' channel-ID intersection. Behavior
scores are the same across those neural scoring selections. Aggregation first
averages folds within a session, then reports the cross-session mean and sample
SEM.

`model_summary.py` writes a status row for every analysis member. It preserves
explicit pending and failed states and does not fabricate scores for them.

## Reports and comparison figures

`reporting.py` aggregates only completed member metrics. `plots.py` expands
configured curve definitions and renders each comparison after all of its
contributing members have been attempted in the current invocation. Reused
validated results count as attempted.

Pending contributions defer a comparison silently. Failed contributions are
shown as annotated gaps or reduced contribution counts and produce one warning
that names the failed members. A comparison with no finite results is skipped.
Unexpected missing completed metrics are rendering errors.

`history.py` renders fit-owned component histories. `presentation.py` owns
figure style, metric labels, and output formatting. Plotting does not fit or
modify a completed checkpoint. Completed components are scanned on resume;
missing or damaged history figures are repaired while valid figures remain
unchanged. `runtime.figure_regeneration: all` explicitly redraws every
expected component-history figure after a style change.
