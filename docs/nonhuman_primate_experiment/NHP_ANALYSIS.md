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

The structure sweep writes `structure_raw.csv` with one case/session/fold/
horizon/target/metric row, including failed and pending members. Its
`structure_summary.csv` retains the fold-then-session mean and SEM with
member counts. Six aggregate horizon figures show the three dynamics choices
in separate panels. Values beyond median ± 2 IQR are labeled at the plot
boundary and remain unchanged in the tables.

`model_summary.py` writes a status row for every analysis member. It preserves
explicit pending and failed states and does not fabricate scores for them.

## Reports and comparison figures

`reporting.py` aggregates only completed member metrics. `plots.py` expands
configured curve definitions and renders each comparison after all of its
contributing members have been attempted in the current invocation. Reused
validated results count as attempted.

Curves may assign an x-axis parameter, grouped lines, and target panels.
Grouped lines use color, line style, and markers together. The latent-dimension
sweep plots both latent dimension by horizon and horizon by latent dimension
for each configured metric.

Pending contributions defer a comparison silently. Failed contributions are
shown as annotated gaps or reduced contribution counts and produce one warning
that names the failed members. A comparison with no finite results is skipped.
Unexpected missing completed metrics are rendering errors. Comparison
figures are replaced when their recorded dependency state changes.

Local `plotting.outliers` rules may select one exact `member` or a `slice`
containing `session` and `fold`, which applies to every resolved case.
Pending selections are deferred. A finite selected value is excluded only when
it lies outside the fold-summary range from nonselected folds; otherwise it is
included normally with one diagnostic warning.

`history.py` renders fit-owned component histories. `presentation.py` owns
figure style, metric labels, and output formatting. Plotting does not fit or
modify a completed checkpoint. Completed components are scanned on resume;
missing or damaged history figures are repaired while valid figures remain
unchanged. `runtime.figure_regeneration: all` explicitly redraws every
expected component-history figure after a style change.
