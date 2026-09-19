# NHP configuration internals

This reference covers resolved YAML configuration and the BRAID adapter. The
public launch instructions are in [NHP experiments](QUICK_START.md).

## Configuration modules

`source/experiments/configuration.py` resolves an experiment manifest and its
four module files: `data`, `model`, `evaluation`, and `plotting`. YAML
files can use `extends`; parents merge in listed order and the declaring file
owns the base directory for inherited references. A `.local.yaml` suffix has
no loading behavior; it is only a Git-ignore convention.

The experiment manifest owns its name, seed, suite, selected sessions/folds,
plugin references, and module references. The data module owns feature and
preprocessing settings, the model module owns fitting settings, the evaluation
module owns forecast horizons and scoring settings, and the plotting module
owns comparison selections and presentation settings. Machine-only `paths`
and `runtime` settings may be inherited through a local manifest.

The launcher accepts `fit`, `preprocess`, `preview`, `evaluate`, and
`plot`. `evaluate` requires an existing completed fit. `plot` reads a
saved analysis and does not load the dataset. CLI values override resolved
runtime and cache settings for that invocation.

## BRAID adapter

`source/experiments/braid_backend.py` adapts time-first experiment arrays to
the native BRAID model. `build_cases()` expands configured latent sizes and
population scales. It sets `n1` to the configured cap or `nx`, whichever is
smaller; remaining latent dimensions form BRAID's residual `n2` stage.

The adapter resolves a session/fold batch cap before fitting and records any
deviation from the configured batch limit. Adapter `fit`, `save`, `load`,
and `predict` methods own conversions between experiment-array and native
BRAID orientations. Public model identity comes from the adapter's
`model_name`, not the Python class name.
