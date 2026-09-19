# NHP data and cache internals

This reference covers dataset ingestion, split-local features, and channel
selection. The public runbook is [NHP experiments](QUICK_START.md).

## Dataset and preprocessing

`source/experiments/nhp.py` reads Indy MATLAB 7.3 sessions. It produces
time-first spike counts `Y`, behavior `Z`, targets `U`, timestamps `t`,
and channel metadata. Waveform snippets are not continuous LFP and are
excluded from this experiment path.

For each fold, the dataset applies configured spike smoothing within temporal
segments, creates training/validation/test roles, and removes guard regions at
role boundaries. When enabled, velocity is inferred by backward differences
inside a segment; the first unsupported sample is removed from all aligned
signals in that segment. `windows.py` only returns complete windows within a
single role and segment.

The selected M1 channel ordering is shared across sessions. `populations.py`
uses this ordering to construct nested populations and maps full-output
predictions to a common channel intersection when required for comparison.
Duplicate channel IDs, empty intersections, or missing output mappings are
errors.

## Cache

`source/experiments/cache.py` stores content-addressed session and fold
entries under the configured cache root. A manifest records its identity and
payload checksums. `reuse` accepts only a validated matching entry, `rebuild`
creates a replacement entry, and `off` bypasses persistent cache reuse.

Preview selection and rendering are deliberately outside numerical feature
identity. Cached fold arrays remain the single source for fitting, scoring,
and preview input selection.
