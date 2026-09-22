# NHP data and cache internals

This reference covers dataset ingestion, split-local features, and channel
selection. The public runbook is [NHP experiments](QUICK_START.md).

## Dataset and preprocessing

`source/experiments/nhp.py` reads Indy MATLAB 7.3 sessions for spike
experiments. It produces time-first spike counts `Y`, behavior `Z`, targets
`U`, timestamps `t`, and channel metadata. Waveform snippets are not
continuous LFP and are excluded from this path.

`source/experiments/nhp_lfp.py` pairs each `raw/indy_*.nwb` broadband
recording with its MATLAB task file. The configured ordered preprocessing
pipeline converts broadband samples to volts and produces LFP `Y` on the same
20-Hz behavior/target grid. Sample ticks must be continuous; isolated invalid
timestamps are repaired only when adjacent ticks establish the missing value.

LFP preprocessing steps are configured as ordered `plugin` and `parameters`
mappings. The tracked default uses SciPy polyphase resampling. Optional common
average referencing is enabled by inserting its plugin before resampling, so
step order and parameters are retained in scientific provenance.

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

Selected LFP preprocessing previews compare short native-rate broadband
excerpts with the final aligned LFP. Only the selected preview interval and
channels are read from the immutable NWB source.
