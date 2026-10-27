# Agent Instructions

## FIRST PRINCIPLE -- GRILL ME

You should always grill me on ALL technical details and design choices rather than making guessworks or silently setting default values by yourself.

Anti-example: I asked you to redesign the artifact figures, and you silently designed the content of one or several figures without checking details like x-y axis meanings, axis ticks, legends, plot timing with me.

## Break-point restoration

All experiments should store artifacts immediately after the results of one fit (the smallest unit of experiment, e.g., be one neural or non-neural fitting of a single random seed) are obtained to avoid losing results of already completed runs. However, failed, invalid, or interrupted run artifacts should be removed or quanrantined in the next invocation, starting a fresh experiment on that fit.

If the failure in one experiment is not fatal (definitely affects the following fits), please report failure and continue trying following experiment rather than stopping the whole process.

NEVER retrain a model on a data / fold / session that already have completion records under the same configurations. (changines in invocation parameters like number of GPU/CPUs used should NOT restart already completed experiments)

## Documentation

DO NOT document about the development cycles. If an initial implementation A is changed to B, DONT document "B is implemented, rather than A". You should ONLY document the current version (B) of the code.

EVERY shell script should have usage documented at the top of the file.

DO NOT overwhelm documentations at public interfaces (shell scripts, major functions etc.) with heavy technial details -- whch make them unreadable. Only include the MOST BRIEF instructions that user need to know to use the interface or script, while leaving technical details to the documentations in the internal modules.

## Constraints

NEVER directly remove or modify a previous completed run artifacts. Appending to an incomplete run directory is allowed, but NOT modifying previous rows.

NEVER reveal local paths (e.g., model checkpoints, dataset paths) in files tracked by GitHub.

## Experiment launch and hand-off

Whenever you are asked to launch a compaign of experiments, please detach and leave it running after smoke test and confirming one fit works successfully.

- DONT save artifacts on the one-fit audit, it is for you to check whether the training works.
- For multi-stage, large-scale pre-training, you DONT need to audit until the last stage. Auditing few updates of the first stage suffices.

## Experiment parallelisation

For the CPU tasks, lower-priority should be assigned to avoid the laucnhed python programme interfering with the server's core operational programmes.

You should parallelise independent neural fits whenever possible. Independent experiments should ALWAYS be launched without waiting other (or even a single stage of ) irrelevant experiments to finish.

For multi-stage experiment with cross-stage dependencies, DONT parallelise.

## Configurations and compaign identity

ONLY hash a model's fit using its scientific parameters (e.g., batch size, number of epochs, architectural parameters).

NEVER use invocation parameters (number of devices, GPU/CPU settings, etc.) that doesn't affect the scientific definition of model fits to generate hash identity.

NEVER use the entire python tree to hash the fit identity. If implementation of a scientific module changes and that has significant effect on the results (test by minimal validation), then reflect this scientific change in the configuration.

Changing the hashing mechanism or minor code implementations (e.g., file tree organisation) should NOT invalidate previous completed fits.

ALWAYS ensure that a compaign launch does NOT restart already fitted models with complete, valid artifacts and results. Scientific parameter changes (e.g., changing the latent dimension) can restart new experiment.

## Artifact saving and logging

The `logs` folder in this repository should ONLY record launch status, logs, failure records, etc. The disk space usage of `logs` should be kept minimal. The actual model fits, checkpoints, result figures, tables, should be saved in a separate artifact root specified by the user. If you are not sure about the artifact and log structure, always ASK ME rather than guesswork.

In saved artifacts, reusable model fits and checkpoints should be placed in separate folders, identified by the run hash. This is to ensure that a new compaign including some already fitted models does NOT restart repeated experiments on the already completed model fits.

Within the artifact root of a compaign, you should NOT save the same information twice. e.g., write new results in a new file with previous results copied.

All compaigns SHOULD share the SAME artifact / log saving logics and modules, rather than each one having its own artifact structure. Minor changes like file addition and field changes are allowed, but NEVER start a completely new artifact or log structure.

DO NOT trigger figure generation after every model fit completion, it should be triggered when ALL relevant model fits are complete.

## Plotting

- use `seaborn` whenever possible rather than `matplotib`
- Scatter plots: set `set_xlim` and (or) `set_ylim` to focus on the main data range.
- Use lines thicker than default, but not too thick that distracts the main data.
- Use `bbox_to_anchor` and `loc'`  to ensure legends are not overlapping with the main figure, not going beyond figure. Legends should have no frame: `frameon=False`.
- Figures should be coloured, but figure legends and labels should make sense even in black-and-white prints. (for example, use different line types or hatches)
- Titles must be wrapped by `textwrap` with suitable width to prevent the title from exceeding boundaries, and title should be kept away from the main figure with at least one line's gap.
- Print a message after saving a figure, pointing to its file path. If a group of images are plotted, please ONLY print ONE message pointing to their common folder. 
- Proper units should be added to the x and y axes if applicable. (DONT add units if no unit)

While the plot layout may change across compaigns (e.g., different axis definitions, different number of panels, additional shaded areas, change of line style), the overall figure style (font size, legend position, colours used) should be consistent unless explicitly allowed to change. To that said -- all compaign's plotting codes should inherit from shared plotting modules rather than writing completely fresh modules.
