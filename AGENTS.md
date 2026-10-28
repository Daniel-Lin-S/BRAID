# Agent Instructions

## FIRST PRINCIPLE -- GRILL ME

You should always grill me on ALL technical details and design choices rather than making guessworks or silently setting default values by yourself.

Anti-example: I asked you to redesign the artifact figures, and you silently designed the content of one or several figures without checking details like x-y axis meanings, axis ticks, legends, plot timing with me.

## Break-point restoration

All experiments should store artifacts immediately after the results of one fit (the smallest unit of experiment, e.g., be one neural or non-neural fitting of a single random seed) are obtained to avoid losing results of already completed runs. However, failed, invalid, or interrupted run artifacts should be removed or quanrantined in the next invocation, starting a fresh experiment on that fit.

If the failure in one experiment is not fatal (definitely affects the following fits), please report failure and continue trying following experiment rather than stopping the whole process.

NEVER retrain a model on a data / fold / session that already have completion records under the same configurations. (changines in invocation parameters like number of GPU/CPUs used should NOT restart already completed experiments)

## Repository structure

Please place files in dedicated modules. 

NEVER stack flat files under the repository root or a module root. Create appropriate modules or submodules (or subfolders).

If a root / module has over 10 files, then you may need to examine whether breaking into subfolders is required. (this is not hard limit, there are situations where large number of files are required, e.g., in a data root with dozens of mat files belonging to different recordings, no appropriate partition exists)

## Documentation

DO NOT document about the development cycles. If an initial implementation A is changed to B, DONT document "B is implemented, rather than A". You should ONLY document the current version (B) of the code.

EVERY shell script should have usage documented at the top of the file.

DO NOT overwhelm documentations at public interfaces (shell scripts, major functions etc.) with heavy technial details -- whch make them unreadable. Only include the MOST BRIEF instructions that user need to know to use the interface or script, while leaving technical details to the documentations in the internal modules.

## Constraints

NEVER directly remove or modify a previous completed model fit artifacts (checkpoints, csv result tables, etc.). Appending to an incomplete run directory is allowed, but NOT modifying previous rows.

- But modifying comparison figures and tables are allowed.

NEVER reveal local paths (e.g., model checkpoints, dataset paths) in files tracked by GitHub.

## Experiment launch and hand-off

Whenever you are asked to launch a compaign of experiments, please detach and leave it running after smoke test and confirming one fit works successfully.

- DONT save artifacts on the one-fit audit, it is for you to check whether the training works.
- For multi-stage, large-scale pre-training, you DONT need to audit until the last stage. Auditing few updates of the first stage suffices.

## Experiment parallelisation

For the CPU tasks, lower-priority should be assigned to avoid the laucnhed python programme interfering with the server's core operational programmes.

You should parallelise independent neural fits whenever possible. Independent experiments should ALWAYS be launched without waiting other (or even a single stage of ) irrelevant experiments to finish.

- If there are multiple levels of parallelisms possible, select a level that doesn't create expensive cross-device communication overhead (e.g., if you assign batches across GPUs, this may have overhead)
- Ensure that you select the minimal level permitted by task independence that doesn't create communication overhead to maximise GPU usage and avoid a single bottleneck holding the entire compaign (e.g., multi-session experiment with weight sharing cannot be parallised at session-level, but session-independent model fitting experiments can)

For multi-stage experiment with cross-stage dependencies, DONT parallelise.

## Configurations and compaign identity

ONLY hash a model's fit using its scientific parameters (e.g., batch size, number of epochs, architectural parameters).

NEVER use invocation parameters (number of devices, GPU/CPU settings, etc.) that doesn't affect the scientific definition of model fits to generate hash identity.

NEVER use the entire python tree to hash the fit identity. If implementation of a scientific module changes and that has significant effect on the results (test by minimal validation), then reflect this scientific change in the configuration.

Changing the hashing mechanism or minor code implementations (e.g., file tree organisation) should NOT invalidate previous completed fits.

ALWAYS ensure that a compaign launch does NOT restart already fitted models with complete, valid artifacts and results. Scientific parameter changes (e.g., changing the latent dimension) can restart new experiment.

## Artifact saving and logging

The `logs` folder in this repository should ONLY record launch status, logs, failure records, etc. The disk space usage of `logs` should be kept minimal. The actual model fits, checkpoints, result figures, tables, should be saved in a separate artifact root specified by the user. If you are not sure about the artifact and log structure, always ASK ME rather than guesswork.

- Assign one log folder to each compaign, named by readable name + launch time. NEVER squash all log files under the root `logs`
- The purpose of the log is the clearly show the lifecycle (number of experiment finished, numbers remaining, numbers failed, etc.), make it clear what the compaign is currently doing, and record information on the failures. DONT include too many irrelevant information. 
- I recommend including a static file `status.json` with finished/pending/total/failure (etc.) job counts that updates as the experiment progresses for lifecycle tracking, and the current experiment status.
- The diskspcae usage of log files should be kept minimal.

In saved artifacts, reusable model fits and checkpoints should be placed in separate folders, identified by the run hash (with a short readable prefix, for example, at least specifying the model name). This is to ensure that a new compaign including some already fitted models does NOT restart repeated experiments on the already completed model fits.

Within the artifact root of a compaign, you should NOT save the same information twice. e.g., write new results in a new file with previous results copied.

All compaigns SHOULD share the SAME artifact / log saving logics and modules, rather than each one having its own artifact structure. Minor changes like file addition and field changes are allowed, but NEVER start a completely new artifact or log structure.

DO NOT trigger figure generation after every model fit completion, it should be triggered when ALL relevant model fits are complete.

## Plotting

- Font sizes must be LARGE and READABLE. This includes ALL texts on the figure (legend, axis ticks, title, etc.) You can use relatively smaller font sizes for minor labels like axis ticks, but they must still remain READABLE.
- use `seaborn` whenever possible rather than `matplotib`
- Scatter plots: set `set_xlim` and (or) `set_ylim` to focus on the main data range.
- Use lines thicker than default, but not too thick that distracts the main data.
- Use `bbox_to_anchor` and `loc'`  to ensure legends are not overlapping with the main figure, not going beyond figure. Legends should have no frame: `frameon=False`.
- DONT leave large gap between title and the main figure.
- Figures should be coloured, but figure legends and labels should make sense even in black-and-white prints. (for example, use different line types or hatches)
- Titles must be wrapped by `textwrap` with suitable width to prevent the title from exceeding boundaries, and title should be kept away from the main figure with at least one line's gap.
- Print a message after saving a figure, pointing to its file path. If a group of images are plotted, please ONLY print ONE message pointing to their common folder. 
- Proper units should be added to the x and y axes if applicable. (DONT add units if no unit)

While the plot layout may change across compaigns (e.g., different axis definitions, different number of panels, additional shaded areas, change of line style), the overall figure style (font size, legend position, colours used) should be consistent unless explicitly allowed to change. To that said -- all compaign's plotting codes should inherit from shared plotting modules rather than writing completely fresh modules.
