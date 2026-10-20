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