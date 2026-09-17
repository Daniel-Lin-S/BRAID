# Agent Instructions

## Documentation

DO NOT document about the development cycles. If an initial implementation A is changed to B, DONT document "B is implemented, rather than A". You should ONLY document the current version (B) of the code.

EVERY shell script should have usage documented at the top of the file.

DO NOT overwhelm documentations at public interfaces (shell scripts, major functions etc.) with heavy technial details -- whch make them unreadable. Only include the MOST BRIEF instructions that user need to know to use the interface or script, while leaving technical details to the documentations in the internal modules.

## Constraints

NEVER directly remove or modify a previous completed run artifacts. Appending to an incomplete run directory is allowed, but NOT modifying previous rows.

NEVER reveal local paths (e.g., model checkpoints, dataset paths) in files tracked by GitHub.
