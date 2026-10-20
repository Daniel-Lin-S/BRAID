"""Define the matched single-session NHP dynamics comparison.

Inputs are session, fold, seed, and dynamics-case identifiers. Outputs are
planned job keys and paths beneath a supplied campaign root. Fit directories
contain identity.json, metrics.json, checkpoint payloads, and complete.json.
"""

from __future__ import annotations

from pathlib import Path


SESSIONS = (
    "indy_20160921_01", "indy_20160627_01",
    "indy_20161220_02", "indy_20170131_02",
)
FOLDS = (0, 2, 4)
SEEDS = (42, 43, 44)
FRACTION = 0.80
LATENT = 32
HORIZONS = (1, 2, 4, 8, 16, 32)
REFERENCE_CASE = "linear"
REFERENCE_SEED = 42
CONTROL_CASES = (REFERENCE_CASE, "mlp", "lstm")
CANDIDATE_CASES = ("transformer_stage1", "transformer_both")
CASES = CONTROL_CASES + CANDIDATE_CASES
CASE_LABELS = {
    "linear": "Linear",
    "mlp": "ReLU MLP (1×64)",
    "lstm": "LSTM",
    "transformer_stage1": "Transformer stage 1",
    "transformer_both": "Transformer both stages",
}
METRICS = ("cc", "r2", "mse")
TARGETS = ("neural", "behavior")
MLP_WIDTH = 64
CPU_NICE = 19
DEFAULT_WORKERS = 2
GPU_MEMORY_LIMIT_MIB = 10240
IQR_MULTIPLIER = 3.0
BIN_MS = 50
LEGACY_CONTROL_TARGET_DIGEST = (
    "667b0c5c302c642dbff54e3218c8bbd597d59969b60e776664ddfc47f609f147"
)
CURRENT_CONTROL_TARGET_DIGEST = (
    "3d3a4fe536f3c4f82ac508d6f9815ead819c19485dab30f502339dc089c86b74"
)
LEGACY_TRANSFORMER_DIGEST = (
    "8ed7a807a82371c8de1c41b7abd19ad11b643e6ea5883b85ef018b7f765b8f56"
)


def target_key(session: str, fold: int, seed: int) -> str:
    """Return the stable key for one matched target fit."""
    return f"{session}__fold{fold}__seed{seed}"


def tasks() -> list[tuple[str, int, int]]:
    """Enumerate all planned session, fold, and seed tuples."""
    return [
        (session, fold, seed)
        for session in SESSIONS for fold in FOLDS for seed in SEEDS
    ]


def preprocessing_directory(root: Path, key: str) -> Path:
    """Return one campaign-local preprocessing directory."""
    return root / "preprocessing" / key


def fit_directory(root: Path, case: str, key: str) -> Path:
    """Return one campaign-local dynamics fit directory."""
    return root / "fits" / f"{case}__{key}"
