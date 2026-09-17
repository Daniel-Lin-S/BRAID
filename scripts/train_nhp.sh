#!/usr/bin/env bash
# Usage: train_nhp.sh --experiment YAML [--stage STAGE]
# Device: --device auto (default), cpu, physical GPU index or UUID.
# CPU tuning: --cpu-threads N --cpu-interop-threads N.
# Add --detach for background execution, or --dry-run to inspect settings.
set -Eeuo pipefail
SCRIPT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
export PYTHONPATH="${SCRIPT_ROOT}/source${PYTHONPATH:+:${PYTHONPATH}}"
exec "${BRAID_PYTHON:-python3}" -m experiments.launcher "$@"
