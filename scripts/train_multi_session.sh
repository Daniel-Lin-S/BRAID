#!/usr/bin/env bash
# Usage: train_multi_session.sh --experiment LOCAL_YAML [--stage plan|run|report]
# Add --detach to launch the queue in the background. Resume with the same YAML.
# Set BRAID_PYTHON to the Python executable containing TensorFlow and BRAID deps.
set -Eeuo pipefail
SCRIPT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
export PYTHONPATH="${SCRIPT_ROOT}/source${PYTHONPATH:+:${PYTHONPATH}}"
export TF_USE_LEGACY_KERAS=1
exec "${BRAID_PYTHON:-python3}" -m experiments.multisession.runner "$@"
