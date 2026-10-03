#!/usr/bin/env bash
set -euo pipefail
SWEEP_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$SWEEP_PROJECT_ROOT"
python -u -m taxosafe_sweep "$@"
