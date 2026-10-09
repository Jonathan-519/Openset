#!/usr/bin/env bash
set -euo pipefail
BOUNDARY_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$BOUNDARY_PROJECT_ROOT"
# CPU cached evidence; the CLI selects a fresh timestamped Boundary output.
python -u -m taxosafe_boundary "$@"
