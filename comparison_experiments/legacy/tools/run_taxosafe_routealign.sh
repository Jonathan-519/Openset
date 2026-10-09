#!/usr/bin/env bash
set -euo pipefail
ROUTEALIGN_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$ROUTEALIGN_PROJECT_ROOT"
python -u -m taxosafe_routealign "$@"
