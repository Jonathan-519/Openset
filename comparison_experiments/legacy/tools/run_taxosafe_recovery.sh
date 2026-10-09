#!/usr/bin/env bash
set -euo pipefail
RECOVERY_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$RECOVERY_PROJECT_ROOT"
python -u -m taxosafe_recovery "$@"
