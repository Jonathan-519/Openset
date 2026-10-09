#!/usr/bin/env bash
set -euo pipefail
DISCOVERY_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$DISCOVERY_PROJECT_ROOT"
python -u -m taxosafe_discovery "$@"
