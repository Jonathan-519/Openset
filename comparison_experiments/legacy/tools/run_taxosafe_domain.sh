#!/usr/bin/env bash
set -euo pipefail
DOMAIN_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$DOMAIN_PROJECT_ROOT"
# CPU cached evidence; the CLI selects a fresh timestamped Domain output.
python -u -m taxosafe_domain "$@"
