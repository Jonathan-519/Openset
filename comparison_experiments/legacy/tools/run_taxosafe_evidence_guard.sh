#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# Preserve the working reference environment. Do not silently change BLAS/OMP
# thread counts, image batch size, precision, reference code, or old receipts.
exec python -u -m taxosafe_evidence_guard "$@"
