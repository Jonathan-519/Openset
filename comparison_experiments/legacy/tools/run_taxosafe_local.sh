#!/usr/bin/env bash
# Run a new frozen-reference local experiment, then package text diagnostics.
set -euo pipefail
if (( $# > 3 )); then
    echo "Usage: bash tools/run_taxosafe_local.sh [reference-run-dir] [new-run-dir] [cuda|cpu]" >&2
    exit 2
fi
cd "$(dirname "${BASH_SOURCE[0]}")/.."
reference_dir=${1:-runs/taxosafe_new/reference/trial_1}
run_dir=${2:-runs/taxosafe_new/reference_local/trial_1}
device=${3:-cuda}
config=configs/Zooplankton_Taxonomic_Tree/TaxoSafe_reference_local.yml
python refine_taxosafe_geometry.py fit --config "$config" \
    --reference-run-dir "$reference_dir" --run-dir "$run_dir" --device "$device" --preflight
for stage in fit calibrate test; do
    python -u refine_taxosafe_geometry.py "$stage" --config "$config" \
        --reference-run-dir "$reference_dir" --run-dir "$run_dir" --device "$device"
done
python tools/pack_taxosafe_new_review.py --run-dir "$run_dir"
