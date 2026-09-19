#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
phase="${1:-all}"
seed="${2:-1}"
run_dir="runs/boundaryshell_v3/seed_${seed}"
mkdir -p "$run_dir/logs"
run_phase() {
  local action="$1"
  shift
  python -u tools/run_boundaryshell_v3.py "$action" --seed "$seed" --run-dir "$run_dir" "$@" \
    2>&1 | tee -a "$run_dir/logs/${action}.log"
}
case "$phase" in
  all)
    run_phase prepare
    run_phase preflight --check-image-content
    run_phase train
    run_phase calibrate
    run_phase test
    run_phase pack
    ;;
  prepare|train|calibrate|test|pack) run_phase "$phase" ;;
  preflight) run_phase preflight --check-image-content ;;
  open) run_phase train --stage open ;;
  classifier) run_phase train --stage classifier ;;
  *) echo "Usage: bash tools/run_boundaryshell_v3.sh {all|prepare|preflight|train|classifier|open|calibrate|test|pack} [seed]" >&2; exit 2 ;;
esac
