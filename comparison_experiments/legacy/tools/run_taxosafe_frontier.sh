#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: bash tools/run_taxosafe_frontier.sh --config CONFIG --reference-run-dir SOURCE --run-dir NEW_RUN [--device cuda|cpu] [--evaluate-test]
Uses the current Python environment; does not upgrade it or delete any run.
Default: read-only preflight, known TRAIN evidence fit, DEV calibration/audit, review pack.
--evaluate-test adds a single evaluation of the locked, previously reviewed TEST.
USAGE
}

config=""; reference=""; destination=""; device="cuda"; evaluate_test=0
while (($#)); do
  case "$1" in
    --config|--reference-run-dir|--run-dir|--device)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      case "$1" in
        --config) config="$2" ;;
        --reference-run-dir) reference="$2" ;;
        --run-dir) destination="$2" ;;
        --device) device="$2" ;;
      esac
      shift 2 ;;
    --evaluate-test) evaluate_test=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done
[[ -n "$config" && -n "$reference" && -n "$destination" ]] || { usage >&2; exit 2; }
[[ "$device" == "cuda" || "$device" == "cpu" ]] || { usage >&2; exit 2; }
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$project_dir"
args=(--config "$config" --reference-run-dir "$reference" --run-dir "$destination" --device "$device")
python -u -m taxosafe_frontier fit "${args[@]}" --preflight
python -u -m taxosafe_frontier fit "${args[@]}"
python -u -m taxosafe_frontier calibrate "${args[@]}" --preflight
python -u -m taxosafe_frontier calibrate "${args[@]}"
if ((evaluate_test)); then
  python -u -m taxosafe_frontier test "${args[@]}" --evaluate-test --preflight
  python -u -m taxosafe_frontier test "${args[@]}" --evaluate-test
fi
python tools/pack_taxosafe_frontier_review.py --run-dir "$destination"
