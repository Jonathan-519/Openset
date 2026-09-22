#!/usr/bin/env bash
# Run from the user's activated ProTeCt environment. No Git checkout required.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ACTION=${1:-develop}
HIER_SUITE=${2:-runs/taxosafe_hier_v4_dev}
HIER_SOURCE=${3:-runs/taxosafe_rs_paper}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
case "$ACTION" in
  develop)
    for SEED in 1 2 3; do
      PLAN="$HIER_SUITE/seed_${SEED}/plan.json"
      if [ ! -f "$PLAN" ]; then
        python tools/run_taxosafe_hier_v4.py plan \
          --source-suite "$HIER_SOURCE" --suite-dir "$HIER_SUITE" --seed "$SEED"
      fi
    done
    for SEED in 1 2 3; do
      python tools/run_taxosafe_hier_v4.py run \
        --plan "$HIER_SUITE/seed_${SEED}/plan.json" --stage develop --device cuda --resume
    done
    echo "Training and validation completed. Review validation_report.json before test."
    ;;
  test)
    for SEED in 1 2 3; do
      python tools/run_taxosafe_hier_v4.py run \
        --plan "$HIER_SUITE/seed_${SEED}/plan.json" --stage test --device cuda --resume
    done
    ;;
  summarize)
    python tools/run_taxosafe_hier_v4.py summarize --suite-dir "$HIER_SUITE" \
      --output-dir "${HIER_SUITE}_diagnosis_$(date +%Y%m%d_%H%M%S)"
    ;;
  pack)
    python - "$HIER_SUITE" <<'PY'
from pathlib import Path
from datetime import datetime
import sys
import tarfile
root = Path.cwd()
suite = Path(sys.argv[1]).resolve()
try:
    suite.relative_to(root)
except ValueError:
    raise SystemExit("Pack expects a suite inside the project directory")
files = sorted(p for p in suite.rglob('*') if p.is_file() and not p.is_symlink()
               and p.suffix in ('.json', '.jsonl', '.log', '.yml', '.md'))
if not files:
    raise SystemExit("No review files found")
archive = root / 'runs' / ('taxosafe_hier_v4_review_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.tar.gz')
with tarfile.open(str(archive), 'w:gz') as tar:
    for path in files:
        tar.add(str(path), arcname=path.relative_to(root).as_posix(), recursive=False)
print('Review files:', len(files))
print('Upload this file:', archive)
PY
    ;;
  *)
    echo "Usage: bash tools/run_taxosafe_hier_v4.sh {develop|test|summarize|pack} [suite-dir] [source-suite]" >&2
    exit 2
    ;;
esac
