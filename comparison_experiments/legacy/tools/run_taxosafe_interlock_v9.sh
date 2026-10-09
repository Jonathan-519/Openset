#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ACTION=${1:-develop}
INTERLOCK_SUITE=${2:-runs/taxosafe_interlock_v9_dev}
INTERLOCK_SOURCE=${3:-runs/taxosafe_hier_v4_dev}
case "$ACTION" in
  develop)
    for SEED in 1 2 3; do
      if [ ! -f "$INTERLOCK_SUITE/seed_${SEED}/plan.json" ]; then
        python tools/run_taxosafe_interlock_v9.py plan --source-suite "$INTERLOCK_SOURCE" \
          --suite-dir "$INTERLOCK_SUITE" --seed "$SEED"
      fi
    done
    for SEED in 1 2 3; do
      python tools/run_taxosafe_interlock_v9.py run --plan "$INTERLOCK_SUITE/seed_${SEED}/plan.json" \
        --stage develop --resume
    done
    ;;
  test)
    for SEED in 1 2 3; do
      python tools/run_taxosafe_interlock_v9.py run --plan "$INTERLOCK_SUITE/seed_${SEED}/plan.json" \
        --stage test --resume
    done
    ;;
  summarize)
    python tools/run_taxosafe_interlock_v9.py summarize --suite-dir "$INTERLOCK_SUITE"
    ;;
  pack)
    python - "$INTERLOCK_SUITE" <<'PY'
from pathlib import Path
from datetime import datetime
import sys, tarfile
root = Path.cwd()
suite = Path(sys.argv[1]).resolve()
suite.relative_to(root)
files = sorted(p for p in suite.rglob('*') if p.is_file() and not p.is_symlink()
               and p.suffix in ('.json', '.jsonl', '.md', '.log', '.yml'))
if not files:
    raise SystemExit('No review files')
out = root / 'runs' / ('taxosafe_interlock_v9_review_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.tar.gz')
with tarfile.open(str(out), 'w:gz') as tar:
    for path in files:
        tar.add(str(path), arcname=path.relative_to(root).as_posix(), recursive=False)
print('Upload this file:', out)
PY
    ;;
  *) echo 'Usage: bash tools/run_taxosafe_interlock_v9.sh {develop|test|summarize|pack} [suite] [source]' >&2; exit 2 ;;
esac
