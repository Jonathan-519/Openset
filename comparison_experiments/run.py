#!/usr/bin/env python3
"""Run an archived comparison in its own Python process and source directory.

Example: python comparison_experiments/run.py -m taxosafe_domain --help
The archived files keep their original relative paths and code signatures.
"""
from pathlib import Path
import subprocess
import sys


def main():
    legacy = Path(__file__).resolve().parent / "legacy"
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    raise SystemExit(subprocess.call([sys.executable, *sys.argv[1:]], cwd=str(legacy)))


if __name__ == "__main__":
    main()
