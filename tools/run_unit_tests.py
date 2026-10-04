"""Collect only the unit-test package, with explicit module identities.

Root test*.py files are historical inference commands. A fixed top-level path
avoids naming collisions when those commands are imported by test fixtures.
"""
import argparse
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]


def discover(pattern="test*.py"):
    # Historical tests use both tests.test_* and bare test_* fixture imports.
    # Keep both import styles without collecting root inference entrypoints.
    for path in (ROOT / "tests", ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return unittest.TestLoader().discover(
        start_dir=str(ROOT / "tests"), pattern=pattern, top_level_dir=str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern", default="test*.py")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=1 if args.quiet else 2).run(discover(args.pattern))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
