#!/usr/bin/env python3
"""Compatibility entry point; implementation lives in tools/run_taxosieve_seeds.py."""
from tools.run_taxosieve_seeds import *  # Preserve imports from the original helper.
from tools.run_taxosieve_seeds import main


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        raise SystemExit("ERROR: " + str(error))
