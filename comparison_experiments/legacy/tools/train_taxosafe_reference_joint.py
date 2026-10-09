from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from taxosafe_reference_joint.entrypoints import run_phase

if __name__ == "__main__":
    run_phase("training")
