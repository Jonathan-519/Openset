"""Separate Frontier provenance; every historical signature remains unchanged.

Use ``python -m taxosafe_frontier``: adding a root-level Python entrypoint
would change the older ParentRisk signature's root-file enumeration.
"""
import copy
import math
from pathlib import Path

import yaml

from taxosafe_support.protocol import (
    PROJECT_ROOT, claim_stage, file_hash, object_hash, read_json,
    require_signature, resolve, run_lock, write_json, write_records,
)
from taxosafe_parentrisk.protocol import (
    artifacts, ensure_fresh_fit, verify_artifacts, code_files as legacy_code_files,
)

SCHEMA_VERSION = "frozen_reference_frontier_v1"
DEFAULT_CONFIG = "configs/Zooplankton_Taxonomic_Tree/TaxoSafe_reference_frontier.yml"
FIT_STAGE = "frontier"
FIT_ARTIFACTS = {
    "model": "frozen_evidence.pth", "cache": "cache.pth", "scales": "scales.json",
    "report": "fit_report.json", "inputs": "inputs.json", "scores": "train_scores.jsonl",
    "config_file": "config.json", "source_file": "source_binding.json",
    "timing": "inference_timing.json",
}


def effective_config(path, seed=None):
    cfg = copy.deepcopy(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
    if not isinstance(cfg, dict) or set(cfg) != {"name", "seed", "geometry", "calibration"}:
        raise ValueError("Frontier requires exactly name, seed, geometry and calibration")
    if seed is not None:
        cfg["seed"] = seed
    if type(cfg["seed"]) is not int or not 0 <= cfg["seed"] < 2 ** 31:
        raise ValueError("seed must be an integer in [0, 2**31)")
    if not isinstance(cfg["name"], str) or not cfg["name"].strip():
        raise ValueError("name must be a nonempty string")
    geo = cfg["geometry"]
    if not isinstance(geo, dict) or set(geo) != {"shrinkage", "ridge"}:
        raise ValueError("geometry requires shrinkage and ridge")
    for key, inclusive in (("shrinkage", True), ("ridge", False)):
        value = geo[key]
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                or value > 1 or (value < 0 if inclusive else value <= 0)):
            raise ValueError("Invalid geometry." + key)
    settings = cfg["calibration"]
    required = {"outer_folds", "inner_folds", "min_rule_sources"}
    if not isinstance(settings, dict) or set(settings) != required:
        raise ValueError("Invalid Frontier calibration fields; source baseline and score families cannot be overridden")
    for key, low, high in (("outer_folds", 2, 8), ("inner_folds", 2, 8), ("min_rule_sources", 2, 8)):
        value = settings[key]
        if type(value) is not int or not low <= value <= high:
            raise ValueError("calibration." + key + " must be an integer in [{},{}]".format(low, high))
    return cfg


def code_files():
    """Bind inherited collection plus this experiment, without legacy edits."""
    paths = set(legacy_code_files())
    paths.update((PROJECT_ROOT / "taxosafe_frontier").rglob("*.py"))
    return sorted(paths)


def signature(cfg, reference_binding):
    return {"method": SCHEMA_VERSION, "config": object_hash(cfg),
            "reference": object_hash(reference_binding),
            "code": object_hash({str(p.relative_to(PROJECT_ROOT)): file_hash(p) for p in code_files()})}
