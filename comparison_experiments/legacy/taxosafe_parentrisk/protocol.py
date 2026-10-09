"""Independent, fail-closed provenance for the frozen ParentRisk experiment.

Historical support/refine/geometry source is imported unchanged. Their reviewed
reference importer remains the sole authority for accepting a source run.
"""
import copy
import math
from pathlib import Path

import yaml

from taxosafe_support.protocol import (
    PROJECT_ROOT, claim_stage, file_hash, object_hash, read_json,
    require_signature, resolve, run_lock, write_json, write_records,
)

SCHEMA_VERSION = "frozen_reference_parentrisk_v1"
DEFAULT_CONFIG = "configs/Zooplankton_Taxonomic_Tree/TaxoSafe_reference_parentrisk.yml"
FIT_STAGE = "parentrisk"
FIT_ARTIFACTS = {
    "model": "frozen_evidence.pth", "cache": "cache.pth", "scales": "scales.json",
    "report": "fit_report.json", "inputs": "inputs.json", "scores": "train_scores.jsonl",
    "config_file": "config.json", "source_file": "source_binding.json",
    "timing": "inference_timing.json",
}


def effective_config(path, seed=None):
    cfg = copy.deepcopy(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
    if not isinstance(cfg, dict) or set(cfg) != {"name", "seed", "geometry", "calibration"}:
        raise ValueError("ParentRisk requires exactly name, seed, geometry and calibration")
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
    required = {"mode", "outer_folds", "inner_folds", "grid_points", "min_rule_sources"}
    if not isinstance(settings, dict) or set(settings) != required:
        raise ValueError("Invalid ParentRisk calibration fields; baseline settings come only from the frozen source")
    if settings["mode"] not in ("audit", "parent_only", "combined"):
        raise ValueError("calibration.mode must be audit, parent_only or combined")
    for key, low, high in (("outer_folds", 2, 8), ("inner_folds", 2, 8),
                           ("grid_points", 2, 11), ("min_rule_sources", 2, 8)):
        value = settings[key]
        if type(value) is not int or not low <= value <= high:
            raise ValueError("calibration." + key + " must be an integer in [{},{}]".format(low, high))
    return cfg


def code_files():
    """Bind inference, preprocessing and all transitive project Python code.

    Tests/reports/runs are deliberately excluded: collecting results must not
    invalidate an immutable experiment. Configuration values have a separate
    semantic digest, so equivalent config formatting is immaterial.
    """
    directories = ("taxosafe_parentrisk", "taxosafe_support", "taxosafe_refine",
                   "taxosafe_geometry", "taxosafe_dcbs", "taxosafe_hier", "models",
                   "loader", "losses", "optim")
    paths = {p for name in directories for p in (PROJECT_ROOT / name).rglob("*.py")}
    paths.update(PROJECT_ROOT.glob("*.py"))
    return sorted(paths)


def signature(cfg, reference_binding):
    return {"method": SCHEMA_VERSION, "config": object_hash(cfg),
            "reference": object_hash(reference_binding),
            "code": object_hash({str(p.relative_to(PROJECT_ROOT)): file_hash(p) for p in code_files()})}


def ensure_fresh_fit(directory):
    """Never add a fit to an old/partial experiment or infer resumability."""
    path = Path(directory)
    if path.exists() and any(p.name != ".dcbs.lock" for p in path.iterdir()):
        raise ValueError("Fit requires a fresh empty --run-dir; preserve the existing directory: " + str(path))


def artifacts(directory, names):
    return {key: {"path": name, "sha256": file_hash(Path(directory) / name)} for key, name in names.items()}


def verify_artifacts(directory, receipt, names):
    """Only fixed local filenames may be read; reject aliases and escaping paths."""
    root = Path(directory).resolve()
    for key, name in names.items():
        item = receipt.get(key)
        if not isinstance(item, dict) or set(item) != {"path", "sha256"} or item["path"] != name:
            raise ValueError("Unexpected ParentRisk artifact descriptor: " + key)
        path = root / name
        if path.is_symlink() or path.resolve().parent != root or not path.is_file():
            raise ValueError("Missing or escaping ParentRisk artifact: " + name)
        if file_hash(path) != item["sha256"]:
            raise ValueError("ParentRisk artifact hash mismatch: " + name)
