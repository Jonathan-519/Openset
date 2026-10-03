"""Configuration and provenance for a TRAIN-only frozen-reference refinement."""
import copy
import math
from pathlib import Path

import yaml

from taxosafe_support.protocol import (
    PROJECT_ROOT, claim_stage, file_hash, object_hash, read_json,
    require_signature, resolve, run_lock, verify_artifact, write_json, write_records,
)


DEFAULT_CONFIG = "configs/Zooplankton_Taxonomic_Tree/TaxoSafe_reference_refine.yml"


def _integer(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(name + " must be an integer in [{},{}]".format(low, high))


def _number(value, name, low, high, lower_inclusive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(name + " must be finite and numeric")
    value = float(value)
    if not math.isfinite(value) or value > high or (value < low if lower_inclusive else value <= low):
        raise ValueError(name + " is outside its permitted range")


def effective_config(path, seed=None):
    cfg = copy.deepcopy(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
    if not isinstance(cfg, dict):
        raise ValueError("Expected a frozen-reference refinement configuration")
    allowed = {"name", "seed", "features", "reconstruction", "training", "calibration"}
    if set(cfg) != allowed:
        raise ValueError("Refinement requires exactly: " + ", ".join(sorted(allowed)))
    if seed is not None:
        cfg["seed"] = seed
    _integer(cfg["seed"], "seed", 0, 2 ** 31 - 1)
    if not isinstance(cfg["name"], str) or not cfg["name"].strip():
        raise ValueError("name must be a nonempty string")
    if cfg["features"] not in ("fine", "raw_spatial"):
        raise ValueError("features must be fine or raw_spatial")
    r = cfg["reconstruction"]
    if not isinstance(r, dict) or set(r) != {"rank", "score_mode", "scale_init", "eps"}:
        raise ValueError("Invalid reconstruction configuration fields")
    _integer(r["rank"], "reconstruction.rank", 1, 512)
    if r["score_mode"] not in ("relative", "cssr"):
        raise ValueError("score_mode must be relative or cssr")
    _number(r["scale_init"], "reconstruction.scale_init", 0, 1e4)
    _number(r["eps"], "reconstruction.eps", 0, 1e-2)
    t = cfg["training"]
    if not isinstance(t, dict) or set(t) != {
            "epochs", "batch_size", "learning_rate", "validation_fraction", "patience"}:
        raise ValueError("Invalid refinement training configuration fields")
    _integer(t["epochs"], "training.epochs", 1, 1000)
    _integer(t["batch_size"], "training.batch_size", 1, 4096)
    _integer(t["patience"], "training.patience", 1, 1000)
    _number(t["learning_rate"], "training.learning_rate", 0, 1)
    _number(t["validation_fraction"], "training.validation_fraction", 0, .5)
    c = cfg["calibration"]
    if not isinstance(c, dict) or set(c) != {"grid_points", "source_loo"}:
        raise ValueError("Invalid refinement calibration configuration fields")
    _integer(c["grid_points"], "calibration.grid_points", 2, 401)
    if not isinstance(c["source_loo"], bool):
        raise ValueError("calibration.source_loo must be a YAML boolean")
    return cfg


def signature(cfg, reference_binding):
    files = sorted((PROJECT_ROOT / "taxosafe_refine").glob("*.py"))
    files += [PROJECT_ROOT / "refine_taxosafe_reference.py"]
    return {"method": "frozen_reference_reconstruction_v1", "config": object_hash(cfg),
            "reference": object_hash(reference_binding),
            "code": object_hash({str(p.relative_to(PROJECT_ROOT)): file_hash(p) for p in files})}
