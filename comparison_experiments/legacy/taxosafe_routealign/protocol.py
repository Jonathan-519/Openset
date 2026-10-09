"""Isolated baseline fine-tuning suite; historical import contracts stay intact."""
import copy
import math
from pathlib import Path

import yaml

from taxosafe_support.protocol import (
    PROJECT_ROOT, file_hash, object_hash, read_json, write_json, write_records,
    resolve, run_lock,
)
from taxosafe_sweep.protocol import code_files as inherited_code_files

SCHEMA_VERSION = "taxosafe_routealign_v1"
DEFAULT_CONFIG = "configs/Zooplankton_Taxonomic_Tree/TaxoSafe_reference_routealign.yml"
ARMS = [
    {"id": "A00_reference", "kind": "baseline", "weights": "reference", "router": "reference"},
    {"id": "A01_evidence_anchor", "kind": "finetune", "scope": "heads", "anchored": True,
     "evidence_anchor": True},
    {"id": "A02_proximity", "kind": "routing", "weights": "reference", "router": "joint"},
    {"id": "A03_combined", "kind": "routing", "weights": "A01_evidence_anchor", "router": "joint"},
    {"id": "A04_parent_rerank", "kind": "routing", "weights": "A01_evidence_anchor", "router": "rerank"},
]
DEFAULT_BUDGET = {"epochs": 20, "patience": 6, "min_epochs": 3,
                  "batches_per_epoch": 240, "prompt_lr": 0.0005, "head_lr": 0.001}


def effective_config(path=DEFAULT_CONFIG):
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or set(cfg) != {"name", "seed", "budget", "arms", "proximity", "router"}:
        raise ValueError("RouteAlign requires name, seed, budget, arms, proximity and router")
    if not isinstance(cfg["name"], str) or not cfg["name"].strip():
        raise ValueError("Sweep name must be nonempty")
    if type(cfg["seed"]) is not int or not 0 <= cfg["seed"] < 2 ** 31:
        raise ValueError("Sweep seed must be an integer in [0,2**31)")
    if object_hash(cfg["arms"]) != object_hash(ARMS):
        raise ValueError("Use the five preregistered arms in their fixed order")
    budget = cfg["budget"]
    if not isinstance(budget, dict) or set(budget) != set(DEFAULT_BUDGET):
        raise ValueError("Unexpected sweep budget fields")
    for key in ("epochs", "patience", "min_epochs", "batches_per_epoch"):
        if type(budget[key]) is not int or budget[key] < 1:
            raise ValueError("budget." + key + " must be a positive integer")
    if budget["min_epochs"] > budget["epochs"]:
        raise ValueError("min_epochs cannot exceed epochs")
    for key in ("prompt_lr", "head_lr"):
        value = budget[key]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not 0 < value <= 0.01):
            raise ValueError("budget." + key + " must be finite in (0,0.01]")
    prox = cfg["proximity"]
    if (not isinstance(prox, dict) or set(prox) != {"k", "shrinkage"}
            or type(prox["k"]) is not int or not 1 <= prox["k"] <= 32
            or isinstance(prox["shrinkage"], bool)
            or not isinstance(prox["shrinkage"], (int, float))
            or not math.isfinite(prox["shrinkage"]) or prox["shrinkage"] <= 0):
        raise ValueError("Invalid TRAIN proximity settings")
    router = cfg["router"]
    if not isinstance(router, dict) or set(router) != {"grid_points", "proximity_weight", "rerank_weight", "proximity_clip", "seed"}:
        raise ValueError("Invalid router setting names")
    if type(router["grid_points"]) is not int or not 3 <= router["grid_points"] <= 49:
        raise ValueError("Router grid_points must be in [3,49]")
    if router["seed"] != cfg["seed"] or type(router["seed"]) is not int:
        raise ValueError("Router and suite seeds must agree")
    for key in ("proximity_weight", "rerank_weight", "proximity_clip"):
        v = router[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
            raise ValueError("Router weights and clip must be finite positive")
    from .calibration import validate_settings
    validate_settings(router)
    return copy.deepcopy(cfg)


def code_files():
    files = set(inherited_code_files())
    files.update((PROJECT_ROOT / "taxosafe_routealign").glob("*.py"))
    for name in ("tools/run_taxosafe_routealign.sh", "tools/pack_taxosafe_routealign_review.py"):
        path = PROJECT_ROOT / name
        if path.is_file():
            files.add(path)
    return sorted(files)


def code_signature():
    return object_hash({str(path.relative_to(PROJECT_ROOT)): file_hash(path)
                        for path in code_files()})


def signature(cfg, source_binding):
    return {"method": SCHEMA_VERSION, "config": object_hash(cfg),
            "reference": object_hash(source_binding), "code": code_signature()}
