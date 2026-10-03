"""Isolated baseline fine-tuning suite; historical import contracts stay intact."""
import copy
import math
from pathlib import Path

import yaml

from taxosafe_support.protocol import (
    PROJECT_ROOT, file_hash, object_hash, read_json, write_json, write_records,
    resolve, run_lock,
)
from taxosafe_frontier.protocol import code_files as inherited_code_files

SCHEMA_VERSION = "taxosafe_sweep_v1"
DEFAULT_CONFIG = "configs/Zooplankton_Taxonomic_Tree/TaxoSafe_reference_sweep.yml"
ARMS = [
    {"id": "E00_reference", "kind": "baseline"},
    {"id": "E01_heads", "kind": "finetune", "scope": "heads",
     "anchored": False, "hierarchy": False},
    {"id": "E02_heads_anchor", "kind": "finetune", "scope": "heads",
     "anchored": True, "hierarchy": False},
    {"id": "E03_prompt_anchor", "kind": "finetune", "scope": "prompts_heads",
     "anchored": True, "hierarchy": False},
    {"id": "E04_hierarchy_anchor", "kind": "finetune", "scope": "prompts_heads",
     "anchored": True, "hierarchy": True},
    {"id": "E05_hierarchy_blend", "kind": "blend",
     "parent_arm": "E04_hierarchy_anchor", "alpha": 0.5},
]
DEFAULT_BUDGET = {"epochs": 20, "patience": 6, "min_epochs": 3,
                  "batches_per_epoch": 240, "prompt_lr": 0.0005, "head_lr": 0.001}


def effective_config(path=DEFAULT_CONFIG):
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or set(cfg) != {"name", "seed", "budget", "arms"}:
        raise ValueError("Sweep requires exactly name, seed, budget and arms")
    if not isinstance(cfg["name"], str) or not cfg["name"].strip():
        raise ValueError("Sweep name must be nonempty")
    if type(cfg["seed"]) is not int or not 0 <= cfg["seed"] < 2 ** 31:
        raise ValueError("Sweep seed must be an integer in [0,2**31)")
    if object_hash(cfg["arms"]) != object_hash(ARMS):
        raise ValueError("Use the six preregistered arms in their fixed order")
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
    return copy.deepcopy(cfg)


def code_files():
    files = set(inherited_code_files())
    files.update((PROJECT_ROOT / "taxosafe_sweep").glob("*.py"))
    for name in ("tools/run_taxosafe_sweep.sh", "tools/pack_taxosafe_sweep_review.py"):
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
