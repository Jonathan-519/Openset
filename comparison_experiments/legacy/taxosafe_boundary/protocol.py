"""Predeclared per-witness boundary and same-bank ranking experiments.

Only new files belong to this family. Historical D05 signatures remain intact.
"""
import copy
import math
from pathlib import Path

import yaml

from taxosafe_support.protocol import (
    PROJECT_ROOT, file_hash, object_hash, read_json, write_json, write_records,
    resolve, run_lock,
)
from taxosafe_recovery.protocol import code_files as inherited_code_files

SCHEMA_VERSION = "taxosafe_boundary_v1"
DEFAULT_CONFIG = "configs/Zooplankton_Taxonomic_Tree/TaxoSafe_d05_boundary.yml"
ARMS = [
    dict(id="G00_reference", kind="reference", mode="reference", policy="source"),
    dict(id="G01_d05", kind="source", mode="d05", policy="source"),
    dict(id="G02_d05_kp", kind="reuse", mode="d05", policy="kp_frontier", weight_source="G01_d05"),
    dict(id="G03_evm_leaf", kind="evm", mode="evm_leaf", policy="kp_frontier"),
    dict(id="G04_bce8", kind="train", mode="bce8", policy="kp_frontier"),
    dict(id="G05_rank8", kind="train", mode="rank8", policy="kp_frontier"),
    dict(id="G06_bce9", kind="train", mode="bce9", policy="kp_frontier"),
    dict(id="G07_rank9", kind="train", mode="rank9", policy="kp_frontier"),
    dict(id="G08_rank9_standard", kind="reuse", mode="rank9", policy="standard", weight_source="G07_rank9"),
    dict(id="G09_rank9_leaf_guard", kind="leaf_guard", mode="leaf_guard", policy="fixed_parent", weight_source="G07_rank9"),
]
DEFAULTS = dict(
    name="D05 per-witness boundaries and same-bank cross-query ranking", seed=1, arms=ARMS,
    boundary=dict(tail_size=32),
    training=dict(steps=300, batch_size=1024, lr=.001, l2sp_weight=.01,
                  ranking_weight=.2, ranking_margin=.2),
    calibration=dict(seed=1, known_target=.92),
)


def validate_config(cfg):
    if not isinstance(cfg, dict) or set(cfg) != set(DEFAULTS):
        raise ValueError("Unexpected boundary configuration fields")
    if not isinstance(cfg["name"], str) or not cfg["name"].strip():
        raise ValueError("Experiment name must be nonempty")
    if type(cfg["seed"]) is not int or not 0 <= cfg["seed"] < 2**31:
        raise ValueError("seed must be a nonnegative integer below 2**31")
    if cfg["arms"] != ARMS:
        raise ValueError("Use the ten predeclared boundary controls in order")
    for section in ("boundary", "training", "calibration"):
        values = cfg[section]
        if not isinstance(values, dict) or set(values) != set(DEFAULTS[section]):
            raise ValueError("Unexpected " + section + " settings")
        for key, value in values.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("Expected a finite numeric setting: " + key)
            if key == "seed":
                if type(value) is not int or value != cfg["seed"]:
                    raise ValueError("Calibration seed must match suite seed")
            elif key in ("steps", "batch_size", "tail_size"):
                if type(value) is not int or value < 1:
                    raise ValueError("Expected a positive integer: " + key)
            elif value <= 0:
                raise ValueError("Expected a positive setting: " + key)
    if cfg["training"]["lr"] > .001:
        raise ValueError("Boundary learning rate must be <= .001")
    if cfg["boundary"]["tail_size"] != 32:
        raise ValueError("Boundary tail_size is preregistered as 32")
    if cfg["calibration"]["known_target"] != .92:
        raise ValueError("The known-priority control uses the declared 92% DEV target")
    return copy.deepcopy(cfg)


def effective_config(path=DEFAULT_CONFIG):
    return validate_config(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def code_files():
    files = set(inherited_code_files())
    files.update((PROJECT_ROOT / "taxosafe_boundary").glob("*.py"))
    for name in ("tools/run_taxosafe_boundary.sh", "tools/pack_taxosafe_boundary_review.py"):
        path = PROJECT_ROOT / name
        if path.is_file():
            files.add(path)
    return sorted(files)


def code_signature():
    return object_hash({str(p.relative_to(PROJECT_ROOT)): file_hash(p) for p in code_files()})


def signature(cfg, source_binding):
    return dict(method=SCHEMA_VERSION, config=object_hash(cfg),
                source=object_hash(source_binding), code=code_signature())
