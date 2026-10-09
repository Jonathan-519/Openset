"""Versioned first-stage matrix; historical source contracts remain unchanged."""
import copy
import math
from pathlib import Path

import yaml

from taxosafe_support.protocol import (
    PROJECT_ROOT, file_hash, object_hash, read_json, write_json, write_records,
    resolve, run_lock,
)
from taxosafe_domain.protocol import code_files as inherited_code_files

SCHEMA_VERSION = "taxosafe_morphology_v1"
DEFAULT_CONFIG = "configs/Zooplankton_Taxonomic_Tree/TaxoSafe_morphology.yml"


def _arm(identifier, kind, root, leaf, level=None):
    return dict(id=identifier, kind=kind, mode=identifier.split("_")[0],
                root=root, leaf=leaf, level=level,
                policy="source" if kind in ("reference", "source") else "staged",
                candidate_policy="original" if kind in ("reference", "source") else "reference_path")


ARMS = [
    _arm("C00_reference", "reference", "reference", "reference"),
    _arm("C01_d05", "source", "d05", "d05"),
    _arm("C02_d05_staged", "evidence", "d05_candidate", "d05"),
    _arm("R01_spatial_parent", "fit", "spatial_parent", "d05", "parent"),
    _arm("L01_spatial_leaf", "fit", "d05_candidate", "spatial_leaf", "leaf"),
]
DEFAULTS = dict(
    name="Frozen reference spatial morphology verification", seed=1, arms=ARMS,
    features=dict(batch_size=8, workers=0, view="reference_full", storage_dtype="float32"),
    matching=dict(epsilon=0.1, iterations=40, coverage_cost=0.5, pair_chunk=4),
    support=dict(references_per_leaf=2, folds=3),
    training=dict(steps=800, batch_size=4, learning_rate=0.001, adapter_dim=32,
                  hidden=64, auxiliary_ce=0.1, weight_decay=0.0001, log_every=25),
    calibration=dict(seed=1, root_known_target=0.92, root_near_target=0.85),
)


def validate_config(cfg):
    if not isinstance(cfg, dict) or set(cfg) != set(DEFAULTS):
        raise ValueError("Unexpected morphology configuration fields")
    if cfg["arms"] != ARMS:
        raise ValueError("Use the five declared controls; RL combination is deferred")
    if not isinstance(cfg["name"], str) or not cfg["name"].strip():
        raise ValueError("Experiment name must be nonempty")
    if type(cfg["seed"]) is not int or not 0 <= cfg["seed"] < 2**31:
        raise ValueError("Invalid seed")
    for section in ("features", "matching", "support", "training", "calibration"):
        if not isinstance(cfg[section], dict) or set(cfg[section]) != set(DEFAULTS[section]):
            raise ValueError("Unexpected settings: " + section)
        for key, default in DEFAULTS[section].items():
            value = cfg[section][key]
            if type(value) is not type(default):
                raise ValueError("Invalid type: " + section + "." + key)
            if isinstance(value, (float, int)) and (not math.isfinite(value) or value < 0):
                raise ValueError("Invalid numeric setting: " + section + "." + key)
    for section, keys in (("features", ("batch_size",)), ("matching", ("iterations", "pair_chunk")),
                          ("support", ("references_per_leaf", "folds")),
                          ("training", ("steps", "batch_size", "adapter_dim", "hidden", "log_every"))):
        if any(cfg[section][key] < 1 for key in keys):
            raise ValueError("Positive setting required in " + section)
    if cfg["support"]["folds"] < 2 or not 0 < cfg["matching"]["epsilon"] <= 1:
        raise ValueError("At least two folds and epsilon in (0,1] required")
    if not 0 < cfg["matching"]["coverage_cost"] < 2 or cfg["training"]["learning_rate"] <= 0:
        raise ValueError("Invalid coverage cost or learning rate")
    if cfg["features"]["view"] != "reference_full" or cfg["features"]["storage_dtype"] != "float32":
        raise ValueError("First experiment uses full reference view and complete float32 spatial tokens")
    if cfg["calibration"] != dict(DEFAULTS["calibration"], seed=cfg["seed"]):
        raise ValueError("Root coverage rules are inherited unchanged; calibration seed must match")
    return copy.deepcopy(cfg)


def effective_config(path=DEFAULT_CONFIG):
    return validate_config(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def code_files():
    files = set(inherited_code_files())
    files.update((PROJECT_ROOT / "taxosafe_morphology").glob("*.py"))
    files.update(PROJECT_ROOT / name for name in
                 ("tools/run_taxosafe_morphology.sh", "tools/pack_taxosafe_morphology_review.py"))
    return sorted(files)


def code_signature():
    return object_hash({str(p.relative_to(PROJECT_ROOT)): file_hash(p) for p in code_files()})


def signature(cfg, source_binding):
    return dict(method=SCHEMA_VERSION, config=object_hash(cfg),
                source=object_hash(source_binding), code=code_signature())
