"""Predeclared D05 continuation controls; historical source stays immutable."""
import copy
import math
from pathlib import Path

import yaml

from taxosafe_support.protocol import (
    PROJECT_ROOT, file_hash, object_hash, read_json, write_json, write_records,
    resolve, run_lock,
)
from taxosafe_discovery.protocol import code_files as inherited_code_files

SCHEMA_VERSION = "taxosafe_recovery_v1"
DEFAULT_CONFIG = "configs/Zooplankton_Taxonomic_Tree/TaxoSafe_d05_recovery.yml"
ARMS = [
    dict(id="F00_reference", kind="reference", mode="reference", policy="source"),
    dict(id="F01_d05", kind="source", mode="d05", policy="source"),
    dict(id="F02_d05_buffer", kind="reuse", mode="d05", policy="buffer92", weight_source="F01_d05"),
    dict(id="F03_continue", kind="finetune", mode="bce", policy="standard"),
    dict(id="F04_hard_positive", kind="finetune", mode="hard", policy="standard"),
    dict(id="F05_negative_anchor", kind="finetune", mode="anchor", policy="standard"),
    dict(id="F06_l2sp", kind="finetune", mode="l2sp", policy="standard"),
    dict(id="F07_buffer", kind="reuse", mode="l2sp", policy="buffer92", weight_source="F06_l2sp"),
    dict(id="F08_leaf_guard", kind="leaf_guard", mode="leaf_guard", policy="fixed_parent", weight_source="F06_l2sp"),
]
DEFAULTS = dict(
    name="D05 known-retention fine-tuning controls", seed=1, arms=ARMS,
    training=dict(steps_per_head=200, batch_size=512, lr=.0001, hard_weight=1.,
                  negative_anchor_weight=1., l2sp_weight=.1, hardness_eta=2.),
    calibration=dict(seed=1, known_target=.92),
)


def validate_config(cfg):
    if not isinstance(cfg, dict) or set(cfg) != set(DEFAULTS):
        raise ValueError("Unexpected recovery configuration fields")
    if not isinstance(cfg["name"], str) or not cfg["name"].strip():
        raise ValueError("Experiment name must be nonempty")
    if type(cfg["seed"]) is not int or not 0 <= cfg["seed"] < 2**31:
        raise ValueError("seed must be a nonnegative integer below 2**31")
    if cfg["arms"] != ARMS:
        raise ValueError("Use the nine predeclared D05 recovery controls in order")
    for section in ("training", "calibration"):
        values = cfg[section]
        if not isinstance(values, dict) or set(values) != set(DEFAULTS[section]):
            raise ValueError("Unexpected " + section + " settings")
        for key, value in values.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("Expected a finite numeric setting: " + key)
            if key == "seed":
                if type(value) is not int or value != cfg["seed"]:
                    raise ValueError("Calibration seed must match suite seed")
            elif key in ("steps_per_head", "batch_size"):
                if type(value) is not int or value < 1:
                    raise ValueError("Expected a positive integer: " + key)
            elif value <= 0:
                raise ValueError("Expected a positive setting: " + key)
    if cfg["training"]["lr"] > .001:
        raise ValueError("Recovery learning rate must be <= .001")
    if cfg["calibration"]["known_target"] != .92:
        raise ValueError("The buffer control uses the predeclared 92% DEV target")
    return copy.deepcopy(cfg)


def effective_config(path=DEFAULT_CONFIG):
    return validate_config(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def code_files():
    files = set(inherited_code_files())
    files.update((PROJECT_ROOT / "taxosafe_recovery").glob("*.py"))
    for name in ("tools/run_taxosafe_recovery.sh", "tools/pack_taxosafe_recovery_review.py"):
        path = PROJECT_ROOT / name
        if path.is_file():
            files.add(path)
    return sorted(files)


def code_signature():
    return object_hash({str(p.relative_to(PROJECT_ROOT)): file_hash(p) for p in code_files()})


def signature(cfg, source_binding):
    return dict(method=SCHEMA_VERSION, config=object_hash(cfg),
                source=object_hash(source_binding), code=code_signature())
