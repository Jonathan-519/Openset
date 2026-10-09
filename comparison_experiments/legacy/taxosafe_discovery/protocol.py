"""Preregistered controls; no mutation of historical reference contracts."""
import copy
import math
from pathlib import Path

import yaml

from taxosafe_support.protocol import (
    PROJECT_ROOT, file_hash, object_hash, read_json, write_json, write_records,
    resolve, run_lock,
)
from taxosafe_routealign.protocol import code_files as inherited_code_files

SCHEMA_VERSION = "taxosafe_discovery_v1"
DEFAULT_CONFIG = "configs/Zooplankton_Taxonomic_Tree/TaxoSafe_reference_discovery.yml"
ARMS = [
    dict(id="D00_reference", kind="baseline", representation="source", method="reference", router="reference", candidate="reference"),
    dict(id="D01_source_rmd", kind="geometry", representation="source", method="rmd", router="global", candidate="reference"),
    dict(id="D02_clip_rmd", kind="geometry", representation="clip", method="rmd", router="global", candidate="reference"),
    dict(id="D03_single_prompt", kind="text", representation="clip", method="single", router="global", candidate="reference"),
    dict(id="D04_domain_prompt", kind="text", representation="clip", method="ensemble", router="global", candidate="reference"),
    dict(id="D05_episode_bce", kind="verifier", representation="clip", method="bce", router="global", candidate="reference"),
    dict(id="D06_episode_rank", kind="verifier", representation="clip", method="bce_rank", router="global", candidate="reference"),
    dict(id="D07_parentwise", kind="reuse", representation="clip", method="bce_rank", router="parentwise", candidate="reference", weight_source="D06_episode_rank"),
    dict(id="D08_residual", kind="projection", representation="clip", method="bce_rank", router="global", candidate="reference"),
    dict(id="D09_coop", kind="prompt", representation="clip", method="coop", router="global", candidate="text"),
    dict(id="D10_forced_prompt", kind="prompt", representation="clip", method="fa", router="global", candidate="text"),
]
DEFAULTS = dict(
    name="TaxoSafe independent rejection and prompt controls",
    seed=1,
    arms=ARMS,
    geometry=dict(shrinkage=.1),
    verifier=dict(folds=3, epochs=100, batch_size=1024, lr=.001, ranking_weight=.2),
    projection=dict(epochs=20, batch_size=64, learning_rate=.001, bottleneck=64,
                    temperature=.1, supcon_weight=.1),
    prompt=dict(epochs=20, batch_size=256, lr=.002, n_ctx=4, reference_weight=3., temperature=1.),
    calibration=dict(seed=1, shrinkage=20., min_parent_known=5),
)


def validate_config(cfg):
    if not isinstance(cfg, dict) or set(cfg) != set(DEFAULTS):
        raise ValueError("Discovery configuration fields differ from the declared protocol")
    if not isinstance(cfg["name"], str) or not cfg["name"].strip():
        raise ValueError("Experiment name must be nonempty")
    if type(cfg["seed"]) is not int or not 0 <= cfg["seed"] < 2**31:
        raise ValueError("seed must be an integer in [0,2**31)")
    if cfg["arms"] != ARMS:
        raise ValueError("Use all eleven preregistered controls in their declared order")
    for section in ("geometry", "verifier", "projection", "prompt", "calibration"):
        values = cfg[section]
        if not isinstance(values, dict) or set(values) != set(DEFAULTS[section]):
            raise ValueError("Unexpected " + section + " settings")
        for key, value in values.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("Non-finite numeric setting: " + section + "." + key)
            if key == "seed":
                if type(value) is not int or value != cfg["seed"]:
                    raise ValueError("Calibration seed must match suite seed")
            elif key in ("epochs", "batch_size", "folds", "bottleneck", "n_ctx", "min_parent_known"):
                if type(value) is not int or value < 1:
                    raise ValueError("Expected a positive integer: " + key)
            elif value <= 0:
                raise ValueError("Expected a positive finite setting: " + key)
    if not 0 < cfg["geometry"]["shrinkage"] <= 1 or not 2 <= cfg["verifier"]["folds"] <= 10:
        raise ValueError("Invalid covariance shrinkage or episodic folds")
    if cfg["prompt"]["n_ctx"] > 16 or cfg["prompt"]["lr"] > .01:
        raise ValueError("Unsafe prompt context length or learning rate")
    return copy.deepcopy(cfg)


def effective_config(path=DEFAULT_CONFIG):
    return validate_config(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def code_files():
    files = set(inherited_code_files())
    files.update((PROJECT_ROOT / "taxosafe_discovery").glob("*.py"))
    for name in ("tools/run_taxosafe_discovery.sh", "tools/pack_taxosafe_discovery_review.py"):
        path = PROJECT_ROOT / name
        if path.is_file():
            files.add(path)
    return sorted(files)


def code_signature():
    return object_hash({str(p.relative_to(PROJECT_ROOT)): file_hash(p) for p in code_files()})


def signature(cfg, source_binding):
    return dict(method=SCHEMA_VERSION, config=object_hash(cfg), reference=object_hash(source_binding), code=code_signature())
