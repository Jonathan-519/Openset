"""Predeclared independent parent-domain and conditional-leaf controls.

Historical family source contracts remain byte-identical. New statistics use
known TRAIN only; DEV fixes the domain gate before fitting the leaf gate.
"""
import copy
from pathlib import Path

import yaml

from taxosafe_support.protocol import (
    PROJECT_ROOT, file_hash, object_hash, read_json, write_json, write_records,
    resolve, run_lock,
)
from taxosafe_boundary.protocol import code_files as inherited_code_files

SCHEMA_VERSION = "taxosafe_domain_v1"
DEFAULT_CONFIG = "configs/Zooplankton_Taxonomic_Tree/TaxoSafe_parent_domain.yml"


def _arm(identifier, kind, mode, root, leaf, policy="staged", candidate_policy="reference_path",
         bank_variant="none", weight_source=None):
    value = dict(id=identifier, kind=kind, mode=mode, root=root, leaf=leaf,
                 policy=policy, candidate_policy=candidate_policy, bank_variant=bank_variant)
    if weight_source is not None:
        value["weight_source"] = weight_source
    return value


ARMS = [
    _arm("H00_reference", "reference", "reference", "reference", "reference", "source", "original"),
    _arm("H01_d05", "source", "d05", "d05", "d05", "source", "original"),
    _arm("H02_d05_staged", "evidence", "d05_staged", "d05_candidate", "d05"),
    _arm("H03_d05_marginal", "evidence", "d05_marginal", "d05_marginal", "d05"),
    _arm("H04_subspace_root", "fit", "subspace_root", "residual", "d05", bank_variant="main"),
    _arm("H05_density_root", "reuse", "density_root", "density", "d05", bank_variant="main", weight_source="H04_subspace_root"),
    _arm("H06_dual_root", "reuse", "dual_root", "dual", "d05", bank_variant="main", weight_source="H04_subspace_root"),
    _arm("H07_conditional_leaf", "reuse", "conditional_leaf", "d05_candidate", "conditional", bank_variant="main", weight_source="H04_subspace_root"),
    _arm("H08_dual_conditional", "reuse", "dual_conditional", "dual", "conditional", bank_variant="main", weight_source="H04_subspace_root"),
    _arm("H09_dual_reroute", "reuse", "dual_reroute", "dual", "conditional", candidate_policy="domain_parent_reference_child", bank_variant="main", weight_source="H04_subspace_root"),
    _arm("H10_dual_joint", "reuse", "dual_joint", "dual", "conditional", policy="joint", bank_variant="main", weight_source="H04_subspace_root"),
    _arm("H11_dual_rank16", "fit", "dual_rank16", "dual", "conditional", bank_variant="wide"),
]

DEFAULTS = dict(
    name="Independent parent domains and conditional leaf evidence", seed=1, arms=ARMS,
    bank=dict(parent_rank=8, leaf_rank=4, global_rank=16, shrinkage=.1, folds=3),
    wide_bank=dict(parent_rank=16, leaf_rank=8, global_rank=32, shrinkage=.1, folds=3),
    calibration=dict(seed=1, root_known_target=.92, root_near_target=.85),
)


def validate_config(cfg):
    if not isinstance(cfg, dict) or set(cfg) != set(DEFAULTS):
        raise ValueError("Unexpected parent-domain configuration fields")
    if not isinstance(cfg["name"], str) or not cfg["name"].strip():
        raise ValueError("Experiment name must be nonempty")
    if type(cfg["seed"]) is not int or not 0 <= cfg["seed"] < 2**31:
        raise ValueError("seed must be an integer in [0, 2**31)")
    if cfg["arms"] != ARMS:
        raise ValueError("Use the twelve predeclared parent-domain controls in order")
    for section in ("bank", "wide_bank"):
        values = cfg[section]
        if not isinstance(values, dict) or set(values) != set(DEFAULTS[section]):
            raise ValueError("Unexpected " + section + " settings")
        for key, default in DEFAULTS[section].items():
            value = values[key]
            if type(value) is not type(default) or value != default:
                raise ValueError("Domain bank settings are preregistered: " + section + "." + key)
    settings = cfg["calibration"]
    if not isinstance(settings, dict) or set(settings) != set(DEFAULTS["calibration"]):
        raise ValueError("Unexpected domain calibration fields")
    if type(settings["seed"]) is not int or settings["seed"] != cfg["seed"]:
        raise ValueError("Calibration seed must match suite seed")
    for key in ("root_known_target", "root_near_target"):
        if type(settings[key]) is not float or settings[key] != DEFAULTS["calibration"][key]:
            raise ValueError("Domain coverage targets are preregistered: " + key)
    return copy.deepcopy(cfg)


def bank_settings(cfg, arm):
    variant = arm.get("bank_variant")
    if variant not in ("main", "wide"):
        raise ValueError("This arm does not fit or use a domain bank")
    return dict(cfg["bank" if variant == "main" else "wide_bank"], seed=cfg["seed"])


def effective_config(path=DEFAULT_CONFIG):
    return validate_config(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def code_files():
    files = set(inherited_code_files())
    files.update((PROJECT_ROOT / "taxosafe_domain").glob("*.py"))
    for name in ("tools/run_taxosafe_domain.sh", "tools/pack_taxosafe_domain_review.py"):
        path = PROJECT_ROOT / name
        if path.is_file():
            files.add(path)
    return sorted(files)


def code_signature():
    return object_hash({str(p.relative_to(PROJECT_ROOT)): file_hash(p) for p in code_files()})


def signature(cfg, source_binding):
    return dict(method=SCHEMA_VERSION, config=object_hash(cfg),
                source=object_hash(source_binding), code=code_signature())
