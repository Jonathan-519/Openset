"""Preregistered C00 ablations; inherited code and run contracts are immutable."""
import copy
import math
from pathlib import Path
import yaml
from taxosafe_support.protocol import PROJECT_ROOT, file_hash, object_hash, read_json, write_json, write_records, resolve, run_lock
from taxosafe_morphology.protocol import code_files as inherited_code_files

SCHEMA_VERSION = "taxosafe_reference_joint_v1"
DEFAULT_CONFIG = "configs/Zooplankton_Taxonomic_Tree/TaxoSafe_reference_joint.yml"
DEFAULT_SOURCE = "runs/taxosafe_new/reference/trial_1_retrain_20261003_185220"


def _arm(name, adapter=False, distill=True, adversarial=False, virtual=False, experts=False):
    return dict(id=name, kind="reference" if name == "A00_reference" else "fit",
                adapter=adapter, distill=distill, adversarial=adversarial, virtual=virtual, experts=experts)


ARMS = [_arm("A00_reference"), _arm("A01_joint_frozen"),
        _arm("A02_distilled_adapter", True), _arm("A03_no_distillation", True, False),
        _arm("A04_fine_adversarial", True, adversarial=True),
        _arm("A05_virtual_outliers", True, virtual=True),
        _arm("A06_combined", True, adversarial=True, virtual=True),
        _arm("A07_parent_experts", True, adversarial=True, virtual=True, experts=True)]
DEFAULTS = dict(name="C00 joint terminal states and protected feature adaptation", seed=1, arms=ARMS,
    support=dict(modes=4, folds=3),
    training=dict(steps=1200, batch_size=64, learning_rate=0.0005, adapter_dim=64, hidden=64,
                  weight_decay=0.0001, classification=0.5, distillation=2.0, feature_anchor=2.0,
                  sibling_margin=0.1, adversarial=0.05, virtual=0.5, log_every=100),
    calibration=dict(seed=1, bias_grid=[-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0]))


def validate_config(cfg):
    if not isinstance(cfg, dict) or set(cfg) != set(DEFAULTS) or cfg["arms"] != ARMS:
        raise ValueError("Use the eight preregistered C00 arms with all declared configuration fields")
    if not isinstance(cfg["name"], str) or not cfg["name"].strip() or type(cfg["seed"]) is not int or not 0 <= cfg["seed"] < 2**31:
        raise ValueError("Invalid experiment name or seed")
    for section in ("support", "training"):
        if not isinstance(cfg[section], dict) or set(cfg[section]) != set(DEFAULTS[section]):
            raise ValueError("Unexpected settings in " + section)
        for key, default in DEFAULTS[section].items():
            value = cfg[section][key]
            if type(value) is not type(default) or not math.isfinite(value) or value <= 0:
                raise ValueError("Positive finite setting required: " + section + "." + key)
    if not 2 <= cfg["support"]["folds"] <= 5 or cfg["support"]["modes"] > 8:
        raise ValueError("Support uses 2-5 image folds and at most eight modes")
    if cfg["calibration"] != dict(DEFAULTS["calibration"], seed=cfg["seed"]):
        raise ValueError("Do not tune the declared bias grid or calibration seed after TEST")
    return copy.deepcopy(cfg)


def effective_config(path=DEFAULT_CONFIG):
    return validate_config(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def code_files():
    paths = set(inherited_code_files())
    paths.update((PROJECT_ROOT / "taxosafe_reference_joint").glob("*.py"))
    paths.update(PROJECT_ROOT / name for name in ("tools/run_taxosafe_reference_joint.sh",
        "tools/pack_taxosafe_reference_joint_review.py", "tools/train_taxosafe_reference_joint.py",
        "tools/calibrate_taxosafe_reference_joint.py", "tools/test_taxosafe_reference_joint.py"))
    return sorted(paths)


def code_signature():
    return object_hash({str(p.relative_to(PROJECT_ROOT)): file_hash(p) for p in code_files()})


def signature(cfg, binding):
    return dict(method=SCHEMA_VERSION, config=object_hash(cfg), source=object_hash(binding), code=code_signature())
