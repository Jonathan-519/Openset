"""Fixed ablations and immutable bindings for expanded TRAIN outlier exposure."""
import copy
import math
from pathlib import Path
import yaml
from taxosafe_support.protocol import PROJECT_ROOT, file_hash, object_hash, read_json, write_json, write_records, resolve, run_lock
from taxosafe_reference_joint.protocol import code_files as inherited_code_files

SCHEMA_VERSION = "taxosafe_evidence_guard_v1"
DEFAULT_CONFIG = "configs/Zooplankton_Taxonomic_Tree/TaxoSafe_evidence_guard.yml"
DEFAULT_SOURCE = "runs/taxosafe_new/reference/trial_1_retrain_20261003_185220"


def _arm(name, parent=False, fine=False, oe=False, geometry=False, anchor=True, guard=False, source=None):
    return dict(id=name, kind="reference" if name == "R00_reference" else "reuse" if source else "fit",
                adapt_parent=parent, adapt_fine=fine, use_oe=oe, geometry=geometry,
                anchor=anchor, root_guard=guard, weight_source=source)


ARMS = [_arm("R00_reference"), _arm("R01_known_control", True, True),
        _arm("R02_OE_fine", False, True, True), _arm("R03_OE_parent", True, False, True),
        _arm("R04_OE_both", True, True, True), _arm("R05_OE_knn", True, True, True, True),
        _arm("R06_no_anchor", True, True, True, anchor=False),
        _arm("R07_root_guard", True, True, True, guard=True, source="R04_OE_both")]
DEFAULTS = dict(name="Preserved C00 evidence with real TRAIN outlier exposure", seed=1, arms=ARMS,
    data=dict(train_intra="prepro/data/Zooplankton_TT_v9_rebuild/gt_train_intra_v10.txt",
              oe_train="prepro/data/Zooplankton_TT_v9_rebuild/gt_oe_train_v10.txt"),
    training=dict(steps=600, batch_size=32, learning_rate=0.0005, adapter_dim=32, weight_decay=0.0001,
                  feature_bound=0.25, classification=0.5, membership=1.0, distillation=2.0,
                  residual=0.1, log_every=100),
    geometry=dict(neighbors=5, shrinkage=5.0, residual_bound=2.0, hidden=16),
    calibration=dict(seed=1, offset_grid=[-1.0, 0.0, 1.0]))


def validate_config(cfg):
    if not isinstance(cfg, dict) or set(cfg) != set(DEFAULTS) or cfg["arms"] != ARMS:
        raise ValueError("Use all eight declared C00 evidence arms and all configuration fields")
    if not isinstance(cfg["name"], str) or not cfg["name"].strip() or type(cfg["seed"]) is not int or not 0 <= cfg["seed"] < 2**31 - 32:
        raise ValueError("Invalid experiment name or seed")
    for section in ("training", "geometry"):
        if not isinstance(cfg[section], dict) or set(cfg[section]) != set(DEFAULTS[section]):
            raise ValueError("Unexpected settings in " + section)
        for key, default in DEFAULTS[section].items():
            value = cfg[section][key]
            if type(value) is not type(default) or not math.isfinite(value) or value <= 0:
                raise ValueError("Positive finite setting required: " + section + "." + key)
    if cfg["training"]["batch_size"] < 3 or cfg["training"]["feature_bound"] > 0.5:
        raise ValueError("Use all three training statuses and a bounded query residual <= 0.5")
    if cfg["calibration"] != dict(DEFAULTS["calibration"], seed=cfg["seed"]):
        raise ValueError("The small offset grid is fixed before TEST; it is not the proposed learning mechanism")
    if not isinstance(cfg["data"], dict) or set(cfg["data"]) != set(DEFAULTS["data"]):
        raise ValueError("Declare both real unknown TRAIN manifests")
    for value in cfg["data"].values():
        if not isinstance(value, str) or not value.strip():
            raise ValueError("TRAIN manifest must be a nonempty path")
    return copy.deepcopy(cfg)


def effective_config(path=DEFAULT_CONFIG):
    return validate_config(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def code_files():
    paths = set(inherited_code_files())
    paths.update((PROJECT_ROOT / "taxosafe_evidence_guard").glob("*.py"))
    paths.update(PROJECT_ROOT / name for name in ("tools/run_taxosafe_evidence_guard.sh",
        "tools/pack_taxosafe_evidence_guard_review.py", "tools/train_taxosafe_evidence_guard.py",
        "tools/calibrate_taxosafe_evidence_guard.py", "tools/test_taxosafe_evidence_guard.py"))
    return sorted(paths)


def code_signature():
    return object_hash({str(path.relative_to(PROJECT_ROOT)): file_hash(path) for path in code_files()})


def signature(cfg, binding):
    return dict(method=SCHEMA_VERSION, config=object_hash(cfg), source=object_hash(binding), code=code_signature(),
                added_TRAIN_manifests={name: file_hash(resolve(path)) for name, path in sorted(cfg["data"].items())})
