"""Reference-only configuration and complete source signatures for H02."""
import copy
import math
from pathlib import Path

import yaml

from .io import (
    PROJECT_ROOT, STAGE_SPLITS, audit_rows, claim_stage, file_hash, object_hash,
    read_json, read_split, require_signature, resolve, run_lock, split_root,
    verify_artifact, write_json, write_records,
)

VARIANTS = ("main",)
EXPECTED_PARENTS = ("Amphipoda", "Appendiculata", "Cladocera", "Copepoda",
                    "Euphausiacea", "Medusae", "Sagittoidea")
REFERENCE_DEPENDENCIES = (
    "models/__init__.py", "models/maple.py", "models/maple_model.py",
    "models/maple_clip.py", "models/model.py", "models/clip.py",
    "models/simple_tokenizer.py", "models/bpe_simple_vocab_16e6.txt.gz",
    "loader/__init__.py", "loader/hierdata.py", "loader/utils.py",
    "loader/hierarchical_episode_sampler.py", "loader/treelibs.py",
    "taxosafe_episode.py", "metrics_open.py",
)


def validate_config(cfg):
    """Validate without changing the frozen effective configuration content."""
    if not isinstance(cfg, dict) or not isinstance(cfg.get("support"), dict):
        raise ValueError("An H02 reference configuration is required")
    if cfg.get("variant", "main") != "main":
        raise ValueError("Only the reference main training path is retained")
    if cfg.get("strict_holdout"):
        raise ValueError("H02 requires the complete reference, not a strict-holdout fold")
    if cfg.get("checkpoint") or cfg.get("init_checkpoint") or cfg["model"].get("pretrained"):
        raise ValueError("Initialize the reference from the original CLIP core")
    if cfg["model"].get("arch") != "maple":
        raise ValueError("The reference encoder requires MaPLe ViT")
    if "dcbs" in cfg or any(float(cfg.get("loss", {}).get(k, 0)) for k in ("lambda_oe", "lambda_real_intra")):
        raise ValueError("Real unknown gradient losses and historical synthesis are not permitted")
    if cfg["data"].get("resize_mode", "center_crop") not in ("center_crop", "letterbox"):
        raise ValueError("Unsupported resize mode")
    settings = cfg["support"]
    if settings.get("membership") != "reference" or settings.get("decoupled") is not True:
        raise ValueError("Only decoupled reference membership v3 is retained")
    if "relation_dim" in settings or "pair_negative_topk" in settings:
        raise ValueError("The reference uses its original all-pair supervision")
    for key, expected in (("shared_encoder", False), ("local_enabled", True),
                          ("episodes_enabled", True), ("parent_holdout_enabled", True)):
        if settings.get(key, expected) is not expected:
            raise ValueError("Reference setting cannot select a comparison experiment: " + key)
    topk = settings.get("reference_topk", 2)
    if isinstance(topk, bool) or not isinstance(topk, int) or not 1 <= topk <= int(settings.get("max_per_leaf", 8)):
        raise ValueError("support.reference_topk must be an integer between 1 and max_per_leaf")
    calibration = cfg.get("calibration", {})
    if calibration.get("decoder") != "membership" or calibration.get("policy") != "known_first":
        raise ValueError("Reference calibration requires membership and known_first")
    if calibration.get("threshold_grid", "quantile") != "quantile":
        raise ValueError("Membership calibration requires a quantile threshold grid")
    grid_points = calibration.get("membership_grid_points", calibration.get("grid_points", 49))
    if isinstance(grid_points, bool) or not isinstance(grid_points, int) or not 2 <= grid_points <= 401:
        raise ValueError("membership_grid_points must be an integer in [2,401]")
    for key in ("parent_threshold_grid", "leaf_threshold_grid"):
        if key in calibration:
            grid = calibration[key]
            if not isinstance(grid, (list, tuple)) or not 1 <= len(grid) <= 403 or any(
                    isinstance(value, bool) or not isinstance(value, (int, float)) or
                    not math.isfinite(float(value)) for value in grid):
                raise ValueError(key + " must contain finite numeric thresholds")
    weights = settings.get("loss", {})
    if any(not math.isfinite(float(v)) or float(v) < 0 for v in weights.values()):
        raise ValueError("Loss weights must be finite and nonnegative")
    margin = float(settings.get("margins", {}).get("representation", .2))
    if not math.isfinite(margin) or not 0 <= margin <= 2:
        raise ValueError("Representation margin must be finite and in [0,2]")
    if int(settings.get("max_per_leaf", 8)) < 2:
        raise ValueError("At least two references per leaf are required before query exclusion")
    training = cfg["training"]
    if training.get("selection", "text") != "text":
        raise ValueError("The reference keeps its original known-text checkpoint selection")
    if int(training["epochs"]) <= int(training["warmup_epochs"]):
        raise ValueError("epochs must exceed warmup_epochs")
    if int(training["warmup_epochs"]) < 1:
        raise ValueError("Known-only warmup is required for the classification anchor")
    if int(training.get("min_episode_epochs", 1)) < 1:
        raise ValueError("At least one episode epoch is required")
    if int(training["epochs"]) < int(training["warmup_epochs"]) + int(training.get("min_episode_epochs", 1)):
        raise ValueError("epochs cannot cover the required episode exposure")
    expected = {"known_e2e": {"operator": ">", "target": 0.90},
                "near_correct_fallback": {"operator": ">=", "target": 0.85},
                "extra_root_rejection": {"operator": ">", "target": 0.90},
                "open_world_leaf_precision": {"operator": ">", "target": 0.90}}
    if cfg.get("evaluation_gates") != expected:
        raise ValueError("The four agreed evaluation gates cannot be silently relaxed")
    return cfg


def effective_config(path, variant="main", seed=1):
    cfg = copy.deepcopy(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
    if not isinstance(cfg, dict) or "support" not in cfg or variant not in VARIANTS:
        raise ValueError("The H02 reference configuration and variant main are required")
    cfg["seed"], cfg["variant"] = int(seed), variant
    cfg["data"]["seed"] = int(seed)
    cfg["data"].setdefault("sampler", {}).update(seed=int(seed), holdout_seed=int(seed))
    cfg["support"].setdefault("loss", {})
    return validate_config(cfg)


def code_digests():
    """Hash actual runtime files; migration tables are deliberately outside here."""
    dependencies = [PROJECT_ROOT / name for name in REFERENCE_DEPENDENCIES]
    support_files = sorted((PROJECT_ROOT / "taxosafe_support").glob("*.py"))
    return {
        "code": object_hash({str(p.relative_to(PROJECT_ROOT)): file_hash(p) for p in dependencies}),
        "support_code": object_hash({str(p.relative_to(PROJECT_ROOT)): file_hash(p) for p in support_files}),
    }


def signature(cfg):
    validate_config(cfg)
    result = {"config": object_hash(cfg), "hierarchy": file_hash(resolve(cfg["data"]["hierarchy"])),
              **code_digests(), "method": "support_reference_v3"}
    if cfg["data"].get("known_preparation_audit"):
        result["known_preparation_audit"] = file_hash(resolve(cfg["data"]["known_preparation_audit"]))
    return result


def load_stage_rows(cfg, stage, meta, forbidden_hashes=(), forbidden_sources=()):
    if stage not in STAGE_SPLITS:
        raise ValueError("Unknown stage: " + stage)
    groups = {split: read_split(cfg, split, meta) for split in STAGE_SPLITS[stage]}
    audit = audit_rows(groups, forbidden_hashes, forbidden_sources, allow_within_split=stage == "test")
    for split in groups:
        audit[split]["manifest_sha256"] = file_hash(resolve(cfg["data"][split]))
    return groups, audit
