"""New-method config and receipts on top of the v11 data isolation contract."""
import copy
import math
from pathlib import Path

import yaml

from taxosafe_dcbs.protocol import (
    PROJECT_ROOT, STAGE_SPLITS, audit_rows, claim_stage, file_hash, object_hash,
    read_json, read_split, require_signature, resolve, run_lock, split_root,
    verify_artifact, write_json, write_records,
)
from taxosafe_dcbs.protocol import signature as dcbs_signature

VARIANTS = ("main", "no_pair", "no_parent", "shared", "no_local", "classification")


def effective_config(path, variant="main", seed=1):
    cfg = copy.deepcopy(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
    if not isinstance(cfg, dict) or "support" not in cfg or variant not in VARIANTS:
        raise ValueError("A support-conditioned config and supported variant are required")
    cfg["seed"], cfg["variant"] = int(seed), variant
    cfg["data"]["seed"] = int(seed)
    cfg["data"].setdefault("sampler", {}).update(seed=int(seed), holdout_seed=int(seed))
    if cfg.get("checkpoint") or cfg.get("init_checkpoint") or cfg["model"].get("pretrained"):
        raise ValueError("Initialize from original CLIP, never a v10/OE checkpoint")
    if cfg["model"].get("arch") != "maple":
        raise ValueError("The shared visual encoder requires MaPLe ViT")
    if "dcbs" in cfg or any(float(cfg.get("loss", {}).get(k, 0)) for k in ("lambda_oe", "lambda_real_intra")):
        raise ValueError("DCBS synthesis and real unknown gradient losses are not permitted")
    if cfg["data"].get("resize_mode", "center_crop") not in ("center_crop", "letterbox"):
        raise ValueError("Unsupported resize mode")
    settings = cfg["support"]
    if not isinstance(settings.get("decoupled", False), bool):
        raise ValueError("support.decoupled must be a YAML boolean")
    membership = settings.get("membership", "prototype")
    if membership not in ("prototype", "reference", "relation"):
        raise ValueError("support.membership must be prototype, reference or relation")
    if membership in ("reference", "relation") and not settings.get("decoupled", False):
        raise ValueError("Reference membership requires support.decoupled=true")
    if membership in ("reference", "relation") or "reference_topk" in settings:
        topk = settings.get("reference_topk", 2)
        if isinstance(topk, bool) or not isinstance(topk, int) or not 1 <= topk <= int(settings.get("max_per_leaf", 8)):
            raise ValueError("support.reference_topk must be an integer between 1 and max_per_leaf")
    if membership == "relation":
        dimension = settings.get("relation_dim", 32)
        if isinstance(dimension, bool) or not isinstance(dimension, int) or not 1 <= dimension <= 512:
            raise ValueError("support.relation_dim must be an integer in [1,512]")
    elif "relation_dim" in settings or "pair_negative_topk" in settings:
        raise ValueError("Relation settings require support.membership=relation")
    if "pair_negative_topk" in settings:
        topk = settings["pair_negative_topk"]
        if isinstance(topk, bool) or not isinstance(topk, int) or not 1 <= topk <= int(settings.get("max_per_leaf", 8)):
            raise ValueError("support.pair_negative_topk must be an integer between 1 and max_per_leaf")
    decoder = cfg.get("calibration", {}).get("decoder", "joint")
    if decoder not in ("joint", "membership"):
        raise ValueError("calibration.decoder must be joint or membership")
    if decoder == "membership" and not settings.get("decoupled", False):
        raise ValueError("Membership decoding requires support.decoupled=true")
    selection = cfg["training"].get("selection", "text")
    if selection not in ("text", "candidate"):
        raise ValueError("training.selection must be text or candidate")
    if selection == "candidate" and decoder != "membership":
        raise ValueError("Candidate checkpoint selection requires membership decoding")
    if decoder == "membership":
        calibration = cfg["calibration"]
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
    if cfg.get("calibration", {}).get("policy", "balanced") not in ("balanced", "known_first"):
        raise ValueError("Unknown calibration policy")
    weights = settings.setdefault("loss", {})
    if variant == "no_pair":
        weights["paired"] = 0.0
    elif variant == "no_parent":
        settings["parent_holdout_enabled"] = False
    elif variant == "shared":
        settings["shared_encoder"] = True
    elif variant == "no_local":
        settings["local_enabled"] = False
    elif variant == "classification":
        settings["episodes_enabled"] = False
        weights.update(episode=0.0, paired=0.0, control=0.0)
    if any(float(v) < 0 for v in weights.values()):
        raise ValueError("Loss weights must be nonnegative")
    if settings.get("decoupled", False):
        if any(not math.isfinite(float(v)) for v in weights.values()):
            raise ValueError("Decoupled loss weights must be finite")
        margin = float(settings.get("margins", {}).get("representation", .2))
        if not math.isfinite(margin) or not 0 <= margin <= 2:
            raise ValueError("Representation margin must be finite and in [0,2]")
    if int(settings.get("max_per_leaf", 8)) < 2:
        raise ValueError("At least two references per leaf are required before query exclusion")
    training = cfg["training"]
    if int(training["epochs"]) <= int(training["warmup_epochs"]):
        raise ValueError("epochs must exceed warmup_epochs")
    if int(training["warmup_epochs"]) < 1:
        raise ValueError("Known-only warmup is required for the classification anchor")
    if int(training.get("min_episode_epochs", 1)) < 1:
        raise ValueError("At least one episode epoch is required")
    if settings.get("episodes_enabled", True) and int(training["epochs"]) < (
            int(training["warmup_epochs"]) + int(training.get("min_episode_epochs", 1))):
        raise ValueError("epochs cannot cover the required episode exposure")
    expected = {"known_e2e": {"operator": ">", "target": 0.90},
                "near_correct_fallback": {"operator": ">=", "target": 0.85},
                "extra_root_rejection": {"operator": ">", "target": 0.90},
                "open_world_leaf_precision": {"operator": ">", "target": 0.90}}
    if cfg.get("evaluation_gates") != expected:
        raise ValueError("The four agreed evaluation gates cannot be silently relaxed")
    return cfg


def signature(cfg):
    result = dcbs_signature(cfg)
    paths = sorted((PROJECT_ROOT / "taxosafe_support").glob("*.py"))
    paths += [PROJECT_ROOT / name for name in (
        "train_taxosafe_new.py", "calibrate_taxosafe_new.py", "test_taxosafe_new.py",
        "tools/validate_taxosafe_support_holdout.py")]
    result["support_code"] = object_hash({str(p.relative_to(PROJECT_ROOT)): file_hash(p) for p in paths})
    if cfg["support"].get("membership", "prototype") == "relation":
        result["method"] = "support_relation_v4"
    elif cfg["support"].get("membership", "prototype") == "reference":
        result["method"] = "support_reference_v3"
    else:
        result["method"] = "support_decoupled_v2" if cfg["support"].get("decoupled", False) else "support_conditioned_v1"
    return result


def load_stage_rows(cfg, stage, meta, forbidden_hashes=(), forbidden_sources=()):
    if stage not in STAGE_SPLITS:
        raise ValueError("Unknown stage: " + stage)
    groups = {split: read_split(cfg, split, meta) for split in STAGE_SPLITS[stage]}
    audit = audit_rows(groups, forbidden_hashes, forbidden_sources, allow_within_split=stage == "test")
    for split in groups:
        audit[split]["manifest_sha256"] = file_hash(resolve(cfg["data"][split]))
    return groups, audit
