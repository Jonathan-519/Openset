"""Explicit, narrowly allowlisted import of a frozen reference-v3 run.

Historical provenance is retained verbatim. The two reviewed source digests
below identify equivalent v3 execution paths; every other signature field and
every artifact/receipt link must still agree. No legacy signature check changes.
"""
from dataclasses import dataclass
import math
from pathlib import Path

from taxosafe_support import protocol as support_protocol
from taxosafe_dcbs.protocol import normalized_name


APPROVED_SOURCES = {
    ("7b6b7e02dc7b0181d93467e6d5c25bc1e42823d3d3c2c90088c28e8aa66b671b",
     "ef5526b8390b838473272eaaf56dcdb128e37e9e2c3a543896d8cd5a30e9aed4"):
        "54dff0ed81771e4cbc14b844f22ebea06e3f9c8d",
    ("7b6b7e02dc7b0181d93467e6d5c25bc1e42823d3d3c2c90088c28e8aa66b671b",
     "77e239bed3c7a8c89d1fbcfaa8da0d097c202a26a68013dbeb644dba8f22faae"):
        "4394be543badc2a7def2fa5a60b45b531b5a861d",
}
CANDIDATE_RULE = "argmax_parent_ranking_then_argmax_leaf_ranking_within_that_parent"


@dataclass
class FrozenReference:
    directory: Path
    config: dict
    meta: dict
    training: dict
    calibration: dict
    router: dict
    audit: dict
    binding: dict
    encoder: object
    evidence: object
    bank: object


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def verify_source_signature(source, runtime):
    """Permit only reviewed code migrations; configuration/data stay exact."""
    if not isinstance(source, dict) or not isinstance(runtime, dict) or set(source) != set(runtime):
        raise ValueError("Reference signature fields differ")
    if source.get("method") != "support_reference_v3" or runtime.get("method") != "support_reference_v3":
        raise ValueError("Only reference-v3 sources may be imported")
    source_key = (source.get("code"), source.get("support_code"))
    runtime_key = (runtime.get("code"), runtime.get("support_code"))
    if source_key not in APPROVED_SOURCES or runtime_key not in APPROVED_SOURCES:
        raise ValueError("Reference source/runtime code is outside the reviewed digest allowlist")
    if any(source[key] != runtime[key] for key in source if key not in ("code", "support_code")):
        raise ValueError("Reference configuration, taxonomy or preparation audit changed")
    return {"source_commit": APPROVED_SOURCES[source_key], "runtime_commit": APPROVED_SOURCES[runtime_key],
            "source_signature": source, "runtime_signature": runtime,
            "migration": "reviewed reference-v3 execution equivalence; no signature fields ignored"}


def _artifact(directory, receipt, key, name):
    descriptor = receipt.get(key, {})
    if descriptor.get("path") != name or not _digest(descriptor.get("sha256")):
        raise ValueError("Invalid reference artifact descriptor: " + key)
    path = directory / name
    try:
        path.resolve().relative_to(directory.resolve())
    except ValueError as exc:
        raise ValueError("Escaping reference artifact: " + str(path)) from exc
    if not path.is_file():
        raise ValueError("Missing or escaping reference artifact: " + str(path))
    if support_protocol.file_hash(path) != descriptor["sha256"]:
        raise ValueError("Reference artifact hash mismatch: " + str(path))
    return path


def _receipt(receipt, signature):
    if receipt.get("schema_version") != 1 or receipt.get("method") != "support_conditioned":
        raise ValueError("Unsupported reference receipt schema")
    support_protocol.require_signature(signature, receipt.get("signature"))
    if receipt.get("test_used_for_fitting") is not False:
        raise ValueError("Reference receipt does not establish test isolation")


def _validate_audit(audit, splits, allow_duplicates=False):
    if not isinstance(audit, dict) or set(audit) != set(splits):
        raise ValueError("Reference split audit is incomplete")
    for split, value in audit.items():
        hashes, sources = value.get("image_hashes"), value.get("sources")
        if (not isinstance(hashes, list) or not hashes or any(not _digest(h) for h in hashes)
                or hashes != sorted(set(hashes))):
            raise ValueError("Invalid reference image identity audit: " + split)
        if not isinstance(sources, list) or not sources or any(not isinstance(s, str) or not s for s in sources):
            raise ValueError("Invalid reference source audit: " + split)
        count, unique = value.get("count"), value.get("unique_image_count")
        if (type(count) is not int or type(unique) is not int or unique != len(hashes)
                or count < unique or (not allow_duplicates and count != unique)
                or not _digest(value.get("manifest_sha256"))):
            raise ValueError("Invalid reference audit counts/manifest digest: " + split)


def _audit_isolation(audits):
    seen = set()
    for split, audit in audits.items():
        hashes = set(audit["image_hashes"])
        if seen & hashes:
            raise ValueError("Reference TRAIN/development/test content overlap: " + split)
        seen.update(hashes)
    development_sources = {normalized_name(name)
        for split in ("val_intra", "val_extra") for name in audits[split]["sources"]}
    for split in ("test_intra", "test_extra"):
        if split in audits and development_sources & {
                normalized_name(name) for name in audits[split]["sources"]}:
            raise ValueError("Reference development/test unknown-source overlap")


def _validate_router(router, config, meta, calibration):
    from taxosafe_support.calibration import TARGETS
    if (router.get("schema_version") != "support_membership_v1" or router.get("decoder") != "membership"
            or router.get("candidate_rule") != CANDIDATE_RULE or router.get("meta") != meta
            or router.get("fit_completed") is not True
            or router.get("fit_splits") != list(support_protocol.STAGE_SPLITS["calibrate"])
            or router.get("fitted_parameters") != ["parent_threshold", "leaf_threshold"]
            or router.get("targets") != TARGETS):
        raise ValueError("Reference router schema, candidate rule or fitting domain differs")
    for key in ("parent_threshold", "leaf_threshold"):
        if isinstance(router.get(key), bool) or not isinstance(router.get(key), (float, int)) or not math.isfinite(router[key]):
            raise ValueError("Reference thresholds must be finite logits")
    if router.get("selection_policy") != config["calibration"].get("policy", "known_first"):
        raise ValueError("Reference router calibration policy differs from its configuration")
    image_hashes = sorted(h for audit in calibration["audit"].values() for h in audit["image_hashes"])
    if (router.get("fit_image_sha256") != image_hashes
            or router.get("unique_image_count") != len(image_hashes)
            or router.get("input_record_count") != sum(a["count"] for a in calibration["audit"].values())
            or router.get("duplicate_record_count") != 0 or not _digest(router.get("evidence_sha256"))):
        raise ValueError("Reference router development identities differ from the audited inputs")
    binding = {"schema_version": "support_membership_v1", "decoder": "membership", "meta": meta,
               "candidate_rule": CANDIDATE_RULE, "evidence_sha256": router["evidence_sha256"],
               "parent_threshold": router["parent_threshold"], "leaf_threshold": router["leaf_threshold"],
               "selection_policy": router["selection_policy"], "grid": router["grid"], "targets": TARGETS}
    if support_protocol.object_hash(binding) != router.get("calibration_sha256"):
        raise ValueError("Reference router semantic calibration hash mismatch")


def inspect_reference(directory):
    """Validate receipts and binary digests without opening any dataset images.

    A completed source TEST is optional. When present only its receipt/audit is
    read; fitting never opens its predictions or images. Source development
    scores are reproduced and checked by the separate calibration stage.
    """
    directory = Path(directory).resolve()
    read = lambda relative: support_protocol.read_json(directory / relative)
    training, calibration = read("training/completed.json"), read("calibration/completed.json")
    config, inputs = read("training/config.json"), read("training/inputs.json")
    if config != training.get("config") or inputs != {
            "signature": training.get("signature"), "audit": training.get("audit"), "meta": training.get("meta")}:
        raise ValueError("Reference configuration/inputs disagree with its training receipt")
    if (config.get("support", {}).get("membership") != "reference"
            or config["support"].get("decoupled") is not True
            or config.get("calibration", {}).get("decoder") != "membership"
            or config.get("model", {}).get("arch") != "maple" or config.get("strict_holdout")
            or config.get("checkpoint") or config.get("init_checkpoint") or config["model"].get("pretrained")):
        raise ValueError("Expected a complete reference-v3 baseline, not a fold or another method")
    signature = training["signature"]
    migration = verify_source_signature(signature, support_protocol.signature(config))
    if signature.get("config") != support_protocol.object_hash(config):
        raise ValueError("Reference configuration digest mismatch")
    _receipt(training, signature)
    _receipt(calibration, signature)
    if (training.get("debug") is not False or training.get("gradient_splits") != ["train"]
            or training.get("support_splits") != ["train"]
            or training.get("unknown_images_used_for_gradients") is not False
            or calibration.get("fit_completed") is not True
            or calibration.get("fit_splits") != list(support_protocol.STAGE_SPLITS["calibrate"])):
        raise ValueError("Reference training/calibration permissions are not established")
    _validate_audit(training.get("audit"), support_protocol.STAGE_SPLITS["train"])
    _validate_audit(calibration.get("audit"), support_protocol.STAGE_SPLITS["calibrate"])
    if training["audit"]["val_known"] != calibration["audit"]["val_known"]:
        raise ValueError("Reference model-selection and calibration known images differ")
    checkpoint_path = _artifact(directory / "training", training, "checkpoint", "best.pth")
    support_path = _artifact(directory / "training", training, "support", "support.pth")
    router_path = _artifact(directory / "calibration", calibration, "router", "router.json")
    router = read("calibration/router.json")
    support_protocol.require_signature(signature, router.get("signature"))
    for key in ("checkpoint", "support"):
        digest = training[key]["sha256"]
        if calibration.get(key + "_sha256") != digest or router.get(key + "_sha256") != digest:
            raise ValueError("Reference checkpoint/support/router hash chain differs")
    _validate_router(router, config, training["meta"], calibration)
    audit = dict(training["audit"], **calibration["audit"])
    receipt_paths = ["training/completed.json", "training/config.json", "training/inputs.json",
                     "calibration/completed.json"]
    test_receipt = directory / "test/completed.json"
    if test_receipt.exists():
        test = read("test/completed.json")
        _receipt(test, signature)
        _validate_audit(test.get("audit"), support_protocol.STAGE_SPLITS["test"], allow_duplicates=True)
        if (test.get("checkpoint_sha256") != training["checkpoint"]["sha256"]
                or test.get("support_sha256") != training["support"]["sha256"]
                or test.get("router_sha256") != calibration["router"]["sha256"]
                or test.get("metric_unit") != "unique_image_sha256"):
            raise ValueError("Reference TEST receipt is not bound to the frozen baseline")
        audit.update(test["audit"])
        receipt_paths.append("test/completed.json")
    _audit_isolation(audit)
    binding = {"schema_version": "reference_v3_import_v1", "directory": str(directory), **migration,
               "receipt_sha256": {name: support_protocol.file_hash(directory / name) for name in receipt_paths},
               "checkpoint_sha256": training["checkpoint"]["sha256"],
               "support_sha256": training["support"]["sha256"],
               "router_sha256": calibration["router"]["sha256"],
               "parent_threshold": router["parent_threshold"], "audit_sha256": support_protocol.object_hash(audit)}
    return {"directory": directory, "config": config, "meta": training["meta"], "training": training,
            "calibration": calibration, "router": router, "audit": audit, "binding": binding,
            "checkpoint_path": checkpoint_path, "support_path": support_path, "router_path": router_path}


def load_reference(directory, device):
    """Load the reviewed v3 path with strict tensor/state and bank validation."""
    from taxosafe_support import pipeline as support_pipeline
    from taxosafe_support.support import SupportBank
    source = inspect_reference(directory)
    config, training, meta = source["config"], source["training"], source["meta"]
    # Match the historical CLI's deterministic backend settings before any
    # CLIP/module construction or image inference takes place.
    support_pipeline.seed_all(config["seed"])
    checkpoint = support_pipeline._load_torch(source["checkpoint_path"])
    payload = support_pipeline._load_torch(source["support_path"])
    if (support_pipeline.hierarchy(config) != meta or checkpoint.get("schema_version") != 1
            or checkpoint.get("method") != "support_conditioned" or checkpoint.get("config") != config
            or checkpoint.get("meta") != meta or checkpoint.get("epoch") != training.get("best_epoch")
            or checkpoint.get("validation") != training.get("known_validation")):
        raise ValueError("Reference checkpoint schema, taxonomy or selection receipt differs")
    if (payload.get("schema_version") != 1 or payload.get("meta") != meta
            or payload.get("checkpoint_sha256") != training["checkpoint"]["sha256"]
            or payload.get("gradient_splits") != ["train"]):
        raise ValueError("Reference support checkpoint/provenance mismatch")
    support_protocol.require_signature(training["signature"], payload.get("signature"))
    bank = SupportBank.from_state_dict(payload["bank"]).to(device)
    if (not set(bank.hashes).issubset(set(training["audit"]["train"]["image_hashes"]))
            or hasattr(bank, "required_leaf_mask")
            or bank.leaf_to_parent.tolist() != meta["leaf_to_parent"]
            or bank.max_per_leaf != int(config["support"].get("max_per_leaf", 8))):
        raise ValueError("Reference support is not the complete audited known-TRAIN bank")
    encoder = support_pipeline.SupportEncoder(support_pipeline.make_backbone(config, meta, device),
                                               meta, config["support"]).to(device)
    if checkpoint.get("dimension") != encoder.dimension:
        raise ValueError("Reference checkpoint encoder dimension mismatch")
    evidence = support_pipeline._make_evidence(encoder, config, meta, device)
    encoder.load_state_dict(checkpoint["encoder"], strict=True)
    evidence.load_state_dict(checkpoint["evidence"], strict=True)
    encoder.requires_grad_(False).eval()
    evidence.requires_grad_(False).eval()
    return FrozenReference(**{key: source[key] for key in (
        "directory", "config", "meta", "training", "calibration", "router", "audit", "binding")},
        encoder=encoder, evidence=evidence, bank=bank)
