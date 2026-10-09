"""Read-only import of a completed discovery D05 and its exact cached inputs.

No migration weakens historical signatures. The parent verifier's geometry,
normalization and templates are loaded, never fitted. Original TEST artifacts
remain unopened until the new recovery development decision is frozen.
"""
import copy
from dataclasses import dataclass
import hashlib
from pathlib import Path

from taxosafe_discovery import protocol as old_protocol
from taxosafe_discovery import runner as old_runner
from taxosafe_discovery import calibration as old_calibration
from taxosafe_refine.importer import inspect_reference

ARM_ID = "D05_episode_bce"
SCHEMA_VERSION = "recovery_d05_import_v1"


def _directory(value):
    path = Path(value).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_dir():
        raise ValueError("D05 import requires a real directory without symlink ancestors: " + str(path))
    return path.resolve()


def _json(path):
    return old_protocol.read_json(old_runner._regular(path))


def _hash(path):
    return old_protocol.file_hash(old_runner._regular(path))


def inspect_d05(discovery_dir):
    """Verify all TRAIN/DEV bytes and links without torch.load or TEST reads."""
    directory = _directory(discovery_dir)
    cfg = old_protocol.validate_config(_json(directory / "config.json"))
    recorded = _json(directory / "source_binding.json")
    reference = inspect_reference(_directory(recorded["directory"]))
    snapshot = old_runner._verify_snapshot(directory, cfg, reference)
    failure_path = directory / "arms" / ARM_ID / "failure.json"
    if failure_path.exists() and _json(failure_path).get("stage") != "test":
        raise ValueError("D05 has a technical training/calibration failure")
    caches = {stage: old_runner._verify_stage(directory, None, "cache_" + stage, snapshot)
              for stage in ("train", "development")}
    training = old_runner._verify_stage(directory, ARM_ID, "training", snapshot)
    calibrated = old_runner._verify_stage(directory, ARM_ID, "calibration", snapshot)
    from taxosafe_discovery.reporting import _stage
    # Saved DEV predictions must reproduce their report/receipt. No original
    # TEST prediction or tensor needs to be opened for this metadata audit.
    _stage(directory, ARM_ID, "calibration", snapshot)
    meta = reference["meta"]
    all_receipts = [*caches.values(), training, calibrated]
    if any(r.get("meta") != meta or r.get("test_used_for_fitting") is not False
           or r.get("unknown_images_used_for_gradients") is not False for r in all_receipts):
        raise ValueError("D05 receipt taxonomy or fit permissions differ")
    if (caches["train"].get("audit") != {"train": reference["training"]["audit"]["train"]}
            or caches["development"].get("audit") != reference["calibration"]["audit"]):
        raise ValueError("D05 cached image audits differ from the immutable reference")
    model = training.get("model")
    if (model != training["artifacts"].get("model") or not isinstance(model, dict)
            or model.get("path") != "model.pth"
            or type(training.get("optimizer_steps")) is not int or training["optimizer_steps"] < 1
            or training.get("weight_source") != ARM_ID
            or training.get("train_cache_sha256") != caches["train"]["artifacts"]["features"]["sha256"]
            or calibrated.get("model_sha256") != model["sha256"]
            or calibrated.get("training_receipt_sha256") != _hash(directory / "arms" / ARM_ID / "training/completed.json")
            or calibrated.get("cache_sha256") != caches["development"]["artifacts"]["features"]["sha256"]
            or len({r.get("inference_spec_sha256") for r in all_receipts}) != 1):
        raise ValueError("D05 model/cache/calibration bindings differ")
    report = _json(directory / "arms" / ARM_ID / "training/training_report.json")
    if report != training.get("fit_report") or report.get("optimizer_steps") != training["optimizer_steps"]:
        raise ValueError("D05 training report differs from its receipt")
    router = _json(directory / "arms" / ARM_ID / "calibration/router.json")
    old_calibration.validate_router(router, meta)
    dev_audit = caches["development"]["audit"]
    dev_hashes = [h for audit in dev_audit.values() for h in audit["image_hashes"]]
    if (router.get("variant") != "global" or router.get("settings") != cfg["calibration"]
            or router.get("fit_completed") is not True or router.get("test_used_for_fitting") is not False
            or router.get("fit_splits") != ["val_known", "val_intra", "val_extra"]
            or sorted(router.get("fit_image_sha256", [])) != sorted(dev_hashes)
            or router.get("unique_image_count") != len(dev_hashes)
            or router.get("input_record_count") != sum(a["count"] for a in dev_audit.values())):
        raise ValueError("D05 router differs from the audited development input")
    paths = ["snapshot.json", "config.json", "source_binding.json"]
    for stage in ("train", "development"):
        paths += ["cache/" + stage + "/" + name for name in ("completed.json", "stage_binding.json", "features.pth")]
    for stage, names in (("training", ("completed.json", "stage_binding.json", "model.pth", "training_report.json")),
                         ("calibration", ("completed.json", "stage_binding.json", "router.json"))):
        paths += ["arms/" + ARM_ID + "/" + stage + "/" + name for name in names]
    binding = dict(schema_version=SCHEMA_VERSION, directory=str(directory),
        reference_binding=copy.deepcopy(reference["binding"]),
        parent=dict(arm_id=ARM_ID, signature=copy.deepcopy(snapshot["signature"]),
                    artifacts={name: _hash(directory / name) for name in paths},
                    model_sha256=model["sha256"], inference_spec_sha256=training["inference_spec_sha256"],
                    optimizer_steps=training["optimizer_steps"]))
    return dict(directory=directory, config=cfg, snapshot=snapshot, reference=reference,
                meta=copy.deepcopy(meta), training=training, calibration=calibrated,
                router=router, caches=caches, binding=binding)


@dataclass
class FrozenD05:
    info: dict
    payload: dict
    geometry: object
    verifier: object
    router: dict
    binding: dict
    meta: dict


def _info(parent):
    return parent.info if isinstance(parent, FrozenD05) else parent


def _refresh(parent):
    previous = _info(parent)
    if not isinstance(previous, dict) or "binding" not in previous:
        raise ValueError("Expected an inspected D05 parent")
    current = inspect_d05(previous["directory"])
    if current["binding"] != previous["binding"]:
        raise ValueError("D05 parent changed after inspection")
    return current


def load_parent_cache(parent, stage):
    if stage not in ("train", "development"):
        raise ValueError("TEST cache requires the separate frozen-development import gate")
    from taxosafe_discovery import backend
    info = _refresh(parent)
    return backend._load_cache(info["directory"], stage, info["config"], info["reference"])


def load_d05(discovery_dir, *, expected_binding=None):
    """Restore exact frozen D05 tensors on CPU, including saved normalization."""
    import torch
    from taxosafe_discovery import backend
    from taxosafe_discovery.geometry import GeometryBank
    from taxosafe_discovery.verifier import SharedVerifier
    from taxosafe_routealign.proximity import _features
    info = inspect_d05(discovery_dir)
    if expected_binding is not None and info["binding"] != expected_binding:
        raise ValueError("D05 parent binding changed")
    arm = next(a for a in info["config"]["arms"] if a["id"] == ARM_ID)
    payload, receipt = backend._load_model(info["directory"], arm, info["config"], info["reference"])
    if (payload.get("projection") is not None or "prompt" in payload
            or not {"geometry", "verifier", "text"} <= set(payload)
            or payload.get("fit_report") != receipt["fit_report"]):
        raise ValueError("D05 payload is not the original geometry/BCE verifier")
    geometry = GeometryBank.from_state_dict(payload["geometry"])
    verifier = SharedVerifier.from_state_dict(payload["verifier"])
    report = receipt["fit_report"]
    if (report.get("geometry") != geometry.fit_report or report.get("verifier") != verifier.fit_report
            or verifier.fit_report["loss"] != "bce" or verifier.dimension != 8
            or verifier.fit_report.get("normalization_fit") != "TRAIN_episode_weighted_mean_std"
            or verifier.fit_report["optimizer_steps"] != receipt["optimizer_steps"]
            or geometry.meta != info["meta"] or geometry.shrinkage != info["config"]["geometry"]["shrinkage"]):
        raise ValueError("D05 tensor training/normalization provenance differs")
    cache, cached = backend._load_cache(info["directory"], "train", info["config"], info["reference"])
    group = cache["groups"]["train"]
    rows = backend._aligned_rows(group)
    labels = torch.tensor([r["true_leaf"] for r in rows], dtype=torch.long)
    expected = _features(group["features"]["clip"], "cached TRAIN CLIP")
    if (list(geometry.image_hashes) != list(group["image_sha256"]) or not torch.equal(geometry.labels, labels)
            or not torch.equal(geometry.fine, expected) or not torch.equal(geometry.parent, expected)):
        raise ValueError("D05 geometry support differs from the exact TRAIN cache")
    episode = verifier.fit_report["episode_report"]
    digest = hashlib.sha256("\n".join(sorted(group["image_sha256"])).encode("utf-8")).hexdigest()
    if (episode.get("image_hash_digest") != digest or episode.get("train_count") != len(labels)
            or episode.get("seed") != info["config"]["seed"]
            or episode.get("folds") != info["config"]["verifier"]["folds"]):
        raise ValueError("D05 normalization/episodes are not bound to this TRAIN cache")
    return FrozenD05(info, payload, geometry, verifier, copy.deepcopy(info["router"]),
                     copy.deepcopy(info["binding"]), copy.deepcopy(info["meta"]))


def load_parent_test_cache(parent, recovery_suite):
    """Read original TEST features only after verifying the new frozen DEV choice."""
    from . import protocol, reporting
    from taxosafe_discovery import backend
    info = _refresh(parent)
    suite = _directory(recovery_suite)
    snapshot = _json(suite / "snapshot.json")
    cfg = protocol.validate_config(_json(suite / "config.json"))
    if (snapshot.get("schema_version") != protocol.SCHEMA_VERSION
            or snapshot.get("source_binding") != info["binding"]
            or _json(suite / "source_binding.json") != info["binding"]
            or snapshot.get("signature") != protocol.signature(cfg, info["binding"])):
        raise ValueError("Recovery TEST cache request belongs to another D05 parent")
    # Checking existence first prevents reporting from creating a missing freeze.
    old_runner._regular(suite / "dev_selection.json")
    reporting.freeze_dev_selection(suite)
    old_runner._verify_stage(info["directory"], None, "cache_test", info["snapshot"])
    cache, receipt = backend._load_cache(info["directory"], "test", info["config"], info["reference"])
    if receipt.get("dev_selection_sha256") != _hash(info["directory"] / "dev_selection.json"):
        raise ValueError("Original TEST cache is not bound to its original DEV decision")
    if receipt["inference_spec_sha256"] != info["training"]["inference_spec_sha256"]:
        raise ValueError("Original TEST cache used a different representation")
    return cache, receipt
