"""TRAIN-only frozen evidence; nested held-known/source DEV audit; opt-in TEST.

All outputs belong to a new directory. This module never changes the frozen
reference or promotes a new leaf prediction. TEST is an evaluation of an
already-selected recipe, and is not a fresh independent benchmark after review.
"""
import argparse
import copy
import json
from pathlib import Path
import time

import torch

from taxosafe_support import pipeline as support_pipeline
from taxosafe_support import protocol as support_protocol
from taxosafe_support.calibration import evaluate_records, unique_records
from taxosafe_refine.importer import inspect_reference, load_reference
from taxosafe_refine.pipeline import (
    _assert_source, _check_baseline_development, _destination, _frozen,
    _metrics, _stage_rows, load_training_rows,
)
from taxosafe_geometry.core import RobustScoreStandardizer
from . import calibration, protocol
from .evidence import ParentEvidenceGeometry, collect_features, fit_evidence
from .reporting import paired_report

SCHEMA_VERSION = protocol.SCHEMA_VERSION
SCALE_NAMES = {"baseline_parent", "baseline_leaf", "geometry_parent", "geometry_leaf",
               "parent_text", "parent_membership", "parent_geometry"}
CAL_ARTIFACTS = {
    "router": "router.json", "fold_plan": "fold_plan.json", "outer_audit": "outer_audit.json",
    "oof_predictions": "oof_predictions.jsonl", "oof_baseline": "oof_baseline_predictions.jsonl",
    "scores": "development_scores.jsonl", "predictions": "development_predictions.jsonl",
    "baseline_predictions": "baseline_predictions.jsonl", "report": "validation_report.json",
    "full_fit_audit": "full_fit_audit.json", "reproduction": "baseline_reproduction.json",
    "component_scores": "component_scores.json", "preservation": "preservation.json",
    "metrics": "development_metrics.json", "baseline_metrics": "baseline_metrics.json",
    "baseline_report": "baseline_validation_report.json", "timing": "inference_timing.json",
}
VALIDATION_SCOPE = "postprocessor_conditional_on_frozen_reference"


def _flat(groups):
    return [row for rows in groups.values() for row in rows]


def _baseline_reproduction(reference, records):
    # The unchanged checker needs an unrelated legacy reconstruction field.
    # A zero on audit copies preserves every score and decision under test.
    copies = {key: [dict(row, reconstruction_score=0.) for row in rows] for key, rows in records.items()}
    return _check_baseline_development(reference, copies)


def _preservation(baseline, routed, mode):
    old = {row["image_sha256"]: row for row in unique_records(_flat(baseline))}
    new = {row["image_sha256"]: row for row in unique_records(_flat(routed))}
    if set(old) != set(new):
        raise ValueError("Baseline and ParentRisk content identities differ")
    changes, all_leaves_preserved = 0, True
    for digest, after in new.items():
        before = old[digest]
        terminal = ("prediction_type", "output_node", "parent", "leaf")
        changed = any(before.get(key) != after.get(key) for key in terminal)
        changes += changed
        if before["prediction_type"] == "known" and changed:
            all_leaves_preserved = False
        if after["prediction_type"] == "known" and (
                before["prediction_type"] != "known" or after["leaf"] != before["leaf"]):
            raise ValueError("ParentRisk must never promote or replace a baseline leaf")
        if mode == "parent_only" and before["prediction_type"] == "known" and changed:
            raise ValueError("parent_only must preserve all original leaf outputs")
    return {"unique_images": len(new), "changed_terminals": changes,
            "new_leaf_promotion_count": 0, "mode": mode,
            "all_original_leaf_outputs_preserved": all_leaves_preserved}


def _routed_metrics(rows):
    metrics = _metrics(rows)
    metrics["score_semantics"] = (
        "Root/local knownness are route indicators, not calibrated probabilities. "
        "Continuous evidence AUROC is reported separately in component_scores.json.")
    return metrics


def fit(cfg, reference_directory, directory, device):
    directory = _destination(reference_directory, directory)
    protocol.ensure_fresh_fit(directory)
    reference = load_reference(reference_directory, device)
    rows, train_audit = load_training_rows(reference)
    output = protocol.claim_stage(directory, protocol.FIT_STAGE)
    sig = protocol.signature(cfg, reference.binding)
    inputs = {"signature": sig, "audit": {"train": train_audit}, "meta": reference.meta}
    protocol.write_json(output / "config.json", cfg)
    protocol.write_json(output / "source_binding.json", reference.binding)
    protocol.write_json(output / "inputs.json", inputs)
    started = time.perf_counter()
    groups, cache, timings = collect_features({"train": rows}, reference, device)
    train = unique_records(groups["train"])
    features = cache["train"]
    if [row["image_sha256"] for row in train] != features["image_sha256"]:
        raise ValueError("TRAIN evidence and feature identity order differ")
    labels = torch.tensor([row["true_leaf"] for row in train], dtype=torch.long)
    cached = {"schema_version": SCHEMA_VERSION, "signature": sig,
              "source_binding": reference.binding, "meta": reference.meta,
              "source_split": "train", **features, "labels": labels}
    support_pipeline._save_torch(output / "cache.pth", cached)
    geometry, scales, scale_audit = fit_evidence(train, features, reference.meta, cfg["geometry"])
    if set(scales) != SCALE_NAMES:
        raise ValueError("ParentRisk requires all seven independent score scales")
    states = {key: value.state_dict() for key, value in scales.items()}
    protocol.write_json(output / "scales.json", {"states": states, "audit": scale_audit})
    report = {"method": SCHEMA_VERSION, "fit_splits": ["train"], "gradient_splits": [],
              "fitting": "closed_form_known_train_statistics", "optimizer_steps": 0,
              "frozen_base_trainable": 0, "unknown_images_used_for_fitting": False,
              "test_used_for_fitting": False, "unique_train_images": len(train),
              "train_query_self_matches_excluded": True,
              "feature_shapes": {key: list(features[key].shape) for key in ("fine", "parent")},
              "score_scale_audit": scale_audit,
              "diagnostics": geometry.diagnostics() if callable(geometry.diagnostics) else geometry.diagnostics,
              "fit_seconds": time.perf_counter() - started}
    protocol.write_json(output / "fit_report.json", report)
    protocol.write_json(output / "inference_timing.json", timings)
    protocol.write_records(output / "train_scores.jsonl", train)
    support_pipeline._save_torch(output / "frozen_evidence.pth", {
        "schema_version": SCHEMA_VERSION, "signature": sig, "config": cfg,
        "meta": reference.meta, "source_binding": reference.binding,
        "geometry": geometry.state_dict(), "scales": states,
        "cache_sha256": protocol.file_hash(output / "cache.pth")})
    _frozen(reference)
    _assert_source(reference)
    protocol.require_signature(sig, protocol.signature(cfg, reference.binding))
    receipt = {"schema_version": SCHEMA_VERSION, "signature": sig, "config": cfg,
               "meta": reference.meta, "source_binding": reference.binding,
               "audit": {"train": train_audit}, "fit_splits": ["train"], "gradient_splits": [],
               "optimizer_steps": 0, "frozen_base_trainable": 0,
               "test_used_for_fitting": False, "unknown_images_used_for_fitting": False,
               "fit_completed": True, **protocol.artifacts(output, protocol.FIT_ARTIFACTS)}
    protocol.write_json(output / "completed.json", receipt)
    return receipt


def _get(reference, key):
    return reference[key] if isinstance(reference, dict) else getattr(reference, key)


def inspect_fit(cfg, reference, directory):
    output = Path(directory) / protocol.FIT_STAGE
    receipt = protocol.read_json(output / "completed.json")
    protocol.verify_artifacts(output, receipt, protocol.FIT_ARTIFACTS)
    binding, meta = _get(reference, "binding"), _get(reference, "meta")
    audit = {"train": _get(reference, "training")["audit"]["train"]}
    expected = {"schema_version": SCHEMA_VERSION, "config": cfg, "meta": meta,
                "source_binding": binding, "audit": audit, "fit_splits": ["train"],
                "gradient_splits": [], "optimizer_steps": 0, "frozen_base_trainable": 0,
                "test_used_for_fitting": False, "unknown_images_used_for_fitting": False,
                "fit_completed": True}
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("ParentRisk fit configuration/source/TRAIN-only audit mismatch")
    protocol.require_signature(receipt.get("signature"), protocol.signature(cfg, binding))
    if (protocol.read_json(output / "config.json") != cfg
            or protocol.read_json(output / "source_binding.json") != binding
            or protocol.read_json(output / "inputs.json") != {
                "signature": receipt["signature"], "audit": audit, "meta": meta}):
        raise ValueError("ParentRisk fit input binding changed")
    return receipt, output / "frozen_evidence.pth"


def _load_fit(cfg, reference, directory):
    receipt, path = inspect_fit(cfg, reference, directory)
    payload = support_pipeline._load_torch(path)
    for key, value in {"schema_version": SCHEMA_VERSION, "config": cfg, "meta": reference.meta,
                       "source_binding": reference.binding, "cache_sha256": receipt["cache"]["sha256"]}.items():
        if payload.get(key) != value:
            raise ValueError("ParentRisk evidence checkpoint mismatch: " + key)
    protocol.require_signature(receipt["signature"], payload.get("signature"))
    saved = protocol.read_json(Path(directory) / protocol.FIT_STAGE / "scales.json")
    if set(payload.get("scales", {})) != SCALE_NAMES or payload["scales"] != saved.get("states"):
        raise ValueError("ParentRisk seven scale states differ from their audit")
    geometry = ParentEvidenceGeometry.from_state_dict(payload["geometry"])
    state = geometry.state_dict()
    if (state["leaf_to_parent"].tolist() != reference.meta["leaf_to_parent"]
            or geometry.fine_dimension != reference.encoder.dimension
            or geometry.parent_dimension != reference.encoder.dimension
            or any(state[key] != value for key, value in cfg["geometry"].items())
            or int(state["leaf_counts"].sum()) != reference.training["audit"]["train"]["unique_image_count"]):
        raise ValueError("ParentRisk geometry differs from frozen taxonomy/TRAIN/configuration")
    scales = {key: RobustScoreStandardizer.from_state_dict(value) for key, value in payload["scales"].items()}
    train_hashes = set(reference.training["audit"]["train"]["image_hashes"])
    if set(saved.get("audit", {})) != SCALE_NAMES:
        raise ValueError("ParentRisk requires seven TRAIN scale provenance audits")
    for key in SCALE_NAMES:
        audit = saved["audit"][key]
        hashes = audit.get("image_sha256", [])
        parent_vector = key in {"parent_text", "parent_membership", "parent_geometry"}
        selection = ("true_parent_per_known_train_image" if parent_vector else
                     "correct_parent_candidate" if key.endswith("parent") else "correct_leaf_candidate")
        if (audit.get("source_split") != "train" or audit.get("selection") != selection
                or type(audit.get("count")) is not int or audit["count"] < 1
                or audit["count"] != len(hashes) or len(set(hashes)) != len(hashes)
                or not set(hashes) <= train_hashes
                or payload["scales"][key]["fit_rows"] != audit["count"]
                or (parent_vector and (set(hashes) != train_hashes or audit.get("parameter_sharing") !=
                                       "one_transform_per_score_family_shared_across_parents"))):
            raise ValueError("ParentRisk score scale is not bound to audited known TRAIN: " + key)
    return geometry, scales, receipt


def _calibration_receipt(cfg, reference, directory, fitted):
    output = Path(directory) / "calibration"
    receipt = protocol.read_json(output / "completed.json")
    protocol.verify_artifacts(output, receipt, CAL_ARTIFACTS)
    binding = _get(reference, "binding")
    if (receipt.get("schema_version") != SCHEMA_VERSION or receipt.get("source_binding") != binding
            or receipt.get("model_sha256") != fitted["model"]["sha256"]
            or receipt.get("fit_receipt_sha256") != protocol.file_hash(
                Path(directory) / protocol.FIT_STAGE / "completed.json")
            or receipt.get("audit") != _get(reference, "calibration")["audit"]
            or receipt.get("fit_splits") != list(support_protocol.STAGE_SPLITS["calibrate"])
            or receipt.get("test_used_for_fitting") is not False
            or receipt.get("validation_scope") != VALIDATION_SCOPE
            or receipt.get("independent_model_level_validation") is not False):
        raise ValueError("ParentRisk calibration is not bound to the audited TRAIN/DEV chain")
    protocol.require_signature(fitted["signature"], receipt.get("signature"))
    router = protocol.read_json(output / "router.json")
    protocol.require_signature(fitted["signature"], router.get("signature"))
    if (router.get("baseline_router") != _get(reference, "router")
            or router.get("model_sha256") != fitted["model"]["sha256"]
            or router.get("source_binding_sha256") != protocol.object_hash(binding)
            or router.get("fold_plan_sha256") != receipt["fold_plan"]["sha256"]):
        raise ValueError("ParentRisk router evidence/source/fold-plan binding changed")
    return receipt, router


def calibrate_run(cfg, reference_directory, directory, device):
    directory = _destination(reference_directory, directory)
    reference = load_reference(reference_directory, device)
    geometry, scales, fitted = _load_fit(cfg, reference, directory)
    groups, audit = _stage_rows(reference, "calibrate")
    output = protocol.claim_stage(directory, "calibration")
    records, _, timings = collect_features(groups, reference, device, geometry, scales, keep_features=False)
    reproduction = _baseline_reproduction(reference, records)
    options = dict(cfg["calibration"], seed=cfg["seed"],
                   baseline_calibration=copy.deepcopy(reference.config["calibration"]))
    router = calibration.calibrate(records["val_known"], records["val_intra"], records["val_extra"],
                                   reference.router, reference.meta, options)
    outer = router["outer_audit"]
    protocol.write_json(output / "fold_plan.json", outer["fold_plan"])
    router.update(signature=fitted["signature"], model_sha256=fitted["model"]["sha256"],
                  source_binding_sha256=protocol.object_hash(reference.binding),
                  fold_plan_sha256=protocol.file_hash(output / "fold_plan.json"))
    routed = {key: calibration.apply_router(rows, router, reference.meta) for key, rows in records.items()}
    baseline = {key: support_pipeline.apply_router(rows, reference.router, reference.meta) for key, rows in records.items()}
    preservation = _preservation(baseline, routed, cfg["calibration"]["mode"])
    report = router["validation_report"]
    checked = evaluate_records(_flat(routed), reference.meta)
    if any(report[key] != checked[key] for key in ("counts", "checks", "metrics", "targets_passed")):
        raise ValueError("ParentRisk production router disagrees with its DEV report")
    for name, value in {
        "router": router, "outer_audit": outer, "full_fit_audit": router["full_fit_audit"],
        "validation_report": report, "baseline_reproduction": reproduction,
        "baseline_validation_report": evaluate_records(_flat(baseline), reference.meta),
        "development_metrics": _routed_metrics(routed), "baseline_metrics": _metrics(baseline),
        "preservation": preservation, "component_scores": calibration.score_diagnostics(_flat(records)),
        "inference_timing": timings,
    }.items():
        protocol.write_json(output / (name + ".json"), value)
    for name, rows in {
        "oof_predictions": outer["predictions"], "oof_baseline_predictions": outer["baseline_predictions"],
        "development_scores": _flat(records), "development_predictions": _flat(routed),
        "baseline_predictions": _flat(baseline),
    }.items():
        protocol.write_records(output / (name + ".jsonl"), rows)
    _frozen(reference)
    _assert_source(reference)
    protocol.require_signature(fitted["signature"], protocol.signature(cfg, reference.binding))
    receipt = {"schema_version": SCHEMA_VERSION, "signature": fitted["signature"],
               "source_binding": reference.binding, "audit": audit, "fit_completed": True,
               "fit_splits": list(support_protocol.STAGE_SPLITS["calibrate"]), "test_used_for_fitting": False,
               "model_sha256": fitted["model"]["sha256"],
               "fit_receipt_sha256": protocol.file_hash(Path(directory) / protocol.FIT_STAGE / "completed.json"),
               "validation_scope": VALIDATION_SCOPE, "independent_model_level_validation": False,
               "targets_passed": report["targets_passed"], "development_targets_passed": report["targets_passed"],
               "workflow_completed": True, **protocol.artifacts(output, CAL_ARTIFACTS)}
    protocol.write_json(output / "completed.json", receipt)
    return receipt


def test_run(cfg, reference_directory, directory, device):
    directory = _destination(reference_directory, directory)
    reference = load_reference(reference_directory, device)
    geometry, scales, fitted = _load_fit(cfg, reference, directory)
    calibrated, router = _calibration_receipt(cfg, reference, directory, fitted)
    groups, audit = _stage_rows(reference, "test")
    output = protocol.claim_stage(directory, "test")
    records, _, timings = collect_features(groups, reference, device, geometry, scales, keep_features=False)
    routed = {key: calibration.apply_router(rows, router, reference.meta) for key, rows in records.items()}
    baseline = {key: support_pipeline.apply_router(rows, reference.router, reference.meta) for key, rows in records.items()}
    preservation = _preservation(baseline, routed, cfg["calibration"]["mode"])
    summary = evaluate_records(_flat(routed), reference.meta)
    unique = {key: unique_records(rows) for key, rows in routed.items()}
    base_unique = {key: unique_records(rows) for key, rows in baseline.items()}
    for predictions in (routed, baseline):
        for rows in predictions.values():
            seen = set()
            for row in rows:
                row["evaluation_weight"] = int(row["image_sha256"] not in seen)
                seen.add(row["image_sha256"])
    values = {"summary": summary, "metrics": _routed_metrics(unique),
              "baseline_summary": evaluate_records(_flat(baseline), reference.meta),
              "baseline_metrics": _metrics(base_unique), "preservation": preservation,
              "paired_risk_report": paired_report(_flat(baseline), _flat(routed), reference.meta),
              "component_scores": calibration.score_diagnostics(_flat(records)), "inference_timing": timings}
    for name, value in values.items():
        protocol.write_json(output / (name + ".json"), value)
    protocol.write_records(output / "predictions.jsonl", _flat(routed))
    protocol.write_records(output / "baseline_predictions.jsonl", _flat(baseline))
    _frozen(reference)
    _assert_source(reference)
    protocol.require_signature(fitted["signature"], protocol.signature(cfg, reference.binding))
    names = {key: key + ".json" for key in values}
    names.update(predictions="predictions.jsonl", baseline_predictions="baseline_predictions.jsonl")
    receipt = {"schema_version": SCHEMA_VERSION, "signature": fitted["signature"],
               "source_binding": reference.binding, "audit": audit, "test_used_for_fitting": False,
               "model_sha256": fitted["model"]["sha256"], "router_sha256": calibrated["router"]["sha256"],
               "calibration_receipt_sha256": protocol.file_hash(Path(directory) / "calibration/completed.json"),
               "metric_unit": "unique_image_sha256", "targets_passed": summary["targets_passed"],
               "evaluation_scope": "previously_reviewed_test_fixed_recipe_evaluation",
               "independent_new_test": False, "workflow_completed": True,
               **preservation, **protocol.artifacts(output, names)}
    protocol.write_json(output / "completed.json", receipt)
    return receipt


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("fit", "calibrate", "test"))
    parser.add_argument("--reference-run-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config", default=protocol.DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--preflight", action="store_true", help="Read-only receipt/code checks; no image or tensor loading")
    parser.add_argument("--evaluate-test", action="store_true", help="Explicitly evaluate the already reviewed locked TEST")
    args = parser.parse_args()
    if args.stage == "test" and not args.evaluate_test:
        parser.error("test requires --evaluate-test; default workflow ends after DEV calibration")
    if args.stage != "test" and args.evaluate_test:
        parser.error("--evaluate-test is only valid with the test stage")
    config_path = Path(args.config) if Path(args.config).is_file() else protocol.resolve(args.config)
    cfg = protocol.effective_config(config_path, seed=args.seed)
    reference_directory = protocol.resolve(args.reference_run_dir)
    directory = _destination(reference_directory, protocol.resolve(args.run_dir))
    if args.preflight:
        source = inspect_reference(reference_directory)
        if args.stage == "fit":
            protocol.ensure_fresh_fit(directory)
        else:
            fitted, _ = inspect_fit(cfg, source, directory)
            if args.stage == "test":
                _calibration_receipt(cfg, source, directory, fitted)
        if (Path(directory) / (protocol.FIT_STAGE if args.stage == "fit" else args.stage.replace("calibrate", "calibration"))).exists():
            raise ValueError("Requested stage already exists; do not delete receipts or resume into it")
        print(json.dumps({"stage": args.stage, "source_binding": source["binding"],
                          "signature": protocol.signature(cfg, source["binding"]),
                          "dataset_images_opened": False, "checkpoint_tensors_loaded": False,
                          "source_writes": False, "destination_created": False}, indent=2))
        return
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; use --preflight or explicitly --device cpu")
    with protocol.run_lock(directory):
        receipt = {"fit": fit, "calibrate": calibrate_run, "test": test_run}[args.stage](
            cfg, reference_directory, directory, torch.device(args.device))
    print(json.dumps({"stage": args.stage, "run_dir": str(directory),
                      "workflow_completed": True, "targets_passed": receipt.get("targets_passed"),
                      "receipt": str(Path(directory) / (protocol.FIT_STAGE if args.stage == "fit" else
                                      "calibration" if args.stage == "calibrate" else "test") / "completed.json")}, indent=2))
