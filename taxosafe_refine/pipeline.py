"""Frozen-reference feature fitting, guarded development calibration, and test.

The original reference run is read-only. Only the reconstruction module learns
from known TRAIN; candidate identities and the source parent threshold stay fixed.
"""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from taxosafe_support import pipeline as support_pipeline
from taxosafe_support import protocol as support_protocol
from taxosafe_support.calibration import raw_records, unique_records as baseline_unique_records
from taxosafe_support.membership_calibration import candidate_scores, RAW_FIELDS
from . import calibration, protocol
from .importer import inspect_reference, load_reference
from .reconstruction import ClassSpecificReconstruction, fit_cached_features

SCHEMA_VERSION = "frozen_reference_reconstruction_v1"
REPRODUCTION_ATOL = 1e-5
REPRODUCTION_RTOL = 1e-5


def _destination(reference_directory, directory):
    reference_directory, directory = Path(reference_directory).resolve(), Path(directory).resolve()
    for child, parent in ((directory, reference_directory), (reference_directory, directory)):
        try:
            child.relative_to(parent)
        except ValueError:
            continue
        raise ValueError("Reference and refinement run directories must be separate, non-nested directories")
    return directory


def _frozen(reference):
    for module in (reference.encoder, reference.evidence):
        if module.training or any(p.requires_grad or p.grad is not None for p in module.parameters()):
            raise ValueError("Every reference parameter must remain frozen in evaluation mode")


@torch.no_grad()
def encode_baseline(encoder, images, text_features=None, classify=True):
    """The original eval arithmetic, plus spatial features, in one visual pass."""
    global_features, spatial = encoder.backbone.encode_image_with_spatial(images, normalize=True)
    parent, parent_local = encoder.parent_branch(global_features, spatial)
    fine, fine_local = (parent, parent_local) if encoder.shared_encoder else encoder.fine_branch(global_features, spatial)
    result = {"parent": parent, "fine": fine,
              "parent_local": parent_local if encoder.local_enabled else None,
              "fine_local": fine_local if encoder.local_enabled else None,
              "raw_global": global_features.float(), "raw_spatial": spatial.float()}
    if classify:
        text = encoder.text_features() if text_features is None else text_features
        count = len(encoder.meta["leaf_names"])
        if text.shape != (count + len(encoder.meta["parent_names"]), encoder.dimension):
            raise ValueError("Reference text feature dimensions changed")
        scale = encoder.backbone.model.logit_scale.exp().float()
        result.update(leaf_logits=scale * fine @ text[:count].T,
                      parent_logits=scale * parent @ text[count:].T)
        if encoder.active_leaf_mask is not None:
            result["leaf_logits"] = result["leaf_logits"].masked_fill(~encoder.active_leaf_mask, -torch.inf)
            result["parent_logits"] = result["parent_logits"].masked_fill(~encoder.active_parent_mask, -torch.inf)
    return result


def load_training_rows(reference):
    """Open only TRAIN, checking its bytes against the source training audit."""
    rows = support_protocol.read_split(reference.config, "train", reference.meta)
    audit = support_protocol.audit_rows({"train": rows})["train"]
    audit["manifest_sha256"] = support_protocol.file_hash(support_protocol.resolve(reference.config["data"]["train"]))
    if audit != reference.training["audit"]["train"]:
        raise ValueError("Known TRAIN data changed since reference training")
    return rows, audit


@torch.no_grad()
def collect_features(groups, cfg, reference, device, model=None, baseline_scores=True, keep_features=True):
    """Content-deduplicated frozen inference; aliases share identical evidence."""
    _frozen(reference)
    text = reference.encoder.text_features() if baseline_scores else None
    records, features, timings = {}, {}, {}
    for split, rows in groups.items():
        canonical = {}
        for row in rows:
            canonical.setdefault(row["image_sha256"], row)
        unique = list(canonical.values())
        if not unique:
            raise ValueError("Empty refinement input split: " + split)
        seen, scored, values = [], [], []
        support_pipeline._sync(device)
        started = time.perf_counter()
        for images, _, indices in support_pipeline.make_loader(unique, reference.config, reference.meta):
            encoded = encode_baseline(reference.encoder, images.to(device), text_features=text, classify=baseline_scores)
            selected = [unique[i] for i in indices.tolist()]
            seen.extend(indices.tolist())
            feature = encoded[cfg["features"]].detach().float()
            if not bool(torch.isfinite(feature).all()):
                raise ValueError("Frozen reference produced non-finite features")
            if keep_features:
                values.append(feature.cpu())
            if baseline_scores:
                output = reference.evidence(encoded, reference.bank)
                batch = raw_records(selected, {"log_probs": output["log_probs"].cpu().numpy()},
                                    encoded["leaf_logits"].cpu().numpy(), reference.meta)
                diagnostic = {key: output[key].cpu().tolist() for key in RAW_FIELDS}
                for i, record in enumerate(batch):
                    record["support_evidence"] = {key: value[i] for key, value in diagnostic.items()}
                if model is not None:
                    identity = candidate_scores(batch, reference.meta)
                    candidate = torch.as_tensor(identity["leaf"], dtype=torch.long, device=device)
                    reconstruction = model.candidate_scores(feature, candidate)
                    if not bool(torch.isfinite(reconstruction).all()):
                        raise ValueError("Reconstruction scores must be finite")
                    for i, record in enumerate(batch):
                        record["reconstruction_score"] = float(reconstruction[i])
                scored.extend(batch)
            else:
                scored.extend(dict(row) for row in selected)
        support_pipeline._sync(device)
        elapsed = time.perf_counter() - started
        if seen != list(range(len(unique))):
            raise ValueError("Frozen inference must visit each unique image once in manifest order")
        by_hash = {row["image_sha256"]: row for row in scored}
        records[split] = [dict(by_hash[row["image_sha256"]], **row) for row in rows]
        if keep_features:
            features[split] = torch.cat(values)
        timings[split] = {"manifest_rows": len(rows), "unique_images": len(unique), "seconds": elapsed,
                          "seconds_per_unique_image": elapsed / len(unique), "visual_passes_per_unique_image": 1}
    return records, features, timings


def _assert_source(reference):
    current = inspect_reference(reference.directory)
    if current["binding"] != reference.binding:
        raise ValueError("Reference source binding changed during this stage")


def fit(cfg, reference_directory, directory, device):
    directory = _destination(reference_directory, directory)
    reference = load_reference(reference_directory, device)
    rows, audit = load_training_rows(reference)
    output = protocol.claim_stage(directory, "refinement")
    sig = protocol.signature(cfg, reference.binding)
    protocol.write_json(output / "config.json", cfg)
    protocol.write_json(output / "source_binding.json", reference.binding)
    protocol.write_json(output / "inputs.json", {"signature": sig, "audit": {"train": audit}, "meta": reference.meta})
    started = time.perf_counter()
    _, cached, timings = collect_features({"train": rows}, cfg, reference, device, baseline_scores=False)
    features = cached["train"].detach().cpu()
    labels = torch.tensor([row["true_leaf"] for row in rows], dtype=torch.long)
    hashes = [row["image_sha256"] for row in rows]
    cache = {"schema_version": SCHEMA_VERSION, "signature": sig, "source_binding": reference.binding,
             "meta": reference.meta, "features": features, "labels": labels, "image_sha256": hashes,
             "source_split": "train", "feature_source": cfg["features"]}
    support_pipeline._save_torch(output / "features.pth", cache)
    fit_started = time.perf_counter()
    model, report = fit_cached_features(features, labels, hashes, len(reference.meta["leaf_names"]),
        **cfg["reconstruction"], **cfg["training"], seed=cfg["seed"], device=device)
    _frozen(reference)
    report = dict(report, gradient_source="known_train_only", feature_source=cfg["features"],
                  frozen_base_trainable=0, refiner_trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  feature_cache_shape=list(features.shape), fit_seconds=time.perf_counter() - fit_started,
                  real_unknown_images_used_for_gradients=False, test_used_for_fitting=False)
    protocol.write_json(output / "training_report.json", report)
    protocol.write_records(output / "train.jsonl", report["history"])
    protocol.write_json(output / "inference_timing.json", timings)
    model_args = dict(model.constructor_arguments(), scale_init=cfg["reconstruction"]["scale_init"])
    support_pipeline._save_torch(output / "reconstruction.pth", {
        "schema_version": SCHEMA_VERSION, "signature": sig, "config": cfg, "meta": reference.meta,
        "source_binding": reference.binding, "model_arguments": model_args,
        "model": support_pipeline._cpu_state(model), "features_sha256": protocol.file_hash(output / "features.pth")})
    _assert_source(reference)
    protocol.require_signature(sig, protocol.signature(cfg, reference.binding))
    receipt = {"schema_version": SCHEMA_VERSION, "signature": sig, "config": cfg, "meta": reference.meta,
               "source_binding": reference.binding, "audit": {"train": audit}, "fit_splits": ["train"],
               "gradient_splits": ["train"],
               "unknown_images_used_for_gradients": False, "test_used_for_fitting": False,
               "frozen_base_trainable": 0, "refiner_trainable": report["refiner_trainable"],
               "feature_cache_shape": list(features.shape), "feature_source": cfg["features"],
               "parent_threshold": reference.router["parent_threshold"], "elapsed_seconds": time.perf_counter() - started,
               "model": {"path": "reconstruction.pth", "sha256": protocol.file_hash(output / "reconstruction.pth")},
               "cache": {"path": "features.pth", "sha256": protocol.file_hash(output / "features.pth")},
               "report": {"path": "training_report.json", "sha256": protocol.file_hash(output / "training_report.json")}}
    protocol.write_json(output / "completed.json", receipt)
    return receipt


def _inspect_fit(cfg, reference, directory):
    output = Path(directory) / "refinement"
    descriptor = protocol.read_json(output / "completed.json")
    for key, name in (("model", "reconstruction.pth"), ("cache", "features.pth"), ("report", "training_report.json")):
        if descriptor.get(key, {}).get("path") != name:
            raise ValueError("Unexpected refinement artifact path: " + key)
    receipt, model_path = protocol.verify_artifact(output, "completed.json", "model")
    protocol.verify_artifact(output, "completed.json", "cache")
    protocol.verify_artifact(output, "completed.json", "report")
    if (receipt.get("schema_version") != SCHEMA_VERSION or receipt.get("config") != cfg
            or protocol.read_json(output / "config.json") != cfg
            or receipt.get("source_binding") != reference.binding
            or protocol.read_json(output / "source_binding.json") != reference.binding
            or receipt.get("meta") != reference.meta
            or receipt.get("audit") != {"train": reference.training["audit"]["train"]}
            or receipt.get("fit_splits") != ["train"]
            or receipt.get("gradient_splits") != ["train"]
            or receipt.get("unknown_images_used_for_gradients") is not False
            or receipt.get("test_used_for_fitting") is not False
            or receipt.get("parent_threshold") != reference.router["parent_threshold"]):
        raise ValueError("Refinement configuration, source binding or fitting audit changed")
    protocol.require_signature(receipt["signature"], protocol.signature(cfg, reference.binding))
    return receipt, model_path


def _load_fit(cfg, reference, directory, device):
    receipt, model_path = _inspect_fit(cfg, reference, directory)
    payload = support_pipeline._load_torch(model_path)
    if (payload.get("schema_version") != SCHEMA_VERSION or payload.get("config") != cfg
            or payload.get("meta") != reference.meta or payload.get("source_binding") != reference.binding
            or payload.get("features_sha256") != receipt["cache"]["sha256"]):
        raise ValueError("Reconstruction checkpoint configuration/source mismatch")
    protocol.require_signature(receipt["signature"], payload.get("signature"))
    arguments = payload["model_arguments"]
    if (arguments.get("dimension") != reference.encoder.dimension
            or arguments.get("num_classes") != len(reference.meta["leaf_names"])
            or any(arguments.get(key) != value for key, value in cfg["reconstruction"].items())):
        raise ValueError("Reconstruction constructor differs from the saved configuration")
    model = ClassSpecificReconstruction(**arguments).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.requires_grad_(False).eval()
    return model, receipt


def _stage_rows(reference, stage):
    forbidden = set(reference.training["audit"]["train"]["image_hashes"])
    sources = set()
    if stage == "test":
        for split, audit in reference.calibration["audit"].items():
            forbidden.update(audit["image_hashes"])
            if split != "val_known":
                sources.update(audit["sources"])
    groups, audit = support_pipeline.load_stage_rows(reference.config, stage, reference.meta,
                                                    forbidden_hashes=forbidden, forbidden_sources=sources)
    for split, value in audit.items():
        if split in reference.audit and value != reference.audit[split]:
            raise ValueError("Reference " + split + " data changed since its frozen stage")
    return groups, audit


def _read_records(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _check_baseline_development(reference, records):
    """Check the source's semantic evidence hash, raw scores, and every decision."""
    old = _read_records(reference.directory / "calibration/development_scores.jsonl")
    old = baseline_unique_records(old)
    evidence = [{"image_sha256": row["image_sha256"], "split": row["split"], "status": row["status"],
        "true_parent": row.get("true_parent"), "true_leaf": row.get("true_leaf"), "source": row.get("source"),
        "support_evidence": {key: np.asarray(row["support_evidence"][key], dtype=float).tolist() for key in RAW_FIELDS},
        "log_probs": np.asarray(row["log_probs"], dtype=float).tolist()}
        for row in sorted(old, key=lambda row: row["image_sha256"])]
    if protocol.object_hash(evidence) != reference.router["evidence_sha256"]:
        raise ValueError("Saved source development scores do not match the source router evidence hash")
    new = calibration.unique_records([row for rows in records.values() for row in rows])
    by_old, by_new = ({row["image_sha256"]: row for row in rows} for rows in (old, new))
    expected = set(reference.router["fit_image_sha256"])
    if set(by_old) != expected or set(by_new) != expected:
        raise ValueError("Reference development reproduction identities differ")
    maximum = {key: 0. for key in ("log_probs", *RAW_FIELDS)}
    for digest in sorted(expected):
        before, after = by_old[digest], by_new[digest]
        for key in ("split", "status", "true_parent", "true_leaf", "source", "global_pred_leaf"):
            if before.get(key) != after.get(key):
                raise ValueError("Reference development metadata/text candidate mismatch: " + key)
        for key in maximum:
            a = np.asarray(before["log_probs"] if key == "log_probs" else before["support_evidence"][key])
            b = np.asarray(after["log_probs"] if key == "log_probs" else after["support_evidence"][key])
            if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all() or not np.allclose(
                    a, b, atol=REPRODUCTION_ATOL, rtol=REPRODUCTION_RTOL):
                raise ValueError("Reference development numeric reproduction failed: " + key)
            maximum[key] = max(maximum[key], float(np.abs(a - b).max()))
    old_routed = support_pipeline.apply_router([by_old[k] for k in sorted(expected)], reference.router, reference.meta)
    new_routed = support_pipeline.apply_router([by_new[k] for k in sorted(expected)], reference.router, reference.meta)
    for before, after in zip(old_routed, new_routed):
        if any(before.get(key) != after.get(key) for key in (
                "prediction_type", "output_node", "parent", "leaf", "candidate_parent", "candidate_leaf")):
            raise ValueError("Reference development routing changed despite numeric tolerance")
    return {"matched_unique_images": len(expected), "raw_scores_match": True, "decisions_match": True,
            "metadata_and_text_candidates_match": True, "atol": REPRODUCTION_ATOL,
            "rtol": REPRODUCTION_RTOL, "maximum_absolute_difference": maximum,
            "source_evidence_sha256": reference.router["evidence_sha256"]}


def _metrics(routed):
    from metrics_open import evaluate_open_set
    rows = [row for records in routed.values() for row in records]
    return evaluate_open_set(*[[row for row in rows if row["status"] == status] for status in ("known", "intra", "extra")])


def _preservation(baseline, refined, router):
    baseline = calibration.unique_records([r for rows in baseline.values() for r in rows])
    refined = calibration.unique_records([r for rows in refined.values() for r in rows])
    old = {r["image_sha256"]: r for r in baseline}
    if len(old) != len(refined) or set(old) != {r["image_sha256"] for r in refined}:
        raise ValueError("Baseline/refined evaluation identities differ")
    fallback = not router.get("reconstruction_gate_enabled", True)
    for new in refined:
        before = old[new["image_sha256"]]
        if any(before.get(key) != new.get(key) for key in (
                "candidate_parent", "candidate_leaf", "parent_threshold", "parent_membership_score")):
            raise ValueError("Frozen parent evidence/threshold or candidate identity changed")
        if (before["prediction_type"] == "global_unknown") != (new["prediction_type"] == "global_unknown"):
            raise ValueError("Frozen parent routing changed")
        if fallback and any(before.get(key) != new.get(key) for key in ("prediction_type", "output_node", "parent", "leaf")):
            raise ValueError("Fallback must reproduce every baseline decision")
    return {"unique_images": len(refined), "parent_route_preserved": True, "candidate_preserved": True,
            "parent_route_preserved_count": len(refined), "candidate_preserved_count": len(refined),
            "fallback": fallback, "fallback_decisions_exact": fallback}


def calibrate_run(cfg, reference_directory, directory, device):
    directory = _destination(reference_directory, directory)
    reference = load_reference(reference_directory, device)
    model, fitted = _load_fit(cfg, reference, directory, device)
    groups, audit = _stage_rows(reference, "calibrate")
    output = protocol.claim_stage(directory, "calibration")
    records, _, timings = collect_features(groups, cfg, reference, device, model=model, keep_features=False)
    reproduction = _check_baseline_development(reference, records)
    protocol.write_json(output / "baseline_reproduction.json", reproduction)
    settings = dict(cfg["calibration"], baseline_calibration=copy.deepcopy(reference.config["calibration"]))
    router = calibration.calibrate(records["val_known"], records["val_intra"], records["val_extra"],
                                   reference.router, reference.meta, settings)
    router.update(signature=fitted["signature"], model_sha256=fitted["model"]["sha256"],
                  source_binding_sha256=protocol.object_hash(reference.binding))
    routed = {split: calibration.apply_router(rows, router, reference.meta) for split, rows in records.items()}
    baseline = {split: support_pipeline.apply_router(rows, reference.router, reference.meta) for split, rows in records.items()}
    preservation = _preservation(baseline, routed, router)
    report = router["validation_report"]
    checked = calibration.evaluate_records([r for rows in routed.values() for r in rows], reference.meta)
    if any(report[key] != checked[key] for key in ("counts", "checks", "metrics", "targets_passed")):
        raise ValueError("Frozen refinement decoder disagrees with its development report")
    protocol.write_json(output / "router.json", router)
    protocol.write_json(output / "validation_report.json", report)
    protocol.write_json(output / "baseline_validation_report.json", calibration.evaluate_records(
        [r for rows in baseline.values() for r in rows], reference.meta))
    protocol.write_json(output / "development_metrics.json", _metrics(routed))
    protocol.write_json(output / "baseline_metrics.json", _metrics(baseline))
    protocol.write_json(output / "preservation.json", preservation)
    protocol.write_json(output / "inference_timing.json", timings)
    protocol.write_records(output / "development_scores.jsonl", [r for rows in records.values() for r in rows])
    protocol.write_records(output / "development_predictions.jsonl", [r for rows in routed.values() for r in rows])
    _assert_source(reference)
    protocol.require_signature(fitted["signature"], protocol.signature(cfg, reference.binding))
    receipt = {"schema_version": SCHEMA_VERSION, "signature": fitted["signature"], "source_binding": reference.binding,
               "audit": audit, "fit_completed": True, "test_used_for_fitting": False,
               "fit_splits": list(groups), "targets_passed": report["targets_passed"],
               "model_sha256": fitted["model"]["sha256"], "parent_threshold": reference.router["parent_threshold"],
               "router": {"path": "router.json", "sha256": protocol.file_hash(output / "router.json")}}
    protocol.write_json(output / "completed.json", receipt)
    return receipt


def test_run(cfg, reference_directory, directory, device):
    directory = _destination(reference_directory, directory)
    reference = load_reference(reference_directory, device)
    model, fitted = _load_fit(cfg, reference, directory, device)
    calibrated, router_path = protocol.verify_artifact(Path(directory) / "calibration", "completed.json", "router")
    router = protocol.read_json(router_path)
    if (calibrated.get("schema_version") != SCHEMA_VERSION or calibrated.get("source_binding") != reference.binding
            or calibrated.get("test_used_for_fitting") is not False
            or calibrated.get("fit_splits") != list(support_protocol.STAGE_SPLITS["calibrate"])
            or calibrated.get("audit") != reference.calibration["audit"]
            or calibrated.get("model_sha256") != fitted["model"]["sha256"]
            or router.get("model_sha256") != fitted["model"]["sha256"]
            or router.get("source_binding_sha256") != protocol.object_hash(reference.binding)
            or router.get("baseline_router") != reference.router):
        raise ValueError("Refinement calibration is not bound to the frozen source/model")
    protocol.require_signature(fitted["signature"], calibrated.get("signature"))
    protocol.require_signature(fitted["signature"], router.get("signature"))
    groups, audit = _stage_rows(reference, "test")
    output = protocol.claim_stage(directory, "test")
    records, _, timings = collect_features(groups, cfg, reference, device, model=model, keep_features=False)
    routed = {split: calibration.apply_router(rows, router, reference.meta) for split, rows in records.items()}
    baseline = {split: support_pipeline.apply_router(rows, reference.router, reference.meta) for split, rows in records.items()}
    preservation = _preservation(baseline, routed, router)
    unique = {split: calibration.unique_records(rows) for split, rows in routed.items()}
    baseline_unique = {split: calibration.unique_records(rows) for split, rows in baseline.items()}
    metrics, baseline_metrics = _metrics(unique), _metrics(baseline_unique)
    summary = calibration.evaluate_records([r for rows in routed.values() for r in rows], reference.meta)
    baseline_summary = calibration.evaluate_records([r for rows in baseline.values() for r in rows], reference.meta)
    gates = calibration.evaluate_gates(metrics)
    for group in (routed, baseline):
        for rows in group.values():
            seen = set()
            for row in rows:
                row["evaluation_weight"] = int(row["image_sha256"] not in seen)
                seen.add(row["image_sha256"])
    for name, value in {"metrics": metrics, "metrics_all_rows": _metrics(routed), "summary": summary,
                        "baseline_metrics": baseline_metrics, "baseline_summary": baseline_summary,
                        "preservation": preservation, "gates": gates, "inference_timing": timings}.items():
        protocol.write_json(output / (name + ".json"), value)
    protocol.write_records(output / "predictions.jsonl", [r for rows in routed.values() for r in rows])
    protocol.write_records(output / "baseline_predictions.jsonl", [r for rows in baseline.values() for r in rows])
    _assert_source(reference)
    protocol.require_signature(fitted["signature"], protocol.signature(cfg, reference.binding))
    receipt = {"schema_version": SCHEMA_VERSION, "signature": fitted["signature"], "source_binding": reference.binding,
               "audit": audit, "test_used_for_fitting": False, "model_sha256": fitted["model"]["sha256"],
               "router_sha256": calibrated["router"]["sha256"], "metric_unit": "unique_image_sha256",
               "targets_passed": summary["targets_passed"], **preservation}
    protocol.write_json(output / "completed.json", receipt)
    return receipt


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("fit", "calibrate", "test"))
    parser.add_argument("--reference-run-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config", default=protocol.DEFAULT_CONFIG)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--preflight", action="store_true", help="Validate configuration and source receipts, without images or CLIP")
    args = parser.parse_args()
    config_path = Path(args.config) if Path(args.config).is_file() else protocol.resolve(args.config)
    cfg = protocol.effective_config(config_path, seed=args.seed)
    reference_directory = protocol.resolve(args.reference_run_dir)
    directory = _destination(reference_directory, protocol.resolve(args.run_dir))
    if args.preflight:
        source = inspect_reference(reference_directory)
        print(json.dumps({"stage": args.stage, "source_binding": source["binding"],
                          "signature": protocol.signature(cfg, source["binding"]),
                          "dataset_images_opened": False, "checkpoint_tensors_loaded": False}, indent=2))
        return
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; use --preflight for receipts or --device cpu explicitly")
    with protocol.run_lock(directory):
        receipt = {"fit": fit, "calibrate": calibrate_run, "test": test_run}[args.stage](
            cfg, reference_directory, directory, torch.device(args.device))
    print(json.dumps(receipt, indent=2))
