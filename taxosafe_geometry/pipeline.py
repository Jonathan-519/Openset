"""Fit TRAIN geometry, calibrate on DEV and evaluate TEST with a frozen model.

Reference weights, support-bank contents and identity ranking remain unchanged.
The new decoder may change both parent and leaf acceptance using independently
fitted relative-distance evidence. No TEST image participates in fitting.
"""
import argparse
import copy
import json
from pathlib import Path
import time

import torch

from taxosafe_support import pipeline as support_pipeline
from taxosafe_support import protocol as support_protocol
from taxosafe_support.calibration import raw_records, unique_records as baseline_unique_records
from taxosafe_support.membership_calibration import candidate_scores, RAW_FIELDS
from taxosafe_refine.importer import inspect_reference, load_reference
from taxosafe_refine.pipeline import (
    _assert_source, _check_baseline_development, _destination, _frozen,
    _metrics, _stage_rows, encode_baseline, load_training_rows,
)
from . import calibration, protocol
from .core import HierarchicalGeometry, RobustScoreStandardizer

SCHEMA_VERSION = protocol.SCHEMA_VERSION
SCORE_NAMES = ("baseline_parent", "baseline_leaf", "geometry_parent", "geometry_leaf")


def _decoder(cfg):
    if cfg["calibration"].get("decoder") == "local_guarded":
        from . import local
        return local
    return calibration


def _routed_metrics(cfg, rows):
    metrics = _metrics(rows)
    if cfg["calibration"].get("decoder") == "local_guarded":
        metrics["score_semantics"] = "root/local knownness are discrete route indicators; see component_scores.json for continuous evidence AUROC"
    return metrics


@torch.no_grad()
def collect_features(groups, reference, device, geometry=None, scales=None, keep_features=True):
    """One original visual pass per unique image, with both frozen branches."""
    _frozen(reference)
    if (geometry is None) != (scales is None):
        raise ValueError("Geometry and fitted score scales must be supplied together")
    text = reference.encoder.text_features()
    records, features, timings = {}, {}, {}
    for split, rows in groups.items():
        canonical = {}
        for row in rows:
            canonical.setdefault(row["image_sha256"], row)
        unique = list(canonical.values())
        if not unique:
            raise ValueError("Empty geometry split: " + split)
        seen, scored, fine_values, parent_values = [], [], [], []
        support_pipeline._sync(device)
        started = time.perf_counter()
        for images, _, indices in support_pipeline.make_loader(unique, reference.config, reference.meta):
            encoded = encode_baseline(reference.encoder, images.to(device), text_features=text)
            selected = [unique[i] for i in indices.tolist()]
            seen.extend(indices.tolist())
            fine, parent = encoded["fine"].detach().float(), encoded["parent"].detach().float()
            if not bool(torch.isfinite(fine).all() and torch.isfinite(parent).all()):
                raise ValueError("Frozen geometry features must be finite")
            if keep_features:
                fine_values.append(fine.cpu())
                parent_values.append(parent.cpu())
            output = reference.evidence(encoded, reference.bank)
            batch = raw_records(selected, {"log_probs": output["log_probs"].cpu().numpy()},
                                encoded["leaf_logits"].cpu().numpy(), reference.meta)
            raw = {key: output[key].cpu().tolist() for key in RAW_FIELDS}
            for i, record in enumerate(batch):
                record["support_evidence"] = {key: value[i] for key, value in raw.items()}
            if geometry is not None:
                _add_geometry(batch, fine, parent, reference.meta, geometry, scales)
            scored.extend(batch)
        support_pipeline._sync(device)
        elapsed = time.perf_counter() - started
        if seen != list(range(len(unique))):
            raise ValueError("Inference must visit each unique image once in manifest order")
        by_hash = {row["image_sha256"]: row for row in scored}
        records[split] = [dict(by_hash[row["image_sha256"]], **row) for row in rows]
        if keep_features:
            features[split] = {"fine": torch.cat(fine_values), "parent": torch.cat(parent_values),
                               "image_sha256": [row["image_sha256"] for row in unique]}
        timings[split] = {"manifest_rows": len(rows), "unique_images": len(unique), "seconds": elapsed,
                          "seconds_per_unique_image": elapsed / len(unique), "visual_passes_per_unique_image": 1}
    _frozen(reference)
    return records, features, timings


def _selected_scores(records, fine, parent, meta, geometry):
    identity = candidate_scores(records, meta)
    scores = geometry.score(fine, parent,
        torch.as_tensor(identity["parent"], dtype=torch.long, device=parent.device),
        torch.as_tensor(identity["leaf"], dtype=torch.long, device=fine.device))
    result = {"baseline_parent": torch.as_tensor(identity["parent_score"], dtype=torch.float64),
              "baseline_leaf": torch.as_tensor(identity["leaf_score"], dtype=torch.float64),
              "geometry_parent": torch.as_tensor(scores["parent_score"]).detach().cpu(),
              "geometry_leaf": torch.as_tensor(scores["leaf_score"]).detach().cpu()}
    if any(v.shape != (len(records),) or not bool(torch.isfinite(v).all()) for v in result.values()):
        raise ValueError("Candidate geometry scores must be finite scalar vectors")
    return identity, result


def _add_geometry(records, fine, parent, meta, geometry, scales):
    _, scores = _selected_scores(records, fine, parent, meta, geometry)
    normalized = {key: scales[key].transform(value) for key, value in scores.items()}
    for i, row in enumerate(records):
        row.update(baseline_parent_z=float(normalized["baseline_parent"][i]),
                   baseline_leaf_z=float(normalized["baseline_leaf"][i]),
                   geometry_parent_score=float(normalized["geometry_parent"][i]),
                   geometry_leaf_score=float(normalized["geometry_leaf"][i]),
                   geometry_parent_raw=float(scores["geometry_parent"][i]),
                   geometry_leaf_raw=float(scores["geometry_leaf"][i]),
                   baseline_parent_raw=float(scores["baseline_parent"][i]),
                   baseline_leaf_raw=float(scores["baseline_leaf"][i]))


def _fit_scales(records, fine, parent, labels, meta, geometry):
    identity, scores = _selected_scores(records, fine, parent, meta, geometry)
    mapping = torch.as_tensor(meta["leaf_to_parent"], dtype=torch.long)
    correct_parent = torch.as_tensor(identity["parent"], dtype=torch.long) == mapping[labels]
    correct_leaf = correct_parent & (torch.as_tensor(identity["leaf"], dtype=torch.long) == labels)
    masks = {"baseline_parent": correct_parent, "geometry_parent": correct_parent,
             "baseline_leaf": correct_leaf, "geometry_leaf": correct_leaf}
    if not bool(correct_parent.any()) or not bool(correct_leaf.any()):
        raise ValueError("TRAIN needs at least one correct parent and leaf candidate for score normalization")
    scales = {key: RobustScoreStandardizer.fit(scores[key][masks[key]]) for key in SCORE_NAMES}
    audit = {key: {"count": int(masks[key].sum()), "source_split": "train",
                  "selection": "correct_parent_candidate" if key.endswith("parent") else "correct_leaf_candidate",
                  "image_sha256": [row["image_sha256"] for row, keep in zip(records, masks[key].tolist()) if keep]}
             for key in SCORE_NAMES}
    return scales, audit


def fit(cfg, reference_directory, directory, device):
    directory = _destination(reference_directory, directory)
    reference = load_reference(reference_directory, device)
    rows, audit = load_training_rows(reference)
    output = protocol.claim_stage(directory, "geometry")
    signature = protocol.signature(cfg, reference.binding)
    protocol.write_json(output / "config.json", cfg)
    protocol.write_json(output / "source_binding.json", reference.binding)
    inputs = {"signature": signature, "audit": {"train": audit}, "meta": reference.meta}
    protocol.write_json(output / "inputs.json", inputs)
    started = time.perf_counter()
    records, cached, timings = collect_features({"train": rows}, reference, device)
    unique = baseline_unique_records(records["train"])
    features = cached["train"]
    if [row["image_sha256"] for row in unique] != features["image_sha256"]:
        raise ValueError("Deduplicated feature and label order differ")
    labels = torch.tensor([row["true_leaf"] for row in unique], dtype=torch.long)
    cache = {"schema_version": SCHEMA_VERSION, "signature": signature, "source_binding": reference.binding,
             "meta": reference.meta, **features, "labels": labels, "source_split": "train"}
    support_pipeline._save_torch(output / "cache.pth", cache)
    geometry = HierarchicalGeometry.fit(features["fine"], features["parent"], labels,
                                        reference.meta["leaf_to_parent"], **cfg["geometry"])
    scales, scale_audit = _fit_scales(unique, features["fine"], features["parent"], labels, reference.meta, geometry)
    _add_geometry(unique, features["fine"], features["parent"], reference.meta, geometry, scales)
    scale_states = {key: value.state_dict() for key, value in scales.items()}
    protocol.write_json(output / "scales.json", {"states": scale_states, "audit": scale_audit})
    diagnostics = geometry.diagnostics() if callable(geometry.diagnostics) else geometry.diagnostics
    report = {"method": "hierarchical_relative_mahalanobis", "fit_splits": ["train"],
              "fitting": "closed_form_known_train_statistics", "gradient_splits": [],
              "optimizer_steps": 0, "frozen_base_trainable": 0, "unknown_images_used_for_fitting": False,
              "test_used_for_fitting": False, "unique_train_images": len(unique),
              "feature_shapes": {k: list(features[k].shape) for k in ("fine", "parent")},
              "score_scale_audit": scale_audit, "diagnostics": diagnostics,
              "fit_seconds": time.perf_counter() - started}
    protocol.write_json(output / "fit_report.json", report)
    protocol.write_json(output / "inference_timing.json", timings)
    protocol.write_records(output / "train_scores.jsonl", unique)
    support_pipeline._save_torch(output / "frozen_geometry.pth", {
        "schema_version": SCHEMA_VERSION, "signature": signature, "config": cfg, "meta": reference.meta,
        "source_binding": reference.binding, "geometry": geometry.state_dict(), "scales": scale_states,
        "cache_sha256": protocol.file_hash(output / "cache.pth")})
    _frozen(reference)
    _assert_source(reference)
    protocol.require_signature(signature, protocol.signature(cfg, reference.binding))
    receipt = {"schema_version": SCHEMA_VERSION, "signature": signature, "config": cfg, "meta": reference.meta,
               "source_binding": reference.binding, "audit": {"train": audit}, "fit_splits": ["train"],
               "gradient_splits": [], "test_used_for_fitting": False, "unknown_images_used_for_fitting": False,
               "frozen_base_trainable": 0, "optimizer_steps": 0, "elapsed_seconds": time.perf_counter() - started}
    for key, name in (("model", "frozen_geometry.pth"), ("cache", "cache.pth"),
                      ("scales", "scales.json"), ("report", "fit_report.json"), ("inputs", "inputs.json")):
        receipt[key] = {"path": name, "sha256": protocol.file_hash(output / name)}
    protocol.write_json(output / "completed.json", receipt)
    return receipt


def _inspect_fit(cfg, reference, directory):
    output = Path(directory) / "geometry"
    descriptor = protocol.read_json(output / "completed.json")
    for key, name in (("model", "frozen_geometry.pth"), ("cache", "cache.pth"),
                      ("scales", "scales.json"), ("report", "fit_report.json"), ("inputs", "inputs.json")):
        if descriptor.get(key, {}).get("path") != name:
            raise ValueError("Unexpected geometry artifact path: " + key)
        protocol.verify_artifact(output, "completed.json", key)
    if (descriptor.get("schema_version") != SCHEMA_VERSION or descriptor.get("config") != cfg
            or protocol.read_json(output / "config.json") != cfg
            or descriptor.get("source_binding") != reference.binding
            or protocol.read_json(output / "source_binding.json") != reference.binding
            or descriptor.get("meta") != reference.meta
            or descriptor.get("audit") != {"train": reference.training["audit"]["train"]}
            or descriptor.get("fit_splits") != ["train"] or descriptor.get("gradient_splits") != []
            or descriptor.get("unknown_images_used_for_fitting") is not False
            or descriptor.get("test_used_for_fitting") is not False
            or descriptor.get("optimizer_steps") != 0 or descriptor.get("frozen_base_trainable") != 0):
        raise ValueError("Geometry configuration, source binding or TRAIN-only audit changed")
    protocol.require_signature(descriptor["signature"], protocol.signature(cfg, reference.binding))
    expected = {"signature": descriptor["signature"], "audit": descriptor["audit"], "meta": reference.meta}
    if protocol.read_json(output / "inputs.json") != expected:
        raise ValueError("Geometry input audit changed")
    return descriptor, output / "frozen_geometry.pth"


def _load_fit(cfg, reference, directory, device=None):
    receipt, path = _inspect_fit(cfg, reference, directory)
    payload = support_pipeline._load_torch(path)
    if (payload.get("schema_version") != SCHEMA_VERSION or payload.get("config") != cfg
            or payload.get("meta") != reference.meta or payload.get("source_binding") != reference.binding
            or payload.get("cache_sha256") != receipt["cache"]["sha256"]):
        raise ValueError("Geometry checkpoint configuration/source mismatch")
    protocol.require_signature(receipt["signature"], payload.get("signature"))
    if set(payload.get("scales", {})) != set(SCORE_NAMES):
        raise ValueError("Geometry requires four fitted score scales")
    saved_scales = protocol.read_json(Path(directory) / "geometry/scales.json")
    if payload["scales"] != saved_scales.get("states"):
        raise ValueError("Geometry checkpoint scale state differs from its audit")
    geometry = HierarchicalGeometry.from_state_dict(payload["geometry"])
    state = geometry.state_dict()
    if (state["leaf_to_parent"].tolist() != reference.meta["leaf_to_parent"]
            or geometry.fine_dimension != reference.encoder.dimension
            or geometry.parent_dimension != reference.encoder.dimension
            or any(state[key] != value for key, value in cfg["geometry"].items())
            or int(state["leaf_counts"].sum()) != reference.training["audit"]["train"]["unique_image_count"]):
        raise ValueError("Geometry statistics disagree with source taxonomy, TRAIN counts or configuration")
    scales = {key: RobustScoreStandardizer.from_state_dict(value) for key, value in payload["scales"].items()}
    train_hashes = set(reference.training["audit"]["train"]["image_hashes"])
    if set(saved_scales.get("audit", {})) != set(SCORE_NAMES):
        raise ValueError("Geometry requires four score-scale TRAIN audits")
    for key in SCORE_NAMES:
        audit = saved_scales["audit"][key]
        hashes = audit.get("image_sha256", [])
        selection = "correct_parent_candidate" if key.endswith("parent") else "correct_leaf_candidate"
        if (audit.get("source_split") != "train" or audit.get("selection") != selection
                or type(audit.get("count")) is not int or audit["count"] < 1
                or audit["count"] != len(hashes) or len(set(hashes)) != len(hashes)
                or not set(hashes) <= train_hashes
                or payload["scales"][key]["fit_rows"] != audit["count"]):
            raise ValueError("Geometry scale normalization is not bound to known TRAIN")
    return geometry, scales, receipt


def _baseline_reproduction(reference, records):
    # The existing checker validates old raw evidence and exact old decisions.
    # Its reconstruction deduplicator needs a score field; a constant on audit
    # copies has no effect on baseline comparisons or on geometry inference.
    copies = {split: [dict(row, reconstruction_score=0.) for row in rows] for split, rows in records.items()}
    return _check_baseline_development(reference, copies)


def _preservation(baseline, routed, router):
    before = {row["image_sha256"]: row for rows in baseline.values() for row in rows}
    after = {row["image_sha256"]: row for rows in routed.values() for row in rows}
    if set(before) != set(after):
        raise ValueError("Baseline/geometry evaluation identities differ")
    fallback = not router["geometry_enabled"]
    changed_parent, changed_leaf, changed_fallback_parent = 0, 0, 0
    for digest, new in after.items():
        old = before[digest]
        if any(old.get(key) != new.get(key) for key in ("candidate_parent", "candidate_leaf")):
            raise ValueError("Geometry must preserve the original candidate ranking")
        changed_parent += int((old["prediction_type"] == "global_unknown") != (new["prediction_type"] == "global_unknown"))
        changed_leaf += int((old["prediction_type"] == "known") != (new["prediction_type"] == "known"))
        changed_fallback_parent += int(new["prediction_type"] == "intra_unknown" and
                                       new["parent"] != old["candidate_parent"])
        if router.get("decoder") == "local_guarded" and new["prediction_type"] == "known":
            if old["prediction_type"] != "known" or new["leaf"] != old["leaf"]:
                raise ValueError("Local corrections cannot promote or replace a known leaf")
        if fallback and any(old.get(key) != new.get(key) for key in ("prediction_type", "output_node", "parent", "leaf")):
            raise ValueError("Geometry fallback must reproduce every baseline decision")
    return {"unique_images": len(after), "candidate_preserved": True, "candidate_preserved_count": len(after),
            "parent_gate_changed_count": changed_parent, "leaf_gate_changed_count": changed_leaf,
            "fallback_parent_changed_count": changed_fallback_parent,
            "fallback": fallback, "fallback_decisions_exact": fallback}


def calibrate_run(cfg, reference_directory, directory, device):
    directory = _destination(reference_directory, directory)
    reference = load_reference(reference_directory, device)
    geometry, scales, fitted = _load_fit(cfg, reference, directory, device)
    groups, audit = _stage_rows(reference, "calibrate")
    output = protocol.claim_stage(directory, "calibration")
    records, _, timings = collect_features(groups, reference, device, geometry, scales, keep_features=False)
    reproduction = _baseline_reproduction(reference, records)
    protocol.write_json(output / "baseline_reproduction.json", reproduction)
    settings = dict(cfg["calibration"], baseline_calibration=copy.deepcopy(reference.config["calibration"]))
    decoder = _decoder(cfg)
    router = decoder.calibrate(records["val_known"], records["val_intra"], records["val_extra"],
                                   reference.router, reference.meta, settings)
    router.update(signature=fitted["signature"], model_sha256=fitted["model"]["sha256"],
                  source_binding_sha256=protocol.object_hash(reference.binding))
    routed = {key: decoder.apply_router(rows, router, reference.meta) for key, rows in records.items()}
    baseline = {key: support_pipeline.apply_router(rows, reference.router, reference.meta) for key, rows in records.items()}
    preservation = _preservation(baseline, routed, router)
    report = router["validation_report"]
    checked = calibration.evaluate_records([row for rows in routed.values() for row in rows], reference.meta)
    if any(report[key] != checked[key] for key in ("counts", "checks", "metrics", "targets_passed")):
        raise ValueError("Geometry decoder disagrees with its development report")
    for name, value in {"router": router, "validation_report": report,
                        "baseline_validation_report": calibration.evaluate_records(
                            [r for rows in baseline.values() for r in rows], reference.meta),
                        "development_metrics": _routed_metrics(cfg, routed), "baseline_metrics": _metrics(baseline),
                        "preservation": preservation, "inference_timing": timings}.items():
        protocol.write_json(output / (name + ".json"), value)
    if decoder is not calibration:
        protocol.write_json(output / "component_scores.json", decoder.score_diagnostics(
            [r for rows in records.values() for r in rows]))
    protocol.write_records(output / "development_scores.jsonl", [r for rows in records.values() for r in rows])
    protocol.write_records(output / "development_predictions.jsonl", [r for rows in routed.values() for r in rows])
    _assert_source(reference)
    protocol.require_signature(fitted["signature"], protocol.signature(cfg, reference.binding))
    receipt = {"schema_version": SCHEMA_VERSION, "signature": fitted["signature"], "source_binding": reference.binding,
               "audit": audit, "fit_completed": True, "test_used_for_fitting": False,
               "fit_splits": list(groups), "targets_passed": report["targets_passed"],
               "model_sha256": fitted["model"]["sha256"], "geometry_enabled": router["geometry_enabled"],
               "router": {"path": "router.json", "sha256": protocol.file_hash(output / "router.json")}}
    protocol.write_json(output / "completed.json", receipt)
    return receipt


def test_run(cfg, reference_directory, directory, device):
    directory = _destination(reference_directory, directory)
    reference = load_reference(reference_directory, device)
    geometry, scales, fitted = _load_fit(cfg, reference, directory, device)
    caldir = Path(directory) / "calibration"
    if protocol.read_json(caldir / "completed.json").get("router", {}).get("path") != "router.json":
        raise ValueError("Unexpected geometry router artifact path")
    calibrated, router_path = protocol.verify_artifact(caldir, "completed.json", "router")
    router = protocol.read_json(router_path)
    if (calibrated.get("schema_version") != SCHEMA_VERSION or calibrated.get("source_binding") != reference.binding
            or calibrated.get("test_used_for_fitting") is not False
            or calibrated.get("fit_splits") != list(support_protocol.STAGE_SPLITS["calibrate"])
            or calibrated.get("audit") != reference.calibration["audit"]
            or calibrated.get("model_sha256") != fitted["model"]["sha256"]
            or router.get("model_sha256") != fitted["model"]["sha256"]
            or router.get("source_binding_sha256") != protocol.object_hash(reference.binding)
            or router.get("baseline_router") != reference.router):
        raise ValueError("Geometry calibration is not bound to the frozen source/model")
    protocol.require_signature(fitted["signature"], calibrated.get("signature"))
    protocol.require_signature(fitted["signature"], router.get("signature"))
    groups, audit = _stage_rows(reference, "test")
    output = protocol.claim_stage(directory, "test")
    records, _, timings = collect_features(groups, reference, device, geometry, scales, keep_features=False)
    routed = {key: _decoder(cfg).apply_router(rows, router, reference.meta) for key, rows in records.items()}
    baseline = {key: support_pipeline.apply_router(rows, reference.router, reference.meta) for key, rows in records.items()}
    preservation = _preservation(baseline, routed, router)
    unique = {key: calibration.unique_records(rows) for key, rows in routed.items()}
    baseline_unique = {key: baseline_unique_records(rows) for key, rows in baseline.items()}
    metrics = _routed_metrics(cfg, unique)
    summary = calibration.evaluate_records([r for rows in routed.values() for r in rows], reference.meta)
    baseline_summary = calibration.evaluate_records([r for rows in baseline.values() for r in rows], reference.meta)
    gates = calibration.evaluate_gates(metrics)
    for group in (routed, baseline):
        for rows in group.values():
            seen = set()
            for row in rows:
                row["evaluation_weight"] = int(row["image_sha256"] not in seen)
                seen.add(row["image_sha256"])
    for name, value in {"metrics": metrics, "metrics_all_rows": _routed_metrics(cfg, routed), "summary": summary,
                        "baseline_metrics": _metrics(baseline_unique), "baseline_summary": baseline_summary,
                        "preservation": preservation, "gates": gates, "inference_timing": timings}.items():
        protocol.write_json(output / (name + ".json"), value)
    if _decoder(cfg) is not calibration:
        protocol.write_json(output / "component_scores.json", _decoder(cfg).score_diagnostics(
            [r for rows in records.values() for r in rows]))
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
    parser.add_argument("--preflight", action="store_true", help="Validate source receipts without images or CLIP")
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
