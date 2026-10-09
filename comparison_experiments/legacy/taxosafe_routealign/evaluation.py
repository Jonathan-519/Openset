"""Audited evaluation of actual neural fine-tuning arms, including failed gates.

Every trained arm keeps its own best-effort membership router. A scientific
gate failure is reported and never replaces that arm with the reference or
prevents TEST. Invalid artifacts, changed data, and failed execution do stop
that stage, because no valid frozen model/router exists to evaluate.
"""
import copy
import csv
from collections import defaultdict
from pathlib import Path

import torch

from taxosafe_support import pipeline as support_pipeline
from taxosafe_support import protocol as support_protocol
from taxosafe_support import calibration as base
from taxosafe_support import membership_calibration as membership
from taxosafe_refine.pipeline import _stage_rows, _assert_source, _check_baseline_development
from taxosafe_dcbs.protocol import normalized_name
from taxosafe_refine.pipeline import _metrics, load_training_rows
from taxosafe_geometry.pipeline import collect_features
from types import SimpleNamespace
from . import calibration
from .proximity import ProximityBank

SCHEMA_VERSION = "taxosafe_routealign_evaluation_v1"
VALIDATION_SCOPE = "exploratory_route_alignment_on_reused_development"


def _hash(value):
    return support_protocol.object_hash(value)


def _read(path):
    return support_protocol.read_json(Path(path))


def _write(path, value):
    support_protocol.write_json(Path(path), value)


def _records(path):
    import json
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _flat(groups):
    return [row for rows in groups.values() for row in rows]


def _correct(row):
    if row["status"] == "known":
        return row["prediction_type"] == "known" and row["leaf"] == row["true_leaf"] and row["parent"] == row["true_parent"]
    if row["status"] == "intra":
        return row["prediction_type"] == "intra_unknown" and row["parent"] == row["true_parent"]
    return row["prediction_type"] == "global_unknown"


def _regular(path):
    path = Path(path)
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
        raise ValueError("Evaluation artifact must be a regular file without symlink ancestors: " + str(path))
    return path


def _verify_binding(source, binding, frozen_baseline=False):
    if (not isinstance(binding, dict) or not isinstance(binding.get("arm_id"), str)
            or not binding["arm_id"] or binding.get("source_binding") != source.binding):
        raise ValueError("Sweep arm/source binding is missing or inconsistent")
    for key in ("checkpoint", "support"):
        descriptor = binding.get(key)
        if not isinstance(descriptor, dict) or set(descriptor) != {"path", "sha256"}:
            raise ValueError("Sweep binding needs an absolute artifact path and digest: " + key)
        path = Path(descriptor["path"])
        if not path.is_absolute() or support_protocol.file_hash(_regular(path)) != descriptor["sha256"]:
            raise ValueError("Sweep artifact binding failed: " + key)
        if frozen_baseline and descriptor["sha256"] != source.training[key]["sha256"]:
            raise ValueError("Frozen baseline must use the original reference " + key)
    _assert_source(source)


def _claim(output_dir, source):
    requested = Path(output_dir)
    if any(p.is_symlink() for p in (requested, *requested.parents)):
        raise ValueError("Evaluation output must not traverse symlinks")
    output = Path(output_dir).resolve()
    source_dir = Path(source.directory).resolve()
    if output == source_dir or source_dir in output.parents or output in source_dir.parents:
        raise ValueError("Evaluation output must be separate from the archived reference")
    if output.exists():
        raise ValueError("Evaluation stage already exists; use a fresh arm/run directory: " + str(output))
    output.mkdir(parents=True)
    return output


def _device(encoder):
    try:
        return next(encoder.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _check_cfg(source, cfg):
    if cfg != source.config or cfg.get("calibration", {}).get("decoder") != "membership":
        raise ValueError("All sweep arms must evaluate the locked reference data/configuration with membership decoding")


def _species(records, meta):
    rows = base.unique_records(records)
    groups = defaultdict(list)
    for row in rows:
        if row["status"] == "known":
            name = meta["leaf_names"][int(row["true_leaf"])]
            key = row["status"], name
        else:
            name = str(row.get("species") or row.get("source") or "unspecified")
            key = row["status"], normalized_name(name)
        groups[key].append(row)
    for name in meta["leaf_names"]:
        groups.setdefault(("known", name), [])
    result = []
    for (status, key), selected in sorted(groups.items()):
        name = key if status == "known" else str(selected[0].get("species") or selected[0].get("source") or key)
        n = len(selected)
        correct = sum(_correct(row) for row in selected)
        leaves = sum(row["prediction_type"] == "known" for row in selected)
        roots = sum(row["prediction_type"] == "global_unknown" for row in selected)
        parents = sum(row["prediction_type"] == "intra_unknown" for row in selected)
        wrong_parent = sum(row["prediction_type"] == "intra_unknown" and row.get("parent") != row.get("true_parent")
                           for row in selected if status != "extra")
        result.append(dict(status=status, species=name, sample_count=n, correct_count=correct,
                           correct_terminal_rate=None if not n else correct / n,
                           accepted_leaf_count=leaves, root_count=roots, parent_count=parents,
                           wrong_parent_count=wrong_parent,
                           false_leaf_count=leaves - correct if status == "known" else leaves,
                           leaf_rejection_rate=None if not n or status == "known" else (n-leaves)/n,
                           evidence_status="not_evaluable" if not n else "insufficient_evidence" if n < 5 else "observed"))
    return result


def _csv(path, records):
    if not records:
        raise ValueError("CSV export requires a declared table schema")
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _write_outputs(output, groups, router, meta, timings, stage, status, binding, crossfit=None):
    raw = _flat(groups)
    decode = membership.decode_records if router.get("decoder") == "membership" else calibration.decode_records
    routed = {split: decode(rows, router, meta) for split, rows in groups.items()}
    all_predictions = _flat(routed)
    base.unique_records(all_predictions)  # Validate aliases before assigning metric weights.
    seen = set()
    for row in all_predictions:
        identity = base._digest(row)
        row["evaluation_weight"] = int(identity not in seen)
        seen.add(identity)
    unique = {split: base.unique_records(rows) for split, rows in routed.items()}
    metrics = _metrics(unique)
    summary = base.evaluate_records(all_predictions, meta)
    report = dict(summary, schema_version=SCHEMA_VERSION, arm_id=binding["arm_id"], stage=stage,
                  calibration_status=status, calibration_gate_is_execution_gate=False,
                  arm_predictions_replaced_by_reference=False, test_used_for_fitting=False,
                  validation_scope=VALIDATION_SCOPE, independent_model_level_validation=False,
                  development_reused_for_method_design=True, confirmatory_validation=False,
                  crossfit_audit=crossfit,
                  score_semantics="route-specific continuous scores; no calibrated probability guarantee")
    names = {"scores": "scores.jsonl", "predictions": "predictions.jsonl", "metrics": "metrics.json",
             "summary": "summary.json", "report": "report.json", "species": "per_species.csv",
             "router": "router.json", "timing": "inference_timing.json", "binding": "binding.json"}
    support_protocol.write_records(output / names["scores"], raw)
    support_protocol.write_records(output / names["predictions"], all_predictions)
    for name, value in (("metrics", metrics), ("summary", summary), ("report", report),
                        ("router", router), ("timing", timings), ("binding", binding)):
        _write(output / names[name], value)
    _csv(output / names["species"], _species(all_predictions, meta))
    # Familiar aliases ease direct review without changing the authoritative files.
    if stage == "calibration":
        for alias, value in (("validation_report.json", report), ("development_metrics.json", metrics)):
            _write(output / alias, value)
            names[alias] = alias
        for alias, value in (("development_scores.jsonl", raw), ("development_predictions.jsonl", all_predictions)):
            support_protocol.write_records(output / alias, value)
            names[alias] = alias
    artifacts = {key: {"path": name, "sha256": support_protocol.file_hash(output / name)} for key, name in names.items()}
    return report, artifacts


def _verify_completed(directory, binding, stage):
    directory = Path(directory)
    receipt = _read(_regular(directory / "completed.json"))
    if (receipt.get("schema_version") != SCHEMA_VERSION or receipt.get("stage") != stage
            or receipt.get("binding") != binding or receipt.get("binding_sha256") != _hash(binding)
            or receipt.get("test_used_for_fitting") is not False):
        raise ValueError("Sweep stage receipt or arm binding changed")
    artifacts = receipt.get("artifacts", {})
    if not isinstance(artifacts, dict) or not {"router", "report", "predictions", "scores", "binding"} <= set(artifacts):
        raise ValueError("Sweep stage artifact inventory is incomplete")
    for item in artifacts.values():
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            raise ValueError("Invalid sweep artifact descriptor")
        path = directory / item["path"]
        if Path(item["path"]).name != item["path"] or support_protocol.file_hash(_regular(path)) != item["sha256"]:
            raise ValueError("Sweep stage artifact hash mismatch")
    if _read(directory / "binding.json") != binding:
        raise ValueError("Sweep binding artifact changed")
    return receipt


def _model_view(source, encoder, evidence, bank):
    return SimpleNamespace(encoder=encoder, evidence=evidence, bank=bank,
                           config=source.config, meta=source.meta)


def _proximity_payload(source, binding, bank, settings):
    return {"schema_version": "routealign_proximity_artifact_v1", "meta": source.meta,
            "source_binding": source.binding, "checkpoint_sha256": binding["checkpoint"]["sha256"],
            "support_sha256": binding["support"]["sha256"], "fit_splits": ["train"],
            "gradient_splits": [], "optimizer_steps": 0, "settings": settings,
            "train_audit": source.training["audit"]["train"], "bank": bank.state_dict()}


def _load_proximity(path, source, binding, settings):
    payload = support_pipeline._load_torch(_regular(path))
    expected = {"schema_version": "routealign_proximity_artifact_v1", "meta": source.meta,
                "source_binding": source.binding, "checkpoint_sha256": binding["checkpoint"]["sha256"],
                "support_sha256": binding["support"]["sha256"], "fit_splits": ["train"],
                "gradient_splits": [], "optimizer_steps": 0, "settings": settings,
                "train_audit": source.training["audit"]["train"]}
    if any(payload.get(k) != v for k, v in expected.items()):
        raise ValueError("Proximity TRAIN/model/source binding changed")
    model = ProximityBank.from_state_dict(payload["bank"])
    expected_hashes = source.training["audit"]["train"]["image_hashes"]
    if (model.meta != source.meta or model.k != settings["k"]
            or model.shrinkage != settings["shrinkage"]
            or len(model.image_hashes) != len(expected_hashes)
            or set(model.image_hashes) != set(expected_hashes)
            or model.fine.shape[1] != source.encoder.dimension
            or model.parent.shape[1] != source.encoder.dimension):
        raise ValueError("Proximity internal TRAIN identities, taxonomy, dimensions or settings changed")
    return model


def _fit_proximity(source, encoder, evidence, bank, binding, output, settings, reuse=None):
    path = output / "proximity.pth"
    if reuse is not None:
        reuse = Path(reuse)
        prior = _read(reuse / "completed.json")
        _verify_completed(reuse, prior["binding"], "calibration")
        fitted = _load_proximity(reuse / "proximity.pth", source, binding, settings)
        origin = {"operation": "reuse_same_checkpoint_train_bank", "source_receipt_sha256": support_protocol.file_hash(reuse / "completed.json")}
    else:
        rows, audit = load_training_rows(source)
        records, features, timing = collect_features({"train": rows}, _model_view(source, encoder, evidence, bank), _device(encoder))
        selected = base.unique_records(records["train"])
        data = features["train"]
        if [r["image_sha256"] for r in selected] != data["image_sha256"]:
            raise ValueError("TRAIN proximity feature/label order changed")
        fitted = ProximityBank.fit(data["fine"], data["parent"],
            torch.tensor([r["true_leaf"] for r in selected], dtype=torch.long), data["image_sha256"],
            source.meta, **settings)
        origin = {"operation": "fit_train_only_leave_self_out_scales", "timing": timing}
    support_pipeline._save_torch(path, _proximity_payload(source, binding, fitted, settings))
    report = {"fit_splits": ["train"], "gradient_splits": [], "optimizer_steps": 0,
              "test_used_for_fitting": False, "unknown_used_for_fitting": False,
              "source_binding": source.binding, "checkpoint_sha256": binding["checkpoint"]["sha256"],
              "settings": settings, "origin": origin,
              "note": "TRAIN local-support statistics are not neural training or strict unseen-species representation fitting"}
    details = getattr(fitted, "fit_report", None)
    if details is not None:
        report["diagnostics"] = details() if callable(details) else details
    _write(output / "proximity_fit.json", report)
    return fitted


def _collect_with_proximity(groups, source, encoder, evidence, bank, fitted):
    records, features, timings = collect_features(groups, _model_view(source, encoder, evidence, bank), _device(encoder))
    for split, rows in records.items():
        f = features[split]
        scores = fitted.score(f["fine"], f["parent"], f["image_sha256"])
        by_hash = {h: {key: values[i].tolist() for key, values in scores.items()} for i, h in enumerate(f["image_sha256"])}
        for row in rows:
            row["proximity"] = by_hash[row["image_sha256"]]
    return records, timings


def _reference_development(source):
    rows = _records(source.directory / "calibration/development_scores.jsonl")
    for row in rows:
        if row.get("split") not in ("val_known", "val_intra", "val_extra"):
            raise ValueError("Reference calibration contains a non-DEV row")
    base.unique_records(rows)
    return rows


def evaluate_development(source, encoder, evidence, bank, cfg, output_dir, binding,
                         frozen_baseline=False, route_variant=None, settings=None,
                         proximity_settings=None, reuse_proximity_dir=None):
    _check_cfg(source, cfg)
    _verify_binding(source, binding, frozen_baseline)
    groups, audit = _stage_rows(source, "calibrate")
    output = _claim(output_dir, source)
    reference_records = _reference_development(source)
    if route_variant is None:
        records, timings = support_pipeline.collect(groups, cfg, source.meta, encoder, evidence, bank, _device(encoder))
        if frozen_baseline:
            reproduction = _check_baseline_development(source, {key: [dict(r, reconstruction_score=0.) for r in rows] for key, rows in records.items()})
            _write(output / "baseline_reproduction.json", reproduction)
            router = copy.deepcopy(source.router)
            status, origin, crossfit = "frozen_reference_reproduced", "frozen_reference_router", None
        else:
            router = membership.calibrate(records["val_known"], records["val_intra"], records["val_extra"], source.meta, copy.deepcopy(cfg["calibration"]))
            status, origin = router["status"], "arm_development_membership_fit"
            crossfit = calibration.crossfit_audit(records["val_known"], records["val_intra"], records["val_extra"], source.meta,
                source.router, settings=dict(settings, baseline_calibration=cfg["calibration"]),
                variant="membership", reference_records=reference_records)
    else:
        fitted = _fit_proximity(source, encoder, evidence, bank, binding, output, proximity_settings, reuse_proximity_dir)
        records, timings = _collect_with_proximity(groups, source, encoder, evidence, bank, fitted)
        cal_settings = dict(settings, baseline_calibration=cfg["calibration"])
        router = calibration.fit_router(records["val_known"], records["val_intra"], records["val_extra"], source.meta,
            source.router, cal_settings, route_variant, reference_records=reference_records)
        crossfit = calibration.crossfit_audit(records["val_known"], records["val_intra"], records["val_extra"], source.meta,
            source.router, cal_settings, route_variant, reference_records=reference_records)
        status, origin = router["status"], "arm_development_proximity_fit"
    router.update(artifact_binding=copy.deepcopy(binding), artifact_binding_sha256=_hash(binding),
                  checkpoint_sha256=binding["checkpoint"]["sha256"], support_sha256=binding["support"]["sha256"], router_origin=origin)
    report, artifacts = _write_outputs(output, records, router, source.meta, timings, "calibration", status, binding, crossfit)
    for key, name in (("proximity", "proximity.pth"), ("proximity_fit", "proximity_fit.json"), ("reproduction", "baseline_reproduction.json")):
        if (output / name).is_file():
            artifacts[key] = {"path": name, "sha256": support_protocol.file_hash(output / name)}
    _verify_binding(source, binding, frozen_baseline)
    receipt = dict(schema_version=SCHEMA_VERSION, stage="calibration", arm_id=binding["arm_id"], binding=copy.deepcopy(binding),
        binding_sha256=_hash(binding), artifacts=artifacts, audit=audit, config_sha256=_hash(cfg), meta=source.meta,
        frozen_baseline=bool(frozen_baseline), fit_completed=True, fit_splits=list(support_protocol.STAGE_SPLITS["calibrate"]),
        route_variant=route_variant, settings=settings, proximity_settings=proximity_settings,
        test_used_for_fitting=False, targets_passed=report["targets_passed"], calibration_status=status,
        status="completed", summary=report, test_allowed_after_failed_gates=True)
    _write(output / "completed.json", receipt)
    return receipt


def evaluate_test(source, encoder, evidence, bank, cfg, output_dir, binding, calibration_dir,
                  frozen_baseline=False, route_variant=None, settings=None, proximity_settings=None):
    _check_cfg(source, cfg)
    _verify_binding(source, binding, frozen_baseline)
    calibrated = _verify_completed(calibration_dir, binding, "calibration")
    if any(calibrated.get(k) != v for k, v in {"config_sha256": _hash(cfg), "meta": source.meta,
        "frozen_baseline": bool(frozen_baseline), "route_variant": route_variant,
        "settings": settings, "proximity_settings": proximity_settings}.items()):
        raise ValueError("TEST configuration differs from calibration")
    router = _read(Path(calibration_dir) / "router.json")
    if (router.get("artifact_binding") != binding or router.get("artifact_binding_sha256") != _hash(binding)
            or router.get("checkpoint_sha256") != binding["checkpoint"]["sha256"]
            or router.get("support_sha256") != binding["support"]["sha256"]):
        raise ValueError("TEST model/support/router chain differs from calibration")
    groups, audit = _stage_rows(source, "test")
    output = _claim(output_dir, source)
    # A failed scientific gate is never an execution gate.
    if route_variant is None:
        records, timings = support_pipeline.collect(groups, cfg, source.meta, encoder, evidence, bank, _device(encoder))
    else:
        fitted = _load_proximity(Path(calibration_dir) / "proximity.pth", source, binding, proximity_settings)
        records, timings = _collect_with_proximity(groups, source, encoder, evidence, bank, fitted)
    report, artifacts = _write_outputs(output, records, router, source.meta, timings, "test", calibrated["calibration_status"], binding)
    _verify_binding(source, binding, frozen_baseline)
    receipt = dict(schema_version=SCHEMA_VERSION, stage="test", arm_id=binding["arm_id"], binding=copy.deepcopy(binding),
        binding_sha256=_hash(binding), artifacts=artifacts, audit=audit, config_sha256=_hash(cfg), meta=source.meta,
        frozen_baseline=bool(frozen_baseline), calibration_receipt_sha256=support_protocol.file_hash(Path(calibration_dir) / "completed.json"),
        calibration_router_sha256=support_protocol.file_hash(Path(calibration_dir) / "router.json"),
        test_used_for_fitting=False, metric_unit="unique_image_sha256", status="completed", targets_passed=report["targets_passed"],
        calibration_status=calibrated["calibration_status"], summary=report)
    _write(output / "completed.json", receipt)
    return receipt
