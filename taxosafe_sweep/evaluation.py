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

SCHEMA_VERSION = "taxosafe_sweep_evaluation_v1"
VALIDATION_SCOPE = "exploratory_finetuning_on_previously_used_development"


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


def _write_outputs(output, groups, router, meta, timings, stage, status, binding):
    raw = _flat(groups)
    routed = {split: base.apply_router(rows, router, meta) for split, rows in groups.items()}
    all_predictions = _flat(routed)
    base.unique_records(all_predictions)  # Validate aliases before assigning metric weights.
    seen = set()
    for row in all_predictions:
        identity = base._digest(row)
        row["evaluation_weight"] = int(identity not in seen)
        seen.add(identity)
    unique = {split: base.unique_records(rows) for split, rows in routed.items()}
    _, metrics = support_pipeline.metrics_for(unique, router, meta)
    summary = base.evaluate_records(all_predictions, meta)
    report = dict(summary, schema_version=SCHEMA_VERSION, arm_id=binding["arm_id"], stage=stage,
                  calibration_status=status, calibration_gate_is_execution_gate=False,
                  arm_predictions_replaced_by_reference=False, test_used_for_fitting=False,
                  validation_scope=VALIDATION_SCOPE, independent_model_level_validation=False,
                  development_reused_for_method_design=True, confirmatory_validation=False)
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


def evaluate_development(source, encoder, evidence, bank, cfg, output_dir, binding, frozen_baseline=False):
    _check_cfg(source, cfg)
    _verify_binding(source, binding, frozen_baseline)
    groups, audit = _stage_rows(source, "calibrate")
    output = _claim(output_dir, source)
    records, timings = support_pipeline.collect(groups, cfg, source.meta, encoder, evidence, bank, _device(encoder))
    if frozen_baseline:
        copies = {key: [dict(row, reconstruction_score=0.) for row in rows] for key, rows in records.items()}
        reproduction = _check_baseline_development(source, copies)
        _write(output / "baseline_reproduction.json", reproduction)
        router = copy.deepcopy(source.router)
        origin, status = "frozen_reference_router", "frozen_reference_reproduced"
    else:
        router = membership.calibrate(records["val_known"], records["val_intra"], records["val_extra"],
                                      source.meta, copy.deepcopy(cfg["calibration"]))
        origin, status = "arm_development_fit", router["status"]
    router.update(sweep_schema_version=SCHEMA_VERSION, artifact_binding=copy.deepcopy(binding),
                  artifact_binding_sha256=_hash(binding), checkpoint_sha256=binding["checkpoint"]["sha256"],
                  support_sha256=binding["support"]["sha256"], router_origin=origin)
    report, artifacts = _write_outputs(output, records, router, source.meta, timings, "calibration", status, binding)
    if frozen_baseline:
        artifacts["reproduction"] = {"path": "baseline_reproduction.json", "sha256": support_protocol.file_hash(output / "baseline_reproduction.json")}
    _verify_binding(source, binding, frozen_baseline)
    receipt = dict(schema_version=SCHEMA_VERSION, stage="calibration", arm_id=binding["arm_id"],
                   binding=copy.deepcopy(binding), binding_sha256=_hash(binding), artifacts=artifacts,
                   audit=audit, config_sha256=_hash(cfg), meta=source.meta, frozen_baseline=bool(frozen_baseline),
                   fit_completed=True, fit_splits=list(support_protocol.STAGE_SPLITS["calibrate"]),
                   test_used_for_fitting=False, targets_passed=report["targets_passed"],
                   calibration_status=status, status="completed", summary=report,
                   test_allowed_after_failed_gates=True)
    _write(output / "completed.json", receipt)
    return receipt


def evaluate_test(source, encoder, evidence, bank, cfg, output_dir, binding, calibration_dir, frozen_baseline=False):
    _check_cfg(source, cfg)
    _verify_binding(source, binding, frozen_baseline)
    calibrated = _verify_completed(calibration_dir, binding, "calibration")
    if (calibrated.get("config_sha256") != _hash(cfg) or calibrated.get("meta") != source.meta
            or calibrated.get("frozen_baseline") != bool(frozen_baseline)):
        raise ValueError("TEST configuration or baseline status differs from calibration")
    router = _read(Path(calibration_dir) / "router.json")
    membership._validate_state(router, source.meta)
    if (router.get("artifact_binding") != binding or router.get("artifact_binding_sha256") != _hash(binding)
            or router.get("checkpoint_sha256") != binding["checkpoint"]["sha256"]
            or router.get("support_sha256") != binding["support"]["sha256"]):
        raise ValueError("TEST model/support/router chain differs from calibration")
    # Do not inspect targets_passed here: failed research gates remain testable.
    groups, audit = _stage_rows(source, "test")
    output = _claim(output_dir, source)
    records, timings = support_pipeline.collect(groups, cfg, source.meta, encoder, evidence, bank, _device(encoder))
    report, artifacts = _write_outputs(output, records, router, source.meta, timings, "test",
                                       calibrated["calibration_status"], binding)
    _verify_binding(source, binding, frozen_baseline)
    receipt = dict(schema_version=SCHEMA_VERSION, stage="test", arm_id=binding["arm_id"],
                   binding=copy.deepcopy(binding), binding_sha256=_hash(binding), artifacts=artifacts,
                   audit=audit, config_sha256=_hash(cfg), meta=source.meta, frozen_baseline=bool(frozen_baseline),
                   calibration_receipt_sha256=support_protocol.file_hash(Path(calibration_dir) / "completed.json"),
                   calibration_router_sha256=calibrated["artifacts"]["router"]["sha256"],
                   test_used_for_fitting=False, metric_unit="unique_image_sha256", status="completed",
                   targets_passed=report["targets_passed"], calibration_status=calibrated["calibration_status"],
                   summary=report)
    _write(output / "completed.json", receipt)
    return receipt
