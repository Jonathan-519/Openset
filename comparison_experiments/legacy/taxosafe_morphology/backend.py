"""Independent spatial verification with immutable reference/D05 controls."""
import copy
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch

from taxosafe_support import pipeline as support
from taxosafe_support import calibration as base
from taxosafe_support import membership_calibration as membership
from taxosafe_refine.pipeline import _metrics, _check_baseline_development
from taxosafe_routealign.evaluation import _species, _csv
from taxosafe_discovery import backend as legacy
from taxosafe_discovery import calibration as legacy_calibration
from . import protocol, calibration, importer
from . import features, episodes, training, diagnostics as score_audit
from taxosafe_domain.backend import aligned_group, _source_report


_regular, _claim, _finish = legacy._regular, legacy._claim, legacy._finish


def _semantic(value):
    def canonical(item):
        if torch.is_tensor(item):
            return {"tensor_sha256": legacy.tensor_hash(item)}
        if isinstance(item, dict):
            return {key: canonical(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [canonical(child) for child in item]
        return item
    return protocol.object_hash(canonical(value))


def _checked(suite):
    suite = Path(suite).resolve()
    cfg = protocol.validate_config(protocol.read_json(_regular(suite / "config.json")))
    binding = protocol.read_json(_regular(suite / "source_binding.json"))
    info = importer.inspect_d05(binding["directory"])
    snapshot = protocol.read_json(_regular(suite / "snapshot.json"))
    if (info["binding"] != binding or snapshot.get("signature") != protocol.signature(cfg, binding)
            or snapshot.get("schema_version") != protocol.SCHEMA_VERSION
            or snapshot.get("config_sha256") != protocol.object_hash(cfg)
            or snapshot.get("source_binding") != binding):
        raise ValueError("Morphology parent/code/configuration changed")
    return suite, cfg, info


def _header(cfg, info, stage, arm_id=None):
    result = dict(schema_version=protocol.SCHEMA_VERSION, stage=stage,
                  signature=protocol.signature(cfg, info["binding"]), source_binding=info["binding"],
                  meta=info["meta"], test_used_for_fitting=False,
                  unknown_images_used_for_gradients=False)
    if arm_id is not None:
        result["arm_id"] = arm_id
    return result


def _receipt(directory, cfg, info, stage, arm_id=None):
    directory = Path(directory)
    receipt = protocol.read_json(_regular(directory / "completed.json"))
    if any(receipt.get(key) != value for key, value in _header(cfg, info, stage, arm_id).items()):
        raise ValueError("Morphology stage receipt/source signature mismatch")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("Missing morphology artifacts")
    for descriptor in artifacts.values():
        if (not isinstance(descriptor, dict) or set(descriptor) != {"path", "sha256"}
                or Path(descriptor["path"]).name != descriptor["path"]
                or protocol.file_hash(_regular(directory / descriptor["path"])) != descriptor["sha256"]):
            raise ValueError("Changed or escaping morphology artifact")
    return receipt


def _arm(cfg, arm_id):
    matches = [arm for arm in cfg["arms"] if arm["id"] == arm_id]
    if len(matches) != 1:
        raise ValueError("Unknown morphology arm")
    return matches[0]


def prepare_cache(suite, stage, device="cpu"):
    if stage not in ("train", "development", "test"):
        raise ValueError("Unknown morphology cache stage")
    suite, cfg, info = _checked(suite)
    if stage == "test":
        _regular(suite / "dev_selection.json")
        cache, parent_receipt = importer.load_parent_test_cache(info, suite)
    else:
        cache, parent_receipt = importer.load_parent_cache(info, stage)
    legacy._validate_cache(cache, info["reference"], stage)
    if legacy._text_contract(cache) != info["training"]["inference_spec_sha256"]:
        raise ValueError("Morphology cache text/core differs from source D05")
    output = _claim(suite / "cache" / stage)
    header = _header(cfg, info, stage)
    binding = dict(parent_cache_sha256=parent_receipt["artifacts"]["features"]["sha256"],
                   parent_cache_receipt_sha256=protocol.file_hash(info["directory"] / "cache" / stage / "completed.json"),
                   inference_spec_sha256=legacy._text_contract(cache))
    _, raw_audit = features.audited_rows(cache, info)
    protocol.write_json(output / "raw_image_audit.json", raw_audit)
    if not raw_audit["valid"]:
        raise ValueError("Raw image audit failed; see " + str(output / "raw_image_audit.json"))
    spatial = features.collect_spatial(cache, info, cfg["features"], device)
    features.validate_spatial(spatial, cache, info, cfg["features"])
    payload = dict(header, **binding, cache=cache, spatial=spatial, audit=parent_receipt["audit"])
    support._save_torch(output / "features.pth", payload)
    receipt = dict(header, **binding, audit=parent_receipt["audit"], provenance=cache["provenance"],
                   spatial_summary=features.spatial_summary(spatial),
                   timings={"original_cache_reused": True, "image_forward_count": spatial["image_forward_count"], "seconds": spatial["seconds"]})
    if stage == "test":
        receipt["dev_selection_sha256"] = protocol.file_hash(suite / "dev_selection.json")
    return _finish(output, receipt, {"features": "features.pth", "raw_image_audit": "raw_image_audit.json"})


def _load_cache(suite, stage, cfg, info):
    directory = Path(suite) / "cache" / stage
    receipt = _receipt(directory, cfg, info, stage)
    payload = support._load_torch(directory / "features.pth")
    if any(payload.get(key) != value for key, value in _header(cfg, info, stage).items()):
        raise ValueError("Morphology cache header changed")
    cache = payload["cache"]
    legacy._validate_cache(cache, info["reference"], stage)
    for key in ("parent_cache_sha256", "parent_cache_receipt_sha256", "inference_spec_sha256", "audit"):
        if payload.get(key) != receipt.get(key):
            raise ValueError("Morphology cache receipt binding differs")
    if (legacy._text_contract(cache) != receipt["inference_spec_sha256"]
            or receipt["inference_spec_sha256"] != info["training"]["inference_spec_sha256"]):
        raise ValueError("Morphology cache changed source text/core semantics")
    if stage == "test" and receipt.get("dev_selection_sha256") != protocol.file_hash(_regular(Path(suite) / "dev_selection.json")):
        raise ValueError("TEST cache differs from frozen morphology DEV selection")
    parent = info["directory"] / "cache" / stage
    if (protocol.file_hash(_regular(parent / "features.pth")) != receipt["parent_cache_sha256"]
            or protocol.file_hash(_regular(parent / "completed.json")) != receipt["parent_cache_receipt_sha256"]):
        raise ValueError("Original discovery cache changed")
    # The child receipt can authenticate its own file without proving that its
    # tensors/records are the frozen source values. Compare to the parent whose
    # artifact digest was just checked, including aliases and candidates.
    original = support._load_torch(parent / "features.pth")
    if not isinstance(original, dict) or _semantic(cache) != _semantic(original.get("cache")):
        raise ValueError("Morphology cache differs from frozen D05 cache contents")
    features.validate_spatial(payload["spatial"], cache, info, cfg["features"])
    if features.spatial_summary(payload["spatial"]) != receipt.get("spatial_summary"):
        raise ValueError("Spatial tensor summary differs from its stage receipt")
    return dict(cache, spatial=payload["spatial"]), receipt


def fit_arm(suite, arm_id, device="cpu"):
    suite, cfg, info = _checked(suite)
    arm = _arm(cfg, arm_id)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    cache, cached = _load_cache(suite, "train", cfg, info)
    output = _claim(suite / "arms" / arm_id / "training")
    header = _header(cfg, info, "training", arm_id)
    source = importer.load_d05(info["directory"], expected_binding=info["binding"])
    digest = _semantic(source.payload)
    payload = dict(header, model_spec=arm, parent_payload=copy.deepcopy(source.payload),
        spatial_model=None, frozen_d05_payload_sha256=digest,
        inference_spec_sha256=legacy._text_contract(source.payload),
        train_cache_sha256=cached["artifacts"]["features"]["sha256"])
    report = _source_report(arm, info)
    if arm["kind"] == "fit":
        view = aligned_group(cache["groups"]["train"], info["meta"], training=True)
        episode_data = episodes.build_episodes(view, source.payload, info["meta"], cfg["support"], cfg["seed"])
        state, report = training.fit_spatial(view, cache["spatial"]["groups"]["train"], episode_data,
                                             info["meta"], cfg, arm["level"], device)
        payload["spatial_model"] = state
    if _semantic(source.payload) != digest:
        raise ValueError("Spatial fitting changed the immutable D05 payload")
    payload["fit_report"] = report
    support._save_torch(output / "model.pth", payload)
    protocol.write_json(output / "training_report.json", report)
    receipt = dict(header, optimizer_steps=report["optimizer_steps"],
        weight_source="reference" if arm["kind"] == "reference" else arm_id if arm["kind"] == "fit" else "original_D05",
        fit_report=report, inference_spec_sha256=payload["inference_spec_sha256"],
        train_cache_sha256=payload["train_cache_sha256"], frozen_d05_payload_sha256=digest,
        model=dict(path="model.pth", sha256=protocol.file_hash(output / "model.pth")))
    return _finish(output, receipt, {"model": "model.pth", "training_report": "training_report.json"})


def _load_model(suite, arm, cfg, info):
    directory = Path(suite) / "arms" / arm["id"] / "training"
    receipt = _receipt(directory, cfg, info, "training", arm["id"])
    expected_steps = cfg["training"]["steps"] if arm["kind"] == "fit" else 0
    if (receipt.get("model") != receipt["artifacts"].get("model")
            or receipt.get("optimizer_steps") != expected_steps
            or protocol.read_json(_regular(directory / "training_report.json")) != receipt.get("fit_report")):
        raise ValueError("Spatial model descriptor or training report differs")
    payload = support._load_torch(directory / "model.pth")
    if any(payload.get(k) != v for k,v in _header(cfg, info, "training", arm["id"]).items()):
        raise ValueError("Spatial model source/schema changed")
    source = importer.load_d05(info["directory"], expected_binding=info["binding"])
    digest = _semantic(source.payload)
    train_receipt = _receipt(Path(suite) / "cache/train", cfg, info, "train")
    if (payload.get("model_spec") != arm or payload.get("fit_report") != receipt["fit_report"]
            or payload["fit_report"]["optimizer_steps"] != expected_steps
            or _semantic(payload["parent_payload"]) != digest
            or payload.get("frozen_d05_payload_sha256") != digest or receipt.get("frozen_d05_payload_sha256") != digest
            or legacy._text_contract(payload["parent_payload"]) != receipt["inference_spec_sha256"]
            or payload.get("inference_spec_sha256") != receipt["inference_spec_sha256"]
            or payload.get("train_cache_sha256") != receipt["train_cache_sha256"]
            or receipt["train_cache_sha256"] != train_receipt["artifacts"]["features"]["sha256"]):
        raise ValueError("Spatial checkpoint changed its frozen D05 or TRAIN binding")
    state = payload.get("spatial_model")
    if (state is not None) != (arm["kind"] == "fit"):
        raise ValueError("Spatial model differs from declared arm kind")
    if state is not None:
        from .verifier import SpatialVerifier
        SpatialVerifier.restore(state["verifier"])
        bank = state["support_bank"]
        cache, _ = _load_cache(suite, "train", cfg, info)
        view = aligned_group(cache["groups"]["train"], info["meta"], training=True)
        indices = torch.as_tensor(bank["selected_train_indices"], dtype=torch.long)
        if (bank.get("schema_version") != "morphology_real_support_v1" or bank["fit_split"] != "known_train"
                or bank["meta"] != info["meta"] or state["level"] != arm["level"]
                or state["matching"] != cfg["matching"] or bank["references_per_leaf"] != cfg["support"]["references_per_leaf"]
                or len(indices) != len(set(indices.tolist())) or bool(((indices < 0) | (indices >= len(view["labels"]))).any())
                or bank["image_sha256"] != [view["image_sha256"][i] for i in indices.tolist()]
                or not torch.equal(bank["labels"], view["labels"][indices])
                or not torch.equal(bank["tokens"], cache["spatial"]["groups"]["train"]["tokens"][indices])
                or not torch.equal(bank["positions"], cache["spatial"]["groups"]["train"]["positions"])
                or payload["fit_report"]["fit_image_sha256"] != view["image_sha256"]):
            raise ValueError("Spatial support changed original TRAIN individuals or token provenance")
    elif payload["fit_report"] != _source_report(arm, info):
        raise ValueError("Source control must preserve inherited training identity")
    return payload, receipt


def _d05_groups(cache, payload, meta):
    return legacy.score_groups(cache, payload["model_spec"], payload, meta)


def _reference_path(rows, meta):
    # The scorer and calibration validator must apply exactly the same repair;
    # the two original baseline decoders never pass through this helper.
    return calibration.reference_path(rows, meta)


def score_groups(cache, arm, payload, meta, device="cpu"):
    if arm["kind"] == "reference":
        return {split: copy.deepcopy(group["records"]) for split,group in cache["groups"].items()}
    original = _d05_groups(cache, payload["parent_payload"], meta)
    if arm["kind"] == "source":
        return original
    result = {}
    for split, group in cache["groups"].items():
        view = aligned_group(group, meta)
        data = legacy_calibration._data(original[split], meta)
        anchors_p, anchors_l, original_leaf, _, mapping = _reference_path(view["raw_records"], meta)
        parents, leaves = data["parent_scores"].copy(), data["leaf_scores"].copy()
        details = None
        if arm["kind"] == "fit":
            up, ul, _, _, _ = _reference_path(view["records"], meta)
            corrections, unique_details = training.score_spatial(payload["spatial_model"], cache["spatial"]["groups"][split],
                up if arm["level"] == "parent" else ul, meta, device)
            indices = group["record_feature_indices"]
            delta = corrections.numpy()[indices]
            if arm["level"] == "parent":
                parents[np.arange(len(parents)), anchors_p] += delta
            else:
                leaves[np.arange(len(leaves)), anchors_l] += delta
            details = [unique_details[j] for j in indices]
        root = parents[np.arange(len(parents)), anchors_p]
        rows = copy.deepcopy(original[split])
        for i,row in enumerate(rows):
            row["morphology"] = dict(root_score=float(root[i]), parent_scores=parents[i].tolist(), leaf_scores=leaves[i].tolist(),
                candidate_parent=int(anchors_p[i]), candidate_leaf=int(anchors_l[i]),
                anchor_parent=int(anchors_p[i]), anchor_leaf=int(anchors_l[i]), root_method=arm["root"],
                leaf_method=arm["leaf"], candidate_policy=arm["candidate_policy"])
            row["morphology_anchor_leaf_reselected"] = bool(original_leaf[i] != anchors_l[i])
            row["morphology_original_reference_leaf"] = int(original_leaf[i])
            if details is not None:
                row["morphology_matching"] = copy.deepcopy(details[i])
            row.update(support_evidence_origin="immutable_reference_diagnostic", log_probs_origin="immutable_reference_diagnostic")
        result[split] = rows
    return result


def _read_records(path):
    import json
    return [json.loads(line) for line in _regular(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _d05_reproduction(info, groups):
    directory = info["directory"] / "arms" / importer.ARM_ID / "calibration"
    old = base.unique_records(_read_records(directory / info["calibration"]["artifacts"]["scores"]["path"]))
    new = base.unique_records([row for rows in groups.values() for row in rows])
    before, after = ({row["image_sha256"]: row for row in rows} for rows in (old, new))
    if set(before) != set(after):
        raise ValueError("D05 development reproduction image identities differ")
    for digest in before:
        for key in ("discovery", "global_pred_leaf", "support_evidence", "log_probs"):
            if before[digest].get(key) != after[digest].get(key):
                raise ValueError("D05 development scores/candidates differ: " + key)
    original = base.unique_records(_read_records(directory / info["calibration"]["artifacts"]["predictions"]["path"]))
    routed = legacy_calibration.decode_records(new, info["router"], info["meta"])
    original = {row["image_sha256"]: row for row in original}
    for row in routed:
        if any(row.get(key) != original[row["image_sha256"]].get(key) for key in
               ("candidate_leaf", "candidate_parent", "prediction_type", "leaf", "parent", "output_node")):
            raise ValueError("D05 development terminal reproduction failed")
    return dict(matched_unique_images=len(new), exact_scores=True, exact_candidates=True,
                exact_terminal_outputs=True, original_router_reused=True)


def _candidate_audit(records, meta):
    unique = base.unique_records(records)
    anchor_changes = parent_changes = leaf_changes = 0
    for row in unique:
        value = row.get("morphology")
        if value is None:
            continue
        anchor_changes += int(row.get("morphology_anchor_leaf_reselected", False))
        parent_changes += int(value["candidate_parent"] != value["anchor_parent"])
        leaf_changes += int(value["candidate_leaf"] != value["anchor_leaf"])
        if (meta["leaf_to_parent"][value["candidate_leaf"]] != value["candidate_parent"]
                or meta["leaf_to_parent"][value["anchor_leaf"]] != value["anchor_parent"]):
            raise ValueError("Exported Morphology candidate/anchor path is inconsistent")
    return dict(unit="unique_image_content_sha256", unique_images=len(unique),
        anchor_leaf_reselected_count=anchor_changes, candidate_parent_changed_count=parent_changes,
        candidate_leaf_changed_count=leaf_changes, selection_uses_query_truth=False,
        reference_leaf_node_offset=1 + len(meta["parent_names"]), reference_log_probs_node_order="root,parent,leaf")


def _export(output, groups, router, info, arm, crossfit, diagnostics, receipt, timings, crossfit_sha256=None):
    from . import reporting
    decode = (membership.decode_records if arm["kind"] == "reference" else
              legacy_calibration.decode_records if arm["kind"] == "source" else calibration.decode_records)
    routed = {split: decode(rows, router, info["meta"]) for split, rows in groups.items()}
    predictions = [row for rows in routed.values() for row in rows]
    base.unique_records(predictions)
    seen = set()
    for row in predictions:
        row["evaluation_weight"] = int(row["image_sha256"] not in seen)
        seen.add(row["image_sha256"])
    names = dict(scores="scores.jsonl", predictions="predictions.jsonl", report="report.json", summary="summary.json",
                 metrics="metrics.json", router="router.json", species="per_species.csv", timing="inference_timing.json")
    if receipt["stage"] == "calibration":
        protocol.write_json(output / "crossfit_audit.json", crossfit)
        crossfit_sha256 = protocol.file_hash(output / "crossfit_audit.json")
        names["crossfit"] = "crossfit_audit.json"
        compact = reporting.compact_crossfit(crossfit)
    else:
        if not isinstance(crossfit_sha256, str) or len(crossfit_sha256) != 64:
            raise ValueError("TEST requires its frozen calibration crossfit artifact digest")
        compact = copy.deepcopy(crossfit)
    summary = base.evaluate_records(predictions, info["meta"])
    summary.update(crossfit_audit=compact, crossfit_audit_sha256=crossfit_sha256,
                   root_stage_outcomes=reporting.root_stage_summary(predictions, info["meta"]))
    dev_targets_passed = (summary["targets_passed"] if receipt["stage"] == "calibration"
                          else receipt.get("inherited_dev_targets_passed"))
    if type(dev_targets_passed) is not bool:
        raise ValueError("TEST report requires its frozen DEV target status")
    details = diagnostics or {}
    report = dict(summary, schema_version="morphology_evaluation_v1", arm_id=arm["id"], stage=receipt["stage"],
        calibration_status="passed" if dev_targets_passed else "best_effort",
        calibration_status_origin="current_development" if receipt["stage"] == "calibration" else "frozen_development",
        evaluation_status="passed" if summary["targets_passed"] else "best_effort",
        calibration_gate_is_execution_gate=False, test_allowed_after_failed_gates=True,
        arm_predictions_replaced_by_reference=False, test_used_for_fitting=False,
        confirmatory_validation=False, independent_model_level_validation=False,
        validation_scope="exploratory_parent_morphology_on_reused_benchmark",
        score_semantics="signed spatial correction of D05 logits; not calibrated posterior probabilities",
        candidate_policy=arm["candidate_policy"], candidate_audit=_candidate_audit(predictions, info["meta"]),
        morphology_policy=details.get("morphology_policy"), root_stage=details.get("root_stage"),
        leaf_stage=details.get("leaf_stage"), calibration_diagnostics=diagnostics,
        inference_constants=dict(image_forward_count=0, optimizer_steps=0, statistics_refitted=False,
                                 cached_unit_clip_preserved=True, spatial_verifier_forward=arm["kind"] == "fit", root_method=arm["root"], leaf_method=arm["leaf"]),
        crossfit_detail_storage="calibration/crossfit_audit.json only",
        crossfit_detail_reference=dict(stage="calibration", artifact="crossfit_audit.json", sha256=crossfit_sha256))
    report["score_diagnostics"] = score_audit.score_diagnostics(predictions)
    metrics = _metrics({split: base.unique_records(rows) for split, rows in routed.items()})
    protocol.write_records(output / "scores.jsonl", [row for rows in groups.values() for row in rows])
    protocol.write_records(output / "predictions.jsonl", predictions)
    for key, value in (("report", report), ("summary", summary), ("metrics", metrics), ("router", router), ("timing", timings)):
        protocol.write_json(output / names[key], value)
    _csv(output / names["species"], _species(predictions, info["meta"]))
    receipt.update(summary=summary, targets_passed=summary["targets_passed"], crossfit_audit_sha256=crossfit_sha256,
                   test_allowed_after_failed_gates=True, calibration_gate_is_execution_gate=False)
    return _finish(output, receipt, names)


def calibrate_arm(suite, arm_id, device="cpu"):
    suite, cfg, info = _checked(suite)
    arm = _arm(cfg, arm_id)
    cache, cached = _load_cache(suite, "development", cfg, info)
    payload, trained = _load_model(suite, arm, cfg, info)
    if legacy._text_contract(cache) != trained["inference_spec_sha256"]:
        raise ValueError("Morphology TRAIN/DEV core or text differs")
    output = _claim(suite / "arms" / arm_id / "calibration")
    started = time.perf_counter()
    groups = score_groups(cache, arm, payload, info["meta"], device)
    d05_groups = _d05_groups(cache, payload["parent_payload"], info["meta"])
    if arm["kind"] == "reference":
        router = copy.deepcopy(info["reference"]["router"])
        copies = {split: [dict(row, reconstruction_score=0.) for row in rows] for split, rows in groups.items()}
        diagnostics = dict(source_reproduction=_check_baseline_development(SimpleNamespace(**info["reference"]), copies))
        crossfit = None
    elif arm["kind"] == "source":
        diagnostics = dict(source_reproduction=_d05_reproduction(info, groups))
        router = copy.deepcopy(info["router"])
        crossfit = copy.deepcopy(info["calibration"]["summary"].get("crossfit_audit"))
    else:
        _d05_reproduction(info, d05_groups)
        reference = dict(records=[row for group in cache["groups"].values() for row in group["records"]],
                         router=info["reference"]["router"], calibration_settings=info["reference"]["config"]["calibration"])
        d05 = dict(records=[row for rows in d05_groups.values() for row in rows], router=info["router"])
        ordered = [groups["val_" + status] for status in base.STATUSES]
        options = dict(reference_records=reference, d05_records=d05)
        router, diagnostics = calibration.fit_router(*ordered, info["meta"], cfg["calibration"], arm["policy"], **options)
        crossfit = calibration.crossfit_audit(*ordered, info["meta"], cfg["calibration"], arm["policy"], **options)
    if arm_id == "L01_spatial_leaf":
        control_dir = suite / "arms/C02_d05_staged/calibration"
        control_receipt = _receipt(control_dir, cfg, info, "calibration", "C02_d05_staged")
        control_router = protocol.read_json(control_dir / "router.json")
        control_oof = protocol.read_json(control_dir / "crossfit_audit.json")
        for key in ("root_threshold", "root_state", "root_state_sha256"):
            if router[key] != control_router[key]:
                raise ValueError("L01 changed the frozen C02 root gate")
        if len(crossfit["folds"]) != len(control_oof["folds"]):
            raise ValueError("L01 changed C02 OOF fold inventory")
        for current, old in zip(crossfit["folds"], control_oof["folds"]):
            for key in ("fit_image_sha256", "held_image_sha256", "status", "root_state_sha256", "root_threshold"):
                if current.get(key) != old.get(key):
                    raise ValueError("L01 changed the corresponding fit-fold C02 root gate")
        diagnostics["c02_root_control"] = dict(exact_full_DEV_root=True, exact_fit_fold_OOF_roots=True,
            receipt_sha256=protocol.file_hash(control_dir / "completed.json"))
    receipt = dict(_header(cfg, info, "calibration", arm_id), model_sha256=trained["model"]["sha256"],
        training_receipt_sha256=protocol.file_hash(suite / "arms" / arm_id / "training/completed.json"),
        cache_sha256=cached["artifacts"]["features"]["sha256"], inference_spec_sha256=trained["inference_spec_sha256"])
    return _export(output, groups, router, info, arm, crossfit, diagnostics, receipt,
                   dict(seconds=time.perf_counter()-started, image_forward_count=0))


def test_arm(suite, arm_id, device="cpu"):
    suite, cfg, info = _checked(suite)
    arm = _arm(cfg, arm_id)
    selection = _regular(suite / "dev_selection.json")
    directory = suite / "arms" / arm_id / "calibration"
    calibrated = _receipt(directory, cfg, info, "calibration", arm_id)
    payload, trained = _load_model(suite, arm, cfg, info)
    if (calibrated["model_sha256"] != trained["model"]["sha256"] or calibrated["training_receipt_sha256"] !=
            protocol.file_hash(suite / "arms" / arm_id / "training/completed.json")):
        raise ValueError("Morphology TEST model differs from calibrated model")
    cache, cached = _load_cache(suite, "test", cfg, info)
    if legacy._text_contract(cache) != trained["inference_spec_sha256"]:
        raise ValueError("Morphology TEST text/core differs from training")
    router = protocol.read_json(_regular(directory / "router.json"))
    crossfit_sha256 = calibrated["artifacts"]["crossfit"]["sha256"]
    if (calibrated.get("crossfit_audit_sha256") != crossfit_sha256
            or calibrated["summary"].get("crossfit_audit_sha256") != crossfit_sha256):
        raise ValueError("Calibration compact crossfit digest differs from its artifact")
    output = _claim(suite / "arms" / arm_id / "test")
    started = time.perf_counter()
    groups = score_groups(cache, arm, payload, info["meta"], device)
    receipt = dict(_header(cfg, info, "test", arm_id), model_sha256=trained["model"]["sha256"],
        calibration_receipt_sha256=protocol.file_hash(directory / "completed.json"),
        router_sha256=calibrated["artifacts"]["router"]["sha256"], inference_spec_sha256=trained["inference_spec_sha256"],
        cache_sha256=cached["artifacts"]["features"]["sha256"], dev_selection_sha256=protocol.file_hash(selection),
        inherited_dev_targets_passed=calibrated["targets_passed"],
        calibration_crossfit_audit_sha256=crossfit_sha256)
    return _export(output, groups, router, info, arm, calibrated["summary"].get("crossfit_audit"), None,
                   receipt, dict(seconds=time.perf_counter()-started, image_forward_count=0), crossfit_sha256)
