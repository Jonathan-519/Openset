"""Exact D05 continuation with frozen evidence and an auditable CPU lifecycle.

Only the verifier may learn. Every continuation starts independently from the
same source state; original feature caches, geometry, templates, normalization
and candidate identities remain immutable. TEST only restores saved states.
"""
import copy
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

import torch

from taxosafe_support import pipeline as support
from taxosafe_support import calibration as base
from taxosafe_support import membership_calibration as membership
from taxosafe_refine.pipeline import _metrics, _check_baseline_development
from taxosafe_routealign.evaluation import _species, _csv
from taxosafe_discovery import backend as legacy
from taxosafe_discovery import calibration as legacy_calibration
from . import protocol, calibration, importer


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
            or snapshot.get("source_binding") != binding):
        raise ValueError("Recovery parent/code/configuration changed")
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
        raise ValueError("Recovery stage receipt/source signature mismatch")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("Missing recovery artifacts")
    for descriptor in artifacts.values():
        if (not isinstance(descriptor, dict) or set(descriptor) != {"path", "sha256"}
                or Path(descriptor["path"]).name != descriptor["path"]
                or protocol.file_hash(_regular(directory / descriptor["path"])) != descriptor["sha256"]):
            raise ValueError("Changed or escaping recovery artifact")
    return receipt


def _arm(cfg, arm_id):
    matches = [arm for arm in cfg["arms"] if arm["id"] == arm_id]
    if len(matches) != 1:
        raise ValueError("Unknown recovery arm")
    return matches[0]


def prepare_cache(suite, stage, device="cpu"):
    del device
    if stage not in ("train", "development", "test"):
        raise ValueError("Unknown recovery cache stage")
    suite, cfg, info = _checked(suite)
    if stage == "test":
        _regular(suite / "dev_selection.json")
        cache, parent_receipt = importer.load_parent_test_cache(info, suite)
    else:
        cache, parent_receipt = importer.load_parent_cache(info, stage)
    legacy._validate_cache(cache, info["reference"], stage)
    if legacy._text_contract(cache) != info["training"]["inference_spec_sha256"]:
        raise ValueError("Recovery cache text/core differs from source D05")
    output = _claim(suite / "cache" / stage)
    header = _header(cfg, info, stage)
    binding = dict(parent_cache_sha256=parent_receipt["artifacts"]["features"]["sha256"],
                   parent_cache_receipt_sha256=protocol.file_hash(info["directory"] / "cache" / stage / "completed.json"),
                   inference_spec_sha256=legacy._text_contract(cache))
    payload = dict(header, **binding, cache=cache, audit=parent_receipt["audit"])
    support._save_torch(output / "features.pth", payload)
    receipt = dict(header, **binding, audit=parent_receipt["audit"], provenance=cache["provenance"],
                   timings={"cache_reused": True, "image_forward_count": 0, "parent": cache["timings"]})
    if stage == "test":
        receipt["dev_selection_sha256"] = protocol.file_hash(suite / "dev_selection.json")
    return _finish(output, receipt, {"features": "features.pth"})


def _load_cache(suite, stage, cfg, info):
    directory = Path(suite) / "cache" / stage
    receipt = _receipt(directory, cfg, info, stage)
    payload = support._load_torch(directory / "features.pth")
    if any(payload.get(key) != value for key, value in _header(cfg, info, stage).items()):
        raise ValueError("Recovery cache header changed")
    cache = payload["cache"]
    legacy._validate_cache(cache, info["reference"], stage)
    for key in ("parent_cache_sha256", "parent_cache_receipt_sha256", "inference_spec_sha256", "audit"):
        if payload.get(key) != receipt.get(key):
            raise ValueError("Recovery cache receipt binding differs")
    if (legacy._text_contract(cache) != receipt["inference_spec_sha256"]
            or receipt["inference_spec_sha256"] != info["training"]["inference_spec_sha256"]):
        raise ValueError("Recovery cache changed source text/core semantics")
    if stage == "test" and receipt.get("dev_selection_sha256") != protocol.file_hash(_regular(Path(suite) / "dev_selection.json")):
        raise ValueError("TEST cache differs from frozen recovery DEV selection")
    parent = info["directory"] / "cache" / stage
    if (protocol.file_hash(_regular(parent / "features.pth")) != receipt["parent_cache_sha256"]
            or protocol.file_hash(_regular(parent / "completed.json")) != receipt["parent_cache_receipt_sha256"]):
        raise ValueError("Original discovery cache changed")
    return cache, receipt


def _episodes(cache, source, info):
    from taxosafe_discovery.verifier import build_episodes
    group = cache["groups"]["train"]
    rows = legacy._aligned_rows(group)
    if any(row["split"] != "train" or row["status"] != "known" for row in rows):
        raise ValueError("Recovery gradients require known TRAIN only")
    feature = group["features"]["clip"]
    labels = torch.tensor([row["true_leaf"] for row in rows], dtype=torch.long)
    episodes = build_episodes(feature, feature, labels, group["image_sha256"], info["meta"],
        template_scores=legacy._templates(feature, source.payload["text"]),
        folds=info["config"]["verifier"]["folds"], seed=info["config"]["seed"], **info["config"]["geometry"])
    original = source.payload["verifier"]["fit_report"]["episode_report"]
    for key in ("image_hash_digest", "train_count", "seed", "folds", "covariance_shrinkage", "feature_names"):
        if episodes["report"].get(key) != original.get(key):
            raise ValueError("Reconstructed TRAIN episode recipe differs from source D05: " + key)
    for level in ("leaf", "parent"):
        for key in ("feature_sha256", "target_sha256", "ranking_pairs_sha256", "rows"):
            if episodes["report"]["examples"][level][key] != original["examples"][level][key]:
                raise ValueError("Reconstructed TRAIN episode bytes differ from source D05: " + level + "." + key)
    candidates = membership.candidate_scores(rows, info["meta"])
    identities = {level: torch.as_tensor(candidates[level], dtype=torch.long) for level in ("leaf", "parent")}
    audit = dict(exact_source_episode_features=True, exact_source_episode_targets=True,
                 exact_source_ranking_pairs=True, episode_report=episodes["report"],
                 candidate_sha256=_semantic(identities), candidate_source="immutable_reference_TRAIN_predictions",
                 episode_seed=info["config"]["seed"], continuation_seed_is_separate=True)
    return episodes, identities, audit


def fit_arm(suite, arm_id, device="cpu"):
    del device
    suite, cfg, info = _checked(suite)
    arm = _arm(cfg, arm_id)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    cache, cached = _load_cache(suite, "train", cfg, info)
    output = _claim(suite / "arms" / arm_id / "training")
    header = _header(cfg, info, "training", arm_id)
    inherited = arm.get("weight_source")
    if arm["kind"] == "reuse":
        origin = suite / "arms" / inherited / "training"
        _, previous = _load_model(suite, _arm(cfg, inherited), cfg, info)
        shutil.copyfile(origin / "model.pth", output / "model.pth")
        report = dict(training_execution="reused", optimizer_steps=0, source_arm=inherited,
                      reused_optimizer_steps=previous["optimizer_steps"])
        receipt = dict(header, optimizer_steps=0, weight_source=inherited, fit_report=report,
                       inference_spec_sha256=previous["inference_spec_sha256"],
                       train_cache_sha256=cached["artifacts"]["features"]["sha256"],
                       frozen_d05_payload_sha256=previous["frozen_d05_payload_sha256"],
                       reused_training_receipt_sha256=protocol.file_hash(origin / "completed.json"))
    else:
        source = importer.load_d05(info["directory"], expected_binding=info["binding"])
        parent_digest = _semantic(source.payload)
        payload = dict(header, model_spec=arm, parent_payload=copy.deepcopy(source.payload),
                       recovery_verifier=None, frozen_d05_payload_sha256=parent_digest,
                       inference_spec_sha256=legacy._text_contract(source.payload),
                       train_cache_sha256=cached["artifacts"]["features"]["sha256"])
        report = dict(training_execution="source_inherited", optimizer_steps=0,
                      initialized_from="original_D05", source_model_sha256=info["training"]["model"]["sha256"],
                      frozen_geometry_text_features_candidates_normalization=True)
        if arm["kind"] == "finetune":
            from .training import finetune
            episodes, candidates, audit = _episodes(cache, source, info)
            state, continued = finetune(source.payload["verifier"], episodes, candidates,
                                       mode=arm["mode"], options=cfg["training"], seed=cfg["seed"])
            payload["recovery_verifier"] = state
            report.update(training_execution="completed", optimizer_steps=continued["optimizer_steps"],
                          continuation=continued, source_episode_reproduction=audit)
        elif arm["kind"] == "leaf_guard":
            from .training import make_leaf_guard
            origin, previous = _load_model(suite, _arm(cfg, inherited), cfg, info)
            state, guard_report = make_leaf_guard(origin["recovery_verifier"], source.payload["verifier"])
            payload["recovery_verifier"] = state
            report.update(training_execution="assembled_leaf_guard", optimizer_steps=0,
                          leaf_guard=guard_report, source_arm=inherited,
                          source_training_receipt_sha256=protocol.file_hash(suite / "arms" / inherited / "training/completed.json"))
        if _semantic(source.payload) != parent_digest:
            raise ValueError("Continuation mutated immutable D05 payload")
        payload["fit_report"] = report
        support._save_torch(output / "model.pth", payload)
        receipt = dict(header, optimizer_steps=report["optimizer_steps"], weight_source=inherited or
                       ("reference" if arm["kind"] == "reference" else "original_D05" if arm["kind"] == "source" else arm_id),
                       fit_report=report, inference_spec_sha256=payload["inference_spec_sha256"],
                       train_cache_sha256=payload["train_cache_sha256"], frozen_d05_payload_sha256=parent_digest)
    protocol.write_json(output / "training_report.json", report)
    receipt["model"] = dict(path="model.pth", sha256=protocol.file_hash(output / "model.pth"))
    return _finish(output, receipt, {"model": "model.pth", "training_report": "training_report.json"})


def _load_model(suite, arm, cfg, info):
    directory = Path(suite) / "arms" / arm["id"] / "training"
    receipt = _receipt(directory, cfg, info, "training", arm["id"])
    payload = support._load_torch(directory / "model.pth")
    owner = arm["weight_source"] if arm["kind"] == "reuse" else arm["id"]
    if any(payload.get(key) != value for key, value in _header(cfg, info, "training", owner).items()):
        raise ValueError("Recovery model owner/source signature differs")
    source = importer.load_d05(info["directory"], expected_binding=info["binding"])
    digest = _semantic(source.payload)
    if (payload.get("model_spec") != _arm(cfg, owner)
            or _semantic(payload["parent_payload"]) != digest
            or payload.get("frozen_d05_payload_sha256") != digest or receipt.get("frozen_d05_payload_sha256") != digest
            or legacy._text_contract(payload["parent_payload"]) != receipt["inference_spec_sha256"]
            or payload.get("inference_spec_sha256") != receipt["inference_spec_sha256"]
            or payload.get("train_cache_sha256") != receipt["train_cache_sha256"]):
        raise ValueError("Recovery changed frozen D05 evidence/model binding")
    owner_arm = _arm(cfg, owner)
    state = payload.get("recovery_verifier")
    if (state is not None) != (owner_arm["kind"] in ("finetune", "leaf_guard")):
        raise ValueError("Recovery verifier state does not match the declared arm")
    if state is not None:
        from .training import RecoveryVerifier
        RecoveryVerifier.from_state_dict(state)
        if _semantic(state["source_state"]) != _semantic(source.payload["verifier"]):
            raise ValueError("Recovery verifier was initialized from another model")
        if (state["audit"]["mode"] != owner_arm["mode"] or state["audit"]["options"] != cfg["training"]
                or state["audit"]["seed"] != cfg["seed"]):
            raise ValueError("Recovery verifier training plan differs from the declared arm")
    owner_receipt = receipt
    if arm["kind"] == "reuse":
        origin = Path(suite) / "arms" / owner / "training"
        previous = _receipt(origin, cfg, info, "training", owner)
        if (receipt.get("reused_training_receipt_sha256") != protocol.file_hash(origin / "completed.json")
                or receipt["model"] != previous["model"]):
            raise ValueError("Recovery reused checkpoint origin changed")
        owner_receipt = previous
    if payload.get("fit_report") != owner_receipt.get("fit_report"):
        raise ValueError("Recovery model training report differs from its receipt")
    if arm["kind"] == "leaf_guard":
        origin = Path(suite) / "arms" / arm["weight_source"] / "training/completed.json"
        if payload["fit_report"].get("source_training_receipt_sha256") != protocol.file_hash(_regular(origin)):
            raise ValueError("Leaf guard source training changed")
        origin_payload, _ = _load_model(suite, _arm(cfg, arm["weight_source"]), cfg, info)
        origin_state = origin_payload["recovery_verifier"]
        if (state["audit"].get("composition_source_state_sha256") != origin_state["state_sha256"]
                or _semantic(state["updated_heads"]["leaf"]) != _semantic(origin_state["updated_heads"]["leaf"])):
            raise ValueError("Leaf guard does not reuse the declared F06 leaf state")
    return payload, receipt


def _d05_groups(cache, payload, meta):
    return legacy.score_groups(cache, payload["model_spec"], payload, meta)


def score_groups(cache, arm, payload, meta):
    if arm["kind"] == "reference":
        return {split: copy.deepcopy(group["records"]) for split, group in cache["groups"].items()}
    parent = payload["parent_payload"]
    if payload["recovery_verifier"] is None:
        return _d05_groups(cache, parent, meta)
    from taxosafe_discovery.geometry import GeometryBank
    from .training import RecoveryVerifier
    geometry = GeometryBank.from_state_dict(parent["geometry"])
    verifier = RecoveryVerifier.from_state_dict(payload["recovery_verifier"])
    result = {}
    for split, group in cache["groups"].items():
        rows = copy.deepcopy(group["records"])
        feature = group["features"]["clip"]
        geometry_scores = geometry.score(feature, feature, group["image_sha256"])
        scored = verifier.score(geometry_scores, legacy._templates(feature, parent["text"]))
        values = {level: scored[level + "_scores"].detach().cpu().double() for level in ("leaf", "parent")}
        if any(not bool(torch.isfinite(value).all()) for value in values.values()):
            raise ValueError("Recovery produced nonfinite evidence")
        identities = membership.candidate_scores(rows, meta)
        for i, row in enumerate(rows):
            index = group["record_feature_indices"][i]
            row["discovery"] = dict(leaf_scores=values["leaf"][index].tolist(),
                parent_scores=values["parent"][index].tolist(), candidate_leaf=int(identities["leaf"][i]),
                candidate_parent=int(identities["parent"][i]))
            row.update(discovery_method="recovery_" + arm["mode"],
                       support_evidence_origin="immutable_reference_diagnostic", log_probs_origin="immutable_reference_diagnostic")
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


def _export(output, groups, router, info, arm, crossfit, diagnostics, receipt, timings, d05_groups=None):
    decode = membership.decode_records if arm["kind"] == "reference" else calibration.decode_records
    routed = {split: decode(rows, router, info["meta"]) for split, rows in groups.items()}
    predictions = [row for rows in routed.values() for row in rows]
    base.unique_records(predictions)
    seen = set()
    for row in predictions:
        row["evaluation_weight"] = int(row["image_sha256"] not in seen)
        seen.add(row["image_sha256"])
    preservation = None
    if arm["kind"] == "leaf_guard":
        reference = {row["image_sha256"]: row for rows in d05_groups.values() for row in
                     legacy_calibration.decode_records(rows, info["router"], info["meta"])}
        originals = {row["image_sha256"]: row for rows in d05_groups.values() for row in rows}
        preservation = dict(parent_vectors_exact=all(row["discovery"]["parent_scores"] == originals[row["image_sha256"]]["discovery"]["parent_scores"] for row in predictions),
            candidates_exact=all(all(row[key] == reference[row["image_sha256"]][key] for key in ("candidate_leaf", "candidate_parent")) for row in predictions),
            root_decisions_exact=all((row["prediction_type"] == "global_unknown") ==
                (reference[row["image_sha256"]]["prediction_type"] == "global_unknown") for row in predictions))
        if not all(preservation.values()):
            raise ValueError("Leaf guard changed D05 parent evidence, candidates or root decisions")
    summary = base.evaluate_records(predictions, info["meta"])
    summary["crossfit_audit"] = crossfit
    report = dict(summary, schema_version="recovery_evaluation_v1", arm_id=arm["id"], stage=receipt["stage"],
        calibration_status="passed" if summary["targets_passed"] else "best_effort",
        calibration_gate_is_execution_gate=False, test_allowed_after_failed_gates=True,
        arm_predictions_replaced_by_reference=False, test_used_for_fitting=False,
        confirmatory_validation=False, independent_model_level_validation=False,
        validation_scope="exploratory_D05_continuation_on_reused_benchmark",
        candidate_policy="immutable_reference", score_semantics="frozen-evidence verifier logits; not probabilities",
        calibration_diagnostics=diagnostics, leaf_guard_preservation=preservation)
    metrics = _metrics({split: base.unique_records(rows) for split, rows in routed.items()})
    protocol.write_records(output / "scores.jsonl", [row for rows in groups.values() for row in rows])
    protocol.write_records(output / "predictions.jsonl", predictions)
    names = dict(scores="scores.jsonl", predictions="predictions.jsonl", report="report.json", summary="summary.json",
                 metrics="metrics.json", router="router.json", species="per_species.csv", timing="inference_timing.json")
    for key, value in (("report", report), ("summary", summary), ("metrics", metrics), ("router", router), ("timing", timings)):
        protocol.write_json(output / names[key], value)
    _csv(output / names["species"], _species(predictions, info["meta"]))
    receipt.update(summary=summary, targets_passed=summary["targets_passed"],
                   test_allowed_after_failed_gates=True, calibration_gate_is_execution_gate=False,
                   leaf_guard_preservation=preservation)
    return _finish(output, receipt, names)


def calibrate_arm(suite, arm_id, device="cpu"):
    del device
    suite, cfg, info = _checked(suite)
    arm = _arm(cfg, arm_id)
    cache, cached = _load_cache(suite, "development", cfg, info)
    payload, trained = _load_model(suite, arm, cfg, info)
    if legacy._text_contract(cache) != trained["inference_spec_sha256"]:
        raise ValueError("Recovery TRAIN/DEV text or core differs")
    output = _claim(suite / "arms" / arm_id / "calibration")
    started = time.perf_counter()
    groups = score_groups(cache, arm, payload, info["meta"])
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
        # Verify the imported numerical baseline independently for every arm.
        # A local F01 output-directory failure must not block other controls.
        _d05_reproduction(info, d05_groups)
        reference = dict(records=[row for group in cache["groups"].values() for row in group["records"]],
                         router=info["reference"]["router"], calibration_settings=info["reference"]["config"]["calibration"])
        d05 = dict(records=[row for rows in d05_groups.values() for row in rows], router=info["router"])
        ordered = [groups["val_" + status] for status in base.STATUSES]
        options = dict(reference_records=reference, d05_records=d05)
        router, diagnostics = calibration.fit_router(*ordered, info["meta"], cfg["calibration"], arm["policy"], **options)
        crossfit = calibration.crossfit_audit(*ordered, info["meta"], cfg["calibration"], arm["policy"], **options)
    receipt = dict(_header(cfg, info, "calibration", arm_id), model_sha256=trained["model"]["sha256"],
        training_receipt_sha256=protocol.file_hash(suite / "arms" / arm_id / "training/completed.json"),
        cache_sha256=cached["artifacts"]["features"]["sha256"], inference_spec_sha256=trained["inference_spec_sha256"])
    return _export(output, groups, router, info, arm, crossfit, diagnostics, receipt,
                   dict(seconds=time.perf_counter()-started, image_forward_count=0), d05_groups)


def test_arm(suite, arm_id, device="cpu"):
    del device
    suite, cfg, info = _checked(suite)
    arm = _arm(cfg, arm_id)
    selection = _regular(suite / "dev_selection.json")
    directory = suite / "arms" / arm_id / "calibration"
    calibrated = _receipt(directory, cfg, info, "calibration", arm_id)
    payload, trained = _load_model(suite, arm, cfg, info)
    if (calibrated["model_sha256"] != trained["model"]["sha256"] or calibrated["training_receipt_sha256"] !=
            protocol.file_hash(suite / "arms" / arm_id / "training/completed.json")):
        raise ValueError("Recovery TEST checkpoint differs from calibrated model")
    cache, cached = _load_cache(suite, "test", cfg, info)
    if legacy._text_contract(cache) != trained["inference_spec_sha256"]:
        raise ValueError("Recovery TEST text/core differs from training")
    router = protocol.read_json(_regular(directory / "router.json"))
    output = _claim(suite / "arms" / arm_id / "test")
    started = time.perf_counter()
    groups = score_groups(cache, arm, payload, info["meta"])
    d05_groups = _d05_groups(cache, payload["parent_payload"], info["meta"]) if arm["kind"] == "leaf_guard" else None
    receipt = dict(_header(cfg, info, "test", arm_id), model_sha256=trained["model"]["sha256"],
        calibration_receipt_sha256=protocol.file_hash(directory / "completed.json"),
        router_sha256=calibrated["artifacts"]["router"]["sha256"], inference_spec_sha256=trained["inference_spec_sha256"],
        cache_sha256=cached["artifacts"]["features"]["sha256"], dev_selection_sha256=protocol.file_hash(selection))
    return _export(output, groups, router, info, arm, calibrated["summary"].get("crossfit_audit"), None, receipt,
                   dict(seconds=time.perf_counter()-started, image_forward_count=0), d05_groups)
