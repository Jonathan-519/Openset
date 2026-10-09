"""Boundary evidence and paired-ranking controls over immutable image caches.

New EVM statistics and verifier heads fit only known TRAIN. Original CLIP,
geometry, text, candidates and the first eight normalization channels remain
fixed. Source controls reproduce D05 exactly; TEST only restores saved states.
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
        raise ValueError("Boundary parent/code/configuration changed")
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
        raise ValueError("Boundary stage receipt/source signature mismatch")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("Missing boundary artifacts")
    for descriptor in artifacts.values():
        if (not isinstance(descriptor, dict) or set(descriptor) != {"path", "sha256"}
                or Path(descriptor["path"]).name != descriptor["path"]
                or protocol.file_hash(_regular(directory / descriptor["path"])) != descriptor["sha256"]):
            raise ValueError("Changed or escaping boundary artifact")
    return receipt


def _arm(cfg, arm_id):
    matches = [arm for arm in cfg["arms"] if arm["id"] == arm_id]
    if len(matches) != 1:
        raise ValueError("Unknown boundary arm")
    return matches[0]


def prepare_cache(suite, stage, device="cpu"):
    del device
    if stage not in ("train", "development", "test"):
        raise ValueError("Unknown boundary cache stage")
    suite, cfg, info = _checked(suite)
    if stage == "test":
        _regular(suite / "dev_selection.json")
        cache, parent_receipt = importer.load_parent_test_cache(info, suite)
    else:
        cache, parent_receipt = importer.load_parent_cache(info, stage)
    legacy._validate_cache(cache, info["reference"], stage)
    if legacy._text_contract(cache) != info["training"]["inference_spec_sha256"]:
        raise ValueError("Boundary cache text/core differs from source D05")
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
        raise ValueError("Boundary cache header changed")
    cache = payload["cache"]
    legacy._validate_cache(cache, info["reference"], stage)
    for key in ("parent_cache_sha256", "parent_cache_receipt_sha256", "inference_spec_sha256", "audit"):
        if payload.get(key) != receipt.get(key):
            raise ValueError("Boundary cache receipt binding differs")
    if (legacy._text_contract(cache) != receipt["inference_spec_sha256"]
            or receipt["inference_spec_sha256"] != info["training"]["inference_spec_sha256"]):
        raise ValueError("Boundary cache changed source text/core semantics")
    if stage == "test" and receipt.get("dev_selection_sha256") != protocol.file_hash(_regular(Path(suite) / "dev_selection.json")):
        raise ValueError("TEST cache differs from frozen boundary DEV selection")
    parent = info["directory"] / "cache" / stage
    if (protocol.file_hash(_regular(parent / "features.pth")) != receipt["parent_cache_sha256"]
            or protocol.file_hash(_regular(parent / "completed.json")) != receipt["parent_cache_receipt_sha256"]):
        raise ValueError("Original discovery cache changed")
    # The child receipt can authenticate its own file without proving that its
    # tensors/records are the frozen source values. Compare to the parent whose
    # artifact digest was just checked, including aliases and candidates.
    original = support._load_torch(parent / "features.pth")
    if not isinstance(original, dict) or _semantic(cache) != _semantic(original.get("cache")):
        raise ValueError("Boundary cache differs from frozen D05 cache contents")
    return cache, receipt


def aligned_group(group, meta, training=False):
    """One validated image/hash order, with explicit raw-record alias expansion."""
    from taxosafe_routealign.proximity import _features
    raw = list(group["records"])
    unique = base.unique_records(raw)
    by_hash = {row["image_sha256"]: row for row in unique}
    hashes = list(group["image_sha256"])
    if len(hashes) != len(set(hashes)) or set(hashes) != set(by_hash):
        raise ValueError("Boundary feature rows and unique record identities differ")
    rows = [by_hash[digest] for digest in hashes]
    indices = group["record_feature_indices"]
    if (len(indices) != len(raw) or any(type(i) is not int or not 0 <= i < len(hashes)
            or hashes[i] != row["image_sha256"] for row, i in zip(raw, indices))):
        raise ValueError("Boundary raw alias-to-feature mapping differs")
    feature = group["features"]["clip"]
    unit = _features(feature, "cached CLIP", count=len(hashes))
    labels = None
    if training:
        if not rows or any(row["split"] != "train" or row["status"] != "known" for row in rows):
            raise ValueError("Boundary fitting requires nonempty known TRAIN only")
        for row in rows:
            leaf = row.get("true_leaf")
            if (type(leaf) is not int or not 0 <= leaf < len(meta["leaf_names"])
                    or row.get("true_parent") != meta["leaf_to_parent"][leaf]):
                raise ValueError("Boundary TRAIN annotations differ from the locked taxonomy")
        labels = torch.tensor([row["true_leaf"] for row in rows], dtype=torch.long)
    selected = membership.candidate_scores(rows, meta)
    candidates = {level: torch.as_tensor(selected[level], dtype=torch.long) for level in ("leaf", "parent")}
    return dict(records=rows, raw_records=raw, raw_clip=feature, unit_clip=unit, image_sha256=hashes,
                labels=labels, record_feature_indices=indices, candidates=candidates)


def _episodes(cache, source, info, cfg):
    from .core import build_episodes
    view = aligned_group(cache["groups"]["train"], info["meta"], training=True)
    feature = view["raw_clip"]
    episodes = build_episodes(feature, feature, view["labels"], view["image_sha256"], info["meta"],
        template_scores=legacy._templates(feature, source.payload["text"]),
        folds=info["config"]["verifier"]["folds"], seed=info["config"]["seed"],
        **info["config"]["geometry"], **cfg["boundary"])
    audit = dict(episode_report=episodes["report"],
                 input_feature_sha256=legacy.tensor_hash(feature), train_image_sha256=view["image_sha256"],
                 label_sha256=legacy.tensor_hash(view["labels"]),
                 candidate_sha256=_semantic(view["candidates"]), candidate_source="immutable_reference_TRAIN_predictions",
                 episode_seed=info["config"]["seed"], folds=info["config"]["verifier"]["folds"],
                 geometry_settings=copy.deepcopy(info["config"]["geometry"]),
                 boundary_settings=copy.deepcopy(cfg["boundary"]),
                 ranking_scope="same_bank_same_candidate_different_query",
                 identical_to_old_cross_bank_pairs=False, continuation_seed_is_separate=True)
    return episodes, audit


def _bank_required(arm):
    return arm["kind"] in ("evm", "leaf_guard") or arm["mode"] in ("bce9", "rank9")


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
                       boundary_verifier=None, boundary_bank=None, frozen_d05_payload_sha256=parent_digest,
                       inference_spec_sha256=legacy._text_contract(source.payload),
                       train_cache_sha256=cached["artifacts"]["features"]["sha256"])
        report = dict(training_execution="source_inherited", optimizer_steps=0,
                      initialized_from="original_D05", source_model_sha256=info["training"]["model"]["sha256"],
                      frozen_geometry_text_features_candidates_and_source8_normalization=True)
        if _bank_required(arm) and arm["kind"] != "leaf_guard":
            from .core import BoundaryBank
            view = aligned_group(cache["groups"]["train"], info["meta"], training=True)
            bank = BoundaryBank.fit(view["raw_clip"], view["raw_clip"], view["labels"],
                                    view["image_sha256"], info["meta"], **cfg["boundary"])
            payload["boundary_bank"] = bank.state_dict()
            report.update(training_execution="deterministic_TRAIN_boundary_fit", boundary_bank=bank.fit_report)
        if arm["kind"] == "train":
            from .core import BoundaryVerifier
            episodes, audit = _episodes(cache, source, info, cfg)
            model = BoundaryVerifier.fit(source.payload["verifier"], episodes,
                                         mode=arm["mode"], seed=cfg["seed"], **cfg["training"])
            payload["boundary_verifier"] = model.state_dict()
            report.update(training_execution="completed", optimizer_steps=model.fit_report["optimizer_steps"],
                          verifier=model.fit_report, episode_audit=audit)
        elif arm["kind"] == "leaf_guard":
            from .core import make_leaf_guard
            origin, previous = _load_model(suite, _arm(cfg, inherited), cfg, info)
            state, guard_report = make_leaf_guard(origin["boundary_verifier"], source.payload["verifier"])
            payload["boundary_verifier"] = state
            payload["boundary_bank"] = copy.deepcopy(origin["boundary_bank"])
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
        raise ValueError("Boundary model owner/source signature differs")
    source = importer.load_d05(info["directory"], expected_binding=info["binding"])
    digest = _semantic(source.payload)
    if (payload.get("model_spec") != _arm(cfg, owner)
            or _semantic(payload["parent_payload"]) != digest
            or payload.get("frozen_d05_payload_sha256") != digest or receipt.get("frozen_d05_payload_sha256") != digest
            or legacy._text_contract(payload["parent_payload"]) != receipt["inference_spec_sha256"]
            or payload.get("inference_spec_sha256") != receipt["inference_spec_sha256"]
            or payload.get("train_cache_sha256") != receipt["train_cache_sha256"]):
        raise ValueError("Boundary changed frozen D05 evidence/model binding")
    owner_arm = _arm(cfg, owner)
    state = payload.get("boundary_verifier")
    if (state is not None) != (owner_arm["kind"] in ("train", "leaf_guard")):
        raise ValueError("Boundary verifier state does not match the declared arm")
    if state is not None:
        from .core import BoundaryVerifier
        model = BoundaryVerifier.from_state_dict(state)
        if _semantic(state["source_state"]) != _semantic(source.payload["verifier"]):
            raise ValueError("Boundary verifier was initialized from another model")
        audit = model.fit_report
        if (audit["mode"] != owner_arm["mode"] or audit["seed"] != cfg["seed"]
                or (owner_arm["kind"] == "train" and any(audit.get(key) != value for key, value in cfg["training"].items()))):
            raise ValueError("Boundary verifier training plan differs from the declared arm")
        for level in ("leaf", "parent"):
            for key in ("mean", "scale"):
                if not torch.equal(state["normalization"][level][key][:8], source.payload["verifier"]["normalization"][level][key]):
                    raise ValueError("Boundary changed a source normalization channel")
    bank_state = payload.get("boundary_bank")
    if (bank_state is not None) != _bank_required(owner_arm):
        raise ValueError("Boundary bank does not match the declared evidence mechanism")
    if bank_state is not None:
        from .core import BoundaryBank
        bank = BoundaryBank.from_state_dict(bank_state)
        if (bank.meta != info["meta"] or bank.tail_size != cfg["boundary"]["tail_size"]
                or list(bank.image_hashes) != list(source.geometry.image_hashes)
                or not torch.equal(bank.labels, source.geometry.labels)
                or not torch.equal(bank.fine, source.geometry.fine)
                or not torch.equal(bank.parent, source.geometry.parent)):
            raise ValueError("Boundary bank differs from the immutable TRAIN support")
    owner_receipt = receipt
    if arm["kind"] == "reuse":
        origin = Path(suite) / "arms" / owner / "training"
        previous = _receipt(origin, cfg, info, "training", owner)
        if (receipt.get("reused_training_receipt_sha256") != protocol.file_hash(origin / "completed.json")
                or receipt["model"] != previous["model"]):
            raise ValueError("Boundary reused checkpoint origin changed")
        owner_receipt = previous
    if payload.get("fit_report") != owner_receipt.get("fit_report"):
        raise ValueError("Boundary model training report differs from its receipt")
    if owner_arm["kind"] == "train" and payload["fit_report"].get("verifier") != state["fit_report"]:
        raise ValueError("Boundary verifier report differs from the trained state")
    if owner_arm["kind"] == "evm" and payload["fit_report"].get("boundary_bank") != bank.fit_report:
        raise ValueError("Boundary fitting report differs from its saved bank")
    if arm["kind"] == "leaf_guard":
        origin = Path(suite) / "arms" / arm["weight_source"] / "training/completed.json"
        if payload["fit_report"].get("source_training_receipt_sha256") != protocol.file_hash(_regular(origin)):
            raise ValueError("Leaf guard source training changed")
        origin_payload, _ = _load_model(suite, _arm(cfg, arm["weight_source"]), cfg, info)
        origin_state = origin_payload["boundary_verifier"]
        if (state["fit_report"].get("composition_trained_state_sha256") != origin_state["state_sha256"]
                or _semantic(state["heads"]["leaf"]) != _semantic(origin_state["heads"]["leaf"])
                or _semantic(payload["boundary_bank"]) != _semantic(origin_payload["boundary_bank"])
                or _semantic(state["heads"]["parent"]) != _semantic(source.payload["verifier"]["heads"]["parent"])
                or payload["fit_report"].get("leaf_guard") != state["fit_report"]):
            raise ValueError("Leaf guard does not reuse the declared G07 leaf state")
    return payload, receipt


def _d05_groups(cache, payload, meta):
    return legacy.score_groups(cache, payload["model_spec"], payload, meta)


def score_groups(cache, arm, payload, meta):
    if arm["kind"] == "reference":
        return {split: copy.deepcopy(group["records"]) for split, group in cache["groups"].items()}
    parent = payload["parent_payload"]
    if payload["boundary_verifier"] is None and payload["boundary_bank"] is None:
        return _d05_groups(cache, parent, meta)
    from taxosafe_discovery.geometry import GeometryBank
    from taxosafe_discovery.verifier import SharedVerifier
    from .core import BoundaryBank, BoundaryVerifier
    geometry = GeometryBank.from_state_dict(parent["geometry"])
    verifier = (BoundaryVerifier.from_state_dict(payload["boundary_verifier"])
                if payload["boundary_verifier"] is not None else None)
    bank = BoundaryBank.from_state_dict(payload["boundary_bank"]) if payload["boundary_bank"] is not None else None
    source_verifier = SharedVerifier.from_state_dict(parent["verifier"]) if verifier is None else None
    result = {}
    for split, group in cache["groups"].items():
        view = aligned_group(group, meta)
        rows = copy.deepcopy(view["raw_records"])
        feature = view["raw_clip"]
        geometry_scores = geometry.score(feature, feature, view["image_sha256"])
        templates = legacy._templates(feature, parent["text"])
        boundary_scores = bank.score(feature, feature, view["image_sha256"]) if bank is not None else None
        if verifier is None:
            # G03 isolates nonparametric leaf evidence; its parent vector is
            # exactly the frozen D05 vector, with no averaged proxy or refit.
            original = source_verifier.score(geometry_scores, templates)
            scored = dict(leaf_scores=boundary_scores["leaf_scores"], parent_scores=original["parent_scores"])
        else:
            scored = verifier.score(geometry_scores, templates, boundary_scores)
        values = {level: scored[level + "_scores"].detach().cpu().double() for level in ("leaf", "parent")}
        if any(not bool(torch.isfinite(value).all()) for value in values.values()):
            raise ValueError("Boundary produced nonfinite evidence")
        identities = membership.candidate_scores(rows, meta)
        for i, row in enumerate(rows):
            index = group["record_feature_indices"][i]
            row["discovery"] = dict(leaf_scores=values["leaf"][index].tolist(),
                parent_scores=values["parent"][index].tolist(), candidate_leaf=int(identities["leaf"][i]),
                candidate_parent=int(identities["parent"][i]))
            row.update(discovery_method="boundary_" + arm["mode"],
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
            parent_thresholds_exact=all(router.get(key) == info["router"].get(key) for key in
                ("global_parent_threshold", "parent_thresholds", "parent_offsets")),
            candidates_exact=all(all(row[key] == reference[row["image_sha256"]][key] for key in ("candidate_leaf", "candidate_parent")) for row in predictions),
            root_decisions_exact=all((row["prediction_type"] == "global_unknown") ==
                (reference[row["image_sha256"]]["prediction_type"] == "global_unknown") for row in predictions))
        if not all(preservation.values()):
            raise ValueError("Leaf guard changed D05 parent evidence, thresholds, candidates or root decisions")
    summary = base.evaluate_records(predictions, info["meta"])
    summary["crossfit_audit"] = crossfit
    report = dict(summary, schema_version="boundary_evaluation_v1", arm_id=arm["id"], stage=receipt["stage"],
        calibration_status="passed" if summary["targets_passed"] else "best_effort",
        calibration_gate_is_execution_gate=False, test_allowed_after_failed_gates=True,
        arm_predictions_replaced_by_reference=False, test_used_for_fitting=False,
        confirmatory_validation=False, independent_model_level_validation=False,
        validation_scope="exploratory_boundary_mechanisms_on_reused_benchmark",
        candidate_policy="immutable_reference", score_semantics="EVM inclusion scores or verifier logits; not calibrated posteriors",
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
        raise ValueError("Boundary TRAIN/DEV text or core differs")
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
        # A local G01 output-directory failure must not block other controls.
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
        raise ValueError("Boundary TEST checkpoint differs from calibrated model")
    cache, cached = _load_cache(suite, "test", cfg, info)
    if legacy._text_contract(cache) != trained["inference_spec_sha256"]:
        raise ValueError("Boundary TEST text/core differs from training")
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
