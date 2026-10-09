"""Audited cache -> TRAIN-only fit -> DEV freeze -> read-only TEST backend.

The immutable reference remains the classifier for D01--D08. New evidence
models estimate acceptance independently. D09/D10 explicitly study a changed
text classifier. No stage silently replaces a failed arm with the reference.
"""
import copy
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import shutil
import time

import torch

from taxosafe_support import pipeline as support
from taxosafe_support import calibration as base
from taxosafe_support import membership_calibration as membership
from taxosafe_refine.importer import inspect_reference, load_reference
from taxosafe_refine.pipeline import _stage_rows, load_training_rows, _check_baseline_development, _metrics
from taxosafe_routealign.evaluation import _species, _csv
from . import protocol, calibration


def _regular(path):
    path = Path(path)
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
        raise ValueError("Expected a regular artifact without symlink ancestors: " + str(path))
    return path


def _claim(path):
    path = Path(path)
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Artifact directories must not traverse symlinks")
    path.mkdir(parents=True, exist_ok=False)
    return path


def _checked(suite):
    suite = Path(suite).resolve()
    cfg = protocol.validate_config(protocol.read_json(_regular(suite / "config.json")))
    binding = protocol.read_json(_regular(suite / "source_binding.json"))
    info = inspect_reference(binding["directory"])
    snapshot = protocol.read_json(_regular(suite / "snapshot.json"))
    if (info["binding"] != binding or snapshot.get("signature") != protocol.signature(cfg, binding)
            or snapshot.get("source_binding") != binding):
        raise ValueError("Discovery source/code/configuration changed")
    return suite, cfg, info


def _header(cfg, info, stage, arm_id=None):
    result = dict(schema_version=protocol.SCHEMA_VERSION, stage=stage,
                  signature=protocol.signature(cfg, info["binding"]), source_binding=info["binding"],
                  meta=info["meta"], test_used_for_fitting=False,
                  unknown_images_used_for_gradients=False)
    if arm_id is not None:
        result["arm_id"] = arm_id
    return result


def _finish(directory, receipt, names):
    receipt["artifacts"] = {key: dict(path=name, sha256=protocol.file_hash(_regular(directory / name)))
                            for key, name in names.items()}
    protocol.write_json(directory / "completed.json", receipt)
    return receipt


def _receipt(directory, cfg, info, stage, arm_id=None):
    directory = Path(directory)
    receipt = protocol.read_json(_regular(directory / "completed.json"))
    expected = _header(cfg, info, stage, arm_id)
    if any(receipt.get(k) != v for k, v in expected.items()):
        raise ValueError("Stage receipt/source signature mismatch: " + str(directory))
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("Missing stage artifacts")
    for value in artifacts.values():
        if (not isinstance(value, dict) or set(value) != {"path", "sha256"}
                or Path(value["path"]).name != value["path"]
                or protocol.file_hash(_regular(directory / value["path"])) != value["sha256"]):
            raise ValueError("Changed/escaping stage artifact")
    return receipt


def tensor_hash(value):
    value = value.detach().cpu().contiguous()
    return hashlib.sha256((str(value.dtype) + str(tuple(value.shape))).encode() + value.numpy().tobytes()).hexdigest()


def _text_contract(cache):
    provenance = cache["provenance"]
    keys = ("clip_core_sha256", "clip_initialization", "templates_sha256", "preprocessing")
    return protocol.object_hash(dict(meta=cache["meta"], text={k: tensor_hash(v) for k, v in cache["text"].items()},
                                     core_and_prompts={k: provenance[k] for k in keys}))


def _validate_cache(cache, info, stage):
    if cache.get("meta") != info["meta"] or set(cache.get("text", {})) != {
            "single_leaf", "single_parent", "ensemble_leaf", "ensemble_parent"}:
        raise ValueError("Cached taxonomy or text variants differ")
    if (cache.get("provenance", {}).get("source_binding") != info["binding"]
            or cache["provenance"].get("preprocessing") != info["config"]["data"]):
        raise ValueError("Cached source core/preprocessing binding differs")
    expected = {"train"} if stage == "train" else {prefix + k for k in base.STATUSES
                                                    for prefix in (["val_"] if stage == "development" else ["test_"])}
    if set(cache.get("groups", {})) != expected:
        raise ValueError("Cached split inventory differs from its declared stage")
    seen = set()
    for split, group in cache["groups"].items():
        hashes, rows = group["image_sha256"], group["records"]
        if (len(hashes) != len(set(hashes)) or seen.intersection(hashes)
                or set(hashes) != {r["image_sha256"] for r in rows}):
            raise ValueError("Cached unique images/aliases overlap or disagree")
        seen.update(hashes)
        expected_status = "known" if stage == "train" else split.split("_", 1)[1]
        if any(r["split"] != split or r["status"] != expected_status for r in rows):
            raise ValueError("Cached rows were moved between splits")
        if stage == "train" and any(r["status"] != "known" for r in rows):
            raise ValueError("Only known TRAIN may enter fitting")
        if set(group["features"]) != {"source_fine", "source_parent", "clip"}:
            raise ValueError("Cached representation inventory differs")
        for value in group["features"].values():
            if not isinstance(value, torch.Tensor) or value.ndim != 2 or len(value) != len(hashes) or not bool(torch.isfinite(value).all()):
                raise ValueError("Invalid cached representation")
        for row, index in zip(rows, group["record_feature_indices"]):
            if type(index) is not int or not 0 <= index < len(hashes) or row["image_sha256"] != hashes[index]:
                raise ValueError("Alias-to-feature index mismatch")
        if len(group["record_feature_indices"]) != len(rows):
            raise ValueError("Alias-to-feature mapping length differs")
    if stage == "train" and seen != set(info["training"]["audit"]["train"]["image_hashes"]):
        raise ValueError("TRAIN cache differs from the immutable source images")
    for key, tensor in cache["text"].items():
        n = len(info["meta"]["leaf_names"] if key.endswith("leaf") else info["meta"]["parent_names"])
        if (not isinstance(tensor, torch.Tensor) or tensor.ndim != 2 or len(tensor) != n
                or not bool(torch.isfinite(tensor).all())):
            raise ValueError("Invalid frozen prompt features")


def prepare_cache(suite, stage, device):
    if stage not in ("train", "development", "test"):
        raise ValueError("Unknown feature-cache stage")
    suite, cfg, info = _checked(suite)
    if stage == "test":
        _regular(suite / "dev_selection.json")
    source = load_reference(info["directory"], torch.device(device))
    if stage == "train":
        rows, audit = load_training_rows(source)
        groups, audits = {"train": rows}, {"train": audit}
    else:
        groups, audits = _stage_rows(source, "calibrate" if stage == "development" else "test")
    from .features import collect_cache
    output = _claim(suite / "cache" / stage)
    cache = collect_cache(source, groups, device)
    _validate_cache(cache, info, stage)
    reproduction = None
    if stage == "development":
        # Historical reproduction helper requires this refine-only field.  It is
        # not used in score/candidate comparison and must not enter our cache.
        reproduction = _check_baseline_development(source, {
            split: [dict(row, reconstruction_score=0.0) for row in group["records"]]
            for split, group in cache["groups"].items()
        })
    header = _header(cfg, info, stage)
    payload = dict(header, cache=cache, audit=audits, inference_spec_sha256=_text_contract(cache))
    support._save_torch(output / "features.pth", payload)
    receipt = dict(header, audit=audits, inference_spec_sha256=payload["inference_spec_sha256"],
                   provenance=cache["provenance"], timings=cache["timings"], baseline_reproduction=reproduction)
    if stage == "test":
        receipt["dev_selection_sha256"] = protocol.file_hash(suite / "dev_selection.json")
    return _finish(output, receipt, {"features": "features.pth"})


def _load_cache(suite, stage, cfg, info):
    directory = Path(suite) / "cache" / stage
    receipt = _receipt(directory, cfg, info, stage)
    payload = support._load_torch(directory / "features.pth")
    if any(payload.get(k) != v for k, v in _header(cfg, info, stage).items()):
        raise ValueError("Feature cache header changed")
    cache = payload["cache"]
    _validate_cache(cache, info, stage)
    if (_text_contract(cache) != receipt["inference_spec_sha256"]
            or payload.get("inference_spec_sha256") != receipt["inference_spec_sha256"]
            or payload.get("audit") != receipt["audit"]):
        raise ValueError("Feature cache prompt/audit binding differs")
    return cache, receipt


def _arm(cfg, arm_id):
    matches = [arm for arm in cfg["arms"] if arm["id"] == arm_id]
    if len(matches) != 1:
        raise ValueError("Unknown arm")
    return matches[0]


def _aligned_rows(group):
    index = {r["image_sha256"]: r for r in group["records"]}
    return [index[h] for h in group["image_sha256"]]


def _representations(group, arm, payload=None):
    values = group["features"]
    if arm["representation"] == "source":
        return values["source_fine"], values["source_parent"]
    feature = values["clip"]
    if payload is not None and payload.get("projection") is not None:
        from .models import load_projection
        from .projection_training import transform_projection
        spec = payload["projection"]
        model = load_projection(spec["state"], spec["dimension"], spec["bottleneck"])
        feature = transform_projection(model, feature)
    return feature, feature


def _templates(feature, text, prefix="ensemble"):
    return {level: feature.float() @ text[prefix + "_" + level].float().T for level in ("leaf", "parent")}


def fit_arm(suite, arm_id, device):
    suite, cfg, info = _checked(suite)
    arm = _arm(cfg, arm_id)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    cache, cache_receipt = _load_cache(suite, "train", cfg, info)
    output = _claim(suite / "arms" / arm_id / "training")
    header = _header(cfg, info, "training", arm_id)
    source_id = arm.get("weight_source")
    if source_id:
        source_dir = suite / "arms" / source_id / "training"
        previous = _receipt(source_dir, cfg, info, "training", source_id)
        shutil.copyfile(source_dir / "model.pth", output / "model.pth")
        fit_report = dict(training_execution="reused", optimizer_steps=0,
                          reused_optimizer_steps=previous["optimizer_steps"], source_arm=source_id)
        receipt = dict(header, optimizer_steps=0, weight_source=source_id, fit_report=fit_report,
                       inference_spec_sha256=previous["inference_spec_sha256"],
                       train_cache_sha256=cache_receipt["artifacts"]["features"]["sha256"],
                       reused_training_receipt_sha256=protocol.file_hash(source_dir / "completed.json"))
    else:
        group = cache["groups"]["train"]
        rows = _aligned_rows(group)
        if any(r["status"] != "known" or r["split"] != "train" for r in rows):
            raise ValueError("Fitting is restricted to known TRAIN")
        labels = torch.tensor([r["true_leaf"] for r in rows], dtype=torch.long)
        payload = dict(header, model_spec=arm, text=cache["text"], provenance=cache["provenance"], projection=None,
                       inference_spec_sha256=_text_contract(cache),
                       train_cache_sha256=cache_receipt["artifacts"]["features"]["sha256"])
        fit_report = dict(training_execution="deterministic_fit", optimizer_steps=0)
        if arm["kind"] == "projection":
            from .projection_training import fit_projection
            model, projection_report = fit_projection(group, cache["text"]["ensemble_leaf"], info["meta"],
                                                      device="cpu", seed=cfg["seed"], options=cfg["projection"])
            payload["projection"] = dict(state=support._cpu_state(model), dimension=group["features"]["clip"].shape[1],
                                         bottleneck=cfg["projection"]["bottleneck"])
            fit_report["projection"] = projection_report
        if arm["kind"] in ("geometry", "verifier", "projection"):
            from .geometry import GeometryBank
            fine, parent = _representations(group, arm, payload)
            geometry = GeometryBank.fit(fine, parent, labels, group["image_sha256"], info["meta"], **cfg["geometry"])
            payload["geometry"] = geometry.state_dict()
            fit_report["geometry"] = geometry.fit_report
            if arm["kind"] in ("verifier", "projection"):
                from .verifier import build_episodes, SharedVerifier
                options = dict(cfg["verifier"])
                folds = options.pop("folds")
                episodes = build_episodes(fine, parent, labels, group["image_sha256"], info["meta"],
                    template_scores=_templates(fine, cache["text"]), folds=folds, seed=cfg["seed"], **cfg["geometry"])
                verifier = SharedVerifier.fit(episodes, loss=arm["method"], seed=cfg["seed"], **options)
                payload["verifier"] = verifier.state_dict()
                fit_report["verifier"] = verifier.fit_report
                fit_report["optimizer_steps"] += verifier.fit_report["optimizer_steps"]
                fit_report["training_execution"] = "completed"
        if arm["kind"] == "projection":
            fit_report["optimizer_steps"] += fit_report["projection"]["optimizer_steps"]
        if arm["kind"] == "prompt":
            from .prompt_training import fit_prompt
            source = load_reference(info["directory"], torch.device("cpu"))
            artifact, prompt_report = fit_prompt(source, group, info["meta"], device=device,
                seed=cfg["seed"], loss=arm["method"], options=cfg["prompt"])
            payload["prompt"] = artifact
            fit_report.update(prompt=prompt_report, optimizer_steps=prompt_report["optimizer_steps"], training_execution="completed")
        payload["fit_report"] = fit_report
        support._save_torch(output / "model.pth", payload)
        receipt = dict(header, optimizer_steps=fit_report["optimizer_steps"], weight_source="reference" if arm["kind"] in ("baseline", "geometry", "text") else arm_id,
                       fit_report=fit_report, inference_spec_sha256=payload["inference_spec_sha256"],
                       train_cache_sha256=payload["train_cache_sha256"])
    protocol.write_json(output / "training_report.json", fit_report)
    model_descriptor = dict(path="model.pth", sha256=protocol.file_hash(output / "model.pth"))
    receipt["model"] = model_descriptor
    return _finish(output, receipt, dict(model="model.pth", training_report="training_report.json"))


def _load_model(suite, arm, cfg, info):
    directory = Path(suite) / "arms" / arm["id"] / "training"
    receipt = _receipt(directory, cfg, info, "training", arm["id"])
    payload = support._load_torch(directory / "model.pth")
    expected_id = arm.get("weight_source", arm["id"])
    if any(payload.get(k) != v for k, v in _header(cfg, info, "training", expected_id).items()):
        raise ValueError("Model source/owner signature changed")
    if (payload.get("model_spec") != _arm(cfg, expected_id)
            or payload.get("inference_spec_sha256") != receipt["inference_spec_sha256"]
            or _text_contract(payload) != receipt["inference_spec_sha256"]
            or payload.get("train_cache_sha256") != receipt["train_cache_sha256"]):
        raise ValueError("Model representation/prompt specification changed")
    if arm.get("weight_source"):
        origin = Path(suite) / "arms" / expected_id / "training"
        old = _receipt(origin, cfg, info, "training", expected_id)
        if (receipt.get("reused_training_receipt_sha256") != protocol.file_hash(origin / "completed.json")
                or receipt["model"]["sha256"] != old["model"]["sha256"]):
            raise ValueError("Reused model origin changed")
    return payload, receipt


def score_groups(cache, arm, payload, meta):
    """Pure inference; models and TRAIN statistics are loaded, never refitted."""
    from .geometry import GeometryBank
    from .verifier import SharedVerifier
    geometry = GeometryBank.from_state_dict(payload["geometry"]) if "geometry" in payload else None
    if geometry is not None and geometry.meta != meta:
        raise ValueError("Geometry taxonomy differs from evaluation")
    verifier = SharedVerifier.from_state_dict(payload["verifier"]) if "verifier" in payload else None
    result = {}
    for split, group in cache["groups"].items():
        rows = copy.deepcopy(group["records"])
        if arm["kind"] == "baseline":
            result[split] = rows
            continue
        fine, parent = _representations(group, arm, payload)
        if geometry is not None:
            evidence = geometry.score(fine, parent, group["image_sha256"])
            scored = (verifier.score(evidence, _templates(fine, payload["text"])) if verifier else
                      dict(leaf_scores=evidence["leaf_rmd"], parent_scores=evidence["parent_rmd"]))
        elif arm["kind"] == "prompt":
            text = payload["prompt"]["text_features"]
            scored = {level + "_scores": fine.float() @ text[level].float().T for level in ("leaf", "parent")}
        else:
            prefix = "single" if arm["method"] == "single" else "ensemble"
            scored = {level + "_scores": value for level, value in _templates(fine, payload["text"], prefix).items()}
        leaf, parents = (scored[key].detach().cpu().double() for key in ("leaf_scores", "parent_scores"))
        if not bool(torch.isfinite(leaf).all() and torch.isfinite(parents).all()):
            raise ValueError("Nonfinite discovery inference scores")
        identities = membership.candidate_scores(rows, meta)
        mapping = torch.tensor(meta["leaf_to_parent"])
        for i, row in enumerate(rows):
            index = group["record_feature_indices"][i]
            if arm["candidate"] == "reference":
                p, c = int(identities["parent"][i]), int(identities["leaf"][i])
            else:
                p = int(parents[index].argmax())
                children = (mapping == p).nonzero(as_tuple=True)[0]
                c = int(children[leaf[index, children].argmax()])
                row["global_pred_leaf"] = int(leaf[index].argmax())
            row["discovery"] = dict(leaf_scores=leaf[index].tolist(), parent_scores=parents[index].tolist(),
                                    candidate_leaf=c, candidate_parent=p)
            row["discovery_method"] = arm["method"]
            row["support_evidence_origin"] = "immutable_reference_diagnostic"
            row["log_probs_origin"] = "immutable_reference_diagnostic"
        result[split] = rows
    return result


def _export(output, groups, router, info, arm, crossfit, calibration_report, receipt, timings):
    decode = membership.decode_records if arm["kind"] == "baseline" else calibration.decode_records
    routed = {split: decode(rows, router, info["meta"]) for split, rows in groups.items()}
    predictions = [row for rows in routed.values() for row in rows]
    base.unique_records(predictions)
    seen = set()
    for row in predictions:
        row["evaluation_weight"] = int(row["image_sha256"] not in seen)
        seen.add(row["image_sha256"])
    summary = base.evaluate_records(predictions, info["meta"])
    summary["crossfit_audit"] = crossfit
    report = dict(summary, schema_version="discovery_evaluation_v1", arm_id=arm["id"], stage=receipt["stage"],
                  calibration_status="passed" if summary["targets_passed"] else "best_effort",
                  calibration_gate_is_execution_gate=False, test_allowed_after_failed_gates=True,
                  arm_predictions_replaced_by_reference=False, test_used_for_fitting=False,
                  confirmatory_validation=False, independent_model_level_validation=False,
                  validation_scope="exploratory_discovery_on_reused_benchmark",
                  score_semantics="independent node evidence and routing margins; not calibrated probabilities",
                  candidate_policy=arm["candidate"], calibration_diagnostics=calibration_report)
    metrics = _metrics({split: base.unique_records(rows) for split, rows in routed.items()})
    raw = [row for rows in groups.values() for row in rows]
    protocol.write_records(output / "scores.jsonl", raw)
    protocol.write_records(output / "predictions.jsonl", predictions)
    names = dict(scores="scores.jsonl", predictions="predictions.jsonl", report="report.json",
                 summary="summary.json", metrics="metrics.json", router="router.json", species="per_species.csv",
                 timing="inference_timing.json")
    for key, value in (("report", report), ("summary", summary), ("metrics", metrics), ("router", router), ("timing", timings)):
        protocol.write_json(output / names[key], value)
    _csv(output / names["species"], _species(predictions, info["meta"]))
    receipt.update(summary=summary, targets_passed=summary["targets_passed"],
                   test_allowed_after_failed_gates=True, calibration_gate_is_execution_gate=False)
    return _finish(output, receipt, names)


def calibrate_arm(suite, arm_id, device):
    del device
    suite, cfg, info = _checked(suite)
    arm = _arm(cfg, arm_id)
    cache, cache_receipt = _load_cache(suite, "development", cfg, info)
    payload, training_receipt = _load_model(suite, arm, cfg, info)
    if _text_contract(cache) != training_receipt["inference_spec_sha256"]:
        raise ValueError("DEV and TRAIN used different text/core configurations")
    output = _claim(suite / "arms" / arm_id / "calibration")
    started = time.perf_counter()
    groups = score_groups(cache, arm, payload, info["meta"])
    crossfit, fit_report = None, None
    if arm["kind"] == "baseline":
        router = copy.deepcopy(info["router"])
    else:
        ordered = [groups["val_" + status] for status in base.STATUSES]
        reference = dict(records=[r for group in cache["groups"].values() for r in group["records"]],
                         router=info["router"], calibration_settings=info["config"]["calibration"])
        router, fit_report = calibration.fit_router(*ordered, info["meta"], cfg["calibration"], arm["router"], reference)
        crossfit = calibration.crossfit_audit(*ordered, info["meta"], cfg["calibration"], arm["router"], reference)
    receipt = dict(_header(cfg, info, "calibration", arm_id),
                   model_sha256=training_receipt["model"]["sha256"],
                   training_receipt_sha256=protocol.file_hash(suite / "arms" / arm_id / "training/completed.json"),
                   cache_sha256=cache_receipt["artifacts"]["features"]["sha256"],
                   inference_spec_sha256=training_receipt["inference_spec_sha256"])
    return _export(output, groups, router, info, arm, crossfit, fit_report, receipt,
                   dict(seconds=time.perf_counter()-started, shared_image_inference=cache["timings"]))


def test_arm(suite, arm_id, device):
    del device
    suite, cfg, info = _checked(suite)
    arm = _arm(cfg, arm_id)
    selection = _regular(suite / "dev_selection.json")
    directory = suite / "arms" / arm_id / "calibration"
    calibrated = _receipt(directory, cfg, info, "calibration", arm_id)
    payload, trained = _load_model(suite, arm, cfg, info)
    if (calibrated["model_sha256"] != trained["model"]["sha256"]
            or calibrated["training_receipt_sha256"] != protocol.file_hash(suite / "arms" / arm_id / "training/completed.json")):
        raise ValueError("TEST model differs from its calibration model")
    cache, cache_receipt = _load_cache(suite, "test", cfg, info)
    if (_text_contract(cache) != trained["inference_spec_sha256"]
            or cache_receipt["dev_selection_sha256"] != protocol.file_hash(selection)):
        raise ValueError("TEST prompt or frozen DEV selection binding differs")
    router = protocol.read_json(directory / "router.json")
    output = _claim(suite / "arms" / arm_id / "test")
    started = time.perf_counter()
    groups = score_groups(cache, arm, payload, info["meta"])
    receipt = dict(_header(cfg, info, "test", arm_id), model_sha256=trained["model"]["sha256"],
                   calibration_receipt_sha256=protocol.file_hash(directory / "completed.json"),
                   router_sha256=calibrated["artifacts"]["router"]["sha256"],
                   inference_spec_sha256=trained["inference_spec_sha256"],
                   cache_sha256=cache_receipt["artifacts"]["features"]["sha256"],
                   dev_selection_sha256=protocol.file_hash(selection))
    return _export(output, groups, router, info, arm, calibrated["summary"].get("crossfit_audit"), None, receipt,
                   dict(seconds=time.perf_counter()-started, shared_image_inference=cache["timings"]))
