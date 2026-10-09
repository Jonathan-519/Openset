"""DEV-only C00 membership calibration and adapter-level source-held-out OOF.

The mechanism being compared is trained query adaptation, not the small fixed
operating-point grid. Each unknown-source OOF fold excludes that entire source
from added unknown TRAIN as well as from DEV fitting. The C00 encoder, original
verifiers, and support bank stay fixed: this is not independent end-to-end
validation of the historically selected C00 baseline.
"""
import copy
import hashlib
import json
import math
import traceback

from taxosafe_dcbs.protocol import normalized_name
from taxosafe_support import calibration as base
from taxosafe_support import membership_calibration as membership
from taxosafe_routealign.calibration import _folds, paired_audit

SCHEMA = "evidence_guard_router_v1"
OOF_SCHEMA = "evidence_guard_source_crossfit_v1"
SCORE_SCHEMA = "evidence_guard_fold_scores_v1"
DEV_SPLITS = {"known": "val_known", "intra": "val_intra", "extra": "val_extra"}


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                      allow_nan=False).encode("utf-8")).hexdigest()


def _arm_id(arm):
    return str(arm.get("id", arm.get("name", "")))


def _is_reference(arm):
    return bool(arm.get("reference", False) or _arm_id(arm) == "R00_reference")


def _settings(settings):
    supplied = dict(settings or {})
    seed = supplied.get("seed", 1)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2 ** 31:
        raise ValueError("Invalid calibration seed")
    grid = supplied.get("offset_grid", [-1., 0., 1.])
    if (not isinstance(grid, (list, tuple)) or not grid or len(grid) > 9
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                   or abs(v) > 1. for v in grid)
            or 0. not in grid):
        raise ValueError("Offsets must be a finite preregistered grid within [-1,1], including zero")
    return dict(seed=seed, offset_grid=sorted(set(float(v) for v in grid)))


def reference_rows(rows):
    """Restore original C00 four heads without mutating scored rows or labels."""
    result = []
    for row in rows:
        value = copy.deepcopy(row)
        value["support_evidence"] = copy.deepcopy(row.get("source_support_evidence", row["support_evidence"]))
        if "source_log_probs" in row:
            value["log_probs"] = copy.deepcopy(row["source_log_probs"])
        result.append(value)
    return result


def _unique(rows):
    raw = list(rows)
    # Legacy unique_records does not compare our additional teacher fields.
    seen = {}
    for row in raw:
        key = base._digest(row)
        reference = _hash(row.get("source_support_evidence", row.get("support_evidence")))
        if key in seen and seen[key] != reference:
            raise ValueError("Same image has conflicting original C00 evidence")
        seen[key] = reference
    return sorted(base.unique_records(raw), key=base._digest)


def _dev_rows(rows, meta):
    rows = _unique(rows)
    if any(r.get("split") != DEV_SPLITS.get(r.get("status")) for r in rows):
        raise ValueError("EvidenceGuard fitting accepts only labelled DEV splits; TEST is forbidden")
    groups = [[r for r in rows if r["status"] == status] for status in base.STATUSES]
    if any(not group for group in groups):
        raise ValueError("EvidenceGuard fitting requires all three DEV statuses")
    fitted, _ = base._fit_inputs(*groups, meta)
    membership.candidate_scores(fitted, meta)
    membership.candidate_scores(reference_rows(fitted), meta)
    return sorted(fitted, key=base._digest)


def _state(meta, root_offset, leaf_offset, hashes, reference_router, arm):
    membership._validate_state(reference_router, meta)
    value = dict(schema_version=SCHEMA, decoder="c00_membership_with_query_adaptation",
                 meta=copy.deepcopy(meta), arm_id=_arm_id(arm),
                 candidate_rule=membership.CANDIDATE_RULE,
                 root_offset=float(root_offset), leaf_offset=float(leaf_offset),
                 parent_threshold=float(reference_router["parent_threshold"]) + float(root_offset),
                 leaf_threshold=float(reference_router["leaf_threshold"]) + float(leaf_offset),
                 reference_router=copy.deepcopy(reference_router),
                 root_guard=bool(arm.get("root_guard", False)),
                 fit_image_sha256=sorted(set(hashes)),
                 fit_splits=list(DEV_SPLITS.values()), test_used_for_fitting=False,
                 threshold_unit="original_C00_membership_logit",
                 architecture="frozen_C00_verifiers_and_support_with_trained_query_residuals")
    value["state_sha256"] = _hash(value)
    return value


def _validate(router, meta):
    if (router.get("schema_version") != SCHEMA or router.get("meta") != meta
            or router.get("candidate_rule") != membership.CANDIDATE_RULE
            or router.get("test_used_for_fitting") is not False
            or router.get("fit_splits") != list(DEV_SPLITS.values())):
        raise ValueError("EvidenceGuard router schema/taxonomy/protocol mismatch")
    if router.get("state_sha256") != _hash({k: v for k, v in router.items() if k != "state_sha256"}):
        raise ValueError("EvidenceGuard router identity changed")
    membership._validate_state(router["reference_router"], meta)
    for key in ("root_offset", "leaf_offset", "parent_threshold", "leaf_threshold"):
        if isinstance(router[key], bool) or not math.isfinite(float(router[key])):
            raise ValueError("EvidenceGuard router requires finite thresholds")


def decode(rows, router, meta):
    """Keep C00 ranking and two-gate semantics; ROOT guard never reads truth."""
    _validate(router, meta)
    rows = list(rows)
    plain = dict(schema_version=membership.SCHEMA_VERSION, decoder="membership", meta=meta,
                 candidate_rule=membership.CANDIDATE_RULE,
                 parent_threshold=router["parent_threshold"], leaf_threshold=router["leaf_threshold"])
    result = membership.decode_records(rows, plain, meta)
    original = (membership.decode_records(reference_rows(rows), router["reference_router"], meta)
                if router["root_guard"] else [None] * len(rows))
    for prediction, before in zip(result, original):
        retained = bool(before is not None and before["prediction_type"] == "global_unknown")
        if retained:
            prediction.update(prediction_type="global_unknown", parent=None, leaf=None, output_node=0)
        prediction.update(decoder="evidence_guard", root_guard_applied=retained,
                          root_guard_source="original_C00_membership_prediction" if retained else None)
    return result


def macro(report, key="per_known_leaf"):
    values = [item["correct_rate"] for item in report[key].values() if item["sample_count"]]
    return sum(values) / len(values) if values else 0.


def preservation(report, original):
    return dict(known_count_preserved=report["counts"]["known_correct"] >= original["counts"]["known_correct"],
                known_macro_preserved=macro(report) + 1e-12 >= macro(original),
                extra_count_preserved=report["counts"]["extra_correct"] >= original["counts"]["extra_correct"])


def fit(rows, meta, settings, reference_router, arm):
    rows = _dev_rows(rows, meta)
    settings = _settings(settings)
    baseline = membership.decode_records(reference_rows(rows), reference_router, meta)
    original = base.evaluate_records(baseline, meta)
    root_grid = settings["offset_grid"] if arm.get("adapt_parent", False) and not _is_reference(arm) else [0.]
    leaf_grid = settings["offset_grid"] if arm.get("adapt_fine", False) and not _is_reference(arm) else [0.]
    best, candidates, zero_report = None, [], None
    for root in root_grid:
        for leaf in leaf_grid:
            router = _state(meta, root, leaf, [base._digest(row) for row in rows], reference_router, arm)
            predictions = decode(rows, router, meta)
            report = base.evaluate_records(predictions, meta)
            checks = preservation(report, original)
            preserved = all(checks.values())
            metrics = [report["metrics"][name] or 0. for name in base.TARGETS]
            ratios = [value / base.TARGETS[name]["threshold"] for name, value in zip(base.TARGETS, metrics)]
            rank = (int(report["targets_passed"] and preserved), int(preserved),
                    int(report["checks"]["known_end_to_end_leaf_accuracy"]), min(ratios), sum(metrics),
                    -abs(root) - abs(leaf), -root, -leaf)
            candidates.append(dict(root_offset=root, leaf_offset=leaf, metrics=report["metrics"],
                counts=report["counts"], targets_passed=report["targets_passed"], **checks))
            if root == 0. and leaf == 0.:
                zero_report = report
            if best is None or rank > best[0]:
                best = rank, router, report, predictions, checks
    _, router, report, predictions, checks = best
    diagnostics = dict(operating_point_grid=candidates, zero_offset_report=zero_report,
                       selected_report=report, reference_report=original, **checks,
                       grid_selected_on="DEV_only", test_used_for_fitting=False,
                       root_offset_locked=not bool(arm.get("adapt_parent", False)) or _is_reference(arm),
                       leaf_offset_locked=not bool(arm.get("adapt_fine", False)) or _is_reference(arm),
                       paired_to_C00=paired_audit(baseline, predictions, meta),
                       selection_rule="four gates and C00 preservation, preservation, known gate, worst target ratio, metric sum, smallest offset",
                       optimization_scope="finite operating-point offsets after query-adapter training")
    return router, diagnostics


def reference_folds(rows, meta, settings, reference_settings):
    rows = _dev_rows(reference_rows(rows), meta)
    settings = _settings(settings)
    by_hash = {base._digest(row): row for row in rows}
    result = []
    for raw_fold in _folds(rows, settings):
        fold = copy.deepcopy(raw_fold)
        if "status" in fold:
            fold["held_status"] = fold.pop("status")
        fitted = [by_hash[h] for h in fold["fit_image_sha256"]]
        held = [by_hash[h] for h in fold["held_image_sha256"]]
        groups = [[row for row in fitted if row["status"] == status] for status in base.STATUSES]
        if not held or any(not group for group in groups):
            result.append(dict(fold, status="not_evaluable", reason="empty held fold or incomplete fit statuses"))
            continue
        router = membership.calibrate(*groups, meta, dict(reference_settings, source_loo=False))
        if set(router["fit_image_sha256"]) & set(fold["held_image_sha256"]):
            raise ValueError("Reference fold includes held images")
        result.append(dict(fold, status="completed", reference_router=router,
                           reference_predictions=membership.decode_records(held, router, meta)))
    return result


def _cache_rows(cache):
    return _unique(row for group in cache["groups"].values() for row in group["records"])


def _fold_signature(fold):
    return _hash({key: fold.get(key) for key in ("fold_id", "kind", "source", "fit_image_sha256", "held_image_sha256", "reference_router")})


def _payload_hash(payload):
    from taxosafe_discovery.features import tensor_hash
    import torch
    def convert(value):
        if torch.is_tensor(value):
            return dict(tensor_sha256=tensor_hash(value), shape=list(value.shape), dtype=str(value.dtype))
        if isinstance(value, dict):
            return {str(key): convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        return value
    return _hash(convert(payload))


def _report(payload):
    report = payload.get("training_report", payload.get("report"))
    if not isinstance(report, dict):
        raise ValueError("Fold training must return an auditable training_report")
    return report


def _audit_train(report, train_rows, dev_rows, excluded):
    if "used_image_sha256" not in report or "used_sources" not in report:
        raise ValueError("Fold training report is missing actual used TRAIN hashes/sources")
    used_hashes = set(report["used_image_sha256"])
    used_sources = {normalized_name(str(source)) for source in report["used_sources"]}
    excluded = {normalized_name(str(source)) for source in excluded}
    by_hash = {base._digest(row): row for row in train_rows}
    if not used_hashes or not used_hashes <= set(by_hash):
        raise ValueError("Fold report has no valid TRAIN identities or includes non-TRAIN identities")
    actual_sources = {normalized_name(str(by_hash[h].get("source") or "unspecified")) for h in used_hashes
                      if by_hash[h]["status"] != "known"}
    forbidden = {base._digest(row) for row in train_rows
                 if row["status"] != "known" and normalized_name(str(row.get("source") or "unspecified")) in excluded}
    eligible = set(report.get("fit_image_sha256", []))
    if not eligible or not used_hashes <= eligible or not eligible <= set(by_hash):
        raise ValueError("Fold eligible TRAIN identities are missing or inconsistent")
    if eligible & forbidden or used_hashes & forbidden or actual_sources & excluded or used_sources & excluded:
        raise ValueError("Held unknown source was used by fold TRAIN")
    if used_hashes & {base._digest(row) for row in dev_rows}:
        raise ValueError("DEV image was used by fold TRAIN")
    if actual_sources != used_sources:
        raise ValueError("Fold TRAIN source report disagrees with actual used images")
    if {normalized_name(str(source)) for source in report.get("excluded_sources", [])} != excluded:
        raise ValueError("Fold TRAIN excluded-source declaration differs from the required holdout")
    return dict(used_image_sha256=sorted(used_hashes), used_sources=sorted(used_sources),
                excluded_train_sources=sorted(excluded), excluded_train_image_sha256=sorted(forbidden),
                excluded_train_image_count=len(forbidden), train_source_exclusion_verified=True,
                dev_images_excluded_from_train=True)


def source_crossfit(traincache, devcache, meta, arm, cfg, folds, device="cpu", reused=None):
    """Refit adapters per fold; never substitute the full-data model on failure."""
    from . import training
    settings = _settings(cfg["calibration"])
    dev_rows = _dev_rows(_cache_rows(devcache), meta)
    train_rows = _cache_rows(traincache)
    dev_hashes = {base._digest(row) for row in dev_rows}
    if any(not (row.get("split") == "train" and row.get("status") == "known"
                or row.get("split") == "train_intra" and row.get("status") == "intra"
                or row.get("split") == "oe_train" and row.get("status") == "extra") for row in train_rows):
        raise ValueError("Source crossfit accepts only declared known/OE TRAIN splits")
    if dev_hashes & {base._digest(row) for row in train_rows}:
        raise ValueError("TRAIN/DEV image overlap in source crossfit")
    binding = _hash(dict(meta=meta, train_hashes=sorted(base._digest(row) for row in train_rows),
                         dev_hashes=sorted(dev_hashes), train_source_binding=traincache.get("source_binding"),
                         dev_source_binding=devcache.get("source_binding"), training_settings=cfg.get("training"),
                         geometry_settings=cfg.get("geometry"), calibration_settings=settings))
    root_guard = bool(arm.get("root_guard", False))
    reused_by_fold = {}
    reuse_error = None
    if root_guard:
        if (not isinstance(reused, dict) or reused.get("schema_version") != SCORE_SCHEMA
                or reused.get("arm_id") != "R04_OE_both" or reused.get("cache_binding") != binding):
            reuse_error = "ROOT guard OOF requires matching R04 adapter-refitted fold scores"
        else:
            reused_by_fold = {item["fold_id"]: item for item in reused["folds"]}
    scores = dict(schema_version=SCORE_SCHEMA, arm_id=_arm_id(arm), cache_binding=binding, folds=[])
    before, after, audits = [], [], []
    for index, fold in enumerate(folds):
        audit = {key: copy.deepcopy(value) for key, value in fold.items()
                 if key not in ("reference_router", "reference_predictions")}
        seed = int(settings["seed"]) + index + 1
        excluded = [fold["source"]] if fold.get("kind") == "unknown_source" else []
        signature = _fold_signature(fold)
        entry = dict(fold_id=fold["fold_id"], fold_signature=signature,
                     source_arm_id="R04_OE_both" if root_guard else _arm_id(arm), seed=seed,
                     adapter_refitted=not _is_reference(arm), status="not_evaluable")
        audit.update(seed=seed, excluded_train_sources=excluded,
                     adapter_refitted=not _is_reference(arm),
                     full_data_adapter_used=False, held_source_used_for_loss_centers=False)
        if fold.get("status") != "completed":
            audits.append(audit)
            scores["folds"].append(entry)
            continue
        try:
            fit_hashes, held_hashes = set(fold["fit_image_sha256"]), set(fold["held_image_sha256"])
            if fit_hashes & held_hashes or fit_hashes | held_hashes != dev_hashes:
                raise ValueError("OOF fold must partition all DEV images")
            if set(fold["reference_router"]["fit_image_sha256"]) != fit_hashes:
                raise ValueError("Fold reference router was not fitted only on its fit partition")
            if _is_reference(arm):
                predictions = copy.deepcopy(fold["reference_predictions"])
                entry.update(status="completed", adapter_refitted=False, rows=reference_rows(dev_rows),
                             training_report=dict(steps_completed=0, reference_only=True))
            else:
                if root_guard:
                    if reuse_error:
                        raise ValueError(reuse_error)
                    old = reused_by_fold.get(fold["fold_id"])
                    if not old or old.get("status") != "completed" or old.get("fold_signature") != signature:
                        raise ValueError("R04 fold score missing, failed, or partition changed")
                    if not old.get("adapter_refitted") or not old.get("model_state_sha256"):
                        raise ValueError("R04 fold score lacks adapter refit identity")
                    scored = copy.deepcopy(old["rows"])
                    report = copy.deepcopy(old["training_report"])
                    model_hash = old["model_state_sha256"]
                    entry["reused_adapter_from"] = "R04_OE_both:" + fold["fold_id"]
                else:
                    fold_cfg = copy.deepcopy(cfg)
                    fold_cfg["seed"] = seed
                    payload, report = training.fit(traincache, meta, arm, fold_cfg, device=device,
                                                   exclude_sources=excluded, reference_router=fold["reference_router"])
                    if not isinstance(report, dict):
                        raise ValueError("Fold training returned no auditable report")
                    model_hash = _payload_hash(payload["state"])
                    scored_groups = training.score(devcache, payload, device=device)
                    scored = [row for group in scored_groups.values() for row in group]
                train_audit = _audit_train(report, train_rows, dev_rows, excluded)
                if report.get("training_seed") != seed:
                    raise ValueError("Fold training seed differs from declared seed")
                for branch in ("parent", "leaf"):
                    if report.get("loss_reference_" + branch + "_threshold") != float(fold["reference_router"][branch + "_threshold"]):
                        raise ValueError("Fold loss center differs from fit-only C00 router")
                scored = _dev_rows(scored, meta)
                by_hash = {base._digest(row): row for row in scored}
                if set(by_hash) != dev_hashes:
                    raise ValueError("Fold model scores do not cover exactly DEV")
                original_by_hash = {base._digest(row): row for row in dev_rows}
                for key, row in by_hash.items():
                    if any(row.get(field) != original_by_hash[key].get(field)
                           for field in ("split", "status", "source", "species", "true_leaf", "true_parent")):
                        raise ValueError("Fold scoring changed DEV annotations")
                    if ("source_support_evidence" not in row or
                            _hash(row["source_support_evidence"]) != _hash(original_by_hash[key]["support_evidence"])):
                        raise ValueError("Fold scoring did not preserve original C00 evidence")
                fit_rows = [by_hash[key] for key in sorted(fit_hashes)]
                held = [by_hash[key] for key in sorted(held_hashes)]
                router, _ = fit(fit_rows, meta, settings, fold["reference_router"], arm)
                if set(router["fit_image_sha256"]) & held_hashes:
                    raise ValueError("Held images entered fold offset calibration")
                predictions = decode(held, router, meta)
                entry.update(status="completed", rows=scored, training_report=copy.deepcopy(report),
                             model_state_sha256=model_hash, **train_audit)
                audit.update(router=router, model_state_sha256=model_hash,
                             training_report=copy.deepcopy(report), **train_audit)
                if root_guard:
                    audit["reused_adapter_from"] = entry["reused_adapter_from"]
            if {base._digest(row) for row in predictions} != held_hashes:
                raise ValueError("Held predictions do not match the declared fold")
            if {base._digest(row) for row in fold["reference_predictions"]} != held_hashes:
                raise ValueError("Reference held predictions do not match the declared fold")
            before.extend(copy.deepcopy(fold["reference_predictions"]))
            after.extend(predictions)
            audit["status"] = "completed"
        except Exception as error:
            # OOF is a quality gate. Technical failure is explicit and false,
            # without preventing independent full-DEV calibration and TEST.
            entry.update(status="not_evaluable", error_type=type(error).__name__, error=str(error))
            audit.update(status="not_evaluable", error_type=type(error).__name__, error=str(error),
                         traceback=traceback.format_exc())
        audits.append(audit)
        scores["folds"].append(entry)
    complete = (len(after) == len(dev_rows) and len({base._digest(row) for row in after}) == len(dev_rows)
                and all(item["status"] == "completed" for item in audits))
    report = base.evaluate_records(after, meta) if after else None
    original = base.evaluate_records(before, meta) if before else None
    checks = preservation(report, original) if report and original else dict(
        known_count_preserved=False, known_macro_preserved=False, extra_count_preserved=False)
    result = dict(schema_version=OOF_SCHEMA, complete=complete,
                  passed=bool(complete and report["targets_passed"] and all(checks.values())),
                  report=report, reference_report=original, folds=audits,
                  reference_predictions=before, predictions=after, **checks,
                  known_count_and_macro_preserved=checks["known_count_preserved"] and checks["known_macro_preserved"],
                  held_image_count=len(after), expected_held_image_count=len(dev_rows),
                  paired_to_C00=paired_audit(before, after, meta) if complete else None,
                  adapter_refitted_in_folds=not _is_reference(arm),
                  reference_encoder_verifiers_and_support_fixed=True,
                  validation_scope="conditional_on_C00_adapter_level_source_heldout_OOF",
                  independent_full_pipeline_validation=False, confirmatory_validation=False,
                  held_data_used_for_offsets=False, held_source_used_for_adapter_training=False,
                  test_used_for_fitting=False, output_used_for_threshold_selection=False,
                  reference_folds=copy.deepcopy(folds) if _is_reference(arm) else None)
    return result, scores
