"""Audited comparison of declared arms, including explicit unavailable results.

DEV freezes the recommendation before any TEST cache exists. TEST rankings
remain descriptive; no missing/failed arm receives invented zero metrics.
"""
from pathlib import Path
from datetime import datetime, timezone

from taxosafe_support import calibration as base
from taxosafe_routealign.evaluation import _csv, _records, _regular
from taxosafe_routealign.reporting import _paired, _source_macro
from . import protocol
from taxosafe_discovery.reporting import _rank, _entry as _base_entry, _comparisons as _base_comparisons
import copy

SCHEMA_VERSION = "taxosafe_morphology_comparison_v1"
VALIDATION_SCOPE = "exploratory_d05_morphology_on_reused_development_and_test"
RANKING = ("DEV qualified first (four gates, known correct count at least baseline, crossfit passed); "
           "then four gates passed, known correct count preserved, mean four metrics descending "
           "(undefined last), known correct count descending, arm ID ascending")

# Descriptions are fixed with the predeclared matrix, not inferred from TEST.
MECHANISMS = {
    "C00_reference": ("exact reference replay", "original evidence", "none; inherited weights"),
    "C01_d05": ("exact D05 replay", "original D05 evidence", "none; inherited weights"),
    "C02_d05_staged": ("staged D05 control", "fixed D05 confidence", "none; threshold calibration only"),
    "R01_spatial_parent": ("signed parent spatial verification", "complete reference spatial grids + constrained real-individual matching", "balanced parent BCE + active-known local CE"),
    "L01_spatial_leaf": ("signed leaf spatial verification; C02 root frozen", "complete reference spatial grids + constrained real-individual matching", "balanced leaf BCE + active-known local CE"),
}


def _matrix_row(arm):
    mechanism, inputs, loss = MECHANISMS[arm["id"]]
    return dict(arm_id=arm["id"], kind=arm["kind"], mode=arm["mode"], mechanism=mechanism,
                inputs=inputs, loss=loss, policy=arm["policy"],
                declared_weight_source=arm.get("weight_source", "reference" if arm["kind"] == "reference" else
                                              "independent_known_TRAIN_spatial_optimization" if arm["kind"] == "fit" else "original_D05"),
                candidate_policy=arm.get("candidate_policy", "source"))


def compact_crossfit(full):
    """Persist one full DEV audit; carry only decision fields into each report."""
    if full is None:
        return None
    if not isinstance(full, dict):
        raise ValueError("Morphology crossfit audit must be a mapping")
    keys = ("passed", "recovery_passed", "reason", "status", "validation_scope", "confirmatory_validation",
            "independent_model_level_validation", "development_reused_for_method_design", "test_used_for_fitting")
    result = {key: copy.deepcopy(full[key]) for key in keys if key in full}
    folds = full.get("folds")
    result["fold_count"] = len(folds) if isinstance(folds, list) else full.get("fold_count")
    if isinstance(full.get("known_recovery"), dict):
        result["known_recovery"] = {key: copy.deepcopy(full["known_recovery"][key])
            for key in ("passed", "checks", "counts") if key in full["known_recovery"]}
    return result


def _unique_morphology_records(records):
    seen = {}
    for row in records:
        digest = base._digest(row)
        if digest in seen and row.get("morphology") != seen[digest].get("morphology"):
            raise ValueError("Same image has conflicting morphology evidence")
        seen[digest] = row
    return base.unique_records(records)


def root_stage_summary(records, meta):
    """Observed terminal outcomes, counted once per image, independent of fits."""
    rows = _unique_morphology_records(records)
    base.evaluate_records(rows, meta)
    result = {name + suffix: 0 for name in ("known", "near", "extra")
              for suffix in ("_count", "_root_rejected", "_passed_root")}
    for key in ("known_leaf_correct_after_root", "known_leaf_wrong_after_root",
                "unknown_leaf_false_accept_after_root", "known_parent_abstentions_after_root",
                "near_parent_correct_after_root"):
        result[key] = 0
    for row in rows:
        status = "near" if row["status"] == "intra" else row["status"]
        result[status + "_count"] += 1
        root = row["prediction_type"] == "global_unknown"
        result[status + ("_root_rejected" if root else "_passed_root")] += 1
        if root:
            continue
        leaf = row["prediction_type"] == "known"
        if status == "known":
            if leaf:
                result["known_leaf_correct_after_root" if row["leaf"] == row["true_leaf"] else
                       "known_leaf_wrong_after_root"] += 1
            else:
                result["known_parent_abstentions_after_root"] += 1
        elif leaf:
            result["unknown_leaf_false_accept_after_root"] += 1
        elif status == "near" and row["parent"] == row["true_parent"]:
            result["near_parent_correct_after_root"] += 1
    result["rates"] = {status + "_root_rejection": result[status + "_root_rejected"] / result[status + "_count"]
                       if result[status + "_count"] else None for status in ("known", "near", "extra")}
    passed = result["known_passed_root"]
    result["rates"]["known_leaf_accuracy_given_root_pass"] = result["known_leaf_correct_after_root"] / passed if passed else None
    result["unit"] = "unique_image_sha256"
    return result


def _timing(root, arm_id, stage):
    marker = root / "arms" / arm_id / stage / "stage_binding.json"
    return protocol.read_json(_regular(marker)).get("elapsed_seconds") if marker.is_file() else None


def _suite(directory):
    root = Path(directory)
    cfg = protocol.validate_config(protocol.read_json(_regular(root / "config.json")))
    snapshot = protocol.read_json(_regular(root / "snapshot.json"))
    ids = [arm["id"] for arm in cfg["arms"]]
    if (not ids or len(ids) != len(set(ids)) or snapshot.get("schema_version") != protocol.SCHEMA_VERSION
            or snapshot.get("config_sha256") != protocol.object_hash(cfg) or snapshot.get("arm_ids") != ids
            or protocol.read_json(_regular(root / "source_binding.json")) != snapshot.get("source_binding")):
        raise ValueError("Morphology configuration/snapshot identity changed")
    return root, cfg, snapshot, ids


def _failure(root, arm_id):
    path = root / "arms" / arm_id / "failure.json"
    if not path.is_file():
        return None
    value = protocol.read_json(_regular(path))
    if value.get("arm_id") != arm_id or value.get("technical_failure") is not True or not value.get("error"):
        raise ValueError("Invalid technical failure record: " + arm_id)
    return value


def _stage(root, arm_id, stage, snapshot):
    from .runner import _verify_stage
    receipt = _verify_stage(root, arm_id, stage, snapshot)
    directory = root / "arms" / arm_id / stage
    artifacts = receipt["artifacts"]
    if not {"predictions", "report", "router"} <= set(artifacts):
        raise ValueError("Evaluation receipt lacks predictions/report/router artifacts")
    rows = _unique_morphology_records(_records(directory / artifacts["predictions"]["path"]))
    calculated = base.evaluate_records(rows, receipt["meta"])
    if any(calculated["counts"][status] == 0 for status in ("known", "intra", "extra")):
        raise ValueError("Evaluation must contain every locked status group")
    report = protocol.read_json(_regular(directory / artifacts["report"]["path"]))
    if not isinstance(report, dict) or report.get("schema_version") != "morphology_evaluation_v1":
        raise ValueError("Invalid morphology evaluation report schema: " + arm_id)
    for key in ("counts", "metrics", "checks", "targets_passed"):
        if report.get(key) != calculated[key] or receipt.get("summary", {}).get(key) != calculated[key]:
            raise ValueError("Saved metrics do not reproduce predictions: " + arm_id)
    if report.get("crossfit_audit") != receipt["summary"].get("crossfit_audit"):
        raise ValueError("Crossfit report/receipt mismatch: " + arm_id)
    root_summary = root_stage_summary(rows, receipt["meta"])
    if report.get("root_stage_outcomes") != root_summary or receipt["summary"].get("root_stage_outcomes") != root_summary:
        raise ValueError("Saved root-stage outcomes do not reproduce predictions: " + arm_id)
    compact = report.get("crossfit_audit")
    digest = report.get("crossfit_audit_sha256")
    if digest != receipt["summary"].get("crossfit_audit_sha256"):
        raise ValueError("Crossfit summary artifact binding differs: " + arm_id)
    if compact is None and digest is None:
        if "crossfit" in artifacts:
            raise ValueError("Crossfit artifact lacks its digest")
    elif stage == "calibration":
        descriptor = artifacts.get("crossfit")
        if not descriptor or digest != descriptor["sha256"] or digest != protocol.file_hash(_regular(directory / descriptor["path"])):
            raise ValueError("Full DEV crossfit artifact changed: " + arm_id)
        full = protocol.read_json(directory / descriptor["path"])
        if compact != compact_crossfit(full):
            raise ValueError("Crossfit compact decision differs from full DEV audit: " + arm_id)
    else:
        if "crossfit" in artifacts:
            raise ValueError("TEST must reference the saved DEV crossfit audit without copying it")
        calibration = _verify_stage(root, arm_id, "calibration", snapshot)
        if (digest != calibration["artifacts"].get("crossfit", {}).get("sha256")
                or receipt.get("calibration_crossfit_audit_sha256") != digest
                or compact != calibration["summary"].get("crossfit_audit")):
            raise ValueError("TEST did not inherit its frozen DEV crossfit audit: " + arm_id)
    return receipt, report, rows


def _entry(arm_id, value, reference, baseline_id, phase, d05=None):
    result = _base_entry(arm_id, value, reference, baseline_id, phase)
    _, report, rows = value
    crossfit = report.get("crossfit_audit") or {}
    observed = False
    checks, audit = {}, None
    if d05 is not None:
        if value[0]["meta"] != d05[0]["meta"]:
            raise ValueError("Morphology comparison taxonomy differs")
        audit = _paired(d05[2], rows, value[0]["meta"])
        current, old = report["counts"], d05[1]["counts"]
        checks = dict(known_strictly_above_90=current["known_correct"] * 10 > current["known"] * 9,
            known_correct_improved=current["known_correct"] > old["known_correct"],
            near_correct_not_decreased=current["intra_correct"] >= old["intra_correct"],
            extra_correct_not_decreased=current["extra_correct"] >= old["extra_correct"],
            precision_strictly_above_90=current["leaf_outputs"] > 0 and current["known_correct"] * 10 > current["leaf_outputs"] * 9)
        observed = all(checks.values())
    oof = crossfit.get("recovery_passed", crossfit.get("known_recovery", {}).get("passed")) is True
    result.update(research_recovery_observed=observed, research_recovery_checks=checks,
        research_recovery_qualified=phase == "development" and observed and oof,
        d05_known_lost_correct_count=None if audit is None else audit["known_lost_correct_count"],
        d05_known_gained_correct_count=None if audit is None else audit["known_gained_correct_count"],
        recovery_is_deployment_qualification=False)
    details = report.get("morphology_policy") or (report.get("calibration_diagnostics") or {}).get("morphology_policy") or {}
    result["morphology_policy"] = details
    result["dev_crossfit_passed" if phase == "development" else "inherited_dev_crossfit_passed"] = result.pop("crossfit_audit_passed")
    result["dev_research_recovery_oof_passed" if phase == "development" else "inherited_dev_research_recovery_oof_passed"] = oof
    result["root_stage_outcomes"] = root_stage_summary(rows, value[0]["meta"])
    result["crossfit_audit_sha256"] = report.get("crossfit_audit_sha256")
    return result


def _development(root, snapshot, ids):
    loaded, failures, inventory = {}, {}, {}
    for arm_id in ids:
        failure = _failure(root, arm_id)
        if failure is not None and failure["stage"] != "test":
            failures[arm_id] = failure
            inventory[arm_id] = {"technical_failure_sha256": protocol.file_hash(root / "arms" / arm_id / "failure.json")}
            continue
        path = root / "arms" / arm_id / "calibration/completed.json"
        if not path.is_file():
            raise ValueError("Every declared arm needs completed calibration or explicit failure: " + arm_id)
        loaded[arm_id] = _stage(root, arm_id, "calibration", snapshot)
        inventory[arm_id] = {"calibration_receipt_sha256": protocol.file_hash(path)}
    reference = loaded.get(ids[0])
    if reference is not None:
        for receipt, report, rows in loaded.values():
            if receipt["meta"] != reference[0]["meta"]:
                raise ValueError("Comparison taxonomy differs across arms")
    ranking = _rank([_entry(arm_id, value, reference, ids[0], "development", loaded.get("C01_d05")) for arm_id, value in loaded.items()])
    roots = _root_invariants({arm_id:_morphology_fingerprints(root, arm_id, "development", value)
                             for arm_id,value in loaded.items()})
    qualified = [row for row in ranking if row["qualified"]]
    chosen = qualified[0]["arm_id"] if qualified else None
    research = sorted([r for r in ranking if r["research_recovery_qualified"]],
        key=lambda r: (-r["counts"]["known_correct"], -(r["mean_four_metrics"] or 0.), r["arm_id"]))
    selection = {"schema_version": SCHEMA_VERSION, "phase": "development", "suite_signature": snapshot["signature"],
        "config_sha256": snapshot["config_sha256"], "snapshot_sha256": protocol.file_hash(root / "snapshot.json"),
        "inventory": inventory, "ranking_rule": RANKING, "selection_uses_test": False, "test_predictions_read": False,
        "validation_scope": VALIDATION_SCOPE, "confirmatory_validation": False, "development_reused_for_method_design": True,
        "parent_chosen_after_prior_test_review": True,
        "root_state_comparison": roots,
        "research_recovery_arm_id": research[0]["arm_id"] if research else None,
        "research_recovery_ranking": research,
        "research_recovery_ranking_rule": "qualified research recovery only; known correct count descending, mean four metrics descending, arm ID ascending",
        "research_recovery_is_deployment_recommendation": False,
        "qualified_candidate_arm_id": chosen, "recommendation_arm_id": chosen or ids[0],
        "recommendation_status": "qualified_on_development" if chosen else "retain_reference_no_qualified_candidate" if reference is not None else "retain_original_reference_comparison_unavailable",
        "best_exploratory_dev_arm": ranking[0] if ranking else None,
        "exploratory_development_ranking": ranking, "technical_failures": failures}
    return selection, loaded


def freeze_dev_selection(root_dir):
    root, cfg, snapshot, ids = _suite(root_dir)
    path = root / "dev_selection.json"
    if not path.exists() and ((root / "cache/test").exists() or any(root.glob("arms/*/test"))):
        raise ValueError("DEV selection must be frozen before creating any TEST cache or stage")
    selection, loaded = _development(root, snapshot, ids)
    if path.exists():
        existing = protocol.read_json(_regular(path))
        stamp = existing.get("created_at_utc")
        if not isinstance(stamp, str) or datetime.fromisoformat(stamp).tzinfo is None:
            raise ValueError("DEV freeze requires a timezone-aware creation timestamp")
        selection["created_at_utc"] = stamp
        if existing != selection:
            raise ValueError("Immutable DEV recommendation/calibration inventory changed")
    else:
        selection["created_at_utc"] = datetime.now(timezone.utc).isoformat()
        protocol.write_json(path, selection)
    return selection


def _morphology_fingerprints(root, arm_id, phase, value):
    receipt, report, rows = value
    ordered = sorted(rows, key=base._digest)
    has_morphology = all(isinstance(row.get("morphology"), dict) for row in ordered)
    result = {}
    for label, fields in (("root_score", ("root_score",)), ("leaf_score", ("leaf_scores",)),
                          ("parent_score", ("parent_scores",)), ("anchor", ("anchor_parent", "anchor_leaf")),
                          ("morphology_candidate", ("candidate_parent", "candidate_leaf"))):
        result[label + "_sha256"] = protocol.object_hash([
            dict(image_sha256=base._digest(row), **{key:row["morphology"][key] for key in fields})
            for row in ordered]) if has_morphology else None
    stage = "calibration" if phase == "development" else "test"
    router = protocol.read_json(_regular(root / "arms" / arm_id / stage / receipt["artifacts"]["router"]["path"]))
    result["root_state_sha256"] = router.get("root_state_sha256")
    result["root_pass_sha256"] = protocol.object_hash([
        [base._digest(row), row["prediction_type"] != "global_unknown"] for row in ordered])
    result["candidate_policy"] = sorted({row["morphology"]["candidate_policy"] for row in ordered}) if has_morphology else ["original"]
    result["score_semantics"] = "separate scalar root confidence, parent ranking evidence and conditional or inherited leaf confidence"
    return result


def _candidate_comparison(before, after):
    left = {base._digest(row):row for row in _unique_morphology_records(before)}
    right = {base._digest(row):row for row in _unique_morphology_records(after)}
    if set(left) != set(right):
        raise ValueError("Candidate comparison image identities differ")
    result = dict(unique_image_count=len(left), candidate_parent_changed=0, candidate_leaf_changed=0,
        candidate_path_changed=0, candidate_changed_terminal_changed=0, candidate_changed_terminal_unchanged=0,
        same_candidate_terminal_changed=0, confidence_vectors_changed=0,
        known_candidate_leaf_repaired=0, known_candidate_leaf_damaged=0)
    for digest, old in left.items():
        row = right[digest]
        parent = old.get("candidate_parent") != row.get("candidate_parent")
        leaf = old.get("candidate_leaf") != row.get("candidate_leaf")
        changed = parent or leaf
        terminal = any(old.get(key) != row.get(key) for key in ("prediction_type", "parent", "leaf"))
        result["candidate_parent_changed"] += int(parent)
        result["candidate_leaf_changed"] += int(leaf)
        result["candidate_path_changed"] += int(changed)
        if changed:
            result["candidate_changed_terminal_changed" if terminal else "candidate_changed_terminal_unchanged"] += 1
        elif terminal:
            result["same_candidate_terminal_changed"] += 1
        def confidence(value):
            evidence = value.get("morphology", value.get("discovery", {}))
            return {key:evidence.get(key) for key in ("root_score", "parent_scores", "leaf_scores")}
        result["confidence_vectors_changed"] += int(confidence(old) != confidence(row))
        if row["status"] == "known":
            was = old.get("candidate_leaf") == row["true_leaf"]
            now = row.get("candidate_leaf") == row["true_leaf"]
            result["known_candidate_leaf_repaired"] += int(now and not was)
            result["known_candidate_leaf_damaged"] += int(was and not now)
    result["interpretation"] = "Candidate changes and confidence/threshold changes are separate observations; terminal differences alone do not identify a causal mechanism."
    return result


def _root_invariants(fingerprints):
    ids = [key for key in ("C02_d05_staged", "L01_spatial_leaf") if key in fingerprints]
    def identical(key):
        if len(ids) != 2:
            return None
        values = [fingerprints[arm][key] for arm in ids]
        return all(value is not None for value in values) and len(set(values)) == 1
    checks = {key:identical(key) for key in ("root_score_sha256", "root_state_sha256", "root_pass_sha256")}
    if any(value is False for value in checks.values()):
        raise ValueError("L01 changed C02 root scores, state or admitted query identities")
    root_leaf_exact = None
    if all(key in fingerprints for key in ("C02_d05_staged", "R01_spatial_parent")):
        root_leaf_exact = fingerprints["C02_d05_staged"]["leaf_score_sha256"] == fingerprints["R01_spatial_parent"]["leaf_score_sha256"]
        if not root_leaf_exact:
            raise ValueError("R01 changed the frozen D05 leaf scores")
    return dict(shared_root_arms=ids, checks=checks, R01_leaf_scores_unchanged=root_leaf_exact,
                validation="verified" if len(ids)==2 else "not_evaluable")


def _comparisons(root, phase, loaded, baseline_id):
    result, species, sources = _base_comparisons(root, phase, loaded, baseline_id)
    species = [dict(row, comparison_baseline=baseline_id) for row in species]
    result["paired_to_d05"] = {}
    if "C01_d05" in loaded:
        before = loaded["C01_d05"]
        for arm_id, value in loaded.items():
            audit = _paired(before[2], value[2], value[0]["meta"])
            result["paired_to_d05"][arm_id] = audit
            species.extend(dict(phase=phase, arm_id=arm_id, comparison_baseline="C01_d05",
                **{key: item for key, item in row.items() if not key.endswith("sha256")}) for row in audit["per_species"])
    for arm_id, value in loaded.items():
        result["fingerprints"][arm_id].update(_morphology_fingerprints(root, arm_id, phase, value))
    result["root_stage_outcomes"] = {arm_id:root_stage_summary(value[2], value[0]["meta"]) for arm_id,value in loaded.items()}
    result["root_state_comparison"] = _root_invariants(result["fingerprints"])
    result["candidate_changes_to_d05"] = {arm_id:_candidate_comparison(loaded["C01_d05"][2], value[2])
        for arm_id,value in loaded.items()} if "C01_d05" in loaded else {}
    result["paired_to_c02"] = {arm_id:_paired(loaded["C02_d05_staged"][2], value[2], value[0]["meta"])
        for arm_id,value in loaded.items()} if "C02_d05_staged" in loaded else {}
    protocol.write_json(root / ("diagnostics_" + phase + ".json"), result)
    return result, species, sources


def summarize_suite(root_dir, phase="complete"):
    if phase not in ("development", "complete"):
        raise ValueError("Unknown morphology reporting phase")
    selection = freeze_dev_selection(root_dir)
    if phase == "development":
        return selection
    root, cfg, snapshot, ids = _suite(root_dir)
    _, dev = _development(root, snapshot, ids)
    tests, failures = {}, dict(selection["technical_failures"])
    for arm_id in dev:
        failure = _failure(root, arm_id)
        if failure is not None:
            if failure["stage"] != "test":
                raise ValueError("Calibrated arm acquired an inconsistent failure record")
            failures[arm_id] = failure
            continue
        value = _stage(root, arm_id, "test", snapshot)
        if value[0].get("calibration_receipt_sha256") != protocol.file_hash(root / "arms" / arm_id / "calibration/completed.json"):
            raise ValueError("TEST result is not bound to its own DEV router")
        tests[arm_id] = value
    ranking = _rank([_entry(arm_id, value, tests.get(ids[0]), ids[0], "test", tests.get("C01_d05")) for arm_id, value in tests.items()])
    diagnostics, species, sources = {}, [], []
    for name, loaded in (("development", dev), ("test", tests)):
        diagnostics[name], per_species, per_source = _comparisons(root, name, loaded, ids[0])
        species.extend(per_species)
        sources.extend(per_source)
    dev_entries = {r["arm_id"]: r for r in selection["exploratory_development_ranking"]}
    test_entries = {r["arm_id"]: r for r in ranking}
    table = []
    from .runner import _verify_stage
    for arm in cfg["arms"]:
        arm_id = arm["id"]
        training = None
        if (root / "arms" / arm_id / "training/stage_binding.json").is_file():
            training = _verify_stage(root, arm_id, "training", snapshot)
        execution = (training.get("fit_report", {}).get("training_execution") or
                     ("spatial_TRAIN_gradient_fit" if arm["kind"] == "fit" else "reused" if arm["kind"] == "reuse" else "source_inherited")) if training is not None else "failed_or_blocked"
        row = dict(_matrix_row(arm), **{"training_execution": execution,
            "optimizer_steps": None if training is None else training.get("optimizer_steps"),
            "weight_source": None if training is None else training.get("weight_source"),
            "model_sha256": None if training is None else training.get("model", {}).get("sha256"),
            "calibration_execution": "completed" if arm_id in dev else "failed_or_blocked",
            "test_execution": "completed" if arm_id in tests else "failed_or_blocked",
            "failure_reason": failures.get(arm_id, {}).get("error"),
            "recommended_by_development": selection["recommendation_arm_id"] == arm_id,
            "training_seconds": _timing(root, arm_id, "training"),
            "calibration_seconds": _timing(root, arm_id, "calibration"),
            "test_seconds": _timing(root, arm_id, "test")})
        for label, entries in (("dev", dev_entries), ("test", test_entries)):
            value = entries.get(arm_id)
            row[label + "_targets_passed"] = None if value is None else value["targets_passed"]
            row[label + "_known_count_preserved"] = None if value is None else value["known_count_preserved"]
            crossfit_key = "dev_crossfit_passed" if label == "dev" else "inherited_dev_crossfit_passed"
            row[crossfit_key] = None if value is None else value[crossfit_key]
            row[label + "_research_recovery_observed"] = None if value is None else value["research_recovery_observed"]
            recovery_key = "dev_research_recovery_oof_passed" if label == "dev" else "inherited_dev_research_recovery_oof_passed"
            row[recovery_key] = None if value is None else value[recovery_key]
            for key in ("name", "selected_tier", "reason"):
                row[label + "_morphology_policy_" + key] = None if value is None else value["morphology_policy"].get(key)
            for status in ("known", "intra", "extra"):
                row[label + "_source_macro_" + status] = None if value is None else value["source_macro"].get(status)
            for key in ("d05_known_lost_correct_count", "d05_known_gained_correct_count"):
                row[label + "_" + key] = None if value is None else value[key]
            for metric in base.TARGETS:
                row[label + "_" + metric] = None if value is None else value["metrics"][metric]
            for key in ("known_root_rejected", "near_root_rejected", "extra_root_rejected",
                        "known_passed_root", "near_passed_root", "extra_passed_root",
                        "known_leaf_correct_after_root", "known_leaf_wrong_after_root",
                        "unknown_leaf_false_accept_after_root", "known_parent_abstentions_after_root", "near_parent_correct_after_root"):
                row[label + "_" + key] = None if value is None else value["root_stage_outcomes"][key]
        table.append(row)
    _csv(root / "comparison_all.csv", table)
    matrix = [_matrix_row(arm) for arm in cfg["arms"]]
    _csv(root / "experiment_matrix.csv", matrix)
    if species:
        _csv(root / "per_species_comparison.csv", species)
    if sources:
        _csv(root / "per_source_comparison.csv", sources)
    summary = {"schema_version": SCHEMA_VERSION, "phase": "complete", "workflow_completed": True,
        "validation_scope": VALIDATION_SCOPE, "confirmatory_validation": False,
        "dev_selection_sha256": protocol.file_hash(root / "dev_selection.json"), "dev_selection": selection,
        "recommendation_arm_id": selection["recommendation_arm_id"], "recommendation_status": selection["recommendation_status"],
        "production_recommendation_uses_test": False,
        "parent_chosen_after_prior_test_review": True,
        "research_recovery_arm_id": selection["research_recovery_arm_id"],
        "research_recovery_is_deployment_recommendation": False, "declared_arm_count": len(ids),
        "completed_calibration_count": len(dev), "completed_test_count": len(tests), "technical_failures": failures,
        "all_arms": table, "experiment_matrix": matrix, "diagnostics": diagnostics,
        "stage_timing_scope": "Worker entry through saved artifacts; includes validation/loading but excludes interpreter startup and parent orchestration. Missing manual-fixture timing remains null.",
        "exploratory_test_ranking": {"production_selection": False, "selection_uses_test": True,
                                     "purpose": "descriptive comparison only", "arms": ranking},
        "best_exploratory_test_arm": ranking[0] if ranking else None}
    protocol.write_json(root / "summary.json", summary)
    lines = ["# D05 morphology experiment comparison", "", "DEV recommendation: " + str(selection["recommendation_arm_id"]),
        "Status: " + selection["recommendation_status"], "",
        "All valid calibrations are tested even when research gates fail.",
        "Failed or unavailable results remain null. C00/C01/C02 inherit weights; R01/L01 record actual optimizer steps.",
        "A DEV candidate must pass all four gates and crossfit, with known correct count at least reference.",
        "Research recovery relative to D05 is recorded separately and cannot qualify deployment.",
        "Both reference and D05 paired losses are reported. TEST cannot change either frozen DEV recommendation.",
        "Root and leaf errors are separated. All new arms keep the fixed reference candidate path.",
        "The TEST crossfit columns inherit frozen DEV audits; no OOF threshold fitting occurs on TEST.",
        "L01 and C02 share identical root scores, thresholds and admitted sets, including corresponding DEV OOF fit folds.",
        "Stage seconds measure worker execution including loading/validation, excluding interpreter startup. Missing timing remains null.",
        "D05 was chosen after prior TEST review; no independent validation claim is made.",
        "This reused benchmark is exploratory, not independent confirmatory validation.", "",
        "Declared arms: {}; completed DEV: {}; completed TEST: {}.".format(len(ids), len(dev), len(tests)), "",
        "| Arm | Mechanism | Inputs | Loss | Policy | Weight source |", "|---|---|---|---|---|---|"]
    lines.extend("| {arm_id} | {mechanism} | {inputs} | {loss} | {policy} | {declared_weight_source} |".format(**row) for row in matrix)
    (root / "comparison_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
