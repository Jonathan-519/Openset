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

SCHEMA_VERSION = "taxosafe_domain_comparison_v1"
VALIDATION_SCOPE = "exploratory_d05_domain_on_reused_development_and_test"
RANKING = ("DEV qualified first (four gates, known correct count at least baseline, crossfit passed); "
           "then four gates passed, known correct count preserved, mean four metrics descending "
           "(undefined last), known correct count descending, arm ID ascending")

# Descriptions are fixed with the predeclared matrix, not inferred from TEST.
MECHANISMS = {
    "H00_reference": ("original reference", "original reference evidence", "none; inherited weights"),
    "H01_d05": ("original D05", "eight D05 verifier features", "none; inherited weights"),
    "H02_d05_staged": ("staged D05 thresholds", "fixed D05 parent and leaf confidence", "none; threshold calibration only"),
    "H03_d05_marginal": ("marginal D05 thresholds", "fixed D05 parent and leaf confidence", "none; threshold calibration only"),
    "H04_subspace_root": ("parent subspace root", "parent PPCA absolute residual; D05 leaf", "known TRAIN statistics; no optimizer"),
    "H05_density_root": ("parent density root", "parent PPCA absolute and relative density; D05 leaf", "none; exact H04 bank reuse"),
    "H06_dual_root": ("dual parent domain root", "parent residual and density; D05 leaf", "none; exact H04 bank reuse"),
    "H07_conditional_leaf": ("conditional leaf density", "D05 root; parent-conditional leaf density", "none; exact H04 bank reuse"),
    "H08_dual_conditional": ("dual root with conditional leaf", "parent residual and density; conditional leaf density", "none; exact H04 bank reuse"),
    "H09_dual_reroute": ("dual parent candidate rerouting", "H08 confidence; parent rerank and within-parent reference leaf", "none; exact H04 bank reuse"),
    "H10_dual_joint": ("joint threshold control", "H08 confidence with joint root and leaf threshold search", "none; exact H04 bank reuse"),
    "H11_dual_rank16": ("subspace rank sensitivity", "parent/leaf/global ranks 16/8/32; dual domain and conditional leaf density", "known TRAIN statistics; no optimizer"),
}


def _matrix_row(arm):
    mechanism, inputs, loss = MECHANISMS[arm["id"]]
    return dict(arm_id=arm["id"], kind=arm["kind"], mode=arm["mode"], mechanism=mechanism,
                inputs=inputs, loss=loss, policy=arm["policy"],
                declared_weight_source=arm.get("weight_source", "reference" if arm["kind"] == "reference" else
                                              "own_known_TRAIN_statistics" if arm["kind"] == "fit" else "original_D05"),
                candidate_policy=arm.get("candidate_policy", "source"))


def compact_crossfit(full):
    """Persist one full DEV audit; carry only decision fields into each report."""
    if full is None:
        return None
    if not isinstance(full, dict):
        raise ValueError("Domain crossfit audit must be a mapping")
    keys = ("passed", "recovery_passed", "reason", "status", "validation_scope", "confirmatory_validation",
            "independent_model_level_validation", "development_reused_for_method_design", "test_used_for_fitting")
    result = {key: copy.deepcopy(full[key]) for key in keys if key in full}
    folds = full.get("folds")
    result["fold_count"] = len(folds) if isinstance(folds, list) else full.get("fold_count")
    if isinstance(full.get("known_recovery"), dict):
        result["known_recovery"] = {key: copy.deepcopy(full["known_recovery"][key])
            for key in ("passed", "checks", "counts") if key in full["known_recovery"]}
    return result


def _unique_domain_records(records):
    seen = {}
    for row in records:
        digest = base._digest(row)
        if digest in seen and row.get("domain") != seen[digest].get("domain"):
            raise ValueError("Same image has conflicting domain evidence")
        seen[digest] = row
    return base.unique_records(records)


def root_stage_summary(records, meta):
    """Observed terminal outcomes, counted once per image, independent of fits."""
    rows = _unique_domain_records(records)
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
        raise ValueError("Domain configuration/snapshot identity changed")
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
    rows = _unique_domain_records(_records(directory / artifacts["predictions"]["path"]))
    calculated = base.evaluate_records(rows, receipt["meta"])
    if any(calculated["counts"][status] == 0 for status in ("known", "intra", "extra")):
        raise ValueError("Evaluation must contain every locked status group")
    report = protocol.read_json(_regular(directory / artifacts["report"]["path"]))
    if not isinstance(report, dict) or report.get("schema_version") != "domain_evaluation_v1":
        raise ValueError("Invalid domain evaluation report schema: " + arm_id)
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
            raise ValueError("Domain comparison taxonomy differs")
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
    details = report.get("domain_policy") or (report.get("calibration_diagnostics") or {}).get("domain_policy") or {}
    result["domain_policy"] = details
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
    ranking = _rank([_entry(arm_id, value, reference, ids[0], "development", loaded.get("H01_d05")) for arm_id, value in loaded.items()])
    roots = _root_invariants({arm_id:_domain_fingerprints(root, arm_id, "development", value)
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


def _domain_fingerprints(root, arm_id, phase, value):
    receipt, report, rows = value
    ordered = sorted(rows, key=base._digest)
    has_domain = all(isinstance(row.get("domain"), dict) for row in ordered)
    result = {}
    for label, fields in (("root_score", ("root_score",)), ("leaf_score", ("leaf_scores",)),
                          ("parent_score", ("parent_scores",)), ("anchor", ("anchor_parent", "anchor_leaf")),
                          ("domain_candidate", ("candidate_parent", "candidate_leaf"))):
        result[label + "_sha256"] = protocol.object_hash([
            dict(image_sha256=base._digest(row), **{key:row["domain"][key] for key in fields})
            for row in ordered]) if has_domain else None
    stage = "calibration" if phase == "development" else "test"
    router = protocol.read_json(_regular(root / "arms" / arm_id / stage / receipt["artifacts"]["router"]["path"]))
    result["root_state_sha256"] = router.get("root_state_sha256")
    result["candidate_policy"] = sorted({row["domain"]["candidate_policy"] for row in ordered}) if has_domain else ["original"]
    result["score_semantics"] = "separate scalar root confidence, parent ranking evidence and conditional or inherited leaf confidence"
    return result


def _candidate_comparison(before, after):
    left = {base._digest(row):row for row in _unique_domain_records(before)}
    right = {base._digest(row):row for row in _unique_domain_records(after)}
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
            evidence = value.get("domain", value.get("discovery", {}))
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
    score_ids = [key for key in ("H06_dual_root", "H08_dual_conditional", "H09_dual_reroute", "H10_dual_joint") if key in fingerprints]
    staged_ids = [key for key in score_ids if key != "H10_dual_joint"]
    def identical(ids, key):
        if len(ids) < 2:
            return None
        values = [fingerprints[arm][key] for arm in ids]
        return all(value is not None for value in values) and len(set(values)) == 1
    score_equal = identical(score_ids, "root_score_sha256")
    state_equal = identical(staged_ids, "root_state_sha256")
    if score_equal is False or state_equal is False:
        raise ValueError("Shared domain root scores or staged root state changed across declared controls")
    return dict(shared_root_score_arms=score_ids, shared_root_scores_exact=score_equal,
                shared_staged_root_arms=staged_ids, staged_root_states_exact=state_equal,
                joint_arm_root_threshold_may_differ=True,
                validation="not_evaluable" if score_equal is None else "verified")


def _comparisons(root, phase, loaded, baseline_id):
    result, species, sources = _base_comparisons(root, phase, loaded, baseline_id)
    species = [dict(row, comparison_baseline=baseline_id) for row in species]
    result["paired_to_d05"] = {}
    if "H01_d05" in loaded:
        before = loaded["H01_d05"]
        for arm_id, value in loaded.items():
            audit = _paired(before[2], value[2], value[0]["meta"])
            result["paired_to_d05"][arm_id] = audit
            species.extend(dict(phase=phase, arm_id=arm_id, comparison_baseline="H01_d05",
                **{key: item for key, item in row.items() if not key.endswith("sha256")}) for row in audit["per_species"])
    for arm_id, value in loaded.items():
        result["fingerprints"][arm_id].update(_domain_fingerprints(root, arm_id, phase, value))
    result["root_stage_outcomes"] = {arm_id:root_stage_summary(value[2], value[0]["meta"]) for arm_id,value in loaded.items()}
    result["root_state_comparison"] = _root_invariants(result["fingerprints"])
    result["candidate_changes_to_d05"] = {arm_id:_candidate_comparison(loaded["H01_d05"][2], value[2])
        for arm_id,value in loaded.items()} if "H01_d05" in loaded else {}
    result["reroute_control"] = (_candidate_comparison(loaded["H08_dual_conditional"][2], loaded["H09_dual_reroute"][2])
        if "H08_dual_conditional" in loaded and "H09_dual_reroute" in loaded else None)
    protocol.write_json(root / ("diagnostics_" + phase + ".json"), result)
    return result, species, sources


def summarize_suite(root_dir, phase="complete"):
    if phase not in ("development", "complete"):
        raise ValueError("Unknown domain reporting phase")
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
    ranking = _rank([_entry(arm_id, value, tests.get(ids[0]), ids[0], "test", tests.get("H01_d05")) for arm_id, value in tests.items()])
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
                     ("statistical_TRAIN_fit" if arm["kind"] == "fit" else "reused" if arm["kind"] == "reuse" else "source_inherited")) if training is not None else "failed_or_blocked"
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
                row[label + "_domain_policy_" + key] = None if value is None else value["domain_policy"].get(key)
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
    lines = ["# D05 domain experiment comparison", "", "DEV recommendation: " + str(selection["recommendation_arm_id"]),
        "Status: " + selection["recommendation_status"], "",
        "All valid calibrations are tested even when research gates fail.",
        "Failed or unavailable results remain null; zero optimizer steps identify statistical fitting or reuse.",
        "A DEV candidate must pass all four gates and crossfit, with known correct count at least reference.",
        "Research recovery relative to D05 is recorded separately and cannot qualify deployment.",
        "Both reference and D05 paired losses are reported. TEST cannot change either frozen DEV recommendation.",
        "Root-stage rejection and post-root leaf outcomes are reported separately; H09 candidate rerouting is explicit.",
        "The TEST crossfit columns inherit frozen DEV audits; no OOF threshold fitting occurs on TEST.",
        "H06/H08/H09 share one staged root state; H10 shares root scores but may choose a different joint threshold.",
        "Stage seconds measure worker execution including loading/validation, excluding interpreter startup. Missing timing remains null.",
        "D05 was chosen after prior TEST review; no independent validation claim is made.",
        "This reused benchmark is exploratory, not independent confirmatory validation.", "",
        "Declared arms: {}; completed DEV: {}; completed TEST: {}.".format(len(ids), len(dev), len(tests)), "",
        "| Arm | Mechanism | Inputs | Loss | Policy | Weight source |", "|---|---|---|---|---|---|"]
    lines.extend("| {arm_id} | {mechanism} | {inputs} | {loss} | {policy} | {declared_weight_source} |".format(**row) for row in matrix)
    (root / "comparison_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
