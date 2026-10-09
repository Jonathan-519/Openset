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

SCHEMA_VERSION = "taxosafe_boundary_comparison_v1"
VALIDATION_SCOPE = "exploratory_d05_boundary_on_reused_development_and_test"
RANKING = ("DEV qualified first (four gates, known correct count at least baseline, crossfit passed); "
           "then four gates passed, known correct count preserved, mean four metrics descending "
           "(undefined last), known correct count descending, arm ID ascending")

# Descriptions are fixed with the predeclared matrix, not inferred from TEST.
MECHANISMS = {
    "G00_reference": ("original reference", "original reference evidence", "none; inherited weights"),
    "G01_d05": ("original D05", "eight D05 verifier features", "none; inherited weights"),
    "G02_d05_kp": ("D05 threshold control", "eight D05 verifier features", "none; exact G01 model reuse"),
    "G03_evm_leaf": ("per-witness EVM leaf boundary", "leaf EVM score; unchanged D05 parent score", "TRAIN boundary fit; no SGD"),
    "G04_bce8": ("eight-feature continuation", "eight D05 verifier features", "BCE + L2-SP"),
    "G05_rank8": ("eight-feature cross-query ranking", "eight D05 verifier features", "BCE + L2-SP + same-bank cross-query ranking"),
    "G06_bce9": ("nine-feature boundary continuation", "eight D05 features plus EVM score", "BCE + L2-SP"),
    "G07_rank9": ("nine-feature boundary ranking", "eight D05 features plus EVM score", "BCE + L2-SP + same-bank cross-query ranking"),
    "G08_rank9_standard": ("standard-threshold control", "exact G07 nine-feature verifier", "none; exact G07 model reuse"),
    "G09_rank9_leaf_guard": ("fixed-parent boundary ranking", "G07 nine-feature leaf; original D05 eight-feature parent", "none; assembled heads"),
}


def _matrix_row(arm):
    mechanism, inputs, loss = MECHANISMS[arm["id"]]
    return dict(arm_id=arm["id"], kind=arm["kind"], mode=arm["mode"], mechanism=mechanism,
                inputs=inputs, loss=loss, policy=arm["policy"],
                declared_weight_source=arm.get("weight_source", "reference" if arm["kind"] == "reference" else "original_D05"))


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
        raise ValueError("Boundary configuration/snapshot identity changed")
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
    rows = base.unique_records(_records(directory / artifacts["predictions"]["path"]))
    calculated = base.evaluate_records(rows, receipt["meta"])
    if any(calculated["counts"][status] == 0 for status in ("known", "intra", "extra")):
        raise ValueError("Evaluation must contain every locked status group")
    report = protocol.read_json(_regular(directory / artifacts["report"]["path"]))
    if not isinstance(report, dict) or report.get("schema_version") != "boundary_evaluation_v1":
        raise ValueError("Invalid boundary evaluation report schema: " + arm_id)
    for key in ("counts", "metrics", "checks", "targets_passed"):
        if report.get(key) != calculated[key] or receipt.get("summary", {}).get(key) != calculated[key]:
            raise ValueError("Saved metrics do not reproduce predictions: " + arm_id)
    if report.get("crossfit_audit") != receipt["summary"].get("crossfit_audit"):
        raise ValueError("Crossfit report/receipt mismatch: " + arm_id)
    return receipt, report, rows


def _entry(arm_id, value, reference, baseline_id, phase, d05=None):
    result = _base_entry(arm_id, value, reference, baseline_id, phase)
    _, report, rows = value
    crossfit = report.get("crossfit_audit") or {}
    observed = False
    checks, audit = {}, None
    if d05 is not None:
        if value[0]["meta"] != d05[0]["meta"]:
            raise ValueError("Boundary comparison taxonomy differs")
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
        research_recovery_oof_passed=oof, research_recovery_qualified=phase == "development" and observed and oof,
        d05_known_lost_correct_count=None if audit is None else audit["known_lost_correct_count"],
        d05_known_gained_correct_count=None if audit is None else audit["known_gained_correct_count"],
        recovery_is_deployment_qualification=False)
    details = report.get("boundary_policy") or (report.get("calibration_diagnostics") or {}).get("boundary_policy") or {}
    result["boundary_policy"] = details
    result["pareto_frontier"] = report.get("pareto_frontier", (report.get("calibration_diagnostics") or {}).get("pareto_frontier"))
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
    ranking = _rank([_entry(arm_id, value, reference, ids[0], "development", loaded.get("G01_d05")) for arm_id, value in loaded.items()])
    qualified = [row for row in ranking if row["qualified"]]
    chosen = qualified[0]["arm_id"] if qualified else None
    research = sorted([r for r in ranking if r["research_recovery_qualified"]],
        key=lambda r: (-r["counts"]["known_correct"], -(r["mean_four_metrics"] or 0.), r["arm_id"]))
    selection = {"schema_version": SCHEMA_VERSION, "phase": "development", "suite_signature": snapshot["signature"],
        "config_sha256": snapshot["config_sha256"], "snapshot_sha256": protocol.file_hash(root / "snapshot.json"),
        "inventory": inventory, "ranking_rule": RANKING, "selection_uses_test": False, "test_predictions_read": False,
        "validation_scope": VALIDATION_SCOPE, "confirmatory_validation": False, "development_reused_for_method_design": True,
        "parent_chosen_after_prior_test_review": True,
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


def _comparisons(root, phase, loaded, baseline_id):
    result, species, sources = _base_comparisons(root, phase, loaded, baseline_id)
    species = [dict(row, comparison_baseline=baseline_id) for row in species]
    result["paired_to_d05"] = {}
    if "G01_d05" in loaded:
        before = loaded["G01_d05"]
        for arm_id, value in loaded.items():
            audit = _paired(before[2], value[2], value[0]["meta"])
            result["paired_to_d05"][arm_id] = audit
            species.extend(dict(phase=phase, arm_id=arm_id, comparison_baseline="G01_d05",
                **{key: item for key, item in row.items() if not key.endswith("sha256")}) for row in audit["per_species"])
    protocol.write_json(root / ("diagnostics_" + phase + ".json"), result)
    return result, species, sources


def summarize_suite(root_dir, phase="complete"):
    if phase not in ("development", "complete"):
        raise ValueError("Unknown boundary reporting phase")
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
    ranking = _rank([_entry(arm_id, value, tests.get(ids[0]), ids[0], "test", tests.get("G01_d05")) for arm_id, value in tests.items()])
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
        row = dict(_matrix_row(arm), **{"training_execution": "completed" if training is not None else "failed_or_blocked",
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
            row[label + "_crossfit_passed"] = None if value is None else value["crossfit_audit_passed"]
            row[label + "_research_recovery_observed"] = None if value is None else value["research_recovery_observed"]
            row[label + "_research_recovery_oof_passed"] = None if value is None else value["research_recovery_oof_passed"]
            for key in ("name", "selected_tier", "reason"):
                row[label + "_boundary_policy_" + key] = None if value is None else value["boundary_policy"].get(key)
            for status in ("known", "intra", "extra"):
                row[label + "_source_macro_" + status] = None if value is None else value["source_macro"].get(status)
            for key in ("d05_known_lost_correct_count", "d05_known_gained_correct_count"):
                row[label + "_" + key] = None if value is None else value[key]
            for metric in base.TARGETS:
                row[label + "_" + metric] = None if value is None else value["metrics"][metric]
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
    lines = ["# D05 boundary experiment comparison", "", "DEV recommendation: " + str(selection["recommendation_arm_id"]),
        "Status: " + selection["recommendation_status"], "",
        "All valid calibrations are tested even when research gates fail.",
        "Failed or unavailable results remain null; zero optimizer steps identify statistical fitting or reuse.",
        "A DEV candidate must pass all four gates and crossfit, with known correct count at least reference.",
        "Research recovery relative to D05 is recorded separately and cannot qualify deployment.",
        "Both reference and D05 paired losses are reported. TEST cannot change either frozen DEV recommendation.",
        "Stage seconds measure worker execution including loading/validation, excluding interpreter startup. Missing timing remains null.",
        "D05 was chosen after prior TEST review; no independent validation claim is made.",
        "This reused benchmark is exploratory, not independent confirmatory validation.", "",
        "Declared arms: {}; completed DEV: {}; completed TEST: {}.".format(len(ids), len(dev), len(tests)), "",
        "| Arm | Mechanism | Inputs | Loss | Policy | Weight source |", "|---|---|---|---|---|---|"]
    lines.extend("| {arm_id} | {mechanism} | {inputs} | {loss} | {policy} | {declared_weight_source} |".format(**row) for row in matrix)
    (root / "comparison_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
