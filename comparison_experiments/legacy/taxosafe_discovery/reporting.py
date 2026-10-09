"""Audited comparison of declared arms, including explicit unavailable results.

DEV freezes the recommendation before any TEST cache exists. TEST rankings
remain descriptive; no missing/failed arm receives invented zero metrics.
"""
from collections import defaultdict
from pathlib import Path

from taxosafe_support import calibration as base
from taxosafe_routealign.evaluation import _csv, _records, _regular
from taxosafe_routealign.reporting import _paired, _source_macro
from . import protocol

SCHEMA_VERSION = "taxosafe_discovery_comparison_v1"
VALIDATION_SCOPE = "exploratory_discovery_on_reused_development_and_test"
RANKING = ("DEV qualified first (four gates, known correct count at least baseline, crossfit passed); "
           "then four gates passed, known correct count preserved, mean four metrics descending "
           "(undefined last), known correct count descending, arm ID ascending")


def _suite(directory):
    root = Path(directory)
    cfg = protocol.read_json(_regular(root / "config.json"))
    snapshot = protocol.read_json(_regular(root / "snapshot.json"))
    ids = [arm["id"] for arm in cfg["arms"]]
    if (not ids or len(ids) != len(set(ids)) or snapshot.get("schema_version") != protocol.SCHEMA_VERSION
            or snapshot.get("config_sha256") != protocol.object_hash(cfg) or snapshot.get("arm_ids") != ids
            or protocol.read_json(_regular(root / "source_binding.json")) != snapshot.get("source_binding")):
        raise ValueError("Discovery configuration/snapshot identity changed")
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
    if not isinstance(report, dict) or report.get("schema_version") != "discovery_evaluation_v1":
        raise ValueError("Invalid discovery evaluation report schema: " + arm_id)
    for key in ("counts", "metrics", "checks", "targets_passed"):
        if report.get(key) != calculated[key] or receipt.get("summary", {}).get(key) != calculated[key]:
            raise ValueError("Saved metrics do not reproduce predictions: " + arm_id)
    if report.get("crossfit_audit") != receipt["summary"].get("crossfit_audit"):
        raise ValueError("Crossfit report/receipt mismatch: " + arm_id)
    return receipt, report, rows


def _rank(rows):
    return sorted(rows, key=lambda r: (-int(r["qualified"]), -int(r["targets_passed"]),
        -int(r["known_count_preserved"]), -(r["mean_four_metrics"] if r["mean_four_metrics"] is not None else -1.),
        -r["counts"]["known_correct"], r["arm_id"]))


def _entry(arm_id, value, reference, baseline_id, phase):
    receipt, report, rows = value
    audit = None if reference is None else _paired(reference[2], rows, receipt["meta"])
    preserved = reference is not None and report["counts"]["known_correct"] >= reference[1]["counts"]["known_correct"]
    crossfit = report.get("crossfit_audit")
    crossfit_passed = isinstance(crossfit, dict) and crossfit.get("passed") is True
    qualified = bool(phase == "development" and arm_id != baseline_id and report["targets_passed"] and preserved and crossfit_passed)
    metrics = report["metrics"]
    if any(metrics.get(name) is None for name in base.TARGETS):
        mean = None
    else:
        mean = sum(metrics[name] for name in base.TARGETS) / len(base.TARGETS)
    return {"arm_id": arm_id, "targets_passed": report["targets_passed"], "qualified": qualified,
        "known_count_preserved": bool(preserved), "crossfit_audit_passed": crossfit_passed if arm_id != baseline_id else None,
        "known_lost_correct_count": None if audit is None else audit["known_lost_correct_count"],
        "known_gained_correct_count": None if audit is None else audit["known_gained_correct_count"],
        "counts": report["counts"], "metrics": metrics, "checks": report["checks"], "mean_four_metrics": mean,
        "source_macro": _source_macro(rows, receipt["meta"])["source_macro"],
        "research_status": "descriptive_test_result" if phase == "test" else "qualified_on_development" if qualified else "best_effort_unqualified"}


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
    ranking = _rank([_entry(arm_id, value, reference, ids[0], "development") for arm_id, value in loaded.items()])
    qualified = [row for row in ranking if row["qualified"]]
    chosen = qualified[0]["arm_id"] if qualified else None
    selection = {"schema_version": SCHEMA_VERSION, "phase": "development", "suite_signature": snapshot["signature"],
        "config_sha256": snapshot["config_sha256"], "snapshot_sha256": protocol.file_hash(root / "snapshot.json"),
        "inventory": inventory, "ranking_rule": RANKING, "selection_uses_test": False, "test_predictions_read": False,
        "validation_scope": VALIDATION_SCOPE, "confirmatory_validation": False, "development_reused_for_method_design": True,
        "qualified_candidate_arm_id": chosen, "recommendation_arm_id": chosen or (ids[0] if reference is not None else None),
        "recommendation_status": "qualified_on_development" if chosen else "retain_reference_no_qualified_candidate" if reference is not None else "reference_unavailable",
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
        if protocol.read_json(_regular(path)) != selection:
            raise ValueError("Immutable DEV recommendation/calibration inventory changed")
    else:
        protocol.write_json(path, selection)
    return selection


def _fingerprints(root, arm_id, stage, value):
    receipt, report, rows = value
    rows = sorted(rows, key=base._digest)
    terminal = [{"image_sha256": base._digest(row), **{key: row[key] for key in ("prediction_type", "parent", "leaf")}} for row in rows]
    candidate_keys = ("candidate_parent", "candidate_leaf")
    candidates = [{"image_sha256": base._digest(row), **{key: row.get(key) for key in candidate_keys}} for row in rows]
    artifact = receipt["artifacts"].get("scores")
    return {"arm_id": arm_id, "unique_image_count": len(rows),
        "model_sha256": receipt.get("model_sha256"),
        "router_file_sha256": receipt["artifacts"]["router"]["sha256"],
        "terminal_prediction_sha256": protocol.object_hash(terminal),
        "candidate_sha256": protocol.object_hash(candidates) if all(all(key in r for key in candidate_keys) for r in rows) else None,
        "raw_scores_file_sha256": None if artifact is None else artifact["sha256"],
        "interpretation": "Equal terminal fingerprints mean identical per-image outputs, not independently successful methods or identical model weights."}


def _comparisons(root, phase, loaded, baseline_id):
    paired, source_macro, fingerprints, pairwise = {}, {}, {}, []
    species_rows, source_rows = [], []
    for arm_id, value in loaded.items():
        source_macro[arm_id] = _source_macro(value[2], value[0]["meta"])
        fingerprints[arm_id] = _fingerprints(root, arm_id, phase, value)
        for row in source_macro[arm_id]["per_source"]:
            source_rows.append({"phase": phase, "arm_id": arm_id, **row})
        if baseline_id in loaded:
            paired[arm_id] = _paired(loaded[baseline_id][2], value[2], value[0]["meta"])
            for row in paired[arm_id]["per_species"]:
                species_rows.append({"phase": phase, "arm_id": arm_id,
                    **{key: v for key, v in row.items() if not key.endswith("sha256")}})
    ids = list(loaded)
    for i, left in enumerate(ids):
        for right in ids[i + 1:]:
            lrows = {base._digest(r): r for r in loaded[left][2]}
            rrows = {base._digest(r): r for r in loaded[right][2]}
            audit = _paired(list(lrows.values()), list(rrows.values()), loaded[left][0]["meta"])
            disagreements = sum(any(lrows[h][key] != rrows[h][key] for key in ("prediction_type", "parent", "leaf")) for h in lrows)
            pairwise.append({"left_arm": left, "right_arm": right, "unique_image_count": len(lrows),
                "terminal_disagreement_count": disagreements, "identical_predictions": disagreements == 0,
                "known_left_correct_right_wrong": audit["known_lost_correct_count"],
                "known_left_wrong_right_correct": audit["known_gained_correct_count"]})
    groups = defaultdict(list)
    for arm_id, value in fingerprints.items():
        groups[value["terminal_prediction_sha256"]].append(arm_id)
    result = {"paired_to_reference": paired, "source_macro": source_macro, "fingerprints": fingerprints,
        "identical_prediction_groups": [ids for ids in groups.values() if len(ids) > 1], "all_pairwise": pairwise}
    protocol.write_json(root / ("diagnostics_" + phase + ".json"), result)
    return result, species_rows, source_rows


def summarize_suite(root_dir, phase="complete"):
    if phase not in ("development", "complete"):
        raise ValueError("Unknown discovery reporting phase")
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
    ranking = _rank([_entry(arm_id, value, tests.get(ids[0]), ids[0], "test") for arm_id, value in tests.items()])
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
        row = {"arm_id": arm_id, "training_execution": "completed" if training is not None else "failed_or_blocked",
            "optimizer_steps": None if training is None else training.get("optimizer_steps"),
            "weight_source": None if training is None else training.get("weight_source"),
            "model_sha256": None if training is None else training.get("model", {}).get("sha256"),
            "calibration_execution": "completed" if arm_id in dev else "failed_or_blocked",
            "test_execution": "completed" if arm_id in tests else "failed_or_blocked",
            "failure_reason": failures.get(arm_id, {}).get("error"),
            "recommended_by_development": selection["recommendation_arm_id"] == arm_id}
        for label, entries in (("dev", dev_entries), ("test", test_entries)):
            value = entries.get(arm_id)
            row[label + "_targets_passed"] = None if value is None else value["targets_passed"]
            row[label + "_known_count_preserved"] = None if value is None else value["known_count_preserved"]
            row[label + "_crossfit_passed"] = None if value is None else value["crossfit_audit_passed"]
            for metric in base.TARGETS:
                row[label + "_" + metric] = None if value is None else value["metrics"][metric]
        table.append(row)
    _csv(root / "comparison_all.csv", table)
    if species:
        _csv(root / "per_species_comparison.csv", species)
    if sources:
        _csv(root / "per_source_comparison.csv", sources)
    summary = {"schema_version": SCHEMA_VERSION, "phase": "complete", "workflow_completed": True,
        "validation_scope": VALIDATION_SCOPE, "confirmatory_validation": False,
        "dev_selection_sha256": protocol.file_hash(root / "dev_selection.json"), "dev_selection": selection,
        "recommendation_arm_id": selection["recommendation_arm_id"], "recommendation_status": selection["recommendation_status"],
        "production_recommendation_uses_test": False, "declared_arm_count": len(ids),
        "completed_calibration_count": len(dev), "completed_test_count": len(tests), "technical_failures": failures,
        "all_arms": table, "diagnostics": diagnostics,
        "exploratory_test_ranking": {"production_selection": False, "selection_uses_test": True,
                                     "purpose": "descriptive comparison only", "arms": ranking},
        "best_exploratory_test_arm": ranking[0] if ranking else None}
    protocol.write_json(root / "summary.json", summary)
    lines = ["# Discovery experiment comparison", "", "DEV recommendation: " + str(selection["recommendation_arm_id"]),
        "Status: " + selection["recommendation_status"], "",
        "All valid calibrations are tested even when research gates fail.",
        "Failed or unavailable results remain null; zero optimizer steps identify statistical fitting or reuse.",
        "A DEV candidate must pass all four gates and crossfit, with known correct count at least reference.",
        "Paired losses are reported separately. TEST cannot change the frozen DEV recommendation.",
        "This reused benchmark is exploratory, not independent confirmatory validation.", "",
        "Declared arms: {}; completed DEV: {}; completed TEST: {}.".format(len(ids), len(dev), len(tests))]
    (root / "comparison_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
