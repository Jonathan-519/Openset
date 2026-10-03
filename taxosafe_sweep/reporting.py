"""Freeze DEV-only recommendations, then describe every completed TEST arm.

TEST rankings describe this reused benchmark. They never change the immutable
development recommendation, model tensors, or calibrated operating points.
"""
from collections import defaultdict
from pathlib import Path

from taxosafe_support import calibration as base
from taxosafe_dcbs.protocol import normalized_name
from . import protocol
from .evaluation import (_read, _write, _records, _hash, _correct, _csv,
                         _regular, _verify_completed, VALIDATION_SCOPE)

SCHEMA_VERSION = "taxosafe_sweep_comparison_v1"
RANKING = ("all four gates passed, known correct count at least reference, "
           "mean of four metrics descending, known correct count descending, arm ID ascending")


def _suite(root):
    root = Path(root)
    cfg = _read(_regular(root / "config.json"))
    snapshot = _read(_regular(root / "snapshot.json"))
    ids = [arm["id"] for arm in cfg["arms"]]
    if (snapshot.get("schema_version") != protocol.SCHEMA_VERSION
            or snapshot.get("config_sha256") != _hash(cfg)
            or snapshot.get("arm_ids") != ids or len(ids) != len(set(ids))
            or not ids or ids[0] != "E00_reference"):
        raise ValueError("Invalid sweep configuration/snapshot for comparison")
    return root, cfg, snapshot, ids


def _stage(root, arm_id, stage, snapshot):
    directory = root / "arms" / arm_id / stage
    receipt = _read(_regular(directory / "completed.json"))
    binding = receipt.get("binding", {})
    if (binding.get("arm_id") != arm_id or binding.get("suite_signature") != snapshot["signature"]
            or binding.get("config_sha256") != snapshot["config_sha256"]
            or binding.get("suite_snapshot_sha256") != protocol.file_hash(root / "snapshot.json")
            or binding.get("source_binding") != snapshot["source_binding"]):
        raise ValueError("Comparison arm does not belong to this suite: " + arm_id)
    receipt = _verify_completed(directory, binding, stage)
    predictions = base.unique_records(_records(directory / "predictions.jsonl"))
    calculated = base.evaluate_records(predictions, receipt["meta"])
    report = _read(directory / "report.json")
    for name in ("counts", "metrics", "checks", "targets_passed"):
        if report.get(name) != calculated[name] or receipt["summary"].get(name) != calculated[name]:
            raise ValueError("Comparison metrics differ from saved predictions: " + arm_id)
    return receipt, report, predictions


def _failure(root, arm_id, stages):
    path = root / "arms" / arm_id / "failure.json"
    failure = _read(_regular(path))
    if (failure.get("arm_id") != arm_id or failure.get("stage") not in stages
            or not failure.get("error")):
        raise ValueError("Missing/invalid technical failure record: " + arm_id)
    return failure, protocol.file_hash(path)


def _rank(rows):
    return sorted(rows, key=lambda row: (-int(row["targets_passed"]),
                -int(row["known_preserved"]), -row["mean_four_metrics"],
                -row["counts"]["known_correct"], row["arm_id"]))


def _entry(arm_id, report, reference):
    mean = sum(report["metrics"][name] or 0. for name in base.TARGETS) / len(base.TARGETS)
    preserved = reference is not None and report["counts"]["known_correct"] >= reference["counts"]["known_correct"]
    return {"arm_id": arm_id, "execution_status": "completed",
            "calibration_status": report["calibration_status"],
            "targets_passed": report["targets_passed"], "known_preserved": bool(preserved),
            "qualified": bool(report["targets_passed"] and preserved),
            "mean_four_metrics": mean, "counts": report["counts"], "metrics": report["metrics"],
            "checks": report["checks"], "research_status": "qualified_on_this_development" if report["targets_passed"] and preserved else "best_effort_unqualified"}


def _paired(reference, selected, meta):
    left = {base._digest(row): row for row in reference}
    right = {base._digest(row): row for row in selected}
    if set(left) != set(right):
        raise ValueError("Cannot compare arms evaluated on different unique images")
    groups, display = defaultdict(list), {}
    changes = []
    for identity in sorted(left):
        before, after = left[identity], right[identity]
        if any(before.get(key) != after.get(key) for key in
               ("status", "split", "true_leaf", "true_parent", "source", "species")):
            raise ValueError("Paired evaluation annotations differ: " + identity)
        status = before["status"]
        species = meta["leaf_names"][int(before["true_leaf"])] if status == "known" else str(before.get("species") or before.get("source"))
        key = (status, species if status == "known" else normalized_name(species))
        display.setdefault(key, species)
        groups[key].append((_correct(before), _correct(after), identity))
        if _correct(before) != _correct(after):
            changes.append({"image_sha256": identity, "path": before.get("path"), "status": status,
                            "species": species, "change": "gained_correct" if _correct(after) else "lost_correct",
                            "before": {key: before[key] for key in ("prediction_type", "parent", "leaf")},
                            "after": {key: after[key] for key in ("prediction_type", "parent", "leaf")}})
    for species in meta["leaf_names"]:
        groups.setdefault(("known", species), [])
        display[("known", species)] = species
    table = []
    for (status, species), values in sorted(groups.items()):
        species = display[(status, species)]
        lost = [identity for b, a, identity in values if b and not a]
        gained = [identity for b, a, identity in values if a and not b]
        table.append({"status": status, "species": species, "sample_count": len(values),
                      "reference_correct": sum(b for b, a, i in values),
                      "arm_correct": sum(a for b, a, i in values),
                      "lost_correct_count": len(lost), "gained_correct_count": len(gained),
                      "net_correct_change": len(gained) - len(lost),
                      "lost_correct_sha256": lost, "gained_correct_sha256": gained,
                      "evidence_status": "not_evaluable" if not values else "insufficient_evidence" if len(values) < 5 else "observed"})
    return {"unit": "unique_image_sha256", "per_species": table, "changed_images": changes,
            "known_lost_correct_count": sum(row["lost_correct_count"] for row in table if row["status"] == "known"),
            "known_gained_correct_count": sum(row["gained_correct_count"] for row in table if row["status"] == "known")}


def _development(root, snapshot, ids):
    loaded, failed, inventory = {}, {}, {}
    for arm_id in ids:
        receipt_path = root / "arms" / arm_id / "calibration" / "completed.json"
        if receipt_path.is_file():
            loaded[arm_id] = _stage(root, arm_id, "calibration", snapshot)
            inventory[arm_id] = {"calibration_completed_sha256": protocol.file_hash(_regular(receipt_path))}
        else:
            failure, digest = _failure(root, arm_id, {"training", "calibration", "dependency"})
            failed[arm_id] = failure
            inventory[arm_id] = {"technical_failure_sha256": digest}
    reference = loaded.get("E00_reference")
    baseline = None if reference is None else reference[1]
    rows = [_entry(arm_id, value[1], baseline) for arm_id, value in loaded.items()]
    if reference is not None:
        for arm_id, value in loaded.items():
            if value[0]["meta"] != reference[0]["meta"]:
                raise ValueError("Comparison taxonomy changed: " + arm_id)
            _paired(reference[2], value[2], reference[0]["meta"])
    ranking = _rank(rows)
    qualified = [row for row in ranking if row["qualified"]]
    chosen = qualified[0]["arm_id"] if qualified else None
    best = ranking[0] if ranking else None
    return {"schema_version": SCHEMA_VERSION, "phase": "development",
            "suite_signature": snapshot["signature"],
            "config_sha256": snapshot["config_sha256"],
            "snapshot_sha256": protocol.file_hash(root / "snapshot.json"),
            "inventory": inventory, "ranking_rule": RANKING,
            "selection_uses_test": False, "test_predictions_read": False,
            "validation_scope": VALIDATION_SCOPE, "confirmatory_validation": False,
            "development_reused_for_method_design": True,
            "qualified_candidate_arm_id": chosen,
            "recommendation_arm_id": chosen or ("E00_reference" if reference is not None else None),
            "recommendation_status": "qualified_on_development" if chosen else "retain_reference_no_qualified_candidate" if reference else "unavailable_reference_failed",
            "best_exploratory_dev_arm": best,
            "exploratory_development_ranking": ranking, "technical_failures": failed}, loaded


def freeze_dev_selection(root_dir):
    """Create once, or verify the exact existing DEV-only decision/inventory."""
    root, cfg, snapshot, ids = _suite(root_dir)
    path = root / "dev_selection.json"
    if not path.exists() and any((root / "arms" / arm_id / "test").exists() for arm_id in ids):
        raise ValueError("DEV selection must be frozen before any TEST stage begins")
    selection, _ = _development(root, snapshot, ids)
    if path.exists():
        if _read(_regular(path)) != selection:
            raise ValueError("Immutable DEV selection or calibration inventory changed")
    else:
        _write(path, selection)
    return selection


def _comparison_csv(root, phase, ranking):
    rows = []
    for rank, row in enumerate(ranking, 1):
        rows.append({"rank": rank, "arm_id": row["arm_id"], "targets_passed": row["targets_passed"],
                     "calibration_status": row["calibration_status"],
                     "qualified_on_test_metrics_only": row.get("qualified_on_test_metrics_only"),
                     "known_preserved": row["known_preserved"], "qualified": row["qualified"],
                     "mean_four_metrics": row["mean_four_metrics"], **row["metrics"], **row["counts"]})
    if rows:
        _csv(root / ("comparison_" + phase + ".csv"), rows)


def _all_comparison(root, cfg, dev, tests, failures, selection):
    rows = []
    for arm in cfg["arms"]:
        arm_id = arm["id"]
        failure = failures.get(arm_id, {})
        training_path = root / "arms" / arm_id / "training" / "completed.json"
        training = _read(_regular(training_path)) if training_path.exists() else {}
        development = dev.get(arm_id)
        test = tests.get(arm_id)
        row = {"arm_id": arm_id, "kind": arm.get("kind", "baseline" if arm_id == "E00_reference" else "finetune"),
               "training_execution": "reference_reused" if arm_id == "E00_reference" else "completed" if training else "failed_or_blocked",
               "optimizer_steps": 0 if arm_id == "E00_reference" else training.get("optimizer_steps"),
               "best_epoch": training.get("best_epoch"),
               "calibration_execution": "completed" if development else "technical_failure",
               "calibration_targets_passed": None if development is None else development[1]["targets_passed"],
               "test_execution": "completed" if test else "technical_failure" if failure.get("stage") == "test" else "blocked_by_technical_failure",
               "test_targets_passed": None if test is None else test[1]["targets_passed"],
               **{"dev_" + name: None if development is None else development[1]["metrics"][name] for name in base.TARGETS},
               **{"test_" + name: None if test is None else test[1]["metrics"][name] for name in base.TARGETS},
               "recommendation": selection["recommendation_arm_id"] == arm_id,
               "failure_reason": failure.get("error", "")}
        rows.append(row)
    _csv(root / "comparison_all.csv", rows)
    return rows


def summarize_suite(root_dir, phase="complete"):
    if phase not in {"development", "complete"}:
        raise ValueError("Comparison phase must be development or complete")
    selection = freeze_dev_selection(root_dir)
    if phase == "development":
        return selection
    root, cfg, snapshot, ids = _suite(root_dir)
    _, dev = _development(root, snapshot, ids)
    tests, failures = {}, dict(selection["technical_failures"])
    for arm_id in dev:
        directory = root / "arms" / arm_id / "test"
        if (directory / "completed.json").is_file():
            tests[arm_id] = _stage(root, arm_id, "test", snapshot)
            receipt = tests[arm_id][0]
            if (receipt.get("calibration_receipt_sha256") != protocol.file_hash(root / "arms" / arm_id / "calibration" / "completed.json")
                    or receipt.get("calibration_router_sha256") != dev[arm_id][0]["artifacts"]["router"]["sha256"]):
                raise ValueError("TEST receipt refers to another calibration: " + arm_id)
        else:
            failures[arm_id], _ = _failure(root, arm_id, {"test"})
    baseline = tests.get("E00_reference")
    ranking = _rank([_entry(arm_id, value[1], None if baseline is None else baseline[1]) for arm_id, value in tests.items()])
    for row in ranking:
        row["research_status"] = "descriptive_test_result"
        row["qualified_on_test_metrics_only"] = row.pop("qualified")
        row["qualified"] = False  # TEST cannot qualify a deployment arm after selection.
    paired, species_table = {}, []
    for name, loaded in (("development", dev), ("test", tests)):
        paired[name] = {}
        if "E00_reference" in loaded:
            reference = loaded["E00_reference"]
            for arm_id, value in loaded.items():
                audit = _paired(reference[2], value[2], reference[0]["meta"])
                paired[name][arm_id] = audit
                for row in audit["per_species"]:
                    species_table.append({"phase": name, "arm_id": arm_id,
                        **{key: val for key, val in row.items() if not key.endswith("sha256")}})
        _write(root / ("paired_" + name + ".json"), paired[name])
    _comparison_csv(root, "development", selection["exploratory_development_ranking"])
    _comparison_csv(root, "test", ranking)
    if species_table:
        _csv(root / "per_species_comparison.csv", species_table)
    all_arms = _all_comparison(root, cfg, dev, tests, failures, selection)
    summary = {"schema_version": SCHEMA_VERSION, "phase": "complete", "workflow_completed": True,
               "validation_scope": VALIDATION_SCOPE, "confirmatory_validation": False,
               "development_reused_for_method_design": True,
               "dev_selection_sha256": protocol.file_hash(root / "dev_selection.json"),
               "dev_selection": selection, "recommendation_arm_id": selection["recommendation_arm_id"],
               "recommendation_status": selection["recommendation_status"],
               "best_exploratory_dev_arm": selection["best_exploratory_dev_arm"],
               "best_exploratory_test_arm": ranking[0] if ranking else None,
               "exploratory_test_ranking": {"selection_uses_test": True, "production_selection": False,
                   "purpose": "descriptive comparison on the already used benchmark", "ranking_rule": RANKING, "arms": ranking},
               "production_recommendation_uses_test": False,
               "technical_failures": failures, "completed_calibration_count": len(dev),
               "completed_test_count": len(tests), "declared_arm_count": len(ids),
               "all_arms": all_arms,
               "paired_correctness": paired}
    _write(root / "summary.json", summary)
    lines = ["# Fine-tuning sweep comparison", "", "Development recommendation: " + str(summary["recommendation_arm_id"]),
             "Status: " + summary["recommendation_status"], "",
             "Every completed arm retains its own calibrated router, including failed research gates.",
             "TEST ranking is descriptive and never changes the frozen DEV recommendation.",
             "This reused benchmark does not provide independent confirmatory validation.", "",
             "Completed calibrations: %d; completed tests: %d; declared arms: %d." % (len(dev), len(tests), len(ids))]
    (root / "comparison_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
