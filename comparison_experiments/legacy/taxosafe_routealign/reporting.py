"""Freeze guarded DEV recommendations, then describe every completed TEST arm.

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

SCHEMA_VERSION = "taxosafe_routealign_comparison_v1"
ARM_IDS = ("A00_reference", "A01_evidence_anchor", "A02_proximity", "A03_combined", "A04_parent_rerank")
RANKING = ("qualified development candidate first, all four gates passed, zero paired known-correct losses, "
           "mean of four metrics descending, known correct count descending, arm ID ascending")
TEST_RANKING = ("all four gates passed, zero paired known-correct losses, "
                "mean of four metrics descending, known correct count descending, arm ID ascending; descriptive only")


def _suite(root):
    root = Path(root)
    cfg = _read(_regular(root / "config.json"))
    snapshot = _read(_regular(root / "snapshot.json"))
    ids = [arm["id"] for arm in cfg["arms"]]
    if (snapshot.get("schema_version") != protocol.SCHEMA_VERSION
            or snapshot.get("config_sha256") != _hash(cfg)
            or snapshot.get("arm_ids") != ids or len(ids) != len(set(ids))
            or tuple(ids) != ARM_IDS):
        raise ValueError("Invalid route-alignment configuration/snapshot for comparison")
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
    if ("crossfit_audit" in report
            and report["crossfit_audit"] != receipt["summary"].get("crossfit_audit")):
        raise ValueError("Cross-fit audit differs between report and calibration receipt: " + arm_id)
    return receipt, report, predictions


def _failure(root, arm_id, stages):
    path = root / "arms" / arm_id / "failure.json"
    failure = _read(_regular(path))
    if (failure.get("arm_id") != arm_id or failure.get("stage") not in stages
            or not failure.get("error")):
        raise ValueError("Missing/invalid technical failure record: " + arm_id)
    return failure, protocol.file_hash(path)


def _rank(rows):
    return sorted(rows, key=lambda row: (-int(row["qualified"]), -int(row["targets_passed"]),
                -int(row["known_preserved"]), -row["mean_four_metrics"],
                -row["counts"]["known_correct"], row["arm_id"]))


def _source_macro(records, meta):
    """Equal-weight observed known classes / unknown sources, never image counts."""
    groups, display = defaultdict(list), {}
    for row in base.unique_records(records):
        status = row["status"]
        if status == "known":
            key = (status, str(int(row["true_leaf"])))
            name = meta["leaf_names"][int(row["true_leaf"])]
        else:
            name = str(row.get("source") or row.get("species") or "")
            if not name:
                raise ValueError("Unknown-source macro metrics require a source name")
            key = (status, normalized_name(name))
        display.setdefault(key, name)
        groups[key].append(row)
    for index, name in enumerate(meta["leaf_names"]):
        key = ("known", str(index))
        groups.setdefault(key, [])
        display[key] = name
    table = []
    for key in sorted(groups):
        rows = groups[key]
        n = len(rows)
        correct = sum(_correct(row) for row in rows)
        table.append({"status": key[0], "source_key": key[1], "source": display[key],
                      "sample_count": n, "correct_count": correct,
                      "correct_rate": correct / n if n else None,
                      "evidence_status": "not_evaluable" if not n else "insufficient_evidence" if n < 5 else "observed"})
    macros, counts = {}, {}
    for status in ("known", "intra", "extra"):
        observed = [row["correct_rate"] for row in table if row["status"] == status and row["sample_count"]]
        macros[status] = sum(observed) / len(observed) if observed else None
        counts[status] = len(observed)
    return {"unit": "unique_image_sha256", "macro_definition": "unweighted mean over observed known leaves or normalized unknown sources",
            "source_macro": macros, "observed_source_counts": counts, "per_source": table}


def _entry(arm_id, value, reference, phase="development", crossfit=None):
    receipt, report, predictions = value
    mean = sum(report["metrics"][name] or 0. for name in base.TARGETS) / len(base.TARGETS)
    audit = None
    if reference is not None:
        if receipt["meta"] != reference[0]["meta"]:
            raise ValueError("Comparison taxonomy changed: " + arm_id)
        audit = _paired(reference[2], predictions, reference[0]["meta"])
    lost = None if audit is None else audit["known_lost_correct_count"]
    gained = None if audit is None else audit["known_gained_correct_count"]
    preserved = audit is not None and lost == 0
    count_preserved = reference is not None and report["counts"]["known_correct"] >= reference[1]["counts"]["known_correct"]
    required = arm_id != ARM_IDS[0]
    if crossfit is None:
        crossfit = receipt["summary"].get("crossfit_audit")
    crossfit_passed = isinstance(crossfit, dict) and crossfit.get("passed") is True
    observed_metrics_passed = bool(report["targets_passed"] and preserved)
    qualified = bool(phase == "development" and required and observed_metrics_passed and crossfit_passed)
    reasons = []
    if not report["targets_passed"]:
        reasons.append("four_metric_gates_not_passed")
    if reference is None:
        reasons.append("reference_unavailable")
    elif not preserved:
        reasons.append("paired_known_correct_losses")
    if required and not crossfit_passed:
        reasons.append("crossfit_audit_missing_or_failed")
    if not required:
        reasons.append("reference_is_retention_option_not_new_candidate")
    macro = _source_macro(predictions, receipt["meta"])
    return {"arm_id": arm_id, "execution_status": "completed",
            "calibration_status": report["calibration_status"],
            "targets_passed": report["targets_passed"], "known_preserved": bool(preserved),
            "known_count_preserved": bool(count_preserved),
            "known_lost_correct_count": lost, "known_gained_correct_count": gained,
            "crossfit_required": required, "crossfit_audit_passed": crossfit_passed if required else None,
            "qualified": qualified, "qualification_reasons": reasons,
            "qualified_on_test_metrics_only": observed_metrics_passed if phase == "test" else None,
            "mean_four_metrics": mean, "counts": report["counts"], "metrics": report["metrics"],
            "checks": report["checks"], "source_macro": macro["source_macro"],
            "observed_source_counts": macro["observed_source_counts"],
            "research_status": "descriptive_test_result" if phase == "test" else "qualified_on_this_development" if qualified else "best_effort_unqualified"}


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
    reference = loaded.get(ARM_IDS[0])
    rows = [_entry(arm_id, value, reference) for arm_id, value in loaded.items()]
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
            "recommendation_arm_id": chosen or (ARM_IDS[0] if reference is not None else None),
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
                     "known_preserved": row["known_preserved"], "known_count_preserved": row["known_count_preserved"],
                     "known_lost_correct_count": row["known_lost_correct_count"],
                     "known_gained_correct_count": row["known_gained_correct_count"],
                     "crossfit_audit_passed": row["crossfit_audit_passed"], "qualified": row["qualified"],
                     "mean_four_metrics": row["mean_four_metrics"],
                     **{status + "_source_macro_accuracy": row["source_macro"][status] for status in ("known", "intra", "extra")},
                     **row["metrics"], **row["counts"]})
    if rows:
        _csv(root / ("comparison_" + phase + ".csv"), rows)


def _all_comparison(root, cfg, dev, tests, failures, selection, test_ranking):
    rows = []
    dev_entries = {row["arm_id"]: row for row in selection["exploratory_development_ranking"]}
    test_entries = {row["arm_id"]: row for row in test_ranking}
    parent_path = root / "arms" / "A01_evidence_anchor" / "training" / "completed.json"
    parent = _read(_regular(parent_path)) if parent_path.exists() else {}
    for arm in cfg["arms"]:
        arm_id = arm["id"]
        failure = failures.get(arm_id, {})
        trained_here = arm_id == "A01_evidence_anchor"
        training = parent if trained_here else {}
        weight_source = "own" if trained_here else "source" if arm_id in ("A00_reference", "A02_proximity") else "A01_evidence_anchor"
        execution = ("completed" if training else "failed_or_blocked") if trained_here else (
            "reference_reused" if weight_source == "source" else "A01_weights_reused")
        development = dev.get(arm_id)
        test = tests.get(arm_id)
        dev_entry, test_entry = dev_entries.get(arm_id), test_entries.get(arm_id)
        row = {"arm_id": arm_id, "kind": arm.get("kind", "baseline" if arm_id == "A00_reference" else "finetune" if trained_here else "router"),
               "weight_source": weight_source, "training_execution": execution,
               "optimizer_steps": training.get("optimizer_steps") if trained_here else 0,
               "best_epoch": training.get("best_epoch"),
               "weight_source_best_epoch": parent.get("best_epoch") if weight_source in ("own", "A01_evidence_anchor") else None,
               "calibration_execution": "completed" if development else "technical_failure",
               "calibration_targets_passed": None if development is None else development[1]["targets_passed"],
               "calibration_crossfit_audit_passed": None if dev_entry is None else dev_entry["crossfit_audit_passed"],
               "dev_known_lost_correct_count": None if dev_entry is None else dev_entry["known_lost_correct_count"],
               "dev_candidate_qualified": False if dev_entry is None else dev_entry["qualified"],
               "test_execution": "completed" if test else "technical_failure" if failure.get("stage") == "test" else "blocked_by_technical_failure",
               "test_targets_passed": None if test is None else test[1]["targets_passed"],
               "test_known_lost_correct_count": None if test_entry is None else test_entry["known_lost_correct_count"],
               **{"dev_" + name: None if development is None else development[1]["metrics"][name] for name in base.TARGETS},
               **{"test_" + name: None if test is None else test[1]["metrics"][name] for name in base.TARGETS},
               **{"dev_" + status + "_source_macro_accuracy": None if dev_entry is None else dev_entry["source_macro"][status] for status in ("known", "intra", "extra")},
               **{"test_" + status + "_source_macro_accuracy": None if test_entry is None else test_entry["source_macro"][status] for status in ("known", "intra", "extra")},
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
    baseline = tests.get(ARM_IDS[0])
    ranking = _rank([_entry(arm_id, value, baseline, phase="test",
        crossfit=dev[arm_id][0]["summary"].get("crossfit_audit")) for arm_id, value in tests.items()])
    paired, species_table, macro_tables, macro_source_rows = {}, [], {}, []
    for name, loaded in (("development", dev), ("test", tests)):
        paired[name] = {}
        macro_tables[name] = {}
        for arm_id, value in loaded.items():
            macro = _source_macro(value[2], value[0]["meta"])
            macro_tables[name][arm_id] = macro
            for row in macro["per_source"]:
                macro_source_rows.append({"phase": name, "arm_id": arm_id, **row})
        _write(root / ("source_macro_" + name + ".json"), macro_tables[name])
        if ARM_IDS[0] in loaded:
            reference = loaded[ARM_IDS[0]]
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
    if macro_source_rows:
        _csv(root / "per_source_comparison.csv", macro_source_rows)
    all_arms = _all_comparison(root, cfg, dev, tests, failures, selection, ranking)
    summary = {"schema_version": SCHEMA_VERSION, "phase": "complete", "workflow_completed": True,
               "validation_scope": VALIDATION_SCOPE, "confirmatory_validation": False,
               "development_reused_for_method_design": True,
               "dev_selection_sha256": protocol.file_hash(root / "dev_selection.json"),
               "dev_selection": selection, "recommendation_arm_id": selection["recommendation_arm_id"],
               "recommendation_status": selection["recommendation_status"],
               "best_exploratory_dev_arm": selection["best_exploratory_dev_arm"],
               "best_exploratory_test_arm": ranking[0] if ranking else None,
               "exploratory_test_ranking": {"selection_uses_test": True, "production_selection": False,
                   "purpose": "descriptive comparison on the already used benchmark", "ranking_rule": TEST_RANKING, "arms": ranking},
               "production_recommendation_uses_test": False,
               "technical_failures": failures, "completed_calibration_count": len(dev),
               "completed_test_count": len(tests), "declared_arm_count": len(ids),
               "all_arms": all_arms,
               "paired_correctness": paired, "source_macro": macro_tables}
    _write(root / "summary.json", summary)
    lines = ["# Route-alignment comparison", "", "Development recommendation: " + str(summary["recommendation_arm_id"]),
             "Status: " + summary["recommendation_status"], "",
             "Every completed arm retains its own calibrated router, including failed research gates.",
             "New DEV recommendations require all metric gates, zero paired known-correct losses, and a passed cross-fit audit.",
             "Only A01 performs optimizer updates; A02 reuses reference weights and A03/A04 reuse A01 weights.",
             "TEST ranking is descriptive and never changes the frozen DEV recommendation.",
             "This reused benchmark does not provide independent confirmatory validation.", "",
             "Completed calibrations: %d; completed tests: %d; declared arms: %d." % (len(dev), len(tests), len(ids))]
    (root / "comparison_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
