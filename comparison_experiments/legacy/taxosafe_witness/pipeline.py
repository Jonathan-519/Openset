"""Reuse verified v4 caches. No backbone/adapter retraining and no old-run writes."""
import copy
import numpy as np
from taxosafe_hier import pipeline as old
from taxosafe_visual import residual
from taxosafe_visual.runtime import resolve, read_json, write_json, write_records, assert_disjoint
from . import core, calibration


def source(plan):
    return read_json(resolve(plan["source_plan"]))


def fit(plan):
    src = source(plan)
    data, rows = old.load_cache(resolve(src["suite"]), "train")
    models, report = core.fit_all(data, rows, src["taxonomy"], plan["settings"], plan["seed"])
    folder = resolve(plan["suite"])
    write_json(folder / "models.json", models)
    write_json(folder / "training_report.json", report)


def anchored_evidence(plan, split):
    """Keep the pre-existing identity candidate; witness scores cannot reroute it."""
    src = source(plan)
    folder = resolve(src["suite"])
    train, train_rows = old.load_cache(folder, "train")
    query, rows = old.load_cache(folder, split)
    forbidden = [r["image_sha256"] for r in train_rows]
    if split == "test":
        _, validation = old.load_cache(folder, "val")
        forbidden += [r["image_sha256"] for r in validation]
    assert_disjoint(rows, forbidden, split)
    root = old.frozen_root(src, query, rows)
    # Reuse the exact published classification implementation, on CPU.
    out = old.scores(src, query, train, train_rows, root, "identity", "cpu")
    anchor = old.apply(src, root, out, read_json(folder / "artifacts/identity/calibration.json"), "coverage", "identity")
    support, labels = old.memory_subset(src, train, train_rows)
    evidence = core.evidence(core.prepare(query, plan["settings"]), core.prepare(support, plan["settings"]),
                             labels, src["taxonomy"]["leaf_to_parent"], out["parent"],
                             plan["settings"], candidates=out["leaf"])
    if not np.array_equal(evidence["leaf"], out["leaf"]):
        raise AssertionError("Verifier changed the classifier")
    # Test-time numerical/platform differences must not silently change labels.
    if split == "test":
        saved_path = folder / "artifacts/identity/test/coverage/predictions.jsonl"
        if saved_path.is_file():
            previous = {r["image_sha256"]: r for r in old.read_rows(saved_path)}
            for row in anchor:
                previous_row = previous[row["image_sha256"]]
                for field in ("candidate_parent", "candidate_leaf"):
                    if row[field] != previous_row[field]:
                        raise AssertionError("Frozen classifier differs from v4 record: " + row["image_sha256"])
    return rows, anchor, evidence


def apply(plan, anchor, e, values, thresholds, variant):
    out = {"score": values, "leaf": e["leaf"], "parent": e["parent"], "neighbour": e["neighbour"]}
    cal = {"profiles": {"known_budget": thresholds}}
    result = residual.apply(anchor, out, cal, plan["taxonomy"], "known_budget", "witness_" + variant)
    for before, after in zip(anchor, result):
        for field in ("candidate_parent", "candidate_leaf", "root_knownness_score", "root_gate_margin"):
            if before[field] != after[field]:
                raise AssertionError("Classifier or root changed: " + field)
        if (before["prediction_type"] == "global_unknown") != (after["prediction_type"] == "global_unknown"):
            raise AssertionError("Root rejection changed")
    return result


def metrics(predictions):
    m = old.audit_metrics(predictions)
    extra = [r for r in predictions if r["status"] == "extra"]
    m["extra"]["final_known_false_acceptance"] = float(np.mean([r["prediction_type"] == "known" for r in extra])) if extra else None
    summary = old.compact_metrics(m)
    summary["extra_final_known_false_acceptance"] = m["extra"]["final_known_false_acceptance"]
    summary["known_correct_acceptance_macro"] = float(np.mean([
        v["end_to_end_leaf_accuracy"] for v in m["per_known_leaf"].values()]))
    return m, summary


def known_metrics(predictions):
    correct = np.asarray([r["candidate_parent"] == r["true_parent"] and r["candidate_leaf"] == r["true_leaf"]
                          for r in predictions])
    accepted = np.asarray([r["prediction_type"] == "known" for r in predictions])
    labels = np.asarray([r["true_leaf"] for r in predictions])
    return {"known_count": len(predictions), "closed_routed_accuracy": float(correct.mean()),
            "known_end_to_end_leaf_accuracy": float((correct & accepted).mean()),
            "known_correct_acceptance_macro": float(np.mean([(correct & accepted)[labels == c].mean()
                                                              for c in np.unique(labels)]))}


def legacy_metrics(plan, anchor, e, indices=None):
    src = source(plan)
    previous = read_json(resolve(src["suite"]) / "artifacts/identity/calibration.json")
    out = {"score": np.asarray([r["child_knownness_score"] for r in anchor]),
           "leaf": e["leaf"], "parent": e["parent"], "neighbour": e["neighbour"]}
    summaries = {}
    for profile in ("coverage", "balanced", "protected"):
        predictions = old.apply(src, anchor, out, previous, profile, "identity")
        if indices is not None:
            predictions = [predictions[i] for i in indices]
        _, summaries[profile] = metrics(predictions)
    return summaries


def gate(reference, full):
    tol = 1e-12
    def compare(key, direction):
        a, b = reference.get(key), full.get(key)
        return None if a is None or b is None else bool(direction * (b - a) >= -tol)
    result = {"closed_accuracy_unchanged": bool(abs(full["closed_routed_accuracy"] - reference["closed_routed_accuracy"]) < tol),
              "known_correct_acceptance_not_lower": compare("known_end_to_end_leaf_accuracy", 1),
              "known_macro_correct_acceptance_not_lower": compare("known_correct_acceptance_macro", 1),
              "intra_false_acceptance_not_higher": compare("intra_oser", -1),
              "fine_grained_auc_not_lower": compare("intra_macro_parent_species_auroc", 1),
              "far_unknown_final_false_acceptance_not_higher": compare("extra_final_known_false_acceptance", -1)}
    result["all_observed_checks_pass"] = all(v is True for v in result.values())
    result["meaning"] = "Observed comparison only; never select a variant or rewrite thresholds on test"
    return result


def calibrate(plan):
    rows, anchor, e = anchored_evidence(plan, "val")
    src, folder = source(plan), resolve(plan["suite"])
    held = old.validation_indices(src, rows)
    cal_ids, audit_ids = calibration.partition(rows, held, plan["settings"]["calibration_fraction"], plan["seed"])
    models = read_json(folder / "models.json")
    scores = {v: core.score(models[v], e) for v in plan["settings"]["variants"]}
    nparents, retention = len(plan["taxonomy"]["parent_names"]), plan["settings"]["known_retention"]
    reference_tau, reference_info = calibration.reference_thresholds(scores["identity"], rows, anchor, cal_ids, nparents, retention)
    reference_accept = np.asarray([scores["identity"][i] >= reference_tau[str(r["candidate_parent"])] for i, r in enumerate(anchor)])
    fitted = {"identity": {"thresholds": reference_tau, "parents": reference_info, "unknowns_used": False}}
    for variant in plan["settings"]["variants"]:
        if variant != "identity":
            fitted[variant] = calibration.matched_thresholds(scores[variant], rows, anchor, cal_ids,
                                                             reference_accept, nparents, retention)
    report = {"test_loaded": False, "verifier_model_selected_on": "whole-species folds inside TRAIN",
              "child_thresholds_use_unknowns": False, "calibration_count": len(cal_ids),
              "audit_count": len(audit_ids), "calibration_image_hashes": [rows[i]["image_sha256"] for i in cal_ids],
              "audit_image_hashes": [rows[i]["image_sha256"] for i in audit_ids], "variants": {},
              "audit_scope": "Disjoint from NEW child calibration. Frozen upstream root used the original validation split; not independent whole-system validation."}
    for variant in plan["settings"]["variants"]:
        predictions = apply(plan, anchor, e, scores[variant], fitted[variant]["thresholds"], variant)
        _, audit = metrics([predictions[i] for i in audit_ids])
        cal = known_metrics([predictions[i] for i in cal_ids])
        report["variants"][variant] = {"audit": audit, "calibration": cal}
    report["gate"] = gate(report["variants"]["identity"]["audit"], report["variants"]["full"]["audit"])
    report["legacy_v4"] = legacy_metrics(plan, anchor, e, audit_ids)
    report["legacy_scope"] = "Original fixed v4 thresholds; their calibration included the original held validation partition"
    write_json(folder / "calibration.json", fitted)
    write_json(folder / "validation_report.json", report)
    # Raw evidence is exported for diagnosis; no need to request large caches.
    exported = []
    for i in sorted(set(cal_ids + audit_ids)):
        row = copy.deepcopy(anchor[i])
        row["witness_evidence"] = dict(zip(core.FEATURES, e["x"][i].tolist()))
        row["evidence_valid"] = e["valid"][i].tolist()
        row["verifier_scores"] = {v: float(scores[v][i]) for v in scores}
        row["child_partition"] = "calibration" if i in cal_ids else "audit"
        exported.append(row)
    write_records(folder / "validation_evidence.jsonl", exported)
    print("Child audit gate:", report["gate"], flush=True)


def test(plan):
    rows, anchor, e = anchored_evidence(plan, "test")
    folder = resolve(plan["suite"])
    models, calibration_state = read_json(folder / "models.json"), read_json(folder / "calibration.json")
    report = {"seed": plan["seed"], "primary_variant": "full", "rows": [],
              "classifier": "frozen_v4_identity", "classification_predictions_identical": True,
              "protocol": "Development test previously inspected; no SOTA claim"}
    for variant in plan["settings"]["variants"]:
        scores = core.score(models[variant], e)
        predictions = apply(plan, anchor, e, scores, calibration_state[variant]["thresholds"], variant)
        for i, r in enumerate(predictions):
            r["witness_evidence"] = dict(zip(core.FEATURES, e["x"][i].tolist()))
            r["evidence_valid"] = e["valid"][i].tolist()
        m, summary = metrics(predictions)
        out = folder / "test" / variant
        write_json(out / "metrics.json", m)
        write_records(out / "predictions.jsonl", predictions)
        report["rows"].append({"method": variant, **summary})
    reference = next(r for r in report["rows"] if r["method"] == "identity")
    full = next(r for r in report["rows"] if r["method"] == "full")
    report["gate"] = gate(reference, full)
    report["legacy_v4"] = legacy_metrics(plan, anchor, e)
    report["gate_against_legacy_coverage"] = gate(report["legacy_v4"]["coverage"], full)
    write_json(folder / "summary.json", report)
    print("Test gate:", report["gate"], flush=True)
