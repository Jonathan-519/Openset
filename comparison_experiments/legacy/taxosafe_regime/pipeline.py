"""Frozen v4 classifier with TRAIN-only dual semantic/morphology rejection."""
import copy
import numpy as np
from taxosafe_hier import pipeline as old
from taxosafe_visual import residual
from taxosafe_visual.runtime import resolve, read_json, write_json, write_records, assert_disjoint
from taxosafe_witness import core as witness_core
from taxosafe_witness import pipeline as witness_pipeline
from . import core

def source(plan): return read_json(resolve(plan["source_plan"]))

def fit(plan):
    src = source(plan); data, rows = old.load_cache(resolve(src["suite"]), "train")
    models, calibration, report = core.fit_all(data, rows, src["taxonomy"], plan["settings"], plan["seed"])
    folder = resolve(plan["suite"])
    write_json(folder / "models.json", models)
    write_json(folder / "train_calibration.json", calibration)
    write_json(folder / "training_report.json", report)

def anchored_evidence(plan, split):
    src = source(plan); source_folder = resolve(src["suite"])
    train, train_rows = old.load_cache(source_folder, "train")
    query, rows = old.load_cache(source_folder, split)
    forbidden = [r["image_sha256"] for r in train_rows]
    if split == "test":
        _, validation = old.load_cache(source_folder, "val")
        forbidden += [r["image_sha256"] for r in validation]
    assert_disjoint(rows, forbidden, split)
    root = old.frozen_root(src, query, rows)
    identity = old.scores(src, query, train, train_rows, root, "identity", "cpu")
    anchor = old.apply(src, root, identity,
                       read_json(source_folder / "artifacts/identity/calibration.json"),
                       "coverage", "identity")
    report = read_json(resolve(plan["suite"]) / "training_report.json")
    support = np.asarray(report["split"]["support"], int)
    hashes = [train_rows[i]["image_sha256"] for i in support]
    if hashes != report["support_image_hashes"]:
        raise ValueError("Recorded regime-adaptive reference bank changed")
    pq, pt = core.prepare(query, plan["settings"]), core.prepare(train, plan["settings"])
    evidence = core.evidence(pq, core.subset(pt, support),
                             [train_rows[i]["true_leaf"] for i in support],
                             src["taxonomy"]["leaf_to_parent"], identity["parent"],
                             plan["settings"], candidates=identity["leaf"])
    if not np.array_equal(evidence["leaf"], identity["leaf"]):
        raise AssertionError("Verifier changed the classifier")
    return rows, anchor, evidence

def apply(plan, anchor, evidence, pvalues, variant):
    alpha = float(plan["settings"]["conformal_alpha"])
    threshold = float(np.nextafter(alpha, np.inf))
    thresholds = {str(i): threshold for i in range(len(plan["taxonomy"]["parent_names"]))}
    out = {"score": pvalues, "leaf": evidence["leaf"], "parent": evidence["parent"],
           "neighbour": evidence["neighbour"]}
    result = residual.apply(anchor, out, {"profiles": {"conformal": thresholds}},
                            plan["taxonomy"], "conformal", "regime_hierarchical_" + variant)
    mapping = np.asarray(plan["taxonomy"]["leaf_to_parent"], int)
    branch_sizes = np.bincount(mapping, minlength=len(plan["taxonomy"]["parent_names"]))
    for i, before in enumerate(anchor):
        if branch_sizes[int(before["candidate_parent"])] == 1:
            result[i] = copy.deepcopy(before)
            result[i]["regime_policy"] = "legacy_coverage_singleton"
        else:
            result[i]["regime_policy"] = "hierarchical_dual_multi_leaf"
    for before, after in zip(anchor, result):
        for field in ("candidate_parent", "candidate_leaf", "root_knownness_score", "root_gate_margin"):
            if before[field] != after[field]: raise AssertionError("Classifier or root changed: " + field)
        if (before["prediction_type"] == "global_unknown") != (after["prediction_type"] == "global_unknown"):
            raise AssertionError("Root rejection changed")
    return result

def _variant_predictions(plan, anchor, evidence, models, state, variant):
    raw = witness_core.score(models[variant], evidence)
    p = core.pvalues(raw, evidence["parent"], evidence["leaf"], state, variant)
    return raw, p, apply(plan, anchor, evidence, p, variant)

def calibrate(plan):
    rows, anchor, evidence = anchored_evidence(plan, "val")
    src, folder = source(plan), resolve(plan["suite"]); audit = old.validation_indices(src, rows)
    models, state = read_json(folder / "models.json"), read_json(folder / "train_calibration.json")
    report = {"test_loaded": False, "verifier_model_selected_on": "known TRAIN fit only",
              "child_thresholds_use_unknowns": False, "child_thresholds_use_validation": False,
              "train_calibration_count": len(state["calibration_query_ids"]),
              "audit_count": len(audit), "audit_image_hashes": [rows[i]["image_sha256"] for i in audit],
              "variants": {}, "audit_scope": "Original held validation partition; development audit only."}
    raw_scores, pvals = {}, {}
    for variant in plan["settings"]["variants"]:
        raw, p, predictions = _variant_predictions(plan, anchor, evidence, models, state, variant)
        raw_scores[variant], pvals[variant] = raw, p
        _, summary = witness_pipeline.metrics([predictions[i] for i in audit])
        report["variants"][variant] = {"audit": summary}
    report["legacy_v4"] = witness_pipeline.legacy_metrics(plan, anchor, evidence, audit)
    report["gate_against_legacy_coverage"] = witness_pipeline.gate(
        report["legacy_v4"]["coverage"], report["variants"]["full"]["audit"])
    report["gate_against_morphology_only"] = witness_pipeline.gate(
        report["variants"]["morphology_only"]["audit"], report["variants"]["full"]["audit"])
    write_json(folder / "validation_report.json", report)
    exported = []
    for i in audit:
        row = copy.deepcopy(anchor[i])
        row["dual_evidence"] = dict(zip(core.FEATURES, evidence["x"][i].tolist()))
        row["evidence_valid"] = evidence["valid"][i].tolist()
        row["verifier_scores"] = {v: float(raw_scores[v][i]) for v in raw_scores}
        row["conformal_pvalues"] = {v: float(pvals[v][i]) for v in pvals}
        row["decision_alpha"] = float(plan["settings"]["conformal_alpha"])
        row["child_partition"] = "validation_audit_only"; exported.append(row)
    write_records(folder / "validation_evidence.jsonl", exported)
    print("Validation gate against legacy coverage:", report["gate_against_legacy_coverage"], flush=True)

def test(plan):
    rows, anchor, evidence = anchored_evidence(plan, "test"); folder = resolve(plan["suite"])
    models, state = read_json(folder / "models.json"), read_json(folder / "train_calibration.json")
    report = {"seed": plan["seed"], "primary_variant": "full", "rows": [],
              "classifier": "frozen_v4_identity", "classification_predictions_identical": True,
              "threshold_source": "known TRAIN conformal split",
              "protocol": "Development test previously inspected; no SOTA claim"}
    for variant in plan["settings"]["variants"]:
        raw, p, predictions = _variant_predictions(plan, anchor, evidence, models, state, variant)
        for i, row in enumerate(predictions):
            row["dual_evidence"] = dict(zip(core.FEATURES, evidence["x"][i].tolist()))
            row["evidence_valid"] = evidence["valid"][i].tolist()
            row["verifier_raw_score"], row["conformal_pvalue"] = float(raw[i]), float(p[i])
        metrics, summary = witness_pipeline.metrics(predictions); out = folder / "test" / variant
        write_json(out / "metrics.json", metrics); write_records(out / "predictions.jsonl", predictions)
        report["rows"].append({"method": variant, **summary})
    full = next(r for r in report["rows"] if r["method"] == "full")
    report["legacy_v4"] = witness_pipeline.legacy_metrics(plan, anchor, evidence)
    report["gate_against_legacy_coverage"] = witness_pipeline.gate(report["legacy_v4"]["coverage"], full)
    write_json(folder / "summary.json", report)
    print("Test gate against legacy coverage:", report["gate_against_legacy_coverage"], flush=True)
