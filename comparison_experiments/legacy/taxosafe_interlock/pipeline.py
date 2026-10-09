"""Frozen v4 classifier with dual-conformal semantic hysteresis."""
import copy
from taxosafe_regime import pipeline as regime
from taxosafe_visual.runtime import resolve, read_json, write_json, write_records
from taxosafe_witness import core as witness_core
from taxosafe_witness import pipeline as witness_pipeline
from . import core

source = regime.source
anchored_evidence = regime.anchored_evidence


def fit(plan):
    regime.fit(plan)
    folder = resolve(plan["suite"])
    report = read_json(folder / "training_report.json")
    report["decision_policy"] = "dual-conformal semantic hysteresis around the frozen v4 decision"
    report["leaf_evidence_role"] = "joint veto evidence below the lower boundary and joint rescue evidence above the upper boundary"
    report["iteration_provenance"] = "v9 designed after inspecting v8 development validation; requires fresh confirmatory data"
    write_json(folder / "training_report.json", report)


def _consensus(record):
    return (int(record["global_pred_leaf"]) == int(record["candidate_leaf"]) and
            int(record["text_pred_parent"]) == int(record["candidate_parent"]))


def apply(plan, anchor, evidence, parent_p, leaf_p, variant):
    """Apply a two-sided hysteresis interlock without changing base scores."""
    import numpy as np
    parent_p = np.asarray(parent_p, dtype=float)
    leaf_p = np.asarray(leaf_p, dtype=float)
    if parent_p.shape != (len(anchor),) or leaf_p.shape != (len(anchor),):
        raise ValueError("P-value/prediction length mismatch")
    if not np.isfinite(parent_p).all() or not np.isfinite(leaf_p).all():
        raise ValueError("Non-finite p-value")
    alpha = float(plan["settings"]["conformal_alpha"])
    veto_leaf_cutoff = float(plan["settings"]["veto_leaf_multiplier"]) * alpha
    rescue_parent_cutoff = float(plan["settings"]["rescue_parent_multiplier"]) * alpha
    rescue_leaf_cutoff = float(plan["settings"]["rescue_leaf_multiplier"]) * alpha
    cuts = (alpha, veto_leaf_cutoff, rescue_parent_cutoff, rescue_leaf_cutoff)
    if not 0.0 < alpha < 1.0 or not all(0.0 < x < 1.0 for x in cuts):
        raise ValueError("All interlock cutoffs must lie strictly inside (0,1)")
    result = copy.deepcopy(anchor)
    mapping = np.asarray(plan["taxonomy"]["leaf_to_parent"], dtype=int)
    branch_sizes = np.bincount(mapping, minlength=len(plan["taxonomy"]["parent_names"]))
    for i, (before, after) in enumerate(zip(anchor, result)):
        parent = int(before["candidate_parent"])
        leaf = int(before["candidate_leaf"])
        if branch_sizes[parent] == 1:
            result[i] = copy.deepcopy(before)
            after = result[i]
            after["interlock_policy"] = "legacy_coverage_singleton"
            after["interlock_parent_pvalue"] = float(parent_p[i])
            after["interlock_leaf_pvalue"] = float(leaf_p[i])
            after["interlock_rescued"] = False
            after["interlock_vetoed"] = False
        else:
            consensus = _consensus(before)
            legacy_accept = before["prediction_type"] == "known"
            eligible = before["prediction_type"] != "global_unknown"
            veto = bool(eligible and legacy_accept and parent_p[i] <= alpha and
                        leaf_p[i] <= veto_leaf_cutoff and not consensus)
            rescue = bool(eligible and not legacy_accept and
                          parent_p[i] > rescue_parent_cutoff and
                          leaf_p[i] > rescue_leaf_cutoff and consensus)
            accept = bool((legacy_accept and not veto) or rescue)
            after["interlock_policy"] = "dual_conformal_semantic_hysteresis"
            after["interlock_parent_pvalue"] = float(parent_p[i])
            after["interlock_leaf_pvalue"] = float(leaf_p[i])
            after["interlock_parent_alpha"] = alpha
            after["interlock_veto_leaf_cutoff"] = veto_leaf_cutoff
            after["interlock_rescue_parent_cutoff"] = rescue_parent_cutoff
            after["interlock_rescue_leaf_cutoff"] = rescue_leaf_cutoff
            after["interlock_semantic_consensus"] = consensus
            after["interlock_rescued"] = rescue
            after["interlock_vetoed"] = veto
            after["interlock_variant"] = variant
            if eligible:
                after["prediction_type"] = "known" if accept else "intra_unknown"
                after["leaf"] = leaf if accept else None
                after["leaf_name"] = plan["taxonomy"]["leaf_names"][leaf] if accept else None
                after["parent"] = parent
                after["parent_name"] = plan["taxonomy"]["parent_names"][parent]
        for field in ("candidate_parent", "candidate_leaf", "root_knownness_score", "root_gate_margin"):
            if before[field] != after[field]:
                raise AssertionError("Classifier or root changed: " + field)
        if (before["prediction_type"] == "global_unknown") != (after["prediction_type"] == "global_unknown"):
            raise AssertionError("Root rejection changed")
    return result


def _variant_predictions(plan, anchor, evidence, models, state, variant):
    raw = witness_core.score(models[variant], evidence)
    parent_p, leaf_p = core.pvalue_components(raw, evidence["parent"], evidence["leaf"], state, variant)
    return raw, parent_p, leaf_p, apply(plan, anchor, evidence, parent_p, leaf_p, variant)


def _policy_counts(predictions):
    return {"rescued": sum(bool(r.get("interlock_rescued")) for r in predictions),
            "vetoed": sum(bool(r.get("interlock_vetoed")) for r in predictions),
            "singleton_fallback": sum(r.get("interlock_policy") == "legacy_coverage_singleton" for r in predictions)}


def calibrate(plan):
    rows, anchor, evidence = anchored_evidence(plan, "val")
    src, folder = source(plan), resolve(plan["suite"])
    audit = regime.old.validation_indices(src, rows)
    models = read_json(folder / "models.json")
    state = read_json(folder / "train_calibration.json")
    report = {"test_loaded": False, "verifier_model_selected_on": "known TRAIN fit only",
              "child_thresholds_use_unknowns": False, "child_thresholds_use_validation": False,
              "development_policy_informed_by_prior_v8_validation": True,
              "confirmatory_requirement": "fresh validation/test cohort or nested TRAIN-only policy selection",
              "train_calibration_count": len(state["calibration_query_ids"]),
              "audit_count": len(audit), "audit_image_hashes": [rows[i]["image_sha256"] for i in audit],
              "variants": {}, "audit_scope": "Development audit only; do not report as confirmatory test."}
    raw_scores, parent_values, leaf_values = {}, {}, {}
    for variant in plan["settings"]["variants"]:
        raw, pp, lp, predictions = _variant_predictions(plan, anchor, evidence, models, state, variant)
        raw_scores[variant], parent_values[variant], leaf_values[variant] = raw, pp, lp
        selected = [predictions[i] for i in audit]
        _, summary = witness_pipeline.metrics(selected)
        report["variants"][variant] = {"audit": summary, "policy_counts": _policy_counts(selected)}
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
        row["parent_conformal_pvalues"] = {v: float(parent_values[v][i]) for v in parent_values}
        row["leaf_conformal_pvalues"] = {v: float(leaf_values[v][i]) for v in leaf_values}
        row["decision_alpha"] = float(plan["settings"]["conformal_alpha"])
        row["veto_leaf_multiplier"] = float(plan["settings"]["veto_leaf_multiplier"])
        row["rescue_parent_multiplier"] = float(plan["settings"]["rescue_parent_multiplier"])
        row["rescue_leaf_multiplier"] = float(plan["settings"]["rescue_leaf_multiplier"])
        row["child_partition"] = "validation_audit_only"
        exported.append(row)
    write_records(folder / "validation_evidence.jsonl", exported)
    print("Validation gate against legacy coverage:", report["gate_against_legacy_coverage"], flush=True)


def test(plan):
    rows, anchor, evidence = anchored_evidence(plan, "test")
    folder = resolve(plan["suite"])
    models = read_json(folder / "models.json")
    state = read_json(folder / "train_calibration.json")
    report = {"seed": plan["seed"], "primary_variant": "full", "rows": [],
              "classifier": "frozen_v4_identity", "classification_predictions_identical": True,
              "threshold_source": "known TRAIN conformal split",
              "protocol": "Development test previously inspected; no confirmatory or SOTA claim"}
    for variant in plan["settings"]["variants"]:
        raw, pp, lp, predictions = _variant_predictions(plan, anchor, evidence, models, state, variant)
        for i, row in enumerate(predictions):
            row["dual_evidence"] = dict(zip(core.FEATURES, evidence["x"][i].tolist()))
            row["evidence_valid"] = evidence["valid"][i].tolist()
            row["verifier_raw_score"] = float(raw[i])
            row["parent_conformal_pvalue"] = float(pp[i])
            row["leaf_conformal_pvalue"] = float(lp[i])
        metrics, summary = witness_pipeline.metrics(predictions)
        out = folder / "test" / variant
        write_json(out / "metrics.json", metrics)
        write_records(out / "predictions.jsonl", predictions)
        report["rows"].append({"method": variant, **summary, **_policy_counts(predictions)})
    full = next(r for r in report["rows"] if r["method"] == "full")
    report["legacy_v4"] = witness_pipeline.legacy_metrics(plan, anchor, evidence)
    report["gate_against_legacy_coverage"] = witness_pipeline.gate(report["legacy_v4"]["coverage"], full)
    write_json(folder / "summary.json", report)
    print("Test gate against legacy coverage:", report["gate_against_legacy_coverage"], flush=True)
