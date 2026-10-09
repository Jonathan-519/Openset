"""Development-only leaf refinement of a frozen reference membership router.

The parent candidate, leaf candidate and parent acceptance are inherited from
the baseline. Only the leaf membership and reconstruction thresholds are fitted.
No model weights or test scores are used by this module.
"""
import copy
import hashlib
import json

import numpy as np

from taxosafe_support import calibration as base
from taxosafe_support import membership_calibration as membership

SCHEMA_VERSION = "reference_refine_v1"
DECODER = "membership_reconstruction"
SELECTION_RULE = ("baseline preservation and strict near/precision improvement first; "
                  "joint four-gate feasibility, normalized count deficit, near plus precision, "
                  "known accuracy, baseline leaf proximity and deterministic threshold ties")
LOCAL_SCORE_NOTE = ("When refinement is enabled, local_knownness_score is the minimum of the leaf-membership "
                    "logit margin and reconstruction-score margin. These have different scales. Its AUROC "
                    "describes this fixed combined route score, not pure reconstruction AUROC or a calibrated "
                    "probability. reconstruction_score and baseline_local_knownness_score remain separate. "
                    "Baseline fallback uses only the original leaf-membership threshold margin.")


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _reconstruction(records):
    values = []
    for row in records:
        value = row.get("reconstruction_score")
        if isinstance(value, bool) or np.shape(value) != ():
            raise ValueError("reconstruction_score must be a finite scalar")
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("reconstruction_score must be a finite scalar") from exc
        if not np.isfinite(value):
            raise ValueError("reconstruction_score must be a finite scalar")
        values.append(value)
    return np.asarray(values, dtype=np.float64)


def unique_records(records):
    """Keep content aliases only when baseline AND reconstruction evidence agree."""
    records = list(records)
    scores = _reconstruction(records)
    seen = {}
    for row, score in zip(records, scores):
        digest = base._digest(row)
        if digest in seen and seen[digest] != score:
            raise ValueError("Same content has inconsistent reconstruction evidence: " + digest)
        seen[digest] = score
    return base.unique_records(records)


def _fit_inputs(known, near, extra, meta):
    groups = [list(known), list(near), list(extra)]
    unique_records(sum(groups, []))  # Check new evidence BEFORE baseline deduplication.
    return base._fit_inputs(*groups, meta)


def _validate_baseline(router, meta):
    # This implementation deliberately cannot refine legacy joint-tree decoders.
    return membership._validate_state(router, meta)


def _validate_state(router, meta):
    if router.get("schema_version") != SCHEMA_VERSION or router.get("decoder") != DECODER:
        raise ValueError("Refinement router schema or decoder mismatch")
    if router.get("meta") != meta:
        raise ValueError("Refinement hierarchy differs from inference hierarchy")
    baseline = router["baseline_router"]
    parent, baseline_leaf = _validate_baseline(baseline, meta)
    leaf, reconstruction = float(router["leaf_threshold"]), float(router["reconstruction_threshold"])
    if not np.isfinite([leaf, reconstruction]).all():
        raise ValueError("Refinement thresholds must be finite")
    if float(router["fixed_parent_threshold"]) != parent:
        raise ValueError("Refinement cannot change the frozen parent threshold")
    enabled = router.get("reconstruction_gate_enabled")
    if not isinstance(enabled, bool):
        raise ValueError("Refinement must declare reconstruction_gate_enabled")
    if not enabled and leaf != baseline_leaf:
        raise ValueError("Baseline fallback must retain the baseline leaf threshold")
    return baseline, parent, leaf, reconstruction, enabled


def apply_router(records, router, meta):
    records = list(records)
    baseline, _, leaf_threshold, reconstruction_threshold, enabled = _validate_state(router, meta)
    recon = _reconstruction(records)
    baseline_predictions = base.apply_router(records, baseline, meta)
    result = []
    parent_count = len(meta["parent_names"])
    for original, reconstruction in zip(baseline_predictions, recon):
        row = dict(original)
        p, c = row["candidate_parent"], row["candidate_leaf"]
        leaf_score = row["leaf_membership_score"]
        row["baseline_prediction_type"] = row["prediction_type"]
        row["baseline_local_knownness_score"] = row["local_knownness_score"]
        row["baseline_leaf_threshold"] = float(baseline["leaf_threshold"])
        if row["prediction_type"] != "global_unknown":
            accepted = leaf_score >= leaf_threshold and (not enabled or reconstruction >= reconstruction_threshold)
            row.update(prediction_type="known" if accepted else "intra_unknown",
                       parent=p, leaf=c if accepted else None,
                       output_node=1 + parent_count + c if accepted else 1 + p)
        margin = float(leaf_score - leaf_threshold)
        if enabled:
            margin = min(margin, float(reconstruction - reconstruction_threshold))
        row.update(decoder=DECODER, leaf_threshold=leaf_threshold,
                   reconstruction_threshold=reconstruction_threshold,
                   reconstruction_gate_enabled=enabled, reconstruction_score=float(reconstruction),
                   local_threshold=0., local_knownness_score=margin, local_known_margin=margin,
                   local_score_type="minimum_leaf_and_reconstruction_threshold_margin" if enabled else "baseline_leaf_threshold_margin")
        result.append(row)
    return result


decode_records = apply_router


def evaluate_records(records, meta=None):
    records = list(records)
    unique_records(records)
    return base.evaluate_records(records, meta)


evaluate_gates = base.evaluate_gates


def _grid(values, settings, name, include=()):
    if settings.get("threshold_grid", "quantile") != "quantile":
        raise ValueError("Refinement threshold_grid must be quantile")
    if name in settings:
        grid = np.asarray(settings[name], dtype=float)
    else:
        count = settings.get("grid_points", 49)
        if isinstance(count, bool) or int(count) != count or not 2 <= count <= 401:
            raise ValueError("Refinement grid_points must be an integer in [2,401]")
        grid = np.quantile(values, np.linspace(0., 1., int(count)))
    if grid.ndim != 1 or not 1 <= len(grid) <= 403 or not np.isfinite(grid).all():
        raise ValueError(name + " must contain finite, nonempty thresholds")
    with np.errstate(over="ignore"):
        grid = np.r_[grid, np.nextafter(values.min(), -np.inf), np.nextafter(values.max(), np.inf), include]
    if not np.isfinite(grid).all():
        raise ValueError("Evidence is too extreme for finite all-pass/all-reject endpoints")
    return np.unique(grid)


def _arrays(rows, baseline_router, meta):
    d = membership._arrays(rows, meta)
    d["reconstruction"] = _reconstruction(rows)
    d["parent_accept"] = d["parent_score"] >= float(baseline_router["parent_threshold"])
    return d


def _counts(d, leaf_accept):
    counts = {s: int(d[s].sum()) for s in ("known", "intra", "extra")}
    counts.update(known_correct=int(np.sum(d["known_candidate_correct"] & leaf_accept)),
                  intra_correct=int(np.sum(d["near_candidate_correct"] & d["parent_accept"] & ~leaf_accept)),
                  extra_correct=int(np.sum(d["extra"] & ~d["parent_accept"])),
                  leaf_outputs=int(leaf_accept.sum()))
    return counts


def _preservation(counts, baseline):
    required_known = max(baseline["known_correct"], 9 * counts["known"] // 10 + 1)
    precision_defined = counts["leaf_outputs"] > 0 and baseline["leaf_outputs"] > 0
    left = counts["known_correct"] * baseline["leaf_outputs"]
    right = baseline["known_correct"] * counts["leaf_outputs"]
    checks = {"known_correct_at_least_baseline": counts["known_correct"] >= baseline["known_correct"],
              "strict_known_gate": counts["known_correct"] >= required_known,
              "near_correct_at_least_baseline": counts["intra_correct"] >= baseline["intra_correct"],
              "leaf_precision_at_least_baseline": precision_defined and left >= right,
              "root_rejection_count_unchanged": counts["extra_correct"] == baseline["extra_correct"]}
    improved = counts["intra_correct"] > baseline["intra_correct"] or (precision_defined and left > right)
    return {"checks": checks, "passed": all(checks.values()), "strict_near_or_precision_improvement": bool(improved),
            "required_known_correct": required_known, "baseline_counts": dict(baseline), "selected_counts": dict(counts),
            "precision_cross_product": {"selected_correct_times_baseline_leaf_outputs": left,
                                        "baseline_correct_times_selected_leaf_outputs": right,
                                        "both_denominators_positive": precision_defined},
            "scope": "aggregate unique-content development counts; not a per-image or future-data guarantee"}


def _select(rows, baseline_router, meta, settings, keep_grid=True):
    _validate_baseline(baseline_router, meta)
    d = _arrays(rows, baseline_router, meta)
    baseline_leaf = float(baseline_router["leaf_threshold"])
    lg = _grid(d["leaf_score"], settings, "leaf_threshold_grid", (baseline_leaf,))
    rg = _grid(d["reconstruction"], settings, "reconstruction_threshold_grid")
    baseline_counts = _counts(d, d["parent_accept"] & (d["leaf_score"] >= baseline_leaf))
    best, points, preservation_count, improvement_count, feasible_count = None, [], 0, 0, 0
    for lt in lg:
        selected_leaf = d["parent_accept"] & (d["leaf_score"] >= lt)
        for rt in rg:
            counts = _counts(d, selected_leaf & (d["reconstruction"] >= rt))
            gates, audit = base._gates(counts), _preservation(counts, baseline_counts)
            preserved = audit["passed"]
            improved = preserved and audit["strict_near_or_precision_improvement"]
            preservation_count += int(preserved)
            improvement_count += int(improved)
            feasible_count += int(preserved and gates["targets_passed"])
            if improved:
                deficit = sum(v["missing_correct"] / max(v["total"], 1) for v in gates["requirements"].values())
                rates = list(gates["metrics"].values())
                key = (gates["targets_passed"], -deficit, rates[1] + rates[3], rates[0],
                       -abs(float(lt) - baseline_leaf), -float(rt), -float(lt))
                if best is None or key > best[0]:
                    best = key, float(lt), float(rt), gates, audit
            if keep_grid:
                points.append({"leaf_threshold": float(lt), "reconstruction_threshold": float(rt),
                               **gates["metrics"], "targets_passed": gates["targets_passed"],
                               "preservation_passed": preserved, "strict_improvement": improved})
    if best is None:
        lt, rt, gates, audit = baseline_leaf, float(rg[0]), base._gates(baseline_counts), _preservation(baseline_counts, baseline_counts)
        status, enabled = "baseline_fallback", False
    else:
        _, lt, rt, gates, audit = best
        status, enabled = ("feasible_refinement" if gates["targets_passed"] else "best_effort_refinement"), True
    return {"leaf_threshold": lt, "reconstruction_threshold": rt, "reconstruction_gate_enabled": enabled,
            "status": status, "gates": gates, "preservation_audit": audit, "grid_tradeoff": points,
            "grid": {"leaf_threshold": lg.tolist(), "reconstruction_threshold": rg.tolist(),
                     "sampled_point_count": len(lg) * len(rg), "sampled_preservation_count": preservation_count,
                     "sampled_improvement_count": improvement_count, "sampled_feasible_count": feasible_count,
                     "baseline_leaf_threshold_included": True, "reconstruction_all_accept_included": True,
                     "tie_rule": "both scores >= their thresholds accept the fixed leaf candidate",
                     "definition": "fit-only empirical quantiles or explicit thresholds, endpoints and baseline leaf threshold",
                     "scope": "finite declared development grid; no continuous infeasibility claim"}}


def _router(selected, baseline_router, meta):
    return {"schema_version": SCHEMA_VERSION, "decoder": DECODER, "meta": copy.deepcopy(meta),
            "baseline_router": copy.deepcopy(baseline_router),
            "baseline_router_sha256": _hash(baseline_router),
            "candidate_rule": membership.CANDIDATE_RULE,
            "fixed_parent_threshold": float(baseline_router["parent_threshold"]),
            "leaf_threshold": selected["leaf_threshold"], "reconstruction_threshold": selected["reconstruction_threshold"],
            "reconstruction_gate_enabled": selected["reconstruction_gate_enabled"],
            "local_score_note": LOCAL_SCORE_NOTE}


def source_loo(known, near, extra, baseline_router, meta, settings=None):
    settings = dict(settings or {})
    rows, _ = _fit_inputs(known, near, extra, meta)
    _validate_baseline(baseline_router, meta)
    baseline_settings = settings.get("baseline_calibration")
    if not isinstance(baseline_settings, dict) or baseline_settings.get("decoder") != "membership":
        raise ValueError("Refinement source LOO requires the original baseline_calibration configuration, not the all-development router grid")
    baseline_settings = dict(baseline_settings, source_loo=False)
    folds, skipped = [], []
    for status in ("intra", "extra"):
        sources = sorted({str(r.get("source", "unspecified")) for r in rows if r["status"] == status})
        if len(sources) < 2:
            skipped.append({"status": status, "reason": "At least two development sources are required"})
            continue
        for held in sources:
            is_held = lambda r: r["status"] == status and str(r.get("source", "unspecified")) == held
            fitted, holdout = [r for r in rows if not is_held(r)], [r for r in rows if is_held(r)]
            groups = [[r for r in fitted if r["status"] == s] for s in ("known", "intra", "extra")]
            fold_base = base.calibrate(*groups, meta, baseline_settings)
            selected = _select(fitted, fold_base, meta, settings, keep_grid=False)
            router = _router(selected, fold_base, meta)
            baseline_held = base.apply_router(holdout, fold_base, meta)
            refined_held = apply_router(holdout, router, meta)
            folds.append({"status": status, "held_source": held, "fit_count": len(fitted),
                          "fit_image_sha256": sorted(base._digest(r) for r in fitted),
                          "baseline_refit_on_fit_sources_only": True,
                          "baseline_parent_threshold": fold_base["parent_threshold"],
                          "baseline_leaf_threshold": fold_base["leaf_threshold"],
                          "baseline_calibration_sha256": fold_base["calibration_sha256"],
                          "leaf_threshold": router["leaf_threshold"], "reconstruction_threshold": router["reconstruction_threshold"],
                          "fit_selection_status": selected["status"], "fit_targets_passed": selected["gates"]["targets_passed"],
                          "fit_preservation_audit": selected["preservation_audit"],
                          "fit_sources": sorted({str(r.get("source", "unspecified")) for r in fitted if r["status"] == status}),
                          "grid_uses_fit_sources_only": True,
                          "baseline_held_metrics": base._group_report(baseline_held, status),
                          "held_metrics": base._group_report(refined_held, status)})
    return {"available": bool(folds), "folds": folds, "skipped": skipped,
            "use": "development diagnostic only; baseline parent/leaf gates and refinement are independently refit without the held source; model weights remain frozen"}


def calibrate(known, near, extra, baseline_router, meta, settings=None):
    settings = dict(settings or {})
    known, near, extra = list(known), list(near), list(extra)
    rows, input_count = _fit_inputs(known, near, extra, meta)
    selected = _select(rows, baseline_router, meta, settings)
    router = _router(selected, baseline_router, meta)
    router.update(fitted_parameters=["leaf_threshold", "reconstruction_threshold"], fit_completed=True,
                  targets_passed=selected["gates"]["targets_passed"], targets=copy.deepcopy(base.TARGETS),
                  status=selected["status"], best_effort=not selected["gates"]["targets_passed"],
                  baseline_fallback=not selected["reconstruction_gate_enabled"], selection_rule=SELECTION_RULE,
                  preservation_audit=selected["preservation_audit"], grid=selected["grid"], grid_tradeoff=selected["grid_tradeoff"],
                  fit_splits=["val_known", "val_intra", "val_extra"], input_record_count=input_count,
                  unique_image_count=len(rows), duplicate_record_count=input_count-len(rows),
                  fit_image_sha256=sorted(base._digest(r) for r in rows))
    router["baseline_validation_report"] = base.evaluate_records(base.apply_router(rows, baseline_router, meta), meta)
    report = evaluate_records(apply_router(rows, router, meta), meta)
    for key in ("fit_completed", "status", "best_effort", "baseline_fallback", "preservation_audit", "selection_rule", "local_score_note"):
        report[key] = copy.deepcopy(router[key])
    report["sampled_feasible_count"] = selected["grid"]["sampled_feasible_count"]
    report["infeasibility"] = {"no_feasible_preserving_sampled_point": not bool(report["sampled_feasible_count"]),
                               "failed_at_selected_point": [k for k, v in report["checks"].items() if not v],
                               "scope": "finite refinement grid with baseline preservation constraints only"}
    candidates = _arrays(rows, baseline_router, meta)
    near_ceiling = int(np.sum(candidates["near_candidate_correct"] & candidates["parent_accept"]))
    near_total = int(candidates["intra"].sum())
    near_required = (17 * near_total + 19) // 20
    report["parent_routing_near_upper_bound"] = {
        "correct_count": near_ceiling, "total": near_total, "rate": near_ceiling / near_total,
        "required_correct": near_required, "can_reach_target": near_ceiling >= near_required,
        "scope": "fixed baseline ranking candidates and parent threshold; even rejecting every leaf cannot exceed this near count"}
    report["fixed_parent_audit"] = {"parent_threshold_unchanged": True,
                                    "root_decisions_unchanged": True,
                                    "extra_recall_can_improve": False,
                                    "baseline_extra_gate_passed": router["baseline_validation_report"]["checks"]["extra_global_unknown_recall"]}
    router["validation_report"] = report
    router["source_loo"] = source_loo(known, near, extra, baseline_router, meta, settings) if settings.get("source_loo", True) else {
        "available": False, "folds": [], "reason": "disabled in configuration"}
    evidence = [{"image_sha256": base._digest(row), "split": row["split"], "status": row["status"],
                 "true_parent": row.get("true_parent"), "true_leaf": row.get("true_leaf"), "source": row.get("source"),
                 "log_probs": np.asarray(row["log_probs"], dtype=float).tolist(),
                 "support_evidence": {field: np.asarray(row["support_evidence"][field], dtype=float).tolist() for field in membership.RAW_FIELDS},
                 "reconstruction_score": float(row["reconstruction_score"])} for row in sorted(rows, key=base._digest)]
    router["evidence_sha256"] = _hash(evidence)
    router["calibration_sha256"] = _hash({k: router[k] for k in ("schema_version", "decoder", "meta", "baseline_router_sha256",
        "fixed_parent_threshold", "candidate_rule", "leaf_threshold", "reconstruction_threshold", "reconstruction_gate_enabled",
        "selection_rule", "grid", "targets", "evidence_sha256")})
    return router
