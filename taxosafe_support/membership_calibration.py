"""Two development-only thresholds on fixed, candidate-aligned membership.

Identity ranking chooses one parent and its highest-ranked child BEFORE either
threshold is applied. Root rejection uses that parent's membership; leaf
acceptance uses that child's membership. A rejected candidate is never replaced
by a sibling or another parent with a more convenient membership score.

This is a decoder/operating-point choice, not a distribution-free guarantee.
The retained joint log_probs are diagnostic probabilities from the model; they
do not define decisions under this decoder.
"""
import copy
import hashlib
import json

import numpy as np

from .calibration import (
    TARGETS, _digest, _fit_inputs, _gates, _group_report, _hierarchy, _scores,
    _selection_policy, evaluate_records,
)

SCHEMA_VERSION = "support_membership_v1"
RAW_FIELDS = ("parent_logits", "leaf_logits", "parent_membership_logits", "leaf_membership_logits")
CANDIDATE_RULE = "argmax_parent_ranking_then_argmax_leaf_ranking_within_that_parent"


def candidate_scores(records, meta):
    """Validate all raw heads and produce candidates independent of truth/gates."""
    records = list(records)
    p, c, mapping = _hierarchy(meta)
    heads = {}
    for field in RAW_FIELDS:
        expected = p if field.startswith("parent_") else c
        try:
            values = np.asarray([row["support_evidence"][field] for row in records], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Membership decoder requires raw support_evidence." + field) from exc
        if not records:
            values = np.empty((0, expected), dtype=np.float64)
        if values.shape != (len(records), expected) or not np.isfinite(values).all():
            raise ValueError("Invalid finite candidate evidence shape: " + field)
        heads[field] = values
    parent = heads["parent_logits"].argmax(1)
    leaf = np.empty(len(records), dtype=int)
    for node in range(p):
        selected = parent == node
        children = np.flatnonzero(mapping == node)
        leaf[selected] = children[heads["leaf_logits"][selected][:, children].argmax(1)]
    index = np.arange(len(records))
    return {"parent": parent, "leaf": leaf,
            "parent_score": heads["parent_membership_logits"][index, parent],
            "leaf_score": heads["leaf_membership_logits"][index, leaf], "heads": heads}


def _validate_state(state, meta):
    if (state.get("schema_version") != SCHEMA_VERSION or state.get("decoder") != "membership"
            or state.get("candidate_rule", CANDIDATE_RULE) != CANDIDATE_RULE):
        raise ValueError("Membership calibration schema, decoder, or candidate rule mismatch")
    if "meta" in state and state["meta"] != meta:
        raise ValueError("Membership calibration hierarchy differs from inference hierarchy")
    thresholds = float(state["parent_threshold"]), float(state["leaf_threshold"])
    if not np.isfinite(thresholds).all():
        raise ValueError("Membership thresholds must be finite logits")
    return thresholds


def decode_records(records, state, meta):
    records = list(records)
    pt, lt = _validate_state(state, meta)
    _scores(records, meta)
    scores = candidate_scores(records, meta)
    parent_count = len(meta["parent_names"])
    result = []
    for i, row in enumerate(records):
        p, c = int(scores["parent"][i]), int(scores["leaf"][i])
        pm, lm = float(scores["parent_score"][i]), float(scores["leaf_score"][i])
        if pm < pt:
            kind, parent, leaf, node = "global_unknown", None, None, 0
        elif lm < lt:
            kind, parent, leaf, node = "intra_unknown", p, None, 1 + p
        else:
            kind, parent, leaf, node = "known", p, c, 1 + parent_count + c
        record = dict(row)
        record.update({
            "prediction_type": kind, "output_node": node, "parent": parent, "leaf": leaf,
            "candidate_parent": p, "candidate_leaf": c,
            "candidate_parent_name": meta["parent_names"][p],
            "candidate_leaf_name": meta["leaf_names"][c],
            "true_parent_name": None if row.get("true_parent") is None else meta["parent_names"][int(row["true_parent"])],
            "true_leaf_name": None if row.get("true_leaf") is None else meta["leaf_names"][int(row["true_leaf"])],
            "global_pred_leaf": int(row.get("global_pred_leaf", scores["heads"]["leaf_logits"][i].argmax())),
            "root_knownness_score": pm, "local_knownness_score": lm, "local_known_margin": lm,
            "root_score_type": "selected_parent_membership_logit",
            "local_score_type": "selected_leaf_membership_logit",
            "parent_membership_score": pm, "leaf_membership_score": lm,
            "parent_threshold": pt, "leaf_threshold": lt,
            "root_threshold": pt, "local_threshold": lt,
            "decoder": "membership", "candidate_rule": CANDIDATE_RULE,
        })
        result.append(record)
    return result


def _grid(values, settings, name):
    if settings.get("threshold_grid", "quantile") != "quantile":
        raise ValueError("Membership threshold_grid must be quantile")
    if name in settings:
        grid = np.asarray(settings[name], dtype=float)
    else:
        count = settings.get("membership_grid_points", settings.get("grid_points", 49))
        if isinstance(count, bool) or int(count) != count or not 2 <= count <= 401:
            raise ValueError("membership_grid_points must be an integer in [2,401]")
        with np.errstate(over="ignore"):
            low, high = np.nextafter(values.min(), -np.inf), np.nextafter(values.max(), np.inf)
        grid = np.r_[low, np.quantile(values, np.linspace(0., 1., int(count))), high]
    if grid.ndim != 1 or not 1 <= len(grid) <= 403 or not np.isfinite(grid).all():
        raise ValueError(name + " must contain finite thresholds; raw scores may be too extreme")
    return np.unique(grid)


def _arrays(rows, meta):
    data = candidate_scores(rows, meta)
    statuses = np.asarray([r["status"] for r in rows])
    data["known"], data["intra"], data["extra"] = (statuses == s for s in ("known", "intra", "extra"))
    truth_parent = np.asarray([-1 if r.get("true_parent") is None else r["true_parent"] for r in rows])
    truth_leaf = np.asarray([-1 if r.get("true_leaf") is None else r["true_leaf"] for r in rows])
    data["known_candidate_correct"] = data["known"] & (data["parent"] == truth_parent) & (data["leaf"] == truth_leaf)
    data["near_candidate_correct"] = data["intra"] & (data["parent"] == truth_parent)
    return data


def _select(rows, meta, settings, keep_grid=True):
    policy = _selection_policy({"policy": settings.get("policy", "known_first")})
    d = _arrays(rows, meta)
    pg = _grid(d["parent_score"], settings, "parent_threshold_grid")
    lg = _grid(d["leaf_score"], settings, "leaf_threshold_grid")
    totals = {s: int(d[s].sum()) for s in ("known", "intra", "extra")}
    best, feasible, known_feasible, points = None, 0, 0, []
    maxima = {name: 0. for name in TARGETS}
    for pt in pg:
        parent_accept = d["parent_score"] >= pt
        for lt in lg:
            leaf_accept = parent_accept & (d["leaf_score"] >= lt)
            parent_only = parent_accept & ~leaf_accept
            counts = dict(totals)
            counts.update(known_correct=int(np.sum(d["known_candidate_correct"] & leaf_accept)),
                          intra_correct=int(np.sum(d["near_candidate_correct"] & parent_only)),
                          extra_correct=int(np.sum(d["extra"] & ~parent_accept)),
                          leaf_outputs=int(leaf_accept.sum()))
            report = _gates(counts)
            rates = [report["metrics"][name] or 0. for name in TARGETS]
            deficit = sum(v["missing_correct"] / max(v["total"], 1) for v in report["requirements"].values())
            known_pass = report["checks"]["known_end_to_end_leaf_accuracy"]
            key = (report["targets_passed"], -deficit, float(np.mean(rates)), rates[0],
                   -abs(float(pt)) - abs(float(lt)), -float(pt), -float(lt))
            if policy == "known_first":
                key = (key[0], known_pass) + key[1:]
            if best is None or key > best[0]:
                best = key, float(pt), float(lt), report
            feasible += int(report["targets_passed"])
            known_feasible += int(known_pass)
            for name, value in report["metrics"].items():
                maxima[name] = max(maxima[name], value or 0.)
            if keep_grid:
                points.append({"parent_threshold": float(pt), "leaf_threshold": float(lt),
                               **report["metrics"], "targets_passed": report["targets_passed"]})
    if feasible:
        status = "feasible"
    elif policy == "known_first":
        status = "best_effort_known_preserved" if known_feasible else "best_effort_known_unavailable"
    else:
        status = "best_effort"
    return {"parent_threshold": best[1], "leaf_threshold": best[2], "report": best[3],
            "selection_policy": policy, "status": status, "best_effort": not bool(feasible),
            "sampled_known_feasible_count": known_feasible, "individual_sampled_maxima": maxima,
            "grid_tradeoff": points,
            "grid": {"parent_threshold": pg.tolist(), "leaf_threshold": lg.tolist(),
                     "definition": "explicit thresholds when supplied, otherwise fit-only empirical quantiles plus all-pass/all-reject endpoints",
                     "score_unit": "logit", "tie_rule": "membership >= threshold accepts",
                     "sampled_point_count": len(pg) * len(lg), "sampled_feasible_count": feasible,
                     "coverage": "finite declared grid; quantile sampling is not an exhaustive operating-point search"}}


def _feasibility(rows, meta):
    """Necessary order bounds for THIS decoder, never the old two-bias proof."""
    d = _arrays(rows, meta)
    kc, nc, extra = d["known_candidate_correct"], d["near_candidate_correct"], d["extra"]
    pm, lm = d["parent_score"], d["leaf_score"]
    required = {"known": 9 * int(d["known"].sum()) // 10 + 1,
                "intra": (17 * int(d["intra"].sum()) + 19) // 20,
                "extra": 9 * int(extra.sum()) // 10 + 1}
    reasons, bounds = [], {}
    if kc.sum() < required["known"]:
        reasons.append("known_ranking_candidate_ceiling_below_target")
    else:
        pu = float(np.sort(pm[kc])[-required["known"]])
        lu = float(np.sort(lm[kc])[-required["known"]])
        bounds.update(known_requires_parent_threshold_at_most=pu, known_requires_leaf_threshold_at_most=lu,
                      near_correct_upper_bound_given_known_ignoring_parent=int(np.sum(nc & (lm < lu))))
    if nc.sum() < required["intra"]:
        reasons.append("near_parent_ranking_candidate_ceiling_below_target")
    else:
        nl = float(np.sort(lm[nc])[required["intra"] - 1])
        npu = float(np.sort(pm[nc])[-required["intra"]])
        bounds.update(near_requires_leaf_threshold_strictly_above=nl,
                      near_requires_parent_threshold_at_most=npu)
        if kc.sum() >= required["known"] and nl >= lu:
            reasons.append("known_and_near_require_incompatible_leaf_membership_threshold")
    el = float(np.sort(pm[extra])[required["extra"] - 1])
    bounds["extra_requires_parent_threshold_strictly_above"] = el
    if kc.sum() >= required["known"]:
        upper = pu if nc.sum() < required["intra"] else min(pu, npu)
        bounds["extra_root_upper_bound_preserving_required_candidate_parent_retention"] = int(np.sum(extra & (pm < upper)))
        if el >= pu:
            reasons.append("known_and_extra_require_incompatible_parent_membership_threshold")
    if nc.sum() >= required["intra"] and el >= npu:
        reasons.append("near_and_extra_require_incompatible_parent_membership_threshold")
    return {"decoder": "membership", "scope": "fixed saved development raw logits and fixed ranking candidates only",
            "required_correct": required, "candidate_correct": {"known": int(kc.sum()), "intra": int(nc.sum())},
            "necessary_bounds": bounds, "continuous_infeasibility_proven": bool(reasons), "contradictions": reasons,
            "interpretation": "Necessary threshold-order bounds only. No contradiction is not proof of feasibility; no claim about retrained evidence or a different candidate rule. Joint two-bias bounds do not apply."}


def fixed_score_feasibility(known, near, extra, meta):
    """Check necessary membership-threshold bounds on development splits only."""
    rows, _ = _fit_inputs(known, near, extra, meta)
    return _feasibility(rows, meta)


def source_loo(known, near, extra, meta, settings=None):
    settings = dict(settings or {})
    rows, _ = _fit_inputs(known, near, extra, meta)
    candidate_scores(rows, meta)
    folds, skipped = [], []
    for status in ("intra", "extra"):
        sources = sorted({str(r.get("source", "unspecified")) for r in rows if r["status"] == status})
        if len(sources) < 2:
            skipped.append({"status": status, "reason": "At least two development sources are required"})
            continue
        for held in sources:
            selected = lambda r: r["status"] == status and str(r.get("source", "unspecified")) == held
            fitted, holdout = [r for r in rows if not selected(r)], [r for r in rows if selected(r)]
            result = _select(fitted, meta, settings, keep_grid=False)
            state = {"schema_version": SCHEMA_VERSION, "decoder": "membership",
                     "parent_threshold": result["parent_threshold"], "leaf_threshold": result["leaf_threshold"]}
            folds.append({"status": status, "held_source": held,
                          "parent_threshold": result["parent_threshold"], "leaf_threshold": result["leaf_threshold"],
                          "fit_count": len(fitted), "fit_targets_passed": result["report"]["targets_passed"],
                          "selection_policy": result["selection_policy"], "fit_selection_status": result["status"],
                          "fit_sources": sorted({str(r.get("source", "unspecified")) for r in fitted if r["status"] == status}),
                          "grid_uses_fit_sources_only": True,
                          "held_metrics": _group_report(decode_records(holdout, state, meta), status)})
    return {"available": bool(folds), "folds": folds, "skipped": skipped,
            "use": "development diagnostic only; each empirical grid and both thresholds exclude the held source; no model updates or final-policy selection"}


def calibrate(known, near, extra, meta, settings=None):
    settings = dict(settings or {})
    known, near, extra = list(known), list(near), list(extra)
    rows, input_count = _fit_inputs(known, near, extra, meta)
    selected = _select(rows, meta, settings)
    state = {"schema_version": SCHEMA_VERSION, "decoder": "membership", "meta": copy.deepcopy(meta),
             "candidate_rule": CANDIDATE_RULE, "parent_threshold": selected["parent_threshold"],
             "leaf_threshold": selected["leaf_threshold"], "threshold_unit": "raw_membership_logit",
             "fitted_parameters": ["parent_threshold", "leaf_threshold"], "fit_completed": True,
             "targets_passed": selected["report"]["targets_passed"], "targets": copy.deepcopy(TARGETS),
             "selection_policy": selected["selection_policy"], "status": selected["status"],
             "best_effort": selected["best_effort"], "fit_splits": ["val_known", "val_intra", "val_extra"],
             "input_record_count": input_count, "unique_image_count": len(rows),
             "duplicate_record_count": input_count - len(rows), "fit_image_sha256": sorted(_digest(r) for r in rows),
             "grid": selected["grid"], "grid_tradeoff": selected["grid_tradeoff"],
             "selection_diagnostics": {"sampled_known_feasible_count": selected["sampled_known_feasible_count"],
                "known_feasible_on_grid": bool(selected["sampled_known_feasible_count"]),
                "selected_known_gate_passed": selected["report"]["checks"]["known_end_to_end_leaf_accuracy"],
                "known_preservation_scope": "declared development threshold grid only"}}
    report = evaluate_records(decode_records(rows, state, meta), meta)
    for key in ("fit_completed", "selection_policy", "status", "best_effort", "selection_diagnostics", "decoder"):
        report[key] = copy.deepcopy(state[key])
    report["sampled_feasible_count"] = state["grid"]["sampled_feasible_count"]
    report["individual_sampled_maxima"] = selected["individual_sampled_maxima"]
    report["infeasibility"] = {
        "no_feasible_sampled_point": not bool(report["sampled_feasible_count"]),
        "failed_at_selected_point": [name for name, passed in report["checks"].items() if not passed],
        "scope": "finite declared threshold grid only",
        "selection_rule": "joint feasibility first; known-first restricts best effort to known-passing points when available; then normalized count deficit, mean metrics and deterministic ties"}
    report["fixed_score_feasibility"] = _feasibility(rows, meta)
    state["validation_report"] = report
    state["source_loo"] = source_loo(known, near, extra, meta, settings) if settings.get("source_loo", True) else {
        "available": False, "folds": [], "reason": "disabled in configuration"}
    evidence = [{"image_sha256": _digest(row), "split": row["split"], "status": row["status"],
                 "true_parent": row.get("true_parent"), "true_leaf": row.get("true_leaf"),
                 "source": row.get("source"),
                 "support_evidence": {key: np.asarray(row["support_evidence"][key], dtype=float).tolist() for key in RAW_FIELDS},
                 "log_probs": np.asarray(row["log_probs"], dtype=float).tolist()}
                for row in sorted(rows, key=_digest)]
    state["evidence_sha256"] = hashlib.sha256(json.dumps(evidence, sort_keys=True, allow_nan=False).encode()).hexdigest()
    binding = {"schema_version": SCHEMA_VERSION, "decoder": "membership", "meta": meta,
               "candidate_rule": CANDIDATE_RULE, "evidence_sha256": state["evidence_sha256"],
               "parent_threshold": state["parent_threshold"], "leaf_threshold": state["leaf_threshold"],
               "selection_policy": state["selection_policy"], "grid": state["grid"], "targets": TARGETS}
    state["calibration_sha256"] = hashlib.sha256(json.dumps(binding, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return state
