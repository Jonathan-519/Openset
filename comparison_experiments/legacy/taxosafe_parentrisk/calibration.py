"""Small shared candidates selected inside held-known/source folds.

Outer predictions audit the entire frozen-reference postprocessor procedure.
They NEVER select a weight, threshold, action, mode, or deployment fallback.
The final production fit uses the archived reference router. Each audit fold
instead recalibrates its membership baseline using fit records only.
"""
import copy

import numpy as np

from taxosafe_geometry import calibration as geometry
from taxosafe_support import calibration as base
from taxosafe_support import membership_calibration as membership

from .decoder import (DECODER, SCHEMA_VERSION, apply_router, digest, evidence_arrays,
                      make_router, parent_candidates)
from .folds import build_folds, unique_records, source_key
from .reporting import paired_report, score_diagnostics, rule_support as matched_rule_support

# Fixed before seeing DEV scores. Every family contains at most one parent
# rule, one shared leaf rule and one shared root rule. No per-parent exceptions.
PARENT_WEIGHTS = ((1., 0., 0.), (.5, .5, 0.), (.5, 0., .5), (1./3., 1./3., 1./3.))
VALIDATION_SCOPE = "postprocessor_conditional_on_frozen_reference"


def settings(options=None):
    result = dict(options or {})
    allowed = {"decoder", "mode", "outer_folds", "inner_folds", "grid_points",
               "min_rule_sources", "seed", "baseline_calibration"}
    if set(result) - allowed or result.get("decoder", DECODER) != DECODER:
        raise ValueError("Invalid parentrisk calibration settings")
    result.setdefault("mode", "combined")
    if result["mode"] not in ("audit", "parent_only", "combined"):
        raise ValueError("mode must be audit, parent_only or combined")
    for name, default, low, high in (("outer_folds", 4, 2, 8), ("inner_folds", 3, 2, 8),
                                    ("grid_points", 7, 2, 11), ("min_rule_sources", 2, 2, 8)):
        result.setdefault(name, default)
        if type(result[name]) is not int or not low <= result[name] <= high:
            raise ValueError(name + " has invalid integer range")
    result.setdefault("seed", 0)
    if type(result["seed"]) is not int or not 0 <= result["seed"] < 2**31:
        raise ValueError("seed must be an integer in [0,2**31)")
    bc = result.setdefault("baseline_calibration", {"decoder": "membership", "policy": "known_first"})
    if not isinstance(bc, dict) or bc.get("decoder", "membership") != "membership":
        raise ValueError("Audit baseline_calibration must use membership")
    return result


def _correct(row):
    if row["status"] == "known":
        return row["prediction_type"] == "known" and row.get("leaf") == row.get("true_leaf")
    if row["status"] == "intra":
        return row["prediction_type"] == "intra_unknown" and row.get("parent") == row.get("true_parent")
    return row["prediction_type"] == "global_unknown"


def _summary(rows):
    d = dict(known=0, intra=0, extra=0, known_correct=0, intra_correct=0, extra_correct=0, leaf_outputs=0)
    sources = {}
    for r in rows:
        status, ok = r["status"], int(_correct(r))
        d[status] += 1
        d[status + "_correct"] += ok
        leaf = r["prediction_type"] == "known"
        d["leaf_outputs"] += int(leaf)
        if status != "known":
            key = status, source_key(r)
            s = sources.setdefault(key, [0, 0, 0])
            s[0] += 1
            s[1] += ok
            s[2] += int(leaf)
    return d, sources


def _fast_audit(before, after):
    """Pair by already aligned unique hashes; no absolute gate blocks search."""
    old, os = _summary(before)
    new, ns = _summary(after)
    known_harm, near_harm, path_harm = 0, 0, 0
    for a, b in zip(before, after):
        known_harm += int(a["status"] == "known" and _correct(a) and not _correct(b))
        near_harm += int(a["status"] == "intra" and _correct(a) and not _correct(b))
        path = (a["status"] == "intra" and a["candidate_parent"] == a.get("true_parent")
                and a["prediction_type"] != "global_unknown")
        path_harm += int(path and b["prediction_type"] == "global_unknown")
    source_safe = all(ns[k][1] >= v[1] for k, v in os.items())
    precision_safe = ((new["leaf_outputs"] == 0) if old["leaf_outputs"] == 0 else
                      new["leaf_outputs"] > 0 and
                      new["known_correct"] * old["leaf_outputs"] >= old["known_correct"] * new["leaf_outputs"])
    passed = not known_harm and not near_harm and not path_harm and source_safe and precision_safe
    gains = [dict(status=k[0], source=k[1], correct_gain=ns[k][1]-v[1], false_leaf_reduction=v[2]-ns[k][2])
             for k, v in os.items() if ns[k][1] > v[1] or ns[k][2] < v[2]]
    improved = (new["intra_correct"] > old["intra_correct"] or new["extra_correct"] > old["extra_correct"]
                or new["known_correct"] * old["leaf_outputs"] > old["known_correct"] * new["leaf_outputs"])
    return dict(passed=bool(passed), improved=bool(improved), gains=gains, counts=new,
                known_added_harm=known_harm, near_added_harm=near_harm, near_parent_path_loss=path_harm)


def _quality(rows, changes=0):
    counts, sources = _summary(rows)
    gates = base._gates(counts)
    deficit = sum(v["missing_correct"] / max(v["total"], 1) for v in gates["requirements"].values())
    macro = sum(v[1] / v[0] for v in sources.values()) / max(len(sources), 1)
    metric_sum = sum(v or 0. for v in gates["metrics"].values())
    return (int(gates["targets_passed"]), -deficit, macro, metric_sum, -changes)


def _grid(values, count, nonnegative=False):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return []
    result = np.unique(np.quantile(values, np.linspace(0., 1., count)))
    if nonnegative:
        result = np.unique(np.r_[0., result])
    return [float(v) for v in result]


def _changed(before, after):
    return sum((a["prediction_type"], a.get("parent"), a.get("leaf")) !=
               (b["prediction_type"], b.get("parent"), b.get("leaf")) for a, b in zip(before, after))


def _support(before, after, meta, action):
    fast = _fast_audit(before, after)
    scopes = []
    for parent, name in enumerate(meta["parent_names"]):
        eligible = [r for r in before if r.get("route_parent", r["candidate_parent"]) == parent
                    and (r["prediction_type"] == "known" if action == "leaf_reject"
                         else r["prediction_type"] != "global_unknown" if action == "root_reject"
                         else r["prediction_type"] != "known")]
        protected = [r for r in eligible if r["status"] == "known" and _correct(r)]
        leaves = []
        for leaf, leaf_name in enumerate(meta["leaf_names"]):
            if meta["leaf_to_parent"][leaf] == parent:
                n = sum(r.get("true_leaf") == leaf for r in protected)
                leaves.append(dict(leaf=leaf, name=leaf_name, protected_known_count=n,
                                   evidence_status="not_evaluable" if n == 0 else "insufficient_evidence" if n < 5 else "observed"))
        scopes.append(dict(parent=parent, name=name, eligible_count=len(eligible),
                           eligible_protected_known_count=len(protected), protected_leaves=leaves))
    return dict(action=action, scope="shared_full_taxonomy", changed_count=_changed(before, after),
                benefiting_sources=fast["gains"], benefiting_source_count=len(fast["gains"]),
                parent_regions=scopes, known_added_harm=fast["known_added_harm"],
                near_added_harm=fast["near_added_harm"], near_parent_path_loss=fast["near_parent_path_loss"],
                matched_region_audit=matched_rule_support(before, before, after, meta, [action]),
                preservation_passed=fast["passed"])


def _fit_family(rows, baseline, meta, options, weights):
    router = make_router(baseline, meta, options["mode"])
    original = apply_router(rows, router, meta)
    current, supports, tried = original, [], 0
    grid_points, minimum = options["grid_points"], options["min_rule_sources"]
    if weights is not None:
        cand = parent_candidates(rows, meta, weights)
        scope = np.asarray([r["prediction_type"] != "known" for r in original])
        best = None
        for threshold in _grid(cand["score"][scope], grid_points):
            for margin in _grid(cand["margin"][scope], grid_points, nonnegative=True):
                trial_router = copy.copy(router)
                trial_router["parent_rule"] = dict(weights=list(weights), threshold=threshold, margin_threshold=margin)
                selected = apply_router(rows, trial_router, meta)
                audit = _fast_audit(original, selected)
                tried += 1
                if not audit["passed"] or not audit["improved"] or len(audit["gains"]) < minimum:
                    continue
                key = _quality(selected, _changed(original, selected)) + (-threshold, -margin)
                if best is None or key > best[0]:
                    best = key, trial_router, selected
        if best is not None:
            _, router, current = best
            supports.append(_support(original, current, meta, "parent"))
    if options["mode"] in ("audit", "combined"):
        e = evidence_arrays(rows, meta)
        for action in ("leaf_reject", "root_reject"):
            if action == "leaf_reject":
                scope = np.asarray([r["prediction_type"] == "known" for r in current])
                x = np.asarray([r["baseline_leaf_z"] for r in rows])
                y = np.asarray([r["geometry_leaf_score"] for r in rows])
            else:
                scope = np.asarray([r["prediction_type"] != "global_unknown" for r in current])
                parents = np.asarray([r["route_parent"] for r in current])
                x, y = (e[k][np.arange(len(rows)), parents] for k in ("membership_z", "geometry_z"))
            best = None
            for xt in _grid(x[scope], grid_points):
                for yt in _grid(y[scope], grid_points):
                    trial_router = copy.copy(router)
                    trial_router["reject_rules"] = router["reject_rules"] + [dict(action=action, x_max=xt, y_max=yt)]
                    selected = apply_router(rows, trial_router, meta)
                    audit, step = _fast_audit(original, selected), _fast_audit(current, selected)
                    tried += 1
                    if not (audit["passed"] and step["passed"] and step["improved"] and len(step["gains"]) >= minimum):
                        continue
                    key = _quality(selected, _changed(current, selected)) + (-xt, -yt)
                    if best is None or key > best[0]:
                        best = key, trial_router, selected
            if best is not None:
                before = current
                _, router, current = best
                supports.append(_support(before, current, meta, action))
    router["rule_support"] = supports
    router["search"] = dict(sampled_candidates=tried, fit_image_sha256=sorted(base._digest(r) for r in rows),
                            local_parent_rules=False, absolute_gate_blocks_search=False)
    return router


def _refit_baseline(rows, meta, options):
    groups = [[r for r in rows if r["status"] == status] for status in base.STATUSES]
    if any(not g for g in groups):
        raise ValueError("Every membership refit requires nonempty fit known, near and extra")
    bc = dict(options["baseline_calibration"], source_loo=False)
    return membership.calibrate(*groups, meta, bc)


def _partition(rows, fold):
    held = set(fold["held_image_sha256"])
    return ([r for r in rows if base._digest(r) not in held],
            [r for r in rows if base._digest(r) in held])


def _nested_fit(rows, baseline, meta, options):
    """Select among fixed families using internal OOF evidence, then refit."""
    plan = build_folds(rows, meta, options["inner_folds"], options["seed"])
    usable = [f for f in plan["folds"] if f["usable"]]
    if len(usable) != len(plan["folds"]) or not any(f["known_held_n"] for f in usable):
        router = make_router(baseline, meta, options["mode"])
        router.update(status="not_searched", rule_support=[], inner_selection=dict(
            status="not_searched", reason="insufficient_fit_or_held_known_source_coverage", fold_plan=plan,
            candidates=[], selected_family=None))
        return router
    cache = []
    for fold in usable:
        fitted, held = _partition(rows, fold)
        fold_baseline = _refit_baseline(fitted, meta, options)
        cache.append((fold, fitted, held, fold_baseline))
    weights = [None] if options["mode"] == "audit" else list(PARENT_WEIGHTS)
    candidates, best = [], None
    for index, weight in enumerate(weights):
        before, after, fold_records = [], [], []
        for fold, fitted, held, fb in cache:
            selected = _fit_family(fitted, fb, meta, options, weight)
            b = base.apply_router(held, fb, meta)
            a = apply_router(held, selected, meta)
            before.extend(b)
            after.extend(a)
            fold_records.append(dict(fold_id=fold["fold_id"],
                fit_image_sha256=sorted(base._digest(r) for r in fitted),
                held_image_sha256=sorted(base._digest(r) for r in held),
                baseline_fit_image_sha256=fb["fit_image_sha256"],
                baseline_calibration_sha256=fb["calibration_sha256"],
                parent_rule=selected["parent_rule"], reject_rules=selected["reject_rules"],
                rule_support=selected["rule_support"], search=selected["search"]))
        audit = _fast_audit(before, after)
        admissible = (audit["passed"] and audit["improved"] and len(audit["gains"]) >= options["min_rule_sources"])
        record = dict(family_index=index, weights=None if weight is None else list(weight),
                      admissible=bool(admissible), audit=paired_report(before, after, meta), folds=fold_records)
        candidates.append(record)
        if admissible:
            key = _quality(after, _changed(before, after)) + (-index,)
            if best is None or key > best[0]:
                best = key, index, weight
    if best is None:
        router = make_router(baseline, meta, options["mode"])
        router.update(status="no_feasible_candidate", rule_support=[])
        chosen = None
    else:
        _, chosen, weight = best
        router = _fit_family(rows, baseline, meta, options, weight)
        final_before = base.apply_router(rows, baseline, meta)
        final_after = apply_router(rows, router, meta)
        final_audit = _fast_audit(final_before, final_after)
        if not final_audit["passed"]:
            raise RuntimeError("Composed full-fit candidate violated preservation safeguards")
        router["status"] = "selected" if router["parent_rule"] or router["reject_rules"] else "no_feasible_candidate"
    router["inner_selection"] = dict(status=router["status"], fold_plan=plan,
                                     candidates=candidates, selected_family=chosen,
                                     scope="internal model selection; not independent performance evaluation")
    return router


def _outer_audit(rows, meta, options):
    plan = build_folds(rows, meta, options["outer_folds"], options["seed"])
    folds, before, after = [], [], []
    for fold in plan["folds"]:
        if not fold["usable"]:
            folds.append(dict(fold, status="not_searched", reason=fold.get("reason", "insufficient_fit_coverage")))
            continue
        fitted, held = _partition(rows, fold)
        baseline = _refit_baseline(fitted, meta, options)
        router = _nested_fit(fitted, baseline, meta, options)
        b, a = base.apply_router(held, baseline, meta), apply_router(held, router, meta)
        before.extend(b)
        after.extend(a)
        report = paired_report(b, a, meta)
        degraded = not _fast_audit(b, a)["passed"]
        folds.append(dict(fold, status="heldout_degraded" if degraded else router["status"],
                          baseline_refit_on_fit_only=True, baseline_fit_image_sha256=baseline["fit_image_sha256"],
                          baseline_calibration_sha256=baseline["calibration_sha256"],
                          parent_rule=router["parent_rule"], reject_rules=router["reject_rules"],
                          inner_selection=router["inner_selection"], rule_support=router["rule_support"],
                          report=report))
    expected = {base._digest(r) for r in rows}
    observed = [base._digest(r) for r in after]
    complete = len(observed) == len(expected) and set(observed) == expected
    if len(observed) != len(set(observed)):
        raise RuntimeError("Outer predictions must contain one output per unique image")
    report = paired_report(before, after, meta) if after else None
    status = "not_searched" if not complete else ("heldout_degraded" if not _fast_audit(before, after)["passed"] else "completed")
    return dict(fold_plan=plan, folds=folds, baseline_predictions=before, predictions=after,
                report=report, complete=complete, status=status,
                unique_images=len(expected), scored_images=len(observed),
                output_used_for_selection=False, validation_scope=VALIDATION_SCOPE,
                independent_model_level_validation=False,
                interpretation="Audit of a postprocessor conditional on a frozen reference selected using DEV. No full-model independence or statistical safety guarantee.")


def calibrate(known, near, extra, baseline_router, meta, options=None):
    options = settings(options)
    groups = [list(known), list(near), list(extra)]
    unique_records(sum(groups, []))
    rows, input_count = geometry._fit_inputs(*groups, meta)
    rows = sorted(rows, key=base._digest)
    evidence_arrays(rows, meta)
    membership._validate_state(baseline_router, meta)
    # Deployment selection finishes BEFORE the independent outer audit. No
    # outer metric ever controls this returned router's rules or status.
    router = _nested_fit(rows, baseline_router, meta, options)
    before = base.apply_router(rows, baseline_router, meta)
    after = apply_router(rows, router, meta)
    report = paired_report(before, after, meta)
    gates = base.evaluate_records(after, meta)
    router.update(fit_completed=True, targets=copy.deepcopy(base.TARGETS),
                  targets_passed=gates["targets_passed"], best_effort=not gates["targets_passed"],
                  baseline_fallback=not bool(router["parent_rule"] or router["reject_rules"]),
                  selection_status=router["status"], target_status="passed" if gates["targets_passed"] else "target_failed",
                  settings=options, validation_scope=VALIDATION_SCOPE, independent_model_level_validation=False,
                  validation_report=gates, baseline_validation_report=base.evaluate_records(before, meta),
                  full_fit_audit=report, full_fit_predictions=after,
                  input_record_count=input_count, unique_image_count=len(rows), duplicate_record_count=input_count-len(rows),
                  fit_image_sha256=[base._digest(r) for r in rows], fit_splits=["val_known", "val_intra", "val_extra"],
                  selection_rule="Fixed shared families selected using inner held-known/source predictions; full-fit image and path preservation; outer audit never selects deployment parameters",
                  evidence_sha256=digest(rows))
    router["outer_audit"] = _outer_audit(rows, meta, options)
    router["calibration_sha256"] = digest(router)
    return router
