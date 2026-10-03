"""Exploratory, reject-only empirical-frontier calibration.

Observed x breakpoints define maximal y thresholds excluding every protected
fit image. Equivalent masks are evaluated once. This removes the old coarse
quantile-grid omission without adding score families or relaxing safeguards.
It is a per-action protected frontier, not an exhaustive joint optimization of
two sequential rules. Three fixed action plans are selected inside DEV folds;
the outer audit never changes deployment rules or selects an action plan.
"""
import copy

import numpy as np

from taxosafe_geometry import calibration as geometry
from taxosafe_support import calibration as base
from taxosafe_support import membership_calibration as membership
from taxosafe_parentrisk import calibration as risk
from taxosafe_parentrisk.decoder import evidence_arrays
from taxosafe_parentrisk.folds import build_folds, unique_records
from taxosafe_parentrisk.reporting import paired_report, score_diagnostics
from .decoder import SCHEMA_VERSION, DECODER, apply_router, make_router, digest

VALIDATION_SCOPE = "exploratory_postprocessor_on_previously_used_development"
ACTION_PLANS = (("leaf_reject",), ("root_reject",), ("leaf_reject", "root_reject"))
CONTEXT = dict(validation_scope=VALIDATION_SCOPE, independent_model_level_validation=False,
               development_reused_for_method_design=True, confirmatory_validation=False)
SEARCH_LIMITATION = (
    "Observed x breakpoints with maximal finite y below protected fit points; "
    "duplicate masks removed. No claim of exhaustive continuous or joint two-action "
    "optimization, optimal minimum-change tie solution, statistical validity after "
    "adaptive DEV reuse, or performance guarantees on unseen data.")


def settings(options=None):
    result = dict(options or {})
    allowed = {"outer_folds", "inner_folds", "min_rule_sources", "seed", "baseline_calibration"}
    if set(result) - allowed:
        raise ValueError("Invalid frontier calibration settings")
    for key, default, low, high in (("outer_folds", 4, 2, 8), ("inner_folds", 3, 2, 8),
                                   ("min_rule_sources", 2, 2, 8)):
        result.setdefault(key, default)
        if type(result[key]) is not int or not low <= result[key] <= high:
            raise ValueError(key + " has invalid integer range")
    result.setdefault("seed", 0)
    if type(result["seed"]) is not int or not 0 <= result["seed"] < 2**31:
        raise ValueError("seed must be an integer in [0,2**31)")
    baseline = result.setdefault("baseline_calibration", {"decoder": "membership", "policy": "known_first"})
    if not isinstance(baseline, dict) or baseline.get("decoder", "membership") != "membership":
        raise ValueError("Frontier baseline calibration must use membership")
    return result


def _report(before, after, meta):
    report = paired_report(before, after, meta)
    report.update(CONTEXT)
    return report


def _fold_plan(rows, meta, count, seed):
    plan = build_folds(rows, meta, count, seed)
    plan.update(CONTEXT)
    plan["schema_version"] = "frontier_folds_v1"
    plan["sha256"] = digest({k: v for k, v in plan.items() if k not in ("sha256", "duplicate_record_count")})
    return plan


def _frontier_masks(x, y, scope, protected):
    """Return unique nonempty maximal protected-safe rectangles and diagnostics.

    Inclusive x ties are processed together before setting the strict y bound.
    An unrepresentable finite bound is skipped explicitly, never serialized as
    infinity. The helper uses fit arrays only and knows nothing about sources.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    scope, protected = np.asarray(scope, dtype=bool), np.asarray(protected, dtype=bool)
    if x.ndim != 1 or any(v.shape != x.shape for v in (y, scope, protected)) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Frontier inputs require aligned finite one-dimensional scores")
    stats = dict(eligible_count=int(scope.sum()), protected_count=int((scope & protected).sum()),
                 observed_x_breakpoints=0, empty_masks=0, duplicate_masks=0,
                 nonfinite_endpoint_skips=0, unique_safe_masks=0)
    result, seen = [], set()
    if not scope.any():
        return result, stats
    for xt in np.unique(x[scope]):
        stats["observed_x_breakpoints"] += 1
        blocked = scope & protected & (x <= xt)
        if blocked.any():
            with np.errstate(over="ignore"):
                yt = float(np.nextafter(y[blocked].min(), -np.inf))
        else:
            yt = float(y[scope].max())
        if not np.isfinite(yt):
            stats["nonfinite_endpoint_skips"] += 1
            continue
        mask = scope & (x <= xt) & (y <= yt)
        if not mask.any():
            stats["empty_masks"] += 1
            continue
        key = mask.tobytes()
        if key in seen:
            stats["duplicate_masks"] += 1
            continue
        if (mask & protected).any():
            raise AssertionError("Protected frontier includes a protected image")
        seen.add(key)
        # Tighten to the smallest observed rectangle with this exact fit mask.
        # The protected bound is an exclusion certificate, not a license to
        # reject the whole unobserved gap up to the next protected point.
        x_observed, y_observed = float(x[mask].max()), float(y[mask].max())
        if not np.array_equal(mask, scope & (x <= x_observed) & (y <= y_observed)):
            raise AssertionError("Observed frontier tightening changed the fit mask")
        result.append((x_observed, y_observed, mask))
    stats["unique_safe_masks"] = len(result)
    return result, stats


def _audit(before, after, minimum):
    fast = risk._fast_audit(before, after)
    old, old_sources = risk._summary(before)
    new, new_sources = risk._summary(after)
    source_loss = any(new_sources[k][1] < value[1] for k, value in old_sources.items())
    precision = ((new["leaf_outputs"] == 0) if old["leaf_outputs"] == 0 else
                 new["leaf_outputs"] > 0 and new["known_correct"] * old["leaf_outputs"] >=
                 old["known_correct"] * new["leaf_outputs"])
    failures = dict(known_added_harm=bool(fast["known_added_harm"]),
                    near_added_harm=bool(fast["near_added_harm"]),
                    near_parent_path_loss=bool(fast["near_parent_path_loss"]),
                    source_correct_decline=bool(source_loss), leaf_precision_decline=not bool(precision),
                    no_improvement=not fast["improved"],
                    benefiting_sources_below_minimum=len(fast["gains"]) < minimum)
    return dict(fast, admissible=not any(failures.values()),
                failure_flags=failures, rejection_reasons=[key for key, value in failures.items() if value])


def _protect(original, current, action):
    flags = []
    for before, now in zip(original, current):
        known = before["status"] == "known" and risk._correct(before)
        near_path = (before["status"] == "intra" and before["candidate_parent"] == before.get("true_parent")
                     and before["prediction_type"] != "global_unknown")
        near_correct = now["status"] == "intra" and risk._correct(now)
        flags.append(known or action == "root_reject" and (near_path or near_correct))
    return np.asarray(flags, dtype=bool)


def _fit(rows, baseline, meta, options, actions):
    rows = sorted(unique_records(rows), key=base._digest)
    if any(r.get("status") not in base.STATUSES or r.get("split") != "val_" + r["status"] for r in rows):
        raise ValueError("Frontier fit requires DEV records only; test fitting is prohibited")
    actions = tuple(actions)
    if actions not in ACTION_PLANS:
        raise ValueError("Frontier actions must be one of the fixed action plans")
    router = make_router(baseline, meta)
    original = apply_router(rows, router, meta)
    current = original
    evidence = evidence_arrays(rows, meta)
    support, action_search = [], []
    for action in actions:
        if action == "leaf_reject":
            scope = np.asarray([r["prediction_type"] == "known" for r in current])
            x, y = (np.asarray([r[key] for r in rows], dtype=float)
                    for key in ("baseline_leaf_z", "geometry_leaf_score"))
        else:
            scope = np.asarray([r["prediction_type"] != "global_unknown" for r in current])
            parent = np.asarray([r["route_parent"] for r in current])
            x, y = (evidence[key][np.arange(len(rows)), parent] for key in ("membership_z", "geometry_z"))
        candidates, stats = _frontier_masks(x, y, scope, _protect(original, current, action))
        failures, best, admissible = {}, None, 0
        for xt, yt, mask in candidates:
            trial = copy.copy(router)
            trial["reject_rules"] = router["reject_rules"] + [dict(action=action, x_max=xt, y_max=yt)]
            selected = apply_router(rows, trial, meta)
            whole = _audit(original, selected, options["min_rule_sources"])
            step = _audit(current, selected, options["min_rule_sources"])
            for name, failed in step["failure_flags"].items():
                failures[name] = failures.get(name, 0) + int(failed)
            if not whole["passed"] or not step["admissible"]:
                continue
            admissible += 1
            key = risk._quality(selected, risk._changed(current, selected)) + (-xt, -yt)
            if best is None or key > best[0]:
                best = key, trial, selected
        entry = dict(action=action, **stats, sampled_candidates=len(candidates),
                     admissible_candidates=admissible, rejection_counts=failures,
                     rejection_counts_are_nonexclusive=True,
                     status="no_eligible_samples" if not scope.any() else
                            "no_protected_safe_candidate" if not candidates else
                            "no_feasible_candidate" if best is None else "selected")
        if best is not None:
            before = current
            _, router, current = best
            support.append(risk._support(before, current, meta, action))
            entry["selected_rule"] = router["reject_rules"][-1]
        action_search.append(entry)
    if not risk._fast_audit(original, current)["passed"]:
        raise RuntimeError("Frontier rule composition violated fit preservation")
    router.update(rule_support=support, action_plan=list(actions),
                  search=dict(actions=action_search, sampled_candidates=sum(x["sampled_candidates"] for x in action_search),
                              fit_image_sha256=sorted(base._digest(r) for r in rows),
                              candidate_generator="observed_x_protected_y_frontier", limitation=SEARCH_LIMITATION,
                              absolute_gate_blocks_search=False, local_parent_rules=False))
    return router


def _nested_fit(rows, baseline, meta, options):
    plan = _fold_plan(rows, meta, options["inner_folds"], options["seed"])
    if any(not fold["usable"] for fold in plan["folds"]) or not any(f["known_held_n"] for f in plan["folds"]):
        router = make_router(baseline, meta)
        router.update(status="not_searched", rule_support=[], inner_selection=dict(status="not_searched",
            reason="insufficient_fit_or_held_known_source_coverage", fold_plan=plan, candidates=[], selected_family=None))
        return router
    cache = []
    for fold in plan["folds"]:
        fitted, held = risk._partition(rows, fold)
        cache.append((fold, fitted, held, risk._refit_baseline(fitted, meta, options)))
    candidates, best = [], None
    for index, actions in enumerate(ACTION_PLANS):
        before, after, details = [], [], []
        component_before, component_after = {a: [] for a in actions}, {a: [] for a in actions}
        for fold, fitted, held, fold_baseline in cache:
            selected = _fit(fitted, fold_baseline, meta, options, actions)
            old, new = base.apply_router(held, fold_baseline, meta), apply_router(held, selected, meta)
            before.extend(old)
            after.extend(new)
            previous, prefix = old, []
            for action in actions:
                prefix.extend(rule for rule in selected["reject_rules"] if rule["action"] == action)
                following = apply_router(held, make_router(fold_baseline, meta, prefix), meta)
                component_before[action].extend(previous)
                component_after[action].extend(following)
                previous = following
            details.append(dict(fold_id=fold["fold_id"], fit_image_sha256=sorted(base._digest(r) for r in fitted),
                held_image_sha256=sorted(base._digest(r) for r in held),
                baseline_fit_image_sha256=fold_baseline["fit_image_sha256"],
                baseline_calibration_sha256=fold_baseline["calibration_sha256"],
                parent_rule=None, reject_rules=selected["reject_rules"], rule_support=selected["rule_support"],
                search=selected["search"]))
        whole = _audit(before, after, options["min_rule_sources"])
        components = {action: dict(selection_audit=_audit(component_before[action], component_after[action], options["min_rule_sources"]),
                                  report=_report(component_before[action], component_after[action], meta)) for action in actions}
        admissible = whole["admissible"] and all(v["selection_audit"]["admissible"] for v in components.values())
        reasons = list(whole["rejection_reasons"])
        reasons.extend("component:" + action + ":" + reason for action, value in components.items()
                       for reason in value["selection_audit"]["rejection_reasons"])
        record = dict(family_index=index, action_plan=list(actions), admissible=bool(admissible),
                      status="admissible" if admissible else "heldout_degraded" if not whole["passed"] else "no_feasible_candidate",
                      rejection_reasons=reasons, selection_audit=whole, audit=_report(before, after, meta),
                      component_audits=components, folds=details)
        candidates.append(record)
        if admissible:
            key = risk._quality(after, risk._changed(before, after)) + (-len(actions), -index)
            if best is None or key > best[0]:
                best = key, index, actions
    refit_reasons = []
    if best is None:
        router, chosen = make_router(baseline, meta), None
        router.update(status="no_feasible_candidate", rule_support=[])
    else:
        _, chosen, actions = best
        router = _fit(rows, baseline, meta, options, actions)
        missing = set(actions) - {r["action"] for r in router["reject_rules"]}
        if missing:
            failed_fit = router
            router = make_router(baseline, meta)
            router.update(status="no_feasible_candidate", rule_support=[], refit_diagnostics=failed_fit["search"])
            refit_reasons = ["refit_component_not_admissible:" + a for a in sorted(missing)]
        else:
            router["status"] = "selected"
    router["inner_selection"] = dict(status=router["status"], fold_plan=plan, candidates=candidates,
                                     selected_family=chosen, refit_rejection_reasons=refit_reasons,
                                     scope="exploratory internal selection; not independent confirmatory validation",
                                     component_source_support_required=True)
    return router


def _outer_audit(rows, meta, options):
    plan = _fold_plan(rows, meta, options["outer_folds"], options["seed"])
    folds, before, after = [], [], []
    for fold in plan["folds"]:
        if not fold["usable"]:
            folds.append(dict(fold, status="not_searched"))
            continue
        fitted, held = risk._partition(rows, fold)
        baseline = risk._refit_baseline(fitted, meta, options)
        router = _nested_fit(fitted, baseline, meta, options)
        old, new = base.apply_router(held, baseline, meta), apply_router(held, router, meta)
        before.extend(old)
        after.extend(new)
        audit = _audit(old, new, options["min_rule_sources"])
        folds.append(dict(fold, status="heldout_degraded" if not audit["passed"] else router["status"],
                          baseline_refit_on_fit_only=True, baseline_fit_image_sha256=baseline["fit_image_sha256"],
                          baseline_calibration_sha256=baseline["calibration_sha256"],
                          parent_rule=None, reject_rules=router["reject_rules"], inner_selection=router["inner_selection"],
                          rule_support=router["rule_support"], selection_audit=audit, report=_report(old, new, meta)))
    expected = {base._digest(r) for r in rows}
    observed = [base._digest(r) for r in after]
    if len(observed) != len(set(observed)):
        raise RuntimeError("Outer audit must contain one prediction per unique image")
    complete = len(observed) == len(expected) and set(observed) == expected
    status = "not_searched" if not complete else "heldout_degraded" if not risk._fast_audit(before, after)["passed"] else "completed"
    return dict(fold_plan=plan, folds=folds, baseline_predictions=before, predictions=after,
                report=_report(before, after, meta) if after else None, complete=complete, status=status,
                unique_images=len(expected), scored_images=len(observed), output_used_for_selection=False,
                interpretation="Exploratory audit on previously used DEV, conditional on an already selected reference; not an independent model or method validation.", **CONTEXT)


def calibrate(known, near, extra, baseline_router, meta, options=None):
    options = settings(options)
    groups = [list(known), list(near), list(extra)]
    unique_records(sum(groups, []))
    rows, input_count = geometry._fit_inputs(*groups, meta)
    rows = sorted(rows, key=base._digest)
    evidence_arrays(rows, meta)
    membership._validate_state(baseline_router, meta)
    # Complete production selection before evaluating any outer held outputs.
    router = _nested_fit(rows, baseline_router, meta, options)
    before, after = base.apply_router(rows, baseline_router, meta), apply_router(rows, router, meta)
    gates = base.evaluate_records(after, meta)
    router.update(fit_completed=True, targets=copy.deepcopy(base.TARGETS), targets_passed=gates["targets_passed"],
                  best_effort=not gates["targets_passed"], baseline_fallback=not bool(router["reject_rules"]),
                  selection_status=router["status"], target_status="passed" if gates["targets_passed"] else "target_failed",
                  settings=options, validation_report=dict(gates, **CONTEXT),
                  baseline_validation_report=base.evaluate_records(before, meta),
                  full_fit_audit=_report(before, after, meta), full_fit_predictions=after,
                  input_record_count=input_count, unique_image_count=len(rows), duplicate_record_count=input_count-len(rows),
                  fit_image_sha256=[base._digest(r) for r in rows], fit_splits=["val_known", "val_intra", "val_extra"],
                  selection_rule="Three fixed rejection plans; fit and pooled inner per-action protection/source support; outer audit never selects deployment parameters",
                  search_limitation=SEARCH_LIMITATION, evidence_sha256=digest(rows), **CONTEXT)
    router["outer_audit"] = _outer_audit(rows, meta, options)
    router["calibration_sha256"] = digest(router)
    return router
