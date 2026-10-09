"""Exact, explicitly tiered calibration of frozen node evidence.

Known accuracy and accepted-leaf precision define feasible domains; open-set
preferences never override an available known-accuracy domain. Pareto points
describe empirical tradeoffs, not population or conformal guarantees.
"""
import copy
from collections import defaultdict
from fractions import Fraction

import numpy as np

from taxosafe_discovery import calibration as discovery
from taxosafe_recovery import calibration as recovery
from taxosafe_dcbs.protocol import normalized_name

base = discovery.base
membership = discovery.membership
DEFAULT_SETTINGS = dict(recovery.DEFAULT_SETTINGS)
POLICIES = ("standard", "kp_frontier", "fixed_parent")
CONTEXT = dict(discovery.CONTEXT, boundary_validation="exploratory_on_reused_development")
ORDER = ("four gates; otherwise known>90% and precision>90%; otherwise known>90%; otherwise maximum known count. "
         "First two domains: capped known count at ceil(0.92*N), then smallest worst near/extra source-macro decline "
         "relative to D05, then mean known-leaf/near-source/extra-source macros and precision. Known-only domain: "
         "precision first, then the same priorities. Final domain: known count, precision, open decline, quality. "
         "Remaining ties: actual known, precision, near correct, extra correct, fewer leaf outputs, higher parent then leaf threshold.")


def validate_settings(settings=None):
    return recovery.validate_settings(settings)


def _policy(policy):
    if policy not in POLICIES:
        raise ValueError("Unknown Boundary calibration policy")


def decode_records(records, router, meta):
    return discovery.decode_records(records, router, meta)


def _masks(counts, totals):
    k, n, e, l = counts.T
    known = 10*k > 9*totals[0]
    precision = (l > 0) & (10*k > 9*l)
    four = known & precision & (20*n >= 17*totals[1]) & (10*e > 9*totals[2])
    return four, known & precision, known


def _choose(counts, quality, regret, parent_thresholds, leaf_thresholds, totals):
    """Pure selector: strict gates use integer arithmetic, including 90% ties."""
    k, n, e, l = counts.T
    four, kp, known = _masks(counts, totals)
    required = (92*totals[0]+99)//100
    capped = np.minimum(k, required)
    precision = np.divide(k, l, out=np.zeros(len(k)), where=l != 0)
    if four.any():
        tier, domain, reason = "four_gates", four, "all_four_gates_feasible"
    elif kp.any():
        tier, domain, reason = "known_precision", kp, "open_gates_not_jointly_feasible"
    elif known.any():
        tier, domain, reason = "known_only", known, "precision_infeasible_given_known_gate"
    else:
        tier, domain, reason = "maximum_known", k == k.max(), "known_gate_infeasible"
    ids = np.flatnonzero(domain)
    # np.lexsort uses the final key as primary. Undefined precision is zero
    # for tie utility only and never passes its gate or enters Pareto export.
    common = (leaf_thresholds, parent_thresholds, -l, e, n, precision, k, quality, -regret)
    if tier in ("four_gates", "known_precision"):
        keys = common + (capped,)
    elif tier == "known_only":
        keys = common + (capped, precision)
    else:
        keys = common + (precision, k)
    index = int(ids[np.lexsort(tuple(value[ids] for value in keys))[-1]])
    feasible = dict(four_gates=int(four.sum()), known_precision=int(kp.sum()), known=int(known.sum()),
                    maximum_known=int(k.max()), maximum_known_candidates=int((k == k.max()).sum()),
                    known_target_candidates=int((k >= required).sum()),
                    known_target_and_precision_candidates=int(((k >= required) & kp).sum()))
    return index, dict(selected_tier=tier, reason=reason, feasible_counts=feasible,
                       selected_domain_candidates=len(ids), known_required_for_target=required,
                       selection_rule=ORDER, precision_undefined_utility=0.)


def _enumerate(rows, meta, before, fixed_parent=None):
    data = discovery._data(rows, meta)
    pg = (np.array([fixed_parent]) if fixed_parent is not None else
          discovery._boundaries(np.r_[data["parent_score"], data["leaf_parent_score"]]))
    lg = discovery._boundaries(data["leaf_score"])
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        identity = row["true_leaf"] if row["status"] == "known" else normalized_name(str(row.get("source") or "unspecified"))
        groups[(row["status"], identity)].append(i)
    km = np.array([r["status"] == "known" for r in rows])
    nm = np.array([r["status"] == "intra" for r in rows])
    em = np.array([r["status"] == "extra" for r in rows])
    kc = km & np.array([data["leaf"][i] == r["true_leaf"] for i, r in enumerate(rows)])
    nc = nm & np.array([data["parent"][i] == r["true_parent"] for i, r in enumerate(rows)])
    old_correct = np.array([r["prediction_type"] == ("intra_unknown" if r["status"] == "intra" else "global_unknown")
                            and (r["status"] != "intra" or r["parent"] == r["true_parent"]) for r in before])
    reference_macro = {s: float(np.mean([old_correct[indices].mean() for (status, _), indices in groups.items() if status == s]))
                       for s in ("intra", "extra")}
    pa = data["parent_score"][None, :] >= pg[:, None]
    lpa = data["leaf_parent_score"][None, :] >= pg[:, None]
    counts, macros = [], []
    for lt in lg:
        leaves = lpa & (data["leaf_score"][None, :] >= lt)
        parents = pa & ~leaves
        roots = ~pa & ~leaves
        correct = (leaves & kc) | (parents & nc) | (roots & em)
        counts.append(np.column_stack([value.sum(1) for value in (leaves & kc, parents & nc, roots & em, leaves)]))
        macros.append(np.column_stack([np.mean([correct[:, indices].mean(1) for (s, _), indices in groups.items() if s == status], axis=0)
                                       for status in base.STATUSES]))
    counts = np.vstack(counts)
    macros = np.vstack(macros)
    precision = np.divide(counts[:, 0], counts[:, 3], out=np.zeros(len(counts)), where=counts[:, 3] != 0)
    regret = np.maximum(0., np.maximum(reference_macro["intra"]-macros[:, 1], reference_macro["extra"]-macros[:, 2]))
    return dict(counts=counts, macros=macros, quality=(macros.sum(1)+precision)/4., regret=regret,
                parent=np.tile(pg, len(lg)), leaf=np.repeat(lg, len(pg)), parent_grid=pg, leaf_grid=lg,
                totals=[int(km.sum()), int(nm.sum()), int(em.sum())], d05_macro=reference_macro,
                parent_threshold_fitted=fixed_parent is None)


def _point(values, index):
    k, n, e, l = (int(x) for x in values["counts"][index])
    return dict(parent_threshold=float(values["parent"][index]), leaf_threshold=float(values["leaf"][index]),
                known_correct=k, intra_correct=n, extra_correct=e, leaf_outputs=l,
                accepted_leaf_precision=None if l == 0 else k/l,
                known_macro=float(values["macros"][index, 0]), near_macro=float(values["macros"][index, 1]),
                extra_macro=float(values["macros"][index, 2]), worst_open_macro_decline=float(values["regret"][index]),
                macro_precision_quality=float(values["quality"][index]))


def _pareto_indices(counts, quality, regret, parent_thresholds, leaf_thresholds):
    """One witness per nondominated four-metric vector; exact rational PPV.

Sorted known/near/extra counts ensure no later point can dominate an earlier
one. Precision comparisons use integer products, avoiding 90% roundoff.
"""
    representatives = {}
    order = np.lexsort((leaf_thresholds, parent_thresholds, quality, -regret))
    for index in order:
        if counts[index, 3]:
            representatives[tuple(int(v) for v in counts[index])] = int(index)
    ordered = sorted(representatives.values(), key=lambda i: (
        int(counts[i, 0]), int(counts[i, 1]), int(counts[i, 2]),
        Fraction(int(counts[i, 0]), int(counts[i, 3])), -int(counts[i, 3])), reverse=True)
    kept, vectors = [], np.empty((len(ordered), 4), dtype=np.int64)
    for i in ordered:
        current = counts[i]
        previous = vectors[:len(kept)]
        dominated = ((previous[:, 1] >= current[1]) & (previous[:, 2] >= current[2])
                     & (previous[:, 0]*current[3] >= current[0]*previous[:, 3]))
        if dominated.any():
            continue
        vectors[len(kept)] = current
        kept.append(i)
    return kept


def _fixed_parent(rows, d05_bundle, meta):
    data, reference = discovery._data(rows, meta), discovery._data(d05_bundle["records"], meta)
    if (not np.array_equal(data["parent_scores"], reference["parent_scores"])
            or not np.array_equal(data["parent"], reference["parent"])
            or not np.array_equal(data["leaf"], reference["leaf"])
            or not np.array_equal(data["leaf_parent"], data["parent"])):
        raise ValueError("Fixed-parent Boundary policy requires unchanged D05 parent scores/candidates and the leaf's own parent")
    return d05_bundle["router"]["global_parent_threshold"]


def fit_router(known, near, extra, meta, settings=None, policy="kp_frontier", reference_records=None, d05_records=None):
    _policy(policy)
    settings = validate_settings(settings)
    input_groups = [list(known), list(near), list(extra)]
    rows, _ = discovery._inputs(*input_groups, meta)
    before, d05_bundle = recovery._d05(rows, d05_records, meta)
    if d05_bundle is None:
        raise ValueError("Boundary calibration requires the D05 reference bundle")
    # Build the standard router without changing a byte of the original policy.
    # Custom policies replace only their own selected thresholds and rebind hash.
    router, legacy_report = discovery.fit_router(*input_groups, meta, recovery._settings(settings), "global", reference_records)
    fixed = _fixed_parent(rows, d05_bundle, meta) if policy == "fixed_parent" else None
    values = _enumerate(rows, meta, before, fixed)
    index, selection = _choose(values["counts"], values["quality"], values["regret"], values["parent"], values["leaf"], values["totals"])
    best_kp = _point(values, index)
    if policy == "standard":
        match = np.flatnonzero((values["parent"] == router["global_parent_threshold"]) & (values["leaf"] == router["global_leaf_threshold"]))
        if len(match) != 1:
            raise AssertionError("Standard router absent from its empirical boundary grid")
        index = int(match[0])
    else:
        pt, lt = float(values["parent"][index]), float(values["leaf"][index])
        p = len(meta["parent_names"])
        feasible = bool(_masks(values["counts"][[index]], values["totals"])[0][0])
        router = copy.deepcopy(router)
        router.update(global_parent_threshold=pt, global_leaf_threshold=lt, parent_thresholds=[pt]*p, leaf_thresholds=[lt]*p,
                      targets_passed=feasible, status="feasible" if feasible else "best_effort", best_effort=not feasible)
        router["router_sha256"] = discovery._router_hash(router)
    predictions = decode_records(rows, router, meta)
    scored = base.evaluate_records(predictions, meta)
    expected = values["counts"][index].tolist()
    actual = [scored["counts"][key] for key in ("known_correct", "intra_correct", "extra_correct", "leaf_outputs")]
    if expected != actual or bool(scored["targets_passed"]) != router["targets_passed"]:
        raise AssertionError("Boundary search and decoder disagree")
    original, _ = discovery._reference(rows, reference_records, meta)
    paired = discovery._paired(original, predictions, meta)
    report = dict(legacy_report) if policy == "standard" else dict(scored, **CONTEXT,
        status=router["status"], best_effort=router["best_effort"], paired_audit=paired,
        known_count_preserved=None if paired is None else paired["known_count_preserved"], selection_rule=ORDER)
    for key in ("input_record_count", "unique_image_count", "duplicate_record_count"):
        report[key] = router[key]
    frontier = _pareto_indices(values["counts"], values["quality"], values["regret"], values["parent"], values["leaf"])
    recovery_report = recovery._recovery(before, predictions, meta)
    same_metric = lambda i: bool(np.array_equal(values["counts"][i, :3], values["counts"][index, :3]) and
                                values["counts"][i, 0]*values["counts"][index, 3] == values["counts"][index, 0]*values["counts"][i, 3])
    report.update(known_recovery=recovery_report, recovery_passed=recovery_report["passed"],
                  exact_threshold_search=dict(parent_grid=values["parent_grid"].tolist(), leaf_grid=values["leaf_grid"].tolist(),
                      candidate_count=len(values["counts"]), parent_threshold_fitted=values["parent_threshold_fitted"],
                      includes_all_leaf_accept_and_reject=True, includes_all_parent_accept_and_reject=fixed is None,
                      feasible_counts=selection["feasible_counts"], scope="all empirical boundaries for frozen supplied candidates and this policy's axes"),
                  boundary_policy=dict(name=policy, settings=settings, **selection,
                      available_top_tier=selection["selected_tier"], legacy_selection_unchanged=policy == "standard",
                      selected_point=_point(values, index), kp_policy_point=best_kp if policy == "standard" else None,
                      d05_router_sha256=d05_bundle["router"]["router_sha256"], d05_macro=values["d05_macro"],
                      baseline_fallback=False, test_allowed_after_failed_gates=True,
                      known_target_attained=actual[0] >= selection["known_required_for_target"],
                      root_outputs_preserved_on_fit=True if fixed is not None else None),
                  pareto_frontier=dict(points=[_point(values, i) for i in frontier], point_count=len(frontier),
                      selected_is_pareto=None if not actual[3] else any(same_metric(i) for i in frontier),
                      dimensions=["known_correct", "intra_correct", "extra_correct", "accepted_leaf_precision"],
                      scope="defined-precision candidates only; one threshold witness per four-metric vector, not per prediction mask",
                      undefined_precision_candidates=int((values["counts"][:, 3] == 0).sum()),
                      used_to_override_policy=False))
    if policy == "standard":
        checks = scored["checks"]
        report["boundary_policy"]["legacy_gate_domain"] = ("four_gates" if scored["targets_passed"] else
            "known_precision" if checks["known_end_to_end_leaf_accuracy"] and checks["open_world_accepted_leaf_precision"] else
            "known_only" if checks["known_end_to_end_leaf_accuracy"] else "known_gate_failed")
        report["boundary_policy"]["selected_tier"] = "legacy_standard"
        report["boundary_policy"]["selected_domain_candidates"] = None
        report["boundary_policy"]["reason"] = "unchanged_legacy_selection"
    if fixed is not None and any(a["prediction_type"] == "global_unknown" and b["prediction_type"] != "global_unknown" for a, b in zip(before, predictions)):
        raise AssertionError("Fixed D05 parent route changed a D05 root output")
    discovery._hash(report)
    return router, report


def crossfit_audit(known, near, extra, meta, settings=None, policy="kp_frontier", reference_records=None, d05_records=None):
    _policy(policy)
    settings = validate_settings(settings)
    rows, _ = discovery._inputs(known, near, extra, meta)
    _, original = discovery._reference(rows, reference_records, meta)
    _, d05_bundle = recovery._d05(rows, d05_records, meta)
    if original is None or d05_bundle is None:
        return dict(**CONTEXT, passed=False, recovery_passed=False, complete=False, status="not_evaluable",
                    known_recovery=dict(passed=False, status="not_evaluable"), reason="Raw original and D05 reference bundles required")
    target_map = {base._digest(r): r for r in rows}
    ref_map = {base._digest(r): r for r in original["records"]}
    d05_map = {base._digest(r): r for r in d05_bundle["records"]}
    folds = discovery._folds(rows, settings)
    before, d05_before, after = [], [], []
    for fold in folds:
        fit, held = fold["fit_image_sha256"], fold["held_image_sha256"]
        fitted, held_rows = [target_map[h] for h in fit], [target_map[h] for h in held]
        ref_fit, ref_held = [ref_map[h] for h in fit], [ref_map[h] for h in held]
        d05_fit, d05_held = [d05_map[h] for h in fit], [d05_map[h] for h in held]
        groups = [[r for r in fitted if r["status"] == s] for s in base.STATUSES]
        if not held or any(not group for group in groups):
            fold.update(status="not_evaluable", reason="empty held or missing fit status")
            continue
        ref_groups = [[r for r in ref_fit if r["status"] == s] for s in base.STATUSES]
        ref_router = membership.calibrate(*ref_groups, meta, dict(original["calibration_settings"], source_loo=False))
        ref_bundle = dict(records=ref_fit, router=ref_router, calibration_settings=original["calibration_settings"])
        d05_groups = [[r for r in d05_fit if r["status"] == s] for s in base.STATUSES]
        d05_router, _ = discovery.fit_router(*d05_groups, meta, d05_bundle["router"]["settings"], "global")
        router, fit_report = fit_router(*groups, meta, settings, policy, ref_bundle, dict(records=d05_fit, router=d05_router))
        b, d, a = membership.decode_records(ref_held, ref_router, meta), decode_records(d05_held, d05_router, meta), decode_records(held_rows, router, meta)
        before.extend(b); d05_before.extend(d); after.extend(a)
        fold.update(status="completed", reference_fit_image_sha256=ref_router["fit_image_sha256"],
            d05_fit_image_sha256=d05_router["fit_image_sha256"], target_fit_image_sha256=router["fit_image_sha256"],
            reference_calibration_sha256=ref_router["calibration_sha256"], d05_router_sha256=d05_router["router_sha256"],
            router_sha256=router["router_sha256"], global_parent_threshold=router["global_parent_threshold"],
            global_leaf_threshold=router["global_leaf_threshold"], d05_parent_threshold=d05_router["global_parent_threshold"],
            fit_targets_passed=fit_report["targets_passed"], fit_boundary_policy=fit_report["boundary_policy"],
            fit_pareto_frontier=fit_report["pareto_frontier"], held_report=discovery._paired(b, a, meta),
            held_known_recovery=recovery._recovery(d, a, meta))
    evaluated = [base._digest(r) for r in after]
    if len(evaluated) != len(set(evaluated)):
        raise AssertionError("Boundary crossfit image evaluated twice")
    complete = set(evaluated) == set(target_map)
    paired = discovery._paired(before, after, meta) if after else None
    recovery_report = recovery._recovery(d05_before, after, meta) if after else dict(passed=False, status="not_evaluable")
    recovery_report["passed"] = bool(complete and recovery_report["passed"])
    scored = None if paired is None else paired["selected"]
    return dict(**CONTEXT, schema_version="boundary_crossfit_v1", policy=policy, folds=folds,
        passed=bool(complete and paired and scored["targets_passed"] and paired["known_count_preserved"]),
        recovery_passed=recovery_report["passed"], complete=complete, status="completed" if complete else "not_evaluable",
        unique_image_count=len(rows), evaluated_image_count=len(after), report=scored,
        counts=None if paired is None else dict(reference=paired["reference"]["counts"], selected=scored["counts"]),
        paired_audit=paired, known_count_preserved=None if paired is None else paired["known_count_preserved"],
        known_recovery=recovery_report, reference_predictions=before, d05_predictions=d05_before, predictions=after,
        output_used_for_threshold_selection=False, source_protection_is_diagnostic=True,
        held_data_used_for_grid_or_offsets=False, full_development_d05_threshold_used_in_fold=False,
        qualification_rule="complete conditional OOF; four gates; known count at least fold-refit original reference",
        interpretation="Fixed model/evidence; 3 known folds and whole unknown sources; both references and all thresholds fitted using fold fit only")
