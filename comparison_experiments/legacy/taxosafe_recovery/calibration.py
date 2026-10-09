"""DEV-only recovery policies around the unchanged Discovery decoder.

The standard policy is delegated verbatim. Buffer calibration changes only
threshold selection, and fixed-parent calibration searches only the leaf axis.
All policies return their own executable best effort when research gates fail.
"""
import copy
import math
from collections import defaultdict

import numpy as np

from taxosafe_discovery import calibration as discovery
from taxosafe_dcbs.protocol import normalized_name

base = discovery.base
membership = discovery.membership
DEFAULT_SETTINGS = {"seed": 1, "known_target": .92}
POLICIES = ("standard", "buffer92", "fixed_parent")
CONTEXT = dict(discovery.CONTEXT, recovery_validation="exploratory_on_reused_development")


def validate_settings(settings=None):
    if settings is not None and not isinstance(settings, dict):
        raise ValueError("Recovery settings must be a mapping")
    result = dict(DEFAULT_SETTINGS, **(settings or {}))
    if set(result) != set(DEFAULT_SETTINGS):
        raise ValueError("Unexpected recovery calibration setting")
    if type(result["seed"]) is not int or not 0 <= result["seed"] < 2**31:
        raise ValueError("seed must be an integer in [0,2**31)")
    value = result["known_target"]
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value != .92:
        raise ValueError("The preregistered known_target is exactly 0.92")
    result["known_target"] = float(value)
    return result


def _settings(settings):
    return dict(discovery.DEFAULT_SETTINGS, seed=settings["seed"])


def _policy(policy):
    if policy not in POLICIES:
        raise ValueError("Unknown recovery calibration policy")


def decode_records(records, router, meta):
    """Inference remains label-free and exactly the existing validated decoder."""
    return discovery.decode_records(records, router, meta)


def _d05(rows, bundle, meta):
    if bundle is None:
        return None, None
    if not isinstance(bundle, dict) or set(bundle) != {"records", "router"}:
        raise ValueError("D05 bundle requires exactly records and router")
    router = bundle["router"]
    discovery.validate_router(router, meta)
    if router["variant"] != "global":
        raise ValueError("D05 reference must use global Discovery calibration")
    raw = discovery._unique(bundle["records"], meta)
    refs = {base._digest(row): row for row in raw}
    identities = {base._digest(row) for row in rows}
    if set(refs) != identities or set(router.get("fit_image_sha256", [])) != identities:
        raise ValueError("D05 records/router and target fit image identities differ")
    if router.get("evidence_sha256") != discovery._hash([row["discovery"] for row in raw]):
        raise ValueError("D05 score evidence differs from its fitted router")
    aligned = []
    for row in rows:
        other = refs[base._digest(row)]
        if any(row.get(key) != other.get(key) for key in ("status", "split", "source", "species", "true_leaf", "true_parent")):
            raise ValueError("D05 and target annotations differ")
        aligned.append(other)
    return decode_records(aligned, router, meta), dict(records=aligned, router=router)


def _recovery(before, after, meta):
    if before is None:
        return dict(passed=False, status="not_evaluable", exploratory_only=True,
                    replaces_four_gate_qualification=False, reason="D05 reference not supplied")
    paired = discovery._paired(before, after, meta)
    old, new = paired["reference"], paired["selected"]
    a, b = old["counts"], new["counts"]
    checks = dict(known_above_90=new["checks"]["known_end_to_end_leaf_accuracy"],
                  known_strictly_improved=b["known_correct"] > a["known_correct"],
                  near_not_worse=b["intra_correct"] >= a["intra_correct"],
                  extra_not_worse=b["extra_correct"] >= a["extra_correct"],
                  precision_above_90=new["checks"]["open_world_accepted_leaf_precision"])
    return dict(passed=bool(all(checks.values())), status="evaluated", checks=checks,
                counts=dict(d05=a, selected=b), paired_audit=paired,
                exploratory_only=True, replaces_four_gate_qualification=False,
                rule="known>90%; strictly more correct known than D05; near/extra correct counts not lower; accepted-leaf precision>90%")


def _search(rows, meta, settings, policy, d05_before, d05_bundle):
    data = discovery._data(rows, meta)
    if policy == "fixed_parent":
        reference = discovery._data(d05_bundle["records"], meta)
        if (not np.array_equal(data["parent_scores"], reference["parent_scores"])
                or not np.array_equal(data["parent"], reference["parent"])
                or not np.array_equal(data["leaf"], reference["leaf"])
                or not np.array_equal(data["leaf_parent"], data["parent"])):
            raise ValueError("Fixed-parent policy requires unchanged D05 parent scores/candidates and the leaf's own parent")
        pg = np.array([d05_bundle["router"]["global_parent_threshold"]])
    else:
        pg = discovery._boundaries(np.r_[data["parent_score"], data["leaf_parent_score"]])
    lg = discovery._boundaries(data["leaf_score"])
    pa = data["parent_score"][None, :] >= pg[:, None]
    lpa = data["leaf_parent_score"][None, :] >= pg[:, None]
    masks = {status: np.array([row["status"] == status for row in rows]) for status in base.STATUSES}
    km, nm, em = (masks[s] for s in base.STATUSES)
    kc = km & np.array([data["leaf"][i] == row["true_leaf"] for i, row in enumerate(rows)])
    nc = nm & np.array([data["parent"][i] == row["true_parent"] for i, row in enumerate(rows)])
    totals = [int(mask.sum()) for mask in (km, nm, em)]
    # Integer arithmetic avoids ceil(.92*N) rounding an exact integer upward.
    required = (92*totals[0]+99)//100
    d05_counts = base.evaluate_records(d05_before, meta)["counts"]
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        identity = row["true_leaf"] if row["status"] == "known" else normalized_name(str(row.get("source") or "unspecified"))
        groups[(row["status"], identity)].append(i)
    best = None
    counts = dict(all_four_gate_candidates=0, known_gate_candidates=0,
                  protected_candidates=0, protected_margin_candidates=0,
                  margin_candidates=0)
    for lt in lg:
        leaves = lpa & (data["leaf_score"][None, :] >= lt)
        parents = pa & ~leaves
        roots = ~pa & ~leaves
        correct = (leaves & kc) | (parents & nc) | (roots & em)
        kn, nn, en, ln = [value.sum(1) for value in (leaves & kc, parents & nc, roots & em, leaves)]
        kpass = kn*10 > 9*totals[0]
        ppass = (ln > 0) & (kn*10 > 9*ln)
        allpass = kpass & (nn*20 >= 17*totals[1]) & (en*10 > 9*totals[2]) & ppass
        protected = (nn >= d05_counts["intra_correct"]) & (en >= d05_counts["extra_correct"]) & ppass
        margin = kn >= required
        for key, value in (("all_four_gate_candidates", allpass), ("known_gate_candidates", kpass),
                           ("protected_candidates", protected), ("protected_margin_candidates", protected & margin),
                           ("margin_candidates", margin)):
            counts[key] += int(value.sum())
        ppv = np.divide(kn, ln, out=np.zeros(len(pg)), where=ln != 0)
        deficit = (np.maximum(0, 9*totals[0]//10+1-kn)/totals[0]
                   + np.maximum(0, (17*totals[1]+19)//20-nn)/totals[1]
                   + np.maximum(0, 9*totals[2]//10+1-en)/totals[2]
                   + np.maximum(0, 9*ln//10+1-kn)/np.maximum(ln, 1))
        macros = {s: np.zeros(len(pg)) for s in base.STATUSES}
        n_groups = defaultdict(int)
        for (status, _), indices in groups.items():
            macros[status] += correct[:, indices].sum(1)/len(indices)
            n_groups[status] += 1
        quality = (sum(macros[s]/n_groups[s] for s in base.STATUSES) + ppv)/4.
        if policy == "buffer92":
            # If no protected point exists, this becomes the unchanged
            # standard best-effort ordering on this arm's own evidence.
            safe = protected.astype(int)
            safety_margin = (protected & margin).astype(int)
            standard_feasible = ((~protected) & allpass).astype(int)
            order = np.lexsort((pg, quality, -deficit, kpass.astype(int), standard_feasible, safety_margin, safe))
            i = int(order[-1])
            key = (int(safe[i]), int(safety_margin[i]), int(standard_feasible[i]), bool(kpass[i]),
                   -float(deficit[i]), float(quality[i]), float(pg[i]), float(lt))
        else:
            order = np.lexsort((pg, quality, -deficit, kpass.astype(int), allpass.astype(int)))
            i = int(order[-1])
            key = (bool(allpass[i]), bool(kpass[i]), -float(deficit[i]), float(quality[i]), float(pg[i]), float(lt))
        if best is None or key > best[0]:
            best = (key, float(pg[i]), float(lt), bool(allpass[i]), bool(protected[i]), bool(margin[i]))
    return best, dict(parent_grid=pg.tolist(), leaf_grid=lg.tolist(), candidate_count=len(pg)*len(lg),
                     known_required_for_margin=required, **counts,
                     includes_all_leaf_accept_and_reject=True,
                     includes_all_parent_accept_and_reject=policy != "fixed_parent",
                     parent_threshold_fitted=policy != "fixed_parent",
                     scope="all empirical thresholds in this policy's allowed axes; fixed supplied candidates")


def fit_router(known, near, extra, meta, settings=None, policy="standard", reference_records=None, d05_records=None):
    _policy(policy)
    settings = validate_settings(settings)
    input_groups = [list(known), list(near), list(extra)]
    rows, _ = discovery._inputs(*input_groups, meta)
    before, d05_bundle = _d05(rows, d05_records, meta)
    if policy != "standard" and d05_bundle is None:
        raise ValueError("Recovery buffer/fixed-parent calibration requires D05 records and router")
    router, report = discovery.fit_router(*input_groups, meta, _settings(settings), "global", reference_records)
    policy_status, search = "standard", None
    if policy != "standard":
        best, search = _search(rows, meta, settings, policy, before, d05_bundle)
        _, pt, lt, feasible, protected, margin = best
        p = len(meta["parent_names"])
        router = copy.deepcopy(router)
        router.update(global_parent_threshold=pt, global_leaf_threshold=lt,
                      parent_thresholds=[pt]*p, leaf_thresholds=[lt]*p,
                      targets_passed=feasible, status="feasible" if feasible else "best_effort", best_effort=not feasible)
        router["router_sha256"] = discovery._router_hash(router)
        predictions = decode_records(rows, router, meta)
        scored = base.evaluate_records(predictions, meta)
        if bool(scored["targets_passed"]) != feasible:
            raise AssertionError("Recovery search and decoded metrics disagree")
        original, _ = discovery._reference(rows, reference_records, meta)
        paired = discovery._paired(original, predictions, meta)
        report = dict(scored, **CONTEXT, status=router["status"], best_effort=router["best_effort"],
                      paired_audit=paired, known_count_preserved=None if paired is None else paired["known_count_preserved"],
                      exact_threshold_search=search,
                      selection_rule="D05 protection; coverage92; known>90; four-gate normalized deficit; source macro and precision" if policy == "buffer92" else
                                     "fixed D05 parent threshold; four gates; known>90; normalized deficit; source macro and precision")
        for key in ("input_record_count", "unique_image_count", "duplicate_record_count"):
            report[key] = router[key]
        if policy == "buffer92":
            policy_status = "protection_infeasible" if not protected else "coverage_satisfied" if margin else "margin_infeasible"
        else:
            policy_status = "fixed_parent"
            if any(a["prediction_type"] == "global_unknown" and b["prediction_type"] != "global_unknown" for a, b in zip(before, predictions)):
                raise AssertionError("Fixed-parent calibration changed a D05 root output")
    predictions = decode_records(rows, router, meta)
    report["known_recovery"] = _recovery(before, predictions, meta)
    report["recovery_passed"] = report["known_recovery"]["passed"]
    report["recovery_policy"] = dict(name=policy, status=policy_status, settings=settings,
        d05_router_sha256=None if d05_bundle is None else d05_bundle["router"]["router_sha256"],
        d05_fit_image_count=None if d05_bundle is None else len(d05_bundle["records"]),
        known_target=settings["known_target"] if policy == "buffer92" else None,
        baseline_fallback=False, test_allowed_after_failed_gates=True,
        score_or_candidate_changed=False,
        root_outputs_preserved_on_fit=True if policy == "fixed_parent" else None)
    discovery._hash(report)
    return router, report


def crossfit_audit(known, near, extra, meta, settings=None, policy="standard", reference_records=None, d05_records=None):
    _policy(policy)
    settings = validate_settings(settings)
    rows, _ = discovery._inputs(known, near, extra, meta)
    _, original = discovery._reference(rows, reference_records, meta)
    _, d05_bundle = _d05(rows, d05_records, meta)
    if original is None or d05_bundle is None:
        return dict(**CONTEXT, passed=False, complete=False, status="not_evaluable",
                    recovery_passed=False,
                    known_recovery=dict(passed=False, status="not_evaluable"),
                    reason="Recovery crossfit requires raw original and D05 reference bundles")
    targets = {base._digest(r): r for r in rows}
    originals = {base._digest(r): r for r in original["records"]}
    d05_raw = {base._digest(r): r for r in d05_bundle["records"]}
    folds = discovery._folds(rows, settings)
    before, d05_before, after = [], [], []
    for fold in folds:
        fit, held = fold["fit_image_sha256"], fold["held_image_sha256"]
        fitted, held_rows = [targets[h] for h in fit], [targets[h] for h in held]
        original_fit, original_held = [originals[h] for h in fit], [originals[h] for h in held]
        d05_fit, d05_held = [d05_raw[h] for h in fit], [d05_raw[h] for h in held]
        groups = [[r for r in fitted if r["status"] == status] for status in base.STATUSES]
        if not held or any(not group for group in groups):
            fold.update(status="not_evaluable", reason="empty held or missing fit status")
            continue
        reference_groups = [[r for r in original_fit if r["status"] == s] for s in base.STATUSES]
        ref_router = membership.calibrate(*reference_groups, meta, dict(original["calibration_settings"], source_loo=False))
        fold_reference = dict(records=original_fit, router=ref_router, calibration_settings=original["calibration_settings"])
        d05_groups = [[r for r in d05_fit if r["status"] == s] for s in base.STATUSES]
        d05_router, _ = discovery.fit_router(*d05_groups, meta, d05_bundle["router"]["settings"], "global")
        router, fit_report = fit_router(*groups, meta, settings, policy, fold_reference, dict(records=d05_fit, router=d05_router))
        b = membership.decode_records(original_held, ref_router, meta)
        d = decode_records(d05_held, d05_router, meta)
        a = decode_records(held_rows, router, meta)
        before.extend(b); d05_before.extend(d); after.extend(a)
        fold.update(status="completed", reference_fit_image_sha256=ref_router["fit_image_sha256"],
                    d05_fit_image_sha256=d05_router["fit_image_sha256"], target_fit_image_sha256=router["fit_image_sha256"],
                    reference_calibration_sha256=ref_router["calibration_sha256"], d05_router_sha256=d05_router["router_sha256"],
                    router_sha256=router["router_sha256"], global_parent_threshold=router["global_parent_threshold"],
                    global_leaf_threshold=router["global_leaf_threshold"], d05_parent_threshold=d05_router["global_parent_threshold"],
                    fit_targets_passed=fit_report["targets_passed"], fit_recovery_policy=fit_report["recovery_policy"],
                    held_report=discovery._paired(b, a, meta), held_known_recovery=_recovery(d, a, meta))
    evaluated = [base._digest(r) for r in after]
    if len(evaluated) != len(set(evaluated)):
        raise AssertionError("Recovery crossfit image evaluated twice")
    complete = set(evaluated) == set(targets)
    paired = discovery._paired(before, after, meta) if after else None
    recovery = _recovery(d05_before, after, meta) if after else dict(passed=False, status="not_evaluable")
    recovery["passed"] = bool(complete and recovery["passed"])
    scored = None if paired is None else paired["selected"]
    return dict(**CONTEXT, schema_version="recovery_crossfit_v1", policy=policy, folds=folds,
                passed=bool(complete and paired and scored["targets_passed"] and paired["known_count_preserved"]),
                complete=complete, status="completed" if complete else "not_evaluable",
                unique_image_count=len(rows), evaluated_image_count=len(after), report=scored,
                counts=None if paired is None else dict(reference=paired["reference"]["counts"], selected=scored["counts"]),
                paired_audit=paired, known_count_preserved=None if paired is None else paired["known_count_preserved"],
                known_recovery=recovery, reference_predictions=before, d05_predictions=d05_before, predictions=after,
                recovery_passed=recovery["passed"],
                output_used_for_threshold_selection=False, source_protection_is_diagnostic=True,
                held_data_used_for_grid_or_offsets=False, full_development_d05_threshold_used_in_fold=False,
                qualification_rule="complete conditional OOF; four gates; known count at least fold-refit original reference",
                interpretation="3 known folds plus whole unknown sources; both references refitted on fit only; fixed model/evidence so no independent model-validation claim")
