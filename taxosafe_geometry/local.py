"""DEV-only, parent-conditional corrections of a frozen reference router.

No new rule can emit a leaf: known predictions retain the original leaf ID.
Each two-evidence rule edits a specific reference outcome. Statistics and score
normalizers remain TRAIN-only. Thresholds and action selection use DEV only.
Source leave-one-out refits the baseline and every rule without the held source.
These empirical safeguards are not guarantees on unseen known/unknown classes.
"""
import copy

import numpy as np

from . import calibration as geometry
from taxosafe_support import calibration as base
from taxosafe_support import membership_calibration as membership

SCHEMA_VERSION = "geometry_local_guarded_v1"
DECODER = "local_guarded"
# Recover correct parent fallbacks before fitting root rejection, so subsequent
# root rules must preserve those newly correct near-unknown decisions too.
ACTIONS = ("leaf_reject", "root_reject", "root_rescue", "parent_repair")
SELECTION_RULE = (
    "Start from reference; sequential two-evidence local rules, global then supported parents; "
    "preserve every baseline-correct known DEV image, every unknown source correct count and "
    "leaf precision; maximize near+extra+precision gain with minimal decision changes. "
    "Screen each action by refitted source LOO, then re-fit and re-check their composition.")


def settings(value=None):
    result = dict(value or {})
    allowed = {"decoder", "actions", "min_known_per_parent", "max_thresholds",
               "source_loo", "source_loo_safeguard", "baseline_calibration"}
    if set(result) - allowed or result.get("decoder", DECODER) != DECODER:
        raise ValueError("Invalid local guarded calibration settings")
    result.setdefault("actions", list(ACTIONS))
    actions = result["actions"]
    if (not isinstance(actions, (tuple, list)) or not actions or
            any(a not in ACTIONS for a in actions) or len(set(actions)) != len(actions)):
        raise ValueError("actions must be distinct supported local operations")
    result["actions"] = [a for a in ACTIONS if a in actions]
    for name, default, low, high in (("min_known_per_parent", 10, 1, 100000),
                                      ("max_thresholds", 257, 2, 4096)):
        result.setdefault(name, default)
        if type(result[name]) is not int or not low <= result[name] <= high:
            raise ValueError(name + " must be an integer in [{},{}]".format(low, high))
    for name in ("source_loo", "source_loo_safeguard"):
        result.setdefault(name, True)
        if type(result[name]) is not bool:
            raise ValueError(name + " must be boolean")
    if result["source_loo_safeguard"] and not result["source_loo"]:
        raise ValueError("source_loo_safeguard requires source_loo")
    return result


def _data(rows, baseline, meta):
    d = geometry._arrays(rows, meta)
    d["bp"], d["bl"] = geometry._baseline_masks(d, baseline)
    d["true_parent"] = np.asarray([-1 if r.get("true_parent") is None else r["true_parent"] for r in rows])
    heads, index = d["heads"], np.arange(len(rows))
    d["alternative_parent"] = np.asarray(meta["leaf_to_parent"])[heads["leaf_logits"].argmax(1)]
    # Independent fine ranking proposes a parent; only a top-two parent-rank
    # candidate with raw membership support may be considered for fallback.
    ranked = np.argsort(-heads["parent_logits"], axis=1, kind="stable")
    second = ranked[:, min(1, ranked.shape[1] - 1)]
    alt = d["alternative_parent"]
    d["alternative_membership"] = heads["parent_membership_logits"][index, alt]
    d["alternative_rank_gap"] = (heads["parent_logits"][index, d["parent"]] -
                                   heads["parent_logits"][index, alt])
    if not np.isfinite(d["alternative_rank_gap"]).all():
        raise ValueError("Parent ranking gap overflow")
    d["alternative_supported"] = ((alt != d["parent"]) & (alt == second) &
                                     (d["alternative_membership"] >= baseline["parent_threshold"]))
    d["protected"] = d["known_candidate_correct"] & d["bl"]
    return d


def _initial(d):
    return d["bp"].copy(), d["bl"].copy(), d["parent"].copy()


def _components(d, state):
    pa, la, parent = state
    kc = la & d["known_candidate_correct"]
    nc = d["intra"] & pa & ~la & (parent == d["true_parent"])
    ec = d["extra"] & ~pa
    return kc, nc, ec


def _counts(d, state):
    kc, nc, ec = _components(d, state)
    return dict({s: int(d[s].sum()) for s in ("known", "intra", "extra")},
                known_correct=int(kc.sum()), intra_correct=int(nc.sum()),
                extra_correct=int(ec.sum()), leaf_outputs=int(state[1].sum()))


def _audit(rows, d, state, meta):
    original = _initial(d)
    before, after = _components(d, original), _components(d, state)
    old, new = _counts(d, original), _counts(d, state)
    lost = before[0] & ~after[0]
    sources, leaves = [], []
    for status, index in (("intra", 1), ("extra", 2)):
        for source in sorted(set(d["sources"][d[status]].tolist())):
            group = d[status] & (d["sources"] == source)
            b, a = int((before[index] & group).sum()), int((after[index] & group).sum())
            sources.append(dict(status=status, source=source, total=int(group.sum()),
                                baseline_correct=b, selected_correct=a, passed=a >= b))
    for leaf, name in enumerate(meta["leaf_names"]):
        group = np.asarray([r["status"] == "known" and r.get("true_leaf") == leaf for r in rows])
        b, a = int((before[0] & group).sum()), int((after[0] & group).sum())
        leaves.append(dict(leaf=leaf, name=name, total=int(group.sum()), baseline_correct=b,
                           selected_correct=a, passed=a >= b))
    checks = dict(every_baseline_correct_known_preserved=not bool(lost.any()),
                  every_known_leaf_count_preserved=all(x["passed"] for x in leaves),
                  every_unknown_source_count_preserved=all(x["passed"] for x in sources),
                  precision_preserved=new["leaf_outputs"] > 0 and
                  new["known_correct"] * old["leaf_outputs"] >= old["known_correct"] * new["leaf_outputs"],
                  known_above_90_percent=new["known_correct"] * 10 > 9 * new["known"])
    return dict(passed=all(checks.values()), checks=checks, baseline_counts=old, selected_counts=new,
                lost_baseline_correct_known_sha256=[base._digest(r) for r, keep in zip(rows, lost) if keep],
                per_unknown_source=sources, per_known_leaf=leaves)


def _condition(d, state, action, parent):
    pa, la, _ = state
    scope = np.ones(len(pa), dtype=bool) if parent == -1 else d["parent"] == parent
    if action == "root_reject":
        return scope & d["bp"] & pa, d["baseline_parent_z"], d["geometry_parent_score"]
    if action == "leaf_reject":
        return scope & d["bl"] & la, d["baseline_leaf_z"], d["geometry_leaf_score"]
    if action == "root_rescue":
        return scope & ~d["bp"] & ~pa, -d["baseline_parent_z"], -d["geometry_parent_score"]
    return (scope & ~d["bl"] & ~la & d["alternative_supported"],
            d["alternative_rank_gap"], -d["alternative_membership"])


def _edit(d, state, mask, action):
    pa, la, parent = [np.broadcast_to(value, mask.shape).copy() for value in state]
    if action == "root_reject":
        pa[mask], la[mask] = False, False
    elif action == "leaf_reject":
        la[mask] = False
    elif action == "root_rescue":
        pa[mask] = True
    else:
        pa[mask] = True
        parent = np.where(mask, d["alternative_parent"], parent)
    return pa, la, parent


def _grid(values, limit):
    unique = np.unique(values)
    if len(unique) <= limit:
        return unique
    return unique[np.unique(np.rint(np.linspace(0, len(unique) - 1, limit)).astype(int))]


def _best_rule(d, state, action, parent, options):
    scope, x, y = _condition(d, state, action, parent)
    if not scope.any():
        return None, 0
    xs, ys = _grid(x[scope], options["max_thresholds"]), _grid(y[scope], options["max_thresholds"])
    kc0, nc0, ec0 = _components(d, state)
    k0, n0, e0, out0 = int(kc0.sum()), int(nc0.sum()), int(ec0.sum()), int(state[1].sum())
    if not out0 or k0 * 10 <= 9 * int(d["known"].sum()):
        return None, 0
    source_groups = [(d[s] & (d["sources"] == source), i, int((co & (d["sources"] == source)).sum()))
                     for s, i, co in (("intra", 1, nc0), ("extra", 2, ec0))
                     for source in sorted(set(d["sources"][d[s]].tolist()))]
    best = None
    for xt in xs:
        masks = scope & (x <= xt) & (y[None, :] <= ys[:, None])
        trial = _edit(d, state, masks, action)
        kc, nc, ec = _components(d, trial)
        k, n, e, outputs = kc.sum(1), nc.sum(1), ec.sum(1), trial[1].sum(1)
        valid = ~(d["protected"] & ~trial[1]).any(1)
        valid &= (outputs > 0) & (k * out0 >= k0 * outputs)
        for group, i, old in source_groups:
            valid &= ((nc if i == 1 else ec) & group).sum(1) >= old
        improved = (n > n0) | (e > e0) | (k * out0 > k0 * outputs)
        quality = ((n - n0) / d["intra"].sum() + (e - e0) / d["extra"].sum() +
                   k / np.maximum(outputs, 1) - k0 / out0)
        for j in np.flatnonzero(valid & improved):
            key = (float(quality[j]), -int(masks[j].sum()), -float(xt), -float(ys[j]))
            if best is None or key > best[0]:
                rule = dict(action=action, parent=int(parent), x_max=float(xt), y_max=float(ys[j]))
                best = key, rule, tuple(v[j].copy() for v in trial)
    return best, len(xs) * len(ys)


def _select(rows, baseline, meta, options, actions):
    d = _data(rows, baseline, meta)
    state, rules, sampled = _initial(d), [], 0
    regions = [-1] + [p for p in range(len(meta["parent_names"]))
                      if int((d["known"] & (d["parent"] == p)).sum()) >= options["min_known_per_parent"]]
    for action in actions:
        for parent in regions:
            best, count = _best_rule(d, state, action, parent, options)
            sampled += count
            if best is not None:
                _, rule, state = best
                rules.append(rule)
    audit = _audit(rows, d, state, meta)
    if rules and not audit["passed"]:
        raise ValueError("Local rule composition violated DEV preservation")
    gates = base._gates(_counts(d, state))
    return dict(rules=rules, geometry_enabled=bool(rules), preservation_audit=audit, gates=gates,
                status=("feasible_local" if gates["targets_passed"] else "best_effort_local") if rules else "baseline_fallback",
                search=dict(sampled_rule_threshold_pairs=sampled, eligible_parent_regions=regions,
                            max_thresholds_per_axis=options["max_thresholds"],
                            threshold_rule="both evidence values <= their stored threshold; eligible fit-only observed values",
                            scope="sequential local search, not exhaustive joint optimization"))


def _router(selection, baseline, meta):
    return dict(schema_version=SCHEMA_VERSION, decoder=DECODER, meta=copy.deepcopy(meta),
                baseline_router=copy.deepcopy(baseline), baseline_router_sha256=geometry._hash(baseline),
                candidate_rule=membership.CANDIDATE_RULE, rules=copy.deepcopy(selection["rules"]),
                geometry_enabled=selection["geometry_enabled"])


def _validate(router, meta):
    geometry._validate_meta(meta)
    if (router.get("schema_version") != SCHEMA_VERSION or router.get("decoder") != DECODER or
            router.get("meta") != meta or router.get("candidate_rule") != membership.CANDIDATE_RULE):
        raise ValueError("Local router schema, hierarchy or candidate rule mismatch")
    baseline = router["baseline_router"]
    membership._validate_state(baseline, meta)
    if router.get("baseline_router_sha256") != geometry._hash(baseline):
        raise ValueError("Local baseline binding changed")
    rules = router.get("rules")
    if (not isinstance(rules, list) or len(rules) > 4 * (len(meta["parent_names"]) + 1) or
            type(router.get("geometry_enabled")) is not bool or router["geometry_enabled"] != bool(rules)):
        raise ValueError("Invalid local rule list or enable state")
    order, seen = -1, set()
    for rule in rules:
        if not isinstance(rule, dict) or set(rule) != {"action", "parent", "x_max", "y_max"}:
            raise ValueError("Invalid local rule fields")
        action, parent = rule["action"], rule["parent"]
        if (action not in ACTIONS or type(parent) is not int or not -1 <= parent < len(meta["parent_names"]) or
                (action, parent) in seen or ACTIONS.index(action) < order):
            raise ValueError("Invalid local rule action, parent or ordering")
        for key in ("x_max", "y_max"):
            value = rule[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError("Local thresholds must be finite scalars")
        order = ACTIONS.index(action)
        seen.add((action, parent))
    return baseline


def apply_router(records, router, meta):
    records = list(records)
    geometry.unique_records(records)
    baseline = _validate(router, meta)
    original = base.apply_router(records, baseline, meta)
    d = _data(records, baseline, meta)
    state, reasons = _initial(d), [[] for _ in records]
    for index, rule in enumerate(router["rules"]):
        scope, x, y = _condition(d, state, rule["action"], rule["parent"])
        mask = scope & (x <= rule["x_max"]) & (y <= rule["y_max"])
        state = _edit(d, state, mask, rule["action"])
        for i in np.flatnonzero(mask):
            reasons[i].append(index)
    pa, la, parents = state
    result = []
    for i, old in enumerate(original):
        row = dict(old)
        parent, leaf = int(parents[i]), int(d["leaf"][i])
        if la[i]:
            kind, node = "known", 1 + len(meta["parent_names"]) + leaf
        elif pa[i]:
            kind, leaf, node = "intra_unknown", None, 1 + parent
        else:
            kind, parent, leaf, node = "global_unknown", None, None, 0
        row.update(prediction_type=kind, parent=parent, leaf=leaf, output_node=node,
                   decoder=DECODER, geometry_enabled=router["geometry_enabled"],
                   baseline_prediction_type=old["prediction_type"],
                   baseline_root_knownness_score=old["root_knownness_score"],
                   baseline_local_knownness_score=old["local_knownness_score"],
                   baseline_parent_threshold=float(baseline["parent_threshold"]),
                   baseline_leaf_threshold=float(baseline["leaf_threshold"]),
                   route_parent=int(parents[i]), alternative_parent=int(d["alternative_parent"][i]),
                   applied_rule_indices=reasons[i],
                   # Local Boolean rules have no single calibrated scalar score.
                   # Keep raw component scores above; expose explicit route flags.
                   root_knownness_score=1. if pa[i] else -1., local_knownness_score=1. if la[i] else -1.,
                   local_known_margin=1. if la[i] else -1., root_threshold=0., local_threshold=0.,
                   parent_threshold=0., leaf_threshold=0.,
                   root_score_type="local_route_indicator_not_probability",
                   local_score_type="local_route_indicator_not_probability")
        result.append(row)
    return result


def score_diagnostics(records):
    """Unthresholded component AUROCs, separate from Boolean route indicators."""
    rows = geometry.unique_records(records)
    status = np.asarray([r["status"] for r in rows])
    result = {}
    for field in geometry.EVIDENCE_FIELDS:
        scores = np.asarray([r[field] for r in rows], dtype=float)
        if "parent" in field:
            pos, neg = scores[status != "extra"], np.sort(scores[status == "extra"])
        else:
            pos, neg = scores[status == "known"], np.sort(scores[status == "intra"])
        auc = None
        if len(pos) and len(neg):
            left, right = np.searchsorted(neg, pos, side="left"), np.searchsorted(neg, pos, side="right")
            auc = float((left + right).sum() / (2. * len(pos) * len(neg)))
        result[field] = dict(auroc=auc, positive_count=len(pos), negative_count=len(neg))
    return dict(unique_images=len(rows), components=result,
                route_score_note="Local root/local scores are +/-1 route indicators. Their metrics_open AUROC/OSCR describe a discrete operating point, not continuous evidence ranking. Component AUROCs here do not imply end-to-end route improvement.")


def _folds(rows, meta, options):
    baseline_settings = options.get("baseline_calibration")
    if not isinstance(baseline_settings, dict) or baseline_settings.get("decoder") != "membership":
        raise ValueError("Local source LOO requires original baseline_calibration")
    folds, skipped = [], []
    for status in ("intra", "extra"):
        sources = sorted({str(r.get("source") or "unspecified") for r in rows if r["status"] == status})
        if len(sources) < 2:
            skipped.append(status)
            continue
        for source in sources:
            held = lambda r: r["status"] == status and str(r.get("source") or "unspecified") == source
            fit, hold = [r for r in rows if not held(r)], [r for r in rows if held(r)]
            groups = [[r for r in fit if r["status"] == s] for s in ("known", "intra", "extra")]
            baseline = membership.calibrate(*groups, meta, dict(baseline_settings, source_loo=False))
            folds.append((status, source, fit, hold, baseline))
    return folds, skipped


def _source_report(folds, skipped, meta, options, actions):
    reports = []
    pooled = {s: dict(total=0, baseline_correct=0, selected_correct=0,
                     baseline_false_leaf=0, selected_false_leaf=0) for s in ("intra", "extra")}
    for status, source, fit, hold, baseline in folds:
        selected = _select(fit, baseline, meta, options, actions)
        original = base.apply_router(hold, baseline, meta)
        predictions = apply_router(hold, _router(selected, baseline, meta), meta)
        old, new = base._group_report(original, status), base._group_report(predictions, status)
        old_leaf = sum(r["prediction_type"] == "known" for r in original)
        new_leaf = sum(r["prediction_type"] == "known" for r in predictions)
        preserved = new["correct_count"] >= old["correct_count"] and new_leaf <= old_leaf
        improved = preserved and (new["correct_count"] > old["correct_count"] or new_leaf < old_leaf)
        reports.append(dict(status=status, held_source=source,
                            fit_image_sha256=sorted(base._digest(r) for r in fit),
                            held_image_sha256=sorted(base._digest(r) for r in hold),
                            baseline_refit_on_fit_sources_only=True, rules_use_fit_sources_only=True,
                            baseline_calibration_sha256=baseline["calibration_sha256"],
                            baseline_parent_threshold=baseline["parent_threshold"],
                            baseline_leaf_threshold=baseline["leaf_threshold"],
                            fit_rules=selected["rules"], baseline_held_metrics=old, held_metrics=new,
                            baseline_false_leaf=old_leaf, selected_false_leaf=new_leaf,
                            preserved=preserved, improved=improved))
        group = pooled[status]
        group["total"] += old["sample_count"]
        group["baseline_correct"] += old["correct_count"]
        group["selected_correct"] += new["correct_count"]
        group["baseline_false_leaf"] += old_leaf
        group["selected_false_leaf"] += new_leaf
    gains = [dict(status=r["status"], source=r["held_source"]) for r in reports if r["improved"]]
    checks = dict(both_unknown_statuses_have_source_support=not skipped,
                  every_held_source_preserved=bool(reports) and all(r["preserved"] for r in reports),
                  at_least_two_distinct_sources_improve=len(gains) >= 2)
    return dict(available=bool(reports), actions=list(actions), folds=reports, skipped=skipped,
                pooled_counts=pooled, safeguard=dict(passed=all(checks.values()), checks=checks,
                    improved_source_count=len(gains), improved_sources=gains),
                use="DEV stability screening, not an unbiased performance estimate. Gains mean more correct unknown routes or fewer false leaf accepts, with neither regressing per held source. Known protection is in-fit, not an unseen-known guarantee.")


def calibrate(known, near, extra, baseline_router, meta, options=None):
    options = settings(options)
    rows, input_count = geometry._fit_inputs(list(known), list(near), list(extra), meta)
    membership._validate_state(baseline_router, meta)
    rows = sorted(rows, key=base._digest)
    provisional = _select(rows, baseline_router, meta, options, options["actions"])
    action_audits, enabled = {}, list(options["actions"])
    empty_loo = dict(available=False, folds=[], safeguard=dict(passed=False), reason="disabled in configuration")
    combined = copy.deepcopy(empty_loo)
    if options["source_loo"]:
        folds, skipped = _folds(rows, meta, options)
        for action in options["actions"]:
            action_audits[action] = _source_report(folds, skipped, meta, options, [action])
        if options["source_loo_safeguard"]:
            enabled = [a for a in enabled if action_audits[a]["safeguard"]["passed"]]
        combined = _source_report(folds, skipped, meta, options, enabled)
    selected = _select(rows, baseline_router, meta, options, enabled)
    rejected = bool(selected["rules"] and options["source_loo_safeguard"] and not combined["safeguard"]["passed"])
    if rejected:
        selected = _select(rows, baseline_router, meta, options, [])
        selected["status"] = "baseline_fallback_source_instability"
    router = _router(selected, baseline_router, meta)
    router.update(fit_completed=True, status=selected["status"], targets=copy.deepcopy(base.TARGETS),
                  targets_passed=selected["gates"]["targets_passed"], best_effort=not selected["gates"]["targets_passed"],
                  baseline_fallback=not selected["geometry_enabled"], preservation_audit=selected["preservation_audit"],
                  search=selected["search"], selection_rule=SELECTION_RULE,
                  action_source_loo=action_audits, enabled_actions=enabled if not rejected else [],
                  rejected_actions=[a for a in options["actions"] if a not in enabled],
                  source_loo=combined, source_loo_safeguard_enabled=options["source_loo_safeguard"],
                  source_loo_safeguard_rejected=rejected, provisional_selection=provisional,
                  fit_splits=["val_known", "val_intra", "val_extra"],
                  input_record_count=input_count, unique_image_count=len(rows), duplicate_record_count=input_count-len(rows),
                  fit_image_sha256=[base._digest(r) for r in rows], settings=copy.deepcopy(options))
    router["baseline_validation_report"] = base.evaluate_records(base.apply_router(rows, baseline_router, meta), meta)
    router["validation_report"] = base.evaluate_records(apply_router(rows, router, meta), meta)
    # Bind every score, label and split that can affect any rule or source fold.
    evidence = [dict(image_sha256=base._digest(r), split=r["split"], status=r["status"],
                     source=r.get("source"), true_parent=r.get("true_parent"), true_leaf=r.get("true_leaf"),
                     log_probs=np.asarray(r["log_probs"], dtype=float).tolist(),
                     support_evidence={k: np.asarray(r["support_evidence"][k], dtype=float).tolist() for k in membership.RAW_FIELDS},
                     **{k: float(r[k]) for k in geometry.EVIDENCE_FIELDS}) for r in rows]
    router["evidence_sha256"] = geometry._hash(evidence)
    router["calibration_sha256"] = geometry._hash(router)
    return router
