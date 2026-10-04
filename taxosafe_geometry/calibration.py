"""Development-only hierarchical RMD fusion with explicit baseline protection.

Class/parent identity is always chosen by the frozen reference ranking heads.
This module fits score fusion and root/leaf thresholds; it never fits features,
reads TEST records, or changes a candidate to obtain a convenient score.
"""
import copy
import hashlib
import json

import numpy as np

from taxosafe_support import calibration as base
from taxosafe_support import membership_calibration as membership

SCHEMA_VERSION = "geometry_hierarchical_v1"
DECODER = "hierarchical_rmd"
EVIDENCE_FIELDS = ("geometry_parent_score", "geometry_leaf_score", "baseline_parent_z", "baseline_leaf_z")
SELECTION_RULE = ("preserve every baseline-correct known DEV image, every unknown source correct count, "
                  "overall leaf precision and known>90%; require strict near/extra/precision improvement; "
                  "then four-gate feasibility, normalized count deficit, near+extra+precision, known, deterministic ties")
SCORE_NOTE = ("Enabled geometry uses convex combinations of TRAIN-normalized scores. Root/local knownness "
              "are the selected fusion scores minus their DEV thresholds; public route thresholds are zero. "
              "Baseline fallback instead uses original raw membership threshold margins, preserving decisions "
              "for every input. Raw membership and normalized component scores remain separate. These are "
              "decision scores, not calibrated probabilities or distribution-free guarantees.")


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _validate_meta(meta):
    for name in ("parent_names", "leaf_names"):
        values = meta.get(name) if isinstance(meta, dict) else None
        if (not isinstance(values, (list, tuple)) or not values or
                any(not isinstance(v, str) or not v for v in values) or len(set(values)) != len(values)):
            raise ValueError("Geometry hierarchy requires unique nonempty " + name)
    mapping = meta.get("leaf_to_parent")
    if (not isinstance(mapping, (list, tuple)) or
            any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in mapping)):
        raise ValueError("Geometry leaf_to_parent must contain integer indices")
    base._hierarchy(meta)


def _evidence(records):
    columns = {}
    for name in EVIDENCE_FIELDS:
        values = []
        for row in records:
            value = row.get(name)
            if isinstance(value, (bool, np.bool_)) or np.shape(value) != ():
                raise ValueError(name + " must be a finite scalar")
            try:
                value = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(name + " must be a finite scalar") from exc
            if not np.isfinite(value):
                raise ValueError(name + " must be a finite scalar")
            values.append(value)
        columns[name] = np.asarray(values, dtype=np.float64)
    return columns


def unique_records(records):
    """Validate every new score before old content deduplication can hide it."""
    records = list(records)
    values = _evidence(records)
    seen = {}
    for index, row in enumerate(records):
        digest = base._digest(row)
        evidence = tuple(values[name][index] for name in EVIDENCE_FIELDS)
        if digest in seen and seen[digest] != evidence:
            raise ValueError("Same content has inconsistent geometry evidence: " + digest)
        seen[digest] = evidence
    return base.unique_records(records)


def _fit_inputs(known, near, extra, meta):
    _validate_meta(meta)
    groups = [list(known), list(near), list(extra)]
    unique_records(sum(groups, []))
    rows, count = base._fit_inputs(*groups, meta)
    for row in rows:
        if any(isinstance(row.get(key), (bool, np.bool_)) for key in ("true_parent", "true_leaf")):
            raise ValueError("Boolean hierarchy labels are invalid")
    return rows, count


def _validate_state(router, meta):
    _validate_meta(meta)
    if (router.get("schema_version") != SCHEMA_VERSION or router.get("decoder") != DECODER or
            router.get("candidate_rule") != membership.CANDIDATE_RULE or router.get("meta") != meta):
        raise ValueError("Geometry router schema, candidate rule or hierarchy mismatch")
    baseline = router["baseline_router"]
    membership._validate_state(baseline, meta)
    if router.get("baseline_router_sha256") != _hash(baseline):
        raise ValueError("Geometry baseline router binding mismatch")
    enabled = router.get("geometry_enabled")
    if not isinstance(enabled, bool):
        raise ValueError("Geometry router must declare geometry_enabled")
    values = []
    for key in ("parent_weight", "leaf_weight", "parent_threshold", "leaf_threshold"):
        value = router.get(key)
        if isinstance(value, (bool, np.bool_)) or np.shape(value) != ():
            raise ValueError("Geometry router parameters must be finite scalars")
        try:
            values.append(float(value))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Geometry router parameters must be finite scalars") from exc
    if not np.isfinite(values).all() or any(not 0. <= v <= 1. for v in values[:2]):
        raise ValueError("Invalid geometry fusion weights or thresholds")
    if enabled and values[0] == values[1] == 0.:
        raise ValueError("Geometry fusion cannot merely retune baseline thresholds")
    return baseline, enabled, values


def apply_router(records, router, meta):
    records = list(records)
    unique_records(records)
    baseline, enabled, (pw, lw, pt, lt) = _validate_state(router, meta)
    evidence = _evidence(records)
    original_predictions = base.apply_router(records, baseline, meta)
    parent_count = len(meta["parent_names"])
    result = []
    for i, original in enumerate(original_predictions):
        row = dict(original)
        if enabled:
            parent_score = (1. - pw) * evidence["baseline_parent_z"][i] + pw * evidence["geometry_parent_score"][i]
            leaf_score = (1. - lw) * evidence["baseline_leaf_z"][i] + lw * evidence["geometry_leaf_score"][i]
            root_margin, leaf_margin = float(parent_score - pt), float(leaf_score - lt)
            p, c = row["candidate_parent"], row["candidate_leaf"]
            if root_margin < 0.:
                kind, parent, leaf, node = "global_unknown", None, None, 0
            elif leaf_margin < 0.:
                kind, parent, leaf, node = "intra_unknown", p, None, 1 + p
            else:
                kind, parent, leaf, node = "known", p, c, 1 + parent_count + c
            row.update(prediction_type=kind, parent=parent, leaf=leaf, output_node=node)
            score_type = "train_normalized_membership_rmd_fusion_threshold_margin"
        else:
            parent_score, leaf_score = original["parent_membership_score"], original["leaf_membership_score"]
            root_margin = float(parent_score - baseline["parent_threshold"])
            leaf_margin = float(leaf_score - baseline["leaf_threshold"])
            score_type = "baseline_raw_membership_threshold_margin"
        if not np.isfinite([parent_score, leaf_score, root_margin, leaf_margin]).all():
            raise ValueError("Geometry route score overflow")
        row.update(decoder=DECODER, geometry_enabled=enabled,
                   baseline_prediction_type=original["prediction_type"],
                   baseline_root_knownness_score=original["root_knownness_score"],
                   baseline_local_knownness_score=original["local_knownness_score"],
                   baseline_parent_threshold=float(baseline["parent_threshold"]),
                   baseline_leaf_threshold=float(baseline["leaf_threshold"]),
                   fusion_parent_score=float(parent_score), fusion_leaf_score=float(leaf_score),
                   fusion_parent_weight=pw, fusion_leaf_weight=lw,
                   fusion_parent_threshold=pt if enabled else float(baseline["parent_threshold"]),
                   fusion_leaf_threshold=lt if enabled else float(baseline["leaf_threshold"]),
                   root_knownness_score=root_margin, local_knownness_score=leaf_margin,
                   local_known_margin=leaf_margin, root_threshold=0., local_threshold=0.,
                   parent_threshold=0., leaf_threshold=0., root_score_type=score_type, local_score_type=score_type)
        result.append(row)
    return result


decode_records = apply_router


def evaluate_records(records, meta=None):
    records = list(records)
    unique_records(records)
    if meta is not None:
        _validate_meta(meta)
    return base.evaluate_records(records, meta)


evaluate_gates = base.evaluate_gates


def _settings(settings):
    settings = dict(settings or {})
    if settings.get("threshold_grid", "quantile") != "quantile":
        raise ValueError("Geometry threshold_grid must be quantile")
    for key in ("source_loo", "source_loo_safeguard"):
        if key in settings and not isinstance(settings[key], bool):
            raise ValueError(key + " must be boolean")
    count = settings.get("grid_points", 31)
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or not 2 <= count <= 101:
        raise ValueError("Geometry grid_points must be an integer in [2,101]")
    for key in ("parent_weights", "leaf_weights"):
        values = settings.get(key, settings.get("weights", [0., .5, 1.]))
        if (not isinstance(values, (list, tuple)) or not 1 <= len(values) <= 11 or
                any(isinstance(v, (bool, np.bool_)) or np.shape(v) != () for v in values)):
            raise ValueError(key + " must contain finite weights in [0,1]")
        try:
            grid = np.asarray(values, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(key + " must contain finite weights in [0,1]") from exc
        if not np.isfinite(grid).all() or ((grid < 0.) | (grid > 1.)).any():
            raise ValueError(key + " must contain finite weights in [0,1]")
        settings[key] = np.unique(grid).tolist()
    if settings["parent_weights"] == [0.] and settings["leaf_weights"] == [0.]:
        raise ValueError("At least one nonzero geometry fusion weight is required")
    return settings


def _grid(values, settings, name):
    explicit = settings.get(name)
    grid = np.asarray(explicit if explicit is not None else np.quantile(
        values, np.linspace(0., 1., settings.get("grid_points", 31))), dtype=float)
    if grid.ndim != 1 or not 1 <= len(grid) <= 103 or not np.isfinite(grid).all():
        raise ValueError(name + " must contain finite thresholds")
    with np.errstate(over="ignore"):
        grid = np.r_[grid, np.nextafter(values.min(), -np.inf), np.nextafter(values.max(), np.inf)]
    if not np.isfinite(grid).all():
        raise ValueError("Geometry scores too extreme for finite grid endpoints")
    return np.unique(grid)


def _arrays(rows, meta):
    data = membership._arrays(rows, meta)
    data.update(_evidence(rows))
    data["sources"] = np.asarray([str(r.get("source") or "unspecified") for r in rows])
    return data


def _counts(d, parent_accept, leaf_accept):
    return dict({s: int(d[s].sum()) for s in ("known", "intra", "extra")},
                known_correct=int(np.sum(d["known_candidate_correct"] & leaf_accept)),
                intra_correct=int(np.sum(d["near_candidate_correct"] & parent_accept & ~leaf_accept)),
                extra_correct=int(np.sum(d["extra"] & ~parent_accept)), leaf_outputs=int(leaf_accept.sum()))


def _correct_masks(d, parent_accept, leaf_accept):
    return {"known": d["known_candidate_correct"] & leaf_accept,
            "intra": d["near_candidate_correct"] & parent_accept & ~leaf_accept,
            "extra": d["extra"] & ~parent_accept}


def _audit(rows, d, bp, bl, parent_accept, leaf_accept, meta):
    before, after = _correct_masks(d, bp, bl), _correct_masks(d, parent_accept, leaf_accept)
    baseline, selected = _counts(d, bp, bl), _counts(d, parent_accept, leaf_accept)
    lost = before["known"] & ~after["known"]
    source_audit = []
    for status in ("intra", "extra"):
        for source in sorted(set(d["sources"][d[status]].tolist())):
            group = d[status] & (d["sources"] == source)
            old, new = int((before[status] & group).sum()), int((after[status] & group).sum())
            source_audit.append(dict(status=status, source=source, total=int(group.sum()),
                                     baseline_correct=old, selected_correct=new, passed=new >= old))
    leaf_audit = []
    for leaf, name in enumerate(meta["leaf_names"]):
        group = d["known"] & np.asarray([r.get("true_leaf") == leaf for r in rows])
        old, new = int((before["known"] & group).sum()), int((after["known"] & group).sum())
        if group.any():
            leaf_audit.append(dict(leaf=name, total=int(group.sum()), baseline_correct=old,
                                   selected_correct=new, passed=new >= old))
    left = selected["known_correct"] * baseline["leaf_outputs"]
    right = baseline["known_correct"] * selected["leaf_outputs"]
    defined = selected["leaf_outputs"] > 0 and baseline["leaf_outputs"] > 0
    checks = dict(every_baseline_correct_known_preserved=not bool(lost.any()),
                  every_known_leaf_count_preserved=all(x["passed"] for x in leaf_audit),
                  every_unknown_source_count_preserved=all(x["passed"] for x in source_audit),
                  leaf_precision_at_least_baseline=defined and left >= right,
                  strict_known_gate=selected["known_correct"] >= 9 * selected["known"] // 10 + 1)
    improved = (selected["intra_correct"] > baseline["intra_correct"] or
                selected["extra_correct"] > baseline["extra_correct"] or (defined and left > right))
    return dict(checks=checks, passed=all(checks.values()), strict_unknown_or_precision_improvement=bool(improved),
                baseline_counts=baseline, selected_counts=selected,
                baseline_correct_known_sha256=sorted(base._digest(r) for r, flag in zip(rows, before["known"]) if flag),
                lost_baseline_correct_known_sha256=sorted(base._digest(r) for r, flag in zip(rows, lost) if flag),
                per_known_leaf=leaf_audit, per_unknown_source=source_audit,
                precision_cross_product=dict(selected_correct_times_baseline_leaf_outputs=left,
                                             baseline_correct_times_selected_leaf_outputs=right),
                scope="unique-content DEV images and sources only; not a future-data guarantee")


def _baseline_masks(d, baseline_router):
    parent = d["parent_score"] >= float(baseline_router["parent_threshold"])
    return parent, parent & (d["leaf_score"] >= float(baseline_router["leaf_threshold"]))


def _select(rows, baseline_router, meta, settings):
    """Vectorize the threshold Cartesian product; build rich reports only once."""
    membership._validate_state(baseline_router, meta)
    d = _arrays(rows, meta)
    bp, bl = _baseline_masks(d, baseline_router)
    baseline = _counts(d, bp, bl)
    before = _correct_masks(d, bp, bl)
    totals = {s: baseline[s] for s in ("known", "intra", "extra")}
    weights = [(p, l) for p in settings["parent_weights"] for l in settings["leaf_weights"] if p or l]
    grids, weight_results, best = [], [], None
    total_sampled = total_preserved = total_improved = total_feasible = 0
    for pw, lw in weights:
        pm = (1. - pw) * d["baseline_parent_z"] + pw * d["geometry_parent_score"]
        lm = (1. - lw) * d["baseline_leaf_z"] + lw * d["geometry_leaf_score"]
        pg, lg = _grid(pm, settings, "parent_threshold_grid"), _grid(lm, settings, "leaf_threshold_grid")
        pa = pm[None, None, :] >= pg[:, None, None]
        la = pa & (lm[None, None, :] >= lg[None, :, None])
        nc = pa & ~la & d["near_candidate_correct"]
        ec = ~pa & d["extra"]
        k = (la & d["known_candidate_correct"]).sum(2)
        n = nc.sum(2)
        e = np.broadcast_to(ec.sum(2), k.shape)
        outputs = la.sum(2)
        protected = ~(before["known"] & ~la).any(2)
        protected &= k >= 9 * totals["known"] // 10 + 1
        precision_left, precision_right = k * baseline["leaf_outputs"], baseline["known_correct"] * outputs
        protected &= (outputs > 0) & (baseline["leaf_outputs"] > 0) & (precision_left >= precision_right)
        for status, correct in (("intra", nc), ("extra", ec)):
            for source in sorted(set(d["sources"][d[status]].tolist())):
                mask = d[status] & (d["sources"] == source)
                protected &= (correct & mask).sum(2) >= int((before[status] & mask).sum())
        improved = ((n > baseline["intra_correct"]) | (e > baseline["extra_correct"]) |
                    (precision_left > precision_right))
        feasible = ((k >= 9 * totals["known"] // 10 + 1) &
                    (n >= (17 * totals["intra"] + 19) // 20) &
                    (e >= 9 * totals["extra"] // 10 + 1) &
                    (outputs > 0) & (k >= 9 * outputs // 10 + 1))
        valid = protected & improved
        sampled, preserving, improving = int(k.size), int(protected.sum()), int((protected & improved).sum())
        total_sampled += sampled
        total_preserved += preserving
        total_improved += improving
        total_feasible += int((feasible & protected).sum())
        grids.append(dict(parent_weight=float(pw), leaf_weight=float(lw), parent_threshold=pg.tolist(),
                          leaf_threshold=lg.tolist(), sampled_point_count=sampled))
        weight_results.append(dict(parent_weight=float(pw), leaf_weight=float(lw), sampled_point_count=sampled,
                                   preservation_count=preserving, improvement_count=improving,
                                   feasible_preserving_count=int((feasible & protected).sum())))
        if not valid.any():
            continue
        safe_outputs = np.maximum(outputs, 1)
        deficit = (np.maximum(0, 9 * totals["known"] // 10 + 1 - k) / totals["known"] +
                   np.maximum(0, (17 * totals["intra"] + 19) // 20 - n) / totals["intra"] +
                   np.maximum(0, 9 * totals["extra"] // 10 + 1 - e) / totals["extra"] +
                   np.maximum(0, 9 * outputs // 10 + 1 - k) / safe_outputs)
        quality = n / totals["intra"] + e / totals["extra"] + k / safe_outputs
        # Only candidate count vectors are inspected in Python, never full records/reports.
        ip, il = np.nonzero(valid)
        order = np.lexsort((-lg[il], -pg[ip], -np.abs(pg[ip]) - np.abs(lg[il]),
                            k[ip, il], quality[ip, il], -deficit[ip, il], feasible[ip, il].astype(int)))
        i, j = int(ip[order[-1]]), int(il[order[-1]])
        key = (bool(feasible[i, j]), -float(deficit[i, j]), float(quality[i, j]), int(k[i, j]),
               -float(pw + lw), -float(pw), -float(lw), -abs(float(pg[i])) - abs(float(lg[j])),
               -float(pg[i]), -float(lg[j]))
        if best is None or key > best[0]:
            best = key, float(pw), float(lw), float(pg[i]), float(lg[j]), pa[i, 0].copy(), la[i, j].copy()
    if best is None:
        pw = lw = pt = lt = 0.
        parent_accept, leaf_accept = bp, bl
        enabled, status = False, "baseline_fallback"
    else:
        _, pw, lw, pt, lt, parent_accept, leaf_accept = best
        enabled = True
        status = "feasible_geometry" if base._gates(_counts(d, parent_accept, leaf_accept))["targets_passed"] else "best_effort_geometry"
    audit = _audit(rows, d, bp, bl, parent_accept, leaf_accept, meta)
    return dict(parent_weight=pw, leaf_weight=lw, parent_threshold=pt, leaf_threshold=lt,
                geometry_enabled=enabled, status=status, preservation_audit=audit,
                gates=base._gates(audit["selected_counts"]),
                grid=dict(weight_threshold_grids=grids, sampled_point_count=total_sampled,
                          sampled_preservation_count=total_preserved, sampled_improvement_count=total_improved,
                          sampled_feasible_count=total_feasible, all_zero_weights_excluded=True,
                          weight_results=weight_results, grid_uses_fit_records_only=True,
                          definition="fit-only empirical quantiles or explicit thresholds plus all-pass/all-reject endpoints",
                          scope="finite declared DEV search; no continuous infeasibility claim"))


def _router(selected, baseline_router, meta):
    return dict(schema_version=SCHEMA_VERSION, decoder=DECODER, meta=copy.deepcopy(meta),
                baseline_router=copy.deepcopy(baseline_router), baseline_router_sha256=_hash(baseline_router),
                candidate_rule=membership.CANDIDATE_RULE, score_note=SCORE_NOTE,
                **{k: selected[k] for k in ("parent_weight", "leaf_weight", "parent_threshold", "leaf_threshold", "geometry_enabled")})


def _source_safeguard(folds, skipped, pooled):
    """Require evidence beyond identity fallback or a single fortunate source."""
    improved = sorted({(f["status"], f["held_source"]) for f in folds
                       if f["geometry_selected"] and
                       f["held_metrics"]["correct_count"] > f["baseline_held_metrics"]["correct_count"]})
    selected_count = sum(int(f["geometry_selected"]) for f in folds)
    checks = dict(both_unknown_statuses_have_source_support=not skipped,
                  at_least_one_fold_selected_geometry=selected_count > 0,
                  at_least_two_distinct_held_sources_improve=len(improved) >= 2,
                  every_held_source_correct_count_preserved=bool(folds) and all(f["held_source_correct_count_preserved"] for f in folds),
                  pooled_near_correct_count_preserved=pooled["intra"]["total"] > 0 and pooled["intra"]["selected_correct"] >= pooled["intra"]["baseline_correct"],
                  pooled_extra_correct_count_preserved=pooled["extra"]["total"] > 0 and pooled["extra"]["selected_correct"] >= pooled["extra"]["baseline_correct"])
    return dict(checks=checks, passed=bool(folds) and all(checks.values()),
                geometry_selected_fold_count=selected_count, improved_source_count=len(improved),
                improved_sources=[dict(status=status, source=source) for status, source in improved],
                required_improved_source_count=2,
                definition="Refit baseline and reselect fusion on retained DEV sources; require gains on at least two distinct held (status,source) pairs and no held-source or pooled regression. Baseline-only folds cannot establish positive geometry evidence.")


def source_loo(known, near, extra, baseline_router, meta, settings=None):
    """Refit the complete baseline and fusion selection without each source.

    The source audit itself influences full-DEV enable/fallback selection. It is
    a DEV stability safeguard, not an unbiased final-model performance estimate.
    """
    settings = _settings(settings)
    rows, _ = _fit_inputs(known, near, extra, meta)
    membership._validate_state(baseline_router, meta)
    baseline_settings = settings.get("baseline_calibration")
    if not isinstance(baseline_settings, dict) or baseline_settings.get("decoder") != "membership":
        raise ValueError("Geometry source LOO requires original baseline_calibration configuration")
    baseline_settings = dict(baseline_settings, source_loo=False)
    folds, skipped = [], []
    pooled = {s: dict(total=0, baseline_correct=0, selected_correct=0) for s in ("intra", "extra")}
    for status in ("intra", "extra"):
        sources = sorted({str(r.get("source") or "unspecified") for r in rows if r["status"] == status})
        if len(sources) < 2:
            skipped.append(dict(status=status, reason="At least two DEV sources are required"))
            continue
        for source in sources:
            held = lambda r: r["status"] == status and str(r.get("source") or "unspecified") == source
            fitted, holdout = [r for r in rows if not held(r)], [r for r in rows if held(r)]
            groups = [[r for r in fitted if r["status"] == s] for s in ("known", "intra", "extra")]
            fold_base = membership.calibrate(*groups, meta, baseline_settings)
            selection = _select(fitted, fold_base, meta, settings)
            predictions = apply_router(holdout, _router(selection, fold_base, meta), meta)
            old = base._group_report(base.apply_router(holdout, fold_base, meta), status)
            fold = dict(status=status, held_source=source, fit_count=len(fitted),
                        fit_image_sha256=sorted(base._digest(r) for r in fitted),
                        held_image_sha256=sorted(base._digest(r) for r in holdout),
                        baseline_refit_on_fit_sources_only=True, grid_uses_fit_sources_only=True,
                        baseline_parent_threshold=fold_base["parent_threshold"],
                        baseline_leaf_threshold=fold_base["leaf_threshold"],
                        baseline_calibration_sha256=fold_base["calibration_sha256"],
                        fit_sources=sorted({str(r.get("source") or "unspecified") for r in fitted if r["status"] == status}),
                        reselected_weights=[selection["parent_weight"], selection["leaf_weight"]],
                        fit_selection_status=selection["status"], fit_preservation_audit=selection["preservation_audit"],
                        baseline_held_metrics=old, held_metrics=base._group_report(predictions, status))
            new = fold["held_metrics"]
            fold.update(reselected_parent_threshold=selection["parent_threshold"],
                        reselected_leaf_threshold=selection["leaf_threshold"],
                        geometry_selected=selection["geometry_enabled"],
                        weights_use_fit_sources_only=True,
                        held_source_correct_count_preserved=new["correct_count"] >= old["correct_count"])
            pooled[status]["total"] += old["sample_count"]
            pooled[status]["baseline_correct"] += old["correct_count"]
            pooled[status]["selected_correct"] += new["correct_count"]
            folds.append(fold)
    safeguard = _source_safeguard(folds, skipped, pooled)
    return dict(available=bool(folds), folds=folds, skipped=skipped, pooled_counts=pooled,
                geometry_selected_fold_count=sum(int(f["geometry_selected"]) for f in folds),
                safeguard=safeguard,
                use="DEV source stability safeguard; feature statistics are TRAIN-only. Each baseline router, fusion weight and threshold grid is refit without held source. This audit influences full-DEV enable/fallback selection and is not an unbiased final-model performance estimate.")


def calibrate(known, near, extra, baseline_router, meta, settings=None):
    settings = _settings(settings)
    groups = [list(known), list(near), list(extra)]
    rows, input_count = _fit_inputs(*groups, meta)
    selected = _select(rows, baseline_router, meta, settings)
    provisional = copy.deepcopy(selected)
    if settings.get("source_loo", True):
        loo = source_loo(*groups, baseline_router, meta, settings)
    else:
        loo = dict(available=False, folds=[], reason="disabled in configuration",
                   safeguard=dict(passed=False, checks={"source_loo_enabled": False}))
    rejected = bool(selected["geometry_enabled"] and settings.get("source_loo_safeguard", True) and not loo["safeguard"]["passed"])
    if rejected:
        d = _arrays(rows, meta)
        bp, bl = _baseline_masks(d, baseline_router)
        selected.update(parent_weight=0., leaf_weight=0., parent_threshold=0., leaf_threshold=0.,
                        geometry_enabled=False, status="baseline_fallback_source_instability",
                        preservation_audit=_audit(rows, d, bp, bl, bp, bl, meta), gates=base._gates(_counts(d, bp, bl)))
    router = _router(selected, baseline_router, meta)
    router.update(fitted_parameters=["parent_weight", "leaf_weight", "parent_threshold", "leaf_threshold"],
                  fit_completed=True, status=selected["status"], targets_passed=selected["gates"]["targets_passed"],
                  targets=copy.deepcopy(base.TARGETS), best_effort=not selected["gates"]["targets_passed"],
                  baseline_fallback=not selected["geometry_enabled"], preservation_audit=selected["preservation_audit"],
                  selection_rule=SELECTION_RULE, grid=selected["grid"], source_loo=loo,
                  source_loo_safeguard_enabled=settings.get("source_loo_safeguard", True),
                  source_loo_safeguard_rejected=rejected,
                  provisional_selection={k: copy.deepcopy(provisional[k]) for k in (
                      "parent_weight", "leaf_weight", "parent_threshold", "leaf_threshold", "geometry_enabled", "status", "gates", "preservation_audit")},
                  fit_splits=["val_known", "val_intra", "val_extra"], input_record_count=input_count,
                  unique_image_count=len(rows), duplicate_record_count=input_count-len(rows),
                  fit_image_sha256=sorted(base._digest(r) for r in rows))
    router["baseline_validation_report"] = base.evaluate_records(base.apply_router(rows, baseline_router, meta), meta)
    predictions = apply_router(rows, router, meta)
    report = evaluate_records(predictions, meta)
    for key in ("fit_completed", "status", "best_effort", "baseline_fallback", "preservation_audit", "selection_rule", "score_note", "source_loo_safeguard_rejected"):
        report[key] = copy.deepcopy(router[key])
    report["sampled_feasible_count"] = selected["grid"]["sampled_feasible_count"]
    report["infeasibility"] = dict(no_feasible_preserving_sampled_point=not bool(report["sampled_feasible_count"]),
                                  failed_at_selected_point=[k for k, v in report["checks"].items() if not v],
                                  scope="finite declared fusion/threshold grid subject to DEV preservation constraints")
    d = _arrays(rows, meta)
    bp, _ = _baseline_masks(d, baseline_router)
    new_parent = np.asarray([p["prediction_type"] != "global_unknown" for p in predictions])
    total = int(d["intra"].sum())
    report["parent_routing_near_upper_bound"] = dict(
        total=total, required_correct=(17 * total + 19) // 20,
        baseline_correct_count=int((d["near_candidate_correct"] & bp).sum()),
        selected_correct_count=int((d["near_candidate_correct"] & new_parent).sum()),
        ranking_candidate_correct_count=int(d["near_candidate_correct"].sum()),
        baseline_rate=float((d["near_candidate_correct"] & bp).sum() / total),
        selected_rate=float((d["near_candidate_correct"] & new_parent).sum() / total),
        scope="fixed identity ranking; parent gate may change, leaf rejection cannot exceed retained correct-parent count")
    router["validation_report"] = report
    evidence = [dict(image_sha256=base._digest(r), split=r["split"], status=r["status"],
                     true_parent=r.get("true_parent"), true_leaf=r.get("true_leaf"), source=r.get("source"),
                     log_probs=np.asarray(r["log_probs"], dtype=float).tolist(),
                     support_evidence={k: np.asarray(r["support_evidence"][k], dtype=float).tolist() for k in membership.RAW_FIELDS},
                     **{k: float(r[k]) for k in EVIDENCE_FIELDS}) for r in sorted(rows, key=base._digest)]
    router["evidence_sha256"] = _hash(evidence)
    router["calibration_sha256"] = _hash({k: router[k] for k in (
        "schema_version", "decoder", "meta", "baseline_router_sha256", "candidate_rule", "parent_weight", "leaf_weight",
        "parent_threshold", "leaf_threshold", "geometry_enabled", "selection_rule", "grid", "targets", "evidence_sha256",
        "source_loo_safeguard_enabled", "source_loo_safeguard_rejected", "source_loo")})
    return router
