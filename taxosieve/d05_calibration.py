"""Exact global DEV calibration for the frozen D05 evidence model.

This module retains the original global threshold search, validation, evidence
hashes and diagnostic fields needed for TaxoSieve and its conditional OOF audit.
"""
import copy
from collections import defaultdict

import numpy as np

from taxosafe_support import calibration as base
from taxosafe_support import membership_calibration as membership
from taxosafe_support.io import normalized_name
from .calibration_utils import _hash, _folds, paired_audit as _legacy_paired

SCHEMA_VERSION = "discovery_calibration_v1"
DECODER = "discovery"
COMPARISON = "(raw_score - parent_group_offset) >= global_threshold"
DEFAULT_SETTINGS = {"seed": 1, "shrinkage": 20., "min_parent_known": 5}
CONTEXT = {"validation_scope": "exploratory_calibration_conditional_on_frozen_discovery_evidence",
           "independent_model_level_validation": False, "confirmatory_validation": False,
           "test_used_for_fitting": False}


def validate_settings(settings=None):
    if settings is not None and not isinstance(settings, dict):
        raise ValueError("Discovery settings must be a mapping")
    result = dict(DEFAULT_SETTINGS, **(settings or {}))
    if set(result) != set(DEFAULT_SETTINGS):
        raise ValueError("Unexpected discovery calibration setting")
    if type(result["seed"]) is not int or not 0 <= result["seed"] < 2**31:
        raise ValueError("seed must be an integer in [0,2**31)")
    if type(result["min_parent_known"]) is not int or not 1 <= result["min_parent_known"] <= 10000:
        raise ValueError("min_parent_known must be a positive integer")
    value = result["shrinkage"]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
        raise ValueError("shrinkage must be finite and nonnegative")
    result["shrinkage"] = float(value)
    return result


def _meta(meta):
    p, c, mapping = base._hierarchy(meta)
    for key in ("leaf_names", "parent_names"):
        values = meta[key]
        if any(not isinstance(value, str) or not value for value in values) or len(set(values)) != len(values):
            raise ValueError("Taxonomy names must be unique nonempty strings")
    if any(type(value) is not int for value in meta["leaf_to_parent"]):
        raise ValueError("Taxonomy mapping must use integer indices")
    return p, c, mapping


def _data(records, meta):
    records = list(records)
    p, c, mapping = _meta(meta)
    vectors, candidates = {}, {}
    for key, size in (("leaf_scores", c), ("parent_scores", p)):
        try:
            raw = [row["discovery"][key] for row in records]
            values = np.asarray(raw, dtype=float) if records else np.empty((0, size), dtype=float)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Missing finite discovery vector: " + key) from exc
        if values.shape != (len(records), size) or not np.isfinite(values).all():
            raise ValueError("Invalid finite discovery vector: " + key)
        if any(isinstance(v, (bool, np.bool_)) for row in raw for v in row):
            raise ValueError("Boolean discovery scores are invalid")
        vectors[key] = values
    for key, size in (("candidate_leaf", c), ("candidate_parent", p)):
        values = []
        for row in records:
            value = row["discovery"].get(key)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not 0 <= value < size:
                raise ValueError("Invalid preselected discovery candidate: " + key)
            values.append(int(value))
        candidates[key] = np.asarray(values, dtype=int)
    index = np.arange(len(records))
    leaf, parent = candidates["candidate_leaf"], candidates["candidate_parent"]
    leaf_parent = mapping[leaf]
    return dict(leaf=leaf, parent=parent, leaf_parent=leaf_parent,
                leaf_score=vectors["leaf_scores"][index, leaf],
                leaf_parent_score=vectors["parent_scores"][index, leaf_parent],
                parent_score=vectors["parent_scores"][index, parent], **vectors)


def _unique(records, meta):
    records = list(records)
    _data(records, meta)
    seen = {}
    for row in records:
        identity, evidence = base._digest(row), _hash(row["discovery"])
        if identity in seen and seen[identity] != evidence:
            raise ValueError("Same image has conflicting discovery evidence")
        seen[identity] = evidence
    return sorted(base.unique_records(records), key=base._digest)


def _inputs(known, near, extra, meta):
    groups = [list(known), list(near), list(extra)]
    _, c, mapping = _meta(meta)
    if any(not rows for rows in groups):
        raise ValueError("Discovery fit requires nonempty known, near and extra DEV")
    for status, rows in zip(base.STATUSES, groups):
        for row in rows:
            if row.get("status") != status or row.get("split") != "val_" + status:
                raise ValueError("DEV splits only; test fitting is prohibited")
            if status != "extra":
                parent = row.get("true_parent")
                if type(parent) is not int or not 0 <= parent < len(meta["parent_names"]):
                    raise ValueError("Invalid true parent")
            if status == "known":
                leaf = row.get("true_leaf")
                if type(leaf) is not int or not 0 <= leaf < c or mapping[leaf] != row["true_parent"]:
                    raise ValueError("Invalid true leaf/parent")
    raw = sum(groups, [])
    return _unique(raw, meta), len(raw)


def _offsets(rows, data, meta, settings, variant):
    count = len(meta["parent_names"])
    result, details = {}, {}
    for level in ("parent", "leaf"):
        values = defaultdict(list)
        for i, row in enumerate(rows):
            if row["status"] != "known":
                continue
            if level == "parent" and data["parent"][i] == row["true_parent"]:
                values[row["true_parent"]].append(float(data["parent_score"][i]))
            elif level == "leaf" and data["leaf"][i] == row["true_leaf"]:
                values[row["true_parent"]].append(float(data["leaf_score"][i]))
        pooled = [value for group in values.values() for value in group]
        location = float(np.median(pooled)) if pooled else 0.
        offsets, report = [], []
        for parent in range(count):
            selected = values[parent]
            n = len(selected)
            local = float(np.median(selected)) if selected else None
            eligible = False  # D05 global calibration retains zero offsets.
            weight = n / (n + settings["shrinkage"]) if eligible else 0.
            with np.errstate(over="ignore", invalid="ignore"):
                delta = float(weight * (local - location)) if weight else 0.
            if not np.isfinite(delta):
                raise ValueError("Parentwise threshold offset overflow")
            offsets.append(delta)
            report.append(dict(parent=parent, known_correct_candidate_count=n, local_median=local,
                               pooling_weight=weight, offset=delta,
                               evidence_status="not_evaluable" if not n else "insufficient_evidence" if n < settings["min_parent_known"] else "observed",
                               global_fallback=not eligible))
        result[level] = np.asarray(offsets, dtype=float)
        details[level] = dict(global_median=location, global_count=len(pooled), groups=report)
    return result, details


def _reference(rows, reference_records, meta):
    if reference_records is None:
        return None, None
    bundle = reference_records if isinstance(reference_records, dict) else None
    raw = bundle["records"] if bundle is not None else reference_records
    references = {base._digest(r): r for r in base.unique_records(raw)}
    if set(references) != {base._digest(r) for r in rows}:
        raise ValueError("Reference and discovery image identities differ")
    aligned = []
    for row in rows:
        other = references[base._digest(row)]
        if any(row.get(key) != other.get(key) for key in ("status", "split", "source", "species", "true_leaf", "true_parent")):
            raise ValueError("Reference and discovery annotations differ")
        aligned.append(other)
    if bundle is not None:
        if set(bundle) != {"records", "router", "calibration_settings"}:
            raise ValueError("Reference bundle requires records, router, calibration_settings")
        before = membership.decode_records(aligned, bundle["router"], meta)
    else:
        if any(row.get("prediction_type") not in base.KINDS or "leaf" not in row or "parent" not in row for row in aligned):
            raise ValueError("Reference lists require decoded predictions; raw records need the reference bundle")
        before = aligned
    return before, dict(bundle, records=aligned) if bundle is not None else None


def _router_hash(state):
    return _hash({key: state[key] for key in ("schema_version", "decoder", "meta", "variant", "settings",
                  "parent_thresholds", "leaf_thresholds", "parent_offsets", "leaf_offsets",
                  "global_parent_threshold", "global_leaf_threshold", "comparison",
                  "candidate_rule", "settings_sha256")})


def validate_router(state, meta):
    p, _, _ = _meta(meta)
    if (state.get("schema_version") != SCHEMA_VERSION or state.get("decoder") != DECODER
            or state.get("meta") != meta or state.get("variant") != "global"
            or state.get("candidate_rule") != "supplied_before_thresholds;leaf_own_parent_admission;single_supplied_fallback"
            or state.get("comparison") != COMPARISON):
        raise ValueError("Discovery router schema, variant, taxonomy or candidate rule differs")
    settings = validate_settings(state.get("settings"))
    if state.get("settings") != settings or state.get("settings_sha256") != _hash(settings):
        raise ValueError("Discovery settings binding changed")
    for key in ("parent_thresholds", "leaf_thresholds", "parent_offsets", "leaf_offsets"):
        values = state.get(key)
        if not isinstance(values, list) or len(values) != p or any(isinstance(v, bool) or not isinstance(v, (float, int)) or not np.isfinite(v) for v in values):
            raise ValueError("Discovery thresholds require finite parent-aligned arrays")
    for level in ("parent", "leaf"):
        value = state.get("global_" + level + "_threshold")
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not np.isfinite(value):
            raise ValueError("Discovery global thresholds must be finite")
        if state["variant"] == "global" and any(state[level + "_offsets"]):
            raise ValueError("Global discovery router cannot use parentwise offsets")
    if state.get("router_sha256") != _router_hash(state):
        raise ValueError("Discovery router checksum changed")
    for level in ("parent", "leaf"):
        with np.errstate(over="ignore", invalid="ignore"):
            nominal = state["global_" + level + "_threshold"] + np.asarray(state[level + "_offsets"])
        if not np.array_equal(nominal, state[level + "_thresholds"]):
            raise ValueError("Nominal discovery thresholds differ from global threshold plus offset")


def _adjusted(data, parent_offsets, leaf_offsets):
    # Keep this exact order in both fitting and inference. Adding the offset
    # back to a threshold can change a tied decision by one floating-point ULP.
    with np.errstate(over="ignore", invalid="ignore"):
        ps = data["parent_score"] - parent_offsets[data["parent"]]
        lps = data["leaf_parent_score"] - parent_offsets[data["leaf_parent"]]
        ls = data["leaf_score"] - leaf_offsets[data["leaf_parent"]]
    if not np.isfinite(np.r_[ps, lps, ls]).all():
        raise ValueError("Adjusted discovery scores overflow")
    return ps, lps, ls


def decode_records(records, router, meta):
    validate_router(router, meta)
    records = list(records)
    data = _data(records, meta)
    pt, lt = router["global_parent_threshold"], router["global_leaf_threshold"]
    ps, lps, ls = _adjusted(data, np.asarray(router["parent_offsets"]), np.asarray(router["leaf_offsets"]))
    with np.errstate(over="ignore", invalid="ignore"):
        leaf_margin = np.minimum(ls-lt, lps-pt)
        parent_margin = ps-pt
    if not np.isfinite(leaf_margin).all() or not np.isfinite(parent_margin).all():
        raise ValueError("Discovery route margin overflow")
    result = []
    for i, record in enumerate(records):
        c, p, lp = (int(data[key][i]) for key in ("leaf", "parent", "leaf_parent"))
        if leaf_margin[i] >= 0.:
            kind, parent, leaf, node = "known", lp, c, 1+len(meta["parent_names"])+c
        elif parent_margin[i] >= 0.:
            kind, parent, leaf, node = "intra_unknown", p, None, 1+p
        else:
            kind, parent, leaf, node = "global_unknown", None, None, 0
        row = dict(record)
        row.update(prediction_type=kind, parent=parent, leaf=leaf, output_node=node,
                   candidate_leaf=c, candidate_parent=p, leaf_candidate_parent=lp,
                   candidate_parent_name=meta["parent_names"][p], candidate_leaf_name=meta["leaf_names"][c],
                   route_parent=lp if kind == "known" else p,
                   selected_leaf_score=float(ls[i]), selected_parent_score=float(ps[i]), selected_leaf_parent_score=float(lps[i]),
                   raw_selected_leaf_score=float(data["leaf_score"][i]), raw_selected_parent_score=float(data["parent_score"][i]),
                   raw_selected_leaf_parent_score=float(data["leaf_parent_score"][i]),
                   leaf_threshold=float(lt), parent_threshold=float(pt), leaf_parent_threshold=float(pt),
                   comparison=COMPARISON, selected_score_space="raw score minus fitted parent-group offset",
                   root_knownness_score=float(max(leaf_margin[i], parent_margin[i])),
                   local_knownness_score=float(leaf_margin[i]), local_known_margin=float(leaf_margin[i]),
                   root_threshold=0., local_threshold=0., decoder=DECODER, variant=router["variant"],
                   root_score_type="max_leaf_path_and_fallback_parent_margin", local_score_type="min_leaf_score_and_own_parent_margin",
                   score_note="Continuous routing margins in evidence units; not probabilities")
        result.append(row)
    return result


def _paired(before, after, meta):
    if before is None:
        return None
    report = _legacy_paired(before, after, meta)
    report["source_protection_diagnostic_passed"] = report.pop("passed")
    report["sources_with_any_improved_component"] = report["benefiting_sources"]
    report["benefiting_sources"] = [key for key, value in report["per_unknown_source"].items()
        if value["selected_correct"] >= value["reference_correct"]
        and value["selected_false_leaves"] <= value["reference_false_leaves"]
        and (value["selected_correct"] > value["reference_correct"]
             or value["selected_false_leaves"] < value["reference_false_leaves"])]
    report.update(CONTEXT)
    report["known_count_preserved"] = report["selected"]["counts"]["known_correct"] >= report["reference"]["counts"]["known_correct"]
    return report


def _boundaries(values):
    with np.errstate(over="ignore"):
        low, high = np.nextafter(values.min(), -np.inf), np.nextafter(values.max(), np.inf)
    result = np.r_[low, np.unique(values), high]
    if not np.isfinite(result).all():
        raise ValueError("No finite all-pass/all-reject boundary")
    return result


def fit_router(known, near, extra, meta, settings=None, variant="global", reference_records=None):
    if variant != "global":
        raise ValueError("TaxoSieve requires the global D05 calibration variant")
    settings = validate_settings(settings)
    rows, input_count = _inputs(known, near, extra, meta)
    before, _ = _reference(rows, reference_records, meta)
    data = _data(rows, meta)
    offsets, offset_report = _offsets(rows, data, meta, settings, variant)
    ps, lps, ls = _adjusted(data, offsets["parent"], offsets["leaf"])
    pg, lg = _boundaries(np.r_[ps, lps]), _boundaries(ls)
    pa, lpa = ps[None, :] >= pg[:, None], lps[None, :] >= pg[:, None]
    km = np.asarray([r["status"] == "known" for r in rows]); nm = np.asarray([r["status"] == "intra" for r in rows]); em = ~(km | nm)
    tp = np.asarray([-1 if r.get("true_parent") is None else r["true_parent"] for r in rows]); tl = np.asarray([-1 if r.get("true_leaf") is None else r["true_leaf"] for r in rows])
    kc, nc = km & (data["leaf"] == tl), nm & (data["parent"] == tp)
    totals = [int(mask.sum()) for mask in (km, nm, em)]
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        key = row["true_leaf"] if row["status"] == "known" else normalized_name(str(row.get("source") or "unspecified"))
        groups[(row["status"], key)].append(index)
    best = None; feasible = known_feasible = known_precision = 0
    max_near_known_extra, max_ppv_known = None, None
    for lt in lg:
        leaves = lpa & (ls[None, :] >= lt)
        parents = pa & ~leaves; roots = ~pa & ~leaves
        correct = (leaves & kc) | (parents & nc) | (roots & em)
        kn, nn, en, ln = (np.sum(value, axis=1) for value in (leaves & kc, parents & nc, roots & em, leaves))
        kpass, npass, epass = kn*10 > 9*totals[0], nn*20 >= 17*totals[1], en*10 > 9*totals[2]
        ppass = (ln > 0) & (kn*10 > ln*9)
        allpass = kpass & npass & epass & ppass
        feasible += int(allpass.sum()); known_feasible += int(kpass.sum()); known_precision += int((kpass & ppass).sum())
        ppv = np.divide(kn, ln, out=np.zeros(len(pg)), where=ln != 0)
        if kpass.any(): max_ppv_known = max(max_ppv_known or 0., float(ppv[kpass].max()))
        if (kpass & epass).any(): max_near_known_extra = max(max_near_known_extra or 0, int(nn[kpass & epass].max()))
        deficit = (np.maximum(0, 9*totals[0]//10+1-kn)/totals[0] + np.maximum(0, (17*totals[1]+19)//20-nn)/totals[1]
                   + np.maximum(0, 9*totals[2]//10+1-en)/totals[2] + np.maximum(0, 9*ln//10+1-kn)/np.maximum(ln, 1))
        macro = {status: np.zeros(len(pg)) for status in base.STATUSES}
        group_count = defaultdict(int)
        for (status, key), indices in groups.items():
            macro[status] += correct[:, indices].sum(1)/len(indices); group_count[status] += 1
        quality = (sum(macro[s]/group_count[s] for s in base.STATUSES) + ppv)/4.
        # Every parent boundary is covered; vectorized metrics do not shortcut
        # threshold combinations or look at held data.
        order = np.lexsort((pg, quality, -deficit, kpass.astype(int), allpass.astype(int)))
        index = int(order[-1])
        key = (bool(allpass[index]), bool(kpass[index]), -float(deficit[index]), float(quality[index]), float(pg[index]), float(lt))
        if best is None or key > best[0]: best = key, float(pg[index]), float(lt)
    key, pt, lt = best
    with np.errstate(over="ignore"):
        parent_thresholds, leaf_thresholds = pt + offsets["parent"], lt + offsets["leaf"]
    if not np.isfinite(np.r_[parent_thresholds, leaf_thresholds]).all():
        raise ValueError("Parentwise threshold overflow")
    router = dict(schema_version=SCHEMA_VERSION, decoder=DECODER, meta=copy.deepcopy(meta), variant=variant,
                  settings=settings, settings_sha256=_hash(settings),
                  candidate_rule="supplied_before_thresholds;leaf_own_parent_admission;single_supplied_fallback",
                  parent_thresholds=parent_thresholds.tolist(), leaf_thresholds=leaf_thresholds.tolist(),
                  parent_offsets=offsets["parent"].tolist(), leaf_offsets=offsets["leaf"].tolist(), comparison=COMPARISON,
                  threshold_array_note="Nominal raw-unit display only; inference subtracts offset before comparing to global threshold",
                  global_parent_threshold=pt, global_leaf_threshold=lt, offsets=offset_report,
                  fit_completed=True, fit_splits=["val_known", "val_intra", "val_extra"],
                  fit_image_sha256=[base._digest(r) for r in rows], evidence_sha256=_hash([r["discovery"] for r in rows]),
                  input_record_count=input_count, unique_image_count=len(rows), duplicate_record_count=input_count-len(rows),
                  status="feasible" if key[0] else "best_effort", targets_passed=bool(key[0]), best_effort=not key[0],
                  baseline_fallback=False, **CONTEXT)
    router["router_sha256"] = _router_hash(router)
    predictions = decode_records(rows, router, meta)
    report = base.evaluate_records(predictions, meta)
    if bool(report["targets_passed"]) != router["targets_passed"]:
        raise AssertionError("Exact threshold search and decoded metrics disagree")
    paired = _paired(before, predictions, meta)
    report.update(**CONTEXT, status=router["status"], best_effort=router["best_effort"], paired_audit=paired,
                  known_count_preserved=None if paired is None else paired["known_count_preserved"],
                  selection_rule="four gates; known>90%; normalized gate deficit; mean known/near/extra source macro and accepted-leaf precision; highest finite parent then leaf boundary",
                  exact_threshold_search=dict(parent_grid=pg.tolist(), leaf_grid=lg.tolist(),
                      candidate_count=len(pg)*len(lg), all_four_gate_candidates=feasible, known_gate_candidates=known_feasible,
                      known_and_precision_gate_candidates=known_precision, maximum_precision_given_known_gate=max_ppv_known,
                      maximum_near_correct_given_known_and_extra_gates=max_near_known_extra,
                      threshold_tie_rule="score >= threshold accepts", includes_all_accept_and_all_reject=True,
                      scope="all empirical decision boundaries for fixed candidates and fixed fitted offsets; no claim beyond these frozen scores"))
    _hash(router); _hash(report)
    return router, report
