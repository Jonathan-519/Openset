"""DEV-only calibration of separate leaf acceptance and parent fallback.

TRAIN-fitted proximity is supplied by the caller. This module never fits that
transform, features, or a neural model. Cross-fitting diagnoses only thresholds
conditional on those fixed objects and the already selected model.
"""
from collections import defaultdict
import copy
import hashlib
import json

import numpy as np

from taxosafe_support import calibration as base
from taxosafe_support import membership_calibration as membership
from taxosafe_dcbs.protocol import normalized_name

SCHEMA_VERSION = "routealign_calibration_v1"
DECODER = "routealign"
DEFAULT_SETTINGS = {"grid_points": 49, "proximity_weight": 1., "rerank_weight": 1., "proximity_clip": 5., "seed": 1}
CONTEXT = dict(validation_scope="exploratory_calibration_conditional_on_previously_selected_model",
               independent_model_level_validation=False, confirmatory_validation=False,
               test_used_for_fitting=False)


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _settings(settings=None):
    if settings is not None and not isinstance(settings, dict):
        raise ValueError("Routealign settings must be a mapping")
    result = copy.deepcopy(settings or {})
    if set(result) - {*DEFAULT_SETTINGS, "baseline_calibration"}:
        raise ValueError("Unexpected routealign calibration settings")
    for key, default, low, high in (("grid_points", 49, 3, 49), ("seed", 1, 0, 2**31-1)):
        result.setdefault(key, default)
        if type(result[key]) is not int or not low <= result[key] <= high:
            raise ValueError("Invalid integer setting: " + key)
    for key, default, high in (("proximity_weight", 1., 4.), ("rerank_weight", 1., 4.), ("proximity_clip", 5., 5.)):
        result.setdefault(key, default)
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 < value <= high:
            raise ValueError("Invalid finite positive setting: " + key)
        result[key] = float(value)
    result.setdefault("baseline_calibration", {"decoder": "membership", "policy": "known_first", "membership_grid_points": 49})
    if not isinstance(result["baseline_calibration"], dict) or result["baseline_calibration"].get("decoder", "membership") != "membership":
        raise ValueError("Reference cross-fit calibration must use membership")
    _hash(result)
    return result


def validate_settings(settings=None):
    """Normalize public settings; reference calibration is runtime-only input."""
    result = _settings(settings)
    if not settings or "baseline_calibration" not in settings:
        result.pop("baseline_calibration")
    return result


def _meta(meta):
    base._hierarchy(meta)
    for key in ("parent_names", "leaf_names"):
        names = meta[key]
        if not isinstance(names, (list, tuple)) or any(not isinstance(v, str) or not v for v in names) or len(names) != len(set(names)):
            raise ValueError("Invalid routealign taxonomy names")
    if any(type(value) is not int for value in meta["leaf_to_parent"]):
        raise ValueError("Taxonomy parent indices must be integers")


def _proximity(records, meta):
    result = {}
    for key, size in (("parent_proximity", len(meta["parent_names"])), ("leaf_proximity", len(meta["leaf_names"]))):
        try:
            values = np.asarray([row["proximity"][key] for row in records], dtype=float)
            if not records:
                values = np.empty((0, size), dtype=float)
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError("Missing finite proximity vector: " + key) from exc
        if values.shape != (len(records), size) or not np.isfinite(values).all():
            raise ValueError("Invalid finite proximity vector: " + key)
        if any(isinstance(v, (bool, np.bool_)) for row in records for v in row["proximity"][key]):
            raise ValueError("Boolean proximity values are invalid")
        result[key] = values
    return result


def _unique(records, meta, require_proximity=True):
    records = list(records)
    base._scores(records, meta)
    membership.candidate_scores(records, meta)
    if require_proximity:
        _proximity(records, meta)
        seen = {}
        for row in records:
            key, value = base._digest(row), _hash(row["proximity"])
            if key in seen and seen[key] != value:
                raise ValueError("Same image has conflicting proximity evidence")
            seen[key] = value
    return sorted(base.unique_records(records), key=base._digest)


def _inputs(known, near, extra, meta, require_proximity=True):
    _meta(meta)
    groups = [list(known), list(near), list(extra)]
    raw = sum(groups, [])
    _unique(raw, meta, require_proximity)
    rows, _ = base._fit_inputs(*groups, meta)
    return sorted(rows, key=base._digest), len(raw)


def _reference(rows, reference_records, baseline_router, meta):
    membership._validate_state(baseline_router, meta)
    reference = _unique(rows if reference_records is None else reference_records, meta, False)
    by_hash = {base._digest(row): row for row in reference}
    if set(by_hash) != {base._digest(row) for row in rows}:
        raise ValueError("Reference and target require identical DEV image hashes")
    aligned = []
    for row in rows:
        other = by_hash[base._digest(row)]
        if any(row.get(key) != other.get(key) for key in ("split", "status", "source", "species", "true_leaf", "true_parent")):
            raise ValueError("Reference and target annotations differ")
        aligned.append(other)
    return aligned


def _arrays(records, meta, settings, variant):
    records = list(records)
    base._scores(records, meta)
    original = membership.candidate_scores(records, meta)
    proximity = _proximity(records, meta)
    index = np.arange(len(records))
    parent, leaf = original["parent"], original["leaf"]
    weight = settings["proximity_weight"]
    bounded = {key: np.clip(value, -settings["proximity_clip"], settings["proximity_clip"])
               for key, value in proximity.items()}
    with np.errstate(over="ignore", invalid="ignore"):
        leaf_score = original["leaf_score"] + weight * bounded["leaf_proximity"][index, leaf]
        fallback = parent.copy()
        if variant == "rerank" and records:
            logits = original["heads"]["parent_logits"]
            centered = logits - logits.max(1, keepdims=True)
            log_probs = centered - np.log(np.exp(centered).sum(1, keepdims=True))
            rerank = log_probs + settings["rerank_weight"] * bounded["parent_proximity"]
            if not np.isfinite(rerank).all():
                raise ValueError("Parent reranking scores must be finite")
            fallback = rerank.argmax(1)
        parent_score = original["heads"]["parent_membership_logits"][index, fallback] + weight * bounded["parent_proximity"][index, fallback]
    if not np.isfinite(leaf_score).all() or not np.isfinite(parent_score).all():
        raise ValueError("Fused route scores must be finite")
    return dict(original=original, parent=parent, leaf=leaf, fallback=fallback,
                leaf_score=leaf_score, parent_score=parent_score, proximity=proximity)


def _validate_state(state, meta):
    _meta(meta)
    if (not isinstance(state, dict) or state.get("schema_version") != SCHEMA_VERSION
            or state.get("decoder") != DECODER or state.get("meta") != meta
            or state.get("variant") not in ("joint", "rerank")
            or state.get("candidate_rule") != membership.CANDIDATE_RULE):
        raise ValueError("Routealign schema, decoder, taxonomy or variant mismatch")
    settings = _settings(state.get("settings"))
    if state.get("settings") != settings or state.get("settings_sha256") != _hash(settings):
        raise ValueError("Routealign settings binding changed")
    for key in ("leaf_parent_floor", "leaf_threshold", "parent_threshold"):
        value = state.get(key)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not np.isfinite(value):
            raise ValueError("Routealign threshold must be a finite scalar: " + key)
    if "router_sha256" in state and state["router_sha256"] != _router_hash(state):
        raise ValueError("Routealign router binding changed")
    return settings


def _router_hash(state):
    return _hash({key: state[key] for key in ("schema_version", "decoder", "meta", "variant", "candidate_rule",
                    "settings", "settings_sha256", "leaf_parent_floor", "leaf_threshold", "parent_threshold")})


def decode_records(records, state, meta):
    """Truth-independent routing; reranking happens only after rejecting leaf."""
    settings = _validate_state(state, meta)
    records = list(records)
    data = _arrays(records, meta, settings, state["variant"])
    result = []
    pcount = len(meta["parent_names"])
    for index, raw in enumerate(records):
        parent, leaf, fallback = (int(data[key][index]) for key in ("parent", "leaf", "fallback"))
        original_pm = float(data["original"]["parent_score"][index])
        original_lm = float(data["original"]["leaf_score"][index])
        with np.errstate(over="ignore"):
            leaf_margin = min(original_pm - state["leaf_parent_floor"], float(data["leaf_score"][index]) - state["leaf_threshold"])
            parent_margin = float(data["parent_score"][index]) - state["parent_threshold"]
        if not np.isfinite([leaf_margin, parent_margin]).all():
            raise ValueError("Route margins must be finite")
        if leaf_margin >= 0.:
            kind, rp, out_leaf, node = "known", parent, leaf, 1 + pcount + leaf
        elif parent_margin >= 0.:
            kind, rp, out_leaf, node = "intra_unknown", fallback, None, 1 + fallback
        else:
            kind, rp, out_leaf, node = "global_unknown", None, None, 0
        row = dict(raw)
        row.update(prediction_type=kind, parent=rp, leaf=out_leaf, output_node=node,
                   candidate_parent=parent, candidate_leaf=leaf,
                   candidate_parent_name=meta["parent_names"][parent], candidate_leaf_name=meta["leaf_names"][leaf],
                   route_parent=parent if kind == "known" else fallback,
                   fallback_candidate_parent=fallback, decoder=DECODER, variant=state["variant"],
                   parent_membership_score=original_pm, leaf_membership_score=original_lm,
                   fused_leaf_score=float(data["leaf_score"][index]), fused_parent_score=float(data["parent_score"][index]),
                   leaf_parent_floor=state["leaf_parent_floor"], parent_threshold=state["parent_threshold"], leaf_threshold=state["leaf_threshold"],
                   root_knownness_score=max(leaf_margin, parent_margin), local_knownness_score=leaf_margin,
                   root_threshold=0., local_threshold=0., local_known_margin=leaf_margin,
                   root_score_type="max_leaf_acceptance_and_parent_fallback_margin",
                   local_score_type="min_fixed_parent_floor_and_fused_leaf_margin",
                   score_note="Membership plus fixed-weight clipped TRAIN proximity; routing margins, not probabilities",
                   candidate_rule=membership.CANDIDATE_RULE)
        result.append(row)
    return result


def _correct(row):
    if row["status"] == "known":
        return row["prediction_type"] == "known" and row["leaf"] == row["true_leaf"] and row["parent"] == row["true_parent"]
    return (row["prediction_type"] == "intra_unknown" and row["parent"] == row["true_parent"]) if row["status"] == "intra" else row["prediction_type"] == "global_unknown"


def _wrong_leaf(row):
    return row["prediction_type"] == "known" and not _correct(row)


def paired_audit(before, after, meta):
    old = {base._digest(r): r for r in base.unique_records(before)}
    new = {base._digest(r): r for r in base.unique_records(after)}
    if set(old) != set(new):
        raise ValueError("Paired audit hashes differ")
    groups = defaultdict(list)
    known_lost, known_gained, wrong_added, near_lost, extra_lost, path_lost = [], [], [], [], [], []
    for key in sorted(old):
        b, a = old[key], new[key]
        if any(b.get(k) != a.get(k) for k in ("status", "split", "true_leaf", "true_parent", "source", "species")):
            raise ValueError("Paired audit annotations differ")
        status = b["status"]
        if status == "known":
            if _correct(b) and not _correct(a): known_lost.append(key)
            if not _correct(b) and _correct(a): known_gained.append(key)
        elif status == "intra":
            if _correct(b) and not _correct(a): near_lost.append(key)
            if b["prediction_type"] != "global_unknown" and b["parent"] == b["true_parent"] and (a["prediction_type"] == "global_unknown" or a["parent"] != b["true_parent"]): path_lost.append(key)
        elif _correct(b) and not _correct(a): extra_lost.append(key)
        if not _wrong_leaf(b) and _wrong_leaf(a): wrong_added.append(key)
        name = str(b.get("source") or "unspecified")
        group = int(b["true_leaf"]) if status == "known" else normalized_name(name)
        groups[(status, group)].append((b, a))
    per_source, per_leaf = {}, {}
    for (status, group), pairs in groups.items():
        entry = dict(sample_count=len(pairs), reference_correct=sum(_correct(b) for b, a in pairs),
                     selected_correct=sum(_correct(a) for b, a in pairs),
                     reference_false_leaves=sum(_wrong_leaf(b) for b, a in pairs),
                     selected_false_leaves=sum(_wrong_leaf(a) for b, a in pairs),
                     lost_correct=sum(_correct(b) and not _correct(a) for b, a in pairs),
                     gained_correct=sum(not _correct(b) and _correct(a) for b, a in pairs),
                     evidence_status="insufficient_evidence" if len(pairs) < 5 else "observed")
        if status == "known":
            per_leaf[meta["leaf_names"][group]] = entry
        else:
            entry.update(status=status, source=group, source_names=sorted({str(b.get("source") or "unspecified") for b, a in pairs}))
            per_source[status + ":" + group] = entry
    for name in meta["leaf_names"]:
        per_leaf.setdefault(name, dict(sample_count=0, reference_correct=0, selected_correct=0,
                         reference_false_leaves=0, selected_false_leaves=0, lost_correct=0, gained_correct=0,
                         evidence_status="not_evaluable"))
    macro = {}
    for status in ("intra", "extra"):
        values = [v for v in per_source.values() if v["status"] == status]
        macro[status] = {"source_count": len(values), **{key: None if not values else float(np.mean([v[key + "_correct"] / v["sample_count"] for v in values])) for key in ("reference", "selected")}}
    safe_sources = all(v["selected_correct"] >= v["reference_correct"] and v["selected_false_leaves"] <= v["reference_false_leaves"] for v in per_source.values())
    gains = [key for key, v in per_source.items() if v["selected_correct"] > v["reference_correct"] or v["selected_false_leaves"] < v["reference_false_leaves"]]
    return dict(**CONTEXT, reference=base.evaluate_records(list(old.values()), meta), selected=base.evaluate_records(list(new.values()), meta),
                known_lost_correct=len(known_lost), known_lost_correct_sha256=known_lost,
                known_gained_correct=len(known_gained), known_gained_correct_sha256=known_gained,
                new_wrong_leaf_count=len(wrong_added), new_wrong_leaf_sha256=wrong_added,
                known_new_wrong_leaf_count=sum(old[k]["status"] == "known" for k in wrong_added),
                near_lost_correct=len(near_lost), extra_lost_correct=len(extra_lost), near_parent_path_loss=len(path_lost),
                per_known_leaf=per_leaf, per_unknown_source=per_source, source_macro=macro,
                benefiting_sources=gains, unknown_sources_preserved=bool(safe_sources),
                passed=not known_lost and bool(safe_sources) and bool(gains))


def _grid(values, count):
    values = np.asarray(values, dtype=float)
    with np.errstate(over="ignore"):
        low, high = float(np.nextafter(values.min(), -np.inf)), float(np.nextafter(values.max(), np.inf))
    if not np.isfinite([low, high]).all():
        raise ValueError("No finite calibration grid endpoint")
    return np.unique(np.r_[low, np.quantile(values, np.linspace(0., 1., count-2)), high])


def fit_router(known, near, extra, meta, baseline_router, settings=None, variant="joint", *, reference_records=None):
    if variant not in ("joint", "rerank"):
        raise ValueError("Routealign fit variant must be joint or rerank")
    settings = _settings(settings)
    rows, input_count = _inputs(known, near, extra, meta)
    references = _reference(rows, reference_records, baseline_router, meta)
    before = membership.decode_records(references, baseline_router, meta)
    data = _arrays(rows, meta, settings, variant)
    state = dict(schema_version=SCHEMA_VERSION, decoder=DECODER, meta=copy.deepcopy(meta), variant=variant,
                 candidate_rule=membership.CANDIDATE_RULE, settings=settings, settings_sha256=_hash(settings),
                 leaf_parent_floor=float(baseline_router["parent_threshold"]))
    known_mask = np.asarray([r["status"] == "known" for r in rows])
    near_mask = np.asarray([r["status"] == "intra" for r in rows])
    extra_mask = ~(known_mask | near_mask)
    truthp = np.asarray([-1 if r.get("true_parent") is None else r["true_parent"] for r in rows])
    truthl = np.asarray([-1 if r.get("true_leaf") is None else r["true_leaf"] for r in rows])
    correct_candidates = known_mask & (data["parent"] == truthp) & (data["leaf"] == truthl)
    protected = np.asarray([r["status"] == "known" and _correct(r) for r in before])
    parent_floor = data["original"]["parent_score"] >= state["leaf_parent_floor"]
    reference_correct = np.asarray([_correct(r) for r in before])
    sources = defaultdict(list)
    for index, row in enumerate(rows):
        if row["status"] != "known": sources[(row["status"], normalized_name(str(row.get("source") or "unspecified")))].append(index)
    old_source = {key: (sum(reference_correct[i] for i in indices), sum(before[i]["prediction_type"] == "known" for i in indices)) for key, indices in sources.items()}
    parent_grid, leaf_grid = _grid(data["parent_score"], settings["grid_points"]), _grid(data["leaf_score"], settings["grid_points"])
    best, feasible, preserved_count = None, 0, 0
    for lt in leaf_grid:
        leaves = parent_floor & (data["leaf_score"] >= lt)
        correct_known = correct_candidates & leaves
        known_lost = int(np.sum(protected & ~correct_known))
        for pt in parent_grid:
            parents = ~leaves & (data["parent_score"] >= pt)
            roots = ~leaves & ~parents
            correct = correct_known | (near_mask & parents & (data["fallback"] == truthp)) | (extra_mask & roots)
            counts = {"known": int(known_mask.sum()), "intra": int(near_mask.sum()), "extra": int(extra_mask.sum()),
                      "known_correct": int(correct_known.sum()), "intra_correct": int(np.sum(correct & near_mask)),
                      "extra_correct": int(np.sum(correct & extra_mask)), "leaf_outputs": int(leaves.sum())}
            report = base._gates(counts)
            source_losses = false_increases = 0
            macros = defaultdict(list)
            for source, indices in sources.items():
                nc, nl = int(correct[indices].sum()), int(leaves[indices].sum())
                source_losses += max(0, old_source[source][0] - nc)
                false_increases += max(0, nl - old_source[source][1])
                macros[source[0]].append(nc / len(indices))
            macro = sum(float(np.mean(macros[s])) for s in ("intra", "extra")) / 2.
            deficit = sum(v["missing_correct"] / max(v["total"], 1) for v in report["requirements"].values())
            quality = (macro * 2. + (report["metrics"]["open_world_accepted_leaf_precision"] or 0.)) / 3.
            key = (-known_lost, report["targets_passed"], -source_losses, -false_increases,
                   -deficit, quality, counts["known_correct"], -abs(float(pt))-abs(float(lt)), -float(pt), -float(lt))
            preserved_count += int(known_lost == 0)
            feasible += int(known_lost == 0 and report["targets_passed"])
            if best is None or key > best[0]: best = key, float(pt), float(lt), report
    _, pt, lt, report = best
    state.update(parent_threshold=pt, leaf_threshold=lt)
    after = decode_records(rows, state, meta)
    paired = paired_audit(before, after, meta)
    state.update(**CONTEXT, status="feasible" if report["targets_passed"] and paired["known_lost_correct"] == 0 else "best_effort",
                 targets_passed=report["targets_passed"], best_effort=not bool(report["targets_passed"] and paired["known_lost_correct"] == 0),
                 baseline_fallback=False, fit_completed=True, fit_splits=["val_known", "val_intra", "val_extra"],
                 fit_image_sha256=[base._digest(r) for r in rows], input_record_count=input_count,
                 unique_image_count=len(rows), duplicate_record_count=input_count-len(rows),
                 reference_scope="same_model_reference" if reference_records is None else "paired_external_reference",
                 reference_router_sha256=_hash(baseline_router), reference_evidence_sha256=_hash(references),
                 validation_report=base.evaluate_records(after, meta), paired_audit=paired,
                 selection_rule="fewest reference-correct known losses; four gates; source correct losses; source false-leaf increases; normalized gate deficit; source-macro and precision; deterministic ties",
                 selection_diagnostics=dict(sampled_candidate_count=len(parent_grid)*len(leaf_grid), sampled_feasible_count=feasible,
                    sampled_known_preserving_count=preserved_count,
                    irrecoverable_reference_known_candidate_count=int(np.sum(protected & ~correct_candidates)),
                    reference_known_blocked_by_parent_floor_count=int(np.sum(protected & ~parent_floor)),
                    selected_known_lost_correct=paired["known_lost_correct"]),
                 grid=dict(parent_threshold=parent_grid.tolist(), leaf_threshold=leaf_grid.tolist(),
                    uses_fit_only=True, maximum_candidates=49*49, definition="fit empirical quantiles plus finite all-pass/all-reject endpoints; no continuous-exhaustiveness claim"))
    state["router_sha256"] = _router_hash(state)
    _hash(state)
    return state


def _folds(rows, settings):
    known_held = [[] for _ in range(3)]
    leaves = defaultdict(list)
    for row in rows:
        if row["status"] == "known": leaves[row["true_leaf"]].append(row)
    for leaf in sorted(leaves, key=lambda key: (-len(leaves[key]), key)):
        counts = [0] * len(known_held)
        for row in sorted(leaves[leaf], key=lambda r: _hash([settings["seed"], base._digest(r)])):
            fold = min(range(len(known_held)), key=lambda index: (counts[index], len(known_held[index]), index))
            known_held[fold].append(base._digest(row)); counts[fold] += 1
    result = [dict(fold_id="known_"+str(index), kind="known", held_image_sha256=sorted(values)) for index, values in enumerate(known_held)]
    sources = defaultdict(list)
    for row in rows:
        if row["status"] != "known": sources[(row["status"], normalized_name(str(row.get("source") or "unspecified")))].append(base._digest(row))
    result.extend(dict(fold_id="source_"+str(index), kind="unknown_source", status=status, source=source,
                       held_image_sha256=sorted(values)) for index, ((status, source), values) in enumerate(sorted(sources.items())))
    hashes = {base._digest(row) for row in rows}
    for fold in result:
        fold["fit_image_sha256"] = sorted(hashes - set(fold["held_image_sha256"]))
    return result


def crossfit_audit(known, near, extra, meta, baseline_router, settings=None, variant="joint", *, reference_records=None):
    """Known-stratified folds and leave-one-unknown-source-out, each fit-only.

Every content hash is held exactly once across the combined diagnostics. This
does not make fixed checkpoint/feature selection independent of development.
"""
    if variant not in ("joint", "rerank", "membership"):
        raise ValueError("Invalid cross-fit variant")
    settings = _settings(settings)
    rows, _ = _inputs(known, near, extra, meta, variant != "membership")
    references = _reference(rows, reference_records, baseline_router, meta)
    targets, reference = {base._digest(r): r for r in rows}, {base._digest(r): r for r in references}
    folds, before, after = _folds(rows, settings), [], []
    for fold in folds:
        fit = [targets[key] for key in fold["fit_image_sha256"]]
        held = [targets[key] for key in fold["held_image_sha256"]]
        ref_fit = [reference[key] for key in fold["fit_image_sha256"]]
        ref_held = [reference[key] for key in fold["held_image_sha256"]]
        groups = [[r for r in fit if r["status"] == status] for status in base.STATUSES]
        if not held or any(not group for group in groups):
            fold.update(execution_status="not_evaluable", reason="empty_held_or_missing_fit_status")
            continue
        rgroups = [[r for r in ref_fit if r["status"] == status] for status in base.STATUSES]
        bsettings = dict(settings["baseline_calibration"], source_loo=False)
        fb = membership.calibrate(*rgroups, meta, bsettings)
        b = membership.decode_records(ref_held, fb, meta)
        if variant == "membership":
            fitted = membership.calibrate(*groups, meta, bsettings)
            a = membership.decode_records(held, fitted, meta)
        else:
            fitted = fit_router(*groups, meta, fb, settings, variant, reference_records=ref_fit)
            a = decode_records(held, fitted, meta)
        before.extend(b); after.extend(a)
        fold.update(execution_status="completed", reference_fit_image_sha256=fb["fit_image_sha256"],
                    target_fit_image_sha256=fitted["fit_image_sha256"],
                    reference_parent_threshold=fb["parent_threshold"],
                    leaf_parent_floor=None if variant == "membership" else fitted["leaf_parent_floor"],
                    parent_threshold=fitted["parent_threshold"], leaf_threshold=fitted["leaf_threshold"],
                    reference_calibration_sha256=fb["calibration_sha256"],
                    target_calibration_sha256=fitted.get("router_sha256", fitted.get("calibration_sha256")),
                    held_report=paired_audit(b, a, meta))
    seen = [base._digest(row) for row in after]
    if len(seen) != len(set(seen)):
        raise AssertionError("Cross-fit image evaluated more than once")
    complete = set(seen) == set(targets)
    paired = paired_audit(before, after, meta) if after else None
    return dict(**CONTEXT, schema_version="routealign_crossfit_v1", variant=variant,
                passed=bool(complete and paired and paired["passed"]), complete=complete,
                status="completed" if complete else "not_evaluable", folds=folds,
                unit="unique_image_sha256", unique_image_count=len(rows), evaluated_image_count=len(after),
                reference_predictions=before, predictions=after, paired_audit=paired,
                counts=None if paired is None else {"reference": paired["reference"]["counts"], "selected": paired["selected"]["counts"]},
                known_lost_correct=None if paired is None else paired["known_lost_correct"],
                new_wrong_leaf_count=None if paired is None else paired["new_wrong_leaf_count"],
                source_macro=None if paired is None else paired["source_macro"],
                source_preservation_required="every held source correct count nondecreasing and false leaves nonincreasing; at least one strict source benefit",
                reference_floor_refit_on_fit_only=True, held_data_used_for_grid=False,
                output_used_for_threshold_selection=False,
                interpretation="Calibration-only diagnostics. Checkpoints and TRAIN proximity are fixed, previously selected objects; no independent model-level or confirmatory validation.")
