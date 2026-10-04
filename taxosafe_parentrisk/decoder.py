"""Truth-independent, non-leaf parent compatibility and shared rejection.

The support top two are ranked once by the new score, before either acceptance
threshold is checked. No candidate is tried again after rejection. The parent
module never changes an original leaf decision. Rejectors can only remove a
leaf/root path, never create or replace a leaf.
"""
import copy
import hashlib
import json

import numpy as np

from taxosafe_support import calibration as base
from taxosafe_support import membership_calibration as membership

SCHEMA_VERSION = "parentrisk_v1"
DECODER = "parentrisk"
VECTOR_FIELDS = ("text_z", "membership_z", "geometry_z", "geometry_raw")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def evidence_arrays(records, meta):
    rows = list(records)
    p = len(meta["parent_names"])
    result = {}
    for name in VECTOR_FIELDS:
        try:
            values = np.asarray([r["parent_evidence"][name] for r in rows], dtype=float)
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError("Missing parent_evidence." + name) from exc
        if not rows:
            values = np.empty((0, p))
        if values.shape != (len(rows), p) or not np.isfinite(values).all():
            raise ValueError("Parent evidence must contain finite full-taxonomy vectors: " + name)
        result[name] = values
    return result


def parent_candidates(records, meta, weights):
    rows = list(records)
    d = membership.candidate_scores(rows, meta)
    e = evidence_arrays(rows, meta)
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (3,) or not np.isfinite(weights).all() or (weights < 0).any() or not np.isclose(weights.sum(), 1.):
        raise ValueError("Parent weights must be three nonnegative values summing to one")
    scores = sum(w * e[k] for w, k in zip(weights, VECTOR_FIELDS[:3]))
    if not np.isfinite(scores).all():
        raise ValueError("Parent compatibility overflow")
    top2 = np.argsort(-d["heads"]["parent_logits"], axis=1, kind="stable")[:, :2]
    restricted = np.take_along_axis(scores, top2, axis=1)
    local = restricted.argmax(1)
    idx = np.arange(len(rows))
    parent = top2[idx, local]
    selected = scores[idx, parent]
    margin = (np.abs(restricted[:, 0] - restricted[:, 1]) if restricted.shape[1] == 2
              else np.zeros(len(rows)))
    return dict(parent=parent, score=selected, margin=margin, top2=top2, scores=scores)


def make_router(baseline, meta, mode="combined", parent_rule=None, reject_rules=None):
    membership._validate_state(baseline, meta)
    return dict(schema_version=SCHEMA_VERSION, decoder=DECODER, mode=mode,
                meta=copy.deepcopy(meta), baseline_router=copy.deepcopy(baseline),
                baseline_router_sha256=digest(baseline), parent_rule=copy.deepcopy(parent_rule),
                reject_rules=copy.deepcopy(reject_rules or []))


def _finite(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and np.isfinite(value)


def validate_router(router, meta):
    if (router.get("schema_version") != SCHEMA_VERSION or router.get("decoder") != DECODER
            or router.get("meta") != meta or router.get("mode") not in ("audit", "parent_only", "combined")):
        raise ValueError("Parentrisk schema, decoder, mode or hierarchy mismatch")
    baseline = router["baseline_router"]
    membership._validate_state(baseline, meta)
    if router.get("baseline_router_sha256") != digest(baseline):
        raise ValueError("Parentrisk baseline binding mismatch")
    pr = router.get("parent_rule")
    if pr is not None:
        if not isinstance(pr, dict) or set(pr) != {"weights", "threshold", "margin_threshold"}:
            raise ValueError("Invalid parent compatibility rule")
        w = pr["weights"]
        if (not isinstance(w, list) or len(w) != 3 or not all(_finite(v) and v >= 0 for v in w)
                or not np.isclose(sum(w), 1.) or not _finite(pr["threshold"])
                or not _finite(pr["margin_threshold"]) or pr["margin_threshold"] < 0):
            raise ValueError("Invalid parent compatibility parameters")
    rules = router.get("reject_rules")
    if not isinstance(rules, list) or len(rules) > 2:
        raise ValueError("At most two shared rejection rules are permitted")
    actions = []
    for rule in rules:
        if (not isinstance(rule, dict) or set(rule) != {"action", "x_max", "y_max"}
                or rule["action"] not in ("leaf_reject", "root_reject")
                or not _finite(rule["x_max"]) or not _finite(rule["y_max"])):
            raise ValueError("Invalid shared rejection rule")
        actions.append(rule["action"])
    if actions != [x for x in ("leaf_reject", "root_reject") if x in actions]:
        raise ValueError("Shared rejection rules must be unique and ordered")
    if router["mode"] == "parent_only" and rules:
        raise ValueError("parent_only cannot change original leaf predictions")
    if router["mode"] == "audit" and pr is not None:
        raise ValueError("audit cannot use the parent compatibility module")
    return baseline


def apply_router(records, router, meta):
    from .folds import unique_records
    rows = list(records)
    unique_records(rows)  # validate aliases; inference preserves caller row order.
    baseline = validate_router(router, meta)
    original = base.apply_router(rows, baseline, meta)
    evidence = evidence_arrays(rows, meta)
    pr = router["parent_rule"]
    choices = parent_candidates(rows, meta, pr["weights"]) if pr else None
    result = []
    for i, old in enumerate(original):
        out = dict(old)
        rp = int(old["candidate_parent"])
        kind, parent, leaf, node = (old[k] for k in ("prediction_type", "parent", "leaf", "output_node"))
        applied = []
        proposed, compat, margin = rp, None, None
        if choices is not None:
            proposed, compat, margin = int(choices["parent"][i]), float(choices["score"][i]), float(choices["margin"][i])
            # Failed compatibility keeps the original root/parent decision.
            # There is no truth-dependent 'repair only wrong parents' clause.
            if (kind != "known" and compat >= pr["threshold"] and margin >= pr["margin_threshold"]):
                if kind != "intra_unknown" or parent != proposed:
                    applied.append("parent")
                rp, kind, parent, leaf, node = proposed, "intra_unknown", proposed, None, 1 + proposed
        for rule in router["reject_rules"]:
            action = rule["action"]
            if action == "leaf_reject":
                scope = kind == "known"
                x, y = float(out["baseline_leaf_z"]), float(out["geometry_leaf_score"])
            else:
                scope = kind != "global_unknown"
                # Both scores belong to the CURRENT route parent, not the old candidate.
                x, y = float(evidence["membership_z"][i, rp]), float(evidence["geometry_z"][i, rp])
            if scope and x <= rule["x_max"] and y <= rule["y_max"]:
                applied.append(action)
                if action == "leaf_reject":
                    kind, parent, leaf, node = "intra_unknown", rp, None, 1 + rp
                else:
                    kind, parent, leaf, node = "global_unknown", None, None, 0
        out.update(prediction_type=kind, parent=parent, leaf=leaf, output_node=node,
                   decoder=DECODER, route_parent=rp, proposed_parent=proposed,
                   parent_compatibility_score=compat, parent_compatibility_margin=margin,
                   baseline_prediction_type=old["prediction_type"], baseline_parent=old["parent"],
                   baseline_leaf=old["leaf"], baseline_output_node=old["output_node"],
                   baseline_parent_threshold=float(baseline["parent_threshold"]),
                   baseline_leaf_threshold=float(baseline["leaf_threshold"]),
                   baseline_root_knownness_score=old["root_knownness_score"],
                   baseline_local_knownness_score=old["local_knownness_score"], applied_rule_indices=applied,
                   root_knownness_score=1. if kind != "global_unknown" else -1.,
                   local_knownness_score=1. if kind == "known" else -1.,
                   local_known_margin=1. if kind == "known" else -1.,
                   root_threshold=0., local_threshold=0., parent_threshold=0., leaf_threshold=0.,
                   root_score_type="parentrisk_route_indicator_not_probability",
                   local_score_type="parentrisk_route_indicator_not_probability")
        result.append(out)
    return result
