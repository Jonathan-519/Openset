"""Development-only ECDF scoring, source LOO, and constrained partial pooling.

Source LOO diagnoses fixed-score transfer; a scalar threshold cannot guarantee
performance on unseen sources. Recheck known constraints AFTER shrinkage.
"""
import copy
import math
import numpy as np

ROOT_FIELDS = ("root_ratio", "root_support")
LOCAL_FIELDS = ("local_ratio", "local_support", "sibling_margin")


def _unit(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def fit_score_support(taxonomy, fine, labels, meta):
    labels = np.asarray(labels, dtype=int)
    taxonomy, fine = _unit(taxonomy), _unit(fine)
    mapping = np.asarray(meta["leaf_to_parent"], dtype=int)
    if len(labels) != len(taxonomy) or len(labels) != len(fine):
        raise ValueError("Training features/labels differ in length")
    if set(labels.tolist()) != set(range(len(mapping))):
        raise ValueError("Every known leaf needs TRAIN-only score support")
    tax_leaves = _unit([taxonomy[labels == c].mean(0) for c in range(len(mapping))])
    parents = _unit([tax_leaves[mapping == p].mean(0) for p in range(len(meta["parent_names"]))])
    leaves = _unit([fine[labels == c].mean(0) for c in range(len(mapping))])
    return {"parent": parents.tolist(), "leaf": leaves.tolist()}


def _logmeanexp(x, temperature):
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    x = np.asarray(x, dtype=float) / temperature
    largest = x.max(axis=-1)
    return temperature * (largest + np.log(np.exp(x - largest[..., None]).mean(axis=-1)))


def raw_records(rows, output, original_leaf_logits, support, meta, settings):
    """Labels are carried for evaluation, never consulted for predictions."""
    pcount = len(meta["parent_names"])
    mapping = np.asarray(meta["leaf_to_parent"], dtype=int)
    a, b, stop = (np.asarray(output[k], dtype=float) for k in ("root", "leaf", "stop"))
    t, f = (_unit(output[k]) for k in ("taxonomy", "fine"))
    old = np.asarray(original_leaf_logits, dtype=float)
    n = len(rows)
    if a.shape != (n, pcount + 1) or b.shape != (n, len(mapping)) or stop.shape != (n, pcount) or old.shape != b.shape:
        raise ValueError("DCBS evidence/hierarchy shapes differ")
    if any(not np.isfinite(x).all() for x in (a, b, stop, t, f, old)):
        raise ValueError("Non-finite model evidence")
    root_ratio = _logmeanexp(a[:, :pcount], float(settings.get("root_temperature", 1.0))) - a[:, pcount]
    root_support = (t @ np.asarray(support["parent"]).T).max(1)
    fine_support = f @ np.asarray(support["leaf"]).T
    pred_parents, global_leaves = a[:, :pcount].argmax(1), old.argmax(1)
    result = []
    for i, row in enumerate(rows):
        p = int(pred_parents[i])
        child = np.where(mapping == p)[0]
        leaf = int(child[old[i, child].argmax()])
        if len(child) > 1:
            top = np.sort(b[i, child] / float(settings.get("logit_scale", 10.0)))[-2:]
            margin = float(top[1] - top[0])
        else:
            margin = None
        value = dict(row)
        value.update({"pred_parent": p, "pred_leaf": leaf, "global_pred_leaf": int(global_leaves[i]),
                      "path_consistent": bool(mapping[global_leaves[i]] == p),
                      "root_ratio": float(root_ratio[i]), "root_support": float(root_support[i]),
                      "local_ratio": float(_logmeanexp(b[i, child], float(settings.get("local_temperature", 1.0))) - stop[i, p]),
                      "local_support": float(fine_support[i, child].max()), "sibling_margin": margin})
        result.append(value)
    return result


def _values(rows, field):
    values = [float(r[field]) for r in rows if r.get(field) is not None]
    if values and not np.isfinite(values).all():
        raise ValueError("Non-finite calibration evidence: " + field)
    return np.sort(values).tolist()


def fit_cdf(known, parent_count, prior_count=30.0, use_margin=True):
    if not known or any(r["status"] != "known" for r in known):
        raise ValueError("ECDF references must be validation known samples only")
    fields = LOCAL_FIELDS if use_margin else LOCAL_FIELDS[:2]
    return {"root": {k: _values(known, k) for k in ROOT_FIELDS},
            "local": {k: _values(known, k) for k in fields},
            "by_parent": {str(p): {k: _values([r for r in known if r["pred_parent"] == p], k) for k in fields}
                          for p in range(parent_count)}, "prior_count": float(prior_count)}


def _cdf(value, reference):
    if not reference or value is None:
        return None
    a = np.asarray(reference, dtype=float)
    if a[-1] - a[0] < 1e-12:
        return None  # Constant channels contain no ranking evidence.
    if not math.isfinite(float(value)):
        raise ValueError("Non-finite inference evidence")
    rank = .5 * (np.searchsorted(a, value, side="left") + np.searchsorted(a, value, side="right"))
    return float((rank + .5) / (len(a) + 1.))


def _geomean(values):
    values = [v for v in values if v is not None]
    if not values:
        raise ValueError("All evidence channels are degenerate; cannot calibrate")
    return float(np.exp(np.log(values).mean()))


def score_records(rows, state):
    result = []
    for row in rows:
        value = dict(row)
        value["root_knownness_score"] = _geomean([_cdf(row[k], v) for k, v in state["root"].items()])
        local = []
        for field, pooled in state["local"].items():
            if row.get(field) is None:
                continue
            qpool = _cdf(row[field], pooled)
            ref = state["by_parent"][str(row["pred_parent"])][field]
            qparent = _cdf(row[field], ref)
            if qparent is not None and qpool is not None:
                weight = len(ref) / (len(ref) + state["prior_count"])
                local.append(weight * qparent + (1 - weight) * qpool)
            else:
                local.append(qpool if qpool is not None else qparent)
        value["local_knownness_score"] = _geomean(local)
        result.append(value)
    return result


def _candidates(rows, field, count):
    x = np.asarray([float(r[field]) for r in rows])
    return np.unique(np.r_[0., np.quantile(x, np.linspace(0, 1, count)), np.nextafter(1., 2.)])


def _mean(values):
    values = list(values)
    return float(np.mean(values)) if values else None


def _root_select(known, near, extra, settings):
    kfloor, nfloor = float(settings.get("root_known_retention", .97)), float(settings.get("root_near_retention", .95))
    by_source = {s: [r for r in extra if r["source"] == s] for s in sorted({r["source"] for r in extra})}
    best = None
    for tau in _candidates(known + near + extra, "root_knownness_score", int(settings.get("grid_points", 101))):
        kr = _mean(r["root_knownness_score"] >= tau for r in known)
        nr = _mean(r["root_knownness_score"] >= tau for r in near)
        if kr + 1e-12 < kfloor or nr + 1e-12 < nfloor:
            continue
        rates = {s: _mean(r["root_knownness_score"] < tau for r in group) for s, group in by_source.items()}
        key = (min(rates.values()), np.mean(list(rates.values())), kr + nr, -float(tau))
        if best is None or key > best[0]:
            best = (key, float(tau), {"known_retention": kr, "near_retention": nr, "extra_by_source": rates})
    if best is None:
        raise ValueError("No root threshold satisfies known/near retention constraints")
    return best[1], best[2]


def source_loo(known, near, extra, settings):
    sources = sorted({r["source"] for r in extra})
    if len(sources) < 2:
        return {"available": False, "reason": "At least two development extra sources are required", "folds": []}
    folds = []
    for held in sources:
        fitted = [r for r in extra if r["source"] != held]
        holdout = [r for r in extra if r["source"] == held]
        tau, operating = _root_select(known, near, fitted, settings)
        folds.append({"held_source": held, "fit_sources": sorted({r["source"] for r in fitted}),
                      "root_threshold": tau, "fit_operating_point": operating, "held_count": len(holdout),
                      "held_rejection_rate": _mean(r["root_knownness_score"] < tau for r in holdout)})
    return {"available": True, "folds": folds,
            "worst_held_rejection": min(r["held_rejection_rate"] for r in folds),
            "note": "Fixed scoring model; held source excluded from threshold candidates/objective. Monotone scalar rejection and retention constraints may give identical folds."}


def _accepted(row, root, local):
    return row["root_knownness_score"] >= root and row["local_knownness_score"] >= local and row["path_consistent"]


def _local_select(known, near, root, settings, candidates=None):
    floor = float(settings.get("local_known_retention", .90))
    if not known:
        return None
    best = None
    candidates = _candidates(known + near, "local_knownness_score", int(settings.get("grid_points", 101))) if candidates is None else candidates
    for tau in candidates:
        retention = _mean(r["local_knownness_score"] >= tau for r in known)
        if retention + 1e-12 < floor:
            continue
        correct = _mean(_accepted(r, root, tau) and r["pred_leaf"] == r["true_leaf"] for r in known)
        fallback = _mean(r["root_knownness_score"] >= root and not _accepted(r, root, tau)
                         and r["pred_parent"] == r["true_parent"] for r in near)
        objective = .5 * (correct + fallback) if near else -retention
        key = (objective, correct, -float(tau))
        if best is None or key > best[0]:
            best = (key, float(tau))
    return None if best is None else best[1]


def _e2e(known, root, thresholds):
    return _mean(_accepted(r, root, thresholds[str(r["pred_parent"])]) and r["pred_leaf"] == r["true_leaf"] for r in known)


def calibrate(known, near, extra, meta, settings):
    known, near, extra = list(known), list(near), list(extra)
    for rows, status, split in ((known, "known", "val_known"), (near, "intra", "val_intra"), (extra, "extra", "val_extra")):
        if not rows or any(r.get("status") != status or r.get("split") != split for r in rows):
            raise ValueError("Calibration requires only {} records".format(split))
    for key in ("root_known_retention", "root_near_retention", "local_known_retention", "min_known_e2e", "max_closed_drop"):
        if key in settings and not 0 <= float(settings[key]) <= 1:
            raise ValueError(key + " must lie in [0,1]")
    prior = float(settings.get("prior_count", 30.))
    if prior < 0 or int(settings.get("grid_points", 101)) < 3:
        raise ValueError("Invalid prior_count or grid_points")
    cdf = fit_cdf(known, len(meta["parent_names"]), prior, settings.get("use_margin", True))
    k, n, e = (score_records(r, cdf) for r in (known, near, extra))
    root, root_report = _root_select(k, n, e, settings)
    pooled = _local_select(k, n, root, settings)
    closed = _mean(r["global_pred_leaf"] == r["true_leaf"] for r in k)
    required = max(float(settings.get("min_known_e2e", .80)), closed - float(settings.get("max_closed_drop", .08)))
    thresholds = {str(p): pooled for p in range(len(meta["parent_names"]))}
    if _e2e(k, root, {p: 0. for p in thresholds}) + 1e-12 < required:
        raise ValueError("Root/path decisions cannot satisfy known E2E floor {:.4f}; improve development representation".format(required))
    if _e2e(k, root, thresholds) + 1e-12 < required:
        valid = [tau for tau in _candidates(k + n, "local_knownness_score", int(settings.get("grid_points", 101)))
                 if _e2e(k, root, {p: float(tau) for p in thresholds}) + 1e-12 >= required]
        pooled = min(pooled, max(valid))
        thresholds = {p: pooled for p in thresholds}
    branches = {}
    for p, name in enumerate(meta["parent_names"]):
        kp, np_ = [r for r in k if r["pred_parent"] == p], [r for r in n if r["pred_parent"] == p]
        raw = _local_select(kp, np_, root, settings)
        effective_n = len(kp) + len(np_) if np_ else len(kp)
        weight = effective_n / (effective_n + prior) if effective_n else 0.
        if raw is None:
            raw, weight = pooled, 0.
        if not settings.get("partial_pooling", True):
            weight = 0.
        tau = weight * raw + (1 - weight) * pooled
        if kp:
            feasible = [t for t in _candidates(kp, "local_knownness_score", int(settings.get("grid_points", 101)))
                        if _mean(r["local_knownness_score"] >= t for r in kp) + 1e-12 >= float(settings.get("local_known_retention", .90))]
            tau = min(tau, max(feasible))
        thresholds[str(p)] = float(tau)
        branches[str(p)] = {"name": name, "raw_threshold": raw, "shrinkage_weight": weight,
                            "known_count": len(kp), "near_count": len(np_),
                            "source": "known+near" if kp and np_ else "known_quantile" if kp else "pooled"}
    if not settings.get("partial_pooling", True):
        thresholds = {p: min(thresholds.values()) for p in thresholds}
    offset = 0.
    if _e2e(k, root, thresholds) + 1e-12 < required:
        offsets = sorted(set([0.] + [max(0., thresholds[str(r["pred_parent"])] - r["local_knownness_score"]) + 1e-12 for r in k] + [1.]))
        for offset in offsets:
            adjusted = {p: max(0., tau - offset) for p, tau in thresholds.items()}
            if _e2e(k, root, adjusted) + 1e-12 >= required:
                thresholds = adjusted
                break
    if _e2e(k, root, thresholds) + 1e-12 < required:
        raise ValueError("Final shrunk thresholds violate known E2E constraint")
    return {"schema_version": 11, "method": "TaxoSafe-DCBS", "meta": copy.deepcopy(meta),
            "settings": dict(settings), "cdf": cdf, "root_threshold": root,
            "pooled_local_threshold": pooled, "local_thresholds": thresholds, "branches": branches,
            "source_loo": source_loo(k, n, e, settings), "root_operating_point": root_report,
            "known_guard": {"closed_accuracy": closed, "required_e2e": required,
                            "final_e2e": _e2e(k, root, thresholds), "local_threshold_offset": offset}}


def apply_router(records, router, meta):
    if router.get("schema_version") != 11 or router.get("meta") != meta:
        raise ValueError("DCBS router version or label mapping mismatch")
    result = []
    for r in score_records(records, router["cdf"]):
        p, c = int(r["pred_parent"]), int(r["pred_leaf"])
        if r["root_knownness_score"] < router["root_threshold"]:
            kind, parent, leaf = "global_unknown", None, None
        elif not r["path_consistent"] or r["local_knownness_score"] < router["local_thresholds"][str(p)]:
            kind, parent, leaf = "intra_unknown", p, None
        else:
            kind, parent, leaf = "known", p, c
        r.update({"prediction_type": kind, "candidate_parent": p, "candidate_leaf": c, "parent": parent, "leaf": leaf,
                  "candidate_parent_name": meta["parent_names"][p], "candidate_leaf_name": meta["leaf_names"][c],
                  "true_parent_name": None if r.get("true_parent") is None else meta["parent_names"][r["true_parent"]],
                  "true_leaf_name": None if r.get("true_leaf") is None else meta["leaf_names"][r["true_leaf"]],
                  "root_threshold": router["root_threshold"], "local_threshold": router["local_thresholds"][str(p)]})
        result.append(r)
    return result
