"""NumPy-only visual retrieval, routing and calibration.

This is a TaxoSafe adaptation inspired by Deep kNN (ICML 2022), Tip-Adapter
(ECCV 2022) and Open-Set Plankton Recognition (2025), NOT a reproduction of
any of those methods. No test examples are used by the fitting functions.
All decision scores increase with support for accepting an image.
"""

import copy
import hashlib
from collections import defaultdict

import numpy as np


ROOT_FEATURES = [
    "parent_cosine", "parent_margin", "parent_neg_entropy",
    "parent_prototype", "global_knn",
]
CHILD_FEATURES = [
    "child_cosine", "child_margin", "child_knn", "child_prototype",
]
# A small, pre-declared candidate set. Select on validation-fit ONLY.
ROOT_CANDIDATES = {
    "text_entropy": [0, 0, 1, 0, 0],
    "visual_knn": [0, 0, 0, 0, 1],
    "visual_parent": [0, 0, 0, 1, 0],
    "semantic_visual": [0.25, 0, 0, 0.25, 0.50],
    "margin_visual": [0.25, 0.25, 0, 0.25, 0.25],
}
CHILD_CANDIDATES = {
    "text_cosine": [1, 0, 0, 0],
    "visual_knn": [0, 0, 1, 0],
    "visual_prototype": [0, 0, 0, 1],
    "semantic_visual": [0.25, 0.25, 0.50, 0],
    "visual_ensemble": [0, 0, 0.50, 0.50],
}


def normalise(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Features must be finite matrices")
    lengths = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(lengths < 1e-12):
        raise ValueError("Zero-norm image/prototype feature")
    return x / lengths


def softmax(x, scale=1.0):
    z = np.asarray(x, dtype=np.float64) * float(scale)
    z = z - np.max(z, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def validate_meta(meta):
    parents = meta["parent_names"]
    leaves = meta["leaf_names"]
    mapping = np.asarray(meta["leaf_to_parent"], dtype=np.int64)
    if len(mapping) != len(leaves) or len(set(leaves)) != len(leaves):
        raise ValueError("Invalid or duplicated leaf taxonomy")
    if set(mapping.tolist()) != set(range(len(parents))):
        raise ValueError("Taxonomy must have at least one leaf per parent")
    return [np.where(mapping == g)[0] for g in range(len(parents))]


def make_bank(features, labels, meta):
    children = validate_meta(meta)
    features = normalise(features)
    labels = np.asarray(labels, dtype=np.int64)
    if len(features) != len(labels):
        raise ValueError("Memory feature/label length mismatch")
    if set(labels.tolist()) != set(range(len(meta["leaf_names"]))):
        raise ValueError("Memory must contain every known leaf and no unknowns")
    leaf_prototypes = normalise(np.stack([
        features[labels == c].mean(axis=0)
        for c in range(len(meta["leaf_names"]))
    ]))
    # Equal weight per species; a frequent leaf must not dominate its parent.
    parent_prototypes = normalise(np.stack([
        leaf_prototypes[ids].mean(axis=0) for ids in children
    ]))
    return {
        "features": features.astype(np.float32),
        "labels": labels,
        "leaf_prototypes": leaf_prototypes.astype(np.float32),
        "parent_prototypes": parent_prototypes.astype(np.float32),
    }


def _kth_similarity(similarities, k):
    if similarities.shape[1] == 0:
        raise ValueError("No support examples for kNN")
    k = min(max(int(k), 1), similarities.shape[1])
    return np.partition(similarities, similarities.shape[1] - k, axis=1)[
        :, similarities.shape[1] - k
    ]


def retrieve(features, bank, meta, k=3, root_k=10, chunk_size=128):
    """Class k-th cosine and global k-th cosine, with bounded query memory.

    For L2-normalized vectors squared distance = 2 - 2 * cosine.
    Thus using k-th largest cosine is equivalent to k-th nearest distance.
    """
    validate_meta(meta)
    if int(k) < 1 or int(root_k) < 1 or int(chunk_size) < 1:
        raise ValueError("child_k, root_k and query_chunk_size must be positive")
    q = normalise(features)
    if not len(q):
        raise ValueError("No query images")
    reference = normalise(bank["features"])
    if q.shape[1] != reference.shape[1]:
        raise ValueError("Memory/query feature dimensions differ")
    labels = np.asarray(bank["labels"], dtype=np.int64)
    class_scores, global_scores, neighbor_indices = [], [], []
    for start in range(0, len(q), int(chunk_size)):
        sims = np.clip(q[start:start + int(chunk_size)] @ reference.T, -1, 1)
        class_scores.append(np.stack([
            _kth_similarity(sims[:, labels == c], k)
            for c in range(len(meta["leaf_names"]))
        ], axis=1))
        global_scores.append(_kth_similarity(sims, root_k))
        neighbor_indices.append(np.stack([
            np.where(labels == c)[0][sims[:, labels == c].argmax(axis=1)]
            for c in range(len(meta["leaf_names"]))
        ], axis=1))
    return {
        "class_knn": np.concatenate(class_scores),
        "global_knn": np.concatenate(global_scores),
        "class_neighbor_index": np.concatenate(neighbor_indices),
        "leaf_prototype": np.clip(q @ normalise(bank["leaf_prototypes"]).T, -1, 1),
        "parent_prototype": np.clip(q @ normalise(bank["parent_prototypes"]).T, -1, 1),
    }


def route(parent_cosine, leaf_cosine, support, meta, scale,
          parent_alpha=0.0, child_alpha=0.0, cache_scale=20.0):
    """Blend frozen text predictions with class-balanced visual retrieval.

    This probability blend is Tip-Adapter-inspired, not its cache-logit rule.
    Children are always restricted to the selected parent.
    """
    children = validate_meta(meta)
    p = np.asarray(parent_cosine)
    l = np.asarray(leaf_cosine)
    parent_probs = ((1.0 - parent_alpha) * softmax(p, scale)
                    + parent_alpha * softmax(support["parent_prototype"], cache_scale))
    pp = parent_probs.argmax(axis=1)
    pl = np.empty(len(pp), dtype=np.int64)
    for g, ids in enumerate(children):
        rows = np.where(pp == g)[0]
        if not len(rows):
            continue
        local = ((1.0 - child_alpha) * softmax(l[rows][:, ids], scale)
                 + child_alpha * softmax(support["class_knn"][rows][:, ids], cache_scale))
        pl[rows] = ids[local.argmax(axis=1)]
    return pp, pl


def evidence(parent_cosine, leaf_cosine, support, meta, scale, routing):
    p = np.asarray(parent_cosine, dtype=np.float64)
    l = np.asarray(leaf_cosine, dtype=np.float64)
    pp, pl = route(p, l, support, meta, scale, **routing)
    children = validate_meta(meta)
    rows = np.arange(len(pp))
    other_parent = p.copy()
    other_parent[rows, pp] = -np.inf
    parent_margin = p[rows, pp] - other_parent.max(axis=1)
    probs = softmax(p, scale)
    parent_entropy = (probs * np.log(np.maximum(probs, 1e-12))).sum(axis=1)
    parent_entropy /= max(np.log(p.shape[1]), 1.0)
    child_margin = np.zeros(len(pp))
    for g, ids in enumerate(children):
        if len(ids) < 2:
            continue
        selected_rows = np.where(pp == g)[0]
        for i in selected_rows:
            others = ids[ids != pl[i]]
            child_margin[i] = l[i, pl[i]] - l[i, others].max()
    root = np.stack([
        p[rows, pp], parent_margin, parent_entropy,
        support["parent_prototype"][rows, pp], support["global_knn"],
    ], axis=1)
    child = np.stack([
        l[rows, pl], child_margin, support["class_knn"][rows, pl],
        support["leaf_prototype"][rows, pl],
    ], axis=1)
    if not np.isfinite(root).all() or not np.isfinite(child).all():
        raise ValueError("Non-finite retrieval evidence")
    return {"pred_parent": pp, "pred_leaf": pl, "root": root,
            "child": child, "global_pred_leaf": l.argmax(axis=1),
            "text_pred_parent": p.argmax(axis=1),
            "neighbor_index": support["class_neighbor_index"][rows, pl]}


def validation_partition(records, fraction=0.5, seed=41):
    """Deterministic stratification by known leaf or unknown source/species.

    This is an IMAGE-level split inside validation, not an additional
    species-disjoint experiment. Selection/calibration indices are disjoint.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError("fit_fraction must be strictly between 0 and 1")
    grouped = defaultdict(list)
    for i, r in enumerate(records):
        key = (r["status"], str(r["true_leaf"]) if r["status"] == "known" else r["source"])
        grouped[key].append(i)
    fit, calibration = [], []
    for key in sorted(grouped):
        indices = sorted(grouped[key], key=lambda i: hashlib.sha256(
            (str(seed) + records[i]["image_sha256"]).encode("utf-8")
        ).hexdigest())
        if len(indices) < 2:
            raise ValueError("Validation stratum {} needs >=2 images".format(key))
        n = min(len(indices) - 1, max(1, int(len(indices) * fraction)))
        fit.extend(indices[:n])
        calibration.extend(indices[n:])
    return np.asarray(sorted(fit)), np.asarray(sorted(calibration))


def _macro(values, groups):
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups, dtype=str)
    if not len(values):
        return 0.0
    return float(np.mean([values[groups == g].mean() for g in np.unique(groups)]))


def routing_fit(p, l, support, meta, scale, records, fit, settings):
    """Choose two small blend weights on validation-fit, never test/calibration."""
    fit = np.asarray(fit)
    inside = fit[[records[i]["status"] != "extra" for i in fit]]
    known = fit[[records[i]["status"] == "known" for i in fit]]
    true_p = np.asarray([records[i]["true_parent"] for i in inside])
    true_l = np.asarray([records[i]["true_leaf"] for i in known])
    p_groups = [records[i]["status"] + ":" + (
        str(records[i]["true_parent"]) if records[i]["status"] == "known"
        else records[i]["source"]) for i in inside]
    cache_scale = float(settings.get("cache_scale", 20.0))
    candidates = settings.get("routing_alphas", [0.0, 0.25, 0.50])
    if not candidates or any(not 0 <= float(a) <= 1 for a in candidates):
        raise ValueError("routing_alphas must be in [0,1]")
    results = []
    for a in candidates:
        pp, _ = route(p, l, support, meta, scale, float(a), 0.0, cache_scale)
        results.append((_macro(pp[inside] == true_p, p_groups), -float(a), float(a)))
    parent_alpha = max(results)[2]
    leaf_results = []
    for a in candidates:
        pp, pl = route(p, l, support, meta, scale, parent_alpha, float(a), cache_scale)
        leaf_results.append((_macro(pl[known] == true_l, true_l), -float(a), float(a)))
    return {"parent_alpha": parent_alpha, "child_alpha": max(leaf_results)[2],
            "cache_scale": cache_scale}, {
        "parent_candidates": [{"alpha": x[2], "macro_accuracy": x[0]} for x in results],
        "child_candidates": [{"alpha": x[2], "macro_accuracy": x[0]} for x in leaf_results],
    }


def fit_moments(x, true_group, reference_mask, group_count, strength=20.0):
    """Shrink small-group moments to pooled moments, fitted on val-fit only."""
    x = np.asarray(x, dtype=float)
    true_group = np.asarray(true_group)
    reference_mask = np.asarray(reference_mask, dtype=bool)
    pool = x[reference_mask]
    if not len(pool):
        raise ValueError("No correctly classified validation-fit references")
    mu0, var0 = pool.mean(axis=0), pool.var(axis=0)
    moments = []
    for g in range(group_count):
        group = x[reference_mask & (true_group == g)]
        n = len(group)
        a = n / float(n + strength) if n else 0.0
        mu, var = (group.mean(axis=0), group.var(axis=0)) if n else (mu0, var0)
        mean = a * mu + (1.0 - a) * mu0
        variance = a * (var + (mu - mean) ** 2) + (1.0 - a) * (var0 + (mu0 - mean) ** 2)
        moments.append({"mean": mean.tolist(), "std": np.sqrt(np.maximum(variance, 1e-6)).tolist(),
                        "n": n, "source": "shrunk_group" if n else "pooled_no_correct_fit_sample"})
    return moments


def standardise(x, predicted_group, moments):
    means = np.asarray([m["mean"] for m in moments])
    stds = np.asarray([m["std"] for m in moments])
    return (np.asarray(x) - means[predicted_group]) / stds[predicted_group]


def weighted_score(z, weights, pred_parent=None, meta=None):
    weights = np.asarray(weights, dtype=float)
    if weights.sum() <= 0 or (weights < 0).any():
        raise ValueError("Invalid evidence weights")
    effective = np.tile(weights, (len(z), 1))
    if pred_parent is not None:
        counts = np.asarray([len(c) for c in validate_meta(meta)])
        # A singleton's child margin is exactly zero and is not evidence.
        effective[counts[pred_parent] == 1, 1] = 0.0
    if (effective.sum(axis=1) <= 0).any():
        raise ValueError("No informative score for a singleton branch")
    return (np.asarray(z) * effective).sum(axis=1) / effective.sum(axis=1)


def risk_threshold(scores, negative_mask, groups, limit=0.05):
    """Lowest scalar threshold satisfying EVERY observed negative source.

    This is an empirical calibration condition, not a population guarantee.
    The >= comparison and ties are handled explicitly, including reject-all.
    No branch-specific threshold substitution can silently break the union.
    """
    scores = np.asarray(scores, dtype=float)
    negative_mask = np.asarray(negative_mask, dtype=bool)
    groups = np.asarray(groups, dtype=str)
    if not 0 <= limit < 1 or not negative_mask.any():
        raise ValueError("Risk calibration needs negatives and 0 <= limit < 1")
    threshold = -np.inf
    for group in np.unique(groups[negative_mask]):
        values = np.sort(scores[negative_mask & (groups == group)])[::-1]
        allowed = int(np.floor(float(limit) * len(values) + 1e-12))
        # threshold > the (allowed+1)-th largest score; ties are all rejected.
        local = np.nextafter(values[allowed], np.inf)
        threshold = max(threshold, local)
    return float(threshold)


def coverage_threshold(scores, correct_mask, rejection=0.10):
    values = np.sort(np.asarray(scores)[np.asarray(correct_mask, dtype=bool)])
    if not len(values) or not 0 <= rejection < 1:
        raise ValueError("Coverage calibration needs correctly routed positives")
    allowed = min(len(values) - 1, int(np.floor(rejection * len(values))))
    return float(values[allowed])


def _source_risks(accepted, mask, groups):
    accepted, mask = np.asarray(accepted), np.asarray(mask, dtype=bool)
    groups = np.asarray(groups, dtype=str)
    return {g: {"n": int(np.sum(mask & (groups == g))),
                "accepted": int(np.sum(accepted & mask & (groups == g))),
                "rate": float(accepted[mask & (groups == g)].mean())}
            for g in np.unique(groups[mask])}


def select_evidence(z, candidates, correct, negative, groups, positive_groups,
                    selection_mask, limit, pred_parent=None, meta=None):
    results = []
    selected_score, selected_weights, selected_name, best = None, None, None, None
    for name, weights in candidates.items():
        score = weighted_score(z, weights, pred_parent, meta)
        tau = risk_threshold(score, negative & selection_mask, groups, limit)
        accept = score >= tau
        positive = selection_mask & ~negative
        utility = _macro(accept[positive] & correct[positive], np.asarray(positive_groups)[positive])
        # Tie-break by micro correct acceptance, then insertion order.
        key = (utility, float(np.mean(accept[positive] & correct[positive])))
        results.append({"name": name, "weights": weights, "fit_threshold": tau,
                        "macro_correct_acceptance": utility,
                        "fit_negative_source_rates": _source_risks(accept, negative & selection_mask, groups)})
        if best is None or key > best:
            best, selected_score, selected_weights, selected_name = key, score, weights, name
    return selected_score, {"name": selected_name, "weights": selected_weights}, results


def fit_calibration(e, records, meta, fit, calibration, settings):
    n = len(records)
    fit_mask = np.zeros(n, bool); fit_mask[fit] = True
    cal_mask = np.zeros(n, bool); cal_mask[calibration] = True
    if np.any(fit_mask & cal_mask) or not np.all(fit_mask | cal_mask):
        raise ValueError("Validation fit/calibration must be a disjoint partition")
    status = np.asarray([r["status"] for r in records])
    true_p = np.asarray([-1 if r["true_parent"] is None else r["true_parent"] for r in records])
    true_l = np.asarray([-1 if r["true_leaf"] is None else r["true_leaf"] for r in records])
    pp, pl = e["pred_parent"], e["pred_leaf"]
    source = np.asarray([r["source"] for r in records])
    root_correct = (status != "extra") & (pp == true_p)
    child_correct = (status == "known") & (pl == true_l) & (pp == true_p)
    strength = float(settings.get("shrinkage_strength", 20.0))
    if strength <= 0:
        raise ValueError("shrinkage_strength must be positive")
    root_moments = fit_moments(e["root"], true_p, fit_mask & root_correct, len(meta["parent_names"]), strength)
    child_moments = fit_moments(e["child"], true_l, fit_mask & child_correct, len(meta["leaf_names"]), strength)
    rz = standardise(e["root"], pp, root_moments)
    cz = standardise(e["child"], pl, child_moments)
    far = float(settings.get("root_far_limit", 0.05))
    oser = float(settings.get("child_oser_limit", 0.05))
    pg = np.asarray([r["status"] + ":" + (
        str(r["true_parent"]) if r["status"] == "known" else r["source"])
        for r in records])
    root_score, root_method, root_search = select_evidence(
        rz, ROOT_CANDIDATES, root_correct, status == "extra", source, pg,
        fit_mask, far)
    # Exclude extra from child fitting; include all intra, including wrong
    # parent routes. Calibrate child rejection independently of root rejection.
    child_score, child_method, child_search = select_evidence(
        cz, CHILD_CANDIDATES, child_correct, status == "intra", source,
        np.asarray([str(r["true_leaf"]) for r in records]),
        fit_mask & (status != "extra"), oser, pp, meta)
    result = {
        "schema_version": 4, "taxonomy": copy.deepcopy(meta),
        "root_features": ROOT_FEATURES, "child_features": CHILD_FEATURES,
        "root_moments": root_moments, "child_moments": child_moments,
        "root_method": root_method, "child_method": child_method,
        "selection_diagnostics": {"root": root_search, "child": child_search},
        "fit_count": int(fit_mask.sum()), "calibration_count": int(cal_mask.sum()),
        "risk_claim": "EMPIRICAL validation-source constraints only; no unseen-species guarantee",
        "profiles": {
            "risk": {
                "tau_root": risk_threshold(root_score, cal_mask & (status == "extra"), source, far),
                "tau_child": risk_threshold(child_score, cal_mask & (status == "intra"), source, oser),
                "description": "Risk-first: each observed calibration negative source <= limit",
            },
            "coverage": {
                "tau_root": coverage_threshold(root_score, cal_mask & root_correct,
                                                 float(settings.get("root_id_rejection", 0.10))),
                "tau_child": coverage_threshold(child_score, cal_mask & child_correct,
                                                  float(settings.get("child_known_rejection", 0.10))),
                "description": "Known-coverage-first quantiles, NOT a 5% OOD-risk claim",
            },
        },
        "risk_limits": {"root_far": far, "child_oser": oser},
    }
    for name, profile in result["profiles"].items():
        root_ok = root_score >= profile["tau_root"]
        leaf_ok = root_ok & (child_score >= profile["tau_child"])
        profile["calibration_source_far"] = _source_risks(root_ok, cal_mask & (status == "extra"), source)
        profile["calibration_source_oser"] = _source_risks(leaf_ok, cal_mask & (status == "intra"), source)
        profile["empirical_constraints_satisfied"] = bool(
            all(x["rate"] <= far + 1e-12 for x in profile["calibration_source_far"].values())
            and all(x["rate"] <= oser + 1e-12 for x in profile["calibration_source_oser"].values()))
    return result


def predict(e, records, calibration, profile):
    meta = calibration["taxonomy"]
    children = validate_meta(meta)
    pp, pl = e["pred_parent"], e["pred_leaf"]
    rz = standardise(e["root"], pp, calibration["root_moments"])
    cz = standardise(e["child"], pl, calibration["child_moments"])
    rs = weighted_score(rz, calibration["root_method"]["weights"])
    cs = weighted_score(cz, calibration["child_method"]["weights"], pp, meta)
    thresholds = calibration["profiles"][profile]
    output = []
    for i, raw in enumerate(records):
        g, leaf = int(pp[i]), int(pl[i])
        if leaf not in children[g]:
            raise ValueError("Leaf prediction outside its parent")
        if rs[i] < thresholds["tau_root"]:
            kind, final_p, final_l = "global_unknown", None, None
        elif cs[i] < thresholds["tau_child"]:
            kind, final_p, final_l = "intra_unknown", g, None
        else:
            kind, final_p, final_l = "known", g, leaf
        r = dict(raw)
        r.update({
            "candidate_parent": g, "candidate_leaf": leaf,
            "pred_parent": g, "pred_leaf": leaf,
            "global_pred_leaf": int(e["global_pred_leaf"][i]),
            "text_pred_parent": int(e["text_pred_parent"][i]),
            "support_neighbor_index": int(e["neighbor_index"][i]),
            "candidate_parent_name": meta["parent_names"][g],
            "candidate_leaf_name": meta["leaf_names"][leaf],
            "parent": final_p, "leaf": final_l, "prediction_type": kind,
            "parent_name": None if final_p is None else meta["parent_names"][final_p],
            "leaf_name": None if final_l is None else meta["leaf_names"][final_l],
            "true_parent_name": None if r["true_parent"] is None else meta["parent_names"][r["true_parent"]],
            "true_leaf_name": None if r["true_leaf"] is None else meta["leaf_names"][r["true_leaf"]],
            "root_knownness_score": float(rs[i]), "child_knownness_score": float(cs[i]),
            "root_gate_margin": float(rs[i] - thresholds["tau_root"]),
            "child_gate_margin": float(cs[i] - thresholds["tau_child"]),
            "root_evidence": dict(zip(ROOT_FEATURES, e["root"][i].tolist())),
            "child_evidence": dict(zip(CHILD_FEATURES, e["child"][i].tolist())),
            "profile": profile,
        })
        output.append(r)
    return output
