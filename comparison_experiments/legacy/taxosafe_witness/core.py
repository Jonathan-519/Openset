"""Local witness evidence and a monotone verifier trained on known TRAIN only.

The verifier never updates features or class labels. Outer leave-species-out
folds remove the held species from ALL fitting queries and reference images.
The upstream frozen encoder may have seen that species: this is verifier-level
generalization, not an unseen-encoder experiment.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import roc_auc_score

FEATURES = ("nearest_log_distance", "multi_support_log_distance",
            "coherent_patch_cost", "unexplained_patch_tail",
            "collage_gap", "sibling_specificity_cost", "support_fragility")
VARIANTS = {"identity": [0], "global_only": [0, 1, 6],
            "no_coherence": [0, 1, 2, 3, 5, 6],
            "no_sibling": [0, 1, 2, 3, 4, 6], "full": list(range(7))}


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    if not np.isfinite(x).all() or (norm < 1e-10).any():
        raise ValueError("Features must be finite nonzero vectors")
    return x / norm


def prepare(data, settings):
    g, patches = unit(data["global"]), unit(data["patches"])
    if patches.ndim != 3 or patches.shape[0] != len(g) or patches.shape[-1] != g.shape[-1]:
        raise ValueError("Incompatible global/patch arrays")
    rank = min(int(settings["projection_dim"]), g.shape[-1])
    if rank < 1:
        raise ValueError("Positive projection dimension required")
    rng = np.random.RandomState(int(settings["projection_seed"]))
    # Data-independent projection: no held species statistics enter the map.
    projection = rng.choice([-1., 1.], (g.shape[-1], rank)).astype(np.float32) / np.sqrt(rank)
    projected = unit(patches @ projection)
    # Within-image contrast weights, not anatomical segmentation or saliency GT.
    center = patches.mean(axis=1, keepdims=True)
    contrast = np.linalg.norm(patches - center, axis=-1) + .05
    weights = contrast / contrast.sum(axis=1, keepdims=True)
    return {"global": g, "patches": projected, "weights": weights}


def subset(data, ids):
    return {k: v[ids] for k, v in data.items()}


def evidence(query, bank, labels, mapping, parents, settings, candidates=None):
    """Prediction API has no query truth. References contain known labels only.

    Exact same reference sample supports all features of one witness. Pooling
    independently best patches across images can hide inconsistent morphology;
    collage_gap quantifies the difference from a coherent single-image witness.
    """
    labels, mapping = np.asarray(labels, int), np.asarray(mapping, int)
    parents = np.asarray(parents, int)
    if len(labels) != len(bank["global"]) or parents.shape != (len(query["global"]),):
        raise ValueError("Evidence lengths differ")
    if (labels < 0).any() or (labels >= len(mapping)).any():
        raise ValueError("Invalid reference labels")
    n = len(parents)
    x, valid = np.zeros((n, 7), np.float64), np.ones((n, 7), bool)
    leaves, neighbours = np.empty(n, int), np.empty(n, int)
    k = int(settings["witnesses"])
    if k < 2:
        raise ValueError("At least two witnesses requested for coherent/collage comparison")
    def patch_cost(qi, ids):
        # [reference, query patch, reference patch], modest k*49*49 memory.
        cost = np.clip(1 - np.einsum('jd,rkd->rjk', query["patches"][qi], bank["patches"][ids]), 0., 2.)
        nearest = cost.min(axis=2)
        costs = .5 * (nearest @ query["weights"][qi]
                      + (cost.min(axis=1) * bank["weights"][ids]).sum(axis=1))
        return nearest, costs
    for i, parent in enumerate(parents):
        available = np.flatnonzero(mapping[labels] == parent)
        if not len(available):
            raise ValueError("Predicted parent has no active support")
        d = np.maximum(0., 2 - 2 * (bank["global"][available] @ query["global"][i]))
        best = available[np.argmin(d)]
        leaf = labels[best] if candidates is None else int(candidates[i])
        own = available[labels[available] == leaf]
        if not len(own):
            raise ValueError("Candidate must have support inside predicted parent")
        od = np.maximum(0., 2 - 2 * (bank["global"][own] @ query["global"][i]))
        order = np.argsort(od, kind="stable")
        chosen, distances = own[order[:k]], od[order[:k]]
        nearest, costs = patch_cost(i, chosen)
        coherent = int(np.argmin(costs))
        pooled = nearest.min(axis=0) @ query["weights"][i]
        tail = np.quantile(nearest[coherent], .9)
        others = available[labels[available] != leaf]
        rival = 0.
        if len(others):
            rd = np.maximum(0., 2 - 2 * (bank["global"][others] @ query["global"][i]))
            _, rc = patch_cost(i, others[np.argsort(rd, kind="stable")[:k]])
            rival = costs[coherent] - rc.min()
        else:
            valid[i, 5] = False
        if len(chosen) < 2:
            valid[i, [4, 6]] = False
        x[i] = [0.5 * np.log(max(float(distances[0]), 1e-12)),
                0.5 * np.log(max(float(distances.mean()), 1e-12)),
                costs[coherent], tail, max(0., nearest[coherent] @ query["weights"][i] - pooled), rival,
                float(distances[-1] - distances[0])]
        leaves[i], neighbours[i] = leaf, chosen[0]
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite witness evidence")
    return {"x": x, "valid": valid, "leaf": leaves, "parent": parents.copy(), "neighbour": neighbours}


def split_train(rows, settings, seed):
    if any(r["status"] != "known" or r["split"] != "train" for r in rows):
        raise ValueError("Verifier accepts only known TRAIN images")
    hashes = [r["image_sha256"] for r in rows]
    if len(hashes) != len(set(hashes)):
        raise ValueError("Duplicate training image contents")
    y = np.asarray([r["true_leaf"] for r in rows], int)
    rng = np.random.RandomState(seed)
    support, fit, audit = [], [], []
    for leaf in sorted(set(y.tolist())):
        ids = np.asarray(sorted(np.flatnonzero(y == leaf), key=lambda i: hashes[i]))
        ids = rng.permutation(ids)
        if len(ids) < 4:
            raise ValueError("Every training leaf needs >=4 distinct images for support/fit/audit")
        ns = min(int(settings["support_per_leaf"]), max(2, len(ids) // 2))
        queries = ids[ns:ns + int(settings["query_per_leaf"])]
        if len(queries) < 2:
            raise ValueError("Insufficient disjoint query images")
        support.extend(ids[:ns]); fit.extend(queries[::2]); audit.extend(queries[1::2])
    return tuple(np.asarray(v, int) for v in (support, fit, audit))


def training_world(data, y, mapping, support, query_ids, settings, outer_leaf=None):
    """No held species appears as a fitting query OR as any positive/negative reference."""
    mapping = np.asarray(mapping, int)
    support = support[y[support] != outer_leaf] if outer_leaf is not None else support
    query_ids = query_ids[y[query_ids] != outer_leaf] if outer_leaf is not None else query_ids
    if set(support.tolist()) & set(query_ids.tolist()):
        raise ValueError("Reference/query overlap")
    xs, masks, targets, groups, query_log = [], [], [], [], []
    def add(qids, refs, target):
        e = evidence(subset(data, qids), subset(data, refs), y[refs], mapping,
                     mapping[y[qids]], settings)
        xs.append(e["x"]); masks.append(e["valid"])
        targets.extend([target] * len(qids))
        groups.extend(["{}:{}".format(target, y[i]) for i in qids])
        query_log.extend(qids.tolist())
    add(query_ids, support, 0)
    for held in sorted(set(y[query_ids].tolist())):
        refs = support[y[support] != held]
        if not np.any(mapping[y[refs]] == mapping[held]):
            continue
        add(query_ids[y[query_ids] == held], refs, 1)
    if len(set(targets)) != 2:
        raise ValueError("Need multiple non-singleton leaves for pseudo-unknown fitting")
    return {"x": np.concatenate(xs), "valid": np.concatenate(masks),
            "y": np.asarray(targets, float), "groups": np.asarray(groups),
            "fit_query_ids": sorted(set(query_log)), "reference_ids": support.tolist()}


def fit_verifier(world, columns, regularizer):
    x, mask, y = world["x"][:, columns], world["valid"][:, columns], world["y"]
    mean = np.asarray([x[mask[:, j], j].mean() if mask[:, j].any() else 0. for j in range(len(columns))])
    scale = np.asarray([max(float(x[mask[:, j], j].std()), 1e-4) if mask[:, j].any() else 1. for j in range(len(columns))])
    z = np.where(mask, np.clip((x - mean) / scale, -8., 8.), 0.)
    groups = world["groups"]
    weights = np.asarray([1. / np.sum(groups == g) for g in groups])
    # Each pseudo/known species is balanced, and total known/unknown mass is equal.
    for target in (0, 1):
        weights[y == target] /= 2 * weights[y == target].sum()
    def objective(theta):
        logits = z @ theta[:-1] + theta[-1]
        loss = np.sum(weights * (np.logaddexp(0., logits) - y * logits))
        loss += .5 * regularizer * np.sum(theta[:-1] ** 2)
        r = weights * (expit(logits) - y)
        grad = np.r_[z.T @ r + regularizer * theta[:-1], r.sum()]
        return loss, grad
    result = minimize(objective, np.zeros(len(columns) + 1), jac=True, method="L-BFGS-B",
                      bounds=[(0., None)] * len(columns) + [(None, None)], options={"maxiter": 500, "ftol": 1e-10})
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError("Verifier optimizer did not converge: " + str(result.message))
    return {"columns": list(columns), "mean": mean.tolist(), "scale": scale.tolist(),
            "weights": result.x[:-1].tolist(), "bias": float(result.x[-1]),
            "regularizer": float(regularizer), "optimization_success": True}


def score(model, e):
    if model.get("identity"):
        return -e["x"][:, 0]
    cols = model["columns"]
    z = np.where(e["valid"][:, cols], np.clip((e["x"][:, cols] - model["mean"]) / model["scale"], -8., 8.), 0.)
    return -(z @ np.asarray(model["weights"]) + model["bias"])


def fit_all(data, rows, meta, settings, seed):
    support, fit, audit = split_train(rows, settings, seed)
    y = np.asarray([r["true_leaf"] for r in rows], int)
    mapping = np.asarray(meta["leaf_to_parent"], int)
    if set(y.tolist()) != set(range(len(mapping))):
        raise ValueError("Missing training leaf")
    prepared = prepare(data, settings)
    cv = {v: {str(reg): [] for reg in settings["regularizers"]} for v in VARIANTS if v != "identity"}
    provenance = []
    for held in sorted(set(y.tolist())):
        parent = mapping[held]
        if np.sum(mapping == parent) < 2:
            continue
        world = training_world(prepared, y, mapping, support, fit, settings, held)
        used = world["fit_query_ids"] + world["reference_ids"]
        if any(y[i] == held for i in used):
            raise AssertionError("Outer held species leaked into verifier fitting")
        refs = support[y[support] != held]
        # Positives are disjoint audit images of same-parent active species.
        # Negatives are audit images of the species absent from the whole fit.
        qids = audit[mapping[y[audit]] == parent]
        target = (y[qids] != held).astype(int)
        if len(set(target.tolist())) != 2:
            raise ValueError("Outer fold needs both audit known and held species")
        e = evidence(subset(prepared, qids), subset(prepared, refs), y[refs], mapping,
                     mapping[y[qids]], settings)
        for variant, columns in VARIANTS.items():
            if variant == "identity":
                continue
            for reg in settings["regularizers"]:
                model = fit_verifier(world, columns, reg)
                values = score(model, e)
                # Give each known sibling equal weight against this held species.
                aucs = []
                for leaf in sorted(set(y[qids][target == 1].tolist())):
                    keep = (y[qids] == leaf) | (target == 0)
                    aucs.append(roc_auc_score(target[keep], values[keep]))
                cv[variant][str(reg)].append({"held_leaf": int(held), "parent": int(parent),
                                              "macro_auc": float(np.mean(aucs))})
        provenance.append({"held_leaf": int(held), "fit_leaves": sorted(set(y[used].tolist())),
                           "fit_query_ids": world["fit_query_ids"], "reference_ids": world["reference_ids"],
                           "audit_query_ids": qids.tolist()})
        print("Verifier whole-species fold complete:", meta["leaf_names"][held], flush=True)
    if not provenance:
        raise ValueError("No evaluable whole-species folds")
    final = training_world(prepared, y, mapping, support, np.r_[fit, audit], settings)
    models, selection = {"identity": {"identity": True}}, {}
    for variant in cv:
        ranked = []
        for reg in settings["regularizers"]:
            r = cv[variant][str(reg)]
            ps = [np.mean([x["macro_auc"] for x in r if x["parent"] == p]) for p in sorted({x["parent"] for x in r})]
            ranked.append((min(ps), float(np.mean(ps)), float(reg)))
        chosen = max(ranked)[2]  # Worst-parent, then macro, then stronger regularization.
        models[variant] = fit_verifier(final, VARIANTS[variant], chosen)
        selection[variant] = {"regularizer": chosen, "candidates": ranked}
    report = {"source": "known TRAIN only", "validation_loaded": False, "test_loaded": False,
              "classifier_updated": False, "encoder_updated": False,
              "encoder_has_seen_outer_species": True, "selection_is_nested_unbiased_estimate": False,
              "selection_rule": "maximize worst-parent held-species AUROC, then macro, then regularization",
              "split": {"support": support.tolist(), "fit": fit.tolist(), "audit": audit.tolist()},
              "outer_folds": provenance, "cross_validation": cv, "selected": selection,
              "final_fit_query_ids": final["fit_query_ids"],
              "limitations": "Verifier-level selection folds, not an unbiased performance estimate or unseen-encoder test"}
    return models, report
