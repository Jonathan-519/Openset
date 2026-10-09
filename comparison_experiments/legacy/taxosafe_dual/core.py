"""TRAIN-only dual semantic/morphology evidence and conformal calibration.

Every statistic uses the same image-disjoint reference bank. A removed leaf is
absent from retrieval, prototypes, patches and candidates in pseudo-novel episodes.
"""
import numpy as np
from sklearn.metrics import roc_auc_score
from taxosafe_witness import core as witness

MORPHOLOGY_FEATURES = witness.FEATURES
SEMANTIC_FEATURES = ("candidate_text_cost", "sibling_margin_cost",
                     "candidate_knn_cost", "candidate_prototype_cost")
FEATURES = MORPHOLOGY_FEATURES + SEMANTIC_FEATURES
VARIANTS = {
    "identity": [0], "morphology_only": list(range(7)),
    "semantic_only": list(range(7, 11)),
    "no_coherence": [0, 1, 4, 5, 6, 7, 8, 9, 10],
    "full": list(range(11)),
}

def prepare(data, settings):
    if any(k not in data for k in ("global", "patches", "leaf_cosine")):
        raise ValueError("Dual evidence requires global, patches and leaf_cosine caches")
    out = witness.prepare(data, settings)
    out["leaf_cosine"] = np.asarray(data["leaf_cosine"], dtype=np.float64)
    if out["leaf_cosine"].ndim != 2 or len(out["leaf_cosine"]) != len(out["global"]):
        raise ValueError("Invalid leaf cosine cache")
    return out

def subset(data, ids):
    ids = np.asarray(ids, dtype=np.int64)
    return {k: v[ids] for k, v in data.items()}

def _semantic_costs(query, bank, labels, mapping, parents, leaves, settings):
    """Candidate-conditioned costs; larger values mean less known-like."""
    qg, bg = witness.unit(query["global"]), witness.unit(bank["global"])
    labels, mapping = np.asarray(labels, int), np.asarray(mapping, int)
    parents, leaves = np.asarray(parents, int), np.asarray(leaves, int)
    logits = np.asarray(query["leaf_cosine"], float)
    if len(bg) != len(labels) or parents.shape != leaves.shape or parents.shape != (len(qg),):
        raise ValueError("Semantic evidence lengths differ")
    if logits.shape != (len(qg), len(mapping)):
        raise ValueError("Leaf cosine/taxonomy mismatch")
    k = int(settings.get("semantic_k", 3))
    if k < 1:
        raise ValueError("semantic_k must be positive")
    x = np.empty((len(qg), 4), dtype=np.float64)
    valid = np.ones((len(qg), 4), dtype=bool)
    for i, (parent, leaf) in enumerate(zip(parents, leaves)):
        active = np.unique(labels[mapping[labels] == parent])
        if leaf not in active:
            raise ValueError("Candidate is absent from its reference branch")
        own = np.flatnonzero(labels == leaf)
        sims = np.clip(bg[own] @ qg[i], -1., 1.)
        kth = np.sort(sims)[-min(k, len(sims))]
        prototype = witness.unit(bg[own].mean(0, keepdims=True))[0]
        rivals = active[active != leaf]
        if len(rivals):
            margin = logits[i, leaf] - logits[i, rivals].max()
        else:
            margin = 0.; valid[i, 1] = False
        x[i] = [-logits[i, leaf], -margin, -kth,
                -float(np.clip(qg[i] @ prototype, -1., 1.))]
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite semantic evidence")
    return x, valid

def evidence(query, bank, labels, mapping, parents, settings, candidates=None):
    local = witness.evidence(query, bank, labels, mapping, parents, settings,
                             candidates=candidates)
    semantic, semantic_valid = _semantic_costs(
        query, bank, labels, mapping, parents, local["leaf"], settings)
    return {"x": np.concatenate([local["x"], semantic], axis=1),
            "valid": np.concatenate([local["valid"], semantic_valid], axis=1),
            "leaf": local["leaf"], "parent": local["parent"],
            "neighbour": local["neighbour"]}

def training_world(data, y, mapping, support, query_ids, settings, outer_leaf=None):
    y, mapping = np.asarray(y, int), np.asarray(mapping, int)
    support, query_ids = np.asarray(support, int), np.asarray(query_ids, int)
    if outer_leaf is not None:
        support = support[y[support] != outer_leaf]
        query_ids = query_ids[y[query_ids] != outer_leaf]
    if set(support.tolist()) & set(query_ids.tolist()):
        raise ValueError("Reference/query overlap")
    if not len(support) or not len(query_ids):
        raise ValueError("Empty reference or fitting query split")
    xs, masks, targets, groups, query_log = [], [], [], [], []
    def append(qids, refs, target, candidates):
        qids, refs = np.asarray(qids, int), np.asarray(refs, int)
        if not len(qids): return
        e = evidence(subset(data, qids), subset(data, refs), y[refs], mapping,
                     mapping[y[qids]], settings, candidates=candidates)
        xs.append(e["x"]); masks.append(e["valid"])
        targets.extend([target] * len(qids))
        groups.extend(["{}:{}".format(target, y[i]) for i in qids])
        query_log.extend(qids.tolist())
    append(query_ids, support, 0, y[query_ids])
    for held in sorted(set(y[query_ids].tolist())):
        refs = support[y[support] != held]
        if np.any(mapping[y[refs]] == mapping[held]):
            append(query_ids[y[query_ids] == held], refs, 1, None)
    if not xs or len(set(targets)) != 2:
        raise ValueError("Need known and pseudo-unknown fitting evidence")
    return {"x": np.concatenate(xs), "valid": np.concatenate(masks),
            "y": np.asarray(targets, float), "groups": np.asarray(groups),
            "fit_query_ids": sorted(set(query_log)), "reference_ids": support.tolist()}

def _outer_audit(data, y, mapping, support, audit, held, settings):
    parent = mapping[held]; refs = support[y[support] != held]
    qids = audit[mapping[y[audit]] == parent]
    known, unknown = qids[y[qids] != held], qids[y[qids] == held]
    if not len(known) or not len(unknown):
        raise ValueError("Outer fold needs held and sibling calibration images")
    a = evidence(subset(data, known), subset(data, refs), y[refs], mapping,
                 mapping[y[known]], settings, candidates=y[known])
    b = evidence(subset(data, unknown), subset(data, refs), y[refs], mapping,
                 mapping[y[unknown]], settings)
    return {"x": np.concatenate([a["x"], b["x"]]),
            "valid": np.concatenate([a["valid"], b["valid"]]),
            "known_leaf": np.r_[y[known], np.full(len(unknown), -1)],
            "target": np.r_[np.ones(len(known), int), np.zeros(len(unknown), int)],
            "query_ids": np.r_[known, unknown]}

def build_calibration(models, data, rows, meta, support, calibration, settings):
    y = np.asarray([r["true_leaf"] for r in rows], int)
    mapping = np.asarray(meta["leaf_to_parent"], int)
    e = evidence(subset(data, calibration), subset(data, support), y[support],
                 mapping, mapping[y[calibration]], settings, candidates=y[calibration])
    variants = {}
    for variant, model in models.items():
        values = witness.score(model, e); parents = {}
        for parent in range(len(meta["parent_names"])):
            ids = np.flatnonzero(mapping[y[calibration]] == parent)
            if not len(ids): raise ValueError("Every parent needs TRAIN calibration images")
            parents[str(parent)] = {"count": int(len(ids)),
                                    "scores": sorted(float(values[i]) for i in ids)}
        variants[variant] = {"parents": parents}
    return {"schema": 1, "method": "parent_mondrian_split_conformal",
            "alpha": float(settings["conformal_alpha"]), "acceptance": "p_value > alpha",
            "unknown_labels_used": False, "validation_used": False, "test_used": False,
            "calibration_query_ids": calibration.tolist(),
            "calibration_image_hashes": [rows[i]["image_sha256"] for i in calibration],
            "variants": variants}

def pvalues(values, parents, state, variant):
    values, parents = np.asarray(values, float), np.asarray(parents, int)
    if values.shape != parents.shape or not np.isfinite(values).all():
        raise ValueError("Invalid score arrays")
    out = np.empty(len(values), float); groups = state["variants"][variant]["parents"]
    for i, (value, parent) in enumerate(zip(values, parents)):
        ref = np.asarray(groups[str(int(parent))]["scores"], float)
        if not len(ref) or not np.isfinite(ref).all(): raise ValueError("Missing conformal reference group")
        out[i] = (1. + np.searchsorted(ref, value, side="right")) / (len(ref) + 1.)
    return out

def fit_all(data, rows, meta, settings, seed):
    support, fit, calibration = witness.split_train(rows, settings, seed)
    y = np.asarray([r["true_leaf"] for r in rows], int)
    mapping = np.asarray(meta["leaf_to_parent"], int)
    if set(y.tolist()) != set(range(len(mapping))): raise ValueError("Missing training leaf")
    prepared = prepare(data, settings)
    world = training_world(prepared, y, mapping, support, fit, settings)
    regularizer = float(settings["regularizer"])
    models = {"identity": {"identity": True}}
    for variant, columns in VARIANTS.items():
        if variant != "identity": models[variant] = witness.fit_verifier(world, columns, regularizer)
    diagnostics = {variant: [] for variant in VARIANTS if variant != "identity"}; folds = []
    for held in sorted(set(y.tolist())):
        parent = mapping[held]
        if np.sum(mapping == parent) < 2: continue
        fold_world = training_world(prepared, y, mapping, support, fit, settings, held)
        audit = _outer_audit(prepared, y, mapping, support, calibration, held, settings)
        for variant, columns in VARIANTS.items():
            if variant == "identity": continue
            model = witness.fit_verifier(fold_world, columns, regularizer)
            values = witness.score(model, audit); aucs = []
            for leaf in sorted(set(audit["known_leaf"][audit["target"] == 1].tolist())):
                keep = (audit["known_leaf"] == leaf) | (audit["target"] == 0)
                aucs.append(roc_auc_score(audit["target"][keep], values[keep]))
            diagnostics[variant].append({"held_leaf": int(held), "parent": int(parent),
                                         "macro_auc": float(np.mean(aucs))})
        folds.append({"held_leaf": int(held), "parent": int(parent),
                      "audit_query_ids": audit["query_ids"].tolist(), "selection_used": False})
        print("Dual-evidence whole-species diagnostic:", meta["leaf_names"][held], flush=True)
    state = build_calibration(models, prepared, rows, meta, support, calibration, settings)
    report = {"source": "known TRAIN only", "validation_loaded": False, "test_loaded": False,
              "classifier_updated": False, "encoder_updated": False,
              "reference_bank_matches_validation_and_test": True,
              "model_selection": "predeclared regularizer; outer folds are diagnostic only",
              "regularizer": regularizer,
              "split": {"support": support.tolist(), "fit": fit.tolist(), "calibration": calibration.tolist()},
              "support_image_hashes": [rows[i]["image_sha256"] for i in support],
              "final_fit_query_ids": world["fit_query_ids"],
              "conformal_calibration_query_ids": calibration.tolist(),
              "outer_folds": folds, "outer_fold_diagnostics": diagnostics,
              "feature_names": list(FEATURES),
              "guarantee_scope": "parent-conditional correctly-routed known queries under exchangeability; classifier/root errors and OOD rejection have no distribution-free guarantee"}
    if set(report["final_fit_query_ids"]) & set(calibration.tolist()):
        raise AssertionError("Conformal calibration leaked into verifier fitting")
    return models, state, report
