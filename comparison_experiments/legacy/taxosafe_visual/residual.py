"""TaxoSafe-RS: parent-residual, locally scaled support for child novelty.

Only the child decision changes. Encoder, parent routing and global rejection
are inherited from visual-support v4. Selection uses TRAIN cross-fit
leave-one-species-out episodes, thresholds use held validation, never test.
This is a research hypothesis, not a population-risk guarantee.
"""
import copy
import hashlib

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from . import core


def project(x, direction, alpha):
    """Do not renormalize: a nearly zero residual must not be amplified."""
    return x - float(alpha) * (x @ direction)[:, None] * direction[None, :]


def distances(a, b):
    squared = (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2 * a @ b.T
    return np.sqrt(np.maximum(squared, 0.0))


def build_state(features, labels, meta, alpha, k, shrinkage=10., local_scaling=True):
    """Fit metric statistics from the supplied support ONLY.

    The caller actually removes the held species before calling this function.
    Self neighbours are excluded from every reference radius.
    """
    core.validate_meta(meta)
    x = core.normalise(features)
    y = np.asarray(labels, dtype=np.int64)
    if y.shape != (len(x),) or np.any(y < 0) or np.any(y >= len(meta["leaf_names"])):
        raise ValueError("Invalid support labels")
    if not 0 <= float(alpha) <= 1 or int(k) < 1 or float(shrinkage) < 0:
        raise ValueError("Require alpha in [0,1], k>=1 and shrinkage>=0")
    mapping = np.asarray(meta["leaf_to_parent"])
    state = {"alpha": float(alpha), "k": int(k), "shrinkage": float(shrinkage),
             "local_scaling": bool(local_scaling), "parents": {}, "classes": {}}
    for parent in sorted(set(mapping[y].tolist())):
        leaves = [int(c) for c in np.unique(y) if mapping[c] == parent]
        means = core.normalise(np.asarray([x[y == c].mean(0) for c in leaves]))
        # Equal leaf weights: abundant species must not define the parent axis.
        direction = means.mean(0)
        norm = np.linalg.norm(direction)
        direction = direction / norm if norm > 1e-12 else np.zeros_like(direction)
        state["parents"][str(parent)] = {"direction": direction.tolist(), "leaves": leaves}
        residual = project(x, direction, alpha)
        reference, medians = {}, []
        for c in leaves:
            indices = np.flatnonzero(y == c)
            if len(indices) < 2:
                raise ValueError("Leaf {} needs at least two distinct support images".format(c))
            effective_k = min(int(k), len(indices) - 1)
            d = distances(residual[indices], residual[indices])
            np.fill_diagonal(d, np.inf)
            radius = np.partition(d, effective_k - 1, axis=1)[:, effective_k - 1]
            radius = np.maximum(radius, 1e-6)
            reference[c] = (indices, radius, effective_k)
            medians.append(float(np.median(radius)))
        # Class-balanced parent prior. Shrink small classes more strongly.
        prior = max(float(np.median(medians)), 1e-6)
        for c in leaves:
            indices, radius, effective_k = reference[c]
            weight = (len(indices) - 1) / (len(indices) - 1 + float(shrinkage))
            radius = np.exp(weight * np.log(radius) + (1 - weight) * np.log(prior))
            state["classes"][str(c)] = {"indices": indices.tolist(), "radii": radius.tolist(),
                                         "effective_k": effective_k}
    return state


def score(features, bank, state, parent_ids, chunk_size=128):
    """Higher score = compatible with a supported leaf; no labels are read."""
    query, reference = core.normalise(features), core.normalise(bank["features"])
    parents = np.asarray(parent_ids, dtype=np.int64)
    if parents.shape != (len(query),) or int(chunk_size) < 1:
        raise ValueError("Invalid parent routing or chunk size")
    scores = np.empty(len(query), dtype=np.float64)
    leaves = np.empty(len(query), dtype=np.int64)
    neighbours = np.empty(len(query), dtype=np.int64)
    for parent in np.unique(parents):
        if str(parent) not in state["parents"]:
            raise ValueError("Parent has no support: {}".format(parent))
        branch = state["parents"][str(parent)]
        direction = np.asarray(branch["direction"])
        ref = project(reference, direction, state["alpha"])
        ids = np.flatnonzero(parents == parent)
        for start in range(0, len(ids), int(chunk_size)):
            idx = ids[start:start + int(chunk_size)]
            q = project(query[idx], direction, state["alpha"])
            candidate_scores, candidate_neighbours = [], []
            for c in branch["leaves"]:
                cls = state["classes"][str(c)]
                support_ids = np.asarray(cls["indices"], dtype=np.int64)
                d = distances(q, ref[support_ids])
                effective_k = cls["effective_k"]
                nearest = np.argsort(d, axis=1, kind="stable")[:, :effective_k]
                boundary = np.take_along_axis(d, nearest, axis=1)[:, -1]
                local_radius = np.median(np.asarray(cls["radii"])[nearest], axis=1)
                if not state["local_scaling"]:
                    local_radius = np.ones_like(boundary)
                candidate_scores.append(np.log(np.maximum(local_radius, 1e-6)) - np.log(np.maximum(boundary, 1e-6)))
                candidate_neighbours.append(support_ids[nearest[:, 0]])
            values = np.stack(candidate_scores, axis=1)
            best = values.argmax(1)
            scores[idx] = values[np.arange(len(idx)), best]
            leaves[idx] = np.asarray(branch["leaves"])[best]
            neighbours[idx] = np.stack(candidate_neighbours, axis=1)[np.arange(len(idx)), best]
    return {"score": scores, "leaf": leaves, "neighbour": neighbours, "parent": parents.copy()}


def training_folds(bank, folds=2, seed=41):
    """Deterministic class-stratified image-disjoint support/query folds."""
    labels = np.asarray(bank["labels"])
    hashes = bank.get("image_hashes")
    if hashes is None or len(hashes) != len(labels) or len(set(map(str, hashes))) != len(labels):
        raise ValueError("Unique image content hashes required for cross-fitting")
    if int(folds) < 2:
        raise ValueError("At least two cross-fitting folds required")
    assignments = np.empty(len(labels), dtype=np.int64)
    for c in np.unique(labels):
        indices = np.flatnonzero(labels == c).tolist()
        if len(indices) < 4:
            raise ValueError("Leaf {} needs >=4 images for cross-fitting; do not duplicate rare samples".format(c))
        indices.sort(key=lambda i: hashlib.sha256((str(seed) + str(hashes[i])).encode()).hexdigest())
        for rank, i in enumerate(indices):
            assignments[i] = rank % int(folds)
    return assignments


def fit(bank, meta, settings):
    """Select a metric by TRAIN pseudo-novel episodes, not validation OOD.

    Encoder pretraining may have seen these known species: episodes simulate
    support removal, they are NOT evidence of truly unseen-species performance.
    """
    alphas = sorted(set(float(x) for x in settings.get("alphas", [0., .5, 1.])))
    ks = sorted(set(int(x) for x in settings.get("ks", [1, 3])))
    if not alphas or not ks or any(not 0 <= a <= 1 for a in alphas) or min(ks) < 1:
        raise ValueError("Invalid residual selection grid")
    folds = int(settings.get("folds", 2))
    assignments = training_folds(bank, folds, int(settings.get("seed", 41)))
    x, y = core.normalise(bank["features"]), np.asarray(bank["labels"])
    mapping = np.asarray(meta["leaf_to_parent"])
    shrinkage = float(settings.get("radius_shrinkage", 10.))
    scaling = bool(settings.get("local_scaling", True))
    reports = []
    for alpha in alphas:
        for k in ks:
            by_parent, acc_by_parent = {}, {}
            episode_count = 0
            for fold in range(folds):
                support, query = np.flatnonzero(assignments != fold), np.flatnonzero(assignments == fold)
                for parent in range(len(meta["parent_names"])):
                    children = np.flatnonzero(mapping == parent)
                    if len(children) < 2:
                        continue  # Singleton has no sibling-novel task; use shared metric.
                    for held in children:
                        s = support[(mapping[y[support]] == parent) & (y[support] != held)]
                        q = query[mapping[y[query]] == parent]
                        local_bank = {"features": x[s], "labels": y[s]}
                        state = build_state(x[s], y[s], meta, alpha, k, shrinkage, scaling)
                        if str(int(held)) in state["classes"]:
                            raise AssertionError("Held species leaked into metric statistics")
                        out = score(x[q], local_bank, state, np.full(len(q), parent))
                        negative = out["score"][y[q] == held]
                        class_aucs, class_acc = [], []
                        for c in children[children != held]:
                            mask = y[q] == c
                            positive = out["score"][mask]
                            if not len(positive) or not len(negative):
                                continue
                            class_aucs.append(float(roc_auc_score([1] * len(positive) + [0] * len(negative), np.r_[positive, negative])))
                            class_acc.append(float(np.mean(out["leaf"][mask] == c)))
                        if class_aucs:
                            by_parent.setdefault(int(parent), []).append(float(np.mean(class_aucs)))
                            acc_by_parent.setdefault(int(parent), []).append(float(np.mean(class_acc)))
                            episode_count += 1
            if not by_parent:
                raise ValueError("No multi-leaf training branch supports leave-one-species-out selection")
            auc = float(np.mean([np.mean(v) for v in by_parent.values()]))
            acc = float(np.mean([np.mean(v) for v in acc_by_parent.values()]))
            reports.append({"alpha": alpha, "k": k, "macro_episode_auroc": auc,
                            "macro_episode_leaf_accuracy": acc, "objective": .75 * auc + .25 * acc,
                            "episode_count": episode_count, "per_parent_auroc": {str(g): float(np.mean(v)) for g, v in by_parent.items()}})
    chosen = max(reports, key=lambda r: (r["objective"], -r["alpha"], -r["k"]))
    state = build_state(x, y, meta, chosen["alpha"], chosen["k"], shrinkage, scaling)
    state["selection"] = {"source": "train_only_crossfit_leave_one_species_out", "folds": folds,
                          "objective": "0.75 * parent/species-macro AUROC + 0.25 * macro known leaf accuracy",
                          "chosen": chosen, "candidates": reports,
                          "encoder_seen_pseudo_species": True}
    return state


def source_balanced_threshold(scores, positive, leaf_ids, negative, sources):
    """Choose a parent threshold with equal leaf/source influence.

    The positive term is the macro acceptance rate over correctly predicted
    known leaves. The negative term is the macro rejection rate over routed
    intra-unknown sources. This is an empirical validation operating point,
    not a conformal or population-risk guarantee. Ties favour known coverage.
    """
    values = np.asarray(scores, dtype=np.float64)
    positive = np.asarray(positive, dtype=bool)
    negative = np.asarray(negative, dtype=bool)
    leaves = np.asarray(leaf_ids, dtype=np.int64)
    groups = np.asarray(sources)
    if not positive.any() or not negative.any():
        raise ValueError("Balanced threshold needs known leaves and intra-unknown sources")
    leaf_groups = sorted(set(leaves[positive].tolist()))
    source_groups = sorted(set(groups[negative].tolist()))
    if not leaf_groups or not source_groups:
        raise ValueError("Balanced threshold has no calibration groups")
    candidates = sorted(set(values[positive | negative].tolist()))
    candidates.append(float(np.nextafter(max(candidates), np.inf)))
    best = None
    for tau in candidates:
        known_macro = float(np.mean([
            np.mean(values[positive & (leaves == leaf)] >= tau)
            for leaf in leaf_groups
        ]))
        unknown_macro = float(np.mean([
            np.mean(values[negative & (groups == source)] < tau)
            for source in source_groups
        ]))
        utility = .5 * (known_macro + unknown_macro)
        # Deterministic lexicographic tie break: preserve known coverage, then
        # choose the least restrictive threshold.
        candidate = (utility, known_macro, -float(tau), float(tau), unknown_macro)
        if best is None or candidate[:3] > best[:3]:
            best = candidate
    return float(best[3]), {
        "objective": "0.5 * macro_known_leaf_acceptance + 0.5 * macro_intra_source_rejection",
        "utility": float(best[0]),
        "macro_known_leaf_acceptance": float(best[1]),
        "macro_intra_source_rejection": float(best[4]),
        "known_leaf_groups": len(leaf_groups),
        "intra_source_groups": len(source_groups),
        "known_count": int(positive.sum()),
        "intra_count": int(negative.sum()),
        "tie_break": "higher known acceptance, then lower threshold",
    }


def calibrate(out, rows, meta, indices, settings):
    """Parent-conditional thresholds on the baseline's held calibration half.

    Coverage protects each observed true known leaf, conditional on correct
    parent routing. Risk limits each observed intra source. Missing branch
    evidence is explicitly marked and never called a safety guarantee.
    """
    indices = np.asarray(indices, dtype=np.int64)
    known = np.asarray([r["status"] == "known" for r in rows])
    intra = np.asarray([r["status"] == "intra" for r in rows])
    correct_parent = np.asarray([r["true_parent"] == int(p) for r, p in zip(rows, out["parent"])])
    sources = np.asarray([r["source"] for r in rows])
    selected = np.zeros(len(rows), bool); selected[indices] = True
    scores = out["score"]
    residual_settings = settings.get("residual", {})
    threshold_mode = str(residual_settings.get("threshold_mode", "branch_min"))
    valid_modes = {"branch_min", "parent_pooled", "leaf_conditional", "hierarchical_shrinkage"}
    if threshold_mode not in valid_modes:
        raise ValueError("Unknown residual threshold_mode: {}".format(threshold_mode))
    threshold_strength = residual_settings.get("threshold_shrinkage", "auto")
    balanced_profile = bool(residual_settings.get("balanced_profile", False))
    if threshold_strength != "auto":
        threshold_strength = float(threshold_strength)
        if threshold_strength < 0:
            raise ValueError("threshold_shrinkage must be non-negative or 'auto'")
    risk_limit = float(settings.get("child_oser_limit", .05))
    reject = float(settings.get("child_known_rejection", .10))
    if not 0 <= reject < 1 or not 0 <= risk_limit < 1:
        raise ValueError("Invalid coverage or risk setting")
    pos = selected & known & correct_parent
    neg = selected & intra
    if not pos.any() or not neg.any():
        raise ValueError("Need held validation known positives and intra negatives")
    true_leaf = np.asarray([r["true_leaf"] if r["true_leaf"] is not None else -1 for r in rows])
    if threshold_mode == "branch_min":
        correct_leaf = np.ones(len(rows), dtype=bool)
    else:
        if "leaf" not in out:
            raise ValueError("Leaf-aware calibration requires candidate leaf predictions")
        predicted_leaf = np.asarray(out["leaf"], dtype=np.int64)
        correct_leaf = predicted_leaf == true_leaf
    # The legacy branch-min decision rule is retained by default. New
    # candidate-leaf rules calibrate only decisions that can be correct: a
    # misrouted leaf cannot be repaired by lowering an acceptance threshold.
    calibrated_pos = pos if threshold_mode == "branch_min" else pos & correct_leaf
    if not calibrated_pos.any():
        raise ValueError("Need correctly routed validation positives for leaf-aware calibration")
    pooled_coverage = core.coverage_threshold(scores, calibrated_pos, reject)
    pooled_risk = core.risk_threshold(scores, neg, sources, risk_limit)
    leaf_counts = []
    if threshold_mode == "hierarchical_shrinkage":
        for c in range(len(meta["leaf_names"])):
            count = int(np.sum(calibrated_pos & (true_leaf == c)))
            if count:
                leaf_counts.append(count)
        if not leaf_counts:
            raise ValueError("No leaf supports hierarchical threshold shrinkage")
        auto_strength = float(np.median(leaf_counts))
        shrinkage = auto_strength if threshold_strength == "auto" else float(threshold_strength)
    else:
        shrinkage = None
    method_name = ("parent_residual_local_support" if threshold_mode == "branch_min" else
                   "candidate_leaf_{}_support".format(threshold_mode))
    profile_names = ["coverage", "risk"] + (["balanced"] if balanced_profile else [])
    result = {"profiles": {name: {} for name in profile_names}, "branches": {},
              "method_name": method_name,
              "threshold_mode": threshold_mode,
              "threshold_shrinkage": shrinkage,
              "threshold_shrinkage_rule": ("median_nonempty_correct_leaf_count" if
                                             threshold_mode == "hierarchical_shrinkage" and
                                             threshold_strength == "auto" else "fixed_or_not_applicable"),
              "no_population_guarantee": True}
    for p, name in enumerate(meta["parent_names"]):
        branch = out["parent"] == p
        negative = neg & branch
        balanced_negative = negative & correct_parent
        boundaries, missing, counts, correct_counts = [], [], {}, {}
        parent_pos = calibrated_pos & branch
        parent_tau = (core.coverage_threshold(scores, parent_pos, reject)
                      if parent_pos.any() else pooled_coverage)
        if threshold_mode == "hierarchical_shrinkage":
            n_parent = int(parent_pos.sum())
            parent_weight = n_parent / (n_parent + shrinkage) if shrinkage > 0 else 1.0
            backoff_tau = parent_weight * parent_tau + (1.0 - parent_weight) * pooled_coverage
        else:
            backoff_tau = parent_tau
        by_leaf = {}
        for c, parent in enumerate(meta["leaf_to_parent"]):
            if parent != p:
                continue
            mask = pos & np.asarray([r["true_leaf"] == c for r in rows])
            counts[meta["leaf_names"][c]] = int(mask.sum())
            correct_mask = calibrated_pos & (true_leaf == c)
            correct_counts[meta["leaf_names"][c]] = int(correct_mask.sum())
            if mask.any():
                boundaries.append(core.coverage_threshold(scores, mask, reject))
            else:
                missing.append(meta["leaf_names"][c])
            if threshold_mode != "branch_min":
                leaf_tau = (core.coverage_threshold(scores, correct_mask, reject)
                            if correct_mask.any() else backoff_tau)
                if threshold_mode == "parent_pooled":
                    tau = parent_tau
                elif threshold_mode == "leaf_conditional":
                    tau = leaf_tau
                else:
                    n_leaf = int(correct_mask.sum())
                    leaf_weight = n_leaf / (n_leaf + shrinkage) if shrinkage > 0 else 1.0
                    tau = leaf_weight * leaf_tau + (1.0 - leaf_weight) * backoff_tau
                by_leaf[str(c)] = float(tau)
        if threshold_mode == "branch_min":
            coverage_tau = min(boundaries) if boundaries else pooled_coverage
            coverage_value = float(coverage_tau)
            feasibility_boundary = coverage_tau
        else:
            coverage_value = {"default": float(backoff_tau), "by_leaf": by_leaf}
            feasibility_boundary = min(by_leaf.values()) if by_leaf else backoff_tau
        risk_tau = core.risk_threshold(scores, negative, sources, risk_limit) if negative.any() else pooled_risk
        result["profiles"]["coverage"][str(p)] = coverage_value
        result["profiles"]["risk"][str(p)] = float(risk_tau)
        balanced_report = None
        if balanced_profile:
            if parent_pos.any() and balanced_negative.any():
                balanced_tau, balanced_report = source_balanced_threshold(
                    scores, parent_pos, true_leaf, balanced_negative, sources)
                balanced_source = "parent_leaf_and_source_macro_utility"
            else:
                balanced_tau = backoff_tau
                balanced_source = "coverage_backoff_no_routed_intra_source"
            result["profiles"]["balanced"][str(p)] = float(balanced_tau)
        result["branches"][name] = {"known_counts_correct_parent": counts,
                                     "known_counts_correct_leaf": correct_counts,
                                     "missing_known_leaves": missing,
                                     "intra_count_routed_here": int(negative.sum()),
                                     "intra_count_correctly_routed_here": int(balanced_negative.sum()),
                                     "risk_threshold_source": "branch_sources" if negative.any() else "pooled_no_branch_negatives",
                                     "balanced_threshold_source": balanced_source if balanced_profile else None,
                                     "balanced_calibration": balanced_report,
                                     "coverage_and_risk_feasible_on_calibration": bool(risk_tau <= feasibility_boundary) if boundaries and negative.any() else None}
    return result


def threshold(calibration, profile, parent, leaf):
    """Resolve a legacy parent scalar or a candidate-leaf threshold."""
    value = calibration["profiles"][profile][str(parent)]
    if isinstance(value, dict):
        by_leaf = value.get("by_leaf", {})
        value = by_leaf.get(str(leaf), value.get("default"))
        if value is None:
            raise ValueError("Missing threshold for parent {} leaf {}".format(parent, leaf))
    return float(value)


def apply(baseline, out, calibration, meta, profile, method="parent_residual_local_support"):
    """Override ONLY child decisions; root scores/decisions are bit-identical."""
    if len(baseline) != len(out["score"]):
        raise ValueError("Prediction/evidence length mismatch")
    result = copy.deepcopy(baseline)
    for i, record in enumerate(result):
        parent, leaf = int(out["parent"][i]), int(out["leaf"][i])
        if parent != record["candidate_parent"] or meta["leaf_to_parent"][leaf] != parent:
            raise ValueError("Residual method must preserve parent routing")
        tau = threshold(calibration, profile, parent, leaf)
        record["baseline_candidate_leaf"] = record["candidate_leaf"]
        record["baseline_child_knownness_score"] = record["child_knownness_score"]
        record["candidate_leaf"] = leaf
        record["pred_leaf"] = leaf
        record["candidate_leaf_name"] = meta["leaf_names"][leaf]
        record["child_knownness_score"] = float(out["score"][i])
        record["residual_child_threshold"] = float(tau)
        record["child_gate_margin"] = float(out["score"][i] - tau)
        record["baseline_child_evidence"] = record["child_evidence"]
        record["child_evidence"] = {method: float(out["score"][i])}
        record["child_method"] = method
        record["profile"] = profile
        record["support_neighbor_index"] = int(out["neighbour"][i])
        if record["prediction_type"] == "global_unknown":
            continue
        accept = out["score"][i] >= tau
        record["prediction_type"] = "known" if accept else "intra_unknown"
        record["leaf"] = leaf if accept else None
        record["leaf_name"] = meta["leaf_names"][leaf] if accept else None
        record["parent"] = parent
        record["parent_name"] = meta["parent_names"][parent]
    return result


def diagnostics(rows):
    """Fine detection, independent of a single operating threshold.

    AUROC uses known as positive (large child score). FPR95 is therefore the
    fraction of intra unknown accepted at >=95% known TPR. Always state this.
    """
    def measure(records):
        labels = np.asarray([r["status"] == "known" for r in records], dtype=int)
        values = np.asarray([r["child_knownness_score"] for r in records])
        if len(set(labels)) < 2:
            return {"known_count": int(labels.sum()), "intra_count": int(len(labels) - labels.sum()), "auroc": None, "fpr95_known": None}
        fpr, tpr, _ = roc_curve(labels, values)
        return {"known_count": int(labels.sum()), "intra_count": int(len(labels) - labels.sum()),
                "auroc": float(roc_auc_score(labels, values)), "fpr95_known": float(fpr[tpr >= .95].min())}
    selected = [r for r in rows if r["status"] in ("known", "intra")]
    correct_route = [r for r in selected if r["candidate_parent"] == r["true_parent"]]
    accepted = [r for r in correct_route if r["prediction_type"] != "global_unknown"]
    per_parent = {str(p): measure([r for r in selected if r["true_parent"] == p]) for p in sorted({r["true_parent"] for r in selected})}
    valid = [v["auroc"] for v in per_parent.values() if v["auroc"] is not None]
    source_results, balanced_parents = {}, {}
    for p in sorted({r["true_parent"] for r in selected}):
        known = [r for r in selected if r["true_parent"] == p and r["status"] == "known"]
        unknown = [r for r in selected if r["true_parent"] == p and r["status"] == "intra"]
        source_aucs = []
        for source in sorted({r["source"] for r in unknown}):
            negatives = [r for r in unknown if r["source"] == source]
            leaf_aucs = []
            for c in sorted({r["true_leaf"] for r in known}):
                metric = measure([r for r in known if r["true_leaf"] == c] + negatives)
                if metric["auroc"] is not None:
                    leaf_aucs.append(metric["auroc"])
            source_results[str(p) + "/" + source] = float(np.mean(leaf_aucs)) if leaf_aucs else None
            if leaf_aucs:
                source_aucs.append(float(np.mean(leaf_aucs)))
        if source_aucs:
            balanced_parents[str(p)] = float(np.mean(source_aucs))
    return {"positive_label": "known; higher child score", "all": measure(selected),
            "correct_parent_only": measure(correct_route), "root_accepted_correct_parent_only": measure(accepted),
            "per_true_parent": per_parent, "macro_parent_auroc": float(np.mean(valid)) if valid else None,
            "known_leaf_macro_auroc_per_novel_source": source_results,
            "macro_parent_species_auroc": float(np.mean(list(balanced_parents.values()))) if balanced_parents else None,
            "warning": "Conditional subsets are diagnostic, not end-to-end success or unseen-species risk guarantees."}
