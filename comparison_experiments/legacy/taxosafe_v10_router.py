"""Learned dual-boundary router for TaxoSafe-v10.

The root gate learns in-taxonomy (known + near unknown) versus global OOD.
The local gate learns known leaf versus unseen sibling. Both fuse semantic
CLIP evidence with TRAIN-only prototype/ViM evidence. Thresholds are selected
only on held-out development partitions by macro deepest-reliable accuracy.
"""

from collections import defaultdict
import numpy as np

from taxolocal_v2_router import (
    add_reference_scores,
    fit_fusion,
    fit_reference_bank,
    fusion_scores,
)

ROOT_FEATURES = (
    "parent_margin",
    "parent_neg_entropy",
    "parent_logsumexp",
    "vim_residual",
    "prototype_distance",
)
LOCAL_FEATURES = (
    "local_known_margin",
    "child_margin",
    "child_neg_entropy",
    "prototype_distance",
)


def _quantiles(values, points):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        raise ValueError("Cannot calibrate an empty score array")
    return np.unique(np.quantile(values, np.linspace(0.0, 1.0, int(points))))


def _macro_dra(records, root_threshold, local_thresholds):
    known_ok, intra_ok, extra_ok = [], [], []
    for row in records:
        root_accept = float(row["global_knownness_score"]) >= root_threshold
        parent_id = int(row["pred_parent"])
        local_threshold = float(local_thresholds.get(parent_id, local_thresholds[-1]))
        local_accept = float(row["local_knownness_score"]) >= local_threshold
        if row["status"] == "known":
            known_ok.append(bool(
                root_accept and local_accept
                and int(row["pred_leaf"]) == int(row["true_leaf"])
            ))
        elif row["status"] == "intra":
            intra_ok.append(bool(
                root_accept and (not local_accept)
                and int(row["pred_parent"]) == int(row["true_parent"])
            ))
        elif row["status"] == "extra":
            extra_ok.append(bool(not root_accept))
    if not known_ok or not intra_ok or not extra_ok:
        raise ValueError("known, intra and extra development groups are required")
    rates = [np.mean(known_ok), np.mean(intra_ok), np.mean(extra_ok)]
    return float(np.mean(rates)), [float(x) for x in rates]


def calibrate_v10(
    train_reference,
    known,
    intra,
    extra,
    parent_names,
    leaf_names,
    settings,
):
    train_reference = list(train_reference)
    known, intra, extra = list(known), list(intra), list(extra)
    if not train_reference or not known or not intra or not extra:
        raise ValueError("All v10 development groups are required")

    bank = fit_reference_bank(train_reference, len(leaf_names), settings)
    known = add_reference_scores(known, bank)
    intra = add_reference_scores(intra, bank)
    extra = add_reference_scores(extra, bank)

    # Root positives include near-unknown samples: they belong to the taxonomy
    # and should be retained for parent-level fallback.
    root_fusion = fit_fusion(
        known + intra, extra, ROOT_FEATURES, settings
    )
    local_fusion = fit_fusion(
        known, intra, LOCAL_FEATURES, settings
    )
    for rows in (known, intra, extra):
        root_scores = fusion_scores(rows, root_fusion)
        local_scores = fusion_scores(rows, local_fusion)
        for i, row in enumerate(rows):
            row["global_knownness_score"] = float(root_scores[i])
            row["local_knownness_score"] = float(local_scores[i])

    all_rows = known + intra + extra
    root_values = [row["global_knownness_score"] for row in all_rows]
    local_values = [row["local_knownness_score"] for row in known + intra]
    root_candidates = _quantiles(
        root_values, settings.get("root_grid_quantiles", 121)
    )
    local_candidates = _quantiles(
        local_values, settings.get("local_grid_quantiles", 121)
    )

    known_floor = float(settings.get("min_known_e2e", 0.65))
    best = None
    pooled_key = -1
    for tau_root in root_candidates:
        for tau_local in local_candidates:
            score, rates = _macro_dra(
                all_rows, float(tau_root), {-1: float(tau_local)}
            )
            if rates[0] < known_floor:
                continue
            key = (score, rates[1] + rates[2], rates[0])
            if best is None or key > best[0]:
                best = (key, float(tau_root), float(tau_local), rates)
    if best is None:
        raise RuntimeError("No v10 operating point satisfies min_known_e2e")

    _, tau_root, pooled_local, pooled_rates = best

    # Parent-specific local thresholds are fitted only where both held-out
    # known and near-unknown evidence exist, then shrunk to the pooled value.
    by_parent_known, by_parent_intra = defaultdict(list), defaultdict(list)
    for row in known:
        by_parent_known[int(row["pred_parent"])].append(row)
    for row in intra:
        by_parent_intra[int(row["pred_parent"])].append(row)
    prior = float(settings.get("local_threshold_prior_count", 30.0))
    min_known = int(settings.get("min_branch_known", 12))
    min_intra = int(settings.get("min_branch_intra", 12))
    branches = {}
    local_thresholds = {-1: pooled_local}
    for parent_id, parent_name in enumerate(parent_names):
        krows = by_parent_known[parent_id]
        irows = by_parent_intra[parent_id]
        raw = pooled_local
        source = "pooled"
        if len(krows) >= min_known and len(irows) >= min_intra:
            candidates = _quantiles(
                [r["local_knownness_score"] for r in krows + irows],
                settings.get("local_grid_quantiles", 121),
            )
            local_best = None
            for cand in candidates:
                known_rate = np.mean([
                    float(r["local_knownness_score"]) >= cand
                    and int(r["pred_leaf"]) == int(r["true_leaf"])
                    for r in krows
                ])
                intra_rate = np.mean([
                    float(r["local_knownness_score"]) < cand
                    and int(r["pred_parent"]) == int(r["true_parent"])
                    for r in irows
                ])
                key = (0.5 * (known_rate + intra_rate), intra_rate, known_rate)
                if local_best is None or key > local_best[0]:
                    local_best = (key, float(cand))
            raw = local_best[1]
            source = "parent_specific"
        n = len(krows) + len(irows)
        weight = 1.0 if prior <= 0 else n / (n + prior)
        threshold = float(weight * raw + (1.0 - weight) * pooled_local)
        local_thresholds[parent_id] = threshold
        branches[parent_name] = {
            "local_knownness_threshold": threshold,
            "raw_local_threshold": float(raw),
            "source": source,
            "shrinkage_weight": float(weight),
            "known_count": len(krows),
            "intra_count": len(irows),
        }

    final_score, final_rates = _macro_dra(
        all_rows, tau_root, local_thresholds
    )
    return {
        "schema_version": 10,
        "method": "TaxoSafe-v10-learned-dual-boundary",
        "decision": "learned semantic+manifold root gate, then learned local gate",
        "parent_names": list(parent_names),
        "leaf_names": list(leaf_names),
        "reference_bank": bank,
        "root_fusion": root_fusion,
        "local_fusion": local_fusion,
        "root_threshold": tau_root,
        "pooled_local_threshold": pooled_local,
        "branches": branches,
        "settings": dict(settings),
        "development_operating_point": {
            "macro_deepest_reliable_accuracy": final_score,
            "known_e2e_leaf_accuracy": final_rates[0],
            "intra_correct_fallback_rate": final_rates[1],
            "extra_global_rejection_rate": final_rates[2],
            "pooled_search_rates": pooled_rates,
        },
    }


def apply_v10(records, router, parent_names, leaf_names):
    if int(router.get("schema_version", -1)) != 10:
        raise ValueError("Expected a TaxoSafe-v10 router")
    if list(router["parent_names"]) != list(parent_names):
        raise ValueError("Parent labels differ from router")
    if list(router["leaf_names"]) != list(leaf_names):
        raise ValueError("Leaf labels differ from router")

    rows = add_reference_scores(records, router["reference_bank"])
    root_scores = fusion_scores(rows, router["root_fusion"])
    local_scores = fusion_scores(rows, router["local_fusion"])
    output = []
    for i, row in enumerate(rows):
        row = dict(row)
        parent_id, leaf_id = int(row["pred_parent"]), int(row["pred_leaf"])
        parent_name = parent_names[parent_id]
        root_score = float(root_scores[i])
        local_score = float(local_scores[i])
        tau_local = float(
            router["branches"][parent_name]["local_knownness_threshold"]
        )
        if root_score < float(router["root_threshold"]):
            prediction_type, parent, leaf = "global_unknown", None, None
        elif local_score < tau_local:
            prediction_type, parent, leaf = "intra_unknown", parent_id, None
        else:
            prediction_type, parent, leaf = "known", parent_id, leaf_id
        row.update({
            "prediction_type": prediction_type,
            "candidate_parent": parent_id,
            "candidate_leaf": leaf_id,
            "parent": parent,
            "leaf": leaf,
            "root_knownness_score": root_score,
            "local_knownness_score": local_score,
            "root_threshold": float(router["root_threshold"]),
            "local_knownness_threshold": tau_local,
            "parent_name": None if parent is None else parent_names[parent],
            "leaf_name": None if leaf is None else leaf_names[leaf],
            "candidate_parent_name": parent_name,
            "candidate_leaf_name": leaf_names[leaf_id],
        })
        output.append(row)
    return output
