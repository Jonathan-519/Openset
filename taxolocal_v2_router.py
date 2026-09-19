"""Dual-boundary hierarchical router for TaxoLocal-v2.

The global gate fuses parent energy evidence with a ViM-style feature
residual.  The local gate fuses the learned unknown-descendant margin with
distance to the candidate leaf prototype.  All fitted state comes from the
known training reference bank and the three development partitions.
"""

from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression

from taxolocal_router import _largest_feasible_threshold


GLOBAL_FEATURES = ("parent_logsumexp", "vim_residual")
LOCAL_FEATURES = ("local_known_margin", "prototype_distance")


def _unit(values, axis=-1, eps=1e-12):
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=axis, keepdims=True)
    return values / np.maximum(norms, eps)


def _features(records):
    if not records:
        raise ValueError("Reference records are empty")
    try:
        values = np.asarray(
            [record["image_feature"] for record in records],
            dtype=np.float64,
        )
    except KeyError as error:
        raise ValueError("Records do not contain image features") from error
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("Image features must have shape [N, D], D >= 2")
    if not np.isfinite(values).all():
        raise ValueError("Image features contain non-finite values")
    return _unit(values)


def fit_reference_bank(records, num_leaves, settings):
    """Fit TRAIN-only ViM subspace and normalized leaf prototypes."""
    records = list(records)
    features = _features(records)
    labels = np.asarray([record["true_leaf"] for record in records], int)
    if np.any(labels < 0) or np.any(labels >= int(num_leaves)):
        raise ValueError("Training reference has an invalid leaf label")
    missing = sorted(set(range(int(num_leaves))).difference(labels.tolist()))
    if missing:
        raise ValueError("Training reference is missing leaves: {}".format(missing))

    mean = features.mean(axis=0)
    centered = features - mean
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    requested = int(settings.get("vim_principal_dim", 128))
    maximum = min(vt.shape[0], features.shape[1] - 1)
    principal_dim = min(maximum, max(0, requested))
    axes = vt[:principal_dim]

    prototypes = []
    counts = []
    for leaf in range(int(num_leaves)):
        selected = features[labels == leaf]
        prototype = _unit(selected.mean(axis=0, keepdims=True))[0]
        prototypes.append(prototype.tolist())
        counts.append(int(len(selected)))

    return {
        "schema_version": 1,
        "source": "known_train_reference_only",
        "feature_dim": int(features.shape[1]),
        "sample_count": int(len(features)),
        "mean": mean.tolist(),
        "principal_axes": axes.tolist(),
        "principal_dim": int(principal_dim),
        "singular_values": singular[:principal_dim].tolist(),
        "leaf_prototypes": prototypes,
        "leaf_counts": counts,
    }


def _reference_scores(records, bank):
    features = _features(records)
    mean = np.asarray(bank["mean"], dtype=np.float64)
    axes = np.asarray(bank["principal_axes"], dtype=np.float64)
    prototypes = np.asarray(bank["leaf_prototypes"], dtype=np.float64)
    if features.shape[1] != int(bank["feature_dim"]):
        raise ValueError("Query and reference feature dimensions differ")
    centered = features - mean
    if axes.size:
        residual = centered - (centered @ axes.T) @ axes
    else:
        residual = centered
    vim_residual = np.linalg.norm(residual, axis=1)
    candidates = np.asarray([record["pred_leaf"] for record in records], int)
    if np.any(candidates < 0) or np.any(candidates >= len(prototypes)):
        raise ValueError("Candidate leaf is outside the prototype bank")
    similarity = np.sum(features * prototypes[candidates], axis=1)
    return vim_residual, 1.0 - np.clip(similarity, -1.0, 1.0), similarity


def add_reference_scores(records, bank):
    """Return copies augmented with ViM and candidate-prototype evidence."""
    records = [dict(record) for record in records]
    residual, distance, similarity = _reference_scores(records, bank)
    for index, record in enumerate(records):
        record["vim_residual"] = float(residual[index])
        record["prototype_distance"] = float(distance[index])
        record["prototype_similarity"] = float(similarity[index])
        # logsumexp(parent logits) is the negative of the conventional energy
        # up to sign; larger values are therefore more known-like.
        record["parent_energy_knownness"] = float(
            record["parent_logsumexp"]
        )
    return records


def _matrix(records, names):
    values = np.asarray(
        [[record[name] for name in names] for record in records],
        dtype=np.float64,
    )
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Fusion evidence must be a finite matrix")
    return values


def fit_fusion(positive, negative, names, settings):
    """Fit a small regularized development-only knownness model."""
    positive, negative = list(positive), list(negative)
    if not positive or not negative:
        raise ValueError("Fusion fitting needs positive and negative records")
    x = _matrix(positive + negative, names)
    y = np.r_[np.ones(len(positive), int), np.zeros(len(negative), int)]
    mean, std = x.mean(axis=0), x.std(axis=0)
    std[std < 1e-8] = 1.0
    model = LogisticRegression(
        C=float(settings.get("fusion_c", 1.0)),
        class_weight="balanced",
        max_iter=int(settings.get("fusion_max_iter", 1000)),
        solver="lbfgs",
        random_state=0,
    )
    model.fit((x - mean) / std, y)
    return {
        "feature_names": list(names),
        "mean": mean.tolist(),
        "scale": std.tolist(),
        "coefficient": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "positive_count": len(positive),
        "negative_count": len(negative),
        "score": "linear_log_odds; larger_is_more_known",
    }


def fusion_scores(records, state):
    x = _matrix(records, state["feature_names"])
    mean = np.asarray(state["mean"], dtype=np.float64)
    scale = np.asarray(state["scale"], dtype=np.float64)
    coefficient = np.asarray(state["coefficient"], dtype=np.float64)
    return ((x - mean) / scale) @ coefficient + float(state["intercept"])


def _rate(values):
    values = list(values)
    return None if not values else float(np.mean(values))


def calibrate_router(
    train_reference,
    known,
    novel,
    extra,
    parent_names,
    leaf_names,
    settings,
):
    """Fit both boundaries without reading any locked-test partition."""
    train_reference = list(train_reference)
    known, novel, extra = list(known), list(novel), list(extra)
    if not train_reference or not known or not novel or not extra:
        raise ValueError("TRAIN reference and all development groups are required")
    bank = fit_reference_bank(train_reference, len(leaf_names), settings)
    known = add_reference_scores(known, bank)
    novel = add_reference_scores(novel, bank)
    extra = add_reference_scores(extra, bank)

    global_model = fit_fusion(known, extra, GLOBAL_FEATURES, settings)
    local_model = fit_fusion(known, novel, LOCAL_FEATURES, settings)
    for records in (known, novel, extra):
        global_values = fusion_scores(records, global_model)
        local_values = fusion_scores(records, local_model)
        for index, record in enumerate(records):
            record["global_knownness_score"] = float(global_values[index])
            record["local_knownness_score"] = float(local_values[index])

    known_floor = float(settings.get("root_known_coverage_floor", 0.95))
    novel_floor = float(settings.get("root_novel_coverage_floor", 0.85))
    local_floor = float(settings.get("local_known_coverage_floor", 0.90))
    min_novel = int(settings.get("per_parent_min_dev_unknown", 20))
    root_threshold = min(
        _largest_feasible_threshold(
            [row["global_knownness_score"] for row in known], known_floor
        ),
        _largest_feasible_threshold(
            [row["global_knownness_score"] for row in novel], novel_floor
        ),
    )
    pooled_local = _largest_feasible_threshold(
        [row["local_knownness_score"] for row in known], local_floor
    )

    by_parent_known, by_parent_novel = defaultdict(list), defaultdict(list)
    for row in known:
        by_parent_known[int(row["pred_parent"])].append(row)
    for row in novel:
        by_parent_novel[int(row["pred_parent"])].append(row)
    branches = {}
    for parent_id, parent_name in enumerate(parent_names):
        branch_known = by_parent_known[parent_id]
        branch_novel = by_parent_novel[parent_id]
        if len(branch_known) >= 10 and len(branch_novel) >= min_novel:
            threshold = _largest_feasible_threshold(
                [row["local_knownness_score"] for row in branch_known],
                local_floor,
            )
            source = "parent_specific"
        else:
            threshold, source = pooled_local, "pooled"
        branches[parent_name] = {
            "local_knownness_threshold": float(threshold),
            "source": source,
            "known_count": len(branch_known),
            "development_unknown_count": len(branch_novel),
        }

    router = {
        "schema_version": 2,
        "method": "TaxoLocal-v2-dual-boundary",
        "decision": "Energy+ViM global gate, then margin+prototype local gate",
        "parent_names": list(parent_names),
        "leaf_names": list(leaf_names),
        "reference_bank": bank,
        "global_fusion": global_model,
        "local_fusion": local_model,
        "root_threshold": float(root_threshold),
        "pooled_local_knownness_threshold": float(pooled_local),
        "branches": branches,
        "settings": {
            "root_known_coverage_floor": known_floor,
            "root_novel_coverage_floor": novel_floor,
            "local_known_coverage_floor": local_floor,
            "per_parent_min_dev_unknown": min_novel,
            "vim_principal_dim": int(bank["principal_dim"]),
            "fusion_c": float(settings.get("fusion_c", 1.0)),
        },
    }
    development = apply_router(
        known + novel + extra, router, parent_names, leaf_names,
        scores_already_added=True,
    )
    router["development_operating_point"] = {
        "known_leaf_acceptance": _rate(
            row["prediction_type"] == "known" for row in development
            if row["status"] == "known"
        ),
        "ancestor_retained_unknown_accuracy": _rate(
            row["prediction_type"] == "intra_unknown"
            and row["parent"] == row["true_parent"]
            for row in development if row["status"] == "intra"
        ),
        "far_unknown_recall": _rate(
            row["prediction_type"] == "global_unknown"
            for row in development if row["status"] == "extra"
        ),
    }
    return router


def apply_router(
    records,
    router,
    parent_names,
    leaf_names,
    scores_already_added=False,
):
    """Apply the frozen two-gate decision rule to score records."""
    if list(router["parent_names"]) != list(parent_names):
        raise ValueError("Router parent labels do not match the model")
    if list(router["leaf_names"]) != list(leaf_names):
        raise ValueError("Router leaf labels do not match the model")
    records = [dict(record) for record in records]
    if not scores_already_added:
        records = add_reference_scores(records, router["reference_bank"])
        global_values = fusion_scores(records, router["global_fusion"])
        local_values = fusion_scores(records, router["local_fusion"])
        for index, record in enumerate(records):
            record["global_knownness_score"] = float(global_values[index])
            record["local_knownness_score"] = float(local_values[index])

    output = []
    for record in records:
        parent_id, leaf_id = int(record["pred_parent"]), int(record["pred_leaf"])
        parent_name = parent_names[parent_id]
        local_threshold = float(
            router["branches"][parent_name]["local_knownness_threshold"]
        )
        if float(record["global_knownness_score"]) < float(router["root_threshold"]):
            prediction_type, parent, leaf = "global_unknown", None, None
        elif float(record["local_knownness_score"]) < local_threshold:
            prediction_type, parent, leaf = "intra_unknown", parent_id, None
        else:
            prediction_type, parent, leaf = "known", parent_id, leaf_id
        record.update({
            "prediction_type": prediction_type,
            "candidate_parent": parent_id,
            "candidate_leaf": leaf_id,
            "parent": parent,
            "leaf": leaf,
            "root_knownness_score": float(record["global_knownness_score"]),
            "root_threshold": float(router["root_threshold"]),
            "local_knownness_threshold": local_threshold,
            "parent_name": None if parent is None else parent_names[parent],
            "leaf_name": None if leaf is None else leaf_names[leaf],
            "candidate_parent_name": parent_name,
            "candidate_leaf_name": leaf_names[leaf_id],
        })
        true_parent, true_leaf = record.get("true_parent"), record.get("true_leaf")
        record["true_parent_name"] = (
            None if true_parent is None else parent_names[true_parent]
        )
        record["true_leaf_name"] = (
            None if true_leaf is None else leaf_names[true_leaf]
        )
        output.append(record)
    return output
