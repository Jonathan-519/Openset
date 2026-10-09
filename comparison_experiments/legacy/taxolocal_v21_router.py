"""TaxoLocal-v2.1 evidence-intersection router.

This is a post-hoc upgrade for an already calibrated TaxoLocal-v2 model.  It
does not update the encoder, prompt parameters, reference bank or local fusion
model.  The root decision rejects a query only when both channels agree:

* semantic channel: parent margin and normalized negative entropy;
* manifold channel: ViM residual and candidate-prototype distance.

Parent-specific local thresholds are shrunk toward the pooled threshold to
reduce small-branch and seed instability.
"""

import copy

import numpy as np

from taxolocal_v2_router import add_reference_scores, fusion_scores


ROOT_FEATURES = (
    "parent_margin",
    "parent_neg_entropy",
    "vim_residual",
    "prototype_distance",
)
ROOT_DIRECTIONS = (1.0, 1.0, -1.0, -1.0)


def _matrix(records, names):
    values = np.asarray(
        [[record[name] for name in names] for record in records],
        dtype=np.float64,
    )
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Root evidence must be a finite matrix")
    return values


def _channel_scores(records, state):
    x = _matrix(records, state["feature_names"])
    directions = np.asarray(state["directions"], dtype=np.float64)
    mean = np.asarray(state["mean"], dtype=np.float64)
    scale = np.asarray(state["scale"], dtype=np.float64)
    z = (x * directions - mean) / scale
    semantic = z[:, :2].mean(axis=1)
    manifold = z[:, 2:].mean(axis=1)
    return semantic, manifold


def _coverage(mask, values):
    if not np.any(mask):
        raise ValueError("A development status group is empty")
    return float(np.mean(values[mask]))


def fit_intersection_gate(records, settings):
    """Fit a two-channel AND-rejection rule on development partitions."""
    records = list(records)
    if not records:
        raise ValueError("Development records are empty")
    x = _matrix(records, ROOT_FEATURES)
    directions = np.asarray(ROOT_DIRECTIONS, dtype=np.float64)
    directed = x * directions
    mean, scale = directed.mean(axis=0), directed.std(axis=0)
    scale[scale < 1e-8] = 1.0
    state = {
        "feature_names": list(ROOT_FEATURES),
        "directions": directions.tolist(),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
    }
    semantic, manifold = _channel_scores(records, state)
    statuses = np.asarray([record["status"] for record in records])
    known, novel, extra = (
        statuses == "known",
        statuses == "intra",
        statuses == "extra",
    )
    if not np.any(known) or not np.any(novel) or not np.any(extra):
        raise ValueError("known, intra and extra development records are required")

    known_floor = float(settings.get("root_known_coverage_floor", 0.95))
    novel_floor = float(settings.get("root_novel_coverage_floor", 0.85))
    grid_size = int(settings.get("root_grid_quantiles", 101))
    if not 0.0 <= known_floor <= 1.0 or not 0.0 <= novel_floor <= 1.0:
        raise ValueError("Root coverage floors must lie in [0, 1]")
    if grid_size < 3:
        raise ValueError("root_grid_quantiles must be at least 3")
    quantiles = np.linspace(0.0, 1.0, grid_size)
    semantic_candidates = np.unique(np.quantile(semantic, quantiles))
    manifold_candidates = np.unique(np.quantile(manifold, quantiles))

    best = None
    for semantic_threshold in semantic_candidates:
        semantic_low = semantic < semantic_threshold
        for manifold_threshold in manifold_candidates:
            # Global rejection requires agreement between both channels.
            accepted = ~(
                semantic_low & (manifold < manifold_threshold)
            )
            known_coverage = _coverage(known, accepted)
            novel_retention = _coverage(novel, accepted)
            if known_coverage < known_floor or novel_retention < novel_floor:
                continue
            extra_recall = _coverage(extra, ~accepted)
            key = (
                extra_recall,
                known_coverage + novel_retention,
                known_coverage,
                novel_retention,
            )
            if best is None or key > best[0]:
                best = (
                    key,
                    float(semantic_threshold),
                    float(manifold_threshold),
                )
    if best is None:
        raise RuntimeError("No root intersection thresholds satisfy coverage floors")

    key, semantic_threshold, manifold_threshold = best
    state.update({
        "semantic_features": list(ROOT_FEATURES[:2]),
        "manifold_features": list(ROOT_FEATURES[2:]),
        "semantic_threshold": semantic_threshold,
        "manifold_threshold": manifold_threshold,
        "decision": (
            "global_unknown iff semantic_score < semantic_threshold AND "
            "manifold_score < manifold_threshold"
        ),
        "known_coverage_floor": known_floor,
        "novel_retention_floor": novel_floor,
        "grid_quantiles": grid_size,
        "development_operating_point": {
            "extra_global_unknown_recall": float(key[0]),
            "known_root_retention": float(key[2]),
            "intra_root_retention": float(key[3]),
        },
    })
    return state


def _rate(values):
    values = list(values)
    return None if not values else float(np.mean(values))


def upgrade_router(base_router, development_records, settings):
    """Upgrade a frozen v2 router using its development predictions only."""
    development_records = [dict(record) for record in development_records]
    if int(base_router.get("schema_version", -1)) != 2:
        raise ValueError("TaxoLocal-v2.1 requires a schema-version 2 router")
    if any(
        key not in base_router
        for key in ("reference_bank", "local_fusion", "branches")
    ):
        raise ValueError("The v2 router is missing reusable fitted state")
    required = set(ROOT_FEATURES) | {
        "local_knownness_score", "status", "pred_parent", "pred_leaf"
    }
    for index, record in enumerate(development_records):
        missing = required.difference(record)
        if missing:
            raise ValueError(
                "Development record {} is missing {}".format(
                    index, sorted(missing)
                )
            )

    router = copy.deepcopy(base_router)
    legacy = {
        "method": router.get("method"),
        "decision": router.get("decision"),
        "root_threshold": router.pop("root_threshold", None),
        "global_fusion": router.pop("global_fusion", None),
        "development_operating_point": router.pop(
            "development_operating_point", None
        ),
    }
    router["schema_version"] = 3
    router["method"] = "TaxoLocal-v2.1-evidence-intersection"
    router["decision"] = (
        "semantic/manifold intersection root gate, then shrunk "
        "margin+prototype local gate"
    )
    router["legacy_v2_root"] = legacy
    router["root_intersection_gate"] = fit_intersection_gate(
        development_records, settings
    )

    prior_count = float(settings.get("local_threshold_prior_count", 50.0))
    if prior_count < 0.0:
        raise ValueError("local_threshold_prior_count must be non-negative")
    pooled = float(router["pooled_local_knownness_threshold"])
    for branch in router["branches"].values():
        original = float(branch["local_knownness_threshold"])
        known_count = int(branch["known_count"])
        weight = (
            1.0 if prior_count == 0.0
            else known_count / (known_count + prior_count)
        )
        branch["v2_local_knownness_threshold"] = original
        branch["local_threshold_shrinkage_weight"] = float(weight)
        branch["local_knownness_threshold"] = float(
            weight * original + (1.0 - weight) * pooled
        )
        branch["source"] = branch["source"] + "_shrunk_to_pooled"

    router["settings"] = dict(router.get("settings", {}))
    router["settings"].update({
        "root_gate": "semantic_manifold_intersection",
        "root_grid_quantiles": int(
            router["root_intersection_gate"]["grid_quantiles"]
        ),
        "local_threshold_prior_count": prior_count,
    })
    development = apply_router(
        development_records,
        router,
        router["parent_names"],
        router["leaf_names"],
        scores_already_added=True,
    )
    router["development_operating_point"] = {
        "known_leaf_acceptance": _rate(
            row["prediction_type"] == "known"
            for row in development if row["status"] == "known"
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
    """Apply the frozen v2.1 evidence-intersection decision rule."""
    if int(router.get("schema_version", -1)) != 3:
        raise ValueError("Expected a TaxoLocal-v2.1 schema-version 3 router")
    if list(router["parent_names"]) != list(parent_names):
        raise ValueError("Router parent labels do not match the model")
    if list(router["leaf_names"]) != list(leaf_names):
        raise ValueError("Router leaf labels do not match the model")
    records = [dict(record) for record in records]
    if not scores_already_added:
        records = add_reference_scores(records, router["reference_bank"])
        local_values = fusion_scores(records, router["local_fusion"])
        for index, record in enumerate(records):
            record["local_knownness_score"] = float(local_values[index])
    elif any("local_knownness_score" not in row for row in records):
        raise ValueError("Pre-scored records lack local_knownness_score")

    gate = router["root_intersection_gate"]
    semantic, manifold = _channel_scores(records, gate)
    semantic_threshold = float(gate["semantic_threshold"])
    manifold_threshold = float(gate["manifold_threshold"])
    output = []
    for index, record in enumerate(records):
        parent_id = int(record["pred_parent"])
        leaf_id = int(record["pred_leaf"])
        parent_name = parent_names[parent_id]
        local_threshold = float(
            router["branches"][parent_name]["local_knownness_threshold"]
        )
        semantic_margin = float(semantic[index] - semantic_threshold)
        manifold_margin = float(manifold[index] - manifold_threshold)
        # The max is non-negative exactly when at least one channel retains
        # the query as in-taxonomy.
        root_knownness = max(semantic_margin, manifold_margin)
        if root_knownness < 0.0:
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
            "root_knownness_score": float(root_knownness),
            "root_semantic_score": float(semantic[index]),
            "root_manifold_score": float(manifold[index]),
            "root_semantic_margin": semantic_margin,
            "root_manifold_margin": manifold_margin,
            "local_knownness_threshold": local_threshold,
            "parent_name": None if parent is None else parent_names[parent],
            "leaf_name": None if leaf is None else leaf_names[leaf],
            "candidate_parent_name": parent_name,
            "candidate_leaf_name": leaf_names[leaf_id],
        })
        true_parent = record.get("true_parent")
        true_leaf = record.get("true_leaf")
        record["true_parent_name"] = (
            None if true_parent is None else parent_names[true_parent]
        )
        record["true_leaf_name"] = (
            None if true_leaf is None else leaf_names[true_leaf]
        )
        output.append(record)
    return output
