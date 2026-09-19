"""Validation-fitted hierarchical router for TaxoLocal predictions."""

from collections import defaultdict

import numpy as np


def _rate(values):
    values = list(values)
    return None if not values else float(np.mean(values))


def _largest_feasible_threshold(positive, floor):
    values = np.asarray(positive, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Calibration scores must be non-empty and finite")
    if not 0.0 <= float(floor) <= 1.0:
        raise ValueError("Coverage floor must be in [0, 1]")
    candidates = np.unique(np.concatenate([
        values,
        [float(values.min()) - 1e-9, float(values.max()) + 1e-9],
    ]))
    feasible = [
        float(value)
        for value in candidates
        if float(np.mean(values >= value)) >= float(floor)
    ]
    if not feasible:
        raise RuntimeError("No threshold satisfies the coverage floor")
    return max(feasible)


def calibrate_router(known, novel, extra, parent_names, settings):
    """Fit only prior-correction cut points; unknown logits remain primary."""
    known = list(known)
    novel = list(novel)
    extra = list(extra)
    if not known or not novel or not extra:
        raise ValueError("known, novel and extra development records are required")
    known_floor = float(settings.get("root_known_coverage_floor", 0.95))
    novel_root_floor = float(
        settings.get("root_novel_coverage_floor", 0.85)
    )
    local_floor = float(settings.get("local_known_coverage_floor", 0.90))
    min_novel = int(settings.get("per_parent_min_dev_unknown", 20))

    root_threshold = min(
        _largest_feasible_threshold(
            [record["parent_score"] for record in known], known_floor
        ),
        _largest_feasible_threshold(
            [record["parent_score"] for record in novel], novel_root_floor
        ),
    )
    pooled_local = _largest_feasible_threshold(
        [record["local_known_margin"] for record in known], local_floor
    )
    # A known leaf is accepted when margin >= threshold.  The learned unknown
    # class wins directly at threshold 0; validation only corrects its prior.
    by_parent_known = defaultdict(list)
    by_parent_novel = defaultdict(list)
    for record in known:
        by_parent_known[int(record["pred_parent"])].append(record)
    for record in novel:
        by_parent_novel[int(record["pred_parent"])].append(record)

    branches = {}
    for parent_id, parent_name in enumerate(parent_names):
        branch_known = by_parent_known[parent_id]
        branch_novel = by_parent_novel[parent_id]
        if len(branch_known) >= 10 and len(branch_novel) >= min_novel:
            local_threshold = _largest_feasible_threshold(
                [record["local_known_margin"] for record in branch_known],
                local_floor,
            )
            source = "parent_specific"
        else:
            local_threshold = pooled_local
            source = "pooled"
        branches[parent_name] = {
            "local_margin_threshold": float(local_threshold),
            "source": source,
            "known_count": len(branch_known),
            "development_unknown_count": len(branch_novel),
        }

    router = {
        "schema_version": 1,
        "decision": (
            "root parent_score gate, then explicit local unknown-vs-child logit"
        ),
        "root_threshold": float(root_threshold),
        "pooled_local_margin_threshold": float(pooled_local),
        "parent_names": list(parent_names),
        "branches": branches,
        "settings": {
            "root_known_coverage_floor": known_floor,
            "root_novel_coverage_floor": novel_root_floor,
            "local_known_coverage_floor": local_floor,
            "per_parent_min_dev_unknown": min_novel,
        },
    }
    development = apply_router(known + novel + extra, router, parent_names, None)
    router["development_operating_point"] = {
        "known_leaf_acceptance": _rate(
            row["prediction_type"] == "known" for row in development
            if row["status"] == "known"
        ),
        "unknown_leaf_rejection": _rate(
            row["prediction_type"] != "known" for row in development
            if row["status"] == "intra"
        ),
        "ancestor_retained_unknown_accuracy": _rate(
            row["prediction_type"] == "intra_unknown"
            and row["parent"] == row["true_parent"]
            for row in development
            if row["status"] == "intra"
        ),
        "far_unknown_recall": _rate(
            row["prediction_type"] == "global_unknown" for row in development
            if row["status"] == "extra"
        ),
    }
    return router


def apply_router(records, router, parent_names, leaf_names):
    output = []
    if list(router["parent_names"]) != list(parent_names):
        raise ValueError("Router parent labels do not match the model")
    for source in records:
        record = dict(source)
        parent_id = int(record["pred_parent"])
        leaf_id = int(record["pred_leaf"])
        parent_name = parent_names[parent_id]
        local_threshold = float(
            router["branches"][parent_name]["local_margin_threshold"]
        )
        if float(record["parent_score"]) < float(router["root_threshold"]):
            prediction_type = "global_unknown"
            parent = None
            leaf = None
        elif float(record["local_known_margin"]) < local_threshold:
            prediction_type = "intra_unknown"
            parent = parent_id
            leaf = None
        else:
            prediction_type = "known"
            parent = parent_id
            leaf = leaf_id
        record.update({
            "prediction_type": prediction_type,
            "candidate_parent": parent_id,
            "candidate_leaf": leaf_id,
            "parent": parent,
            "leaf": leaf,
            "root_knownness_score": float(record["parent_score"]),
            "local_margin_threshold": local_threshold,
            "root_threshold": float(router["root_threshold"]),
            "parent_name": None if parent is None else parent_names[parent],
            "candidate_parent_name": parent_name,
        })
        if leaf_names is not None:
            record["leaf_name"] = (
                None if leaf is None else leaf_names[leaf]
            )
            record["candidate_leaf_name"] = leaf_names[leaf_id]
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
