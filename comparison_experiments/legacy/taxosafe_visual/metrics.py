"""Open-set, hierarchy and safety metrics for TaxoSafe."""

from collections import Counter, defaultdict
import math
import os

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def _rate(mask):
    values = np.asarray(list(mask), dtype=np.float64)
    if values.size == 0:
        return None
    return float(values.mean())


def _count(records, prediction_type):
    return sum(
        record["prediction_type"] == prediction_type for record in records
    )


def _wilson_interval(successes, total, z=1.959963984540054):
    """Return a dependency-free 95% Wilson interval for a binomial rate."""
    total = int(total)
    successes = int(successes)
    if total <= 0:
        return None
    proportion = float(successes) / float(total)
    denominator = 1.0 + z * z / total
    centre = (
        proportion + z * z / (2.0 * total)
    ) / denominator
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / total
        + z * z / (4.0 * total * total)
    ) / denominator
    return {
        "low": float(max(0.0, centre - radius)),
        "high": float(min(1.0, centre + radius)),
        "successes": successes,
        "total": total,
        "method": "wilson_95",
    }


def _fpr_at_95_tpr(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.size == 0 or np.unique(labels).size < 2:
        return None
    fpr, tpr, _ = roc_curve(labels, scores)
    eligible = fpr[tpr >= 0.95]
    if eligible.size == 0:
        return 1.0
    return float(eligible.min())


def _root_detection_metrics(known_records, intra_records, extra_records):
    in_taxonomy = list(known_records) + list(intra_records)
    labels = [1] * len(in_taxonomy) + [0] * len(extra_records)
    scores = [
        float(record["root_knownness_score"])
        for record in in_taxonomy + list(extra_records)
    ]
    if not labels or len(set(labels)) < 2:
        return {"auroc": None, "aupr": None, "fpr95": None}
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "aupr": float(average_precision_score(labels, scores)),
        "fpr95": _fpr_at_95_tpr(labels, scores),
    }


def _known_group_metrics(records):
    records = list(records)
    accepted_correct = [
        record["prediction_type"] == "known"
        and record["candidate_parent"] == record["true_parent"]
        and record["candidate_leaf"] == record["true_leaf"]
        for record in records
    ]
    accepted_count = _count(records, "known")
    accepted_correct_count = int(sum(accepted_correct))
    return {
        "sample_count": len(records),
        "parent_accuracy": _rate(
            record["candidate_parent"] == record["true_parent"]
            for record in records
        ),
        "global_leaf_accuracy": _rate(
            record.get("global_pred_leaf", record["candidate_leaf"])
            == record["true_leaf"]
            for record in records
        ),
        "branch_restricted_leaf_accuracy": _rate(
            record["candidate_leaf"] == record["true_leaf"]
            for record in records
        ),
        "closed_hca": _rate(
            record["candidate_parent"] == record["true_parent"]
            and record.get("global_pred_leaf", record["candidate_leaf"])
            == record["true_leaf"]
            for record in records
        ),
        "end_to_end_leaf_accuracy": _rate(accepted_correct),
        "known_leaf_coverage": _rate(
            record["prediction_type"] == "known" for record in records
        ),
        "accepted_leaf_precision": (
            None
            if accepted_count == 0
            else float(accepted_correct_count / accepted_count)
        ),
        "accepted_leaf_count": int(accepted_count),
        "accepted_correct_leaf_count": accepted_correct_count,
        "under_specification_rate": _rate(
            record["prediction_type"] == "intra_unknown"
            for record in records
        ),
        "known_global_rejection_rate": _rate(
            record["prediction_type"] == "global_unknown"
            for record in records
        ),
    }


def _intra_group_metrics(records):
    records = list(records)
    return {
        "sample_count": len(records),
        "correct_fallback_rate": _rate(
            record["prediction_type"] == "intra_unknown"
            and record["parent"] == record["true_parent"]
            for record in records
        ),
        "over_specification_error_rate": _rate(
            record["prediction_type"] == "known" for record in records
        ),
        "wrong_parent_rate": _rate(
            record["candidate_parent"] != record["true_parent"]
            for record in records
        ),
        "accepted_wrong_parent_rate": _rate(
            record["prediction_type"] != "global_unknown"
            and record["parent"] != record["true_parent"]
            for record in records
        ),
        "parent_localization_accuracy": _rate(
            record["candidate_parent"] == record["true_parent"]
            for record in records
        ),
        "intra_global_rejection_rate": _rate(
            record["prediction_type"] == "global_unknown"
            for record in records
        ),
    }


def _extra_group_metrics(records):
    records = list(records)
    global_count = _count(records, "global_unknown")
    known_count = _count(records, "known")
    intra_count = _count(records, "intra_unknown")
    return {
        "sample_count": len(records),
        "global_unknown_recall": (
            None if not records else float(global_count / len(records))
        ),
        "false_parent_acceptance_rate": (
            None
            if not records
            else float((known_count + intra_count) / len(records))
        ),
        "false_known_leaf_rate": (
            None if not records else float(known_count / len(records))
        ),
        "prediction_type_counts": dict(sorted(Counter(
            record["prediction_type"] for record in records
        ).items())),
        "accepted_parent_distribution": dict(sorted(Counter(
            record.get(
                "candidate_parent_name", str(record["candidate_parent"])
            )
            for record in records
            if record["prediction_type"] != "global_unknown"
        ).items())),
    }


def _per_group(records, status):
    grouped = defaultdict(list)
    for record in records:
        name = record.get("true_parent_name")
        if name is None:
            name = str(record.get("true_parent"))
        grouped[name].append(record)
    metric_fn = (
        _known_group_metrics if status == "known" else _intra_group_metrics
    )
    return {
        name: metric_fn(grouped[name]) for name in sorted(grouped)
    }


def _per_leaf(records):
    grouped = defaultdict(list)
    for record in records:
        name = record.get("true_leaf_name")
        if name is None:
            name = str(record.get("true_leaf"))
        grouped[name].append(record)
    return {
        name: _known_group_metrics(grouped[name]) for name in sorted(grouped)
    }


def _source_name(record):
    path = str(record.get("path", "")).replace("\\", "/")
    parent = os.path.basename(os.path.dirname(path))
    return parent or "unknown_source"


def _per_source(records, metric_fn):
    grouped = defaultdict(list)
    for record in records:
        grouped[_source_name(record)].append(record)
    return {
        name: metric_fn(grouped[name]) for name in sorted(grouped)
    }


def evaluate_open_set(known_records, intra_records, extra_records):
    """Evaluate one fixed TaxoSafe operating point; rates are in [0, 1]."""
    known_records = list(known_records)
    intra_records = list(intra_records)
    extra_records = list(extra_records)
    if not known_records or not intra_records or not extra_records:
        raise ValueError(
            "known, intra and extra test records must all be non-empty"
        )

    known_metrics = _known_group_metrics(known_records)
    intra_metrics = _intra_group_metrics(intra_records)
    extra_metrics = {
        "sample_count": len(extra_records),
        "global_unknown_recall": _rate(
            record["prediction_type"] == "global_unknown"
            for record in extra_records
        ),
        "false_parent_acceptance_rate": _rate(
            record["prediction_type"] != "global_unknown"
            for record in extra_records
        ),
        "false_known_leaf_rate": _rate(
            record["prediction_type"] == "known"
            for record in extra_records
        ),
        "candidate_parent_distribution": dict(sorted(Counter(
            record.get(
                "candidate_parent_name", str(record["candidate_parent"])
            )
            for record in extra_records
        ).items())),
        "accepted_parent_distribution": dict(sorted(Counter(
            record.get(
                "candidate_parent_name", str(record["candidate_parent"])
            )
            for record in extra_records
            if record["prediction_type"] != "global_unknown"
        ).items())),
    }
    extra_metrics.update(
        _root_detection_metrics(known_records, intra_records, extra_records)
    )

    correct_known = [
        record["prediction_type"] == "known"
        and record["parent"] == record["true_parent"]
        and record["leaf"] == record["true_leaf"]
        for record in known_records
    ]
    correct_intra = [
        record["prediction_type"] == "intra_unknown"
        and record["parent"] == record["true_parent"]
        for record in intra_records
    ]
    correct_extra = [
        record["prediction_type"] == "global_unknown"
        for record in extra_records
    ]
    all_correct = correct_known + correct_intra + correct_extra

    all_records = known_records + intra_records + extra_records
    depth = {"global_unknown": 0.0, "intra_unknown": 1.0, "known": 2.0}
    mean_depth = float(np.mean([
        depth[record["prediction_type"]] for record in all_records
    ]))
    over_specification_events = [
        record["prediction_type"] == "known" for record in intra_records
    ] + [
        record["prediction_type"] != "global_unknown"
        for record in extra_records
    ]
    accepted_leaf_count = sum(
        record["prediction_type"] == "known" for record in all_records
    )
    correct_leaf_count = int(np.sum(correct_known))
    unknown_known_leaf_count = sum(
        record["prediction_type"] == "known" for record in intra_records
    ) + sum(
        record["prediction_type"] == "known" for record in extra_records
    )

    overall_metrics = {
        "sample_count": len(all_records),
        "deepest_reliable_taxon_accuracy": _rate(all_correct),
        "macro_deepest_reliable_taxon_accuracy": float(np.mean([
            known_metrics["end_to_end_leaf_accuracy"],
            intra_metrics["correct_fallback_rate"],
            extra_metrics["global_unknown_recall"],
        ])),
        "correct_count": int(np.sum(all_correct)),
        "mean_prediction_depth": mean_depth,
        "normalized_specificity": mean_depth / 2.0,
        "over_specification_risk": _rate(over_specification_events),
        "prediction_type_counts": dict(sorted(Counter(
            record["prediction_type"] for record in all_records
        ).items())),
        "open_world_accepted_leaf_precision": (
            None
            if accepted_leaf_count == 0
            else float(correct_leaf_count / accepted_leaf_count)
        ),
        "accepted_leaf_count": int(accepted_leaf_count),
        "correct_leaf_count": int(correct_leaf_count),
        "unknown_as_known_leaf_count": int(unknown_known_leaf_count),
        "rate_unit": "fraction_in_[0,1]",
    }
    confidence_intervals = {
        "known_end_to_end_leaf_accuracy": _wilson_interval(
            int(np.sum(correct_known)), len(known_records)
        ),
        "intra_correct_fallback_rate": _wilson_interval(
            int(np.sum(correct_intra)), len(intra_records)
        ),
        "intra_over_specification_error_rate": _wilson_interval(
            _count(intra_records, "known"), len(intra_records)
        ),
        "extra_global_unknown_recall": _wilson_interval(
            _count(extra_records, "global_unknown"), len(extra_records)
        ),
        "extra_false_parent_acceptance_rate": _wilson_interval(
            len(extra_records) - _count(extra_records, "global_unknown"),
            len(extra_records),
        ),
        "deepest_reliable_taxon_accuracy": _wilson_interval(
            int(np.sum(all_correct)), len(all_records)
        ),
        "open_world_accepted_leaf_precision": _wilson_interval(
            correct_leaf_count, accepted_leaf_count
        ),
    }
    return {
        "known": known_metrics,
        "intra": intra_metrics,
        "extra": extra_metrics,
        "overall": overall_metrics,
        "confidence_intervals_95": confidence_intervals,
        "per_parent": {
            "known": _per_group(known_records, "known"),
            "intra": _per_group(intra_records, "intra"),
        },
        "per_known_leaf": _per_leaf(known_records),
        "per_intra_species": _per_source(
            intra_records, _intra_group_metrics
        ),
        "per_extra_source": _per_source(
            extra_records, _extra_group_metrics
        ),
    }


def risk_specificity_point(metrics, threshold_shift):
    """Extract a compact point for the risk-specificity curve."""
    return {
        "threshold_shift": float(threshold_shift),
        "normalized_specificity": float(
            metrics["overall"]["normalized_specificity"]
        ),
        "over_specification_risk": float(
            metrics["overall"]["over_specification_risk"]
        ),
        "deepest_reliable_taxon_accuracy": float(
            metrics["overall"]["deepest_reliable_taxon_accuracy"]
        ),
    }
