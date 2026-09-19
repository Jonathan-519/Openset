"""Calibrate TaxoSafe thresholds on validation data only.

Two pre-declared policies are supported. ``balanced`` protects a configured
coverage floor and reports any unresolved risk violation. ``safe`` enforces
the FAR/OSER limits first and reports any resulting coverage violation.
"""

import argparse
import copy
import datetime
import json
import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve

from taxosafe_eval_utils import (
    collect_score_records,
    load_model_and_data,
    load_yaml,
    resolve_run_dir,
    sha256_file,
    write_json,
    write_jsonl,
)


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CALIBRATION_SPLITS = ("val_known", "val_intra", "val_extra")
DEFAULT_ROOT_EVIDENCE = (
    "parent_score",
    "parent_margin",
    "parent_msp",
    "parent_neg_entropy",
    "parent_logsumexp",
)
DEFAULT_CHILD_EVIDENCE_WEIGHTS = {
    "child_score": 0.34,
    "child_margin": 0.33,
    "child_neg_entropy": 0.33,
}


def _safe_std(values, epsilon):
    return max(float(np.std(values)), float(epsilon))


def _shrunk_moments(values, pooled_values, strength, epsilon):
    """Shrink small-branch moments toward the pooled reference."""
    values = np.asarray(values, dtype=np.float64)
    pooled_values = np.asarray(pooled_values, dtype=np.float64)
    if values.size == 0 or pooled_values.size == 0:
        raise ValueError("Moment estimation received an empty sample")
    branch_mean = float(values.mean())
    branch_var = float(values.var())
    pooled_mean = float(pooled_values.mean())
    pooled_var = float(pooled_values.var())
    strength = max(float(strength), 0.0)
    weight = float(values.size / (values.size + strength))
    mean = weight * branch_mean + (1.0 - weight) * pooled_mean
    variance = (
        weight * (branch_var + (branch_mean - mean) ** 2)
        + (1.0 - weight) * (pooled_var + (pooled_mean - mean) ** 2)
    )
    return {
        "mean": float(mean),
        "std": max(float(np.sqrt(max(variance, 0.0))), float(epsilon)),
        "raw_mean": branch_mean,
        "raw_std": _safe_std(values, epsilon),
        "shrinkage_weight": weight,
    }


def _rate(values):
    values = list(values)
    if not values:
        return None
    return float(np.mean(values))


def _normalise_score_weights(raw):
    """Validate and normalise a non-negative evidence-weight mapping."""
    if raw is None:
        raw = DEFAULT_CHILD_EVIDENCE_WEIGHTS
    if not isinstance(raw, dict) or not raw:
        raise ValueError(
            "calibration.child_score_weights must be a non-empty mapping"
        )
    weights = {str(name): float(weight) for name, weight in raw.items()}
    if any(weight < 0.0 for weight in weights.values()):
        raise ValueError("Child evidence weights must be non-negative")
    total = float(sum(weights.values()))
    if total <= 0.0:
        raise ValueError("At least one child evidence weight must be positive")
    return {
        name: float(weight / total)
        for name, weight in weights.items()
        if weight > 0.0
    }


def _fpr_at_95_tpr(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores)
    eligible = fpr[tpr >= 0.95]
    return 1.0 if eligible.size == 0 else float(eligible.min())


def compute_branch_statistics(
    records,
    hier_meta,
    root_score_name="parent_score",
    child_score_weights=None,
    shrinkage_strength=20.0,
    epsilon=1e-6,
):
    """Estimate branch moments without letting unknown counts move the child reference.

    Root evidence treats known and intra-unknown images as in-taxonomy
    positives. Child evidence is standardised from correctly routed known
    images only, so its zero point means "typical known member" and is not
    changed by the number of unknown validation images.
    """
    child_score_weights = _normalise_score_weights(child_score_weights)
    pooled_root_records = [
        record
        for record in records
        if record["status"] in {"known", "intra"}
        and record["true_parent"] == record["pred_parent"]
    ]
    pooled_child_records = [
        record
        for record in records
        if record["status"] == "known"
        and record["true_parent"] == record["pred_parent"]
    ]
    pooled_root_values = [
        record[root_score_name] for record in pooled_root_records
    ]
    pooled_child_values = {
        score_name: [
            record[score_name] for record in pooled_child_records
        ]
        for score_name in child_score_weights
    }
    statistics = {}
    for parent_id, parent_name in enumerate(hier_meta["parent_names"]):
        children_by_parent = hier_meta.get("children_by_parent")
        if children_by_parent is None:
            child_count = None
        else:
            child_count = len(children_by_parent[parent_id])
        # Margin and entropy are structurally constant for a one-leaf branch
        # and therefore contain no novelty information.  Let such branches
        # use absolute child cosine only instead of injecting artificial
        # negative z-scores from pooled multi-leaf statistics.
        branch_child_weights = dict(child_score_weights)
        if child_count == 1:
            if "child_score" not in child_score_weights:
                raise ValueError(
                    "Singleton branches require child_score in "
                    "calibration.child_score_weights"
                )
            branch_child_weights = {"child_score": 1.0}
        root_selected = [
            record
            for record in records
            if record["status"] in {"known", "intra"}
            and record["true_parent"] == parent_id
            and record["pred_parent"] == parent_id
        ]
        child_selected = [
            record
            for record in records
            if record["status"] == "known"
            and record["true_parent"] == parent_id
            and record["pred_parent"] == parent_id
        ]
        if not root_selected:
            raise RuntimeError(
                "No correctly routed val_known/val_intra samples for '{}'; "
                "branch calibration is impossible.".format(parent_name)
            )
        if not child_selected:
            raise RuntimeError(
                "No correctly routed val_known samples for '{}'; child "
                "calibration is impossible.".format(parent_name)
            )
        if any(root_score_name not in record for record in root_selected):
            raise KeyError(
                "Validation records do not contain root evidence '{}'"
                .format(root_score_name)
            )
        missing_child = [
            score_name
            for score_name in child_score_weights
            if any(score_name not in record for record in child_selected)
        ]
        if missing_child:
            raise KeyError(
                "Validation records lack child evidence: {}".format(
                    sorted(missing_child)
                )
            )

        root_values = [record[root_score_name] for record in root_selected]
        child_evidence_stats = {}
        for score_name in child_score_weights:
            values = [record[score_name] for record in child_selected]
            moments = _shrunk_moments(
                values,
                pooled_child_values[score_name],
                shrinkage_strength,
                epsilon,
            )
            moments["sample_count"] = len(values)
            child_evidence_stats[score_name] = moments
        root_moments = _shrunk_moments(
            root_values,
            pooled_root_values,
            shrinkage_strength,
            epsilon,
        )
        root_mean = root_moments["mean"]
        root_std = root_moments["std"]
        alias_stats = child_evidence_stats.get(
            "child_score", next(iter(child_evidence_stats.values()))
        )
        statistics[parent_name] = {
            "root_score_name": root_score_name,
            "root_evidence_mean": root_mean,
            "root_evidence_std": root_std,
            # Backward-compatible aliases for existing analysis notebooks.
            "parent_mean": root_mean,
            "parent_std": root_std,
            "root_raw_mean": root_moments["raw_mean"],
            "root_raw_std": root_moments["raw_std"],
            "root_shrinkage_weight": root_moments["shrinkage_weight"],
            "child_evidence_stats": child_evidence_stats,
            "child_score_weights": branch_child_weights,
            "known_child_count": child_count,
            "child_mean": float(alias_stats["mean"]),
            "child_std": float(alias_stats["std"]),
            "statistics_sample_count": len(root_selected),
            "statistics_known_count": sum(
                record["status"] == "known" for record in root_selected
            ),
            "statistics_intra_count": sum(
                record["status"] == "intra" for record in root_selected
            ),
            "child_statistics_known_count": len(child_selected),
        }
    return statistics


def add_standardized_evidence(
    records,
    hier_meta,
    statistics,
    root_score_name="parent_score",
    child_score_weights=None,
):
    """Attach branch-normalised root/child evidence and their gap."""
    child_score_weights = _normalise_score_weights(child_score_weights)
    output = []
    for source in records:
        record = copy.deepcopy(source)
        parent_name = hier_meta["parent_names"][record["pred_parent"]]
        params = statistics[parent_name]
        branch_child_weights = _normalise_score_weights(
            params.get("child_score_weights", child_score_weights)
        )
        z_parent = (
            float(record[root_score_name])
            - float(params["root_evidence_mean"])
        ) / max(float(params["root_evidence_std"]), 1e-12)
        child_components = {}
        z_child = 0.0
        for score_name, weight in branch_child_weights.items():
            score_stats = params["child_evidence_stats"][score_name]
            component = (
                float(record[score_name]) - float(score_stats["mean"])
            ) / max(float(score_stats["std"]), 1e-12)
            child_components[score_name] = float(component)
            z_child += float(weight) * float(component)
        record["root_score_name"] = root_score_name
        record["root_evidence"] = float(record[root_score_name])
        record["z_parent"] = float(z_parent)
        record["z_child"] = float(z_child)
        record["child_knownness"] = float(z_child)
        record["child_evidence_components"] = child_components
        record["gap"] = float(z_parent - z_child)
        output.append(record)
    return output


def select_root_evidence(
    records,
    hier_meta,
    candidates,
    child_score_weights=None,
    shrinkage_strength=20.0,
    epsilon=1e-6,
    target_tpr=0.85,
):
    """Select one validation-supported root score; never inspect test data."""
    labels = np.asarray([
        0 if record["status"] == "extra" else 1 for record in records
    ], dtype=np.int64)
    if np.unique(labels).size != 2:
        raise ValueError("Root evidence selection needs in-taxa and extra data")

    results = {}
    viable = []
    for score_name in candidates:
        if not all(score_name in record for record in records):
            results[score_name] = {"available": False}
            continue
        statistics = compute_branch_statistics(
            records,
            hier_meta,
            root_score_name=score_name,
            child_score_weights=child_score_weights,
            shrinkage_strength=shrinkage_strength,
            epsilon=epsilon,
        )
        standardized = add_standardized_evidence(
            records,
            hier_meta,
            statistics,
            root_score_name=score_name,
            child_score_weights=child_score_weights,
        )
        scores = np.asarray(
            [record["z_parent"] for record in standardized],
            dtype=np.float64,
        )
        auc = float(roc_auc_score(labels, scores))
        fpr95 = _fpr_at_95_tpr(labels, scores)
        fpr, tpr, _ = roc_curve(labels, scores)
        eligible = fpr[tpr >= float(target_tpr)]
        fpr_at_target = (
            1.0 if eligible.size == 0 else float(eligible.min())
        )
        results[score_name] = {
            "available": True,
            "validation_auroc": auc,
            "validation_fpr95": fpr95,
            "validation_target_tpr": float(target_tpr),
            "validation_fpr_at_target_tpr": fpr_at_target,
        }
        viable.append(
            (-fpr_at_target, auc, -fpr95, score_name, statistics, standardized)
        )

    if not viable:
        raise RuntimeError(
            "None of calibration.root_score_candidates is present in the "
            "validation records"
        )
    viable.sort(
        key=lambda item: (item[0], item[1], item[2], item[3]),
        reverse=True,
    )
    _, _, _, selected_name, statistics, standardized = viable[0]
    return selected_name, statistics, standardized, results


def _candidate_line(values, points):
    values = np.asarray(list(values), dtype=np.float64)
    if values.size == 0:
        raise ValueError("Cannot build a threshold grid from no values")
    lower = float(values.min()) - 1e-6
    upper = float(values.max()) + 1e-6
    if lower == upper:
        return np.asarray([lower, upper + 1e-6])
    return np.linspace(lower, upper, int(points))


def _select_operating_point(
    candidates,
    risk_key,
    risk_limit,
    coverage_key,
    floor,
    policy="balanced",
):
    """Choose either a coverage-balanced or strictly risk-first point."""
    policy = str(policy).lower()
    if policy not in {"balanced", "safe"}:
        raise ValueError("selection policy must be 'balanced' or 'safe'")

    if policy == "safe":
        risk_feasible = [
            item
            for item in candidates
            if item[risk_key] <= risk_limit + 1e-12
        ]
        if risk_feasible:
            selected = max(
                risk_feasible,
                key=lambda item: (
                    item[coverage_key],
                    -item[risk_key],
                    item["conservatism"],
                ),
            )
            reason = "risk_constrained_maximum_coverage"
        else:
            selected = min(
                candidates,
                key=lambda item: (
                    item[risk_key],
                    -item[coverage_key],
                    -item["conservatism"],
                ),
            )
            reason = "minimum_available_risk"
    else:
        selected, reason = _select_balanced_point(
            candidates,
            risk_key,
            risk_limit,
            coverage_key,
            floor,
        )

    output = dict(selected)
    output.pop("conservatism", None)
    output["selection_reason"] = reason
    output["selection_policy"] = policy
    output["risk_constraint_satisfied"] = bool(
        output[risk_key] <= risk_limit + 1e-12
    )
    output["coverage_constraint_satisfied"] = bool(
        output[coverage_key] >= floor - 1e-12
    )
    return output


def _select_balanced_point(
    candidates, risk_key, risk_limit, coverage_key, floor
):
    """Preserve v2's coverage-guarded selection for balanced reporting."""
    both = [
        item for item in candidates
        if item[risk_key] <= risk_limit + 1e-12
        and item[coverage_key] >= floor - 1e-12
    ]
    if both:
        selected = max(
            both,
            key=lambda item: (
                item[coverage_key], -item[risk_key], item["conservatism"]
            ),
        )
        reason = "both_constraints_satisfied"
    else:
        coverage_feasible = [
            item for item in candidates
            if item[coverage_key] >= floor - 1e-12
        ]
        if coverage_feasible:
            selected = min(
                coverage_feasible,
                key=lambda item: (
                    item[risk_key], -item[coverage_key],
                    -item["conservatism"]
                ),
            )
            reason = "coverage_preserved_risk_constraint_relaxed"
        else:
            selected = max(
                candidates,
                key=lambda item: (
                    item[coverage_key], -item[risk_key],
                    item["conservatism"]
                ),
            )
            reason = "maximum_available_coverage"
    return selected, reason


def _prefix_constraint_status(result, prefix):
    """Prevent root and child decision diagnostics from overwriting."""
    output = copy.deepcopy(result)
    for key in (
        "selection_reason",
        "risk_constraint_satisfied",
        "coverage_constraint_satisfied",
    ):
        if key in output:
            output["{}_{}".format(prefix, key)] = output.pop(key)
    return output


def search_root_threshold(
    positive_records,
    negative_records,
    far_limit,
    points,
    min_correct_acceptance,
    selection_policy="balanced",
):
    """Search a root gate with a minimum correct-parent acceptance guard."""
    if not positive_records:
        raise ValueError("Root calibration needs positive in-taxonomy samples")
    if not negative_records:
        return None

    positive_scores = np.asarray([
        record["z_parent"]
        for record in positive_records
        if record["pred_parent"] == record["true_parent"]
    ], dtype=np.float64)
    negative_scores = np.asarray(
        [record["z_parent"] for record in negative_records],
        dtype=np.float64,
    )
    if positive_scores.size == 0:
        return None
    thresholds = _candidate_line(
        np.concatenate([positive_scores, negative_scores]), points
    )
    denominator = float(len(positive_records))
    candidates = []
    for threshold in thresholds:
        candidates.append({
            "tau_root": float(threshold),
            "root_correct_parent_acceptance": float(
                np.sum(positive_scores >= threshold) / denominator
            ),
            "root_false_acceptance": float(
                np.mean(negative_scores >= threshold)
            ),
            "root_positive_count": len(positive_records),
            "root_correctly_routed_count": int(positive_scores.size),
            "root_negative_count": len(negative_records),
            "conservatism": float(threshold),
        })
    return _select_operating_point(
        candidates,
        risk_key="root_false_acceptance",
        risk_limit=float(far_limit),
        coverage_key="root_correct_parent_acceptance",
        floor=float(min_correct_acceptance),
        policy=selection_policy,
    )


def _root_pass(record, branch_parameters):
    parent_name = branch_parameters["parent_names"][record["pred_parent"]]
    threshold = branch_parameters["branches"][parent_name]["tau_root"]
    return record["z_parent"] >= threshold


def search_child_thresholds(
    positive_records,
    negative_records,
    branch_parameters,
    over_specification_limit,
    min_known_correct_acceptance,
    child_points=121,
    gap_points=161,
    selection_policy="balanced",
):
    """Search child/gap gates without allowing reject-all to win silently."""
    if not positive_records:
        raise ValueError("Child calibration needs known validation samples")
    if not negative_records:
        return None

    positive_child = np.asarray(
        [record["z_child"] for record in positive_records], dtype=np.float64
    )
    positive_gap = np.asarray(
        [record["gap"] for record in positive_records], dtype=np.float64
    )
    positive_base = np.asarray([
        record["pred_parent"] == record["true_parent"]
        and record["pred_leaf"] == record["true_leaf"]
        and _root_pass(record, branch_parameters)
        for record in positive_records
    ], dtype=bool)
    negative_child = np.asarray(
        [record["z_child"] for record in negative_records], dtype=np.float64
    )
    negative_gap = np.asarray(
        [record["gap"] for record in negative_records], dtype=np.float64
    )
    negative_base = np.asarray([
        _root_pass(record, branch_parameters)
        for record in negative_records
    ], dtype=bool)

    all_child = np.concatenate([positive_child, negative_child])
    all_gap = np.concatenate([positive_gap, negative_gap])
    child_low = min(-3.0, float(all_child.min()) - 1e-6)
    child_high = max(3.0, float(all_child.max()) + 1e-6)
    gap_low = min(-3.0, float(all_gap.min()) - 1e-6)
    gap_high = max(5.0, float(all_gap.max()) + 1e-6)
    child_candidates = np.linspace(child_low, child_high, int(child_points))
    gap_candidates = np.linspace(gap_low, gap_high, int(gap_points))
    child_candidates = np.unique(np.concatenate([
        child_candidates,
        [all_child.min() - 1e-6, all_child.max() + 1e-6],
    ]))
    gap_candidates = np.unique(np.concatenate([
        gap_candidates,
        [all_gap.min() - 1e-6, all_gap.max() + 1e-6],
    ]))

    candidates = []
    for tau_child in child_candidates:
        positive_accept = (
            positive_base[:, None]
            & (positive_child[:, None] >= tau_child)
            & (positive_gap[:, None] <= gap_candidates[None, :])
        )
        negative_accept = (
            negative_base[:, None]
            & (negative_child[:, None] >= tau_child)
            & (negative_gap[:, None] <= gap_candidates[None, :])
        )
        correct_rates = positive_accept.mean(axis=0)
        osers = negative_accept.mean(axis=0)
        for index, tau_gap in enumerate(gap_candidates):
            candidates.append({
                "tau_child": float(tau_child),
                "tau_gap": float(tau_gap),
                "known_correct_leaf_acceptance": float(correct_rates[index]),
                "intra_over_specification_rate": float(osers[index]),
                "child_positive_count": len(positive_records),
                "child_negative_count": len(negative_records),
                "conservatism": float(tau_child - tau_gap),
            })

    return _select_operating_point(
        candidates,
        risk_key="intra_over_specification_rate",
        risk_limit=float(over_specification_limit),
        coverage_key="known_correct_leaf_acceptance",
        floor=float(min_known_correct_acceptance),
        policy=selection_policy,
    )


def _validation_operating_point(records, calibration):
    """Summarise the validation point before final test is allowed."""
    decisions = []
    root_score_name = calibration["metadata"]["root_score_name"]
    for record in records:
        parent_name = calibration["parent_names"][record["pred_parent"]]
        params = calibration["branches"][parent_name]
        # ``records`` were standardised with the same branch statistics and
        # fixed child-evidence ensemble used to build this calibration.
        z_parent = float(record["z_parent"])
        z_child = float(record["z_child"])
        gap = z_parent - z_child
        if z_parent < float(params["tau_root"]):
            prediction_type = "global_unknown"
        elif (
            z_child < float(params["tau_child"])
            or gap > float(params["tau_gap"])
        ):
            prediction_type = "intra_unknown"
        else:
            prediction_type = "known"
        decisions.append((record, prediction_type))

    known = [item for item in decisions if item[0]["status"] == "known"]
    intra = [item for item in decisions if item[0]["status"] == "intra"]
    extra_records = [
        item for item in decisions if item[0]["status"] == "extra"
    ]
    correct_known = [
        decision == "known"
        and record["pred_parent"] == record["true_parent"]
        and record["pred_leaf"] == record["true_leaf"]
        for record, decision in known
    ]
    correct_intra = [
        decision == "intra_unknown"
        and record["pred_parent"] == record["true_parent"]
        for record, decision in intra
    ]
    correct_extra = [
        decision == "global_unknown" for _, decision in extra_records
    ]
    all_correct = correct_known + correct_intra + correct_extra
    return {
        "known_count": len(known),
        "known_leaf_coverage": _rate(
            decision == "known" for _, decision in known
        ),
        "known_end_to_end_leaf_accuracy": _rate(correct_known),
        "intra_count": len(intra),
        "intra_correct_fallback_rate": _rate(correct_intra),
        "intra_over_specification_rate": _rate(
            decision == "known" for _, decision in intra
        ),
        "extra_count": len(extra_records),
        "extra_false_parent_acceptance_rate": _rate(
            decision != "global_unknown" for _, decision in extra_records
        ),
        "deepest_reliable_taxon_accuracy": _rate(all_correct),
    }


def calibrate(records, hier_meta, cfg, checkpoint_path):
    calibration_cfg = cfg.get("calibration", {})
    selection_policy = str(
        calibration_cfg.get("selection_policy", "balanced")
    ).lower()
    if selection_policy not in {"balanced", "safe"}:
        raise ValueError(
            "calibration.selection_policy must be balanced or safe"
        )
    child_score_weights = _normalise_score_weights(
        calibration_cfg.get("child_score_weights")
    )
    far_limit = float(
        calibration_cfg.get("root_false_accept_limit", 0.05)
    )
    oser_limit = float(
        calibration_cfg.get("over_specification_limit", 0.05)
    )
    min_root_acceptance = float(
        calibration_cfg.get("min_root_correct_parent_acceptance", 0.85)
    )
    min_leaf_acceptance = float(
        calibration_cfg.get("min_known_correct_leaf_acceptance", 0.65)
    )
    for name, value in (
        ("root_false_accept_limit", far_limit),
        ("over_specification_limit", oser_limit),
        ("min_root_correct_parent_acceptance", min_root_acceptance),
        ("min_known_correct_leaf_acceptance", min_leaf_acceptance),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError("calibration.{} must be in [0, 1]".format(name))

    min_branch_extra = int(
        calibration_cfg.get("min_branch_extra_samples", 20)
    )
    min_branch_root_positive = int(
        calibration_cfg.get("min_branch_root_positive_samples", 30)
    )
    min_branch_intra = int(
        calibration_cfg.get("min_branch_intra_samples", 30)
    )
    min_branch_known = int(
        calibration_cfg.get("min_branch_known_samples", 10)
    )
    root_points = int(calibration_cfg.get("root_grid_points", 800))
    child_points = int(calibration_cfg.get("child_grid_points", 121))
    gap_points = int(calibration_cfg.get("gap_grid_points", 161))
    epsilon = float(calibration_cfg.get("std_epsilon", 1e-6))
    shrinkage_strength = float(
        calibration_cfg.get("moment_shrinkage_strength", 20.0)
    )
    if shrinkage_strength < 0.0:
        raise ValueError(
            "calibration.moment_shrinkage_strength must be non-negative"
        )
    evidence_candidates = calibration_cfg.get(
        "root_score_candidates", list(DEFAULT_ROOT_EVIDENCE)
    )
    if isinstance(evidence_candidates, str):
        evidence_candidates = [evidence_candidates]
    evidence_candidates = [str(value) for value in evidence_candidates]

    root_score_name, statistics, records, evidence_results = (
        select_root_evidence(
            records,
            hier_meta,
            evidence_candidates,
            child_score_weights=child_score_weights,
            shrinkage_strength=shrinkage_strength,
            epsilon=epsilon,
            target_tpr=min_root_acceptance,
        )
    )
    positives_all = [
        record for record in records
        if record["status"] in {"known", "intra"}
    ]
    known_all = [record for record in records if record["status"] == "known"]
    intra_all = [record for record in records if record["status"] == "intra"]
    extra_all = [record for record in records if record["status"] == "extra"]

    pooled_root = search_root_threshold(
        positives_all,
        extra_all,
        far_limit,
        root_points,
        min_root_acceptance,
        selection_policy=selection_policy,
    )
    if pooled_root is None:
        raise RuntimeError("val_extra is required for root calibration")

    output = {
        "schema_version": 3,
        "metadata": {
            "created_utc": datetime.datetime.utcnow().isoformat() + "Z",
            "checkpoint": os.path.abspath(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "calibration_splits": list(CALIBRATION_SPLITS),
            "selection_policy": selection_policy,
            "root_score_name": root_score_name,
            "root_score_candidates": evidence_results,
            "child_score_name": "standardized_weighted_ensemble",
            "child_score_weights": child_score_weights,
            "child_statistics_source": "correctly_routed_val_known_only",
            "moment_shrinkage_strength": shrinkage_strength,
            "root_false_accept_limit": far_limit,
            "over_specification_limit": oser_limit,
            "min_root_correct_parent_acceptance": min_root_acceptance,
            "min_known_correct_leaf_acceptance": min_leaf_acceptance,
            "rate_unit": "fraction_in_[0,1]",
            "branch_specific": bool(
                calibration_cfg.get("branch_specific", True)
            ),
            "min_branch_extra_samples": min_branch_extra,
            "min_branch_root_positive_samples": min_branch_root_positive,
            "min_branch_intra_samples": min_branch_intra,
            "min_branch_known_samples": min_branch_known,
            "record_count": len(records),
        },
        "parent_names": list(hier_meta["parent_names"]),
        "leaf_names": list(hier_meta["leaf_names"]),
        "pooled": {"root": pooled_root},
        "branches": {},
    }

    for parent_id, parent_name in enumerate(hier_meta["parent_names"]):
        params = copy.deepcopy(statistics[parent_name])
        branch_positive = [
            record for record in positives_all
            if record["true_parent"] == parent_id
        ]
        branch_extra = [
            record for record in extra_all
            if record["pred_parent"] == parent_id
        ]
        branch_root = None
        if output["metadata"]["branch_specific"] and (
            len(branch_extra) >= min_branch_extra
            and len(branch_positive) >= min_branch_root_positive
        ):
            branch_root = search_root_threshold(
                branch_positive,
                branch_extra,
                far_limit,
                root_points,
                min_root_acceptance,
                selection_policy=selection_policy,
            )
        if branch_root is None:
            branch_root = copy.deepcopy(pooled_root)
            params["root_threshold_source"] = "pooled"
            params["root_constraint_scope"] = "pooled_validation"
        else:
            params["root_threshold_source"] = "branch_specific"
            params["root_constraint_scope"] = "branch_validation"
        params.update(_prefix_constraint_status(branch_root, "root"))
        output["branches"][parent_name] = params

    root_context = {
        "parent_names": output["parent_names"],
        "branches": output["branches"],
    }
    pooled_child = search_child_thresholds(
        known_all,
        intra_all,
        root_context,
        oser_limit,
        min_leaf_acceptance,
        child_points,
        gap_points,
        selection_policy=selection_policy,
    )
    if pooled_child is None:
        raise RuntimeError("val_intra is required for child calibration")
    output["pooled"]["child"] = pooled_child

    for parent_id, parent_name in enumerate(hier_meta["parent_names"]):
        branch_known = [
            record for record in known_all
            if record["true_parent"] == parent_id
        ]
        branch_intra = [
            record for record in intra_all
            if record["pred_parent"] == parent_id
        ]
        branch_child = None
        if output["metadata"]["branch_specific"] and (
            len(branch_intra) >= min_branch_intra
            and len(branch_known) >= min_branch_known
        ):
            branch_child = search_child_thresholds(
                branch_known,
                branch_intra,
                root_context,
                oser_limit,
                min_leaf_acceptance,
                child_points,
                gap_points,
                selection_policy=selection_policy,
            )
        if branch_child is None:
            branch_child = copy.deepcopy(pooled_child)
            source = "pooled"
            constraint_scope = "pooled_validation"
        else:
            source = "branch_specific"
            constraint_scope = "branch_validation"
        output["branches"][parent_name].update(
            _prefix_constraint_status(branch_child, "child")
        )
        output["branches"][parent_name]["child_threshold_source"] = source
        output["branches"][parent_name][
            "child_constraint_scope"
        ] = constraint_scope
        output["branches"][parent_name]["threshold_source"] = source

    output["metadata"]["constraint_violations"] = {
        "root_risk": [
            name for name, params in output["branches"].items()
            if not params.get("root_risk_constraint_satisfied", False)
        ],
        "child_risk": [
            name for name, params in output["branches"].items()
            if not params.get("child_risk_constraint_satisfied", False)
        ],
        "root_coverage": [
            name for name, params in output["branches"].items()
            if not params.get("root_coverage_constraint_satisfied", False)
        ],
        "child_coverage": [
            name for name, params in output["branches"].items()
            if not params.get("child_coverage_constraint_satisfied", False)
        ],
    }
    output["metadata"]["validation_operating_point"] = (
        _validation_operating_point(records, output)
    )
    return output, records


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calibrate TaxoSafe thresholds using validation splits"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--trial", default="1")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--policy",
        choices=("balanced", "safe"),
        default=None,
        help=(
            "Override calibration.selection_policy. 'safe' honours the "
            "risk limit first; 'balanced' preserves the coverage floor."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg, config_path = load_yaml(args.config)
    del config_path
    if args.policy is not None:
        cfg.setdefault("calibration", {})["selection_policy"] = args.policy
    os.chdir(PROJECT_ROOT)
    if any("test" in split for split in CALIBRATION_SPLITS):
        raise RuntimeError("Calibration is not allowed to load test splits")

    run_dir = resolve_run_dir(cfg, args.trial, PROJECT_ROOT, args.run_dir)
    checkpoint_path = os.path.abspath(
        args.checkpoint or os.path.join(run_dir, "ckpt", "best.pth")
    )
    default_name = (
        "thresholds.json"
        if args.policy is None
        else "thresholds_{}.json".format(args.policy)
    )
    output_path = os.path.abspath(
        args.output
        or os.path.join(run_dir, "calibration", default_name)
    )
    output_stem = os.path.splitext(os.path.basename(output_path))[0]
    score_suffix = (
        output_stem[len("thresholds"):]
        if output_stem.startswith("thresholds")
        else "_{}".format(output_stem)
    )
    score_path = os.path.join(
        os.path.dirname(output_path),
        "validation_scores{}.jsonl".format(score_suffix),
    )

    if (
        cfg.get("model", {}).get("prec", "fp16") == "fp16"
        and not torch.cuda.is_available()
    ):
        raise SystemExit(
            "TaxoSafe fp16 calibration requires CUDA; use a GPU or a "
            "separately validated fp32 configuration"
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, loaders, hier_meta = load_model_and_data(
        cfg, CALIBRATION_SPLITS, checkpoint_path, device
    )
    records = []
    records.extend(collect_score_records(
        model, loaders["val_known"], "known", hier_meta, device
    ))
    records.extend(collect_score_records(
        model, loaders["val_intra"], "intra", hier_meta, device
    ))
    records.extend(collect_score_records(
        model, loaders["val_extra"], "extra", hier_meta, device
    ))

    calibration, records = calibrate(
        records, hier_meta, cfg, checkpoint_path
    )
    write_json(output_path, calibration)
    write_jsonl(score_path, records)

    print("Calibration complete")
    print("thresholds: {}".format(output_path))
    print("validation scores: {}".format(score_path))
    print("selected root evidence: {}".format(
        calibration["metadata"]["root_score_name"]
    ))
    print("selection policy: {}".format(
        calibration["metadata"]["selection_policy"]
    ))
    for parent_name in calibration["parent_names"]:
        params = calibration["branches"][parent_name]
        print(
            "{}: root={:.4f} ({}), child={:.4f}, gap={:.4f} ({})"
            .format(
                parent_name,
                params["tau_root"],
                params["root_threshold_source"],
                params["tau_child"],
                params["tau_gap"],
                params["child_threshold_source"],
            )
        )
        if not params.get("root_risk_constraint_satisfied", False):
            print(
                "WARNING: {} root FAR constraint is not satisfied at this "
                "operating point.".format(parent_name)
            )
        if not params.get("root_coverage_constraint_satisfied", False):
            print(
                "WARNING: {} root coverage floor was infeasible.".format(
                    parent_name
                )
            )
        if not params.get("child_risk_constraint_satisfied", False):
            print(
                "WARNING: {} child OSER constraint is not satisfied at this "
                "operating point.".format(parent_name)
            )
        if not params.get("child_coverage_constraint_satisfied", False):
            print(
                "WARNING: {} child coverage floor was infeasible.".format(
                    parent_name
                )
            )
    print("validation operating point:")
    print(json.dumps(
        calibration["metadata"]["validation_operating_point"],
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
