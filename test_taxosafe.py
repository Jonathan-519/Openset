"""Final three-level TaxoSafe inference and open-set evaluation."""

import argparse
import copy
import json
import os

import numpy as np
import torch

from metrics_open import evaluate_open_set, risk_specificity_point
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
TEST_SPLITS = ("test_known", "test_intra", "test_extra")


def load_calibration(path):
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "Calibration file does not exist: {}".format(path)
        )
    with open(path, "r", encoding="utf-8") as stream:
        calibration = json.load(stream)
    required = {"metadata", "parent_names", "leaf_names", "branches"}
    missing = required.difference(calibration)
    if missing:
        raise KeyError(
            "thresholds.json is missing keys: {}".format(sorted(missing))
        )
    if any(
        "test" in split
        for split in calibration["metadata"].get("calibration_splits", [])
    ):
        raise RuntimeError(
            "Rejected thresholds.json because it was calibrated on test data"
        )
    return calibration, path


def predict_taxosafe(
    parent_cosine,
    leaf_cosine,
    hier_meta,
    calibration,
):
    """Legacy matrix API for parent-score calibrations.

    The final test path uses :func:`attach_predictions`, which consumes saved
    evidence records and therefore supports every calibrated root score.
    """
    if parent_cosine.ndim != 2 or leaf_cosine.ndim != 2:
        raise ValueError("parent_cosine and leaf_cosine must be matrices")
    if parent_cosine.shape[0] != leaf_cosine.shape[0]:
        raise ValueError("Parent and leaf score batches must have equal size")
    if parent_cosine.shape[1] != len(hier_meta["parent_names"]):
        raise ValueError("Unexpected number of parent scores")
    if leaf_cosine.shape[1] != len(hier_meta["leaf_names"]):
        raise ValueError("Unexpected number of leaf scores")

    if int(calibration.get("schema_version", 1)) >= 3:
        raise ValueError(
            "TaxoSafe v3 child evidence needs margin/entropy records. Use "
            "attach_predictions on collect_score_records output."
        )

    root_score_name = calibration.get("metadata", {}).get(
        "root_score_name", "parent_score"
    )
    if root_score_name not in {"parent_score", "parent_margin"}:
        raise ValueError(
            "Matrix-only predict_taxosafe cannot reconstruct '{}'. Use "
            "attach_predictions on collect_score_records output.".format(
                root_score_name
            )
        )

    temporary_records = []
    for row in range(parent_cosine.shape[0]):
        parent_id = int(parent_cosine[row].argmax().item())
        parent_score = float(parent_cosine[row, parent_id].item())
        if parent_cosine.shape[1] >= 2:
            top_two = torch.topk(parent_cosine[row], 2).values
            parent_margin = float((top_two[0] - top_two[1]).item())
        else:
            parent_margin = 0.0

        children = torch.as_tensor(
            hier_meta["children_by_parent"][parent_id],
            dtype=torch.long,
            device=leaf_cosine.device,
        )
        local_scores = leaf_cosine[row, children]
        local_index = int(local_scores.argmax().item())
        leaf_id = int(children[local_index].item())
        child_score = float(local_scores[local_index].item())
        temporary_records.append({
            "pred_parent": parent_id,
            "pred_leaf": leaf_id,
            "parent_score": parent_score,
            "parent_margin": parent_margin,
            "child_score": child_score,
        })
    return [
        _predict_record(record, hier_meta, calibration)
        for record in temporary_records
    ]


def _child_knownness(record, params, calibration):
    """Compute the fixed validation-defined child evidence ensemble."""
    metadata = calibration.get("metadata", {})
    weights = params.get(
        "child_score_weights", metadata.get("child_score_weights")
    )
    statistics = params.get("child_evidence_stats")
    if not weights or not statistics:
        child_std = max(float(params["child_std"]), 1e-8)
        value = (
            float(record["child_score"]) - float(params["child_mean"])
        ) / child_std
        return float(value), {"child_score": float(value)}

    components = {}
    knownness = 0.0
    weight_sum = float(sum(float(value) for value in weights.values()))
    if weight_sum <= 0.0:
        raise ValueError("Calibration child evidence weights sum to zero")
    for score_name, raw_weight in weights.items():
        if score_name not in record:
            raise KeyError(
                "Test record lacks calibrated child evidence '{}'".format(
                    score_name
                )
            )
        if score_name not in statistics:
            raise KeyError(
                "Branch statistics lack child evidence '{}'".format(
                    score_name
                )
            )
        score_stats = statistics[score_name]
        component = (
            float(record[score_name]) - float(score_stats["mean"])
        ) / max(float(score_stats["std"]), 1e-8)
        components[score_name] = float(component)
        knownness += float(raw_weight) * float(component) / weight_sum
    return float(knownness), components


def _predict_record(record, hier_meta, calibration):
    """Apply the frozen three-level decision policy to one score record."""
    parent_id = int(record["pred_parent"])
    leaf_id = int(record["pred_leaf"])
    parent_name = hier_meta["parent_names"][parent_id]
    params = calibration["branches"][parent_name]
    children = torch.as_tensor(
        hier_meta["children_by_parent"][parent_id]
    ).detach().cpu().reshape(-1).tolist()
    children = [int(value) for value in children]
    if leaf_id not in children:
        raise RuntimeError(
            "Candidate leaf {} is outside predicted parent {}".format(
                leaf_id, parent_name
            )
        )

    root_score_name = calibration.get("metadata", {}).get(
        "root_score_name", params.get("root_score_name", "parent_score")
    )
    if root_score_name not in record:
        raise KeyError(
            "Test score record lacks calibrated root evidence '{}'".format(
                root_score_name
            )
        )
    root_value = float(record[root_score_name])
    root_mean = float(
        params["root_evidence_mean"]
        if "root_evidence_mean" in params
        else params["parent_mean"]
    )
    root_std = max(float(
        params["root_evidence_std"]
        if "root_evidence_std" in params
        else params["parent_std"]
    ), 1e-8)
    child_score = float(record["child_score"])
    z_parent = (root_value - root_mean) / root_std
    z_child, child_components = _child_knownness(
        record, params, calibration
    )
    gap = z_parent - z_child

    common = {
        "candidate_parent": parent_id,
        "candidate_leaf": leaf_id,
        "root_score_name": root_score_name,
        "root_evidence": root_value,
        "parent_score": float(record["parent_score"]),
        "child_score": child_score,
        "z_parent": float(z_parent),
        "z_child": float(z_child),
        "child_knownness": float(z_child),
        "child_evidence_components": child_components,
        "gap": float(gap),
        "root_knownness_score": float(z_parent),
    }
    if z_parent < float(params["tau_root"]):
        return dict(
            common,
            prediction_type="global_unknown",
            parent=None,
            leaf=None,
        )
    if (
        z_child < float(params["tau_child"])
        or gap > float(params["tau_gap"])
    ):
        return dict(
            common,
            prediction_type="intra_unknown",
            parent=parent_id,
            leaf=None,
        )
    return dict(
        common,
        prediction_type="known",
        parent=parent_id,
        leaf=leaf_id,
    )


def attach_predictions(records, hier_meta, calibration):
    predictions = [
        _predict_record(record, hier_meta, calibration)
        for record in records
    ]
    if len(predictions) != len(records):
        raise RuntimeError("Prediction count mismatch")

    output = []
    for source, prediction in zip(records, predictions):
        record = dict(source)
        if prediction["candidate_parent"] != record["pred_parent"]:
            raise RuntimeError("Parent argmax changed after score collection")
        if prediction["candidate_leaf"] != record["pred_leaf"]:
            raise RuntimeError("Branch-restricted leaf argmax changed")
        record.update(prediction)
        parent = record["parent"]
        leaf = record["leaf"]
        record["parent_name"] = (
            None if parent is None else hier_meta["parent_names"][parent]
        )
        record["leaf_name"] = (
            None if leaf is None else hier_meta["leaf_names"][leaf]
        )
        candidate_parent = record["candidate_parent"]
        candidate_leaf = record["candidate_leaf"]
        record["candidate_parent_name"] = hier_meta["parent_names"][
            candidate_parent
        ]
        record["candidate_leaf_name"] = hier_meta["leaf_names"][
            candidate_leaf
        ]
        true_parent = record.get("true_parent")
        true_leaf = record.get("true_leaf")
        record["true_parent_name"] = (
            None
            if true_parent is None
            else hier_meta["parent_names"][true_parent]
        )
        record["true_leaf_name"] = (
            None
            if true_leaf is None
            else hier_meta["leaf_names"][true_leaf]
        )
        output.append(record)
    return output


def shifted_calibration(calibration, shift):
    """Increase specificity as shift grows; used only for the risk curve."""
    output = copy.deepcopy(calibration)
    shift = float(shift)
    for params in output["branches"].values():
        params["tau_root"] = float(params["tau_root"]) - shift
        params["tau_child"] = float(params["tau_child"]) - shift
        params["tau_gap"] = float(params["tau_gap"]) + shift
    return output


def build_risk_specificity_curve(
    raw_by_status,
    hier_meta,
    calibration,
    points=21,
):
    curve = []
    for shift in np.linspace(-2.0, 2.0, int(points)):
        shifted = shifted_calibration(calibration, float(shift))
        known = attach_predictions(
            raw_by_status["known"], hier_meta, shifted
        )
        intra = attach_predictions(
            raw_by_status["intra"], hier_meta, shifted
        )
        extra = attach_predictions(
            raw_by_status["extra"], hier_meta, shifted
        )
        metrics = evaluate_open_set(known, intra, extra)
        curve.append(risk_specificity_point(metrics, shift))
    return curve


def parse_args():
    parser = argparse.ArgumentParser(
        description="Final TaxoSafe open-set test"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--trial", default="1")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--thresholds", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing final metrics file",
    )
    parser.add_argument(
        "--allow-risk-violation",
        action="store_true",
        help=(
            "Run even when validation could not satisfy configured FAR/OSER "
            "constraints at the minimum coverage floor"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg, config_path = load_yaml(args.config)
    del config_path
    os.chdir(PROJECT_ROOT)

    # Fixed test-only tuple: validation data cannot be supplied to this entry.
    if any("val" in split for split in TEST_SPLITS):
        raise RuntimeError("Final testing is not allowed to load validation")

    run_dir = resolve_run_dir(
        cfg,
        args.trial,
        PROJECT_ROOT,
        args.run_dir,
    )
    checkpoint_path = os.path.abspath(
        args.checkpoint
        or os.path.join(run_dir, "ckpt", "best.pth")
    )
    thresholds_path = os.path.abspath(
        args.thresholds
        or os.path.join(run_dir, "calibration", "thresholds.json")
    )
    calibration, thresholds_path = load_calibration(thresholds_path)
    policy_name = str(
        calibration.get("metadata", {}).get("selection_policy", "legacy")
    )
    default_output_name = (
        "test_taxosafe"
        if policy_name in {"legacy", "balanced"}
        else "test_taxosafe_{}".format(policy_name)
    )
    output_dir = os.path.abspath(
        args.output_dir or os.path.join(run_dir, default_output_name)
    )
    metrics_path = os.path.join(output_dir, "metrics.json")
    predictions_path = os.path.join(output_dir, "predictions.jsonl")
    existing_outputs = [
        path for path in (metrics_path, predictions_path)
        if os.path.exists(path)
    ]
    if existing_outputs and not args.overwrite:
        raise FileExistsError(
            "Final test output already exists at {}. Refusing to overwrite; use "
            "--overwrite only when intentionally rerunning.".format(
                existing_outputs
            )
        )

    violations = calibration.get("metadata", {}).get(
        "constraint_violations", {}
    )
    risk_violation = bool(violations.get("root_risk")) or bool(
        violations.get("child_risk")
    )
    if risk_violation and not args.allow_risk_violation:
        raise RuntimeError(
            "Validation calibration constraints were not all satisfied: {}. "
            "Inspect thresholds.json and the printed validation operating "
            "point. If this is an intentional research run, rerun with "
            "--allow-risk-violation; do not describe it as a constrained "
            "safe operating point.".format(violations)
        )
    checkpoint_hash = sha256_file(checkpoint_path)
    expected_hash = calibration["metadata"].get("checkpoint_sha256")
    if expected_hash and checkpoint_hash != expected_hash:
        raise RuntimeError(
            "best.pth differs from the checkpoint used for calibration"
        )

    if not torch.cuda.is_available():
        raise SystemExit("TaxoSafe ViT-B/16 fp16 testing requires a CUDA GPU")
    device = torch.device("cuda")
    model, loaders, hier_meta = load_model_and_data(
        cfg,
        TEST_SPLITS,
        checkpoint_path,
        device,
    )
    if calibration["parent_names"] != list(hier_meta["parent_names"]):
        raise RuntimeError("Parent labels differ from thresholds.json")
    if calibration["leaf_names"] != list(hier_meta["leaf_names"]):
        raise RuntimeError("Leaf labels differ from thresholds.json")

    raw_by_status = {
        "known": collect_score_records(
            model,
            loaders["test_known"],
            "known",
            hier_meta,
            device,
            include_vectors=False,
        ),
        "intra": collect_score_records(
            model,
            loaders["test_intra"],
            "intra",
            hier_meta,
            device,
            include_vectors=False,
        ),
        "extra": collect_score_records(
            model,
            loaders["test_extra"],
            "extra",
            hier_meta,
            device,
            include_vectors=False,
        ),
    }

    known = attach_predictions(
        raw_by_status["known"], hier_meta, calibration
    )
    intra = attach_predictions(
        raw_by_status["intra"], hier_meta, calibration
    )
    extra = attach_predictions(
        raw_by_status["extra"], hier_meta, calibration
    )
    metrics = evaluate_open_set(known, intra, extra)
    curve_points = int(
        cfg.get("calibration", {}).get("risk_curve_points", 21)
    )
    metrics["open_set_hierarchical_risk_specificity_curve"] = (
        build_risk_specificity_curve(
            raw_by_status,
            hier_meta,
            calibration,
            points=curve_points,
        )
    )
    metrics["metadata"] = {
        "checkpoint": checkpoint_path,
        "checkpoint_sha256": checkpoint_hash,
        "thresholds": thresholds_path,
        "calibration_schema_version": calibration.get("schema_version", 1),
        "calibration_selection_policy": calibration.get(
            "metadata", {}
        ).get("selection_policy", "legacy"),
        "root_score_name": calibration.get("metadata", {}).get(
            "root_score_name", "parent_score"
        ),
        "child_score_name": calibration.get("metadata", {}).get(
            "child_score_name", "child_score"
        ),
        "calibration_validation_operating_point": calibration.get(
            "metadata", {}
        ).get("validation_operating_point"),
        "calibration_constraint_violations": violations,
        "risk_violation_override_used": bool(
            risk_violation and args.allow_risk_violation
        ),
        "test_splits": list(TEST_SPLITS),
        "rate_unit": "fraction_in_[0,1]",
    }

    all_records = known + intra + extra
    write_json(metrics_path, metrics)
    write_jsonl(
        predictions_path,
        all_records,
        drop_vector_fields=True,
    )

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("metrics: {}".format(metrics_path))
    print("predictions: {}".format(predictions_path))


if __name__ == "__main__":
    main()
