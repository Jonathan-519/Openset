"""Freeze TaxoSafe hierarchical partial-pooling calibration experiments.

The score is fixed to the train-selected simple cell (alpha=0, k=1, no local
scaling).  Only the coverage-threshold estimator changes.  Existing clean
checkpoints are reused; the old completed suite is never modified.
"""
import argparse
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import run_taxosafe_suite as suite


def variants():
    score = {"alphas": [0.0], "ks": [1], "local_scaling": False}
    return {
        # The preregistered candidate uses a data-adaptive but tuning-free
        # prior strength: median nonempty correct-leaf calibration count.
        "full": dict(score, threshold_mode="hierarchical_shrinkage",
                     threshold_shrinkage="auto"),
        "branch_min_reference": dict(score, threshold_mode="branch_min"),
        "parent_pooled": dict(score, threshold_mode="parent_pooled"),
        "leaf_conditional": dict(score, threshold_mode="leaf_conditional"),
        # Sensitivity only; never choose the best of these on test.
        "shrinkage_5": dict(score, threshold_mode="hierarchical_shrinkage",
                            threshold_shrinkage=5.0),
        "shrinkage_10": dict(score, threshold_mode="hierarchical_shrinkage",
                             threshold_shrinkage=10.0),
        "shrinkage_20": dict(score, threshold_mode="hierarchical_shrinkage",
                             threshold_shrinkage=20.0),
    }


def verified_training_source(source_suite, seed):
    source = suite.resolve(source_suite) / ("seed_" + str(seed))
    plan_path = source / "plan.json"
    original = suite.read(plan_path)
    if original["seed"] != seed or original["reuse_checkpoint"]:
        raise ValueError("Source must be an original completed clean-training plan")
    training = suite.steps(original, "train")[0]
    receipt_path = source / "receipts/train.json"
    receipt = suite.read(receipt_path)
    if receipt.get("plan_sha256") != suite.file_hash(plan_path):
        raise ValueError("Original training receipt/plan mismatch")
    if set(receipt.get("outputs_sha256", {})) != set(training["outputs"]):
        raise ValueError("Original training receipt output list mismatch")
    for name, expected in receipt["outputs_sha256"].items():
        path = suite.resolve(name)
        if not path.is_file() or suite.file_hash(path) != expected:
            raise ValueError("Original training output changed or missing: " + str(path))
    archive = suite.resolve(original["archived_training_config"])
    frozen_copy = suite.resolve(original["training_config"])
    if archive.read_bytes() != frozen_copy.read_bytes():
        raise ValueError("Archived training YAML differs from original frozen copy")
    cfg = yaml.safe_load(archive.read_text(encoding="utf-8-sig"))
    # Source code is intentionally changing in this experiment. Preserve and
    # verify the trained checkpoint plus all data/taxonomy lists instead of
    # pretending that the entire old source tree is unchanged.
    data_keys = ("train", "val_known", "val_intra", "val_extra", "test_known",
                 "test_intra", "test_extra", "oe_train", "hierarchy")
    for key in data_keys:
        path = suite.resolve(cfg["data"][key])
        portable = suite.portable(path)
        expected = original["inputs_sha256"].get(portable)
        if expected is None or not path.is_file() or suite.file_hash(path) != expected:
            raise ValueError("Original data/taxonomy input changed or missing: " + str(path))
    return original, plan_path, receipt_path, archive


def make_plan(source_suite, output_suite, seed):
    original, source_plan, train_receipt, archive = verified_training_source(source_suite, seed)
    original_variants = suite.VARIANTS
    try:
        suite.VARIANTS = variants()
        plan = suite.make_plan(
            archive,
            suite.resolve(output_suite) / ("seed_" + str(seed)),
            trial=original["trial"],
            run_dir=original["run_dir"],
        )
    finally:
        suite.VARIANTS = original_variants
    plan.update({
        "study_type": "post_factorial_hierarchical_partial_pooling_development",
        "primary_method": "full",
        "source_plan": suite.portable(source_plan),
        "factorial_conclusion": (
            "Score fixed before this run to alpha=0,k=1,local_scaling=false. "
            "Full is automatic hierarchical shrinkage; other threshold modes are ablations/sensitivity."),
        "protocol": (
            "Development-only because the current test set was already inspected. Full is fixed before "
            "execution. Report all seven variants and both profiles; never select a fixed shrinkage value "
            "on test. Root policy and checkpoint are shared. No conformal or population-risk guarantee."),
    })
    for path in (source_plan, train_receipt, Path(__file__)):
        plan["inputs_sha256"][suite.portable(path)] = suite.file_hash(path)
    suite.dump(suite.resolve(plan["suite"]) / "plan.json", plan)
    return plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-suite", default="runs/taxosafe_rs_paper")
    p.add_argument("--suite-dir", default="runs/taxosafe_partial_pooling_dev")
    p.add_argument("--seed", type=int, required=True, choices=(1, 2, 3))
    args = p.parse_args()
    plan = make_plan(args.source_suite, args.suite_dir, args.seed)
    print("Frozen development plan:", suite.resolve(plan["suite"]) / "plan.json")
    print("Reused checkpoint:", suite.resolve(plan["run_dir"]) / "ckpt/best.pth")
    print("Primary method: full (automatic hierarchical shrinkage)")
    print("Variants:", ", ".join(plan["variants"]))
    print("Training is omitted. Run memory, calibrate and test; report every variant.")


if __name__ == "__main__":
    main()
