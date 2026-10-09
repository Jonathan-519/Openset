"""Freeze the TaxoSafe source-balanced operating-point development run.

The checkpoint and child score are fixed from the earlier experiments.  This
run adds one validation-only child operating point between the permissive
coverage and conservative risk profiles.  The root always uses the frozen
coverage decision for the balanced profile.
"""
import argparse
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import run_taxosafe_suite as suite
from tools.plan_taxosafe_partial_pooling import verified_training_source


def variants():
    return {
        "full": {
            "alphas": [0.0],
            "ks": [1],
            "local_scaling": False,
            "threshold_mode": "hierarchical_shrinkage",
            "threshold_shrinkage": "auto",
            "balanced_profile": True,
        },
    }


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

    # The base extension keeps coverage as its historical default.  Freeze the
    # new primary profile in both the generated config and the hashed plan.
    config_path = suite.resolve(plan["variants"]["full"]["config"])
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["visual_support"]["primary_profile"] = "balanced"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    plan["inputs_sha256"][suite.portable(config_path)] = suite.file_hash(config_path)
    plan.update({
        "study_type": "post_partial_pooling_source_balanced_development",
        "primary_method": "full",
        "primary_profile": "balanced",
        "test_profiles": "all",
        "source_plan": suite.portable(source_plan),
        "development_rationale": (
            "The v2 score ranking was strong, while coverage and risk thresholds were too permissive "
            "and too conservative. The new parent threshold maximizes an equal-weight macro utility "
            "over correctly predicted known leaves and correctly routed validation intra sources."),
        "protocol": (
            "Development-only: the existing test set was inspected before this rule was defined. "
            "The balanced rule uses only the frozen threshold-calibration partition, keeps the coverage "
            "root bit-identical, reports coverage/balanced/risk from one test feature pass, and performs "
            "no test fitting. A fresh outer species split is required for confirmation. No population-risk guarantee."),
    })
    for path in (source_plan, train_receipt, Path(__file__)):
        plan["inputs_sha256"][suite.portable(path)] = suite.file_hash(path)
    suite.dump(suite.resolve(plan["suite"]) / "plan.json", plan)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-suite", default="runs/taxosafe_rs_paper")
    parser.add_argument("--suite-dir", default="runs/taxosafe_balanced_v3_dev")
    parser.add_argument("--seed", type=int, required=True, choices=(1, 2, 3))
    args = parser.parse_args()
    plan = make_plan(args.source_suite, args.suite_dir, args.seed)
    print("Frozen development plan:", suite.resolve(plan["suite"]) / "plan.json")
    print("Reused checkpoint:", suite.resolve(plan["run_dir"]) / "ckpt/best.pth")
    print("Primary: full / balanced (coverage root + source-balanced child)")
    print("Training is omitted. Run memory, calibrate and test; all three profiles are mandatory.")


if __name__ == "__main__":
    main()
