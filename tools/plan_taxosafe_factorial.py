"""Freeze post-v1 development controls using the existing clean checkpoint.

Fix alpha, k and local scaling in a 3x2x2 factorial design. This separates
scaling from the k reselection that confounded the original ablation.
No retraining, no modification of the original suite, no new novelty claim.
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import run_taxosafe_suite as suite


def controls():
    # full retains the exact v1 adaptive recipe as the paired reference.
    variants = {"full": {}}
    for alpha, tag in ((0., "a0"), (.5, "a05"), (1., "a1")):
        for k in (1, 3):
            for scaling, suffix in ((False, "raw"), (True, "scaled")):
                variants["{}_k{}_{}".format(tag, k, suffix)] = {
                    "alphas": [alpha], "ks": [k], "local_scaling": scaling}
    return variants


def make_controls(source_suite, output_suite, seed):
    source = suite.resolve(source_suite) / ("seed_" + str(seed))
    original = suite.read(source / "plan.json")
    if original["seed"] != seed or original["reuse_checkpoint"]:
        raise ValueError("Source must be the original completed clean training plan")
    suite.verify_inputs(original)
    training = suite.steps(original, "train")[0]
    if not suite.verify_receipt(original, training):
        raise ValueError("Original training stage has no verified receipt")
    archive = suite.resolve(original["archived_training_config"])
    if suite.file_hash(archive) != suite.file_hash(suite.resolve(original["training_config"])):
        raise ValueError("Training archive differs from original frozen configuration")
    original_variants = suite.VARIANTS
    try:
        suite.VARIANTS = controls()
        plan = suite.make_plan(archive, suite.resolve(output_suite) / ("seed_" + str(seed)),
                               trial=original["trial"], run_dir=original["run_dir"])
    finally:
        suite.VARIANTS = original_variants
    plan["study_type"] = "post_v1_development_factorial_controls"
    plan["protocol"] = (
        "Original test already inspected. Full is the v1 reference; twelve fixed alpha/k/scaling "
        "cells isolate factorial effects. Root and checkpoint shared. No test-based winner "
        "selection or confirmatory novelty claim. All methods/profiles reported.")
    plan["source_plan"] = suite.portable(source / "plan.json")
    for path in (source / "plan.json", source / "receipts/train.json", Path(__file__)):
        plan["inputs_sha256"][suite.portable(path)] = suite.file_hash(path)
    suite.dump(suite.resolve(plan["suite"]) / "plan.json", plan)
    return plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-suite", default="runs/taxosafe_rs_paper")
    p.add_argument("--suite-dir", default="runs/taxosafe_factorial_dev")
    p.add_argument("--seed", type=int, required=True, choices=(1, 2, 3))
    args = p.parse_args()
    plan = make_controls(args.source_suite, args.suite_dir, args.seed)
    print("Frozen development plan:", suite.resolve(plan["suite"]) / "plan.json")
    print("Reused checkpoint:", suite.resolve(plan["run_dir"]) / "ckpt/best.pth")
    print("Variants:", ", ".join(plan["variants"]))
    print("Training is omitted. Next: run_taxosafe_suite.py run --stage memory, then calibrate, then test.")


if __name__ == "__main__":
    main()
