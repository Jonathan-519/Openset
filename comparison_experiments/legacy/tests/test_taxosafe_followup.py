"""CPU checks for post-v1 diagnostics and checkpoint-reusing controls."""
import copy
from pathlib import Path
import tempfile
import unittest

import yaml

from tools.diagnose_taxosafe_v1 import combine_gates, root_signature
from tools.plan_taxosafe_factorial import controls, make_controls
from tools.plan_taxosafe_partial_pooling import (
    variants as pooling_variants, make_plan as make_pooling_plan, verified_training_source,
)
from tools.plan_taxosafe_balanced_v3 import variants as balanced_variants, make_plan as make_balanced_plan
from tools import run_taxosafe_suite as suite
from taxosafe_visual import pipeline
from tests._fixtures import freeze_legacy_oe_input, synthetic_suite_config


class FollowupTests(unittest.TestCase):
    def rows(self):
        rows = []
        for i, prediction in enumerate(("known", "intra_unknown", "global_unknown")):
            rows.append(dict(status="intra", path=str(i), image_sha256=str(i),
                             candidate_parent=0, candidate_parent_name="P",
                             candidate_leaf=1, candidate_leaf_name="L",
                             root_knownness_score=.2, root_gate_margin=.1 if i < 2 else -.1,
                             child_knownness_score=.4, residual_child_threshold=.3,
                             child_gate_margin=.1, prediction_type=prediction,
                             parent=0 if i < 2 else None, parent_name="P" if i < 2 else None,
                             leaf=1 if i == 0 else None, leaf_name="L" if i == 0 else None,
                             true_parent=0, true_leaf=None))
        return rows

    def test_crossing_preserves_root_and_uses_inclusive_child_boundary(self):
        root = self.rows()
        child = copy.deepcopy(root)
        for row, threshold in zip(child, (.5, .4, .1)):
            row["residual_child_threshold"] = threshold
        crossed = combine_gates(root, child, "crossed")
        self.assertEqual(root_signature(crossed), root_signature(root))
        self.assertEqual([r["prediction_type"] for r in crossed],
                         ["intra_unknown", "known", "global_unknown"])
        self.assertEqual([r["leaf"] for r in crossed], [None, 1, None])
        self.assertEqual(root[0]["prediction_type"], "known")

    def test_crossing_ignores_truth_and_rejects_mismatched_scores(self):
        root = self.rows()
        expected = combine_gates(root, root, "same")
        changed = copy.deepcopy(root)
        for row in changed:
            row.update(true_parent=999, true_leaf=888, status="extra")
        actual = combine_gates(changed, changed, "same")
        for a, b in zip(expected, actual):
            for field in ("prediction_type", "leaf", "parent", "child_gate_margin"):
                self.assertEqual(a[field], b[field])
        changed = copy.deepcopy(root)
        changed[0]["child_knownness_score"] += .01
        with self.assertRaisesRegex(ValueError, "scores/routing"):
            combine_gates(root, changed, "bad")

    def test_factorial_covers_all_original_alpha_k_scaling_combinations(self):
        values = controls()
        self.assertEqual(values.pop("full"), {})
        cells = {(v["alphas"][0], v["ks"][0], v["local_scaling"]) for v in values.values()}
        self.assertEqual(cells, {(a, k, s) for a in (0., .5, 1.) for k in (1, 3) for s in (False, True)})
        self.assertEqual(len(values), 12)

    def test_partial_pooling_plan_fixes_score_and_predeclares_main_method(self):
        values = pooling_variants()
        self.assertEqual(set(values), {"full", "branch_min_reference", "parent_pooled",
                                      "leaf_conditional", "shrinkage_5", "shrinkage_10",
                                      "shrinkage_20"})
        for value in values.values():
            self.assertEqual(value["alphas"], [0.0])
            self.assertEqual(value["ks"], [1])
            self.assertFalse(value["local_scaling"])
        self.assertEqual(values["full"]["threshold_mode"], "hierarchical_shrinkage")
        self.assertEqual(values["full"]["threshold_shrinkage"], "auto")
        self.assertEqual(pipeline._calibration_schema({"residual": {"enabled": True}}), 5)
        self.assertEqual(pipeline._calibration_schema({"residual": {
            "enabled": True, "threshold_mode": "hierarchical_shrinkage"}}), 6)

    def test_balanced_v3_is_opt_in_and_reuses_coverage_root(self):
        values = balanced_variants()
        self.assertEqual(set(values), {"full"})
        self.assertTrue(values["full"]["balanced_profile"])
        settings = {"residual": {"enabled": True, "balanced_profile": True,
                                 "threshold_mode": "hierarchical_shrinkage"}}
        self.assertEqual(pipeline._calibration_schema(settings), 7)
        self.assertEqual(pipeline._profile_pairs(settings, "all"),
                         [("coverage", "coverage"), ("balanced", "coverage"),
                          ("risk", "risk")])
        self.assertEqual(len(suite.stage_outputs("test", "artifact", "all")), 19)

    def test_plan_reuses_verified_checkpoint_and_freezes_source_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = synthetic_suite_config(root, "SYNTHETIC_FOLLOWUP_TEST")
            source = root / "source.yml"
            source.write_text(yaml.safe_dump(cfg))
            original = suite.make_plan(source, root / "original/seed_1", seed=1)
            run = root / "run"
            (run / "ckpt").mkdir(parents=True)
            (run / "ckpt/best.pth").write_bytes(b"SYNTHETIC, NOT A TORCH CHECKPOINT")
            archive = run / "training.yml"
            archive.write_bytes(suite.resolve(original["training_config"]).read_bytes())
            original.update(run_dir=str(run), archived_training_config=str(archive))
            plan_path = root / "original/seed_1/plan.json"
            suite.dump(plan_path, original)
            training = suite.steps(original, "train")[0]
            suite.dump(suite.receipt_path(original, training), {
                "plan_sha256": suite.file_hash(plan_path),
                "outputs_sha256": {p: suite.file_hash(suite.resolve(p)) for p in training["outputs"]}})
            plan = make_controls(root / "original", root / "controls", 1)
            self.assertTrue(plan["reuse_checkpoint"])
            self.assertEqual(plan["run_dir"], str(run))
            self.assertEqual(len(plan["variants"]), 13)
            self.assertEqual(len(suite.steps(plan)), 39)
            self.assertNotIn("train", [step["stage"] for step in suite.steps(plan)])
            self.assertEqual(suite.resolve(plan["training_config"]).read_bytes(), archive.read_bytes())
            suite.verify_inputs(plan)
            with self.assertRaises(FileExistsError):
                make_controls(root / "original", root / "controls", 1)
            archive.write_bytes(archive.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "Frozen input changed"):
                suite.verify_inputs(plan)

    def test_partial_pooling_plan_accepts_intentional_residual_change_but_verifies_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = synthetic_suite_config(root, "SYNTHETIC_POOLING_TEST")
            Path(cfg["data"]["oe_train"]).unlink()
            source = root / "source.yml"
            source.write_text(yaml.safe_dump(cfg))
            original = suite.make_plan(source, root / "original/seed_1", seed=1)
            run = root / "run"
            (run / "ckpt").mkdir(parents=True)
            checkpoint = run / "ckpt/best.pth"
            checkpoint.write_bytes(b"SYNTHETIC, NOT A TORCH CHECKPOINT")
            archive = run / "training.yml"
            archive.write_bytes(suite.resolve(original["training_config"]).read_bytes())
            original.update(run_dir=str(run), archived_training_config=str(archive))
            plan_path = root / "original/seed_1/plan.json"
            suite.dump(plan_path, original)
            training = suite.steps(original, "train")[0]
            suite.dump(suite.receipt_path(original, training), {
                "plan_sha256": suite.file_hash(plan_path),
                "outputs_sha256": {p: suite.file_hash(suite.resolve(p)) for p in training["outputs"]}})
            plan = make_pooling_plan(root / "original", root / "pooling", 1)
            self.assertTrue(plan["reuse_checkpoint"])
            self.assertEqual(plan["primary_method"], "full")
            self.assertEqual(len(plan["variants"]), 7)
            self.assertEqual(len(suite.steps(plan)), 21)
            self.assertNotIn("train", [step["stage"] for step in suite.steps(plan)])
            suite.verify_inputs(plan)
            checkpoint.write_bytes(b"CHANGED")
            with self.assertRaisesRegex(ValueError, "Frozen input changed"):
                suite.verify_inputs(plan)

    def test_balanced_v3_plan_freezes_all_profiles_without_retraining(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = synthetic_suite_config(root, "SYNTHETIC_BALANCED_TEST")
            Path(cfg["data"]["oe_train"]).unlink()
            source = root / "source.yml"
            source.write_text(yaml.safe_dump(cfg))
            original = suite.make_plan(source, root / "original/seed_1", seed=1)
            run = root / "run"
            (run / "ckpt").mkdir(parents=True)
            checkpoint = run / "ckpt/best.pth"
            checkpoint.write_bytes(b"SYNTHETIC, NOT A TORCH CHECKPOINT")
            archive = run / "training.yml"
            archive.write_bytes(suite.resolve(original["training_config"]).read_bytes())
            original.update(run_dir=str(run), archived_training_config=str(archive))
            plan_path = root / "original/seed_1/plan.json"
            suite.dump(plan_path, original)
            training = suite.steps(original, "train")[0]
            suite.dump(suite.receipt_path(original, training), {
                "plan_sha256": suite.file_hash(plan_path),
                "outputs_sha256": {p: suite.file_hash(suite.resolve(p)) for p in training["outputs"]}})
            plan = make_balanced_plan(root / "original", root / "balanced", 1)
            self.assertTrue(plan["reuse_checkpoint"])
            self.assertEqual(plan["primary_profile"], "balanced")
            self.assertEqual(plan["test_profiles"], "all")
            self.assertEqual(len(plan["variants"]), 1)
            self.assertEqual(len(suite.steps(plan)), 3)
            test = suite.steps(plan, "test")[0]
            self.assertEqual(test["argv"][-1], "all")
            self.assertIn("test/balanced/metrics.json", "\n".join(test["outputs"]))
            suite.verify_inputs(plan)

    def training_source(self, root, oe_mode):
        """Build real plan/receipt hashes with synthetic training outputs."""
        cfg = synthetic_suite_config(root, "SYNTHETIC_OE_PROVENANCE_TEST")
        if oe_mode == "enabled_missing_hash":
            cfg["loss"]["lambda_oe"] = 0.1
        source = root / "source.yml"
        source.write_text(yaml.safe_dump(cfg))
        original = suite.make_plan(source, root / "original/seed_1", seed=1)
        if oe_mode == "legacy_frozen":
            freeze_legacy_oe_input(original, cfg)
        elif oe_mode == "enabled_missing_hash":
            original["inputs_sha256"].pop(suite.portable(suite.resolve(cfg["data"]["oe_train"])))
        elif oe_mode == "enabled_missing_config":
            frozen = suite.resolve(original["training_config"])
            frozen_cfg = yaml.safe_load(frozen.read_text())
            frozen_cfg["loss"]["lambda_oe"] = 0.1
            frozen_cfg["data"].pop("oe_train")
            frozen.write_text(yaml.safe_dump(frozen_cfg))
        run = root / "run"
        (run / "ckpt").mkdir(parents=True)
        (run / "ckpt/best.pth").write_bytes(b"SYNTHETIC, NOT A TORCH CHECKPOINT")
        archive = run / "training.yml"
        archive.write_bytes(suite.resolve(original["training_config"]).read_bytes())
        original.update(run_dir=str(run), archived_training_config=str(archive))
        plan_path = root / "original/seed_1/plan.json"
        suite.dump(plan_path, original)
        training = suite.steps(original, "train")[0]
        suite.dump(suite.receipt_path(original, training), {
            "plan_sha256": suite.file_hash(plan_path),
            "outputs_sha256": {p: suite.file_hash(suite.resolve(p)) for p in training["outputs"]}})
        return cfg

    def test_legacy_frozen_disabled_oe_remains_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = self.training_source(root, "legacy_frozen")
            verified_training_source(root / "original", 1)
            Path(cfg["data"]["oe_train"]).write_text("CHANGED", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Original data/taxonomy input changed or missing"):
                verified_training_source(root / "original", 1)

    def test_enabled_oe_requires_config_and_frozen_hash(self):
        for mode, error in (("enabled_missing_hash", "Original data/taxonomy input changed or missing"),
                            ("enabled_missing_config", "Original enabled OE training requires data.oe_train")):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.training_source(root, mode)
                with self.assertRaisesRegex(ValueError, error):
                    verified_training_source(root / "original", 1)


if __name__ == "__main__":
    unittest.main()
