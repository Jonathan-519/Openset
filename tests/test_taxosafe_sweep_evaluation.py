"""Evaluation receipts preserve each arm even when research gates fail.

Only artifact/model loading and image scoring are replaced by deterministic
fixtures. Membership fitting, baseline reproduction, routing, metrics, CSVs,
artifact verification, and TEST use the production implementations.
"""
from contextlib import ExitStack
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from taxosafe_sweep import evaluation
from taxosafe_support import calibration as base
from taxosafe_support import membership_calibration as membership
from taxosafe_support import protocol
from tests.test_taxosafe_geometry_calibration import META, row


class SweepEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.archive = self.root / "reference"
        (self.archive / "training").mkdir(parents=True)
        (self.archive / "calibration").mkdir()
        self.cfg = {"calibration": {"decoder": "membership", "source_loo": False,
                                    "membership_grid_points": 3, "policy": "known_first"}}
        self.groups = {}
        for prefix in ("val", "test"):
            for status in base.STATUSES:
                values = []
                for i in range(2):
                    # Known and near evidence is identical, so their terminal
                    # requirements cannot both be met by any two thresholds.
                    record = row(prefix + "-" + status + str(i), status,
                                 pm=-1. if status == "extra" else 1.)
                    record.update(split=prefix + "_" + status,
                                  path=prefix + "/" + status + str(i),
                                  source=status + "-source")
                    values.append(record)
                self.groups[prefix + "_" + status] = values
        self.router = membership.calibrate(*(self.groups["val_" + s] for s in base.STATUSES),
                                            META, self.cfg["calibration"])
        # Deliberately use a distinguishable archived operating point. The
        # baseline must reproduce it; a new arm must retain its own DEV fit.
        self.router.update(parent_threshold=1000., leaf_threshold=1000.)
        protocol.write_records(self.archive / "calibration/development_scores.jsonl",
                               [r for s in base.STATUSES for r in self.groups["val_" + s]])
        descriptors = {}
        for key in ("checkpoint", "support"):
            path = self.archive / "training" / (key + ".pt")
            path.write_bytes(("fixture-" + key).encode())
            descriptors[key] = {"path": str(path), "sha256": protocol.file_hash(path)}
        self.source = SimpleNamespace(directory=self.archive, config=self.cfg, meta=copy.deepcopy(META),
                                      router=copy.deepcopy(self.router), binding={"source_id": "archived-fixture"},
                                      training=copy.deepcopy(descriptors))
        self.binding = {"arm_id": "E01_fixture", "source_binding": copy.deepcopy(self.source.binding),
                        **copy.deepcopy(descriptors)}
        self.encoder = torch.nn.Linear(1, 1)
        self.stage_calls, self.collect_calls = [], []
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.object(evaluation, "_assert_source"))
        stack.enter_context(patch.object(evaluation, "_stage_rows", side_effect=self.stage_rows))
        stack.enter_context(patch.object(evaluation.support_pipeline, "collect", side_effect=self.collect))

    def stage_rows(self, source, stage):
        self.assertIs(source, self.source)
        self.stage_calls.append(stage)
        splits = protocol.STAGE_SPLITS[stage]
        return {key: copy.deepcopy(self.groups[key]) for key in splits}, {"fixture_stage": stage}

    def collect(self, groups, cfg, meta, encoder, evidence, bank, device):
        self.assertEqual(cfg, self.cfg)
        self.assertEqual(meta, META)
        self.collect_calls.append(tuple(groups))
        return copy.deepcopy(groups), {key: {"seconds": 0., "manifest_rows": len(rows)}
                                      for key, rows in groups.items()}

    def dev(self, name="calibration", frozen=False):
        return evaluation.evaluate_development(self.source, self.encoder, None, None, self.cfg,
                                               self.root / name, self.binding, frozen_baseline=frozen)

    def run_test_stage(self, name="test", calibration="calibration", frozen=False):
        return evaluation.evaluate_test(self.source, self.encoder, None, None, self.cfg,
                                       self.root / name, self.binding, self.root / calibration,
                                       frozen_baseline=frozen)

    def test_failed_scientific_gates_keep_arm_router_and_allow_test_without_refitting(self):
        with patch.object(evaluation.membership, "calibrate", wraps=membership.calibrate) as fitted:
            calibrated = self.dev()
        self.assertEqual(fitted.call_count, 1)
        self.assertFalse(calibrated["targets_passed"])
        self.assertTrue(calibrated["test_allowed_after_failed_gates"])
        actual_router = protocol.read_json(self.root / "calibration/router.json")
        self.assertEqual(actual_router["router_origin"], "arm_development_fit")
        self.assertTrue(actual_router["best_effort"])
        self.assertNotEqual(actual_router["parent_threshold"], self.source.router["parent_threshold"])
        fit_hashes = set(actual_router["fit_image_sha256"])
        test_hashes = {r["image_sha256"] for key, rows in self.groups.items() if key.startswith("test_") for r in rows}
        self.assertFalse(fit_hashes & test_hashes)
        # Remove every fit row and make all calibration calls fatal at TEST.
        for key in list(self.groups):
            if key.startswith("val_"):
                del self.groups[key]
        with patch.object(evaluation.membership, "calibrate", side_effect=AssertionError("TEST must not fit")):
            tested = self.run_test_stage()
        self.assertEqual(tested["status"], "completed")
        self.assertFalse(tested["targets_passed"])
        self.assertEqual(self.stage_calls, ["calibrate", "test"])
        self.assertEqual(actual_router, protocol.read_json(self.root / "test/router.json"))
        self.assertFalse(tested["summary"]["arm_predictions_replaced_by_reference"])
        self.assertFalse(tested["summary"]["calibration_gate_is_execution_gate"])
        self.assertFalse(tested["test_used_for_fitting"])
        scores = evaluation._records(self.root / "test/scores.jsonl")
        expected = base.apply_router(scores, actual_router, META)
        predictions = evaluation._records(self.root / "test/predictions.jsonl")
        self.assertEqual([(r["prediction_type"], r["output_node"]) for r in expected],
                         [(r["prediction_type"], r["output_node"]) for r in predictions])

    def test_frozen_baseline_uses_saved_router_and_real_reproduction_check(self):
        original = copy.deepcopy(self.source.router)
        with patch.object(evaluation, "_check_baseline_development", wraps=evaluation._check_baseline_development) as reproduce:
            with patch.object(evaluation.membership, "calibrate", side_effect=AssertionError("Frozen baseline cannot refit")):
                calibrated = self.dev(frozen=True)
        self.assertEqual(reproduce.call_count, 1)
        self.assertEqual(calibrated["calibration_status"], "frozen_reference_reproduced")
        self.assertEqual(self.source.router, original)
        actual = protocol.read_json(self.root / "calibration/router.json")
        self.assertEqual(actual["router_origin"], "frozen_reference_router")
        self.assertEqual(actual["parent_threshold"], original["parent_threshold"])
        self.assertEqual(actual["leaf_threshold"], original["leaf_threshold"])
        reproduction = protocol.read_json(self.root / "calibration/baseline_reproduction.json")
        self.assertTrue(reproduction["raw_scores_match"])
        self.assertTrue(reproduction["decisions_match"])
        self.assertEqual(reproduction["matched_unique_images"], 6)
        tested = self.run_test_stage(frozen=True)
        self.assertEqual(tested["status"], "completed")
        self.assertEqual(tested["summary"]["counts"]["leaf_outputs"], 0)

    def test_changed_checkpoint_or_support_stops_before_test_scoring(self):
        self.dev()
        for key in ("checkpoint", "support"):
            with self.subTest(artifact=key):
                path = Path(self.binding[key]["path"])
                original = path.read_bytes()
                path.write_bytes(original + b"tampered")
                previous_collect = len(self.collect_calls)
                try:
                    with self.assertRaisesRegex(ValueError, "artifact binding failed"):
                        self.run_test_stage(name="test-" + key)
                    self.assertEqual(len(self.collect_calls), previous_collect)
                    self.assertFalse((self.root / ("test-" + key)).exists())
                finally:
                    path.write_bytes(original)

    def test_changed_router_stops_before_test_scoring(self):
        self.dev()
        path = self.root / "calibration/router.json"
        router = protocol.read_json(path)
        router["parent_threshold"] += 1.
        protocol.write_json(path, router)
        previous_collect = len(self.collect_calls)
        with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
            self.run_test_stage()
        self.assertEqual(len(self.collect_calls), previous_collect)
        self.assertFalse((self.root / "test").exists())

    def test_completed_stages_are_immutable_and_reference_path_cannot_be_claimed(self):
        self.dev()
        original = (self.root / "calibration/completed.json").read_bytes()
        count = len(self.collect_calls)
        with self.assertRaisesRegex(ValueError, "stage already exists"):
            self.dev()
        self.assertEqual(len(self.collect_calls), count)
        self.assertEqual((self.root / "calibration/completed.json").read_bytes(), original)
        self.run_test_stage()
        test_original = (self.root / "test/completed.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "stage already exists"):
            self.run_test_stage()
        self.assertEqual((self.root / "test/completed.json").read_bytes(), test_original)
        with self.assertRaisesRegex(ValueError, "separate from the archived reference"):
            evaluation._claim(self.archive / "another-stage", self.source)

    def test_unique_metrics_and_exported_alias_weights(self):
        alias = copy.deepcopy(self.groups["test_known"][0])
        alias["path"] = "alias/known"
        self.groups["test_known"].append(alias)
        self.dev()
        tested = self.run_test_stage()
        self.assertEqual(tested["summary"]["counts"]["known"], 2)
        self.assertEqual(tested["summary"]["unique_image_count"], 6)
        self.assertEqual(tested["summary"]["input_record_count"], 7)
        self.assertEqual(tested["summary"]["duplicate_record_count"], 1)
        metrics = protocol.read_json(self.root / "test/metrics.json")
        self.assertEqual(metrics["known"]["sample_count"], 2)
        predictions = evaluation._records(self.root / "test/predictions.jsonl")
        same = [r for r in predictions if r["image_sha256"] == alias["image_sha256"]]
        self.assertEqual([r["evaluation_weight"] for r in same], [1, 0])
        self.assertEqual(sum(r["evaluation_weight"] for r in predictions), 6)

    def test_frozen_baseline_reproduction_failure_never_becomes_completed(self):
        self.groups["val_known"][0]["support_evidence"]["parent_membership_logits"][0] += .1
        with self.assertRaisesRegex(ValueError, "numeric reproduction failed"):
            self.dev(frozen=True)
        self.assertFalse((self.root / "calibration/completed.json").exists())
        with self.assertRaisesRegex(ValueError, "regular file"):
            self.run_test_stage(frozen=True)

    def test_symlink_artifacts_receipts_and_output_ancestors_are_rejected(self):
        checkpoint_alias = self.root / "checkpoint_alias.pt"
        checkpoint_alias.symlink_to(self.binding["checkpoint"]["path"])
        alias_binding = copy.deepcopy(self.binding)
        alias_binding["checkpoint"]["path"] = str(checkpoint_alias)
        with self.assertRaisesRegex(ValueError, "symlink ancestors"):
            evaluation._verify_binding(self.source, alias_binding)
        output_alias = self.root / "output_alias"
        output_alias.symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "must not traverse symlinks"):
            evaluation._claim(output_alias / "new-stage", self.source)
        self.dev()
        receipt = self.root / "calibration/completed.json"
        relocated = self.root / "receipt-original.json"
        receipt.rename(relocated)
        receipt.symlink_to(relocated)
        with self.assertRaisesRegex(ValueError, "symlink ancestors"):
            self.run_test_stage()
        self.assertFalse((self.root / "test").exists())


if __name__ == "__main__":
    unittest.main()
