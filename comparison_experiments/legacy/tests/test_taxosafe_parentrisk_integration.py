"""Production fit/calibrate/test contracts on receipt-compatible toy images.

The reference training/importer, statistical fitting, collectors, decoders,
artifact binding and full filesystem lifecycle run normally. Only image/model
I/O uses the existing deterministic tiny reference fixture (no CLIP download).
"""
import copy
import unittest
from unittest.mock import patch

import torch

from taxosafe_support import protocol as source_protocol
from tests import test_taxosafe_refine_pipeline as fixture


class ParentRiskIntegrationTests(unittest.TestCase):
    setUpClass = classmethod(fixture.FrozenPipelineContracts.setUpClass.__func__)
    tearDownClass = classmethod(fixture.FrozenPipelineContracts.tearDownClass.__func__)
    setUp = fixture.FrozenPipelineContracts.setUp
    make_source = fixture.FrozenPipelineContracts.make_source
    load_stage = fixture.FrozenPipelineContracts.load_stage
    read_split = fixture.FrozenPipelineContracts.read_split
    assert_frozen = fixture.FrozenPipelineContracts.assert_frozen

    def configure(self, mode="parent_only"):
        from taxosafe_parentrisk import protocol
        self.cfg = protocol.effective_config(protocol.PROJECT_ROOT / protocol.DEFAULT_CONFIG)
        self.cfg["calibration"].update(mode=mode, outer_folds=2, inner_folds=2,
                                       grid_points=2, min_rule_sources=2)

    def test_real_full_cycle_test_never_fits_and_parent_only_preserves_all_leaves(self):
        from taxosafe_parentrisk import calibration, pipeline
        from taxosafe_geometry.core import HierarchicalGeometry, RobustScoreStandardizer
        from tools.diagnose_taxosafe_failure_modes import audit_run
        self.make_source(with_test=True)
        self.configure()
        original = fixture.artifact_snapshot(self.source)
        pipeline.fit(self.cfg, self.source, self.directory, self.device)
        pipeline.calibrate_run(self.cfg, self.source, self.directory, self.device)
        router = source_protocol.read_json(self.directory / "calibration/router.json")
        test_hashes = {r["image_sha256"] for name in source_protocol.STAGE_SPLITS["test"] for r in self.groups[name]}
        self.assertFalse(test_hashes & set(router["fit_image_sha256"]))
        self.assertFalse((self.directory / "test").exists())
        diagnostic = audit_run(self.directory)
        self.assertFalse(diagnostic["test_opened"])
        self.assertTrue(diagnostic["paired_risks"]["original_leaf_invariance"]["passed"])
        # Removing all fitting inputs prevents an accidental hidden train/DEV
        # read at TEST, in addition to spies guarding the fitting entrypoints.
        for name in ("train", "val_known", "val_intra", "val_extra"):
            del self.groups[name]
        self.calls.clear()
        with patch.object(calibration, "calibrate", side_effect=AssertionError("TEST cannot calibrate")), \
                patch.object(HierarchicalGeometry, "fit", side_effect=AssertionError("TEST cannot fit geometry")), \
                patch.object(RobustScoreStandardizer, "fit", side_effect=AssertionError("TEST cannot fit scales")):
            pipeline.test_run(self.cfg, self.source, self.directory, self.device)
        self.assertEqual(self.calls, [("test", source_protocol.STAGE_SPLITS["test"])])
        before = fixture.read_records(self.directory / "test/baseline_predictions.jsonl")
        after = fixture.read_records(self.directory / "test/predictions.jsonl")
        self.assertEqual(len(before), len(after))
        for old, new in zip(before, after):
            for key in ("image_sha256", "candidate_parent", "candidate_leaf", "support_evidence", "log_probs"):
                self.assertEqual(old[key], new[key], key)
            if old["prediction_type"] == "known":
                for key in ("prediction_type", "output_node", "parent", "leaf"):
                    self.assertEqual(old[key], new[key], key)
        aliases = [r for r in after if r["image_sha256"] == self.groups["test_known"][0]["image_sha256"]]
        self.assertEqual([r["evaluation_weight"] for r in aliases], [1, 0])
        self.assertEqual(fixture.artifact_snapshot(self.source), original)
        self.assertTrue(audit_run(self.directory, evaluate_test=True)["test_opened"])

    def test_fit_leaves_reference_tensors_and_support_immutable(self):
        from taxosafe_parentrisk import pipeline
        from taxosafe_refine import importer
        self.make_source()
        self.configure()
        reference = importer.load_reference(self.source, self.device)
        encoder = fixture.tensor_snapshot(reference.encoder)
        evidence = fixture.tensor_snapshot(reference.evidence)
        bank = copy.deepcopy(reference.bank.state_dict())
        original = fixture.artifact_snapshot(self.source)
        with patch.object(pipeline, "load_reference", return_value=reference):
            receipt = pipeline.fit(self.cfg, self.source, self.directory, self.device)
        self.assert_frozen(reference.encoder, encoder)
        self.assert_frozen(reference.evidence, evidence)
        for key, old in bank.items():
            new = reference.bank.state_dict()[key]
            if torch.is_tensor(old):
                self.assertTrue(torch.equal(old, new), key)
            else:
                self.assertEqual(old, new, key)
        self.assertEqual(fixture.artifact_snapshot(self.source), original)
        self.assertEqual(self.calls, [("train", ("train",))])
        self.assertEqual(receipt["fit_splits"], ["train"])
        self.assertEqual(receipt["optimizer_steps"], 0)
        self.assertEqual(receipt["gradient_splits"], [])
        self.assertIs(receipt["test_used_for_fitting"], False)

    def test_mutated_fit_artifacts_fail_before_dev_or_new_stage_creation(self):
        from taxosafe_parentrisk import pipeline, protocol
        self.make_source()
        self.configure()
        pipeline.fit(self.cfg, self.source, self.directory, self.device)
        for filename in protocol.FIT_ARTIFACTS.values():
            path = self.directory / protocol.FIT_STAGE / filename
            original = path.read_bytes()
            try:
                path.write_bytes(original + b"changed")
                self.calls.clear()
                with self.subTest(artifact=filename), self.assertRaises(ValueError):
                    pipeline.calibrate_run(self.cfg, self.source, self.directory, self.device)
                self.assertEqual(self.calls, [])
                self.assertFalse((self.directory / "calibration").exists())
            finally:
                path.write_bytes(original)

    def test_changed_valid_source_receipt_is_rejected_before_dev(self):
        from taxosafe_parentrisk import pipeline
        from taxosafe_refine import importer
        self.make_source()
        self.configure()
        pipeline.fit(self.cfg, self.source, self.directory, self.device)
        path = self.source / "training/completed.json"
        receipt = source_protocol.read_json(path)
        receipt["review_note"] = "The source now differs from frozen fit binding"
        source_protocol.write_json(path, receipt)
        importer.inspect_reference(self.source)
        self.calls.clear()
        with self.assertRaises(ValueError):
            pipeline.calibrate_run(self.cfg, self.source, self.directory, self.device)
        self.assertEqual(self.calls, [])
        self.assertFalse((self.directory / "calibration").exists())

    def test_existing_partial_run_is_not_resumed_or_overwritten(self):
        from taxosafe_parentrisk import pipeline
        self.make_source()
        self.configure()
        self.directory.mkdir()
        marker = self.directory / "partial.json"
        marker.write_text('{"keep": true}', encoding="utf-8")
        with self.assertRaises(ValueError):
            pipeline.fit(self.cfg, self.source, self.directory, self.device)
        self.assertEqual(marker.read_text(encoding="utf-8"), '{"keep": true}')
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
