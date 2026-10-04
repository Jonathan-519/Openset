"""Actual frozen collection, statistics and artifact lifecycle on tiny images."""
import copy
import builtins
import io
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from taxosafe_support import protocol as source_protocol
from tests import test_taxosafe_refine_pipeline as fixture


class FrontierPipelineTests(unittest.TestCase):
    setUpClass = classmethod(fixture.FrozenPipelineContracts.setUpClass.__func__)
    tearDownClass = classmethod(fixture.FrozenPipelineContracts.tearDownClass.__func__)
    setUp = fixture.FrozenPipelineContracts.setUp
    make_source = fixture.FrozenPipelineContracts.make_source
    load_stage = fixture.FrozenPipelineContracts.load_stage
    read_split = fixture.FrozenPipelineContracts.read_split
    assert_frozen = fixture.FrozenPipelineContracts.assert_frozen

    def configure(self):
        from taxosafe_frontier import protocol
        self.cfg = protocol.effective_config(protocol.PROJECT_ROOT / protocol.DEFAULT_CONFIG)
        self.cfg["calibration"].update(outer_folds=2, inner_folds=2, min_rule_sources=2)

    def test_complete_cycle_test_never_fits_and_preserves_candidate_identity(self):
        from taxosafe_frontier import calibration, pipeline, protocol
        from taxosafe_geometry.core import HierarchicalGeometry, RobustScoreStandardizer
        from tools.pack_taxosafe_frontier_review import pack
        import tarfile
        self.make_source(with_test=True)
        self.configure()
        original = fixture.artifact_snapshot(self.source)
        pipeline.fit(self.cfg, self.source, self.directory, self.device)
        pipeline.calibrate_run(self.cfg, self.source, self.directory, self.device)
        router = source_protocol.read_json(self.directory / "calibration/router.json")
        test_hashes = {r["image_sha256"] for name in source_protocol.STAGE_SPLITS["test"] for r in self.groups[name]}
        self.assertFalse(test_hashes & set(router["fit_image_sha256"]))
        self.assertFalse((self.directory / "test").exists())
        cal_receipt = source_protocol.read_json(self.directory / "calibration/completed.json")
        self.assertTrue(cal_receipt["development_reused_for_method_design"])
        self.assertFalse(cal_receipt["confirmatory_validation"])
        self.assertFalse(cal_receipt["independent_model_level_validation"])
        self.assertEqual(cal_receipt["validation_scope"], pipeline.VALIDATION_SCOPE)
        protocol.verify_artifacts(self.directory / "calibration", cal_receipt, pipeline.CAL_ARTIFACTS)
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
            if new["prediction_type"] == "known":
                self.assertEqual(old["prediction_type"], "known")
                self.assertEqual(old["leaf"], new["leaf"])
        aliases = [r for r in after if r["image_sha256"] == self.groups["test_known"][0]["image_sha256"]]
        self.assertEqual([r["evaluation_weight"] for r in aliases], [1, 0])
        self.assertEqual(fixture.artifact_snapshot(self.source), original)
        target = self.directory.parent / "frontier_review.tar.gz"
        pack(self.directory, target)
        with tarfile.open(target) as archive:
            names = archive.getnames()
            self.assertIn("run/calibration/oof_predictions.jsonl", names)
            self.assertIn("run/test/paired_risk_report.json", names)
            self.assertFalse(any(name.endswith(".pth") for name in names))

    def test_fit_never_changes_reference_parameters_support_or_source_files(self):
        from taxosafe_frontier import pipeline
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
        self.assertEqual(receipt["gradient_splits"], [])
        self.assertEqual(receipt["optimizer_steps"], 0)
        self.assertIs(receipt["test_used_for_fitting"], False)
        self.assertIs(receipt["unknown_images_used_for_fitting"], False)

    def test_frontier_fresh_fit_reproduces_inherited_frozen_evidence_exactly(self):
        from taxosafe_frontier import pipeline
        from taxosafe_parentrisk import pipeline as previous_pipeline, protocol as previous_protocol
        self.make_source()
        self.configure()
        previous_cfg = previous_protocol.effective_config(previous_protocol.PROJECT_ROOT / previous_protocol.DEFAULT_CONFIG)
        previous_run = self.directory.with_name("previous_parentrisk")
        previous_pipeline.fit(previous_cfg, self.source, previous_run, self.device)
        previous_bytes = fixture.artifact_snapshot(previous_run)
        pipeline.fit(self.cfg, self.source, self.directory, self.device)
        old = fixture.read_records(previous_run / "parentrisk/train_scores.jsonl")
        new = fixture.read_records(self.directory / "frontier/train_scores.jsonl")
        self.assertEqual(old, new)
        self.assertEqual(source_protocol.read_json(previous_run / "parentrisk/scales.json"),
                         source_protocol.read_json(self.directory / "frontier/scales.json"))
        self.assertEqual(fixture.artifact_snapshot(previous_run), previous_bytes)
        old_receipt = source_protocol.read_json(previous_run / "parentrisk/completed.json")
        new_receipt = source_protocol.read_json(self.directory / "frontier/completed.json")
        self.assertNotEqual(old_receipt["signature"]["method"], new_receipt["signature"]["method"])
        self.assertEqual(old_receipt["source_binding"], new_receipt["source_binding"])

    def test_signed_text_replay_never_opens_test_or_binaries_and_rejects_score_tampering(self):
        from taxosafe_parentrisk import pipeline as previous_pipeline, protocol as previous_protocol
        from tools.replay_taxosafe_frontier import replay
        self.make_source(with_test=True)
        previous_cfg = previous_protocol.effective_config(previous_protocol.PROJECT_ROOT / previous_protocol.DEFAULT_CONFIG)
        previous_cfg["calibration"].update(outer_folds=2, inner_folds=2, grid_points=2)
        previous_run = self.directory.with_name("previous_parentrisk")
        previous_pipeline.fit(previous_cfg, self.source, previous_run, self.device)
        previous_pipeline.calibrate_run(previous_cfg, self.source, previous_run, self.device)
        previous_pipeline.test_run(previous_cfg, self.source, previous_run, self.device)
        old_bytes = fixture.artifact_snapshot(previous_run)
        self.calls.clear()
        opened = []

        def guard(original):
            def checked(file, *args, **kwargs):
                if isinstance(file, (str, bytes, Path)):
                    path = Path(file.decode() if isinstance(file, bytes) else file)
                    if "test" in path.parts or path.suffix in {".pth", ".pt", ".npy", ".npz", ".jpg", ".png"}:
                        raise AssertionError("DEV text replay cannot read TEST, images, or model/cache binaries: " + str(path))
                    opened.append(str(path))
                return original(file, *args, **kwargs)
            return checked

        output = self.directory.with_name("frontier_replay")
        with patch("builtins.open", guard(builtins.open)), patch("io.open", guard(io.open)):
            summary = replay(previous_run, output)
        provenance = summary["provenance"]
        self.assertEqual(provenance["kind"], "exploratory_dev_text_replay_not_production_receipt")
        for key in ("test_files_opened", "image_pipeline_executed", "checkpoint_and_cache_binary_verified",
                    "independent_model_level_validation", "confirmatory_validation"):
            self.assertIs(provenance[key], False, key)
        self.assertTrue(provenance["development_reused_for_method_design"])
        self.assertFalse((output / "completed.json").exists())
        self.assertFalse((output / "calibration/completed.json").exists())
        self.assertEqual(self.calls, [])
        self.assertTrue(any(path.endswith("development_scores.jsonl") for path in opened))
        self.assertEqual(fixture.artifact_snapshot(previous_run), old_bytes)
        scores = previous_run / "calibration/development_scores.jsonl"
        scores.write_bytes(scores.read_bytes() + b"modified")
        rejected_output = self.directory.with_name("frontier_replay_corrupt")
        with self.assertRaisesRegex(ValueError, "hash/path mismatch"):
            replay(previous_run, rejected_output)
        self.assertFalse(rejected_output.exists())

    def test_fit_artifact_tampering_is_rejected_before_dev_images(self):
        from taxosafe_frontier import pipeline, protocol
        self.make_source()
        self.configure()
        pipeline.fit(self.cfg, self.source, self.directory, self.device)
        for filename in protocol.FIT_ARTIFACTS.values():
            path = self.directory / protocol.FIT_STAGE / filename
            original = path.read_bytes()
            try:
                path.write_bytes(original + b"modified")
                self.calls.clear()
                with self.subTest(artifact=filename), self.assertRaises(ValueError):
                    pipeline.calibrate_run(self.cfg, self.source, self.directory, self.device)
                self.assertEqual(self.calls, [])
                self.assertFalse((self.directory / "calibration").exists())
            finally:
                path.write_bytes(original)

    def test_completed_calibration_audit_is_bound_before_test_images(self):
        from taxosafe_frontier import pipeline
        self.make_source()
        self.configure()
        pipeline.fit(self.cfg, self.source, self.directory, self.device)
        pipeline.calibrate_run(self.cfg, self.source, self.directory, self.device)
        path = self.directory / "calibration/outer_audit.json"
        path.write_bytes(path.read_bytes() + b"modified")
        self.calls.clear()
        with self.assertRaises(ValueError):
            pipeline.test_run(self.cfg, self.source, self.directory, self.device)
        self.assertEqual(self.calls, [])
        self.assertFalse((self.directory / "test").exists())

    def test_changed_source_receipt_is_rejected_before_dev(self):
        from taxosafe_frontier import pipeline
        from taxosafe_refine import importer
        self.make_source()
        self.configure()
        pipeline.fit(self.cfg, self.source, self.directory, self.device)
        path = self.source / "training/completed.json"
        receipt = source_protocol.read_json(path)
        receipt["review_note"] = "different immutable source"
        source_protocol.write_json(path, receipt)
        importer.inspect_reference(self.source)
        self.calls.clear()
        with self.assertRaises(ValueError):
            pipeline.calibrate_run(self.cfg, self.source, self.directory, self.device)
        self.assertEqual(self.calls, [])
        self.assertFalse((self.directory / "calibration").exists())

    def test_partial_run_requires_new_directory(self):
        from taxosafe_frontier import pipeline
        self.make_source()
        self.configure()
        self.directory.mkdir()
        marker = self.directory / "partial.json"
        marker.write_text('{"preserve": true}', encoding="utf-8")
        with self.assertRaises(ValueError):
            pipeline.fit(self.cfg, self.source, self.directory, self.device)
        self.assertEqual(marker.read_text(), '{"preserve": true}')
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
