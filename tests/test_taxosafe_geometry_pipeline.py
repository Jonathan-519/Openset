"""CPU integration: frozen weights, TRAIN-only statistics and audited TEST."""
import copy
import unittest
from unittest.mock import patch

import torch

from taxosafe_support import protocol as reference_protocol
from tests import test_taxosafe_refine_pipeline as fixture


class GeometryPipelineContracts(unittest.TestCase):
    # Reuse receipt-compatible synthetic fixtures, without inheriting their
    # reconstruction tests or downloading any CLIP model / real image.
    setUpClass = classmethod(fixture.FrozenPipelineContracts.setUpClass.__func__)
    tearDownClass = classmethod(fixture.FrozenPipelineContracts.tearDownClass.__func__)
    setUp = fixture.FrozenPipelineContracts.setUp
    make_source = fixture.FrozenPipelineContracts.make_source
    load_stage = fixture.FrozenPipelineContracts.load_stage
    read_split = fixture.FrozenPipelineContracts.read_split
    assert_frozen = fixture.FrozenPipelineContracts.assert_frozen

    def configure_geometry(self):
        from taxosafe_geometry import protocol
        self.cfg = protocol.effective_config(protocol.PROJECT_ROOT / protocol.DEFAULT_CONFIG)
        self.cfg["calibration"].update(grid_points=3, weights=[0., 1.],
                                        source_loo=False, source_loo_safeguard=False)

    def test_local_full_cycle_uses_new_decoder_without_test_fitting(self):
        from taxosafe_geometry import local, pipeline, protocol
        self.make_source(with_test=True)
        self.cfg = protocol.effective_config(protocol.PROJECT_ROOT /
            "configs/Zooplankton_Taxonomic_Tree/TaxoSafe_reference_local.yml")
        self.cfg["calibration"].update(source_loo=False, source_loo_safeguard=False,
                                      min_known_per_parent=1, max_thresholds=3)
        original = fixture.artifact_snapshot(self.source)
        pipeline.fit(self.cfg, self.source, self.directory, self.device)
        pipeline.calibrate_run(self.cfg, self.source, self.directory, self.device)
        router = reference_protocol.read_json(self.directory / "calibration/router.json")
        self.assertEqual(router["decoder"], "local_guarded")
        self.calls.clear()
        with patch.object(local, "calibrate", side_effect=AssertionError("TEST cannot fit local rules")), \
                patch.object(pipeline.HierarchicalGeometry, "fit", side_effect=AssertionError("TEST cannot fit geometry")):
            receipt = pipeline.test_run(self.cfg, self.source, self.directory, self.device)
        self.assertEqual(self.calls, [("test", reference_protocol.STAGE_SPLITS["test"])])
        self.assertTrue(receipt["candidate_preserved"])
        self.assertEqual(fixture.artifact_snapshot(self.source), original)
        old = fixture.read_records(self.directory / "test/baseline_predictions.jsonl")
        new = fixture.read_records(self.directory / "test/predictions.jsonl")
        for before, after in zip(old, new):
            self.assertEqual(after["decoder"], "local_guarded")
            if after["prediction_type"] == "known":
                self.assertEqual(before["prediction_type"], "known")
                self.assertEqual(before["leaf"], after["leaf"])

    def test_fit_keeps_source_frozen_and_fits_only_unique_known_train(self):
        from taxosafe_geometry import pipeline
        from taxosafe_refine import importer
        self.make_source()
        self.configure_geometry()
        source_bytes = fixture.artifact_snapshot(self.source)
        reference = importer.load_reference(self.source, self.device)
        encoder, evidence = fixture.tensor_snapshot(reference.encoder), fixture.tensor_snapshot(reference.evidence)
        bank = copy.deepcopy(reference.bank.state_dict())
        with patch.object(pipeline, "load_reference", return_value=reference):
            receipt = pipeline.fit(self.cfg, self.source, self.directory, self.device)
        self.assert_frozen(reference.encoder, encoder)
        self.assert_frozen(reference.evidence, evidence)
        for key, before in bank.items():
            after = reference.bank.state_dict()[key]
            if torch.is_tensor(before):
                self.assertTrue(torch.equal(before, after), key)
                self.assertFalse(after.requires_grad)
            else:
                self.assertEqual(before, after, key)
        self.assertEqual(fixture.artifact_snapshot(self.source), source_bytes)
        self.assertEqual(self.calls, [("train", ("train",))])
        self.assertEqual(receipt["fit_splits"], ["train"])
        self.assertEqual(receipt["gradient_splits"], [])
        self.assertEqual(receipt["optimizer_steps"], 0)
        self.assertIs(receipt["test_used_for_fitting"], False)
        report = reference_protocol.read_json(self.directory / "geometry/fit_report.json")
        self.assertEqual(report["unique_train_images"], len(self.groups["train"]))
        train_hashes = {row["image_sha256"] for row in self.groups["train"]}
        for value in report["score_scale_audit"].values():
            self.assertLessEqual(set(value["image_sha256"]), train_hashes)
            self.assertGreater(value["count"], 0)

    def test_full_cycle_test_never_refits_and_candidates_are_preserved(self):
        from taxosafe_geometry import calibration, pipeline
        self.make_source(with_test=True)
        self.configure_geometry()
        source_bytes = fixture.artifact_snapshot(self.source)
        pipeline.fit(self.cfg, self.source, self.directory, self.device)
        pipeline.calibrate_run(self.cfg, self.source, self.directory, self.device)
        router = reference_protocol.read_json(self.directory / "calibration/router.json")
        test_hashes = {row["image_sha256"] for name in reference_protocol.STAGE_SPLITS["test"] for row in self.groups[name]}
        self.assertFalse(test_hashes & set(router["fit_image_sha256"]))
        for name in ("train", "val_known", "val_intra", "val_extra"):
            del self.groups[name]
        self.calls.clear()
        with patch.object(calibration, "calibrate", side_effect=AssertionError("TEST cannot fit thresholds")), \
                patch.object(pipeline.HierarchicalGeometry, "fit", side_effect=AssertionError("TEST cannot fit statistics")), \
                patch.object(pipeline.RobustScoreStandardizer, "fit", side_effect=AssertionError("TEST cannot fit scales")):
            pipeline.test_run(self.cfg, self.source, self.directory, self.device)
        self.assertEqual(self.calls, [("test", reference_protocol.STAGE_SPLITS["test"])])
        actual = fixture.read_records(self.directory / "test/predictions.jsonl")
        baseline = fixture.read_records(self.directory / "test/baseline_predictions.jsonl")
        for before, after in zip(baseline, actual):
            for field in ("image_sha256", "support_evidence", "log_probs", "candidate_parent", "candidate_leaf"):
                self.assertEqual(before[field], after[field], field)
            self.assertTrue(all(torch.isfinite(torch.tensor(after[field])) for field in (
                "geometry_parent_score", "geometry_leaf_score", "baseline_parent_z", "baseline_leaf_z")))
        aliases = [row for row in actual if row["image_sha256"] == self.groups["test_known"][0]["image_sha256"]]
        self.assertEqual([row["evaluation_weight"] for row in aliases], [1, 0])
        self.assertEqual(aliases[0]["geometry_leaf_score"], aliases[1]["geometry_leaf_score"])
        self.assertEqual(fixture.artifact_snapshot(self.source), source_bytes)

    def test_changed_source_binding_rejected_before_opening_dev(self):
        from taxosafe_geometry import pipeline
        from taxosafe_refine import importer
        self.make_source()
        self.configure_geometry()
        pipeline.fit(self.cfg, self.source, self.directory, self.device)
        path = self.source / "training/completed.json"
        receipt = reference_protocol.read_json(path)
        receipt["review_note"] = "internally valid changed source"
        reference_protocol.write_json(path, receipt)
        importer.inspect_reference(self.source)
        self.calls.clear()
        with self.assertRaises(ValueError):
            pipeline.calibrate_run(self.cfg, self.source, self.directory, self.device)
        self.assertEqual(self.calls, [])
        self.assertFalse((self.directory / "calibration").exists())

    def test_model_cache_and_scale_tampering_rejected_before_dev_images(self):
        from taxosafe_geometry import pipeline
        self.make_source()
        self.configure_geometry()
        pipeline.fit(self.cfg, self.source, self.directory, self.device)
        for name in ("frozen_geometry.pth", "cache.pth", "scales.json"):
            path = self.directory / "geometry" / name
            saved = path.read_bytes()
            try:
                path.write_bytes(saved + b"changed")
                self.calls.clear()
                with self.subTest(artifact=name), self.assertRaises(ValueError):
                    pipeline.calibrate_run(self.cfg, self.source, self.directory, self.device)
                self.assertEqual(self.calls, [])
            finally:
                path.write_bytes(saved)

    def test_content_aliases_share_one_visual_pass_and_aligned_features(self):
        from taxosafe_geometry import pipeline
        from taxosafe_refine import importer
        self.make_source()
        reference = importer.load_reference(self.source, self.device)
        before = reference.encoder.backbone.visual_calls
        groups = {"test_known": self.groups["test_known"]}
        rows, cached, timings = pipeline.collect_features(groups, reference, self.device)
        self.assertEqual(reference.encoder.backbone.visual_calls - before, 1)
        self.assertEqual(cached["test_known"]["fine"].shape, (4, 4))
        self.assertEqual(cached["test_known"]["parent"].shape, (4, 4))
        self.assertEqual(timings["test_known"]["visual_passes_per_unique_image"], 1)
        self.assertEqual(len(rows["test_known"]), 5)
        self.assertEqual(len(cached["test_known"]["image_sha256"]), 4)

    def test_coherently_rehashed_model_with_wrong_fit_configuration_is_rejected(self):
        from taxosafe_geometry import pipeline
        self.make_source()
        self.configure_geometry()
        pipeline.fit(self.cfg, self.source, self.directory, self.device)
        model_path = self.directory / "geometry/frozen_geometry.pth"
        state = pipeline.support_pipeline._load_torch(model_path)
        state["geometry"]["shrinkage"] = .9
        pipeline.support_pipeline._save_torch(model_path, state)
        receipt_path = self.directory / "geometry/completed.json"
        receipt = reference_protocol.read_json(receipt_path)
        receipt["model"]["sha256"] = reference_protocol.file_hash(model_path)
        reference_protocol.write_json(receipt_path, receipt)
        self.calls.clear()
        with self.assertRaisesRegex(ValueError, "statistics disagree"):
            pipeline.calibrate_run(self.cfg, self.source, self.directory, self.device)
        self.assertEqual(self.calls, [])

    def test_preflight_uses_receipts_without_opening_images_or_checkpoints(self):
        from taxosafe_geometry import pipeline
        self.make_source()
        argv = ["refine_taxosafe_geometry.py", "fit", "--reference-run-dir", str(self.source),
                "--run-dir", str(self.directory), "--preflight"]
        with patch("sys.argv", argv), patch.object(pipeline, "load_reference", side_effect=AssertionError("no tensors")):
            pipeline.run()
        self.assertEqual(self.calls, [])
        self.assertFalse(self.directory.exists())


if __name__ == "__main__":
    unittest.main()
