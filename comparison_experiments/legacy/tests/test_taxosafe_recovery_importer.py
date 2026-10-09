"""Restore a genuine tiny D05 run without fitting or consulting old TEST output.

The historical reference, CLIP caches, two verifier heads and DEV router all use
their production implementations. Only filesystem image input is synthetic.
Adversarial cases repair ordinary receipt hashes so semantic checks are tested,
rather than merely detecting a stale checksum.
"""
import copy
from pathlib import Path
import shutil
from unittest.mock import patch
import unittest

import torch

from tests import test_taxosafe_discovery_integration as discovery_fixture
from tests.test_taxosafe_refine_pipeline import artifact_snapshot
from taxosafe_discovery import backend, protocol, runner
from taxosafe_support import pipeline as support


ARM_ID = "D05_episode_bce"


class RecoveryImporterContracts(discovery_fixture.DiscoveryLifecycle):
    def setUp(self):
        super().setUp()
        self._prepare()
        backend.fit_arm(self.suite, ARM_ID, "cpu")
        runner._complete_stage(self.suite, ARM_ID, "training", self.snapshot)
        backend.calibrate_arm(self.suite, ARM_ID, "cpu")
        runner._complete_stage(self.suite, ARM_ID, "calibration", self.snapshot)
        from taxosafe_recovery import importer
        self.importer = importer
        self.training_dir = self.suite / "arms" / ARM_ID / "training"
        self.calibration_dir = self.suite / "arms" / ARM_ID / "calibration"
        self.calls.clear()

    def _remark(self, arm_id, stage):
        directory = runner._directory(self.suite, arm_id, stage)
        (directory / runner.STAGE_MARKER).unlink()
        runner._complete_stage(self.suite, arm_id, stage, self.snapshot)

    def _rebind_stage_artifact(self, arm_id, stage, key):
        directory = runner._directory(self.suite, arm_id, stage)
        receipt_path = directory / "completed.json"
        receipt = protocol.read_json(receipt_path)
        descriptor = receipt["artifacts"][key]
        descriptor["sha256"] = protocol.file_hash(directory / descriptor["path"])
        if key == "model":
            receipt["model"] = copy.deepcopy(descriptor)
        protocol.write_json(receipt_path, receipt)
        self._remark(arm_id, stage)
        return receipt

    def _rebind_model(self, payload):
        """Preserve a coherent receipt chain while changing a model's contents."""
        support._save_torch(self.training_dir / "model.pth", payload)
        trained = self._rebind_stage_artifact(ARM_ID, "training", "model")
        calibration = protocol.read_json(self.calibration_dir / "completed.json")
        calibration["model_sha256"] = trained["model"]["sha256"]
        calibration["training_receipt_sha256"] = protocol.file_hash(self.training_dir / "completed.json")
        protocol.write_json(self.calibration_dir / "completed.json", calibration)
        self._remark(ARM_ID, "calibration")

    def _rebind_train_cache(self, payload):
        path = self.suite / "cache/train/features.pth"
        support._save_torch(path, payload)
        cached = self._rebind_stage_artifact(None, "cache_train", "features")
        trained = protocol.read_json(self.training_dir / "completed.json")
        trained["train_cache_sha256"] = cached["artifacts"]["features"]["sha256"]
        protocol.write_json(self.training_dir / "completed.json", trained)
        model = support._load_torch(self.training_dir / "model.pth")
        model["train_cache_sha256"] = trained["train_cache_sha256"]
        self._rebind_model(model)

    def test_inspection_is_checkpoint_free_and_preserves_failed_gate_d05(self):
        before = artifact_snapshot(self.suite)
        with patch.object(support, "_load_torch", side_effect=AssertionError("Inspection loaded checkpoint")), \
                patch.object(torch, "load", side_effect=AssertionError("Inspection called torch.load")):
            info = self.importer.inspect_d05(self.suite)
        self.assertTrue({"directory", "config", "snapshot", "reference", "meta", "training",
                         "calibration", "router", "caches", "binding"}.issubset(info))
        self.assertEqual(Path(info["directory"]), self.suite)
        self.assertEqual(info["meta"], self.info["meta"])
        self.assertEqual(info["router"], protocol.read_json(self.calibration_dir / "router.json"))
        self.assertGreater(info["training"]["optimizer_steps"], 0)
        self.assertFalse(info["calibration"]["targets_passed"])
        self.assertEqual(set(info["caches"]), {"train", "development"})
        self.assertEqual(self.calls, [])
        self.assertEqual(artifact_snapshot(self.suite), before)

    def test_restoration_keeps_exact_frozen_heads_geometry_and_normalization(self):
        original = support._load_torch(self.training_dir / "model.pth")
        before = artifact_snapshot(self.suite)
        with patch("taxosafe_discovery.geometry.GeometryBank.fit", side_effect=AssertionError("Recovery refitted geometry")), \
                patch("taxosafe_discovery.verifier.SharedVerifier.fit", side_effect=AssertionError("Recovery refitted verifier")), \
                patch("taxosafe_discovery.verifier.build_episodes", side_effect=AssertionError("Recovery rebuilt TRAIN statistics")), \
                patch("taxosafe_discovery.calibration.fit_router", side_effect=AssertionError("Recovery recalibrated D05")):
            loaded = self.importer.load_d05(self.suite)
        self.assertEqual(loaded.binding, loaded.info["binding"])
        self.assertEqual(loaded.meta, self.info["meta"])
        self.assertEqual(loaded.router, protocol.read_json(self.calibration_dir / "router.json"))
        for level in ("leaf", "parent"):
            head = loaded.verifier.heads[level]
            self.assert_frozen(head, original["verifier"]["heads"][level])
            for key in ("mean", "scale"):
                value = loaded.verifier.normalization[level][key]
                self.assertTrue(torch.equal(value, original["verifier"]["normalization"][level][key]))
                self.assertFalse(value.requires_grad)
        for key in ("fine", "parent", "labels"):
            value = getattr(loaded.geometry, key)
            self.assertTrue(torch.equal(value, original["geometry"][key]))
            self.assertFalse(value.requires_grad)
        self.assertEqual(list(loaded.geometry.image_hashes), original["geometry"]["image_hashes"])
        self.assertEqual(artifact_snapshot(self.suite), before)
        self._assert_source_unchanged()

    def test_expected_binding_rejects_a_different_parent(self):
        expected = self.importer.inspect_d05(self.suite)["binding"]
        loaded = self.importer.load_d05(self.suite, expected_binding=expected)
        self.assertEqual(loaded.binding, expected)
        changed = copy.deepcopy(expected)
        changed["unexpected_parent"] = "different provenance"
        with self.assertRaises(ValueError):
            self.importer.load_d05(self.suite, expected_binding=changed)

    def test_bound_normalization_cannot_be_replaced_by_rewriting_all_receipts(self):
        expected = self.importer.inspect_d05(self.suite)["binding"]
        payload = support._load_torch(self.training_dir / "model.pth")
        payload["verifier"]["normalization"]["parent"]["mean"][0] += .25
        self._rebind_model(payload)
        with self.assertRaisesRegex(ValueError, "binding"):
            self.importer.load_d05(self.suite, expected_binding=expected)

    def test_known_train_and_development_cache_loading_uses_saved_inputs_only(self):
        loaded = self.importer.load_d05(self.suite)
        before = artifact_snapshot(self.suite)
        with patch.object(support, "make_loader", side_effect=AssertionError("Reopened image dataset")), \
                patch.object(support, "load_stage_rows", side_effect=AssertionError("Reopened split dataset")):
            for stage, owner in (("train", loaded), ("development", loaded.info)):
                with self.subTest(stage=stage):
                    cache, receipt = self.importer.load_parent_cache(owner, stage)
                    original = support._load_torch(self.suite / "cache" / stage / "features.pth")["cache"]
                    self.assertEqual(receipt, loaded.info["caches"][stage])
                    self.assertEqual(cache["meta"], loaded.meta)
                    self.assertEqual(set(cache["groups"]), set(original["groups"]))
                    for split, group in cache["groups"].items():
                        self.assertEqual(group["image_sha256"], original["groups"][split]["image_sha256"])
                        self.assertEqual(group["records"], original["groups"][split]["records"])
                        for feature, values in group["features"].items():
                            self.assertTrue(torch.equal(values, original["groups"][split]["features"][feature]))
        self.assertEqual(artifact_snapshot(self.suite), before)
        self.assertEqual(self.calls, [])

    def test_binary_and_calibration_file_tampering_rejected_before_loading(self):
        for path in (self.training_dir / "model.pth", self.calibration_dir / "router.json",
                     self.suite / "cache/train/features.pth", self.suite / "cache/development/features.pth"):
            original = path.read_bytes()
            try:
                path.write_bytes(original + b"\nchanged")
                with self.subTest(artifact=str(path.relative_to(self.suite))), \
                        patch.object(support, "_load_torch", side_effect=AssertionError("Loaded corrupted checkpoint")), \
                        self.assertRaises(ValueError):
                    self.importer.load_d05(self.suite)
            finally:
                path.write_bytes(original)

    def test_calibration_report_must_agree_with_predictions_and_receipt(self):
        report_path = self.calibration_dir / "report.json"
        report = protocol.read_json(report_path)
        report["counts"]["known_correct"] += 1
        protocol.write_json(report_path, report)
        self._rebind_stage_artifact(ARM_ID, "calibration", "report")
        with self.assertRaises(ValueError):
            self.importer.inspect_d05(self.suite)

    def test_invalid_saved_normalization_rejected_after_hash_chain_rebinding(self):
        payload = support._load_torch(self.training_dir / "model.pth")
        payload["verifier"]["normalization"]["leaf"]["scale"][0] = -1.
        self._rebind_model(payload)
        with self.assertRaisesRegex(ValueError, "normalization|Normalization"):
            self.importer.load_d05(self.suite)

    def test_train_identity_label_and_feature_changes_cannot_replace_geometry_support(self):
        # Restore bytes between cases, including all linked receipts/markers.
        files = {path: path.read_bytes() for path in self.suite.rglob("*") if path.is_file()}
        for kind in ("identity_order", "label", "feature"):
            with self.subTest(kind=kind):
                payload = support._load_torch(self.suite / "cache/train/features.pth")
                group = payload["cache"]["groups"]["train"]
                if kind == "identity_order":
                    # Keep the cache's alias indexing self-consistent, but move
                    # image identities relative to its already fitted geometry.
                    group["image_sha256"][0], group["image_sha256"][1] = group["image_sha256"][1], group["image_sha256"][0]
                    index = {identity: i for i, identity in enumerate(group["image_sha256"])}
                    group["record_feature_indices"] = [index[row["image_sha256"]] for row in group["records"]]
                elif kind == "label":
                    group["records"][0]["true_leaf"] = (group["records"][0]["true_leaf"] + 1) % len(self.info["meta"]["leaf_names"])
                else:
                    group["features"]["clip"][0] = torch.roll(group["features"]["clip"][0], 1)
                try:
                    self._rebind_train_cache(payload)
                    with self.assertRaises(ValueError):
                        self.importer.load_d05(self.suite)
                finally:
                    for path, content in files.items():
                        path.write_bytes(content)

    def test_old_test_artifacts_are_never_opened_during_parent_recovery(self):
        traps = (self.suite / "arms" / ARM_ID / "test/predictions.jsonl",
                 self.suite / "cache/test/features.pth", self.source / "test/predictions.jsonl")
        for path in traps:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"invalid old TEST artifact: never read")
        forbidden = set(traps)
        original_open = Path.open

        def guarded_open(path, *args, **kwargs):
            if path in forbidden:
                raise AssertionError("Old TEST artifact was opened: " + str(path))
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "open", guarded_open):
            info = self.importer.inspect_d05(self.suite)
            loaded = self.importer.load_d05(self.suite, expected_binding=info["binding"])
        self.assertEqual(loaded.binding, info["binding"])
        self.assertEqual(self.calls, [])

    def test_review_without_checkpoints_is_not_a_recoverable_source(self):
        review = self.root / "review_only"
        shutil.copytree(self.suite, review, ignore=shutil.ignore_patterns("*.pth"))
        with self.assertRaises(ValueError):
            self.importer.inspect_d05(review)

    def test_training_api_forbids_test_and_test_api_requires_frozen_dev(self):
        from taxosafe_recovery import protocol as recovery_protocol
        from taxosafe_recovery import reporting as recovery_reporting
        from taxosafe_recovery import runner as recovery_runner
        loaded = self.importer.load_d05(self.suite)
        with self.assertRaises(ValueError):
            self.importer.load_parent_cache(loaded, "test")
        with self.assertRaises(ValueError):
            self.importer.load_parent_cache(loaded, "validation")
        recovery = self.root / "new_recovery"
        recovery_runner._initialize(recovery, copy.deepcopy(recovery_protocol.DEFAULTS), loaded.info, "cpu")
        with patch.object(recovery_reporting, "freeze_dev_selection", side_effect=AssertionError("Missing DEV choice was auto-created")), \
                patch.object(backend, "_load_cache", side_effect=AssertionError("TEST features were read early")), \
                self.assertRaisesRegex(ValueError, "dev_selection.json"):
            self.importer.load_parent_test_cache(loaded, recovery)
        # Merely adding a filename cannot stand in for a completed DEV decision.
        protocol.write_json(recovery / "dev_selection.json", {})
        with patch.object(backend, "_load_cache", side_effect=AssertionError("TEST features were read after a forged DEV choice")), \
                self.assertRaises(ValueError):
            self.importer.load_parent_test_cache(loaded, recovery)


for _name in dir(discovery_fixture.DiscoveryLifecycle):
    if _name.startswith("test_") and _name not in RecoveryImporterContracts.__dict__:
        setattr(RecoveryImporterContracts, _name, None)


if __name__ == "__main__":
    unittest.main()
