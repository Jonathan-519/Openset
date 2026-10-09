"""Real nine-arm continuation of a receipt-compatible, actually trained D05."""
import copy
from pathlib import Path
from unittest.mock import patch
import unittest

import torch

from tests import test_taxosafe_discovery_integration as discovery_fixture
from tests import test_taxosafe_refine_pipeline as reference_fixture
from taxosafe_support import pipeline as support
from taxosafe_recovery import backend, importer, protocol, reporting, runner, training


class RecoveryLifecycle(discovery_fixture.DiscoveryLifecycle):
    def _recovery_json(self, arm_id, stage, filename="completed.json"):
        return protocol.read_json(self.recovery / "arms" / arm_id / stage / filename)

    def _recovery_model(self, arm_id):
        return support._load_torch(self.recovery / "arms" / arm_id / "training/model.pth")

    def test_real_d05_warm_start_nine_arms_no_test_fit_and_immutable_parent(self):
        # This executes the existing genuine tiny-CLIP/SGD discovery lifecycle;
        # its inherited I/O fixture is the only substitute for external images.
        discovery_fixture.DiscoveryLifecycle.test_eleven_real_arms_failed_gates_still_test_without_any_refitting(self)
        discovery = self.suite
        parent_files = reference_fixture.artifact_snapshot(discovery)
        parent = importer.load_d05(discovery)
        parent_digest = backend._semantic(parent.payload)
        parent_state = copy.deepcopy(parent.payload["verifier"])
        cfg = copy.deepcopy(protocol.DEFAULTS)
        cfg["training"].update(steps_per_head=2, batch_size=16)
        cfg = protocol.validate_config(cfg)
        self.recovery = self.root / "recovery"
        snapshot = runner._initialize(self.recovery, cfg, parent.info, "cpu")
        test_reads = []
        real_load = support._load_torch
        real_read_text = Path.read_text

        def guarded_load(path):
            path = Path(path)
            if path.resolve() == (discovery / "cache/test/features.pth").resolve():
                self.assertTrue((self.recovery / "dev_selection.json").is_file(),
                                "Parent TEST cache was opened before new DEV freeze")
                test_reads.append(str(path))
            return real_load(path)

        def no_parent_test_predictions(path, *args, **kwargs):
            try:
                relative = Path(path).resolve().relative_to(discovery.resolve())
            except ValueError:
                relative = None
            if relative is not None and "test" in relative.parts and relative.name in ("predictions.jsonl", "scores.jsonl"):
                raise AssertionError("Recovery must not parse old TEST predictions")
            return real_read_text(path, *args, **kwargs)

        with patch.object(support, "_load_torch", side_effect=guarded_load), \
                patch.object(Path, "read_text", new=no_parent_test_predictions):
            with self.assertRaises((ValueError, FileNotFoundError)):
                backend.prepare_cache(self.recovery, "test", "cpu")
            self.assertEqual(test_reads, [])
            for stage in ("train", "development"):
                receipt = backend.prepare_cache(self.recovery, stage, "cpu")
                self.assertEqual(receipt["timings"]["image_forward_count"], 0)
                runner._complete_stage(self.recovery, None, "cache_" + stage, snapshot)
            for arm in cfg["arms"]:
                arm_id = arm["id"]
                backend.fit_arm(self.recovery, arm_id, "cpu")
                runner._complete_stage(self.recovery, arm_id, "training", snapshot)
                backend.calibrate_arm(self.recovery, arm_id, "cpu")
                runner._complete_stage(self.recovery, arm_id, "calibration", snapshot)
                self.assertFalse(self._recovery_json(arm_id, "calibration")["targets_passed"], arm_id)
            self.assertEqual(test_reads, [])
            self.assertEqual(self._recovery_json("F00_reference", "calibration", "router.json"), parent.info["reference"]["router"])
            self.assertEqual(self._recovery_json("F01_d05", "calibration", "router.json"), parent.router)
            reproduction = self._recovery_json("F01_d05", "calibration", "report.json")["calibration_diagnostics"]["source_reproduction"]
            self.assertTrue(reproduction["exact_scores"])
            self.assertTrue(reproduction["exact_candidates"])
            self.assertTrue(reproduction["exact_terminal_outputs"])
            source_heads = {level: training.state_hash(value) for level, value in parent_state["heads"].items()}
            for arm_id in ("F03_continue", "F04_hard_positive", "F05_negative_anchor", "F06_l2sp"):
                payload = self._recovery_model(arm_id)
                state = payload["recovery_verifier"]
                self.assertEqual(backend._semantic(payload["parent_payload"]), parent_digest)
                self.assertEqual(training.state_hash(state["source_state"]), training.state_hash(parent_state))
                self.assertEqual(state["audit"]["source_head_sha256"], source_heads)
                self.assertEqual(state["audit"]["source_normalization_sha256"], state["audit"]["normalization_sha256"])
                self.assertEqual(state["audit"]["optimizer_steps"], 4)
                self.assertTrue(all(delta > 0 for delta in state["audit"]["parameter_delta_l2"].values()))
                audit = payload["fit_report"]["source_episode_reproduction"]
                self.assertTrue(audit["exact_source_episode_features"])
                self.assertTrue(audit["exact_source_episode_targets"])
                self.assertTrue(audit["exact_source_ranking_pairs"])
                self.assertEqual(audit["episode_seed"], parent.info["config"]["seed"])
            for base_id, reused_id in (("F01_d05", "F02_d05_buffer"), ("F06_l2sp", "F07_buffer")):
                original, reused = (self._recovery_json(arm_id, "training") for arm_id in (base_id, reused_id))
                self.assertEqual(original["model"], reused["model"])
                self.assertEqual(reused["optimizer_steps"], 0)
            f06 = self._recovery_model("F06_l2sp")["recovery_verifier"]
            guard = self._recovery_model("F08_leaf_guard")["recovery_verifier"]
            self.assertEqual(training.state_hash(guard["updated_heads"]["leaf"]), training.state_hash(f06["updated_heads"]["leaf"]))
            self.assertEqual(training.state_hash(guard["updated_heads"]["parent"]), source_heads["parent"])
            self.assertEqual(guard["audit"]["optimizer_steps"], 0)
            guard_report = self._recovery_json("F08_leaf_guard", "calibration", "report.json")
            self.assertTrue(all(guard_report["leaf_guard_preservation"].values()))
            selection = reporting.freeze_dev_selection(self.recovery)
            selection_digest = protocol.file_hash(self.recovery / "dev_selection.json")
            self.assertIsNone(selection["qualified_candidate_arm_id"])
            # Source TRAIN/DEV images were removed by the original lifecycle.
            # TEST additionally forbids every optimizer/statistics/router fit.
            with patch("taxosafe_recovery.training.finetune", side_effect=AssertionError("TEST continuation")), \
                    patch("taxosafe_recovery.training.make_leaf_guard", side_effect=AssertionError("TEST guard assembly")), \
                    patch("taxosafe_recovery.calibration.fit_router", side_effect=AssertionError("TEST router fit")), \
                    patch("taxosafe_recovery.calibration.crossfit_audit", side_effect=AssertionError("TEST OOF fit")), \
                    patch("taxosafe_discovery.geometry.GeometryBank.fit", side_effect=AssertionError("TEST geometry fit")), \
                    patch("taxosafe_discovery.verifier.build_episodes", side_effect=AssertionError("TEST episodes")), \
                    patch("taxosafe_discovery.verifier.SharedVerifier.fit", side_effect=AssertionError("TEST verifier fit")), \
                    patch("taxosafe_discovery.calibration.fit_router", side_effect=AssertionError("TEST old router fit")), \
                    patch.object(support, "make_loader", side_effect=AssertionError("TEST image re-encoding")):
                backend.prepare_cache(self.recovery, "test", "cpu")
                runner._complete_stage(self.recovery, None, "cache_test", snapshot)
                for arm in cfg["arms"]:
                    arm_id = arm["id"]
                    backend.test_arm(self.recovery, arm_id, "cpu")
                    runner._complete_stage(self.recovery, arm_id, "test", snapshot)
                    receipt = self._recovery_json(arm_id, "test")
                    self.assertFalse(receipt["calibration_gate_is_execution_gate"])
                    self.assertTrue(receipt["test_allowed_after_failed_gates"])
                    self.assertEqual(receipt["summary"]["counts"]["known"], 4)
                    rows = reference_fixture.read_records(self.recovery / "arms" / arm_id / "test/predictions.jsonl")
                    self.assertEqual(len(rows), 11)
                    self.assertEqual(sum(row["evaluation_weight"] for row in rows), 10)
                guard_test = self._recovery_json("F08_leaf_guard", "test", "report.json")
                self.assertTrue(all(guard_test["leaf_guard_preservation"].values()))
            self.assertGreater(len(test_reads), 0)
            self.assertEqual(protocol.file_hash(self.recovery / "dev_selection.json"), selection_digest)
            reporting.summarize_suite(self.recovery)
            self._assert_recovery_tamper_rejected(cfg, parent.info, snapshot)
        self.assertEqual(backend._semantic(parent.payload), parent_digest)
        self.assertEqual(reference_fixture.artifact_snapshot(discovery), parent_files)
        self._assert_source_unchanged()

    def _assert_recovery_tamper_rejected(self, cfg, info, snapshot):
        arm = next(arm for arm in cfg["arms"] if arm["id"] == "F03_continue")
        directory = self.recovery / "arms" / arm["id"] / "training"
        model_path, receipt_path = directory / "model.pth", directory / "completed.json"
        original, original_receipt = model_path.read_bytes(), receipt_path.read_bytes()
        try:
            value = support._load_torch(model_path)
            value["parent_payload"]["text"]["ensemble_leaf"][0, 0] += .01
            support._save_torch(model_path, value)
            receipt = protocol.read_json(receipt_path)
            receipt["model"]["sha256"] = protocol.file_hash(model_path)
            receipt["artifacts"]["model"] = copy.deepcopy(receipt["model"])
            protocol.write_json(receipt_path, receipt)
            with self.assertRaisesRegex(ValueError, "frozen D05"):
                backend._load_model(self.recovery, arm, cfg, info)
        finally:
            model_path.write_bytes(original)
            receipt_path.write_bytes(original_receipt)
        runner._verify_stage(self.recovery, arm["id"], "training", snapshot)


for _name in dir(discovery_fixture.DiscoveryLifecycle):
    if _name.startswith("test_") and _name not in RecoveryLifecycle.__dict__:
        setattr(RecoveryLifecycle, _name, None)


if __name__ == "__main__":
    unittest.main()
