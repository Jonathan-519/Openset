"""Lifecycle tests: every evaluable arm tests, after a frozen DEV decision."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import taxosafe_sweep
from taxosafe_sweep import protocol, runner


class SweepRunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source_dir = self.root / "reference"
        self.source_dir.mkdir()
        (self.source_dir / "untouched.txt").write_text("frozen source", encoding="utf-8")
        self.suite = self.root / "suite"
        self.cfg = {"name": "lifecycle_test", "seed": 1,
                    "budget": copy.deepcopy(protocol.DEFAULT_BUDGET), "arms": copy.deepcopy(protocol.ARMS)}
        self.source = {"directory": self.source_dir, "binding": {"directory": str(self.source_dir), "hash": "source"},
                       "config": {}, "meta": {}, "audit": {}, "calibration": {"audit": {}},
                       "training": {"audit": {}}}
        self.calls = []
        self.failures = set()
        self.exit_without_receipt = set()
        self.frozen = False
        self.code = "stable code"
        self.source_before = self._source_files()
        silent = patch("builtins.print")
        silent.start()
        self.addCleanup(silent.stop)
        reporter = SimpleNamespace(freeze_dev_selection=self.freeze, summarize_suite=self.summarize)
        for target, value in (("_source", lambda unused: copy.deepcopy(self.source)),
                              ("_launch_stage", self.launch)):
            entered = patch.object(runner, target, value)
            entered.start()
            self.addCleanup(entered.stop)
        entered = patch.object(protocol, "signature", self.signature)
        entered.start()
        self.addCleanup(entered.stop)
        entered = patch.object(taxosafe_sweep, "reporting", reporter, create=True)
        entered.start()
        self.addCleanup(entered.stop)

    def signature(self, cfg, source):
        return {"config": protocol.object_hash(cfg), "reference": protocol.object_hash(source), "code": self.code}

    def _source_files(self):
        return {str(p.relative_to(self.source_dir)): p.read_bytes() for p in self.source_dir.rglob("*") if p.is_file()}

    def launch(self, suite, arm, stage, device):
        self.calls.append((arm, stage))
        if stage == "test":
            self.assertTrue(self.frozen, "No TEST may precede DEV freezing")
        logs = suite / "logs" / arm
        logs.mkdir(parents=True, exist_ok=True)
        stdout, stderr = logs / (stage + ".stdout.log"), logs / (stage + ".stderr.log")
        stdout.write_text("real subprocess replacement for lifecycle testing\n", encoding="utf-8")
        stderr.write_text("synthetic technical failure" if (arm, stage) in self.failures else "", encoding="utf-8")
        output = suite / "arms" / arm / stage
        output.mkdir(parents=True)
        if (arm, stage) in self.failures:
            (output / "partial.json").write_text("{}", encoding="utf-8")
            return 2, stdout, stderr
        if (arm, stage) in self.exit_without_receipt:
            return 0, stdout, stderr
        protocol.write_json(output / "completed.json", {"arm_id": arm, "stage": stage,
                            "targets_passed": False, "calibration_status": "targets_unmet"})
        protocol.write_json(output / "evidence.json", {"stage": stage, "arm": arm})
        snapshot = protocol.read_json(suite / "snapshot.json")
        runner._complete_stage(suite, arm, stage, snapshot)
        return 0, stdout, stderr

    def freeze(self, suite):
        inputs = {}
        for arm in self.cfg["arms"]:
            directory = suite / "arms" / arm["id"]
            calibration = directory / "calibration" / "completed.json"
            failure = directory / "failure.json"
            self.assertTrue(calibration.is_file() or failure.is_file(), "Every arm must finish before freezing")
            inputs[arm["id"]] = protocol.file_hash(calibration if calibration.is_file() else failure)
        value = {"calibration_inputs": inputs, "test_used_for_selection": False}
        target = suite / "dev_selection.json"
        if target.exists():
            self.assertEqual(value, protocol.read_json(target))
        else:
            protocol.write_json(target, value)
        self.frozen = True
        return value

    def summarize(self, suite, phase="complete"):
        self.assertTrue(self.frozen)
        return {"phase": phase, "test_used_for_selection": False}

    def execute(self, **kwargs):
        return runner.execute_suite(self.cfg, self.source_dir, self.suite, device="cpu", run_preflight=False, **kwargs)

    def test_all_six_failed_gate_arms_test_after_all_calibrations(self):
        self.execute()
        self.assertEqual(17, len(self.calls))  # five training, six DEV, six TEST
        first_test = next(i for i, (_, stage) in enumerate(self.calls) if stage == "test")
        self.assertEqual(6, sum(stage == "calibration" for _, stage in self.calls[:first_test]))
        self.assertEqual([a["id"] for a in self.cfg["arms"]], [arm for arm, stage in self.calls if stage == "test"])
        receipt = protocol.read_json(self.suite / "suite_completed.json")
        self.assertTrue(receipt["all_valid_calibrations_test_attempted_regardless_of_gate"])
        self.assertEqual(6, len(receipt["completed_test_arms"]))
        self.assertEqual([], receipt["technical_failure_arms"])
        self.assertEqual(self.source_before, self._source_files())

    def test_calibration_technical_failure_does_not_block_trained_parent_blend(self):
        self.failures.add(("E04_hierarchy_anchor", "calibration"))
        self.execute()
        self.assertNotIn(("E04_hierarchy_anchor", "test"), self.calls)
        self.assertIn(("E05_hierarchy_blend", "training"), self.calls)
        self.assertIn(("E05_hierarchy_blend", "test"), self.calls)
        failure = protocol.read_json(self.suite / "arms/E04_hierarchy_anchor/failure.json")
        self.assertTrue(failure["test_blocked"])
        self.assertIn("synthetic technical failure", failure["error"])

    def test_training_failure_blocks_only_its_arm_and_dependent_blend(self):
        self.failures.add(("E04_hierarchy_anchor", "training"))
        self.execute()
        self.assertNotIn(("E04_hierarchy_anchor", "calibration"), self.calls)
        self.assertNotIn(("E05_hierarchy_blend", "training"), self.calls)
        self.assertEqual(4, sum(stage == "test" for _, stage in self.calls))
        failure = protocol.read_json(self.suite / "arms/E05_hierarchy_blend/failure.json")
        self.assertIn("Parent training is unavailable", failure["error"])

    def test_test_failure_is_exported_but_other_arms_still_test(self):
        self.failures.add(("E00_reference", "test"))
        self.execute()
        self.assertEqual(6, sum(stage == "test" for _, stage in self.calls))
        receipt = protocol.read_json(self.suite / "suite_completed.json")
        self.assertEqual(5, len(receipt["completed_test_arms"]))
        self.assertEqual(["E00_reference"], receipt["technical_failure_arms"])
        failure = protocol.read_json(self.suite / "arms/E00_reference/failure.json")
        self.assertFalse(failure["test_blocked"])

    def test_zero_exit_without_completed_receipt_is_arm_failure(self):
        self.exit_without_receipt.add(("E01_heads", "training"))
        self.execute()
        self.assertNotIn(("E01_heads", "test"), self.calls)
        self.assertIn(("E05_hierarchy_blend", "test"), self.calls)
        failure = protocol.read_json(self.suite / "arms/E01_heads/failure.json")
        self.assertIn("verified completion", failure["error"])

    def test_completed_resume_revalidates_and_never_relaunches(self):
        self.execute()
        before = list(self.calls)
        self.execute(resume=True)
        self.assertEqual(before, self.calls)

    def test_resume_rejects_modified_stage_artifact(self):
        self.execute()
        (self.suite / "arms/E01_heads/training/evidence.json").write_text("{}", encoding="utf-8")
        before = list(self.calls)
        with self.assertRaisesRegex(ValueError, "artifacts changed"):
            self.execute(resume=True)
        self.assertEqual(before, self.calls)

    def test_resume_rejects_partial_stage_and_failed_arm(self):
        runner._initialize(self.suite, self.cfg, self.source, "cpu")
        partial = self.suite / "arms/E00_reference/calibration"
        partial.mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            self.execute(resume=True)
        runner._failure(self.suite, "E00_reference", "calibration", "failure")
        with self.assertRaisesRegex(ValueError, "technically failed"):
            self.execute(resume=True)

    def test_source_or_code_change_aborts_instead_of_becoming_arm_failure(self):
        original = self.launch
        def changing_launch(*args):
            result = original(*args)
            self.code = "changed code"
            return result
        with patch.object(runner, "_launch_stage", changing_launch):
            with self.assertRaisesRegex(ValueError, "snapshot changed"):
                self.execute()
        self.assertEqual(1, len(self.calls))
        self.assertFalse((self.suite / "arms/E00_reference/failure.json").exists())
        self.assertFalse((self.suite / "dev_selection.json").exists())

    def test_reference_and_output_must_be_disjoint(self):
        for target in (self.source_dir, self.source_dir / "child", self.source_dir.parent):
            with self.assertRaisesRegex(ValueError, "non-nested"):
                runner._destinations(self.source_dir, target)
        self.assertEqual(self.source_before, self._source_files())

    def test_manifest_does_not_skip_nested_files_named_like_marker(self):
        stage = self.root / "manifest_test"
        (stage / "nested").mkdir(parents=True)
        protocol.write_json(stage / "completed.json", {})
        protocol.write_json(stage / runner.STAGE_MARKER, {})
        protocol.write_json(stage / "nested" / runner.STAGE_MARKER, {"must": "be checked"})
        manifest = runner._manifest(stage)
        self.assertNotIn(runner.STAGE_MARKER, manifest)
        self.assertIn("nested/" + runner.STAGE_MARKER, manifest)

    def test_device_is_part_of_immutable_resume_snapshot(self):
        runner._initialize(self.suite, self.cfg, self.source, "cuda")
        with self.assertRaisesRegex(ValueError, "Device differs"):
            self.execute(resume=True)
        self.assertFalse(self.calls)

    def test_preflight_does_not_create_run_or_read_predictions(self):
        audit = {"count": 1, "unique_image_count": 1, "image_hashes": ["a"], "sources": ["A"]}
        self.source["training"]["audit"] = {"train": audit, "val_known": audit}
        with patch.object(runner, "_device_check", return_value={"device": "cpu"}), \
             patch("taxosafe_support.pipeline.load_stage_rows", return_value=({}, {"train": audit, "val_known": audit})), \
             patch("taxosafe_refine.pipeline._stage_rows", side_effect=[({}, {"val_extra": audit}), ({}, {"test_extra": audit})]):
            report = runner.preflight(self.cfg, self.source_dir, self.suite, "cpu")
        self.assertFalse(report["destination_created"])
        self.assertFalse(report["model_forward_performed"])
        self.assertFalse(report["test_predictions_read"])
        self.assertFalse(self.suite.exists())
        self.assertEqual(self.source_before, self._source_files())


class SweepPreflightErrors(unittest.TestCase):
    def test_missing_weights_error_is_actionable_and_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "text review archive cannot replace model weights"):
                runner._source(directory)
            self.assertEqual([], list(Path(directory).iterdir()))

    def test_unavailable_cuda_does_not_silently_use_cpu(self):
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(ValueError, "CUDA is unavailable"):
                runner._device_check("cuda")
            self.assertEqual("cpu", runner._device_check("cpu")["device"])


if __name__ == "__main__":
    unittest.main()
