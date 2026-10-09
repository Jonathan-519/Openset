"""Execute-suite failure isolation using explicitly synthetic stage artifacts."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from taxosafe_domain import protocol, reporting, runner
from tests import test_taxosafe_domain_reporting as fixtures


class DomainRunnerFailures(unittest.TestCase):
    _file = fixtures.DomainReporting._file
    _stage = fixtures.DomainReporting._stage

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.root = self.base / "suite"
        self.source = self.base / "discovery"
        self.source.mkdir()
        reference = self.base / "reference"
        reference.mkdir()
        self.info = dict(directory=self.source, reference={"directory":reference},
                         binding={"directory":str(self.source), "fixture":"synthetic runner I/O"})
        self.cfg = copy.deepcopy(protocol.DEFAULTS)
        self.calls, self.failures = [], set()
        for target, replacement in (("_source", lambda unused:copy.deepcopy(self.info)),
                                    ("_launch_stage", self._launch)):
            mocked = patch.object(runner, target, replacement)
            mocked.start()
            self.addCleanup(mocked.stop)
        quiet = patch("builtins.print")
        quiet.start()
        self.addCleanup(quiet.stop)

    def _launch(self, suite, arm_id, stage, device):
        self.calls.append((arm_id, stage))
        self.snapshot = protocol.read_json(suite / "snapshot.json")
        if stage in ("cache_test", "test"):
            self.assertTrue((suite / "dev_selection.json").is_file())
            frozen = reporting.freeze_dev_selection(suite)
            self.assertFalse(frozen["selection_uses_test"])
        logs = suite / "logs" / (arm_id or "cache")
        logs.mkdir(parents=True, exist_ok=True)
        stdout, stderr = logs / (stage + ".stdout.log"), logs / (stage + ".stderr.log")
        stdout.write_text("Synthetic execution-lifecycle test, no model training\n")
        stderr.write_text("")
        directory = runner._directory(suite, arm_id, stage)
        if (arm_id, stage) in self.failures:
            directory.mkdir(parents=True)
            (directory / "partial.json").write_text("{}")
            stderr.write_text("Intentional synthetic technical failure")
            return 2, stdout, stderr
        if arm_id is None:
            directory.mkdir(parents=True)
            receipt = dict(schema_version=protocol.SCHEMA_VERSION, stage=stage[6:],
                signature=self.snapshot["signature"], source_binding=self.snapshot["source_binding"],
                artifacts={"features":self._file(directory, "features.pth", b"synthetic cache")})
            protocol.write_json(directory / "completed.json", receipt)
            runner._complete_stage(suite, arm_id, stage, self.snapshot)
        else:
            self._stage(arm_id, stage, pass_gates=False, passed=False)
        return 0, stdout, stderr

    def _execute(self):
        return runner.execute_suite(self.cfg, self.source, self.root, device="cpu", run_preflight=False)

    def test_main_bank_training_failure_blocks_only_its_reuse_arms(self):
        self.failures.add(("H04_subspace_root", "training"))
        summary = self._execute()
        dependents = [arm["id"] for arm in self.cfg["arms"] if arm.get("weight_source") == "H04_subspace_root"]
        self.assertEqual(len(dependents), 6)
        for arm_id in dependents:
            self.assertNotIn((arm_id, "training"), self.calls)
            self.assertNotIn((arm_id, "test"), self.calls)
            self.assertEqual(summary["technical_failures"][arm_id]["stage"], "dependency")
        self.assertNotIn(("H04_subspace_root", "calibration"), self.calls)
        self.assertIn(("H11_dual_rank16", "training"), self.calls)
        self.assertIn(("H11_dual_rank16", "test"), self.calls)
        self.assertEqual(summary["completed_test_count"], 5)
        failed = next(row for row in summary["all_arms"] if row["arm_id"] == "H04_subspace_root")
        self.assertIsNone(failed["dev_known_end_to_end_leaf_accuracy"])
        self.assertIsNone(failed["test_known_end_to_end_leaf_accuracy"])

    def test_main_bank_calibration_failure_does_not_block_valid_weight_reuse(self):
        self.failures.add(("H04_subspace_root", "calibration"))
        summary = self._execute()
        self.assertNotIn(("H04_subspace_root", "test"), self.calls)
        self.assertTrue((self.root / "arms/H04_subspace_root/training/stage_binding.json").is_file())
        for arm in self.cfg["arms"]:
            if arm.get("weight_source") == "H04_subspace_root":
                self.assertIn((arm["id"], "training"), self.calls)
                self.assertIn((arm["id"], "calibration"), self.calls)
                self.assertIn((arm["id"], "test"), self.calls)
        self.assertEqual(summary["completed_test_count"], 11)
        self.assertEqual(set(summary["technical_failures"]), {"H04_subspace_root"})

    def test_failed_quality_gates_still_test_every_arm_after_dev_freeze(self):
        summary = self._execute()
        ids = [arm["id"] for arm in self.cfg["arms"]]
        index = self.calls.index((None, "cache_test"))
        self.assertEqual(sum(stage == "calibration" for _, stage in self.calls[:index]), 12)
        self.assertEqual(self.calls[index+1:], [(arm_id, "test") for arm_id in ids])
        self.assertEqual(summary["completed_calibration_count"], 12)
        self.assertEqual(summary["completed_test_count"], 12)
        self.assertEqual(summary["technical_failures"], {})
        self.assertTrue(all(row["dev_targets_passed"] is False for row in summary["all_arms"]))
        self.assertEqual(summary["recommendation_arm_id"], "H00_reference")
        self.assertFalse(summary["production_recommendation_uses_test"])


if __name__ == "__main__":
    unittest.main()
