"""Lifecycle/report tests with synthetic stage outputs, not neural experiments."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from taxosafe_discovery import protocol, reporting, runner
from taxosafe_support import calibration as base


META = {"leaf_names": ["a", "b"], "parent_names": ["P"], "leaf_to_parent": [0, 0]}


def predictions(stage, pass_gates=False):
    rows = []
    for status in ("known", "intra", "extra"):
        for i in range(2):
            name = stage + "_" + status + "_" + str(i)
            kind = "known" if status == "known" else "intra_unknown" if status == "intra" and pass_gates else "global_unknown"
            rows.append({"image_sha256": hashlib.sha256(name.encode()).hexdigest(),
                "path": name + ".jpg", "split": stage + "_" + status, "status": status,
                "source": "a" if status == "known" and i == 0 else "b" if status == "known" else status + str(i),
                "true_leaf": i if status == "known" else None,
                "true_parent": 0 if status != "extra" else None,
                "prediction_type": kind, "parent": 0 if kind != "global_unknown" else None,
                "leaf": i if kind == "known" else None,
                "candidate_parent": 0, "candidate_leaf": i if status == "known" else 0})
    return rows + [dict(rows[0], path="alias.jpg")]


class DiscoveryLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source_dir = self.root / "reference"
        self.source_dir.mkdir()
        (self.source_dir / "weights.pth").write_bytes(b"unchanged source fixture")
        self.suite = self.root / "suite"
        self.cfg = copy.deepcopy(protocol.DEFAULTS)
        self.source = {"directory": self.source_dir, "binding": {"directory": str(self.source_dir), "source": "fixed"}}
        self.calls, self.failures, self.no_marker, self.pass_arms = [], set(), set(), set()
        self.crossfit_failed = set()
        self.code = "fixed test code"
        for target, replacement in (("_source", lambda unused: copy.deepcopy(self.source)), ("_launch_stage", self.launch)):
            entered = patch.object(runner, target, replacement)
            entered.start()
            self.addCleanup(entered.stop)
        entered = patch.object(protocol, "signature", side_effect=lambda cfg, binding: {
            "method": protocol.SCHEMA_VERSION, "config": protocol.object_hash(cfg),
            "reference": protocol.object_hash(binding), "code": self.code})
        entered.start()
        self.addCleanup(entered.stop)
        entered = patch("builtins.print")
        entered.start()
        self.addCleanup(entered.stop)

    def write(self, directory, name, value):
        path = directory / name
        if isinstance(value, bytes):
            path.write_bytes(value)
        elif name.endswith(".jsonl"):
            protocol.write_records(path, value)
        else:
            protocol.write_json(path, value)
        return {"path": name, "sha256": protocol.file_hash(path)}

    def launch(self, suite, arm_id, stage, device):
        self.calls.append((arm_id, stage))
        if stage in ("cache_test", "test"):
            self.assertTrue((suite / "dev_selection.json").is_file())
        logs = suite / "logs" / (arm_id or "cache")
        logs.mkdir(parents=True, exist_ok=True)
        stdout, stderr = logs / (stage + ".stdout.log"), logs / (stage + ".stderr.log")
        stdout.write_text("synthetic lifecycle fixture\n")
        stderr.write_text("synthetic technical stage failure")
        directory = runner._directory(suite, arm_id, stage)
        directory.mkdir(parents=True)
        if (arm_id, stage) in self.failures:
            (directory / "partial.json").write_text("{}")
            return 2, stdout, stderr
        snapshot = protocol.read_json(suite / "snapshot.json")
        receipt = {"schema_version": protocol.SCHEMA_VERSION, "stage": stage[6:] if arm_id is None else stage,
            "arm_id": arm_id, "signature": snapshot["signature"], "source_binding": snapshot["source_binding"],
            "test_used_for_fitting": False, "artifacts": {}}
        if arm_id is None:
            receipt["artifacts"]["features"] = self.write(directory, "features.pth", b"synthetic cache fixture")
        elif stage == "training":
            source = "D06_episode_rank" if arm_id == "D07_parentwise" else arm_id
            receipt.update(optimizer_steps=4 if arm_id in ("D05_episode_bce", "D06_episode_rank", "D08_residual", "D09_coop", "D10_forced_prompt") else 0,
                           weight_source=source, fit_report={"synthetic_lifecycle_fixture": True})
            receipt["model"] = self.write(directory, "model.pth", (source + " synthetic model").encode())
            receipt["artifacts"]["model"] = receipt["model"]
        else:
            rows = predictions(stage, arm_id in self.pass_arms)
            report = dict(base.evaluate_records(rows, META), schema_version="discovery_evaluation_v1",
                crossfit_audit={"passed": arm_id not in self.crossfit_failed})
            receipt.update(meta=META, summary=report, targets_passed=report["targets_passed"])
            for key, name, value in (("predictions", "predictions.jsonl", rows), ("scores", "scores.jsonl", rows),
                                      ("report", "report.json", report), ("router", "router.json", {"frozen": True})):
                receipt["artifacts"][key] = self.write(directory, name, value)
            if stage == "test":
                receipt["calibration_receipt_sha256"] = protocol.file_hash(suite / "arms" / arm_id / "calibration/completed.json")
        protocol.write_json(directory / "completed.json", receipt)
        if (arm_id, stage) not in self.no_marker:
            runner._complete_stage(suite, arm_id, stage, snapshot)
        return 0, stdout, stderr

    def execute(self, **kwargs):
        return runner.execute_suite(self.cfg, self.source_dir, self.suite, device="cpu", run_preflight=False, **kwargs)

    def test_failed_gate_all_arms_test_after_dev_freeze_and_test_cache(self):
        summary = self.execute()
        ids = [a["id"] for a in self.cfg["arms"]]
        self.assertEqual(self.calls[:2], [(None, "cache_train"), (None, "cache_development")])
        index = self.calls.index((None, "cache_test"))
        self.assertEqual(sum(stage == "calibration" for _, stage in self.calls[:index]), len(ids))
        self.assertEqual(self.calls[index + 1:], [(arm_id, "test") for arm_id in ids])
        self.assertEqual(summary["completed_test_count"], len(ids))
        self.assertTrue(all(not row["dev_targets_passed"] for row in summary["all_arms"]))
        self.assertEqual(summary["recommendation_arm_id"], ids[0])
        self.assertEqual(summary["diagnostics"]["test"]["identical_prediction_groups"], [ids])
        self.assertTrue(all(v["unique_image_count"] == 6 for v in summary["diagnostics"]["test"]["fingerprints"].values()))
        self.assertEqual(len(summary["diagnostics"]["test"]["all_pairwise"]), len(ids) * (len(ids) - 1) // 2)
        self.assertEqual((self.source_dir / "weights.pth").read_bytes(), b"unchanged source fixture")

    def test_training_failure_blocks_only_its_dependency(self):
        self.failures.add(("D06_episode_rank", "training"))
        summary = self.execute()
        self.assertNotIn(("D06_episode_rank", "calibration"), self.calls)
        self.assertNotIn(("D07_parentwise", "training"), self.calls)
        self.assertIn(("D10_forced_prompt", "test"), self.calls)
        self.assertEqual(summary["completed_test_count"], len(self.cfg["arms"]) - 2)
        self.assertEqual(summary["technical_failures"]["D07_parentwise"]["stage"], "dependency")

    def test_calibration_failure_does_not_block_weight_reuse_and_no_fake_metrics(self):
        self.failures.add(("D06_episode_rank", "calibration"))
        summary = self.execute()
        self.assertIn(("D07_parentwise", "training"), self.calls)
        self.assertIn(("D07_parentwise", "test"), self.calls)
        row = next(r for r in summary["all_arms"] if r["arm_id"] == "D06_episode_rank")
        self.assertIsNone(row["dev_known_end_to_end_leaf_accuracy"])
        self.assertIsNone(row["test_known_end_to_end_leaf_accuracy"])

    def test_bad_zero_exit_completed_receipt_is_failure_even_if_file_exists(self):
        self.no_marker.add(("D01_source_rmd", "calibration"))
        summary = self.execute()
        self.assertNotIn(("D01_source_rmd", "test"), self.calls)
        self.assertEqual(summary["completed_test_count"], len(self.cfg["arms"]) - 1)
        self.assertIn("verified completion", summary["technical_failures"]["D01_source_rmd"]["error"])

    def test_baseline_test_failure_keeps_every_other_test_and_dev_recommendation(self):
        self.failures.add(("D00_reference", "test"))
        summary = self.execute()
        self.assertEqual(sum(stage == "test" for _, stage in self.calls), len(self.cfg["arms"]))
        self.assertEqual(summary["recommendation_arm_id"], "D00_reference")
        row = summary["all_arms"][0]
        self.assertIsNone(row["test_known_end_to_end_leaf_accuracy"])
        self.assertIsNotNone(row["dev_known_end_to_end_leaf_accuracy"])

    def test_shared_train_cache_failure_exports_every_arm_without_fake_outputs(self):
        self.failures.add((None, "cache_train"))
        summary = self.execute()
        self.assertEqual(summary["completed_calibration_count"], 0)
        self.assertEqual(summary["completed_test_count"], 0)
        self.assertEqual(len(summary["all_arms"]), len(self.cfg["arms"]))
        self.assertFalse((self.suite / "cache/test").exists())
        self.assertTrue(all(r["optimizer_steps"] is None and r["dev_targets_passed"] is None and r["test_targets_passed"] is None for r in summary["all_arms"]))
        self.assertIsNone(summary["best_exploratory_test_arm"])

    def test_test_cache_failure_keeps_calibrations_and_null_test_metrics(self):
        self.failures.add((None, "cache_test"))
        summary = self.execute()
        self.assertEqual(summary["completed_calibration_count"], len(self.cfg["arms"]))
        self.assertEqual(summary["completed_test_count"], 0)
        self.assertTrue(all(r["test_targets_passed"] is None for r in summary["all_arms"]))

    def test_complete_resume_validates_without_relaunch(self):
        self.execute()
        calls = list(self.calls)
        self.execute(resume=True)
        self.assertEqual(calls, self.calls)

    def test_resume_rejects_cache_or_model_tampering(self):
        self.execute()
        for path in (self.suite / "cache/train/features.pth", self.suite / "arms/D07_parentwise/training/model.pth"):
            original = path.read_bytes()
            path.write_bytes(original + b"tampered")
            try:
                with self.assertRaisesRegex(ValueError, "artifacts changed"):
                    self.execute(resume=True)
            finally:
                path.write_bytes(original)

    def test_source_code_change_is_fatal_not_an_arm_failure(self):
        original = self.launch
        def changing(*args):
            result = original(*args)
            self.code = "changed"
            return result
        with patch.object(runner, "_launch_stage", side_effect=changing):
            with self.assertRaisesRegex(ValueError, "snapshot changed"):
                self.execute()
        self.assertEqual(len(self.calls), 1)
        self.assertFalse((self.suite / "dev_selection.json").exists())

    def test_dev_qualification_requires_crossfit_and_selection_stays_frozen(self):
        self.pass_arms.update(("D01_source_rmd", "D02_clip_rmd"))
        self.crossfit_failed.add("D01_source_rmd")
        summary = self.execute()
        self.assertEqual(summary["recommendation_arm_id"], "D02_clip_rmd")
        frozen = (self.suite / "dev_selection.json").read_bytes()
        reporting.freeze_dev_selection(self.suite)
        self.assertEqual(frozen, (self.suite / "dev_selection.json").read_bytes())
        value = json.loads(frozen)
        value["recommendation_arm_id"] = "D10_forced_prompt"
        protocol.write_json(self.suite / "dev_selection.json", value)
        with self.assertRaisesRegex(ValueError, "Immutable DEV"):
            reporting.freeze_dev_selection(self.suite)

    def test_freeze_rejects_opened_test_cache_when_decision_absent(self):
        self.execute()
        (self.suite / "dev_selection.json").unlink()
        with self.assertRaisesRegex(ValueError, "before creating any TEST cache"):
            reporting.freeze_dev_selection(self.suite)


if __name__ == "__main__":
    unittest.main()
