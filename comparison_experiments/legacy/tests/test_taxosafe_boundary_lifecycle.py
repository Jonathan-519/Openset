"""Lifecycle/report tests with synthetic stage outputs, not neural experiments."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from taxosafe_boundary import protocol, reporting, runner
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


class BoundaryLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source_dir = self.root / "reference"
        self.source_dir.mkdir()
        (self.source_dir / "weights.pth").write_bytes(b"unchanged source fixture")
        self.suite = self.root / "suite"
        self.cfg = copy.deepcopy(protocol.DEFAULTS)
        self.source = {"directory": self.source_dir, "reference": {"directory": self.root / "original_reference"},
            "binding": {"directory": str(self.source_dir), "source": "fixed"}}
        self.calls, self.failures, self.no_marker, self.pass_arms = [], set(), set(), set()
        self.crossfit_failed, self.recovery_oof_pass, self.known_reject_arms = set(), set(), set()
        self.invalid_report = set()
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
            arm = next(a for a in self.cfg["arms"] if a["id"] == arm_id)
            source = arm.get("weight_source", arm_id)
            receipt.update(optimizer_steps=4 if arm_id in ("G04_bce8", "G05_rank8", "G06_bce9", "G07_rank9") else 0,
                           weight_source=source, fit_report={"synthetic_lifecycle_fixture": True})
            receipt["model"] = self.write(directory, "model.pth", (source + " synthetic model").encode())
            receipt["artifacts"]["model"] = receipt["model"]
        else:
            rows = predictions(stage, arm_id in self.pass_arms)
            if arm_id in self.known_reject_arms:
                for row in rows:
                    if row["status"] == "known" and row["true_leaf"] == 0:
                        row.update(prediction_type="global_unknown", parent=None, leaf=None)
            report = dict(base.evaluate_records(rows, META), schema_version="boundary_evaluation_v1",
                crossfit_audit={"passed": arm_id not in self.crossfit_failed, "recovery_passed": arm_id in self.recovery_oof_pass})
            report["boundary_policy"] = {"name": "kp_frontier", "selected_tier": "best_effort",
                "reason": "synthetic fixture", "feasible_counts": {"strict": 0}}
            if (arm_id, stage) in self.invalid_report:
                report["counts"]["known_correct"] += 1
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
        self.failures.add(("G07_rank9", "training"))
        summary = self.execute()
        self.assertNotIn(("G07_rank9", "calibration"), self.calls)
        self.assertNotIn(("G08_rank9_standard", "training"), self.calls)
        self.assertNotIn(("G09_rank9_leaf_guard", "training"), self.calls)
        self.assertIn(("G06_bce9", "test"), self.calls)
        self.assertEqual(summary["completed_test_count"], len(self.cfg["arms"]) - 3)
        self.assertEqual(summary["technical_failures"]["G08_rank9_standard"]["stage"], "dependency")

    def test_calibration_failure_does_not_block_weight_reuse_and_no_fake_metrics(self):
        self.failures.add(("G07_rank9", "calibration"))
        summary = self.execute()
        self.assertIn(("G08_rank9_standard", "training"), self.calls)
        self.assertIn(("G08_rank9_standard", "test"), self.calls)
        row = next(r for r in summary["all_arms"] if r["arm_id"] == "G07_rank9")
        self.assertIsNone(row["dev_known_end_to_end_leaf_accuracy"])
        self.assertIsNone(row["test_known_end_to_end_leaf_accuracy"])

    def test_bad_zero_exit_completed_receipt_is_failure_even_if_file_exists(self):
        self.no_marker.add(("G02_d05_kp", "calibration"))
        summary = self.execute()
        self.assertNotIn(("G02_d05_kp", "test"), self.calls)
        self.assertEqual(summary["completed_test_count"], len(self.cfg["arms"]) - 1)
        self.assertIn("verified completion", summary["technical_failures"]["G02_d05_kp"]["error"])

    def test_baseline_test_failure_keeps_every_other_test_and_dev_recommendation(self):
        self.failures.add(("G00_reference", "test"))
        summary = self.execute()
        self.assertEqual(sum(stage == "test" for _, stage in self.calls), len(self.cfg["arms"]))
        self.assertEqual(summary["recommendation_arm_id"], "G00_reference")
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
        for path in (self.suite / "cache/train/features.pth", self.suite / "arms/G08_rank9_standard/training/model.pth"):
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
        self.pass_arms.update(("G02_d05_kp", "G04_bce8"))
        self.crossfit_failed.add("G02_d05_kp")
        summary = self.execute()
        self.assertEqual(summary["recommendation_arm_id"], "G04_bce8")
        frozen = (self.suite / "dev_selection.json").read_bytes()
        reporting.freeze_dev_selection(self.suite)
        self.assertEqual(frozen, (self.suite / "dev_selection.json").read_bytes())
        value = json.loads(frozen)
        value["recommendation_arm_id"] = "G09_rank9_leaf_guard"
        protocol.write_json(self.suite / "dev_selection.json", value)
        with self.assertRaisesRegex(ValueError, "Immutable DEV"):
            reporting.freeze_dev_selection(self.suite)

    def test_research_recovery_does_not_replace_failed_deployment_gates(self):
        self.known_reject_arms.add("G01_d05")
        self.recovery_oof_pass.add("G04_bce8")
        summary = self.execute()
        self.assertEqual(summary["recommendation_arm_id"], "G00_reference")
        self.assertEqual(summary["research_recovery_arm_id"], "G04_bce8")
        self.assertFalse(summary["research_recovery_is_deployment_recommendation"])
        entries = {r["arm_id"]: r for r in summary["dev_selection"]["exploratory_development_ranking"]}
        self.assertTrue(entries["G04_bce8"]["research_recovery_qualified"])
        self.assertFalse(entries["G04_bce8"]["qualified"])
        self.assertTrue(summary["parent_chosen_after_prior_test_review"])
        for phase in ("development", "test"):
            self.assertEqual(set(summary["diagnostics"][phase]["paired_to_d05"]), {a["id"] for a in self.cfg["arms"]})
        with (self.suite / "per_species_comparison.csv").open(encoding="utf-8-sig") as handle:
            self.assertEqual({row["comparison_baseline"] for row in __import__("csv").DictReader(handle)},
                             {"G00_reference", "G01_d05"})

    def test_research_recovery_requires_its_own_out_of_fold_guard(self):
        self.known_reject_arms.add("G01_d05")
        summary = self.execute()
        self.assertIsNone(summary["research_recovery_arm_id"])
        self.assertEqual(summary["completed_test_count"], len(self.cfg["arms"]))

    def test_freeze_rejects_opened_test_cache_when_decision_absent(self):
        self.execute()
        (self.suite / "dev_selection.json").unlink()
        with self.assertRaisesRegex(ValueError, "before creating any TEST cache"):
            reporting.freeze_dev_selection(self.suite)

    def test_inconsistent_saved_metrics_are_an_isolated_technical_failure(self):
        self.invalid_report.add(("G07_rank9", "calibration"))
        summary = self.execute()
        self.assertNotIn(("G07_rank9", "test"), self.calls)
        self.assertIn(("G08_rank9_standard", "test"), self.calls)
        failed = next(r for r in summary["all_arms"] if r["arm_id"] == "G07_rank9")
        self.assertIsNone(failed["dev_known_end_to_end_leaf_accuracy"])
        self.assertEqual(summary["completed_test_count"], len(self.cfg["arms"]) - 1)

    def test_matrix_policy_macro_and_time_evidence_are_exported(self):
        summary = self.execute()
        self.assertEqual(len(summary["experiment_matrix"]), len(self.cfg["arms"]))
        self.assertTrue((self.suite / "experiment_matrix.csv").is_file())
        self.assertIn("created_at_utc", summary["dev_selection"])
        from datetime import datetime
        freeze = datetime.fromisoformat(summary["dev_selection"]["created_at_utc"])
        for row in summary["all_arms"]:
            self.assertTrue(all(row[k] for k in ("mechanism", "inputs", "loss", "policy", "declared_weight_source")))
            self.assertEqual(row["dev_boundary_policy_name"], "kp_frontier")
            self.assertIsNotNone(row["dev_source_macro_known"])
            marker = protocol.read_json(self.suite / "arms" / row["arm_id"] / "test/stage_binding.json")
            self.assertGreaterEqual(datetime.fromisoformat(marker["completed_at_utc"]), freeze)


if __name__ == "__main__":
    unittest.main()
