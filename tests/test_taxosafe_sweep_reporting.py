"""DEV immutability and descriptive TEST comparisons of failed-gate arms."""
import copy
import csv
import tempfile
import unittest
from pathlib import Path

from taxosafe_sweep import evaluation, reporting, protocol
from taxosafe_support import calibration as base
from tests.test_taxosafe_geometry_calibration import META, BASELINE, row


class SweepReportingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ids = ["E00_reference", "E01_heads", "E02_heads_anchor"]
        self.cfg = {"arms": [{"id": value} for value in self.ids]}
        self.snapshot = {"schema_version": protocol.SCHEMA_VERSION, "arm_ids": self.ids,
                         "config_sha256": protocol.object_hash(self.cfg), "signature": {"x": "fixed"},
                         "source_binding": {"reference": "immutable"}}
        protocol.write_json(self.root / "config.json", self.cfg)
        protocol.write_json(self.root / "snapshot.json", self.snapshot)

    def stage(self, arm_id, stage, *, known_wrong=False, near_good=False, extra_good=False):
        directory = self.root / "arms" / arm_id / stage
        directory.mkdir(parents=True)
        prefix = "dev" if stage == "calibration" else "test"
        raw = [row(prefix + "k", "known", lm=-1. if known_wrong else 1.),
               row(prefix + "n", "intra", lm=-1. if near_good else 1., source="near_species"),
               row(prefix + "e", "extra", pm=-1. if extra_good else 1., source="extra_species")]
        if stage == "test":
            for item in raw:
                item["split"] = "test_" + item["status"]
        pred = base.apply_router(raw, BASELINE, META)
        report = dict(base.evaluate_records(pred, META), calibration_status="best_effort")
        binding = {"arm_id": arm_id, "suite_signature": self.snapshot["signature"],
                   "suite_snapshot_sha256": protocol.file_hash(self.root / "snapshot.json"),
                   "config_sha256": self.snapshot["config_sha256"],
                   "source_binding": self.snapshot["source_binding"]}
        protocol.write_json(directory / "report.json", report)
        protocol.write_json(directory / "binding.json", binding)
        protocol.write_json(directory / "router.json", BASELINE)
        protocol.write_records(directory / "scores.jsonl", raw)
        protocol.write_records(directory / "predictions.jsonl", pred)
        receipt = {"schema_version": evaluation.SCHEMA_VERSION, "stage": stage,
                   "binding": binding, "binding_sha256": protocol.object_hash(binding),
                   "test_used_for_fitting": False, "meta": META, "summary": report,
                   "artifacts": {name: {"path": name + suffix, "sha256": protocol.file_hash(directory / (name + suffix))}
                                 for name, suffix in (("report", ".json"), ("binding", ".json"),
                                                      ("router", ".json"), ("scores", ".jsonl"), ("predictions", ".jsonl"))}}
        if stage == "test":
            cal = self.root / "arms" / arm_id / "calibration"
            receipt["calibration_receipt_sha256"] = protocol.file_hash(cal / "completed.json")
            receipt["calibration_router_sha256"] = protocol.file_hash(cal / "router.json")
        protocol.write_json(directory / "completed.json", receipt)

    def failure(self, arm_id, stage):
        path = self.root / "arms" / arm_id / "failure.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        protocol.write_json(path, {"arm_id": arm_id, "stage": stage, "error": "technical fixture failure",
                                   "test_blocked": stage != "test"})

    def dev(self):
        for arm_id in self.ids:
            self.stage(arm_id, "calibration")

    def test_failed_gates_retains_reference_and_exposes_best_effort(self):
        self.dev()
        selection = reporting.freeze_dev_selection(self.root)
        self.assertIsNone(selection["qualified_candidate_arm_id"])
        self.assertEqual(selection["recommendation_arm_id"], "E00_reference")
        self.assertEqual(selection["best_exploratory_dev_arm"]["arm_id"], "E00_reference")
        self.assertFalse(selection["selection_uses_test"])
        self.assertFalse(selection["test_predictions_read"])

    def test_test_ranking_cannot_change_frozen_development_choice(self):
        self.dev()
        selection = reporting.freeze_dev_selection(self.root)
        frozen = (self.root / "dev_selection.json").read_bytes()
        self.stage(self.ids[0], "test")
        self.stage(self.ids[1], "test", near_good=True, extra_good=True)
        self.stage(self.ids[2], "test", known_wrong=True)
        summary = reporting.summarize_suite(self.root)
        self.assertEqual(summary["recommendation_arm_id"], selection["recommendation_arm_id"])
        self.assertEqual(summary["best_exploratory_test_arm"]["arm_id"], "E01_heads")
        self.assertTrue(summary["exploratory_test_ranking"]["selection_uses_test"])
        self.assertFalse(summary["production_recommendation_uses_test"])
        self.assertEqual((self.root / "dev_selection.json").read_bytes(), frozen)
        paired = summary["paired_correctness"]["test"][self.ids[2]]
        self.assertEqual(paired["known_lost_correct_count"], 1)
        self.assertEqual(paired["known_gained_correct_count"], 0)
        self.assertEqual(next(r for r in paired["per_species"] if r["species"] == "b")["evidence_status"], "not_evaluable")

    def test_every_arm_must_be_attempted_before_freeze(self):
        self.stage(self.ids[0], "calibration")
        with self.assertRaises((ValueError, FileNotFoundError)):
            reporting.freeze_dev_selection(self.root)
        self.failure(self.ids[1], "training")
        self.failure(self.ids[2], "dependency")
        result = reporting.freeze_dev_selection(self.root)
        self.assertEqual(len(result["technical_failures"]), 2)

    def test_technical_reference_calibration_failure_no_forced_winner(self):
        self.failure(self.ids[0], "calibration")
        for arm_id in self.ids[1:]:
            self.stage(arm_id, "calibration", near_good=True, extra_good=True)
        selection = reporting.freeze_dev_selection(self.root)
        self.assertIsNone(selection["recommendation_arm_id"])
        self.assertIsNotNone(selection["best_exploratory_dev_arm"])
        for arm_id in self.ids[1:]:
            self.stage(arm_id, "test", near_good=True, extra_good=True)
        summary = reporting.summarize_suite(self.root)
        self.assertEqual(summary["completed_test_count"], 2)
        self.assertEqual(summary["paired_correctness"]["test"], {})
        with open(self.root / "comparison_all.csv", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([value["arm_id"] for value in rows], self.ids)
        self.assertEqual(rows[0]["calibration_execution"], "technical_failure")
        self.assertEqual(rows[0]["dev_known_end_to_end_leaf_accuracy"], "")
        self.assertEqual(rows[1]["test_execution"], "completed")

    def test_technical_reference_test_failure_does_not_change_dev(self):
        self.dev()
        selection = reporting.freeze_dev_selection(self.root)
        self.failure(self.ids[0], "test")
        for arm_id in self.ids[1:]:
            self.stage(arm_id, "test")
        summary = reporting.summarize_suite(self.root)
        self.assertEqual(summary["dev_selection"], selection)
        self.assertEqual(summary["completed_test_count"], 2)
        self.assertEqual(summary["paired_correctness"]["test"], {})

    def test_selection_and_saved_calibration_tampering_are_rejected(self):
        self.dev()
        selection = reporting.freeze_dev_selection(self.root)
        changed = copy.deepcopy(selection)
        changed["recommendation_arm_id"] = "E02_heads_anchor"
        protocol.write_json(self.root / "dev_selection.json", changed)
        with self.assertRaisesRegex(ValueError, "Immutable"):
            reporting.freeze_dev_selection(self.root)
        protocol.write_json(self.root / "dev_selection.json", selection)
        cal = self.root / "arms" / self.ids[1] / "calibration"
        (cal / "predictions.jsonl").write_text("", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            reporting.freeze_dev_selection(self.root)

    def test_initial_freeze_refuses_prior_test_access(self):
        self.dev()
        (self.root / "arms" / self.ids[0] / "test").mkdir()
        with self.assertRaisesRegex(ValueError, "before any TEST"):
            reporting.freeze_dev_selection(self.root)


if __name__ == "__main__":
    unittest.main()
