"""DEV recommendations require paired protection and crossfit evidence.

These fixtures write genuine hash-bound stage artifacts and use production
routing, metrics, receipt validation, comparison CSVs and recommendation code.
The fixture differences concern predictions, not mocked gate/selection results.
"""
import copy
import csv
from pathlib import Path
import tempfile
import unittest

from taxosafe_routealign import evaluation, protocol, reporting
from taxosafe_support import calibration as base
from tests.test_taxosafe_geometry_calibration import BASELINE, META, row


MISSING = object()
ARM_IDS = ("A00_reference", "A01_evidence_anchor", "A02_proximity",
           "A03_combined", "A04_parent_rerank")


class RouteAlignReportingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.initialize(Path(self.temporary.name))

    def initialize(self, root):
        self.root = root
        self.ids = list(ARM_IDS)
        self.cfg = {"arms": copy.deepcopy(protocol.ARMS)}
        self.assertEqual([arm["id"] for arm in self.cfg["arms"]], self.ids)
        self.snapshot = {
            "schema_version": protocol.SCHEMA_VERSION,
            "arm_ids": self.ids,
            "config_sha256": protocol.object_hash(self.cfg),
            "signature": {"fixture_signature": "fixed"},
            "source_binding": {"reference": "immutable-fixture"},
        }
        protocol.write_json(self.root / "config.json", self.cfg)
        protocol.write_json(self.root / "snapshot.json", self.snapshot)

    def stage(self, arm_id, stage, *, known_wrong=(), near_correct=0,
              extra_correct=0, crossfit=MISSING, crossfit_location="both",
              source_aliases=False, duplicate_unknown=False):
        directory = self.root / "arms" / arm_id / stage
        directory.mkdir(parents=True, exist_ok=True)
        prefix = "dev" if stage == "calibration" else "test"
        raw = [row(prefix + "-known-" + str(i), "known",
                   lm=-1. if i in known_wrong else 1.) for i in range(12)]
        for status, correct in (("intra", near_correct), ("extra", extra_correct)):
            for i in range(10):
                source = status + "-large" if i < 9 else status + "-small"
                if source_aliases and i < 4:
                    source = status.upper() + "_LARGE"
                item = row(prefix + "-" + status + "-" + str(i), status,
                           pm=-1. if status == "extra" and i < correct else 1.,
                           lm=-1. if status == "intra" and i < correct else 1.,
                           source=source)
                raw.append(item)
        for item in raw:
            item["path"] = prefix + "/" + item["image_sha256"] + ".png"
            if stage == "test":
                item["split"] = "test_" + item["status"]
        if duplicate_unknown:
            alias = copy.deepcopy(raw[12])
            alias["path"] = prefix + "/same-content-alias.png"
            raw.append(alias)
        predictions = base.apply_router(raw, BASELINE, META)
        report = dict(base.evaluate_records(predictions, META),
                      calibration_status="frozen_reference_reproduced" if arm_id == self.ids[0]
                      else "arm_development_fit")
        summary = copy.deepcopy(report)
        if crossfit is not MISSING:
            audit = {"passed": crossfit, "validation_scope": "fixture_crossfit"}
            if crossfit_location in {"both", "report"}:
                report["crossfit_audit"] = copy.deepcopy(audit)
            if crossfit_location in {"both", "summary"}:
                summary["crossfit_audit"] = copy.deepcopy(audit)
        binding = {
            "arm_id": arm_id,
            "suite_signature": self.snapshot["signature"],
            "suite_snapshot_sha256": protocol.file_hash(self.root / "snapshot.json"),
            "config_sha256": self.snapshot["config_sha256"],
            "source_binding": self.snapshot["source_binding"],
        }
        for name, value in (("report", report), ("binding", binding), ("router", BASELINE)):
            protocol.write_json(directory / (name + ".json"), value)
        protocol.write_records(directory / "scores.jsonl", raw)
        protocol.write_records(directory / "predictions.jsonl", predictions)
        receipt = {
            "schema_version": evaluation.SCHEMA_VERSION,
            "stage": stage,
            "binding": binding,
            "binding_sha256": protocol.object_hash(binding),
            "test_used_for_fitting": False,
            "meta": META,
            "summary": summary,
            "artifacts": {
                name: {"path": name + suffix,
                       "sha256": protocol.file_hash(directory / (name + suffix))}
                for name, suffix in (("report", ".json"), ("binding", ".json"),
                                     ("router", ".json"), ("scores", ".jsonl"),
                                     ("predictions", ".jsonl"))
            },
        }
        if stage == "test":
            calibration = self.root / "arms" / arm_id / "calibration"
            receipt["calibration_receipt_sha256"] = protocol.file_hash(calibration / "completed.json")
            receipt["calibration_router_sha256"] = protocol.file_hash(calibration / "router.json")
        protocol.write_json(directory / "completed.json", receipt)
        return report, predictions

    def development(self, **options_by_arm):
        for arm_id in self.ids:
            self.stage(arm_id, "calibration", **options_by_arm.get(arm_id, {}))

    def run_test_stages(self, **options_by_arm):
        for arm_id in self.ids:
            self.stage(arm_id, "test", **options_by_arm.get(arm_id, {}))

    def failure(self, arm_id, stage):
        protocol.write_json(self.root / "arms" / arm_id / "failure.json", {
            "arm_id": arm_id, "stage": stage,
            "error": "technical fixture failure: " + stage,
            "test_blocked": stage != "test",
        })

    @staticmethod
    def entry(selection, arm_id):
        return next(item for item in selection["exploratory_development_ranking"]
                    if item["arm_id"] == arm_id)

    def comparison_rows(self):
        with open(self.root / "comparison_all.csv", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    def test_equal_known_totals_cannot_hide_one_lost_correct_image(self):
        self.development(**{
            self.ids[0]: dict(known_wrong=(11,), near_correct=10, extra_correct=10),
            self.ids[1]: dict(known_wrong=(0,), near_correct=10, extra_correct=10, crossfit=True),
        })
        selection = reporting.freeze_dev_selection(self.root)
        baseline = self.entry(selection, self.ids[0])
        candidate = self.entry(selection, self.ids[1])
        self.assertTrue(candidate["targets_passed"])
        self.assertEqual(baseline["counts"]["known_correct"], candidate["counts"]["known_correct"])
        self.assertTrue(candidate["known_count_preserved"])
        self.assertEqual(candidate["known_lost_correct_count"], 1)
        self.assertEqual(candidate["known_gained_correct_count"], 1)
        self.assertFalse(candidate["known_preserved"])
        self.assertFalse(candidate["qualified"])
        self.assertIsNone(selection["qualified_candidate_arm_id"])
        self.assertEqual(selection["recommendation_arm_id"], self.ids[0])

    def test_missing_false_and_truthy_nonboolean_crossfit_cannot_qualify(self):
        original_root = self.root
        for name, crossfit in (("missing", MISSING), ("false", False),
                               ("integer", 1), ("string", "true")):
            with self.subTest(crossfit=name):
                self.initialize(original_root / name)
                self.development(**{self.ids[1]: dict(near_correct=10, extra_correct=10,
                                                     crossfit=crossfit)})
                selection = reporting.freeze_dev_selection(self.root)
                candidate = self.entry(selection, self.ids[1])
                self.assertTrue(candidate["targets_passed"])
                self.assertTrue(candidate["known_preserved"])
                self.assertTrue(candidate["crossfit_required"])
                self.assertFalse(candidate["crossfit_audit_passed"])
                self.assertFalse(candidate["qualified"])
                self.assertIsNone(selection["qualified_candidate_arm_id"])
                self.assertEqual(selection["recommendation_arm_id"], self.ids[0])

    def test_all_gates_crossfit_and_zero_paired_loss_qualify_candidate(self):
        original_root = self.root
        for location in ("both", "summary"):
            with self.subTest(location=location):
                self.initialize(original_root / location)
                self.development(**{self.ids[1]: dict(near_correct=10, extra_correct=10,
                                                     crossfit=True, crossfit_location=location)})
                selection = reporting.freeze_dev_selection(self.root)
                candidate = self.entry(selection, self.ids[1])
                self.assertTrue(candidate["qualified"])
                self.assertTrue(candidate["crossfit_audit_passed"])
                self.assertTrue(candidate["known_preserved"])
                self.assertEqual(candidate["known_lost_correct_count"], 0)
                self.assertEqual(selection["qualified_candidate_arm_id"], self.ids[1])
                self.assertEqual(selection["recommendation_arm_id"], self.ids[1])
                self.assertFalse(selection["selection_uses_test"])
                self.assertFalse(selection["test_predictions_read"])

    def test_baseline_is_retained_without_claiming_it_is_a_new_qualified_candidate(self):
        self.development(**{self.ids[0]: dict(near_correct=10, extra_correct=10)})
        selection = reporting.freeze_dev_selection(self.root)
        baseline = self.entry(selection, self.ids[0])
        self.assertTrue(baseline["targets_passed"])
        self.assertFalse(baseline["crossfit_required"])
        self.assertFalse(baseline["qualified"])
        self.assertIsNone(selection["qualified_candidate_arm_id"])
        self.assertEqual(selection["recommendation_arm_id"], self.ids[0])

    def test_conflicting_crossfit_receipt_and_report_are_rejected(self):
        self.development(**{self.ids[1]: dict(near_correct=10, extra_correct=10, crossfit=True)})
        receipt_path = self.root / "arms" / self.ids[1] / "calibration/completed.json"
        receipt = protocol.read_json(receipt_path)
        receipt["summary"]["crossfit_audit"]["passed"] = False
        protocol.write_json(receipt_path, receipt)
        with self.assertRaises(ValueError):
            reporting.freeze_dev_selection(self.root)
        self.assertFalse((self.root / "dev_selection.json").exists())

    def test_test_winner_never_changes_frozen_dev_recommendation(self):
        self.development(**{self.ids[1]: dict(near_correct=10, extra_correct=10, crossfit=True)})
        selection = reporting.freeze_dev_selection(self.root)
        frozen = (self.root / "dev_selection.json").read_bytes()
        self.run_test_stages(**{self.ids[2]: dict(near_correct=10, extra_correct=10)})
        summary = reporting.summarize_suite(self.root, phase="complete")
        self.assertEqual(summary["recommendation_arm_id"], self.ids[1])
        self.assertEqual(summary["dev_selection"], selection)
        self.assertEqual(summary["best_exploratory_test_arm"]["arm_id"], self.ids[2])
        self.assertTrue(summary["exploratory_test_ranking"]["selection_uses_test"])
        self.assertFalse(summary["production_recommendation_uses_test"])
        self.assertEqual((self.root / "dev_selection.json").read_bytes(), frozen)
        self.assertTrue(all(item["qualified"] is False
                            for item in summary["exploratory_test_ranking"]["arms"]))

    def test_baseline_and_other_technical_failures_keep_all_comparison_rows(self):
        self.failure(self.ids[0], "calibration")
        self.failure(self.ids[4], "dependency")
        for arm_id in self.ids[1:4]:
            self.stage(arm_id, "calibration", near_correct=10, extra_correct=10, crossfit=True)
        selection = reporting.freeze_dev_selection(self.root)
        self.assertIsNone(selection["recommendation_arm_id"])
        self.assertIsNone(selection["qualified_candidate_arm_id"])
        for arm_id in self.ids[1:4]:
            self.stage(arm_id, "test", near_correct=10, extra_correct=10)
        summary = reporting.summarize_suite(self.root)
        self.assertEqual(summary["completed_test_count"], 3)
        self.assertIsNone(summary["recommendation_arm_id"])
        self.assertEqual(summary["paired_correctness"]["test"], {})
        rows = self.comparison_rows()
        self.assertEqual([item["arm_id"] for item in rows], self.ids)
        for index in (0, 4):
            self.assertEqual(rows[index]["calibration_execution"], "technical_failure")
            self.assertEqual(rows[index]["dev_known_end_to_end_leaf_accuracy"], "")
            self.assertIn("technical fixture failure", rows[index]["failure_reason"])
        for index in (1, 2, 3):
            self.assertEqual(rows[index]["test_execution"], "completed")

    def test_test_failure_keeps_dev_choice_and_comparison_row(self):
        self.development(**{self.ids[1]: dict(near_correct=10, extra_correct=10, crossfit=True)})
        selection = reporting.freeze_dev_selection(self.root)
        self.failure(self.ids[1], "test")
        for arm_id in self.ids:
            if arm_id != self.ids[1]:
                self.stage(arm_id, "test")
        summary = reporting.summarize_suite(self.root)
        self.assertEqual(summary["dev_selection"], selection)
        self.assertEqual(summary["completed_test_count"], 4)
        rows = self.comparison_rows()
        self.assertEqual(len(rows), 5)
        self.assertEqual(rows[1]["test_execution"], "technical_failure")
        self.assertEqual(rows[1]["test_known_end_to_end_leaf_accuracy"], "")

    def test_csv_distinguishes_owned_weights_and_zero_step_reused_weights(self):
        self.development()
        protocol.write_json(self.root / "arms" / self.ids[1] / "training/completed.json",
                            {"optimizer_steps": 37, "best_epoch": 2})
        reporting.freeze_dev_selection(self.root)
        self.run_test_stages()
        reporting.summarize_suite(self.root)
        rows = self.comparison_rows()
        self.assertEqual([item["weight_source"] for item in rows],
                         ["source", "own", "source", self.ids[1], self.ids[1]])
        self.assertEqual([item["optimizer_steps"] for item in rows], ["0", "37", "0", "0", "0"])
        self.assertEqual(rows[1]["best_epoch"], "2")

    def test_source_macro_uses_unique_sources_not_micro_or_duplicate_rows(self):
        options = dict(near_correct=9, extra_correct=9, crossfit=True,
                       source_aliases=True, duplicate_unknown=True)
        arms = {arm_id: dict(source_aliases=True) for arm_id in self.ids}
        arms[self.ids[2]] = options
        self.development(**arms)
        selection = reporting.freeze_dev_selection(self.root)
        candidate = self.entry(selection, self.ids[2])
        self.assertEqual(candidate["counts"]["intra"], 10)
        self.assertAlmostEqual(candidate["metrics"]["intra_correct_fallback_rate"], .9)
        self.assertAlmostEqual(candidate["metrics"]["extra_global_unknown_recall"], .9)
        self.assertAlmostEqual(candidate["source_macro"]["intra"], .5)
        self.assertAlmostEqual(candidate["source_macro"]["extra"], .5)
        self.run_test_stages(**arms)
        reporting.summarize_suite(self.root)
        csv_row = self.comparison_rows()[2]
        for prefix in ("dev", "test"):
            for status in ("intra", "extra"):
                self.assertAlmostEqual(float(csv_row[prefix + "_" + status + "_source_macro_accuracy"]), .5)
        for phase in ("development", "test"):
            detail = protocol.read_json(self.root / ("source_macro_" + phase + ".json"))
            self.assertAlmostEqual(detail[self.ids[2]]["source_macro"]["intra"], .5)
            self.assertAlmostEqual(detail[self.ids[2]]["source_macro"]["extra"], .5)
            for status in ("intra", "extra"):
                self.assertEqual(detail[self.ids[2]]["observed_source_counts"][status], 2)
                sources = [item for item in detail[self.ids[2]]["per_source"]
                           if item["status"] == status]
                self.assertEqual(sorted(item["sample_count"] for item in sources), [1, 9])
            empty_known = [item for item in detail[self.ids[2]]["per_source"]
                           if item["status"] == "known" and item["sample_count"] == 0]
            self.assertEqual(len(empty_known), 2)
            self.assertTrue(all(item["correct_rate"] is None and item["evidence_status"] == "not_evaluable"
                                for item in empty_known))

    def test_changed_frozen_selection_and_calibration_artifacts_are_rejected(self):
        self.development()
        selection = reporting.freeze_dev_selection(self.root)
        changed = copy.deepcopy(selection)
        changed["recommendation_arm_id"] = self.ids[4]
        protocol.write_json(self.root / "dev_selection.json", changed)
        with self.assertRaisesRegex(ValueError, "Immutable"):
            reporting.freeze_dev_selection(self.root)
        protocol.write_json(self.root / "dev_selection.json", selection)
        path = self.root / "arms" / self.ids[1] / "calibration/predictions.jsonl"
        path.write_text("", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            reporting.freeze_dev_selection(self.root)

    def test_initial_dev_freeze_refuses_any_existing_test_stage(self):
        self.development()
        (self.root / "arms" / self.ids[4] / "test").mkdir()
        with self.assertRaisesRegex(ValueError, "before any TEST"):
            reporting.freeze_dev_selection(self.root)


if __name__ == "__main__":
    unittest.main()
