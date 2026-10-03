"""Failure audits count terminal errors and paired harm without fitting."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from taxosafe_support import calibration as base
from tests.test_taxosafe_geometry_calibration import BASELINE, META, row
from tools.diagnose_taxosafe_failure_modes import audit_run, diagnose


def predictions():
    rows = [row("correct-known", "known"), row("near-leaf", "intra"),
            row("near-parent", "intra", lm=-1.), row("far-root", "extra", pm=-1.)]
    return base.apply_router(rows, BASELINE, META)


def root(record, rule="root_reject"):
    record.update(prediction_type="global_unknown", parent=None, leaf=None,
                  output_node=0, applied_rule_indices=[rule])


class FailureDiagnosticTests(unittest.TestCase):
    def test_near_leaf_to_root_is_path_harm_even_if_both_terminal_wrong(self):
        before = predictions()
        after = copy.deepcopy(before)
        root(after[1])
        report = diagnose(before, after, META)
        risk = report["paired_risks"]["near_parent_path_harm"]
        self.assertEqual((risk["numerator"], risk["denominator"]), (1, 2))
        self.assertEqual(report["selected"]["near"]["terminal_decomposition"], {
            "correct_parent_fallback": 1, "root_rejection": 1,
            "leaf_false_acceptance": 0, "wrong_parent_fallback": 0})
        self.assertEqual(report["rule_attribution"]["root_reject"]["near_parent_path_harm"], 1)
        self.assertEqual(report["rule_attribution"]["root_reject"]["terminal_losses"], 0)

    def test_known_harm_counts_wrong_leaf_replacement_not_only_rejection(self):
        before = predictions()
        after = copy.deepcopy(before)
        after[0].update(leaf=1, output_node=4)
        report = diagnose(before, after, META)
        self.assertEqual(report["paired_risks"]["known_new_harm"]["numerator"], 1)
        self.assertFalse(report["paired_risks"]["original_leaf_invariance"]["passed"])

    def test_wrong_parent_is_not_a_retained_correct_parent_path(self):
        before = predictions()
        after = copy.deepcopy(before)
        after[2].update(parent=1, output_node=2, route_parent=1)
        report = diagnose(before, after, META)
        self.assertEqual(report["paired_risks"]["near_parent_path_retention"]["numerator"], 1)
        self.assertEqual(report["paired_risks"]["near_parent_path_harm"]["numerator"], 0)

    def test_aliases_deduplicated_and_empty_leaves_not_certified(self):
        before = predictions()
        report = diagnose(before + [copy.deepcopy(before[0])], before, META)
        self.assertEqual(report["input_counts"]["unique_images"], 4)
        coverage = report["known_leaf_coverage"]
        self.assertEqual(coverage[0]["evidence_status"], "insufficient_evidence")
        self.assertEqual(coverage[1]["evidence_status"], "not_evaluable")
        self.assertEqual(coverage[1]["evaluation_count"], 0)

    def test_conflicting_new_evidence_aliases_and_pair_annotations_fail(self):
        before = predictions()
        alias = copy.deepcopy(before[0])
        alias["encoder_evidence"] = {"parent_text_logits": [1., 2.]}
        with self.assertRaisesRegex(ValueError, "Conflicting diagnostic evidence"):
            diagnose(before + [alias], before, META)
        after = copy.deepcopy(before)
        after[1]["true_parent"] = 1
        with self.assertRaisesRegex(ValueError, "inconsistent truth"):
            diagnose(before, after, META)

    def test_top2_and_upper_bounds_have_explicit_denominators(self):
        before = predictions()
        report = diagnose(before, before, META)
        bounds = report["reference"]["near"]["conditional_upper_bounds"]
        self.assertEqual(bounds["fixed_support_candidate_ideal_two_gates"]["numerator"], 2)
        self.assertEqual(bounds["fixed_actual_parent_gate_and_route_ideal_leaf_gate"]["numerator"], 2)
        self.assertEqual(report["reference"]["near"]["support_parent_top2_recall"]["rate"], 1.)

    def test_source_macro_does_not_weight_large_sources(self):
        before = predictions()
        before[1]["source"] = "near-large"
        before[2]["source"] = "near-small"
        extra = base.apply_router([row("near-leaf-two", "intra", source="near-large")], BASELINE, META)
        report = diagnose(before + extra, before + extra, META)
        self.assertEqual(report["reference"]["source_macro"]["intra"]["correct_rate"], .5)
        self.assertAlmostEqual(report["reference"]["metrics"]["metrics"]["intra_correct_fallback_rate"], 1 / 3)

    def test_source_spelling_variants_do_not_create_extra_sources(self):
        before = predictions()
        before[1]["source"] = "Near Species"
        before[2]["source"] = "near_species"
        report = diagnose(before, before, META)
        self.assertEqual(report["reference"]["source_macro"]["intra"]["source_count"], 1)

    def test_default_refuses_test_even_when_input_path_is_named_development(self):
        before = predictions()
        for record in before:
            record["split"] = "test_" + record["status"]
        with self.assertRaisesRegex(ValueError, "Expected only DEV"):
            diagnose(before, before, META)
        self.assertTrue(diagnose(before, before, META, evaluate_test=True)["test_opened"])

    def test_train_coverage_deduplicates_and_checks_overlap(self):
        before = predictions()
        training = row("train-other", "known", leaf=1)
        training["split"] = "train"
        report = diagnose(before, before, META, train=[training, copy.deepcopy(training)])
        self.assertEqual(report["known_leaf_coverage"][1]["train_count"], 1)
        training["image_sha256"] = before[0]["image_sha256"]
        with self.assertRaisesRegex(ValueError, "TRAIN/evaluation"):
            diagnose(before, before, META, train=[training])

    def test_run_default_never_opens_test_or_mutates_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            calibration = directory / "calibration"
            calibration.mkdir()
            (directory / "test").mkdir()
            (directory / "test/predictions.jsonl").write_text("must never open", encoding="utf-8")
            router = {"meta": META, "decoder": "parentrisk", "baseline_router": BASELINE}
            (calibration / "router.json").write_text(json.dumps(router), encoding="utf-8")
            (calibration / "development_predictions.jsonl").write_text(
                "\n".join(json.dumps(r) for r in predictions()), encoding="utf-8")
            files_before = {str(p): p.read_bytes() for p in directory.rglob("*") if p.is_file()}
            with patch("taxosafe_support.membership_calibration.calibrate", side_effect=AssertionError("must not fit")):
                report = audit_run(directory)
            files_after = {str(p): p.read_bytes() for p in directory.rglob("*") if p.is_file()}
            self.assertEqual(files_before, files_after)
            self.assertFalse(report["test_opened"])
            self.assertFalse(report["fitting_performed"])
            self.assertFalse(report["independent_model_level_validation"])

    def test_reference_raw_development_scores_are_supported(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            calibration = directory / "calibration"
            calibration.mkdir()
            (calibration / "router.json").write_text(json.dumps(BASELINE), encoding="utf-8")
            rows = [row("known", "known"), row("near", "intra"), row("far", "extra")]
            (calibration / "development_scores.jsonl").write_text(
                "\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            report = audit_run(directory)
            self.assertTrue(report["paired_risks"]["original_leaf_invariance"]["passed"])
            self.assertEqual(report["paired_risks"]["known_new_harm"]["numerator"], 0)


if __name__ == "__main__":
    unittest.main()
