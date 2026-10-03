"""Known holdout, content isolation, paired harms and honest small-n reports."""
import copy
import unittest

from taxosafe_parentrisk.folds import build_folds, unique_records
from taxosafe_parentrisk.reporting import paired_report, score_diagnostics, rule_support
from taxosafe_support import calibration as base
from tests.test_taxosafe_geometry_calibration import META, BASELINE, row


def sample(identity, status, **kwargs):
    result = row(identity, status, **kwargs)
    result["parent_evidence"] = {"text_z": [1., 0.], "membership_z": [result["baseline_parent_z"], 0.],
                                 "geometry_z": [result["geometry_parent_score"], 0.], "geometry_raw": [1., 0.]}
    result["encoder_evidence"] = {"parent_text_logits": [2., 1.]}
    return result


def dataset():
    records = [sample("known-" + str(i), "known") for i in range(8)]
    records += [sample("near-" + str(i), "intra", source="near" + str(i // 2)) for i in range(8)]
    records += [sample("extra-" + str(i), "extra", source="extra" + str(i // 2)) for i in range(8)]
    return records


class ParentRiskFoldTests(unittest.TestCase):
    def test_every_hash_held_once_known_and_unknown_sources_are_held(self):
        rows = dataset()
        plan = build_folds(rows, META, n_splits=4, seed=7)
        held = [h for fold in plan["folds"] for h in fold["held_image_sha256"]]
        self.assertEqual(len(set(held)), len(rows))
        self.assertEqual(set(held), {r["image_sha256"] for r in rows})
        for fold in plan["folds"]:
            self.assertTrue(fold["usable"])
            self.assertEqual(fold["known_held_n"], 2)
            self.assertEqual(fold["known_fit_n"], 6)
            self.assertEqual(len(fold["held_sources"]["intra"]), 1)
            self.assertEqual(len(fold["held_sources"]["extra"]), 1)
            self.assertFalse(set(fold["fit_image_sha256"]) & set(fold["held_image_sha256"]))
        self.assertFalse(plan["independent_model_level_validation"])

    def test_order_aliases_and_scores_do_not_change_plan_identity(self):
        rows = dataset()
        plan = build_folds(rows, META)
        changed = copy.deepcopy(list(reversed(rows)))
        for r in changed:
            r["support_evidence"]["parent_logits"] = [1000., -1000.]
            r["parent_evidence"]["text_z"] = [-1000., 1000.]
            r["geometry_leaf_score"] = -999.
        changed.append(copy.deepcopy(changed[0]))
        actual = build_folds(changed, META)
        self.assertEqual(plan["folds"], actual["folds"])
        self.assertEqual(plan["sha256"], actual["sha256"])
        self.assertEqual(actual["duplicate_record_count"], 1)

    def test_duplicate_new_evidence_cannot_be_hidden(self):
        r = dataset()[0]
        changed = copy.deepcopy(r)
        changed["parent_evidence"]["text_z"][0] += 1.
        with self.assertRaisesRegex(ValueError, "inconsistent parent/encoder evidence"):
            unique_records([r, changed])
        del changed["parent_evidence"]
        with self.assertRaisesRegex(ValueError, "inconsistent parent/encoder evidence"):
            unique_records([r, changed])

    def test_tiny_folds_report_missing_evidence_without_fake_counts(self):
        rows = [sample("k", "known"), sample("n", "intra"), sample("e", "extra")]
        plan = build_folds(rows, META)
        self.assertTrue(any(f["known_held_n"] == 0 for f in plan["folds"]))
        for fold in plan["folds"]:
            if not fold["known_held_n"]:
                self.assertEqual(fold["known_held_evidence"], "not_evaluable")
        self.assertTrue(any(not f["usable"] and f["reason"].startswith("missing_fit_statuses") for f in plan["folds"]))

    def test_test_data_is_never_folded_for_calibration(self):
        rows = dataset()
        rows[0]["split"] = "test_known"
        with self.assertRaisesRegex(ValueError, "test fitting is prohibited"):
            build_folds(rows, META)

    def test_source_spelling_aliases_are_kept_in_the_same_held_fold(self):
        rows = dataset()
        rows[8]["source"], rows[9]["source"] = "Acartia hongi", "ACARTIA_hongi"
        plan = build_folds(rows, META)
        matches = [f for f in plan["folds"] if "acartiahongi" in f["held_sources"]["intra"]]
        self.assertEqual(len(matches), 1)
        self.assertTrue({rows[8]["image_sha256"], rows[9]["image_sha256"]}.issubset(matches[0]["held_image_sha256"]))
        self.assertIn("Acartia hongi", matches[0]["held_source_names"]["intra"])
        self.assertIn("ACARTIA_hongi", matches[0]["held_source_names"]["intra"])


class ParentRiskReportingTests(unittest.TestCase):
    def _routed(self):
        rows = [sample("k", "known"), sample("n", "intra"),
                sample("n2", "intra", lm=-1., source="near-second"), sample("e", "extra", pm=-1.)]
        return base.apply_router(rows, BASELINE, META)

    def test_wrong_leaf_to_root_near_is_measured_as_parent_path_harm(self):
        before = self._routed()
        after = copy.deepcopy(before)
        after[1].update(prediction_type="global_unknown", parent=None, leaf=None, output_node=0)
        report = paired_report(before, after, META)
        self.assertEqual(report["baseline"]["counts"]["intra_correct"], report["selected"]["counts"]["intra_correct"])
        self.assertEqual(report["near_parent_path_loss"]["numerator"], 1)
        self.assertEqual(report["near_parent_path_loss"]["denominator"], 2)
        self.assertFalse(report["preservation_audit"]["near_parent_paths_preserved"])
        self.assertFalse(report["preservation_audit"]["passed"])

    def test_known_damage_is_paired_and_empty_leaves_unevaluable(self):
        before = self._routed()
        after = copy.deepcopy(before)
        after[0].update(prediction_type="intra_unknown", leaf=None, output_node=1)
        report = paired_report(before, after, META)
        self.assertEqual(report["known_added_harm"]["numerator"], 1)
        self.assertEqual(report["known_added_harm"]["denominator"], 1)
        self.assertEqual(report["per_known_leaf"]["a"]["evidence_status"], "insufficient_evidence")
        self.assertEqual(report["per_known_leaf"]["b"]["evidence_status"], "not_evaluable")
        self.assertFalse(report["preservation_audit"]["known_correct_images_preserved"])

    def test_near_correct_fallback_is_protected_despite_same_source_correct_count(self):
        records = [sample("correct-near", "intra", lm=-1., source="same-source"),
                   sample("wrong-near", "intra", source="same-source")]
        before = base.apply_router(records, BASELINE, META)
        after = copy.deepcopy(before)
        # Swap correctness between two images without changing the source total.
        after[0].update(prediction_type="intra_unknown", parent=1, leaf=None, output_node=2)
        after[1].update(prediction_type="intra_unknown", parent=0, leaf=None, output_node=1)
        report = paired_report(before, after, META)
        self.assertEqual(report["near_added_harm"]["numerator"], 1)
        self.assertEqual(report["near_added_harm"]["denominator"], 2)
        self.assertEqual(report["near_added_harm"]["image_sha256"], [before[0]["image_sha256"]])
        self.assertTrue(report["preservation_audit"]["unknown_source_correct_counts_preserved"])
        self.assertTrue(report["preservation_audit"]["near_parent_paths_preserved"])
        self.assertFalse(report["preservation_audit"]["near_correct_images_preserved"])
        self.assertFalse(report["preservation_audit"]["passed"])

    def test_exact_content_sets_and_annotations_are_required(self):
        before = self._routed()
        with self.assertRaisesRegex(ValueError, "exact same"):
            paired_report(before, before[:-1], META)
        after = copy.deepcopy(before)
        after[1]["true_parent"] = 1
        with self.assertRaisesRegex(ValueError, "annotations changed"):
            paired_report(before, after, META)

    def test_alias_dedup_recall_and_macro(self):
        before = self._routed()
        report = paired_report(before + [copy.deepcopy(before[0])], list(reversed(before)), META)
        self.assertEqual(report["unique_image_count"], 4)
        self.assertEqual(report["known_added_harm"]["denominator"], 1)
        self.assertEqual(report["source_macro"]["intra"]["selected"], .5)
        self.assertEqual(report["parent_recall"]["intra"]["support"]["top2"]["rate"], 1.)
        self.assertTrue(report["preservation_audit"]["passed"])

    def test_continuous_evidence_and_rule_local_source_support(self):
        before = self._routed()
        diagnostics = score_diagnostics(before)
        self.assertIn("selected_parent_text_z", diagnostics["components"])
        self.assertEqual(diagnostics["components"]["geometry_leaf_score"]["positive_count"], 1)
        after = copy.deepcopy(before)
        after[1].update(prediction_type="intra_unknown", leaf=None, output_node=1, applied_rule_indices=[0])
        report = rule_support(before, before, after, META)
        self.assertEqual(report["rules"]["0"]["improved_source_count"], 1)
        self.assertEqual(report["rules"]["0"]["evidence_status"], "insufficient_evidence")
        self.assertEqual(report["rules"]["0"]["per_parent"]["p"]["improved_source_count"], 1)
        self.assertEqual(report["rules"]["0"]["per_parent"]["q"]["evidence_status"], "not_evaluable")


if __name__ == "__main__":
    unittest.main()
