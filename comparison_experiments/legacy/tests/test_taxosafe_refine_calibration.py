"""Preservation, isolation and exact baseline fallback for leaf refinement."""
import copy
import hashlib
import unittest
from unittest.mock import patch

import numpy as np

from taxosafe_support import calibration as base
from taxosafe_refine import calibration as refine

META = {"parent_names": ["p", "q"], "leaf_names": ["a", "b", "c"], "leaf_to_parent": [0, 0, 1]}
BASELINE = {"schema_version": "support_membership_v1", "decoder": "membership", "meta": META,
            "parent_threshold": 0., "leaf_threshold": 0.}
SETTINGS = {"source_loo": False, "grid_points": 3}


def row(identity, status, pm=1., lm=1., reconstruction=1., source=None, parent=0, leaf=0):
    return {"image_sha256": hashlib.sha256(identity.encode()).hexdigest(),
            "split": {"known": "val_known", "intra": "val_intra", "extra": "val_extra"}[status],
            "status": status, "source": source or status,
            "true_parent": None if status == "extra" else parent,
            "true_leaf": leaf if status == "known" else None,
            "log_probs": [-float(np.log(6))] * 6,
            "support_evidence": {"parent_logits": [2., 1.], "leaf_logits": [2., 1., 10.],
                                 "parent_membership_logits": [pm, 100.],
                                 "leaf_membership_logits": [lm, 100., 100.]},
            "reconstruction_score": reconstruction}


def fixture():
    return ([row("k", "known")], [row("n", "intra", reconstruction=-1.)],
            [row("e", "extra", pm=-1.)])


class RefineCalibrationTests(unittest.TestCase):
    def test_improves_near_and_precision_preserving_known_and_root(self):
        router = refine.calibrate(*fixture(), BASELINE, META, SETTINGS)
        self.assertEqual(router["schema_version"], refine.SCHEMA_VERSION)
        self.assertEqual(router["status"], "feasible_refinement")
        self.assertTrue(router["targets_passed"])
        self.assertTrue(router["preservation_audit"]["passed"])
        self.assertTrue(router["preservation_audit"]["strict_near_or_precision_improvement"])
        self.assertEqual(router["fitted_parameters"], ["leaf_threshold", "reconstruction_threshold"])
        self.assertEqual(router["fixed_parent_threshold"], BASELINE["parent_threshold"])
        self.assertEqual(router["baseline_router"], BASELINE)
        self.assertEqual(router["validation_report"]["counts"], {"known": 1, "intra": 1, "extra": 1,
                         "known_correct": 1, "intra_correct": 1, "extra_correct": 1, "leaf_outputs": 1})
        self.assertEqual(router["baseline_validation_report"]["counts"]["leaf_outputs"], 2)

    def test_baseline_fallback_is_exact_for_arbitrary_unseen_reconstruction(self):
        known, near, extra = fixture()
        near[0]["support_evidence"]["leaf_membership_logits"][0] = -1.
        router = refine.calibrate(known, near, extra, BASELINE, META, SETTINGS)
        self.assertEqual(router["status"], "baseline_fallback")
        self.assertFalse(router["reconstruction_gate_enabled"])
        self.assertEqual(router["leaf_threshold"], BASELINE["leaf_threshold"])
        self.assertTrue(router["targets_passed"])  # Fallback status is independent of original gates.
        records = known + near + extra
        for i, record in enumerate(records):
            record["reconstruction_score"] = -1.e100 if i % 2 else 1.e100
            record["split"] = "test_" + record["status"]
        before, after = base.apply_router(records, BASELINE, META), refine.apply_router(records, router, META)
        fields = ("prediction_type", "parent", "leaf", "output_node", "candidate_parent", "candidate_leaf",
                  "parent_membership_score", "leaf_membership_score", "root_knownness_score", "parent_threshold")
        for left, right in zip(before, after):
            self.assertEqual({k: left[k] for k in fields}, {k: right[k] for k in fields})

    def test_no_improvement_fallback_does_not_pretend_gates_passed(self):
        known, near, extra = fixture()
        near[0]["reconstruction_score"] = 1.
        router = refine.calibrate(known, near, extra, BASELINE, META, SETTINGS)
        self.assertEqual(router["status"], "baseline_fallback")
        self.assertFalse(router["targets_passed"])
        self.assertTrue(router["fit_completed"])
        self.assertTrue(router["best_effort"])
        self.assertTrue(router["preservation_audit"]["passed"])
        self.assertFalse(router["preservation_audit"]["strict_near_or_precision_improvement"])

    def test_known_preservation_exact_precision_comparison_and_zero_leaf_failure(self):
        original = {"known": 20, "intra": 20, "extra": 20, "known_correct": 19,
                    "intra_correct": 10, "extra_correct": 20, "leaf_outputs": 25}
        worse_known = dict(original, known_correct=18, intra_correct=20, leaf_outputs=18)
        self.assertFalse(refine._preservation(worse_known, original)["passed"])
        less_near = dict(original, intra_correct=9, leaf_outputs=19)
        self.assertFalse(refine._preservation(less_near, original)["passed"])
        less_precision = dict(original, leaf_outputs=26, intra_correct=20)
        self.assertFalse(refine._preservation(less_precision, original)["passed"])
        equal_precision = dict(original, known_correct=20, leaf_outputs=40)
        half_precision_base = dict(original, known_correct=19, leaf_outputs=38)
        audit = refine._preservation(equal_precision, half_precision_base)
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["precision_cross_product"]["selected_correct_times_baseline_leaf_outputs"],
                         audit["precision_cross_product"]["baseline_correct_times_selected_leaf_outputs"])
        self.assertFalse(refine._preservation(dict(original, leaf_outputs=0), original)["passed"])
        below_known_gate = dict(original, known_correct=18)
        self.assertFalse(refine._preservation(below_known_gate, below_known_gate)["passed"])

    def test_root_and_candidate_invariant_for_any_new_leaf_threshold(self):
        router = refine.calibrate(*fixture(), BASELINE, META, SETTINGS)
        records = [row("x%d" % i, "extra", pm=pm, lm=lm, reconstruction=rc)
                   for i, (pm, lm, rc) in enumerate(((-1., 100., 100.), (0., 0., 0.), (1., -2., 3.), (2., 5., -9.)))]
        before = base.apply_router(records, BASELINE, META)
        for lt, rt in ((-20., -20.), (0., 0.), (20., 20.)):
            changed = dict(router, leaf_threshold=lt, reconstruction_threshold=rt, reconstruction_gate_enabled=True)
            after = refine.apply_router(records, changed, META)
            for left, right in zip(before, after):
                self.assertEqual(left["prediction_type"] == "global_unknown", right["prediction_type"] == "global_unknown")
                self.assertEqual(left["candidate_parent"], right["candidate_parent"])
                self.assertEqual(left["candidate_leaf"], right["candidate_leaf"])
                self.assertEqual(left["root_knownness_score"], right["root_knownness_score"])
                self.assertEqual(left["support_evidence"], right["support_evidence"])
                self.assertEqual(left["log_probs"], right["log_probs"])
        tied = refine.apply_router([records[1]], dict(router, leaf_threshold=0., reconstruction_threshold=0.), META)[0]
        self.assertEqual(tied["prediction_type"], "known")
        self.assertEqual(tied["local_known_margin"], 0.)

    def test_parent_routing_ceiling_uses_actual_fixed_candidates_and_threshold(self):
        known, near, extra = fixture()
        near.extend([row("nroot", "intra", pm=-1.), row("nwrong", "intra", parent=1)])
        router = refine.calibrate(known, near, extra, BASELINE, META, SETTINGS)
        cap = router["validation_report"]["parent_routing_near_upper_bound"]
        self.assertEqual((cap["correct_count"], cap["total"], cap["required_correct"]), (1, 3, 3))
        self.assertFalse(cap["can_reach_target"])
        self.assertFalse(router["targets_passed"])
        self.assertEqual(router["status"], "best_effort_refinement")

    def test_no_test_fitting_or_empty_groups_in_calibration_and_loo(self):
        for function in (refine.calibrate, refine.source_loo):
            for idx in range(3):
                args = fixture()
                args[idx][0]["split"] = "test_" + args[idx][0]["status"]
                with self.assertRaisesRegex(ValueError, "test fitting is prohibited"):
                    function(*args, BASELINE, META, SETTINGS)
            args = list(fixture())
            args[1] = []
            with self.assertRaisesRegex(ValueError, "nonempty val_intra"):
                function(*args, BASELINE, META, SETTINGS)

    def test_reconstruction_finite_scalar_required_and_hash_bound(self):
        for bad in (None, True, [1.], float("nan"), float("inf")):
            args = fixture()
            args[0][0]["reconstruction_score"] = bad
            with self.subTest(value=bad), self.assertRaisesRegex(ValueError, "finite scalar"):
                refine.calibrate(*args, BASELINE, META, SETTINGS)
        args = fixture()
        first = refine.calibrate(*args, BASELINE, META, SETTINGS)
        args[2][0]["reconstruction_score"] += .1  # Root is invariant, but evidence identity changes.
        second = refine.calibrate(*args, BASELINE, META, SETTINGS)
        self.assertNotEqual(first["evidence_sha256"], second["evidence_sha256"])
        self.assertNotEqual(first["calibration_sha256"], second["calibration_sha256"])

    def test_duplicate_reconstruction_conflict_rejected_before_dedup(self):
        known, near, extra = fixture()
        alias = copy.deepcopy(known[0])
        alias["path"] = "alias.png"
        router = refine.calibrate(known + [alias], near, extra, BASELINE, META, SETTINGS)
        self.assertEqual((router["unique_image_count"], router["duplicate_record_count"]), (3, 1))
        predictions = refine.apply_router(known + [alias] + near + extra, router, META)
        self.assertEqual(len(predictions), 4)
        self.assertEqual(refine.evaluate_records(predictions)["unique_image_count"], 3)
        alias["reconstruction_score"] += 1.
        with self.assertRaisesRegex(ValueError, "inconsistent reconstruction evidence"):
            refine.calibrate(known + [alias], near, extra, BASELINE, META, SETTINGS)

    def test_grid_always_contains_baseline_and_all_accept_point(self):
        settings = dict(SETTINGS, leaf_threshold_grid=[99.], reconstruction_threshold_grid=[99.])
        router = refine.calibrate(*fixture(), BASELINE, META, settings)
        self.assertIn(BASELINE["leaf_threshold"], router["grid"]["leaf_threshold"])
        self.assertLess(router["grid"]["reconstruction_threshold"][0], -1.)
        self.assertGreater(router["grid"]["reconstruction_threshold"][-1], 1.)
        self.assertTrue(router["grid"]["reconstruction_all_accept_included"])

    def test_schema_parent_override_and_invalid_baseline_rejected(self):
        router = refine.calibrate(*fixture(), BASELINE, META, SETTINGS)
        for bad in (dict(router, schema_version="other"), dict(router, fixed_parent_threshold=1.),
                    dict(router, leaf_threshold=float("nan")), dict(router, meta={}),
                    dict(router, reconstruction_gate_enabled=False, leaf_threshold=1.)):
            with self.subTest(state=bad), self.assertRaises(ValueError):
                refine.apply_router(sum(fixture(), []), bad, META)
        with self.assertRaises(ValueError):
            refine.calibrate(*fixture(), {"parent_bias": 0., "leaf_bias": 0.}, META, SETTINGS)
        with self.assertRaisesRegex(ValueError, "baseline_calibration"):
            refine.source_loo(*fixture(), BASELINE, META, SETTINGS)

    def test_loo_refits_baseline_and_refinement_without_held_source(self):
        known, near, extra = fixture()
        near.append(row("n2", "intra", pm=3., lm=-1., reconstruction=-10., source="near2"))
        extra.append(row("e2", "extra", pm=-3., reconstruction=10., source="extra2"))
        settings = dict(SETTINGS, baseline_calibration={"decoder": "membership", "policy": "known_first", "membership_grid_points": 3})
        calls, refinement_calls = [], []
        original_base, original_select = base.calibrate, refine._select
        def fit_baseline(known, near, extra, meta, settings):
            fitted = original_base(known, near, extra, meta, settings)
            calls.append((known + near + extra, dict(settings), fitted))
            return fitted
        def fit_refine(rows, baseline_router, *args, **kwargs):
            refinement_calls.append((list(rows), baseline_router))
            return original_select(rows, baseline_router, *args, **kwargs)
        with patch.object(base, "calibrate", side_effect=fit_baseline), patch.object(refine, "_select", side_effect=fit_refine):
            report = refine.source_loo(known, near, extra, dict(BASELINE, parent_threshold=999.), META, settings)
        self.assertEqual(len(report["folds"]), 4)
        for fold, (rows, settings, fitted), (refine_rows, refine_base) in zip(report["folds"], calls, refinement_calls):
            self.assertFalse(any(r["status"] == fold["status"] and r["source"] == fold["held_source"] for r in rows))
            self.assertEqual(rows, refine_rows)
            self.assertIs(fitted, refine_base)
            self.assertFalse(settings["source_loo"])
            self.assertEqual(fold["baseline_parent_threshold"], fitted["parent_threshold"])
            self.assertNotEqual(fold["baseline_parent_threshold"], 999.)
            self.assertTrue(fold["baseline_refit_on_fit_sources_only"])
            self.assertTrue(fold["grid_uses_fit_sources_only"])


if __name__ == "__main__":
    unittest.main()
