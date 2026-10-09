"""Exact-count tests for fixed-candidate, development-only membership gates."""
import copy
import hashlib
import unittest
from unittest.mock import patch

import numpy as np

from taxosafe_support.calibration import apply_router, calibrate, evaluate_records, source_loo, unique_records
from taxosafe_support import membership_calibration as module

META = {"parent_names": ["p", "q"], "leaf_names": ["a", "b", "c"], "leaf_to_parent": [0, 0, 1]}
SETTINGS = {"decoder": "membership", "policy": "known_first", "source_loo": False,
            "parent_threshold_grid": [0.], "leaf_threshold_grid": [0.]}


def row(identity, status, pm, lm, parent=0, leaf=0, source=None):
    return {"image_sha256": hashlib.sha256(identity.encode()).hexdigest(),
            "split": {"known": "val_known", "intra": "val_intra", "extra": "val_extra"}[status],
            "status": status, "source": source or status,
            "true_parent": None if status == "extra" else parent,
            "true_leaf": leaf if status == "known" else None,
            "log_probs": [-float(np.log(6))] * 6,
            "support_evidence": {"parent_logits": [2., 1.], "leaf_logits": [2., 1., 10.],
                                 "parent_membership_logits": [pm, 100.],
                                 "leaf_membership_logits": [lm, 100., 100.]}}


def fixture():
    return ([row("k", "known", 1., 1.)], [row("n", "intra", 1., -1.)], [row("e", "extra", -1., 1.)])


def state(parent_threshold=0., leaf_threshold=0.):
    return {"schema_version": module.SCHEMA_VERSION, "decoder": "membership",
            "parent_threshold": parent_threshold, "leaf_threshold": leaf_threshold}


class MembershipCalibrationTests(unittest.TestCase):
    def test_candidates_fixed_before_thresholds_and_no_evidence_borrowing(self):
        original = row("x", "extra", -1., -2.)
        kinds = []
        for pt, lt in ((0., 0.), (-1., 0.), (-1., -2.)):
            routed = apply_router([original], state(pt, lt), META)[0]
            kinds.append(routed["prediction_type"])
            self.assertEqual((routed["candidate_parent"], routed["candidate_leaf"]), (0, 0))
            self.assertEqual((routed["parent_membership_score"], routed["leaf_membership_score"]), (-1., -2.))
            self.assertEqual(routed["support_evidence"], original["support_evidence"])
            self.assertEqual(routed["root_score_type"], "selected_parent_membership_logit")
        self.assertEqual(kinds, ["global_unknown", "intra_unknown", "known"])
        self.assertNotIn("prediction_type", original)

    def test_ranking_ties_choose_lowest_index_independent_of_membership(self):
        original = row("tie", "extra", 0., 0.)
        original["support_evidence"]["parent_logits"] = [1., 1.]
        original["support_evidence"]["leaf_logits"] = [1., 1., 100.]
        output = apply_router([original], state(), META)[0]
        self.assertEqual((output["parent"], output["leaf"], output["output_node"]), (0, 0, 3))
        self.assertEqual(output["prediction_type"], "known")  # Both gate ties accept.
        self.assertEqual(apply_router([], state(), META), [])

    def test_joint_feasible_report_and_schema_dispatch_for_both_policies(self):
        for policy in ("known_first", "balanced"):
            fitted = calibrate(*fixture(), META, dict(SETTINGS, policy=policy))
            self.assertEqual(fitted["schema_version"], "support_membership_v1")
            self.assertEqual(fitted["status"], "feasible")
            self.assertTrue(fitted["targets_passed"])
            self.assertTrue(fitted["fit_completed"])
            self.assertFalse(fitted["best_effort"])
            self.assertEqual(fitted["fitted_parameters"], ["parent_threshold", "leaf_threshold"])
            self.assertNotIn("parent_bias", fitted)
            self.assertEqual(fitted["validation_report"]["fixed_score_feasibility"]["decoder"], "membership")
            routed = apply_router(sum(fixture(), []), fitted, META)
            self.assertEqual([r["prediction_type"] for r in routed], ["known", "intra_unknown", "global_unknown"])
            self.assertEqual(evaluate_records(routed)["counts"], fitted["validation_report"]["counts"])

    def test_known_first_preserves_strict_known_floor_without_claiming_success(self):
        known = [row("k%d" % i, "known", 5., 1. if i < 2 else 5.) for i in range(20)]
        near = [row("n%d" % i, "intra", 5., 1.) for i in range(20)]
        extra = [row("e%d" % i, "extra", -5., 1.) for i in range(20)]
        settings = dict(SETTINGS, leaf_threshold_grid=[0., 2.])
        fitted = calibrate(known, near, extra, META, settings)
        balanced = calibrate(known, near, extra, META, dict(settings, policy="balanced"))
        self.assertEqual(fitted["leaf_threshold"], 0.)
        self.assertEqual(balanced["leaf_threshold"], 2.)
        self.assertEqual(balanced["validation_report"]["metrics"]["known_end_to_end_leaf_accuracy"], .9)
        self.assertFalse(balanced["validation_report"]["checks"]["known_end_to_end_leaf_accuracy"])
        self.assertEqual(fitted["status"], "best_effort_known_preserved")
        self.assertFalse(fitted["targets_passed"])
        self.assertTrue(fitted["best_effort"])
        self.assertEqual(fitted["selection_diagnostics"]["sampled_known_feasible_count"], 1)
        impossible = calibrate(known, near, extra, META, dict(settings, leaf_threshold_grid=[2.]))
        self.assertEqual(impossible["status"], "best_effort_known_unavailable")

    def test_precision_denominator_includes_wrong_known_and_all_unknown_leaves(self):
        records = [row("k", "known", 1., 1.), row("kw", "known", 1., 1., leaf=1),
                   row("n", "intra", 1., 1.), row("e", "extra", 1., 1.)]
        report = evaluate_records(apply_router(records, state(), META))
        self.assertEqual(report["counts"]["leaf_outputs"], 4)
        self.assertEqual(report["metrics"]["open_world_accepted_leaf_precision"], .25)
        empty_leaf = evaluate_records(apply_router(records, state(2.), META))
        self.assertIsNone(empty_leaf["metrics"]["open_world_accepted_leaf_precision"])
        self.assertFalse(empty_leaf["targets_passed"])

    def test_evidence_must_be_complete_finite_and_correct_shape(self):
        for field in module.RAW_FIELDS:
            for defect in ("missing", "nan", "shape"):
                original = row("k", "known", 1., 1.)
                if defect == "missing":
                    del original["support_evidence"][field]
                elif defect == "nan":
                    original["support_evidence"][field][0] = float("nan")
                else:
                    original["support_evidence"][field].append(0.)
                with self.subTest(field=field, defect=defect), self.assertRaises(ValueError):
                    apply_router([original], state(), META)
        original.pop("support_evidence")
        with self.assertRaisesRegex(ValueError, "requires raw"):
            apply_router([original], state(), META)

    def test_malformed_or_mismatched_state_fails_closed(self):
        for bad in (dict(state(), schema_version="support_v1"), dict(state(), decoder="joint"),
                    dict(state(), candidate_rule="best_membership"), dict(state(), leaf_threshold=float("inf")),
                    dict(state(), meta={}), dict(state(), decoder="unknown", schema_version="other")):
            with self.subTest(state=bad), self.assertRaises(ValueError):
                apply_router([row("x", "extra", 1., 1.)], bad, META)
        with self.assertRaisesRegex(ValueError, "Unknown calibration decoder"):
            calibrate(*fixture(), META, dict(SETTINGS, decoder="unknown"))

    def test_dedup_rejects_one_sided_or_conflicting_raw_evidence(self):
        original = row("k", "known", 1., 1.)
        alias = copy.deepcopy(original)
        alias["path"] = "alias.png"
        self.assertEqual(len(unique_records([original, alias])), 1)
        alias.pop("support_evidence")
        with self.assertRaisesRegex(ValueError, "missing raw support evidence"):
            unique_records([original, alias])
        alias = copy.deepcopy(original)
        alias["support_evidence"]["leaf_membership_logits"][1] += 1.
        with self.assertRaisesRegex(ValueError, "inconsistent raw support evidence"):
            unique_records([original, alias])
        known, near, extra = fixture()
        fitted = calibrate(known + copy.deepcopy(known), near, extra, META, SETTINGS)
        self.assertEqual((fitted["unique_image_count"], fitted["duplicate_record_count"]), (3, 1))
        self.assertEqual(len(apply_router(known + known + near + extra, fitted, META)), 4)

    def test_hash_binds_even_unselected_raw_heads_and_policy(self):
        args = fixture()
        before = calibrate(*args, META, SETTINGS)
        args[0][0]["support_evidence"]["leaf_membership_logits"][1] += 1.
        after = calibrate(*args, META, SETTINGS)
        self.assertNotEqual(before["evidence_sha256"], after["evidence_sha256"])
        self.assertNotEqual(before["calibration_sha256"], after["calibration_sha256"])
        self.assertEqual(before["validation_report"]["counts"], after["validation_report"]["counts"])
        balanced = calibrate(*args, META, dict(SETTINGS, policy="balanced"))
        self.assertNotEqual(after["calibration_sha256"], balanced["calibration_sha256"])

    def test_quantile_grid_has_finite_all_pass_and_all_reject_endpoints(self):
        values = np.asarray([-5., 0., 3., 8.])
        grid = module._grid(values, {"membership_grid_points": 3}, "parent_threshold_grid")
        self.assertTrue(np.isfinite(grid).all())
        self.assertLess(grid[0], values.min())
        self.assertGreater(grid[-1], values.max())
        self.assertEqual(grid.tolist()[1:-1], [-5., 1.5, 8.])
        for settings in ({"threshold_grid": "test"}, {"membership_grid_points": 1},
                         {"membership_grid_points": True}, {"parent_threshold_grid": []},
                         {"parent_threshold_grid": [float("inf")]}, {"parent_threshold_grid": [[1.]]}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                module._grid(values, settings, "parent_threshold_grid")

    def test_test_splits_and_missing_groups_forbidden_for_fit_loo_and_diagnostic(self):
        for function in (calibrate, source_loo, module.fixed_score_feasibility):
            kwargs = {} if function is module.fixed_score_feasibility else {"settings": SETTINGS}
            args = fixture()
            args[0][0]["split"] = "test_known"
            with self.assertRaisesRegex(ValueError, "test fitting is prohibited"):
                function(*args, META, **kwargs)
            args = list(fixture())
            args[2] = []
            with self.assertRaisesRegex(ValueError, "nonempty val_extra"):
                function(*args, META, **kwargs)

    def test_loo_grid_and_threshold_fit_exclude_held_source(self):
        known, near, extra = fixture()
        near.append(row("n2", "intra", 100., -100., source="near2"))
        extra.append(row("e2", "extra", -100., 100., source="extra2"))
        calls = []
        original = module._select
        def track(rows, *args, **kwargs):
            result = original(rows, *args, **kwargs)
            calls.append((list(rows), result))
            return result
        with patch.object(module, "_select", side_effect=track):
            report = source_loo(known, near, extra, META,
                                {"decoder": "membership", "membership_grid_points": 3})
        self.assertEqual(len(report["folds"]), 4)
        for fold, (rows, result) in zip(report["folds"], calls):
            self.assertFalse(any(r["status"] == fold["status"] and r["source"] == fold["held_source"] for r in rows))
            self.assertTrue(fold["grid_uses_fit_sources_only"])
            scores = module.candidate_scores(rows, META)
            self.assertEqual(result["grid"]["parent_threshold"][0],
                             np.nextafter(scores["parent_score"].min(), -np.inf))
            self.assertEqual(result["grid"]["leaf_threshold"][-1],
                             np.nextafter(scores["leaf_score"].max(), np.inf))

    def test_membership_necessary_bounds_respect_strict_tie_rejection(self):
        known, near, extra = fixture()
        near[0]["support_evidence"]["leaf_membership_logits"][0] = 1.
        report = module.fixed_score_feasibility(known, near, extra, META)
        self.assertTrue(report["continuous_infeasibility_proven"])
        self.assertIn("known_and_near_require_incompatible_leaf_membership_threshold", report["contradictions"])
        self.assertEqual(report["necessary_bounds"]["near_correct_upper_bound_given_known_ignoring_parent"], 0)
        self.assertNotIn("known_requires_parent_minus_leaf_bias_strictly_below", report["necessary_bounds"])
        near[0]["support_evidence"]["leaf_membership_logits"][0] = -1.
        extra[0]["support_evidence"]["parent_membership_logits"][0] = 1.
        report = module.fixed_score_feasibility(known, near, extra, META)
        self.assertIn("known_and_extra_require_incompatible_parent_membership_threshold", report["contradictions"])
        self.assertIn("near_and_extra_require_incompatible_parent_membership_threshold", report["contradictions"])

    def test_candidate_ceiling_is_separate_from_membership_overlap(self):
        known, near, extra = fixture()
        known[0]["true_leaf"] = 1
        report = module.fixed_score_feasibility(known, near, extra, META)
        self.assertEqual(report["candidate_correct"]["known"], 0)
        self.assertIn("known_ranking_candidate_ceiling_below_target", report["contradictions"])
        self.assertNotIn("known_and_extra_require_incompatible_parent_membership_threshold", report["contradictions"])


if __name__ == "__main__":
    unittest.main()
