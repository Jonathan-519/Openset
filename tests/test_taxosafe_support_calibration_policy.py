"""Regression and exact-count tests for optional known-first calibration."""
import hashlib
import unittest
from unittest.mock import patch

import numpy as np

from taxosafe_support.calibration import calibrate, fixed_score_feasibility, source_loo

META = {"parent_names": ["parent"], "leaf_names": ["leaf"], "leaf_to_parent": [0]}
SETTINGS = {"parent_bias_grid": [0.], "leaf_bias_grid": [-2., 0.], "source_loo": False}


def record(identity, status, scores, parent=None, leaf=None, source=None):
    scores = np.asarray(scores, dtype=float)
    log_probs = scores - scores.max() - np.log(np.exp(scores - scores.max()).sum())
    return {"image_sha256": hashlib.sha256(identity.encode()).hexdigest(),
            "split": {"known": "val_known", "intra": "val_intra", "extra": "val_extra"}[status],
            "status": status, "source": source or status, "true_parent": parent, "true_leaf": leaf,
            "log_probs": log_probs.tolist()}


def conflict_fixture(weak_known=3):
    known = [record("k%d" % i, "known", [-10, 0, 1 if i < weak_known else 5], 0, 0)
             for i in range(20)]
    near = [record("n%d" % i, "intra", [-10, 0, 1], 0, source="near%d" % (i % 2))
            for i in range(20)]
    extra = [record("e%d" % i, "extra", [10, 0, 0], source="extra%d" % (i % 2))
             for i in range(20)]
    return known, near, extra


def feasible_fixture():
    return ([record("k", "known", [-10, 0, 5], 0, 0)],
            [record("n", "intra", [-10, 5, 0], 0)],
            [record("e", "extra", [10, 0, 0])])


class CalibrationPolicyTests(unittest.TestCase):
    def test_default_balanced_keeps_legacy_compromise(self):
        args = conflict_fixture()
        default = calibrate(*args, META, SETTINGS)
        explicit = calibrate(*args, META, dict(SETTINGS, policy="balanced"))
        self.assertEqual((default["parent_bias"], default["leaf_bias"]), (0., -2.))
        self.assertEqual(default["validation_report"]["counts"],
                         {"known": 20, "intra": 20, "extra": 20,
                          "known_correct": 17, "intra_correct": 20, "extra_correct": 20,
                          "leaf_outputs": 17})
        self.assertEqual(default["calibration_sha256"], explicit["calibration_sha256"])
        # Verified against calibration.py from the pre-change 1d7e4e4 commit.
        self.assertEqual(default["calibration_sha256"],
                         "954173941eec46af97f7ec715b5f243e149856011ea512ddf2e460d8f6d84d42")
        self.assertEqual(default["status"], "best_effort")
        self.assertFalse(default["targets_passed"])

    def test_known_first_preserves_known_but_never_claims_joint_success(self):
        state = calibrate(*conflict_fixture(), META, dict(SETTINGS, policy="known_first"))
        self.assertEqual(state["leaf_bias"], 0.)
        self.assertTrue(state["fit_completed"])
        self.assertFalse(state["targets_passed"])
        self.assertTrue(state["best_effort"])
        self.assertEqual(state["status"], "best_effort_known_preserved")
        self.assertTrue(state["validation_report"]["checks"]["known_end_to_end_leaf_accuracy"])
        self.assertEqual(state["selection_diagnostics"]["sampled_known_feasible_count"], 1)
        self.assertEqual(state["validation_report"]["status"], state["status"])

    def test_exactly_ninety_percent_is_not_known_preserved(self):
        state = calibrate(*conflict_fixture(weak_known=2), META,
                          dict(SETTINGS, policy="known_first"))
        self.assertEqual(state["leaf_bias"], 0.)
        self.assertEqual(state["selection_diagnostics"]["sampled_known_feasible_count"], 1)
        self.assertEqual(state["validation_report"]["requirements"]["known_end_to_end_leaf_accuracy"]["required_correct"], 19)

    def test_known_unavailable_explicitly_retains_best_effort(self):
        settings = dict(SETTINGS, leaf_bias_grid=[-2.], policy="known_first")
        state = calibrate(*conflict_fixture(), META, settings)
        self.assertEqual(state["status"], "best_effort_known_unavailable")
        self.assertFalse(state["selection_diagnostics"]["known_feasible_on_grid"])
        self.assertFalse(state["targets_passed"])
        self.assertTrue(state["fit_completed"])

    def test_joint_feasible_dominates_and_policy_is_hash_bound(self):
        balanced = calibrate(*feasible_fixture(), META, SETTINGS)
        known_first = calibrate(*feasible_fixture(), META, dict(SETTINGS, policy="known_first"))
        self.assertEqual((balanced["parent_bias"], balanced["leaf_bias"]),
                         (known_first["parent_bias"], known_first["leaf_bias"]))
        self.assertEqual(known_first["status"], "feasible")
        self.assertTrue(known_first["targets_passed"])
        self.assertFalse(known_first["best_effort"])
        self.assertNotEqual(known_first["calibration_sha256"], balanced["calibration_sha256"])
        self.assertFalse(known_first["validation_report"]["fixed_score_feasibility"]["continuous_infeasibility_proven"])

    def test_equal_order_statistic_bounds_are_incompatible_due_to_leaf_tie(self):
        args = ([record("k", "known", [-10, 0, 1], 0, 0)],
                [record("n", "intra", [-10, 0, 1], 0)],
                [record("e", "extra", [10, 0, 0])])
        report = fixed_score_feasibility(*args, META)
        bounds = report["necessary_bounds"]
        self.assertEqual(bounds["known_requires_parent_minus_leaf_bias_strictly_below"],
                         bounds["near_requires_parent_minus_leaf_bias_at_least"])
        self.assertTrue(report["continuous_infeasibility_proven"])
        self.assertIn("known_and_near_require_incompatible_depth_difference", report["contradictions"])
        self.assertEqual(bounds["near_correct_upper_bound_given_known_ignoring_root"], 0)

    def test_candidate_ceiling_is_not_reported_as_bias_failure(self):
        meta = {"parent_names": ["p"], "leaf_names": ["a", "b"], "leaf_to_parent": [0, 0]}
        known = [record("k%d" % i, "known", [-10, 0, 5 if i else 3, 4], 0, 0)
                 for i in range(10)]
        near = [record("n", "intra", [-10, 5, 0, 0], 0)]
        extra = [record("e", "extra", [10, 0, 0, 0])]
        report = fixed_score_feasibility(known, near, extra, meta)
        self.assertEqual(report["candidate_correct"]["known"], 9)
        self.assertEqual(report["required_correct"]["known"], 10)
        self.assertIn("known_candidate_accuracy_below_required", report["contradictions"])

    def test_test_and_empty_groups_are_forbidden_for_fit_and_diagnostic(self):
        for function in (calibrate, fixed_score_feasibility):
            args = list(feasible_fixture())
            args[0][0]["split"] = "test_known"
            with self.assertRaisesRegex(ValueError, "test fitting is prohibited"):
                function(*args, META)
            args = list(feasible_fixture())
            args[1] = []
            with self.assertRaisesRegex(ValueError, "nonempty val_intra"):
                function(*args, META)
        with self.assertRaisesRegex(ValueError, "policy"):
            calibrate(*feasible_fixture(), META, dict(SETTINGS, policy="silently_relax_targets"))

    def test_source_loo_uses_same_policy_without_held_source(self):
        import taxosafe_support.calibration as module
        original = module._select
        calls = []
        def track(rows, *args, **kwargs):
            calls.append((list(rows), kwargs["policy"]))
            return original(rows, *args, **kwargs)
        with patch.object(module, "_select", side_effect=track):
            report = source_loo(*conflict_fixture(), META, dict(SETTINGS, policy="known_first"))
        self.assertEqual(len(report["folds"]), 4)
        for fold, (rows, policy) in zip(report["folds"], calls):
            self.assertEqual(policy, "known_first")
            self.assertEqual(fold["selection_policy"], policy)
            self.assertFalse(any(r["status"] == fold["status"] and r["source"] == fold["held_source"] for r in rows))
            self.assertEqual(fold["fit_selection_status"], "best_effort_known_preserved")


if __name__ == "__main__":
    unittest.main()
