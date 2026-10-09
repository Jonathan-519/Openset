"""Tier feasibility, truthful Pareto export and fit-only Boundary audits."""
import copy
import json
import unittest
from fractions import Fraction
from unittest.mock import patch

import numpy as np

from taxosafe_discovery import calibration as discovery
from taxosafe_boundary import calibration as boundary
from tests.test_taxosafe_discovery_calibration import META, fixture, row, bundle
from tests.test_taxosafe_recovery_calibration import d05_bundle, buffered_fixture


def choose(counts, quality=None, regret=None, totals=(219, 69, 88)):
    counts = np.array(counts, dtype=np.int64)
    size = len(counts)
    return boundary._choose(counts, np.zeros(size) if quality is None else np.array(quality),
        np.zeros(size) if regret is None else np.array(regret), np.arange(size, dtype=float),
        np.arange(size, dtype=float), totals)


class BoundaryCalibrationTests(unittest.TestCase):
    def test_strict_precision_domain_beats_90_percent_lower_deficit(self):
        # Recovery's real failure pattern: 198/220 is exactly 90%, not >90%.
        index, report = choose([[198, 44, 60, 220], [199, 44, 59, 221]], quality=[1., 0.], regret=[0., 1.])
        self.assertEqual(index, 1)
        self.assertEqual(report["selected_tier"], "known_precision")
        self.assertEqual(report["feasible_counts"]["known_precision"], 1)

    def test_precision_and_open_preferences_cannot_collapse_available_known_gate(self):
        index, report = choose([[184, 69, 88, 199], [198, 38, 58, 220]], quality=[1., 0.], regret=[0., 1.])
        self.assertEqual(index, 1)
        self.assertEqual(report["selected_tier"], "known_only")
        self.assertEqual(report["reason"], "precision_infeasible_given_known_gate")

    def test_four_gates_beat_more_known_when_extra_or_near_would_fail(self):
        index, report = choose([[198, 59, 80, 210], [202, 58, 80, 215]], quality=[0., 1.])
        self.assertEqual(index, 0)
        self.assertEqual(report["selected_tier"], "four_gates")

    def test_known_infeasible_maximizes_known_before_precision(self):
        index, report = choose([[170, 69, 88, 175], [197, 20, 30, 300]], quality=[1., 0.])
        self.assertEqual(index, 1)
        self.assertEqual(report["selected_tier"], "maximum_known")
        self.assertEqual(report["reason"], "known_gate_infeasible")

    def test_capped_known_then_open_regret_then_quality_and_deterministic_ties(self):
        points = [[198, 50, 70, 215], [200, 48, 68, 219], [202, 46, 66, 224], [205, 45, 65, 226]]
        index, report = choose(points, quality=[1., 1., .9, 1.], regret=[0., 0., .01, .02])
        self.assertEqual(index, 2)  # cap 202 ties with205; less open decline wins
        self.assertEqual(report["known_required_for_target"], 202)
        index, _ = choose([[202, 46, 66, 224], [202, 46, 66, 224]])
        self.assertEqual(index, 1)  # final threshold tie: larger parent then leaf

    def test_known_only_prefers_precision_before_further_coverage(self):
        index, report = choose([[198, 40, 60, 223], [205, 40, 60, 240]])
        self.assertEqual(index, 0)
        self.assertEqual(report["selected_tier"], "known_only")

    def test_exact_pareto_matches_independent_bruteforce_including_equal_ratios(self):
        random = np.random.RandomState(7)
        counts = np.array([[int(k), int(n), int(e), int(l)] for k, n, e, l in
            zip(random.randint(0, 21, 100), random.randint(0, 12, 100), random.randint(0, 12, 100), random.randint(21, 41, 100))])
        counts = np.vstack([counts, [10, 12, 12, 20], [5, 12, 12, 10], [0, 15, 15, 0], [0, 14, 14, 1], [0, 14, 14, 2]])
        size = len(counts)
        result = boundary._pareto_indices(counts, np.zeros(size), np.zeros(size), np.arange(size), np.arange(size))
        metrics = lambda i: (*map(int, counts[i, :3]), Fraction(int(counts[i, 0]), int(counts[i, 3])))
        values = {metrics(i) for i in range(size) if counts[i, 3]}
        expected = {a for a in values if not any(all(x >= y for x, y in zip(b, a)) and b != a for b in values)}
        self.assertEqual({metrics(i) for i in result}, expected)
        self.assertEqual(len(result), len(expected))

    def test_standard_returns_exact_legacy_router_while_exposing_new_policy_comparison(self):
        groups = buffered_fixture()
        groups[0].append(copy.deepcopy(groups[0][0]))
        original, old_report = discovery.fit_router(*groups, META)
        actual, report = boundary.fit_router(*groups, META, policy="standard", d05_records=d05_bundle(groups))
        self.assertEqual(actual, original)
        self.assertEqual(report["metrics"], old_report["metrics"])
        self.assertTrue(report["boundary_policy"]["legacy_selection_unchanged"])
        self.assertEqual(report["boundary_policy"]["selected_tier"], "legacy_standard")
        self.assertIsNone(report["boundary_policy"]["selected_domain_candidates"])
        self.assertIsNotNone(report["boundary_policy"]["kp_policy_point"])
        self.assertEqual(report["duplicate_record_count"], 1)

    def test_fit_with_no_joint_precision_feasibility_keeps_known_and_own_router(self):
        baseline = buffered_fixture(correct=51)
        groups = copy.deepcopy(baseline)
        for item in groups[0]: item["discovery"]["leaf_scores"][0] = 0.
        for item in groups[1]+groups[2]:
            item["discovery"]["parent_scores"][0] = 2.
            item["discovery"]["leaf_scores"][0] = 5.
        state, report = boundary.fit_router(*groups, META, d05_records=d05_bundle(baseline))
        self.assertEqual(report["boundary_policy"]["selected_tier"], "known_only")
        self.assertEqual(report["counts"]["known_correct"], 51)
        self.assertFalse(report["targets_passed"])
        self.assertFalse(state["baseline_fallback"])
        self.assertTrue(report["boundary_policy"]["test_allowed_after_failed_gates"])
        self.assertEqual(len(boundary.decode_records(sum(groups, []), state, META)), 59)

    def test_exact_grid_matches_independent_decode_feasibility(self):
        groups = fixture(); groups[0][0]["discovery"]["leaf_scores"][0] = .17
        baseline = d05_bundle(groups)
        state, report = boundary.fit_router(*groups, META, d05_records=baseline)
        search = report["exact_threshold_search"]
        four = kp = known = 0
        for pt in search["parent_grid"]:
            for lt in search["leaf_grid"]:
                candidate = copy.deepcopy(state)
                candidate.update(global_parent_threshold=pt, global_leaf_threshold=lt,
                                 parent_thresholds=[pt, pt], leaf_thresholds=[lt, lt])
                candidate["router_sha256"] = discovery._router_hash(candidate)
                scored = discovery.base.evaluate_records(boundary.decode_records(sum(groups, []), candidate, META), META)
                k = scored["checks"]["known_end_to_end_leaf_accuracy"]
                p = scored["checks"]["open_world_accepted_leaf_precision"]
                four += int(scored["targets_passed"]); kp += int(k and p); known += int(k)
        self.assertEqual([four, kp, known], [search["feasible_counts"][key] for key in ("four_gates", "known_precision", "known")])
        self.assertTrue(search["includes_all_leaf_accept_and_reject"])
        self.assertTrue(search["includes_all_parent_accept_and_reject"])
        json.dumps(report, allow_nan=False)

    def test_fixed_parent_uses_new_selector_and_keeps_original_roots(self):
        baseline = fixture(); groups = copy.deepcopy(baseline)
        for item in groups[2]: item["discovery"]["leaf_scores"][0] = 1000.
        reference = d05_bundle(baseline)
        state, report = boundary.fit_router(*groups, META, policy="fixed_parent", d05_records=reference)
        self.assertEqual(state["global_parent_threshold"], reference["router"]["global_parent_threshold"])
        self.assertEqual(report["exact_threshold_search"]["parent_grid"], [state["global_parent_threshold"]])
        self.assertFalse(report["exact_threshold_search"]["parent_threshold_fitted"])
        self.assertEqual(report["boundary_policy"]["selection_rule"], boundary.ORDER)
        self.assertTrue(all(x["prediction_type"] == "global_unknown" for x in boundary.decode_records(groups[2], state, META)))
        groups[0][0]["discovery"]["parent_scores"][0] += .001
        with self.assertRaisesRegex(ValueError, "unchanged D05"):
            boundary.fit_router(*groups, META, policy="fixed_parent", d05_records=reference)

    def test_crossfit_does_not_use_held_rows_for_either_reference_or_threshold_grid(self):
        groups = buffered_fixture(correct=51)
        plan = discovery._folds(sum(groups, []), {"seed": 1})
        folds = [f for f in plan if f["kind"] == "known"]
        low = set(folds[0]["held_image_sha256"][:4]+folds[1]["held_image_sha256"][:2])
        for item in groups[0]:
            if item["image_sha256"] in low: item["discovery"]["parent_scores"][0] = 1.
        groups[2][0]["discovery"]["parent_scores"][0] = 1.5
        d05 = d05_bundle(groups)
        calls, original_calls, grid_calls = [], [], []
        old_fit, old_reference, enumerate_ = discovery.fit_router, boundary.membership.calibrate, boundary._enumerate
        def fitted(*args, **kwargs):
            result = old_fit(*args, **kwargs); calls.append(set(result[0]["fit_image_sha256"])); return result
        def refit(*args, **kwargs):
            result = old_reference(*args, **kwargs); original_calls.append(set(result["fit_image_sha256"])); return result
        def grid(rows, *args, **kwargs):
            grid_calls.append({r["image_sha256"] for r in rows}); return enumerate_(rows, *args, **kwargs)
        with patch.object(discovery, "fit_router", side_effect=fitted), patch.object(boundary.membership, "calibrate", side_effect=refit), patch.object(boundary, "_enumerate", side_effect=grid):
            audit = boundary.crossfit_audit(*groups, META, policy="fixed_parent", reference_records=bundle(sum(groups, [])), d05_records=d05)
        self.assertTrue(audit["complete"])
        self.assertEqual(audit["evaluated_image_count"], 59)
        for i, fold in enumerate(audit["folds"]):
            fit, held = set(fold["fit_image_sha256"]), set(fold["held_image_sha256"])
            self.assertFalse(fit & held)
            for fitted_ids in (calls[2*i], calls[2*i+1], original_calls[i], grid_calls[i]): self.assertEqual(fit, fitted_ids)
            self.assertEqual(fold["global_parent_threshold"], fold["d05_parent_threshold"])
        self.assertTrue(any(f["global_parent_threshold"] != d05["router"]["global_parent_threshold"] for f in audit["folds"]))
        self.assertFalse(audit["output_used_for_threshold_selection"])
        self.assertFalse(audit["full_development_d05_threshold_used_in_fold"])

    def test_settings_reference_test_guard_and_inference_label_independence(self):
        groups = fixture(); d05 = d05_bundle(groups)
        for settings in ({"known_target": .94}, {"seed": True}, {"parent_grid": 10}):
            with self.assertRaises(ValueError): boundary.fit_router(*groups, META, settings, d05_records=d05)
        with self.assertRaises(ValueError): boundary.fit_router(*groups, META, policy="buffer92", d05_records=d05)
        state, report = boundary.fit_router(*groups, META, d05_records=d05)
        x, y = boundary.decode_records([groups[0][0], {"discovery": groups[0][0]["discovery"]}], state, META)
        self.assertEqual(x["prediction_type"], y["prediction_type"])
        self.assertEqual(x["local_knownness_score"], y["local_knownness_score"])
        self.assertFalse(report["known_recovery"]["replaces_four_gate_qualification"])
        groups[0][0]["split"] = "test_known"
        for function in (boundary.fit_router, boundary.crossfit_audit):
            with self.assertRaisesRegex(ValueError, "test fitting is prohibited"): function(*groups, META, d05_records=d05)
        self.assertEqual(boundary.crossfit_audit(*fixture(), META)["status"], "not_evaluable")


if __name__ == "__main__":
    unittest.main()
