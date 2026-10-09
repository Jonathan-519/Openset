"""Recovery policy isolation, exact empirical masks and conditional folds."""
import copy
import json
import unittest
from unittest.mock import patch

from taxosafe_discovery import calibration as discovery
from taxosafe_recovery import calibration as recovery
from tests.test_taxosafe_discovery_calibration import META, fixture, row, bundle


def d05_bundle(groups):
    router, _ = discovery.fit_router(*groups, META)
    return dict(records=copy.deepcopy(sum(groups, [])), router=router)


def buffered_fixture(correct=46):
    return ([row("k"+str(i), "known", ls=2. if i < correct else -1.) for i in range(51)],
            [row("n"+str(i), "intra", ls=-1., source="near"+str(i % 2)) for i in range(4)],
            [row("e"+str(i), "extra", ps=-2., ls=-1., source="extra"+str(i % 2)) for i in range(4)])


class RecoveryCalibrationTests(unittest.TestCase):
    def test_settings_are_prespecified_and_strict(self):
        self.assertEqual(recovery.validate_settings(), {"seed": 1, "known_target": .92})
        for settings in ({"known_target": .94}, {"known_target": True}, {"seed": True}, {"seed": -1}, {"grid": 49}):
            with self.assertRaises(ValueError): recovery.validate_settings(settings)
        with self.assertRaises(ValueError): recovery.fit_router(*fixture(), META, policy="unknown")

    def test_standard_router_is_exactly_old_policy_including_duplicate_metadata(self):
        groups = fixture()
        groups[0].append(copy.deepcopy(groups[0][0]))
        reference = bundle(sum(groups, []))
        old, old_report = discovery.fit_router(*groups, META, reference_records=reference)
        actual, report = recovery.fit_router(*groups, META, reference_records=reference, d05_records=d05_bundle(groups))
        self.assertEqual(old, actual)
        self.assertEqual(old_report["metrics"], report["metrics"])
        self.assertEqual(actual["input_record_count"], 15)
        self.assertEqual(actual["duplicate_record_count"], 1)
        self.assertEqual(report["input_record_count"], old_report["input_record_count"])
        self.assertFalse(report["recovery_passed"])  # no strict gain over itself

    def test_buffer_margin_infeasible_retains_safe_point_not_unsafe_coverage(self):
        groups = buffered_fixture()
        state, report = recovery.fit_router(*groups, META, policy="buffer92", d05_records=d05_bundle(groups))
        self.assertEqual(report["recovery_policy"]["status"], "margin_infeasible")
        self.assertEqual(report["counts"]["known_correct"], 46)
        self.assertEqual(report["counts"]["intra_correct"], 4)
        self.assertEqual(report["counts"]["extra_correct"], 4)
        search = report["exact_threshold_search"]
        self.assertEqual(search["known_required_for_margin"], 47)
        self.assertGreater(search["margin_candidates"], 0)
        self.assertEqual(search["protected_margin_candidates"], 0)
        self.assertTrue(report["targets_passed"])  # research margin and four gates are distinct
        self.assertFalse(state["baseline_fallback"])

    def test_buffer_coverage_can_improve_with_changed_scores(self):
        baseline, groups = buffered_fixture(), buffered_fixture(correct=47)
        state, report = recovery.fit_router(*groups, META, policy="buffer92", d05_records=d05_bundle(baseline))
        self.assertEqual(report["recovery_policy"]["status"], "coverage_satisfied")
        self.assertEqual(report["counts"]["known_correct"], 47)
        self.assertTrue(report["recovery_passed"])
        self.assertTrue(report["known_recovery"]["checks"]["known_strictly_improved"])
        self.assertEqual(state["fit_image_sha256"], sorted(r["image_sha256"] for r in sum(groups, [])))

    def test_empty_protection_still_returns_own_executable_best_effort(self):
        baseline = fixture(); groups = copy.deepcopy(baseline)
        for item in groups[0]: item["discovery"]["leaf_scores"][0] = -2.
        for item in groups[1]: item["discovery"]["leaf_scores"][0] = 2.
        state, report = recovery.fit_router(*groups, META, policy="buffer92", d05_records=d05_bundle(baseline))
        self.assertEqual(report["recovery_policy"]["status"], "protection_infeasible")
        self.assertEqual(report["exact_threshold_search"]["protected_candidates"], 0)
        self.assertFalse(state["baseline_fallback"])
        self.assertTrue(report["recovery_policy"]["test_allowed_after_failed_gates"])
        self.assertFalse(report["targets_passed"])
        scored = discovery.base.evaluate_records(recovery.decode_records(sum(groups, []), state, META), META)
        self.assertEqual(scored["counts"], report["counts"])

    def test_exact_buffer_feasibility_matches_independent_decoding(self):
        baseline, groups = buffered_fixture(), buffered_fixture(correct=47)
        state, report = recovery.fit_router(*groups, META, policy="buffer92", d05_records=d05_bundle(baseline))
        search = report["exact_threshold_search"]
        safe = margin = feasible = 0
        for pt in search["parent_grid"]:
            for lt in search["leaf_grid"]:
                candidate = copy.deepcopy(state)
                candidate.update(global_parent_threshold=pt, global_leaf_threshold=lt,
                                 parent_thresholds=[pt, pt], leaf_thresholds=[lt, lt])
                candidate["router_sha256"] = discovery._router_hash(candidate)
                result = discovery.base.evaluate_records(recovery.decode_records(sum(groups, []), candidate, META), META)
                counts = result["counts"]
                protected = counts["intra_correct"] >= 4 and counts["extra_correct"] >= 4 and result["checks"]["open_world_accepted_leaf_precision"]
                safe += int(protected)
                margin += int(protected and counts["known_correct"] >= 47)
                feasible += int(result["targets_passed"])
        self.assertEqual(safe, search["protected_candidates"])
        self.assertEqual(margin, search["protected_margin_candidates"])
        self.assertEqual(feasible, search["all_four_gate_candidates"])

    def test_fixed_parent_searches_only_leaf_and_never_promotes_original_roots(self):
        baseline = fixture(); groups = copy.deepcopy(baseline)
        for item in groups[2]: item["discovery"]["leaf_scores"][0] = 1000.
        d05 = d05_bundle(baseline)
        state, report = recovery.fit_router(*groups, META, policy="fixed_parent", d05_records=d05)
        self.assertEqual(state["global_parent_threshold"], d05["router"]["global_parent_threshold"])
        search = report["exact_threshold_search"]
        self.assertEqual(search["parent_grid"], [state["global_parent_threshold"]])
        self.assertFalse(search["parent_threshold_fitted"])
        self.assertTrue(search["includes_all_leaf_accept_and_reject"])
        self.assertTrue(all(r["prediction_type"] == "global_unknown" for r in recovery.decode_records(groups[2], state, META)))
        self.assertTrue(report["recovery_policy"]["root_outputs_preserved_on_fit"])
        groups[0][0]["discovery"]["parent_scores"][0] += .01
        with self.assertRaisesRegex(ValueError, "unchanged D05"):
            recovery.fit_router(*groups, META, policy="fixed_parent", d05_records=d05)

    def test_d05_bundle_is_bound_to_same_fit_images_and_annotations(self):
        groups = fixture(); d05 = d05_bundle(groups)
        for wrong in ({"records": d05["records"]}, dict(d05, records=d05["records"][:-1])):
            with self.assertRaises(ValueError): recovery.fit_router(*groups, META, policy="buffer92", d05_records=wrong)
        changed = copy.deepcopy(d05); changed["records"][0]["source"] = "other"
        with self.assertRaisesRegex(ValueError, "annotations differ"):
            recovery.fit_router(*groups, META, d05_records=changed)
        changed = copy.deepcopy(d05); changed["router"]["fit_image_sha256"].pop()
        with self.assertRaisesRegex(ValueError, "identities differ"):
            recovery.fit_router(*groups, META, d05_records=changed)
        changed = copy.deepcopy(d05); changed["records"][0]["discovery"]["leaf_scores"][0] += .01
        with self.assertRaisesRegex(ValueError, "evidence differs"):
            recovery.fit_router(*groups, META, d05_records=changed)

    def test_known_recovery_is_strictly_improved_not_just_equal(self):
        baseline = fixture(); d05 = d05_bundle(baseline)
        _, report = recovery.fit_router(*baseline, META, d05_records=d05)
        self.assertTrue(report["targets_passed"])
        self.assertFalse(report["recovery_passed"])
        self.assertFalse(report["known_recovery"]["checks"]["known_strictly_improved"])
        self.assertFalse(report["known_recovery"]["replaces_four_gate_qualification"])

    def test_crossfit_refits_both_references_without_held_records_and_fixed_floor_changes(self):
        groups = buffered_fixture(correct=51)
        plan = discovery._folds(sum(groups, []), {"seed": 1})
        known_folds = [f for f in plan if f["kind"] == "known"]
        low = set(known_folds[0]["held_image_sha256"][:4] + known_folds[1]["held_image_sha256"][:2])
        for item in groups[0]:
            if item["image_sha256"] in low: item["discovery"]["parent_scores"][0] = 1.
        groups[2][0]["discovery"]["parent_scores"][0] = 1.5
        d05 = d05_bundle(groups); calls = []; originals = []
        old_fit, old_reference = discovery.fit_router, recovery.membership.calibrate
        def fitted(*args, **kwargs):
            value = old_fit(*args, **kwargs); calls.append(value[0]); return value
        def refit(*args, **kwargs):
            value = old_reference(*args, **kwargs); originals.append(value); return value
        with patch.object(discovery, "fit_router", side_effect=fitted), patch.object(recovery.membership, "calibrate", side_effect=refit):
            audit = recovery.crossfit_audit(*groups, META, policy="fixed_parent", reference_records=bundle(sum(groups, [])), d05_records=d05)
        self.assertTrue(audit["complete"])
        self.assertEqual(audit["evaluated_image_count"], 59)
        self.assertEqual(len(calls), 2*len(audit["folds"]))
        self.assertFalse(audit["full_development_d05_threshold_used_in_fold"])
        for i, fold in enumerate(audit["folds"]):
            fit, held = set(fold["fit_image_sha256"]), set(fold["held_image_sha256"])
            self.assertFalse(fit & held)
            for state in (calls[2*i], calls[2*i+1], originals[i]):
                self.assertEqual(set(state["fit_image_sha256"]), fit)
            self.assertEqual(fold["global_parent_threshold"], fold["d05_parent_threshold"])
        self.assertTrue(any(f["global_parent_threshold"] != d05["router"]["global_parent_threshold"] for f in audit["folds"]))
        self.assertFalse(audit["recovery_passed"])  # same evidence is not a new recovery
        json.dumps(audit, allow_nan=False)

    def test_test_fitting_is_forbidden_and_inference_ignores_labels(self):
        groups = fixture(); d05 = d05_bundle(groups)
        state, _ = recovery.fit_router(*groups, META, policy="buffer92", d05_records=d05)
        labelled = groups[0][0]; unlabelled = {"discovery": labelled["discovery"]}
        a, b = recovery.decode_records([labelled, unlabelled], state, META)
        for key in ("prediction_type", "leaf", "parent", "root_knownness_score"):
            self.assertEqual(a[key], b[key])
        groups[0][0]["split"] = "test_known"
        for function in (recovery.fit_router, recovery.crossfit_audit):
            with self.assertRaisesRegex(ValueError, "test fitting is prohibited"):
                function(*groups, META, d05_records=d05)
        self.assertEqual(recovery.crossfit_audit(*fixture(), META)["status"], "not_evaluable")


if __name__ == "__main__":
    unittest.main()
