"""Local corrections: image protection, label-free inference and source isolation."""
import copy
import unittest
from unittest.mock import patch

from taxosafe_geometry import local
from taxosafe_support import calibration as base
from taxosafe_support import membership_calibration as membership
from tests.test_taxosafe_geometry_calibration import META, BASELINE, row

OPTIONS = dict(decoder="local_guarded", min_known_per_parent=1, max_thresholds=31,
               source_loo=False, source_loo_safeguard=False)


def fixture():
    return ([row("k", "known", pm=3., lm=3., gp=3., gl=3.)],
            [row("n", "intra", lm=.2, gl=-2.)],
            [row("e", "extra", pm=.2, gp=-2.)])


def routed(groups, options=None):
    router = local.calibrate(*groups, BASELINE, META, options or OPTIONS)
    return router, local.apply_router(sum(groups, []), router, META)


class LocalGeometryTests(unittest.TestCase):
    def test_local_corrections_improve_both_unknown_routes_and_preserve_known(self):
        router, predictions = routed(fixture())
        self.assertTrue(router["geometry_enabled"])
        self.assertTrue(router["targets_passed"])
        self.assertTrue(router["preservation_audit"]["passed"])
        self.assertEqual([r["prediction_type"] for r in predictions], ["known", "intra_unknown", "global_unknown"])
        self.assertEqual(router["validation_report"]["counts"]["known_correct"], 1)
        self.assertEqual(predictions[0]["applied_rule_indices"], [])

    def test_parent_conditional_rule_avoids_global_known_outlier(self):
        known, near, extra = fixture()
        weak = row("weak-known-q", "known", parent=1, leaf=2, gp=-20., gl=-20.)
        weak["support_evidence"].update(parent_logits=[1., 2.], parent_membership_logits=[1., 1.],
                                        leaf_membership_logits=[1., 1., 1.])
        weak["baseline_parent_z"] = weak["baseline_leaf_z"] = -20.
        known.append(weak)
        router, predictions = routed((known, near, extra))
        self.assertTrue(router["preservation_audit"]["passed"])
        self.assertEqual([p["prediction_type"] for p in predictions[:2]], ["known", "known"])
        self.assertTrue(any(rule["parent"] == 0 for rule in router["rules"]))
        self.assertTrue(router["targets_passed"])

    def test_no_leaf_promotion_or_replacement_even_for_strong_geometry(self):
        groups = fixture()
        router, _ = routed(groups)
        records = [row("new-known-root", "known", pm=-1., gp=100., gl=100.),
                   row("new-known-parent", "known", lm=-1., gp=100., gl=100.)]
        before = base.apply_router(records, BASELINE, META)
        after = local.apply_router(records, router, META)
        for old, new in zip(before, after):
            self.assertNotEqual(new["prediction_type"], "known")
            self.assertEqual(new["candidate_leaf"], old["candidate_leaf"])

    def test_fallback_is_exact_for_unseen_scores_and_tied_boundaries(self):
        groups = ([row("k", "known")], [row("n", "intra", lm=-1.)], [row("e", "extra", pm=-1.)])
        router, _ = routed(groups)
        self.assertFalse(router["geometry_enabled"])
        records = sum(groups, []) + [row("tie", "extra", pm=0., lm=0., gp=-1e90, gl=1e90)]
        for old, new in zip(base.apply_router(records, BASELINE, META), local.apply_router(records, router, META)):
            for key in ("prediction_type", "parent", "leaf", "output_node", "candidate_parent", "candidate_leaf"):
                self.assertEqual(old[key], new[key])

    def test_parent_repair_only_changes_nonleaf_outputs_with_two_head_support(self):
        selection = dict(geometry_enabled=True, rules=[dict(action="parent_repair", parent=-1, x_max=2., y_max=0.)])
        router = local._router(selection, BASELINE, META)
        rejected = row("root", "intra", pm=-1., parent=1)
        accepted = row("leaf", "known")
        unsupported = copy.deepcopy(rejected)
        unsupported["image_sha256"] = "f" * 64
        unsupported["support_evidence"]["parent_membership_logits"][1] = -1.
        predictions = local.apply_router([rejected, accepted, unsupported], router, META)
        self.assertEqual((predictions[0]["prediction_type"], predictions[0]["parent"]), ("intra_unknown", 1))
        self.assertEqual(predictions[0]["candidate_parent"], 0)
        self.assertEqual(predictions[0]["candidate_leaf"], 0)
        self.assertEqual(predictions[1]["prediction_type"], "known")
        self.assertEqual(predictions[1]["leaf"], 0)
        self.assertEqual(predictions[2]["prediction_type"], "global_unknown")

    def test_prediction_does_not_depend_on_test_labels_or_source_names(self):
        router, _ = routed(fixture())
        rows = sum(fixture(), [])
        expected = local.apply_router(rows, router, META)
        for r in rows:
            r.update(source="unseen", split="test_" + r["status"])
            if r["status"] != "extra":
                r["true_parent"] = 1
                r["true_leaf"] = 2 if r["status"] == "known" else None
        actual = local.apply_router(rows, router, META)
        self.assertEqual([(r["prediction_type"], r["output_node"]) for r in expected],
                         [(r["prediction_type"], r["output_node"]) for r in actual])

    def test_test_fitting_empty_splits_and_inconsistent_aliases_rejected(self):
        for index in range(3):
            groups = list(fixture())
            groups[index][0]["split"] = "test_" + groups[index][0]["status"]
            with self.assertRaisesRegex(ValueError, "test fitting is prohibited"):
                routed(groups)
            groups[index] = []
            with self.assertRaisesRegex(ValueError, "nonempty"):
                routed(groups)
        groups = fixture()
        alias = copy.deepcopy(groups[0][0])
        alias["geometry_leaf_score"] += 1.
        groups[0].append(alias)
        with self.assertRaisesRegex(ValueError, "inconsistent geometry evidence"):
            routed(groups)

    def test_unique_image_fit_and_determinism_under_reordering(self):
        groups = fixture()
        original, _ = routed(groups)
        groups[0].append(copy.deepcopy(groups[0][0]))
        duplicate, predicted = routed(groups)
        self.assertEqual(duplicate["unique_image_count"], 3)
        self.assertEqual(duplicate["duplicate_record_count"], 1)
        self.assertEqual(len(predicted), 4)
        self.assertEqual(original["rules"], duplicate["rules"])
        self.assertEqual(original["evidence_sha256"], duplicate["evidence_sha256"])
        self.assertEqual(original, routed(fixture())[0])

    def test_rule_and_configuration_validation(self):
        router, _ = routed(fixture())
        for edit in (dict(geometry_enabled=1), dict(geometry_enabled=False), dict(decoder="other"),
                     dict(meta={}), dict(baseline_router=dict(BASELINE, leaf_threshold=1.))):
            with self.subTest(edit=edit), self.assertRaises(ValueError):
                local.apply_router(sum(fixture(), []), dict(router, **edit), META)
        for value in (True, None, [], float("nan"), float("inf")):
            changed = copy.deepcopy(router)
            changed["rules"][0]["x_max"] = value
            with self.assertRaises(ValueError):
                local.apply_router(sum(fixture(), []), changed, META)
        for edit in (dict(min_known_per_parent=True), dict(max_thresholds=0), dict(actions=[]),
                     dict(actions=["test_tuning"]), dict(actions=["leaf_reject"] * 2),
                     dict(source_loo="yes"), dict(test_tuning=True), dict(source_loo_safeguard=True)):
            with self.subTest(edit=edit), self.assertRaises(ValueError):
                local.settings(dict(OPTIONS, **edit))

    def test_source_folds_refit_baseline_and_rules_without_held_source(self):
        known, near, extra = fixture()
        near.append(row("n2", "intra", lm=.1, gl=-2., source="near-b"))
        extra.append(row("e2", "extra", pm=.1, gp=-2., source="extra-b"))
        options = dict(OPTIONS, source_loo=True, source_loo_safeguard=True,
                       baseline_calibration=dict(decoder="membership", membership_grid_points=3))
        called = []
        original = membership.calibrate
        def spy(k, n, e, meta, config):
            called.append(k + n + e)
            return original(k, n, e, meta, config)
        with patch.object(membership, "calibrate", side_effect=spy):
            router = local.calibrate(known, near, extra, BASELINE, META, options)
        self.assertEqual(len(called), 4)  # Shared folds, no repeated baseline fitting per action.
        for audit in list(router["action_source_loo"].values()) + [router["source_loo"]]:
            for fold, fit in zip(audit["folds"], called):
                self.assertFalse(any(r["status"] == fold["status"] and r["source"] == fold["held_source"] for r in fit))
                self.assertEqual(set(fold["fit_image_sha256"]), {r["image_sha256"] for r in fit})
                self.assertTrue(set(fold["fit_image_sha256"]).isdisjoint(fold["held_image_sha256"]))
                self.assertTrue(fold["baseline_refit_on_fit_sources_only"])
                self.assertTrue(fold["rules_use_fit_sources_only"])

    def test_unstable_action_is_disabled_without_disabling_stable_actions(self):
        options = dict(OPTIONS, source_loo=True, source_loo_safeguard=True,
                       baseline_calibration=dict(decoder="membership", membership_grid_points=3))
        def report(folds, skipped, meta, config, actions):
            return dict(available=True, folds=[], safeguard=dict(passed=not any(a in actions for a in ("root_rescue", "parent_repair"))))
        with patch.object(local, "_source_report", side_effect=report):
            router, _ = routed(fixture(), options)
        self.assertEqual(router["enabled_actions"], ["leaf_reject", "root_reject"])
        self.assertEqual(router["rejected_actions"], ["root_rescue", "parent_repair"])
        self.assertTrue(router["geometry_enabled"])
        self.assertTrue(router["targets_passed"])

    def test_combination_gets_its_own_safeguard_and_exact_fallback(self):
        options = dict(OPTIONS, source_loo=True, source_loo_safeguard=True,
                       baseline_calibration=dict(decoder="membership", membership_grid_points=3))
        def report(folds, skipped, meta, config, actions):
            return dict(available=True, folds=[], safeguard=dict(passed=len(actions) == 1))
        with patch.object(local, "_source_report", side_effect=report):
            router, predictions = routed(fixture(), options)
        self.assertFalse(router["geometry_enabled"])
        self.assertTrue(router["source_loo_safeguard_rejected"])
        old = base.apply_router(sum(fixture(), []), BASELINE, META)
        self.assertEqual([x["output_node"] for x in predictions], [x["output_node"] for x in old])

    def test_source_safeguard_needs_two_real_gains_and_rejects_any_regression(self):
        known, near, extra = fixture()
        options = local.settings(dict(OPTIONS, source_loo=True,
                                     baseline_calibration=dict(decoder="membership", membership_grid_points=3)))
        folds, skipped = local._folds(known + near + extra, META, options)
        report = local._source_report(folds, skipped, META, options, ["leaf_reject"])
        self.assertFalse(report["safeguard"]["passed"])
        self.assertEqual(report["safeguard"]["improved_source_count"], 0)
        self.assertEqual(set(skipped), {"intra", "extra"})

    def test_component_auc_handles_ties_and_deduplicates(self):
        rows = sum(fixture(), [])
        rows[1]["geometry_leaf_score"] = rows[0]["geometry_leaf_score"]
        report = local.score_diagnostics(rows + [copy.deepcopy(rows[0])])
        self.assertEqual(report["unique_images"], 3)
        self.assertEqual(report["components"]["geometry_leaf_score"]["auroc"], .5)
        self.assertEqual(report["components"]["geometry_parent_score"]["auroc"], 1.)
        self.assertEqual(report["components"]["geometry_leaf_score"]["positive_count"], 1)


if __name__ == "__main__":
    unittest.main()
