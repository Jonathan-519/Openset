"""Hierarchical evidence fusion: protection, source isolation and fallback."""
import copy
import hashlib
import unittest
from unittest.mock import patch

import numpy as np

from taxosafe_geometry import calibration as geometry
from taxosafe_support import calibration as base
from taxosafe_support import membership_calibration as membership


META = {"parent_names": ["p", "q"], "leaf_names": ["a", "b", "c"], "leaf_to_parent": [0, 0, 1]}
BASELINE = {"schema_version": "support_membership_v1", "decoder": "membership", "meta": META,
            "parent_threshold": 0., "leaf_threshold": 0.}
SETTINGS = {"source_loo": False, "source_loo_safeguard": False, "grid_points": 3}


def row(identity, status, pm=1., lm=1., gp=1., gl=1., source=None, parent=0, leaf=0):
    return {"image_sha256": hashlib.sha256(identity.encode()).hexdigest(),
            "split": {"known": "val_known", "intra": "val_intra", "extra": "val_extra"}[status],
            "status": status, "source": source or status,
            "true_parent": None if status == "extra" else parent,
            "true_leaf": leaf if status == "known" else None,
            "log_probs": [-float(np.log(6))] * 6,
            "support_evidence": {"parent_logits": [2., 1.], "leaf_logits": [2., 1., 10.],
                                 "parent_membership_logits": [pm, 100.],
                                 "leaf_membership_logits": [lm, 100., 100.]},
            "geometry_parent_score": gp, "geometry_leaf_score": gl,
            "baseline_parent_z": pm, "baseline_leaf_z": lm}


def fixture():
    return ([row("k", "known")],
            [row("n", "intra", pm=-1., gp=1., gl=-1.)],
            [row("e", "extra", pm=1., gp=-1., gl=-1.)])


class GeometryCalibrationTests(unittest.TestCase):
    def test_improves_root_and_near_without_changing_ranking_candidates(self):
        groups = fixture()
        router = geometry.calibrate(*groups, BASELINE, META, SETTINGS)
        self.assertEqual(router["status"], "feasible_geometry")
        self.assertTrue(router["targets_passed"])
        self.assertTrue(router["geometry_enabled"])
        self.assertTrue(router["preservation_audit"]["passed"])
        self.assertEqual(router["validation_report"]["counts"], dict(known=1, intra=1, extra=1,
                         known_correct=1, intra_correct=1, extra_correct=1, leaf_outputs=1))
        before = base.apply_router(sum(groups, []), BASELINE, META)
        after = geometry.apply_router(sum(groups, []), router, META)
        for original, predicted in zip(before, after):
            for field in ("candidate_parent", "candidate_leaf", "support_evidence", "log_probs",
                          "parent_membership_score", "leaf_membership_score"):
                self.assertEqual(original[field], predicted[field])
            self.assertEqual(predicted["root_threshold"], 0.)
            self.assertEqual(predicted["local_threshold"], 0.)
        ceiling = router["validation_report"]["parent_routing_near_upper_bound"]
        self.assertEqual(ceiling["baseline_correct_count"], 0)
        self.assertEqual(ceiling["selected_correct_count"], 1)
        self.assertEqual(ceiling["ranking_candidate_correct_count"], 1)

    def test_exact_baseline_fallback_even_for_unseen_geometry_scores(self):
        groups = ([row("k", "known")], [row("n", "intra", lm=-1., gl=-1.)],
                  [row("e", "extra", pm=-1., gp=-1.)])
        router = geometry.calibrate(*groups, BASELINE, META, SETTINGS)
        self.assertEqual(router["status"], "baseline_fallback")
        self.assertTrue(router["targets_passed"])
        self.assertFalse(router["geometry_enabled"])
        records = sum(groups, []) + [row("unseen", "extra", pm=0., lm=0.)]
        for i, record in enumerate(records):
            record["geometry_parent_score"] = 1.e100 if i % 2 else -1.e100
            record["geometry_leaf_score"] = -record["geometry_parent_score"]
            record["split"] = "test_" + record["status"]
        before = base.apply_router(records, BASELINE, META)
        after = geometry.apply_router(records, router, META)
        for original, predicted in zip(before, after):
            for field in ("prediction_type", "parent", "leaf", "output_node", "candidate_parent", "candidate_leaf"):
                self.assertEqual(original[field], predicted[field])
            self.assertEqual(predicted["root_knownness_score"] >= 0., original["parent_membership_score"] >= BASELINE["parent_threshold"])
            self.assertEqual(predicted["local_knownness_score"] >= 0., original["leaf_membership_score"] >= BASELINE["leaf_threshold"])

    def test_known_image_protection_prevents_aggregate_swap(self):
        known = [row("k%d" % i, "known") for i in range(10)]
        known.append(row("recover", "known", lm=-1.))
        near, extra = fixture()[1:]
        records = known + near + extra
        data = geometry._arrays(records, META)
        bp, bl = geometry._baseline_masks(data, BASELINE)
        after_leaf = bl.copy()
        after_leaf[0], after_leaf[10] = False, True
        audit = geometry._audit(records, data, bp, bl, bp, after_leaf, META)
        self.assertEqual(audit["baseline_counts"]["known_correct"], audit["selected_counts"]["known_correct"])
        self.assertTrue(audit["checks"]["every_known_leaf_count_preserved"])
        self.assertFalse(audit["checks"]["every_baseline_correct_known_preserved"])
        self.assertFalse(audit["passed"])
        self.assertEqual(audit["lost_baseline_correct_known_sha256"], [known[0]["image_sha256"]])

    def test_unknown_source_preservation_prevents_aggregate_exchange(self):
        records = [row("k", "known"), row("n1", "intra", lm=-1., source="source-a"),
                   row("n2", "intra", source="source-b"), row("e", "extra", pm=-1.)]
        data = geometry._arrays(records, META)
        bp, bl = geometry._baseline_masks(data, BASELINE)
        leaves = bl.copy()
        leaves[1], leaves[2] = True, False
        audit = geometry._audit(records, data, bp, bl, bp, leaves, META)
        self.assertEqual(audit["baseline_counts"]["intra_correct"], audit["selected_counts"]["intra_correct"])
        self.assertFalse(audit["checks"]["every_unknown_source_count_preserved"])
        self.assertFalse(audit["passed"])
        failed = [x for x in audit["per_unknown_source"] if not x["passed"]]
        self.assertEqual(failed[0]["source"], "source-a")

    def test_no_test_fitting_and_empty_groups_are_rejected(self):
        settings = dict(SETTINGS, baseline_calibration={"decoder": "membership", "membership_grid_points": 3})
        for function in (geometry.calibrate, geometry.source_loo):
            for index in range(3):
                groups = list(fixture())
                groups[index][0]["split"] = "test_" + groups[index][0]["status"]
                with self.assertRaisesRegex(ValueError, "test fitting is prohibited"):
                    function(*groups, BASELINE, META, settings)
                groups = list(fixture())
                groups[index] = []
                with self.assertRaisesRegex(ValueError, "nonempty"):
                    function(*groups, BASELINE, META, settings)

    def test_finite_scalar_evidence_and_duplicate_checks_before_dedup(self):
        for name in geometry.EVIDENCE_FIELDS:
            for invalid in (None, True, [1.], float("nan"), float("inf")):
                groups = fixture()
                groups[0][0][name] = invalid
                with self.subTest(name=name, value=invalid), self.assertRaisesRegex(ValueError, "finite scalar"):
                    geometry.calibrate(*groups, BASELINE, META, SETTINGS)
            known, near, extra = fixture()
            alias = copy.deepcopy(known[0])
            alias["path"] = "same-content-alias.png"
            router = geometry.calibrate(known + [alias], near, extra, BASELINE, META, SETTINGS)
            self.assertEqual(router["unique_image_count"], 3)
            self.assertEqual(router["duplicate_record_count"], 1)
            alias[name] += .1
            with self.assertRaisesRegex(ValueError, "inconsistent geometry evidence"):
                geometry.calibrate(known + [alias], near, extra, BASELINE, META, SETTINGS)

    def test_inference_preserves_aliases_and_evaluation_deduplicates(self):
        known, near, extra = fixture()
        router = geometry.calibrate(known, near, extra, BASELINE, META, SETTINGS)
        records = known + copy.deepcopy(known) + near + extra
        predictions = geometry.apply_router(records, router, META)
        self.assertEqual(len(predictions), 4)
        report = geometry.evaluate_records(predictions, META)
        self.assertEqual(report["unique_image_count"], 3)
        self.assertEqual(report["duplicate_record_count"], 1)

    def test_evidence_identity_and_deterministic_selection(self):
        groups = fixture()
        original = geometry.calibrate(*groups, BASELINE, META, SETTINGS)
        again = geometry.calibrate(*groups, BASELINE, META, SETTINGS)
        self.assertEqual(original, again)
        groups[2][0]["geometry_leaf_score"] += .01
        changed = geometry.calibrate(*groups, BASELINE, META, SETTINGS)
        self.assertNotEqual(original["evidence_sha256"], changed["evidence_sha256"])
        self.assertNotEqual(original["calibration_sha256"], changed["calibration_sha256"])

    def test_invalid_router_settings_and_hierarchy_fail_closed(self):
        router = geometry.calibrate(*fixture(), BASELINE, META, SETTINGS)
        for changed in (dict(router, schema_version="other"), dict(router, meta={}),
                        dict(router, geometry_enabled=1), dict(router, parent_threshold=float("nan")),
                        dict(router, parent_weight=-.1), dict(router, parent_weight=0., leaf_weight=0.),
                        dict(router, baseline_router=dict(BASELINE, leaf_threshold=1.))):
            with self.subTest(state=changed), self.assertRaises(ValueError):
                geometry.apply_router(sum(fixture(), []), changed, META)
        for change in ({"weights": [0.]}, {"weights": [True]}, {"weights": [.5, float("inf")]},
                       {"grid_points": True}, {"grid_points": 1}, {"source_loo": 1},
                       {"threshold_grid": "uniform"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                geometry.calibrate(*fixture(), BASELINE, META, dict(SETTINGS, **change))
        for meta in (dict(META, leaf_to_parent=[0., 0., 1.]), dict(META, leaf_to_parent=[True, 0, 1]),
                     dict(META, parent_names=["p", "p"])):
            with self.assertRaises(ValueError):
                geometry.calibrate(*fixture(), BASELINE, meta, SETTINGS)

    def test_zero_weight_candidate_excluded_and_grid_endpoints_finite(self):
        router = geometry.calibrate(*fixture(), BASELINE, META, SETTINGS)
        self.assertTrue(router["grid"]["all_zero_weights_excluded"])
        for grid in router["grid"]["weight_threshold_grids"]:
            self.assertTrue(grid["parent_weight"] or grid["leaf_weight"])
            self.assertTrue(np.isfinite(grid["parent_threshold"]).all())
            self.assertTrue(np.isfinite(grid["leaf_threshold"]).all())
        extreme = fixture()
        extreme[0][0]["geometry_parent_score"] = np.finfo(np.float64).max
        with self.assertRaisesRegex(ValueError, "too extreme"):
            geometry.calibrate(*extreme, BASELINE, META, SETTINGS)

    def test_source_loo_refits_baseline_weights_and_thresholds_on_fit_only(self):
        known, near, extra = fixture()
        near.append(row("n2", "intra", pm=10., lm=-1., gp=10., gl=-10., source="near2"))
        extra.append(row("e2", "extra", pm=-10., gp=-10., source="extra2"))
        settings = dict(SETTINGS, baseline_calibration={"decoder": "membership", "policy": "known_first", "membership_grid_points": 3})
        base_calls, geometry_calls = [], []
        fit_baseline, fit_geometry = membership.calibrate, geometry._select
        def baseline_spy(known, near, extra, meta, options):
            fitted = fit_baseline(known, near, extra, meta, options)
            base_calls.append((known + near + extra, dict(options), fitted))
            return fitted
        def geometry_spy(rows, baseline, meta, options):
            selected = fit_geometry(rows, baseline, meta, options)
            geometry_calls.append((list(rows), baseline, dict(options), selected))
            return selected
        with patch.object(membership, "calibrate", side_effect=baseline_spy), patch.object(geometry, "_select", side_effect=geometry_spy):
            report = geometry.source_loo(known, near, extra, dict(BASELINE, parent_threshold=999.), META, settings)
        self.assertEqual(len(report["folds"]), 4)
        for fold, (records, options, fitted), (geo_records, geo_base, geo_options, selected) in zip(report["folds"], base_calls, geometry_calls):
            held = lambda r: r["status"] == fold["status"] and r["source"] == fold["held_source"]
            self.assertFalse(any(held(r) for r in records))
            self.assertEqual(records, geo_records)
            self.assertIs(fitted, geo_base)
            self.assertFalse(options["source_loo"])
            self.assertEqual(geo_options["parent_weights"], [0., .5, 1.])
            self.assertEqual(geo_options["leaf_weights"], [0., .5, 1.])
            self.assertEqual(fold["reselected_weights"], [selected["parent_weight"], selected["leaf_weight"]])
            self.assertTrue(fold["weights_use_fit_sources_only"])
            self.assertTrue(fold["grid_uses_fit_sources_only"])
            self.assertNotEqual(fold["baseline_parent_threshold"], 999.)
            self.assertTrue(set(fold["fit_image_sha256"]).isdisjoint(fold["held_image_sha256"]))

    def test_source_safeguard_falls_back_when_audit_missing_or_unstable(self):
        full = geometry.calibrate(*fixture(), BASELINE, META, SETTINGS)
        self.assertTrue(full["geometry_enabled"])
        disabled = geometry.calibrate(*fixture(), BASELINE, META, dict(SETTINGS, source_loo_safeguard=True))
        self.assertFalse(disabled["geometry_enabled"])
        self.assertEqual(disabled["status"], "baseline_fallback_source_instability")
        self.assertTrue(disabled["provisional_selection"]["geometry_enabled"])
        settings = dict(SETTINGS, source_loo=True, source_loo_safeguard=True,
                        baseline_calibration={"decoder": "membership", "membership_grid_points": 3})
        insufficient = geometry.calibrate(*fixture(), BASELINE, META, settings)
        self.assertFalse(insufficient["geometry_enabled"])
        self.assertEqual(len(insufficient["source_loo"]["skipped"]), 2)
        self.assertFalse(insufficient["source_loo"]["safeguard"]["passed"])
        unstable = {"available": True, "folds": [], "safeguard": {"passed": False,
                    "checks": {"every_held_source_correct_count_preserved": False}}}
        with patch.object(geometry, "source_loo", return_value=unstable):
            guarded = geometry.calibrate(*fixture(), BASELINE, META, settings)
        self.assertFalse(guarded["geometry_enabled"])
        self.assertEqual(guarded["validation_report"]["counts"], guarded["baseline_validation_report"]["counts"])
        self.assertTrue(guarded["source_loo_safeguard_rejected"])

    def test_source_loo_needs_actual_baseline_configuration(self):
        with self.assertRaisesRegex(ValueError, "baseline_calibration"):
            geometry.source_loo(*fixture(), BASELINE, META, SETTINGS)

    def test_source_safeguard_needs_two_real_held_source_gains(self):
        folds = [dict(status=status, held_source=status + str(index), geometry_selected=False,
                      baseline_held_metrics={"correct_count": 1}, held_metrics={"correct_count": 1},
                      held_source_correct_count_preserved=True)
                 for status in ("intra", "extra") for index in range(2)]
        pooled = {status: dict(total=4, baseline_correct=2, selected_correct=2) for status in ("intra", "extra")}
        audit = geometry._source_safeguard(folds, [], pooled)
        self.assertFalse(audit["passed"])
        self.assertFalse(audit["checks"]["at_least_one_fold_selected_geometry"])
        self.assertEqual(audit["improved_source_count"], 0)
        folds[0].update(geometry_selected=True, held_metrics={"correct_count": 2})
        pooled["intra"]["selected_correct"] += 1
        audit = geometry._source_safeguard(folds, [], pooled)
        self.assertFalse(audit["passed"])
        self.assertEqual(audit["improved_source_count"], 1)
        self.assertTrue(audit["checks"]["at_least_one_fold_selected_geometry"])
        folds[2].update(geometry_selected=True, held_metrics={"correct_count": 2})
        pooled["extra"]["selected_correct"] += 1
        audit = geometry._source_safeguard(folds, [], pooled)
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["improved_source_count"], 2)
        self.assertEqual(audit["improved_sources"], [{"status": "extra", "source": "extra0"},
                                                   {"status": "intra", "source": "intra0"}])
        folds[3]["held_source_correct_count_preserved"] = False
        self.assertFalse(geometry._source_safeguard(folds, [], pooled)["passed"])


if __name__ == "__main__":
    unittest.main()
