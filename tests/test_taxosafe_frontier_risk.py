"""Independent regressions for exact protected rejection and nested auditing."""
import copy
import json
import unittest
from unittest.mock import patch

import numpy as np

from taxosafe_frontier import calibration as frontier
from taxosafe_frontier.decoder import apply_router, make_router
from taxosafe_parentrisk import calibration as sampled
from taxosafe_support import calibration as base
from tests.test_taxosafe_geometry_calibration import META, BASELINE
from tests.test_taxosafe_parentrisk_risk import row


FIELDS = ("prediction_type", "parent", "leaf", "output_node", "candidate_parent", "candidate_leaf")


def observations():
    known = [row("k" + str(i), "known", pm=3., lm=3., gp=3., gl=3.) for i in range(12)]
    near = [row("n" + str(i), "intra", source="near" + str(i), pm=2., lm=.1, gp=2., gl=-2.) for i in range(4)]
    extra = [row("e" + str(i), "extra", source="extra" + str(i), pm=.1, lm=2., gp=-2., gl=2.) for i in range(4)]
    return known, near, extra


def opts(**values):
    return frontier.settings(dict(outer_folds=4, inner_folds=3, min_rule_sources=2,
        baseline_calibration=dict(decoder="membership", membership_grid_points=3, source_loo=False), **values))


class FrontierRiskTests(unittest.TestCase):
    def test_exact_frontier_finds_safe_rectangle_missed_by_sampled_grid(self):
        low = row("protected-low-x", "known", lm=3., gl=100.)
        low["baseline_leaf_z"] = 0.
        high = row("protected-high-x", "known", lm=3., gl=-100.)
        high["baseline_leaf_z"] = 10.
        near = [row("n1", "intra", source="one", lm=1., gl=1.),
                row("n2", "intra", source="two", lm=2., gl=2.)]
        records = [low, high] + near + [row("root", "extra", pm=-1.)]
        old = sampled._fit_family(records, BASELINE, META,
            sampled.settings(dict(mode="audit", grid_points=2)), None)
        self.assertFalse(any(r["action"] == "leaf_reject" for r in old["reject_rules"]))
        new = frontier._fit(records, BASELINE, META, opts(), ("leaf_reject",))
        self.assertEqual([r["action"] for r in new["reject_rules"]], ["leaf_reject"])
        predicted = apply_router(records, new, META)
        self.assertEqual([r["prediction_type"] for r in predicted[:4]], ["known", "known", "intra_unknown", "intra_unknown"])

    def test_frontier_tied_protected_x_and_inclusive_y_boundary_are_safe(self):
        x = np.array([1., 1., 2., 0.])
        y = np.array([1., .5, 0., 2.])
        scope = np.ones(4, dtype=bool)
        protected = np.array([True, False, True, False])
        candidates, _ = frontier._frontier_masks(x, y, scope, protected)
        self.assertTrue(candidates)
        for xt, yt, mask in candidates:
            expected = scope & (x <= xt) & (y <= yt)
            self.assertTrue(np.array_equal(mask, expected))
            self.assertFalse(bool((mask & protected).any()))
            self.assertTrue(np.isfinite([xt, yt]).all())
            self.assertEqual(xt, float(x[mask].max()))
            self.assertEqual(yt, float(y[mask].max()))
        self.assertTrue(any(mask[1] for _, _, mask in candidates))

    def test_frontier_covers_every_safe_observed_rectangle_by_a_safe_superset(self):
        x = np.array([0., 1., 1., 2., 3., 4., 4.])
        y = np.array([3., 0., 2., 1., 4., 0., 3.])
        scope = np.array([True] * 6 + [False])
        protected = np.array([True, False, True, False, False, True, True])
        candidates, _ = frontier._frontier_masks(x, y, scope, protected)
        for xt in np.unique(x[scope]):
            for yt in np.unique(y[scope]):
                rectangle = scope & (x <= xt) & (y <= yt)
                if rectangle.any() and not (rectangle & protected).any():
                    self.assertTrue(any(np.all(~rectangle | mask) for _, _, mask in candidates), (xt, yt))

    def test_known_and_near_parent_path_protection_cannot_be_bought_by_precision(self):
        records = [row("known", "known", pm=.1, lm=3., gp=-3., gl=3.),
                   row("near", "intra", pm=.1, lm=.1, gp=-3., gl=3.),
                   row("far-one", "extra", source="one", pm=.1, gp=-3.),
                   row("far-two", "extra", source="two", pm=.1, gp=-3.)]
        router = frontier._fit(records, BASELINE, META, opts(), ("root_reject",))
        prediction = apply_router(records, router, META)
        self.assertEqual(prediction[0]["prediction_type"], "known")
        self.assertNotEqual(prediction[1]["prediction_type"], "global_unknown")
        self.assertEqual(router["reject_rules"], [])

    def test_components_cannot_pool_one_source_each_to_pass_two_source_gate(self):
        known = [row("known", "known", pm=3., lm=3., gp=3., gl=3.)]
        near = [row("near", "intra", source="near-only", pm=2., lm=.1, gp=2., gl=-2.)]
        # The extra point shares leaf coordinates with protected known, so it
        # can only benefit the root component, never the leaf component.
        extra = [row("extra", "extra", source="extra-only", pm=.1, lm=3., gp=-2., gl=3.)]
        router = frontier._fit(known + near + extra, BASELINE, META, opts(), ("leaf_reject", "root_reject"))
        self.assertEqual(router["reject_rules"], [])

    def test_source_aliases_and_duplicate_images_do_not_inflate_support(self):
        known = [row("known", "known", pm=3., lm=3., gp=3., gl=3.)]
        near = [row("near-one", "intra", source="Near Species", pm=2., lm=.1, gp=2., gl=-2.),
                row("near-two", "intra", source="near_species", pm=2., lm=.1, gp=2., gl=-2.)]
        extra = [row("extra", "extra", pm=-1.)]
        records = known + near + extra + [copy.deepcopy(near[0])]
        router = frontier._fit(records, BASELINE, META, opts(), ("leaf_reject",))
        self.assertEqual(router["reject_rules"], [])

    def test_inner_components_cannot_borrow_other_actions_held_sources(self):
        known, near, extra = observations()
        for r in near[1:]:
            r["support_evidence"]["leaf_membership_logits"][0] = -1.
            r["baseline_leaf_z"] = -1.
        for r in extra:
            r["baseline_leaf_z"] = 3.
            r["geometry_leaf_score"] = 3.
        for r in extra[1:]:
            r["support_evidence"]["parent_membership_logits"][0] = -1.
            r["baseline_parent_z"] = -1.
            r["parent_evidence"]["membership_z"][0] = -1.
        records = known + near + extra

        def baseline(fitted, meta, options):
            return dict(BASELINE, fit_image_sha256=sorted(r["image_sha256"] for r in fitted),
                        calibration_sha256="synthetic-baseline")

        def fit(fitted, fitted_baseline, meta, options, actions):
            # Intentionally bypass FIT's own source guard to test the
            # independent inner-held component safeguard in isolation.
            bounds = {"leaf_reject": dict(x_max=.5, y_max=0.),
                      "root_reject": dict(x_max=.5, y_max=0.)}
            router = make_router(fitted_baseline, meta,
                [dict(action=a, **bounds[a]) for a in actions])
            router.update(rule_support=[], search=dict(fit_image_sha256=sorted(r["image_sha256"] for r in fitted)))
            return router

        with patch.object(frontier.risk, "_refit_baseline", side_effect=baseline), \
                patch.object(frontier, "_fit", side_effect=fit):
            router = frontier._nested_fit(records, BASELINE, META, opts())
        joint = next(c for c in router["inner_selection"]["candidates"] if len(c["action_plan"]) == 2)
        self.assertTrue(joint["selection_audit"]["admissible"])
        self.assertFalse(joint["admissible"])
        for component in joint["component_audits"].values():
            self.assertIn("benefiting_sources_below_minimum", component["selection_audit"]["rejection_reasons"])
        self.assertEqual(router["reject_rules"], [])

    def test_empty_fallback_and_active_decisions_do_not_read_truth(self):
        records = sum(observations(), [])
        empty = make_router(BASELINE, META)
        before, after = base.apply_router(records, BASELINE, META), apply_router(records, empty, META)
        self.assertEqual([[r[k] for k in FIELDS] for r in before], [[r[k] for k in FIELDS] for r in after])
        active = make_router(BASELINE, META, [dict(action="leaf_reject", x_max=.5, y_max=0.)])
        expected = apply_router(records, active, META)
        changed = copy.deepcopy(records)
        for r in changed:
            r.update(status="extra", split="test_extra", source="renamed", true_parent=1, true_leaf=2)
        actual = apply_router(changed, active, META)
        self.assertEqual([[r[k] for k in FIELDS] for r in expected], [[r[k] for k in FIELDS] for r in actual])

    def test_no_nonfinite_thresholds_even_at_representable_extremes(self):
        limit = np.finfo(np.float64).max
        candidates, info = frontier._frontier_masks(np.array([0., 1.]), np.array([-limit, 0.]),
                                                   np.ones(2, dtype=bool), np.array([True, False]))
        json.dumps({"bounds": [(x, y) for x, y, _ in candidates], "diagnostics": info}, allow_nan=False)
        self.assertTrue(all(not mask[0] for _, _, mask in candidates))

    def test_bad_router_or_nonfinite_evidence_fails_closed(self):
        records = [row("r", "known")]
        router = make_router(BASELINE, META)
        for edit in (dict(schema_version="parentrisk_v1"), dict(decoder="parentrisk"),
                     dict(parent_rule={"weights": [1., 0., 0.]}),
                     dict(reject_rules=[dict(action="parent_repair", x_max=0., y_max=0.)]),
                     dict(reject_rules=[dict(action="leaf_reject", x_max=float("inf"), y_max=0.)])):
            with self.subTest(edit=edit), self.assertRaises(ValueError):
                apply_router(records, dict(router, **edit), META)
        records[0]["geometry_leaf_score"] = float("nan")
        with self.assertRaises(ValueError):
            apply_router(records, router, META)

    def test_test_records_cannot_enter_calibration_and_outer_outputs_cannot_select(self):
        groups = observations()
        groups[0][0]["split"] = "test_known"
        with self.assertRaisesRegex(ValueError, "test fitting is prohibited"):
            frontier.calibrate(*groups, BASELINE, META, opts())
        with patch.object(frontier, "_outer_audit", return_value={"status": "heldout_degraded"}):
            a = frontier.calibrate(*observations(), BASELINE, META, opts())
        with patch.object(frontier, "_outer_audit", return_value={"status": "completed"}):
            b = frontier.calibrate(*observations(), BASELINE, META, opts())
        self.assertEqual(a["reject_rules"], b["reject_rules"])
        self.assertEqual(a["selection_status"], b["selection_status"])
        json.dumps(a, allow_nan=False)

    def test_outer_known_and_source_hashes_are_absent_from_all_inner_fits(self):
        router = frontier.calibrate(*observations(), BASELINE, META, opts())
        outer = router["outer_audit"]
        self.assertTrue(outer["complete"])
        self.assertFalse(outer["output_used_for_selection"])
        hashes = [r["image_sha256"] for r in outer["predictions"]]
        self.assertEqual(len(hashes), len(set(hashes)))
        self.assertEqual(len(hashes), 20)
        for fold in outer["folds"]:
            held = set(fold["held_image_sha256"])
            self.assertGreater(fold["known_held_n"], 0)
            self.assertFalse(held & set(fold["baseline_fit_image_sha256"]))
            for candidate in fold["inner_selection"]["candidates"]:
                for inner in candidate["folds"]:
                    fit = set(inner["fit_image_sha256"])
                    inner_held = set(inner["held_image_sha256"])
                    self.assertFalse(held & (fit | inner_held))
                    self.assertFalse(inner_held & set(inner["baseline_fit_image_sha256"]))
                    self.assertFalse(inner_held & set(inner["search"]["fit_image_sha256"]))


if __name__ == "__main__":
    unittest.main()
