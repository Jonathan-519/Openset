"""Regression tests for real routing and validation leakage risks."""
import copy
import unittest
from unittest.mock import patch

from taxosafe_parentrisk import calibration as risk
from taxosafe_parentrisk.decoder import apply_router, make_router, parent_candidates
from taxosafe_support import calibration as base
from tests.test_taxosafe_geometry_calibration import META, BASELINE, row as old_row


def row(identity, status, text=None, **kw):
    r = old_row(identity, status, **kw)
    r["encoder_evidence"] = {"parent_text_logits": text or [3., 1.]}
    r["parent_evidence"] = dict(text_z=text or [3., 1.],
        membership_z=[r["baseline_parent_z"], -2.],
        geometry_z=[r["geometry_parent_score"], -2.], geometry_raw=[r["geometry_parent_score"], -2.])
    return r


def groups():
    known = [row("known" + str(i), "known", pm=3., lm=3., gp=3., gl=3.) for i in range(12)]
    near = [row("near" + str(i), "intra", source="n" + str(i), pm=-1., lm=-1., gp=2., gl=-2.) for i in range(4)]
    extra = [row("extra" + str(i), "extra", source="e" + str(i), text=[-3., -4.], pm=-2., lm=1., gp=-2., gl=-2.) for i in range(4)]
    return known, near, extra


class ParentRiskTests(unittest.TestCase):
    def test_parent_only_preserves_every_original_leaf_and_never_promotes_leaf(self):
        parent_rule = dict(weights=[1., 0., 0.], threshold=0., margin_threshold=.1)
        router = make_router(BASELINE, META, "parent_only", parent_rule)
        records = [row("leaf", "extra", text=[0., 9.]), row("root", "intra", pm=-1., text=[0., 9.], parent=1),
                   row("parent", "known", lm=-1., text=[0., 9.])]
        before, after = base.apply_router(records, BASELINE, META), apply_router(records, router, META)
        self.assertEqual((after[0]["prediction_type"], after[0]["leaf"]), ("known", before[0]["leaf"]))
        self.assertEqual((after[1]["prediction_type"], after[1]["parent"], after[1]["leaf"]), ("intra_unknown", 1, None))
        self.assertEqual((after[2]["parent"], after[2]["leaf"]), (1, None))

    def test_choose_once_then_gate_and_second_parent_uses_own_geometry(self):
        r = row("r", "intra", pm=-1., text=[2., 4.])
        r["parent_evidence"]["geometry_z"] = [100., -8.]
        picks = parent_candidates([r], META, [0., 0., 1.])
        self.assertEqual(picks["parent"].tolist(), [0])
        router = make_router(BASELINE, META, "combined", dict(weights=[1., 0., 0.], threshold=3., margin_threshold=3.))
        after = apply_router([r], router, META)[0]
        self.assertEqual(after["proposed_parent"], 1)
        self.assertEqual(after["prediction_type"], "global_unknown")
        # Accept parent 1; root rule MUST see its -8 geometry, not parent's 0 +100.
        router["parent_rule"]["margin_threshold"] = 0.
        router["reject_rules"] = [dict(action="root_reject", x_max=0., y_max=0.)]
        self.assertEqual(apply_router([r], router, META)[0]["prediction_type"], "global_unknown")

    def test_empty_rules_exact_terminal_fallback_and_truth_independence(self):
        records = sum(groups(), [])
        router = make_router(BASELINE, META, "combined")
        before, after = base.apply_router(records, BASELINE, META), apply_router(records, router, META)
        fields = ("prediction_type", "parent", "leaf", "output_node", "candidate_parent", "candidate_leaf")
        self.assertEqual([[r[k] for k in fields] for r in before], [[r[k] for k in fields] for r in after])
        router["parent_rule"] = dict(weights=[1., 0., 0.], threshold=0., margin_threshold=0.)
        expected = apply_router(records, router, META)
        changed = copy.deepcopy(records)
        for r in changed:
            r.update(status="extra", source="arbitrary", split="test_extra", true_parent=1, true_leaf=2)
        actual = apply_router(changed, router, META)
        self.assertEqual([[r[k] for k in fields] for r in expected], [[r[k] for k in fields] for r in actual])

    def test_wrong_leaf_to_root_is_blocked_by_near_parent_path_guard(self):
        records = [row("k", "known", pm=3., lm=3., gp=3., gl=3.),
                   row("n", "intra", pm=.1, lm=.1, gp=-3., gl=3.),
                   row("e1", "extra", source="one", pm=.1, gp=-3.),
                   row("e2", "extra", source="two", pm=.1, gp=-3.)]
        settings = risk.settings(dict(mode="audit", grid_points=2))
        router = risk._fit_family(records, BASELINE, META, settings, None)
        out = apply_router(records, router, META)
        self.assertNotEqual(out[1]["prediction_type"], "global_unknown")
        self.assertFalse(any(r["action"] == "root_reject" for r in router["reject_rules"]))

    def test_fit_preserves_known_images_and_each_rule_has_own_two_source_support(self):
        known, near, extra = groups()
        settings = risk.settings(dict(mode="parent_only", grid_points=2))
        router = risk._fit_family(known + near + extra, BASELINE, META, settings, [1., 0., 0.])
        self.assertIsNotNone(router["parent_rule"])
        self.assertTrue(all(s["benefiting_source_count"] >= 2 for s in router["rule_support"]))
        audit = risk.paired_report(base.apply_router(known + near + extra, BASELINE, META),
                                 apply_router(known + near + extra, router, META), META)
        self.assertTrue(audit["preservation_audit"]["passed"])

    def test_nested_fold_baselines_and_rule_grids_exclude_outer_held_hashes(self):
        opts = dict(mode="parent_only", outer_folds=4, inner_folds=3, grid_points=2,
                    baseline_calibration=dict(decoder="membership", membership_grid_points=3, source_loo=False))
        with patch.object(risk, "PARENT_WEIGHTS", ((1., 0., 0.),)):
            router = risk.calibrate(*groups(), BASELINE, META, opts)
        outer = router["outer_audit"]
        self.assertTrue(outer["complete"])
        self.assertFalse(outer["output_used_for_selection"])
        self.assertEqual(len(outer["predictions"]), 20)
        self.assertEqual(len({r["image_sha256"] for r in outer["predictions"]}), 20)
        for fold in outer["folds"]:
            held = set(fold["held_image_sha256"])
            self.assertFalse(held & set(fold["baseline_fit_image_sha256"]))
            self.assertGreater(fold["known_held_n"], 0)
            for candidate in fold["inner_selection"]["candidates"]:
                for inner in candidate["folds"]:
                    self.assertFalse(held & set(inner["fit_image_sha256"]))
                    self.assertFalse(held & set(inner["held_image_sha256"]))
                    self.assertFalse(set(inner["held_image_sha256"]) & set(inner["baseline_fit_image_sha256"]))
                    self.assertFalse(set(inner["held_image_sha256"]) & set(inner["search"]["fit_image_sha256"]))

    def test_outer_audit_cannot_toggle_production_rules(self):
        opts = dict(mode="parent_only", grid_points=2,
                    baseline_calibration=dict(decoder="membership", membership_grid_points=3))
        with patch.object(risk, "PARENT_WEIGHTS", ((1., 0., 0.),)), patch.object(risk, "_outer_audit", return_value={"status": "heldout_degraded"}):
            first = risk.calibrate(*groups(), BASELINE, META, opts)
        with patch.object(risk, "PARENT_WEIGHTS", ((1., 0., 0.),)), patch.object(risk, "_outer_audit", return_value={"status": "completed"}):
            second = risk.calibrate(*groups(), BASELINE, META, opts)
        self.assertEqual(first["parent_rule"], second["parent_rule"])
        self.assertEqual(first["reject_rules"], second["reject_rules"])
        self.assertEqual(first["selection_status"], second["selection_status"])

    def test_missing_status_small_folds_report_not_searched_and_test_fit_rejected(self):
        k, n, e = [g[:1] for g in groups()]
        opts = dict(mode="parent_only", grid_points=2,
                    baseline_calibration=dict(decoder="membership", membership_grid_points=3))
        router = risk.calibrate(k, n, e, BASELINE, META, opts)
        self.assertEqual(router["selection_status"], "not_searched")
        self.assertTrue(router["baseline_fallback"])
        k[0]["split"] = "test_known"
        with self.assertRaisesRegex(ValueError, "test fitting is prohibited"):
            risk.calibrate(k, n, e, BASELINE, META, opts)

    def test_router_tampering_invalid_evidence_and_mode_rejected(self):
        r = row("r", "known")
        router = make_router(BASELINE, META, "parent_only")
        router["reject_rules"] = [dict(action="leaf_reject", x_max=0., y_max=0.)]
        with self.assertRaisesRegex(ValueError, "parent_only"):
            apply_router([r], router, META)
        router = make_router(BASELINE, META)
        r["parent_evidence"]["geometry_z"][0] = None
        with self.assertRaises(ValueError):
            apply_router([r], router, META)


if __name__ == "__main__":
    unittest.main()
