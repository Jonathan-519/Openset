"""Separate leaf/fallback routing, finite binding and fit-only diagnostics."""
import copy
import hashlib
import json
import math
import unittest
from unittest.mock import patch

from taxosafe_routealign import calibration as route
from taxosafe_support import membership_calibration as membership

META = {"parent_names": ["p", "q"], "leaf_names": ["a", "b", "c"], "leaf_to_parent": [0, 0, 1]}
BASELINE = {"schema_version": "support_membership_v1", "decoder": "membership", "meta": META,
            "parent_threshold": 0., "leaf_threshold": 0.}
OPTIONS = {"grid_points": 5, "baseline_calibration": {"decoder": "membership", "membership_grid_points": 3, "source_loo": False}}


def row(name, status, pm=2., lm=2., *, source=None, parent=0, leaf=0, lp=None, pp=None):
    return dict(image_sha256=hashlib.sha256(name.encode()).hexdigest(), status=status, split="val_"+status,
                source=source or status, true_parent=None if status == "extra" else parent,
                true_leaf=leaf if status == "known" else None, log_probs=[-math.log(6)]*6,
                support_evidence=dict(parent_logits=[2., 1.], leaf_logits=[2., 1., 3.],
                    parent_membership_logits=[pm, pm], leaf_membership_logits=[lm, lm, lm]),
                proximity=dict(parent_proximity=pp or [0., 0.], leaf_proximity=lp or [0., 0., 0.]))


def fixture():
    return ([row("k"+str(i), "known") for i in range(6)],
            [row("n"+str(i), "intra", lm=-2., source="near"+str(i % 2)) for i in range(4)],
            [row("e"+str(i), "extra", pm=-2., lm=-2., source="extra"+str(i % 2)) for i in range(4)])


def manual(variant="joint", pt=0., lt=0., floor=0.):
    settings = route._settings(OPTIONS)
    return dict(schema_version=route.SCHEMA_VERSION, decoder=route.DECODER, meta=META,
                candidate_rule=membership.CANDIDATE_RULE, variant=variant,
                settings=settings, settings_sha256=route._hash(settings),
                parent_threshold=pt, leaf_threshold=lt, leaf_parent_floor=floor)


class RoutealignCalibrationTests(unittest.TestCase):
    def test_parent_fallback_threshold_cannot_delete_accepted_leaf(self):
        record = row("k", "known", pm=1., lm=2.)
        result = route.decode_records([record], manual(pt=100., lt=0.), META)[0]
        self.assertEqual((result["prediction_type"], result["leaf"]), ("known", 0))
        self.assertGreaterEqual(result["root_knownness_score"], 0.)
        self.assertEqual(result["fused_parent_score"], 1.)

    def test_rerank_occurs_after_leaf_decision_and_never_tries_another_parent(self):
        record = row("n", "intra", pm=2., lm=2., pp=[0., 5.], parent=1)
        state = manual("rerank")
        leaf = route.decode_records([record], state, META)[0]
        self.assertEqual((leaf["parent"], leaf["leaf"]), (0, 0))
        state["leaf_threshold"] = 10.
        parent = route.decode_records([record], state, META)[0]
        self.assertEqual((parent["prediction_type"], parent["parent"], parent["leaf"]), ("intra_unknown", 1, None))
        record["support_evidence"]["parent_membership_logits"] = [100., -100.]
        root = route.decode_records([record], state, META)[0]
        self.assertEqual(root["fallback_candidate_parent"], 1)
        self.assertEqual(root["prediction_type"], "global_unknown")
        self.assertIsNone(root["leaf"])

    def test_predictions_are_independent_of_labels_and_source(self):
        record = row("n", "intra", lm=-2., pp=[0., 5.])
        unlabeled = {k: v for k, v in record.items() if k not in ("status", "source", "true_leaf", "true_parent", "split")}
        state = manual("rerank")
        a, b = route.decode_records([record, unlabeled], state, META)
        for key in ("prediction_type", "parent", "leaf", "candidate_parent", "candidate_leaf", "fused_parent_score", "fused_leaf_score"):
            self.assertEqual(a[key], b[key])

    def test_train_proximity_has_bounded_influence_in_fusion_and_reranking(self):
        extreme = row("extreme", "extra", lp=[1.e300, -1.e300, 100.], pp=[-1.e300, 1.e300])
        clipped = copy.deepcopy(extreme)
        clipped["proximity"] = {"leaf_proximity": [5., -5., 5.], "parent_proximity": [-5., 5.]}
        for variant in ("joint", "rerank"):
            state = manual(variant, lt=100.)
            a, b = route.decode_records([extreme, clipped], state, META)
            for key in ("fused_leaf_score", "fused_parent_score", "fallback_candidate_parent", "prediction_type"):
                self.assertEqual(a[key], b[key])
            self.assertEqual(a["fused_leaf_score"], 7.)
            self.assertLessEqual(abs(a["fused_parent_score"]-2.), 5.)

    def test_fit_preserves_known_first_and_keeps_failed_gate_router(self):
        known = [row("k", "known")]
        near = [row("n", "intra")]
        extra = [row("e", "extra")]
        state = route.fit_router(known, near, extra, META, BASELINE, OPTIONS)
        self.assertEqual(state["status"], "best_effort")
        self.assertTrue(state["best_effort"])
        self.assertFalse(state["baseline_fallback"])
        self.assertEqual(state["paired_audit"]["known_lost_correct"], 0)
        self.assertEqual(route.decode_records(known, state, META)[0]["prediction_type"], "known")
        self.assertLessEqual(len(state["grid"]["parent_threshold"]), OPTIONS["grid_points"])
        json.dumps(state, allow_nan=False)

    def test_external_reference_exposes_irrecoverable_known_ranking_loss(self):
        groups = fixture()
        reference = copy.deepcopy(sum(groups, []))
        for record in groups[0]:
            record["support_evidence"]["leaf_logits"] = [1., 2., 3.]
        state = route.fit_router(*groups, META, BASELINE, OPTIONS, reference_records=reference)
        self.assertEqual(state["reference_scope"], "paired_external_reference")
        self.assertEqual(state["selection_diagnostics"]["irrecoverable_reference_known_candidate_count"], 6)
        self.assertEqual(state["paired_audit"]["known_lost_correct"], 6)
        self.assertEqual(state["status"], "best_effort")

    def test_evidence_validation_precedes_deduplication(self):
        for value in (float("nan"), float("inf"), True):
            groups = fixture()
            groups[0][0]["proximity"]["leaf_proximity"][0] = value
            with self.assertRaises(ValueError): route.fit_router(*groups, META, BASELINE, OPTIONS)
        groups = fixture()
        alias = copy.deepcopy(groups[0][0]); alias["proximity"]["leaf_proximity"][0] = 1.
        groups[0].append(alias)
        with self.assertRaisesRegex(ValueError, "conflicting proximity"):
            route.fit_router(*groups, META, BASELINE, OPTIONS)
        groups = fixture()
        groups[0].append(copy.deepcopy(groups[0][0]))
        state = route.fit_router(*groups, META, BASELINE, OPTIONS)
        self.assertEqual(state["duplicate_record_count"], 1)

    def test_router_settings_and_threshold_tampering_is_rejected(self):
        groups = fixture(); state = route.fit_router(*groups, META, BASELINE, OPTIONS)
        for key, value in (("decoder", "membership"), ("variant", "anything"), ("parent_threshold", float("nan")), ("leaf_threshold", 123.)):
            changed = copy.deepcopy(state); changed[key] = value
            with self.assertRaises(ValueError): route.decode_records(groups[0], changed, META)
        for options in ({"grid_points": 50}, {"proximity_weight": False}, {"proximity_clip": 6.}, {"unknown": 1}):
            with self.assertRaises(ValueError): route.validate_settings(options)
        self.assertEqual(route.validate_settings(), route.DEFAULT_SETTINGS)

    def test_test_rows_never_fit_or_enter_crossfit(self):
        for method in (route.fit_router, route.crossfit_audit):
            groups = fixture(); groups[0][0]["split"] = "test_known"
            with self.assertRaisesRegex(ValueError, "test fitting is prohibited"):
                method(*groups, META, BASELINE, OPTIONS)

    def test_every_crossfit_reference_grid_and_target_fit_excludes_held_hashes(self):
        groups = fixture()
        fit_calls = []
        ref_calls = []
        original_fit, original_ref = route.fit_router, membership.calibrate
        def fit(*args, **kwargs):
            state = original_fit(*args, **kwargs)
            fit_calls.append(state["fit_image_sha256"])
            return state
        def reference(*args, **kwargs):
            state = original_ref(*args, **kwargs)
            ref_calls.append(state["fit_image_sha256"])
            return state
        with patch.object(route, "fit_router", side_effect=fit), patch.object(membership, "calibrate", side_effect=reference):
            report = route.crossfit_audit(*groups, META, BASELINE, OPTIONS, reference_records=sum(groups, []))
        self.assertTrue(report["complete"])
        self.assertEqual(len(report["folds"]), 7)
        self.assertEqual(len(report["predictions"]), 14)
        for index, fold in enumerate(report["folds"]):
            held = set(fold["held_image_sha256"])
            self.assertFalse(held & set(fit_calls[index]))
            self.assertFalse(held & set(ref_calls[index]))
            self.assertEqual(fold["leaf_parent_floor"], fold["reference_parent_threshold"])
        self.assertFalse(report["independent_model_level_validation"])
        self.assertFalse(report["output_used_for_threshold_selection"])

    def test_membership_crossfit_does_not_require_proximity(self):
        groups = fixture()
        for record in sum(groups, []): record.pop("proximity")
        report = route.crossfit_audit(*groups, META, BASELINE, OPTIONS, variant="membership")
        self.assertTrue(report["complete"])
        self.assertFalse(report["passed"])  # identical models provide no strict unknown gain
        self.assertEqual(report["known_lost_correct"], 0)
        self.assertEqual(report["new_wrong_leaf_count"], 0)

    def test_source_audit_forbids_false_leaf_increase_without_correct_loss(self):
        records = [row("k", "known"), row("near1", "intra", parent=1, lm=-1.),
                   row("near2", "intra", lm=-1., source="other"), row("extra", "extra", pm=-1.)]
        old = membership.decode_records(records, BASELINE, META)
        new = copy.deepcopy(old)
        new[1].update(prediction_type="known", parent=0, leaf=0)
        audit = route.paired_audit(old, new, META)
        self.assertEqual(audit["known_lost_correct"], 0)
        self.assertEqual(audit["new_wrong_leaf_count"], 1)
        self.assertFalse(audit["unknown_sources_preserved"])
        self.assertFalse(audit["passed"])

    def test_source_aliases_are_held_together_and_rare_known_retained(self):
        groups = fixture()
        groups[1][0]["source"], groups[1][2]["source"] = "Near Species", "near_species"
        report = route.crossfit_audit(*groups, META, BASELINE, OPTIONS)
        folded = [f for f in report["folds"] if f.get("source") == "nearspecies"]
        self.assertEqual(len(folded), 1)
        self.assertEqual(len(folded[0]["held_image_sha256"]), 2)
        self.assertEqual(report["paired_audit"]["per_known_leaf"]["b"]["evidence_status"], "not_evaluable")


if __name__ == "__main__":
    unittest.main()
