"""Independent node evidence, exact boundaries and conditional DEV folds."""
import copy
import hashlib
import json
import math
import unittest
from unittest.mock import patch

import numpy as np

from taxosafe_discovery import calibration as discovery
from taxosafe_support import membership_calibration as membership

META = {"parent_names": ["p", "q"], "leaf_names": ["a", "b", "c"], "leaf_to_parent": [0, 0, 1]}
BASELINE = {"schema_version": "support_membership_v1", "decoder": "membership", "meta": META,
            "parent_threshold": 0., "leaf_threshold": 0.}


def row(name, status, parent=0, leaf=0, ps=2., ls=2., source=None):
    parent_scores = [-10., -10.]; parent_scores[parent] = ps
    leaf_scores = [-10., -10., -10.]; leaf_scores[leaf] = ls
    return dict(image_sha256=hashlib.sha256(name.encode()).hexdigest(), status=status, split="val_"+status,
                source=source or status, true_parent=None if status == "extra" else parent,
                true_leaf=leaf if status == "known" else None,
                discovery=dict(parent_scores=parent_scores, leaf_scores=leaf_scores, candidate_parent=parent, candidate_leaf=leaf),
                log_probs=[-math.log(6.)]*6,
                support_evidence=dict(parent_logits=[2., 1.], leaf_logits=[2., 1., 3.],
                    parent_membership_logits=[ps, ps], leaf_membership_logits=[ls, ls, ls]))


def fixture():
    return ([row("k"+str(i), "known") for i in range(6)],
            [row("n"+str(i), "intra", ls=-2., source="near"+str(i % 2)) for i in range(4)],
            [row("e"+str(i), "extra", ps=-2., ls=-2., source="extra"+str(i % 2)) for i in range(4)])


def bundle(records):
    return dict(records=records, router=BASELINE,
                calibration_settings={"decoder": "membership", "membership_grid_points": 3, "source_loo": False})


def at_thresholds(pt=0., lt=0.):
    settings = discovery.validate_settings()
    state = dict(schema_version=discovery.SCHEMA_VERSION, decoder=discovery.DECODER, meta=META,
                 variant="global", settings=settings, settings_sha256=discovery._hash(settings),
                 candidate_rule="supplied_before_thresholds;leaf_own_parent_admission;single_supplied_fallback",
                 global_parent_threshold=pt, global_leaf_threshold=lt,
                 parent_offsets=[0., 0.], leaf_offsets=[0., 0.], comparison=discovery.COMPARISON,
                 parent_thresholds=[pt, pt], leaf_thresholds=[lt, lt])
    state["router_sha256"] = discovery._router_hash(state)
    return state


class DiscoveryCalibrationTests(unittest.TestCase):
    def test_leaf_uses_own_parent_gate_and_fallback_uses_preselected_parent(self):
        item = row("n", "intra")
        item["discovery"].update(candidate_parent=1, parent_scores=[-1., 2.])
        result = discovery.decode_records([item], at_thresholds(), META)[0]
        self.assertEqual((result["prediction_type"], result["parent"], result["leaf"]), ("intra_unknown", 1, None))
        item["discovery"]["parent_scores"] = [2., -1.]
        result = discovery.decode_records([item], at_thresholds(), META)[0]
        self.assertEqual((result["prediction_type"], result["parent"], result["leaf"]), ("known", 0, 0))
        item["discovery"]["leaf_scores"][0] = -1.
        result = discovery.decode_records([item], at_thresholds(), META)[0]
        self.assertEqual(result["prediction_type"], "global_unknown")  # cannot try p=0 after fixed p=1 fails

    def test_discovery_inference_ignores_support_scores_truth_and_source(self):
        item = row("n", "intra", ls=-1.)
        unlabelled = {"discovery": copy.deepcopy(item["discovery"])}
        item["support_evidence"] = {"invalid": "ignored by discovery"}
        a, b = discovery.decode_records([item, unlabelled], at_thresholds(), META)
        for key in ("prediction_type", "parent", "leaf", "root_knownness_score", "local_knownness_score"):
            self.assertEqual(a[key], b[key])
        self.assertGreaterEqual(a["root_knownness_score"], 0.)
        self.assertLess(a["local_knownness_score"], 0.)

    def test_exact_search_finds_feasible_router_and_returns_separate_report(self):
        groups = fixture()
        router, report = discovery.fit_router(*groups, META, reference_records=bundle(sum(groups, [])))
        self.assertEqual(router["status"], "feasible")
        self.assertTrue(report["targets_passed"])
        self.assertTrue(report["known_count_preserved"])
        grid = report["exact_threshold_search"]
        self.assertTrue(grid["includes_all_accept_and_all_reject"])
        self.assertGreater(grid["all_four_gate_candidates"], 0)
        self.assertLess(grid["parent_grid"][0], -2.)
        self.assertGreater(grid["parent_grid"][-1], 2.)
        json.dumps([router, report], allow_nan=False)

    def test_exact_search_matches_independent_bruteforce_with_different_parents(self):
        groups = fixture()
        for i, item in enumerate(sum(groups, [])):
            item["discovery"]["parent_scores"][1] = (i % 5)-2.25
            item["discovery"]["leaf_scores"][0] += i*.031
            if i % 3 == 0: item["discovery"]["candidate_parent"] = 1
        router, report = discovery.fit_router(*groups, META)
        pg, lg = report["exact_threshold_search"]["parent_grid"], report["exact_threshold_search"]["leaf_grid"]
        feasible = 0
        for pt in pg:
            for lt in lg:
                predictions = discovery.decode_records(sum(groups, []), at_thresholds(pt, lt), META)
                scored = discovery.base.evaluate_records(predictions, META)
                feasible += int(scored["targets_passed"])
        self.assertEqual(feasible, report["exact_threshold_search"]["all_four_gate_candidates"])
        self.assertEqual(report["targets_passed"], feasible > 0)

    def test_failed_gates_still_return_executable_best_effort(self):
        groups = ([row("k", "known")], [row("n", "intra")], [row("e", "extra")])
        router, report = discovery.fit_router(*groups, META)
        self.assertEqual(router["status"], "best_effort")
        self.assertFalse(router["baseline_fallback"])
        self.assertTrue(report["checks"]["known_end_to_end_leaf_accuracy"])
        self.assertEqual(len(discovery.decode_records(sum(groups, []), router, META)), 3)

    def test_reference_image_preservation_is_reported_without_hard_fit_constraint(self):
        groups = fixture()
        groups[0].extend(row("additional"+str(i), "known") for i in range(14))
        reference = bundle(copy.deepcopy(sum(groups, [])))
        groups[0][0]["discovery"]["candidate_leaf"] = 1
        groups[0][0]["discovery"]["leaf_scores"][1] = 2.
        state, report = discovery.fit_router(*groups, META, reference_records=reference)
        self.assertTrue(report["targets_passed"])
        self.assertEqual(report["paired_audit"]["known_lost_correct"], 1)
        self.assertFalse(report["known_count_preserved"])
        self.assertEqual(state["status"], "feasible")
        audit = discovery.crossfit_audit(*groups, META, reference_records=reference)
        self.assertTrue(audit["report"]["targets_passed"])
        self.assertFalse(audit["known_count_preserved"])
        self.assertFalse(audit["passed"])

    def test_parentwise_changes_only_thresholds_with_fixed_partial_pooling(self):
        groups = fixture()
        groups[0].extend(row("q"+str(i), "known", parent=1, leaf=2, ps=102., ls=102.) for i in range(6))
        state, _ = discovery.fit_router(*groups, META, variant="parentwise")
        self.assertNotEqual(state["parent_thresholds"][0], state["parent_thresholds"][1])
        for level in ("parent", "leaf"):
            diagnostics = state["offsets"][level]
            self.assertEqual(diagnostics["global_count"], 12)
            self.assertEqual(diagnostics["groups"][0]["pooling_weight"], 6/26)
            self.assertAlmostEqual(diagnostics["groups"][0]["offset"], -50*6/26)
        changed_unknowns = copy.deepcopy(groups)
        for item in changed_unknowns[1] + changed_unknowns[2]:
            item["discovery"]["leaf_scores"] = [900., 900., 900.]
        changed, _ = discovery.fit_router(*changed_unknowns, META, variant="parentwise")
        self.assertEqual(state["offsets"], changed["offsets"])  # offsets use known-fit only

    def test_sparse_and_absent_parent_statistics_use_global_threshold(self):
        groups = fixture(); groups = (groups[0][:3], groups[1], groups[2])
        state, _ = discovery.fit_router(*groups, META, variant="parentwise")
        for level in ("parent", "leaf"):
            self.assertEqual(state["offsets"][level]["groups"][0]["evidence_status"], "insufficient_evidence")
            self.assertEqual(state["offsets"][level]["groups"][1]["evidence_status"], "not_evaluable")
            self.assertTrue(all(g["offset"] == 0 for g in state["offsets"][level]["groups"]))

    def test_parentwise_fractional_boundary_uses_identical_centering_at_inference(self):
        a, b = 8.459776330097977, 115.15908805880605
        groups = fixture()
        for item in groups[0]:
            item["discovery"]["parent_scores"][0] = a
            item["discovery"]["leaf_scores"][0] = a
        groups[0].extend(row("q"+str(i), "known", parent=1, leaf=2, ps=b, ls=b) for i in range(6))
        for item in groups[1]: item["discovery"]["parent_scores"][0] = a
        state, report = discovery.fit_router(*groups, META, variant="parentwise")
        self.assertTrue(state["targets_passed"])
        self.assertTrue(report["targets_passed"])
        self.assertEqual(report["counts"]["known_correct"], 12)
        self.assertEqual(report["counts"]["intra_correct"], 4)
        actual = discovery.decode_records(sum(groups, []), state, META)
        self.assertEqual(actual[0]["selected_parent_score"], a-state["parent_offsets"][0])
        self.assertEqual(actual[0]["parent_threshold"], state["global_parent_threshold"])
        # Include every tied boundary in an independent decoder re-evaluation.
        feasible = 0
        for pt in report["exact_threshold_search"]["parent_grid"]:
            for lt in report["exact_threshold_search"]["leaf_grid"]:
                candidate = copy.deepcopy(state)
                candidate.update(global_parent_threshold=pt, global_leaf_threshold=lt,
                    parent_thresholds=(pt+np.asarray(state["parent_offsets"])).tolist(),
                    leaf_thresholds=(lt+np.asarray(state["leaf_offsets"])).tolist())
                candidate["router_sha256"] = discovery._router_hash(candidate)
                scored = discovery.base.evaluate_records(discovery.decode_records(sum(groups, []), candidate, META), META)
                feasible += int(scored["targets_passed"])
        self.assertEqual(feasible, report["exact_threshold_search"]["all_four_gate_candidates"])
        for key in ("parent_offsets", "leaf_offsets", "global_parent_threshold", "global_leaf_threshold"):
            tampered = copy.deepcopy(state)
            if isinstance(tampered[key], list): tampered[key][0] += .1
            else: tampered[key] += .1
            with self.assertRaisesRegex(ValueError, "checksum"):
                discovery.decode_records(groups[0], tampered, META)

    def test_nan_candidates_duplicates_and_router_tampering_rejected(self):
        for value in (float("nan"), float("inf"), True):
            groups = fixture(); groups[0][0]["discovery"]["leaf_scores"][0] = value
            with self.assertRaises(ValueError): discovery.fit_router(*groups, META)
        groups = fixture(); alias = copy.deepcopy(groups[0][0]); alias["discovery"]["candidate_leaf"] = 1; groups[0].append(alias)
        with self.assertRaisesRegex(ValueError, "conflicting discovery"):
            discovery.fit_router(*groups, META)
        groups = fixture(); state, _ = discovery.fit_router(*groups, META)
        state["parent_thresholds"][0] += .01
        with self.assertRaisesRegex(ValueError, "checksum"):
            discovery.decode_records(groups[0], state, META)

    def test_test_never_enters_fitting_and_reference_identities_are_locked(self):
        for function in (discovery.fit_router, discovery.crossfit_audit):
            groups = fixture(); groups[0][0]["split"] = "test_known"
            with self.assertRaisesRegex(ValueError, "test fitting is prohibited"):
                function(*groups, META)
        groups = fixture()
        with self.assertRaisesRegex(ValueError, "identities differ"):
            discovery.fit_router(*groups, META, reference_records=bundle(sum(groups, [])[:-1]))

    def test_crossfit_refits_offsets_and_reference_only_on_fit_hashes(self):
        groups = fixture(); fit_calls = []; ref_calls = []
        original_fit, original_reference = discovery.fit_router, membership.calibrate
        def fit(*args, **kwargs):
            result = original_fit(*args, **kwargs); fit_calls.append(result[0]); return result
        def reference(*args, **kwargs):
            result = original_reference(*args, **kwargs); ref_calls.append(result); return result
        with patch.object(discovery, "fit_router", side_effect=fit), patch.object(membership, "calibrate", side_effect=reference):
            report = discovery.crossfit_audit(*groups, META, variant="parentwise", reference_records=bundle(sum(groups, [])))
        self.assertTrue(report["complete"])
        self.assertTrue(report["passed"])
        self.assertEqual(len(report["predictions"]), 14)
        self.assertEqual(len(report["folds"]), 7)
        for i, fold in enumerate(report["folds"]):
            held = set(fold["held_image_sha256"])
            self.assertFalse(held & set(fit_calls[i]["fit_image_sha256"]))
            self.assertFalse(held & set(ref_calls[i]["fit_image_sha256"]))
            if fold["kind"] == "known":
                self.assertEqual(fit_calls[i]["offsets"]["leaf"]["global_count"], 4)
        self.assertTrue(report["source_protection_is_diagnostic"])
        self.assertFalse(report["independent_model_level_validation"])

    def test_without_raw_reference_crossfit_is_explicitly_unevaluable(self):
        report = discovery.crossfit_audit(*fixture(), META)
        self.assertFalse(report["passed"])
        self.assertEqual(report["status"], "not_evaluable")


if __name__ == "__main__":
    unittest.main()
