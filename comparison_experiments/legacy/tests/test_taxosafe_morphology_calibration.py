"""Public-API checks for frozen root-first gates and exact DEV-only calibration."""
import copy
import json
import math
import unittest
from collections import defaultdict
from fractions import Fraction
from unittest.mock import patch

import numpy as np

from taxosafe_boundary import calibration as boundary
from taxosafe_discovery import calibration as discovery
from taxosafe_morphology import calibration as morphology
from tests.test_taxosafe_discovery_calibration import META, bundle, fixture, row
from tests.test_taxosafe_recovery_calibration import buffered_fixture, d05_bundle


def as_morphology(groups):
    """Keep the old D05 evidence intact for an independently fitted reference."""
    result = copy.deepcopy(groups)
    for item in sum(result, []):
        evidence = item["discovery"]
        parent, leaf = evidence["candidate_parent"], evidence["candidate_leaf"]
        item["morphology"] = dict(
            root_score=evidence["parent_scores"][parent],
            parent_scores=copy.deepcopy(evidence["parent_scores"]),
            leaf_scores=copy.deepcopy(evidence["leaf_scores"]),
            candidate_parent=parent, candidate_leaf=leaf,
            anchor_parent=parent, anchor_leaf=leaf,
            root_method="d05", leaf_method="d05", candidate_policy="anchor")
    return result


def fit(groups, **kwargs):
    kwargs.setdefault("d05_records", d05_bundle(groups))
    return morphology.fit_router(*groups, META, **kwargs)


def boundaries(values):
    values = sorted(set(values))
    return [float(np.nextafter(values[0], -math.inf)), *values,
            float(np.nextafter(values[-1], math.inf))]


def manual_predictions(records, root_threshold, leaf_threshold):
    """Independent scalar decoder; deliberately does not call production decode."""
    result = []
    for item in records:
        value = item["morphology"]
        parent, leaf = value["candidate_parent"], value["candidate_leaf"]
        if value["root_score"] < root_threshold:
            result.append(("global_unknown", None, None))
        elif value["leaf_scores"][leaf] >= leaf_threshold:
            result.append(("known", parent, leaf))
        else:
            result.append(("intra_unknown", parent, None))
    return result


def staged_leaf_oracle(records, root_threshold):
    """Enumerate the stated staged policy with exact rational precision/macros."""
    totals = {status: sum(r["status"] == status for r in records)
              for status in ("known", "intra", "extra")}
    points = []
    for threshold in boundaries([r["morphology"]["leaf_scores"][r["morphology"]["candidate_leaf"]]
                                 for r in records]):
        predictions = manual_predictions(records, root_threshold, threshold)
        k = n = e = leaves = 0
        near_sources = defaultdict(list)
        for item, (kind, parent, leaf) in zip(records, predictions):
            leaves += kind == "known"
            k += item["status"] == "known" and kind == "known" and leaf == item["true_leaf"]
            near_correct = kind == "intra_unknown" and parent == item["true_parent"]
            n += item["status"] == "intra" and near_correct
            e += item["status"] == "extra" and kind == "global_unknown"
            if item["status"] == "intra":
                near_sources[item["source"]].append(near_correct)
        precision = Fraction(k, leaves) if leaves else Fraction(0)
        near_macro = sum((Fraction(sum(v), len(v)) for v in near_sources.values()), Fraction(0)) / len(near_sources)
        known_pass = 10*k > 9*totals["known"]
        precision_pass = leaves > 0 and 10*k > 9*leaves
        four = known_pass and precision_pass and 20*n >= 17*totals["intra"] and 10*e > 9*totals["extra"]
        tier = 3 if four else 2 if known_pass and precision_pass else 1 if known_pass else 0
        if tier >= 2:
            preference = (near_macro, k, precision, threshold)
        elif tier == 1:
            preference = (precision, near_macro, k, threshold)
        else:
            preference = (k, near_macro, precision, threshold)
        points.append((tier, preference, threshold, (k, n, e, leaves), predictions))
    return max(points, key=lambda point: (point[0], point[1]))


class MorphologyCalibrationTests(unittest.TestCase):
    def test_separable_fixture_has_executable_finite_router_and_report(self):
        groups = as_morphology(fixture())
        state, report = fit(groups)
        self.assertTrue(report["targets_passed"])
        self.assertEqual(report["counts"]["known_correct"], 6)
        self.assertEqual(report["counts"]["intra_correct"], 4)
        self.assertEqual(report["counts"]["extra_correct"], 4)
        self.assertEqual(state["root_threshold"], state["root_state"]["threshold"])
        actual = morphology.decode_records(sum(groups, []), state, META)
        expected = manual_predictions(sum(groups, []), state["root_threshold"], state["leaf_threshold"])
        self.assertEqual([(p["prediction_type"], p["parent"], p["leaf"]) for p in actual], expected)
        json.dumps([state, report], allow_nan=False)

    def test_rejected_root_cannot_be_rescued_by_any_leaf_or_parent_score(self):
        groups = as_morphology(fixture()); state, _ = fit(groups)
        item = copy.deepcopy(groups[0][0])
        item["morphology"]["root_score"] = state["root_threshold"] - 1.
        item["morphology"]["parent_scores"] = [1e20, 1e20]
        item["morphology"]["leaf_scores"] = [1e20, 1e20, 1e20]
        predicted = morphology.decode_records([item], state, META)[0]
        self.assertEqual((predicted["prediction_type"], predicted["parent"], predicted["leaf"]),
                         ("global_unknown", None, None))

    def test_root_state_is_identical_when_only_leaf_evidence_or_candidates_change(self):
        groups = as_morphology(fixture()); reference = d05_bundle(groups)
        original, _ = fit(groups, d05_records=reference)
        changed = copy.deepcopy(groups)
        for i, item in enumerate(sum(changed, [])):
            item["morphology"].update(leaf_scores=[-100.+i, 7.*i, 100.-i],
                                  parent_scores=[-100., 100.], candidate_parent=1, candidate_leaf=2,
                                  leaf_method="rank9", candidate_policy="rerouted")
        other, _ = fit(changed, d05_records=reference)
        self.assertEqual(original["root_state"], other["root_state"])
        self.assertEqual(original["root_state_sha256"], other["root_state_sha256"])
        self.assertEqual(original["root_threshold"], other["root_threshold"])
        self.assertEqual([p[0] == "global_unknown" for p in manual_predictions(sum(groups, []), original["root_threshold"], original["leaf_threshold"])],
                         [p[0] == "global_unknown" for p in manual_predictions(sum(changed, []), other["root_threshold"], other["leaf_threshold"])])

    def test_root_state_binds_root_evidence_even_when_selected_threshold_is_unchanged(self):
        groups = as_morphology(fixture()); reference = d05_bundle(groups)
        original, _ = fit(groups, d05_records=reference)
        changed = copy.deepcopy(groups); changed[2][0]["morphology"]["root_score"] -= .125
        other, _ = fit(changed, d05_records=reference)
        self.assertEqual(original["root_threshold"], other["root_threshold"])
        self.assertNotEqual(original["root_state_sha256"], other["root_state_sha256"])

    def test_root_boundary_keeps_all_tied_known_images_when_dropping_tie_exceeds_floor(self):
        groups = as_morphology(([row("k"+str(i), "known") for i in range(25)],
                            [row("n"+str(i), "intra", ls=-2.) for i in range(20)],
                            [row("e"+str(i), "extra", ps=-2., ls=-2.) for i in range(20)]))
        for item in groups[0]+groups[1]: item["morphology"]["root_score"] = 3.
        for item in groups[0][:3]: item["morphology"]["root_score"] = 1.
        for item in groups[2]: item["morphology"]["root_score"] = 2.
        state, _ = fit(groups)
        self.assertEqual(math.ceil(.92*25), 23)
        self.assertEqual(state["root_threshold"], 1.)
        self.assertTrue(all(p["prediction_type"] != "global_unknown"
                            for p in morphology.decode_records(groups[0], state, META)))
        # Equality is accepted, independently of source/labels at inference.
        item = {"morphology": copy.deepcopy(groups[0][0]["morphology"])}
        self.assertNotEqual(morphology.decode_records([item], state, META)[0]["prediction_type"], "global_unknown")

    def test_near_source_support_constraint_is_not_redundant_with_aggregate_floor(self):
        groups = as_morphology(([row("k"+str(i), "known") for i in range(25)],
                            [row("n"+str(i), "intra", ls=-2., source="rare" if i == 0 else "common") for i in range(20)],
                            [row("e"+str(i), "extra", ps=-2., ls=-2.) for i in range(20)]))
        for item in groups[0]+groups[1]: item["morphology"]["root_score"] = 3.
        groups[1][0]["morphology"]["root_score"] = 1.
        for item in groups[2]: item["morphology"]["root_score"] = 2.
        state, report = fit(groups)
        self.assertEqual(state["root_threshold"], 1.)
        self.assertEqual(report["root_stage"]["requested_constraints"]["near_sources"]["rare"], 1)
        self.assertEqual(report["root_stage"]["coverage"]["correct_near_parent_anchor_retained"], 20)
        self.assertEqual(morphology.decode_records([groups[1][0]], state, META)[0]["prediction_type"], "intra_unknown")
        # A threshold of3 satisfies both aggregate floors, but destroys rare's
        # previously reachable correct parent. The source constraint blocks it.
        self.assertGreaterEqual(25, math.ceil(.92*25))
        self.assertGreaterEqual(19, math.ceil(.85*20))

    def test_zero_structural_near_cap_produces_explicit_executable_best_effort(self):
        groups = as_morphology(fixture())
        for item in groups[1]:
            # Both immutable D05 evidence and Morphology anchors choose p, while
            # the unseen leaf's true parent is q: a valid zero-candidate ceiling.
            item["true_parent"] = 1
        state, report = fit(groups)
        self.assertFalse(report["targets_passed"])
        self.assertTrue(report["root_stage"]["structural_infeasible"])
        self.assertEqual(report["root_stage"]["candidate_ceiling"]["near"], 0)
        self.assertEqual(report["root_stage"]["requested_constraints"]["near"], 4)
        self.assertEqual(report["root_stage"]["effective_constraints"]["near"], 0)
        self.assertEqual(report["counts"]["intra_correct"], 0)
        self.assertEqual(len(morphology.decode_records(sum(groups, []), state, META)), 14)
        json.dumps([state, report], allow_nan=False)

    def test_staged_leaf_selector_matches_independent_exhaustive_decode(self):
        groups = as_morphology(buffered_fixture(correct=49))
        for i, item in enumerate(groups[0]):
            item["morphology"]["leaf_scores"][0] = 1. + (i % 7)*.173 if i < 49 else -.5
        for i, item in enumerate(groups[1]): item["morphology"]["leaf_scores"][0] = .9+i*.15
        for i, item in enumerate(groups[2]): item["morphology"]["leaf_scores"][0] = 100.+i
        known_only = as_morphology(buffered_fixture(correct=51))
        for item in sum(known_only, []): item["morphology"]["root_score"] = 2.
        for item in known_only[0]: item["morphology"]["leaf_scores"][0] = 0.
        for item in known_only[1]+known_only[2]: item["morphology"]["leaf_scores"][0] = 5.
        maximum_known = as_morphology(fixture())
        for item in maximum_known[0]: item["morphology"]["candidate_leaf"] = 1
        for expected_tier, case in [("four_gates", as_morphology(fixture())),
                                    ("known_precision", groups),
                                    ("known_only", known_only),
                                    ("maximum_known", maximum_known)]:
            with self.subTest(tier=expected_tier):
                state, report = fit(case)
                expected = staged_leaf_oracle(sum(case, []), state["root_threshold"])
                self.assertEqual(report["leaf_stage"]["selected_tier"], expected_tier)
                self.assertEqual(state["leaf_threshold"], expected[2])
                self.assertEqual(tuple(report["counts"][key] for key in
                                       ("known_correct", "intra_correct", "extra_correct", "leaf_outputs")), expected[3])
                predictions = morphology.decode_records(sum(case, []), state, META)
                self.assertEqual([(p["prediction_type"], p["parent"], p["leaf"]) for p in predictions], expected[4])

    def test_joint_same_score_control_matches_boundary_kp_public_selection(self):
        groups = as_morphology(buffered_fixture(correct=49)); reference = d05_bundle(groups)
        old_state, old_report = boundary.fit_router(*groups, META, d05_records=reference)
        state, report = fit(groups, policy="joint", d05_records=reference)
        self.assertEqual(state["root_threshold"], old_state["global_parent_threshold"])
        self.assertEqual(state["leaf_threshold"], old_state["global_leaf_threshold"])
        self.assertEqual(report["counts"], old_report["counts"])
        self.assertEqual(report["targets_passed"], old_report["targets_passed"])

    def test_decode_is_independent_of_truth_source_and_reference_evidence(self):
        groups = as_morphology(fixture()); state, _ = fit(groups)
        labelled = copy.deepcopy(groups[1][0]); only_morphology = {"morphology": copy.deepcopy(labelled["morphology"])}
        labelled.update(source="irrelevant", true_leaf=999, true_parent=999,
                        discovery={"ignored": True}, support_evidence={"ignored": True})
        a, b = morphology.decode_records([labelled, only_morphology], state, META)
        for key in ("prediction_type", "parent", "leaf", "root_knownness_score", "local_knownness_score"):
            self.assertEqual(a[key], b[key])

    def test_invalid_morphology_schema_nonfinite_values_and_mapping_are_rejected(self):
        bad_updates = [dict(root_score=float("nan")), dict(root_score=float("inf")), dict(root_score=True),
                       dict(leaf_scores=[2., float("inf"), -2.]), dict(parent_scores=[2.]),
                       dict(candidate_leaf=3), dict(candidate_parent=True),
                       dict(candidate_parent=1, candidate_leaf=0), dict(anchor_parent=1, anchor_leaf=0)]
        for updates in bad_updates:
            with self.subTest(updates=updates):
                groups = as_morphology(fixture()); groups[0][0]["morphology"].update(updates)
                with self.assertRaises(ValueError): fit(groups)
        groups = as_morphology(fixture()); del groups[0][0]["morphology"]["root_score"]
        with self.assertRaises(ValueError): fit(groups)
        groups = as_morphology(fixture()); groups[0][0]["morphology"]["unexpected"] = 1
        with self.assertRaises(ValueError): fit(groups)

    def test_router_checksum_binds_root_state_and_both_thresholds(self):
        groups = as_morphology(fixture()); state, _ = fit(groups)
        for key in ("root_threshold", "leaf_threshold"):
            bad = copy.deepcopy(state); bad[key] += .001
            with self.assertRaises(ValueError): morphology.decode_records(groups[0], bad, META)
        bad = copy.deepcopy(state); bad["root_state"]["threshold"] += .001
        with self.assertRaises(ValueError): morphology.decode_records(groups[0], bad, META)

    def test_conflicting_duplicate_evidence_is_rejected(self):
        groups = as_morphology(fixture()); duplicate = copy.deepcopy(groups[0][0])
        duplicate["morphology"]["root_score"] += .1; groups[0].append(duplicate)
        with self.assertRaises(ValueError): fit(groups)

    def test_test_split_is_never_used_for_fitting(self):
        for function in (morphology.fit_router, morphology.crossfit_audit):
            groups = as_morphology(fixture()); reference = d05_bundle(groups)
            groups[0][0]["split"] = "test_known"
            with self.assertRaisesRegex(ValueError, "test fitting is prohibited"):
                function(*groups, META, d05_records=reference)

    def test_crossfit_refits_root_leaf_and_d05_only_on_fit_images(self):
        groups = as_morphology(buffered_fixture(correct=51)); reference = d05_bundle(groups)
        calls, d05_calls = [], []
        real_fit, real_d05 = morphology.fit_router, discovery.fit_router
        def checked_fit(known, near, extra, *args, **kwargs):
            result = real_fit(known, near, extra, *args, **kwargs)
            calls.append(({r["image_sha256"] for r in known+near+extra}, result[0]))
            return result
        def checked_d05(known, near, extra, *args, **kwargs):
            d05_calls.append({r["image_sha256"] for r in known+near+extra})
            return real_d05(known, near, extra, *args, **kwargs)
        with patch.object(morphology, "fit_router", side_effect=checked_fit), patch.object(discovery, "fit_router", side_effect=checked_d05):
            audit = morphology.crossfit_audit(*groups, META, reference_records=bundle(sum(groups, [])), d05_records=reference)
        self.assertTrue(audit["complete"])
        self.assertEqual(audit["evaluated_image_count"], 59)
        self.assertEqual(len(calls), len(audit["folds"]))
        self.assertEqual(len(d05_calls), len(audit["folds"]))
        for fold, (fitted, state), old_fitted in zip(audit["folds"], calls, d05_calls):
            fit_ids, held = set(fold["fit_image_sha256"]), set(fold["held_image_sha256"])
            self.assertFalse(fit_ids & held)
            self.assertEqual(fitted, fit_ids)
            self.assertEqual(old_fitted, fit_ids)
            self.assertEqual(set(state["fit_image_sha256"]), fit_ids)
        self.assertFalse(audit["output_used_for_threshold_selection"])


if __name__ == "__main__":
    unittest.main()
