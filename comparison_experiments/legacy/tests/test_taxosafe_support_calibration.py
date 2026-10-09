"""Small exact fixtures for support calibration protocol and target accounting."""
import copy
import hashlib
import unittest
from unittest.mock import patch

import numpy as np

from taxosafe_support.calibration import (
    apply_router, calibrate, evaluate_gates, evaluate_records, raw_records,
    source_loo, unique_records,
)

META = {"parent_names": ["copepod", "jelly"], "leaf_names": ["a", "b", "c"],
        "leaf_to_parent": [0, 0, 1]}


def row(index, status, logits, parent=None, leaf=None, source=None, split=None):
    values = np.asarray(logits, dtype=float)
    values -= np.log(np.exp(values - values.max()).sum()) + values.max()
    return {"image_sha256": hashlib.sha256(str(index).encode()).hexdigest(),
            "path": str(index) + ".png", "status": status,
            "split": split or {"known": "val_known", "intra": "val_intra", "extra": "val_extra"}[status],
            "true_parent": parent, "true_leaf": leaf, "source": source or status,
            "log_probs": values.tolist()}


def fixture():
    return ([row("k", "known", [-8, -8, -8, 8, -8, -8], 0, 0)],
            [row("n", "intra", [-8, 8, -8, -8, -8, -8], 0)],
            [row("e", "extra", [8, -8, -8, -8, -8, -8])])


def prediction(index, status, kind, correct=True):
    r = row(index, status, [0] * 6, 0 if status != "extra" else None,
            0 if status == "known" else None)
    if kind == "known":
        r.update(prediction_type=kind, parent=0, leaf=0 if correct else 1,
                 candidate_parent=0, candidate_leaf=0 if correct else 1)
    elif kind == "intra_unknown":
        r.update(prediction_type=kind, parent=0 if correct else 1, leaf=None,
                 candidate_parent=0 if correct else 1, candidate_leaf=0)
    else:
        r.update(prediction_type=kind, parent=None, leaf=None, candidate_parent=0, candidate_leaf=0)
    return r


class SupportCalibrationTests(unittest.TestCase):
    def test_two_bias_fit_and_separate_completion(self):
        k, n, e = fixture()
        settings = {"parent_bias_grid": [-1, 0, 1], "leaf_bias_grid": [-1, 0, 1], "source_loo": False}
        state = calibrate(k, n, e, META, settings)
        self.assertTrue(state["fit_completed"])
        self.assertTrue(state["targets_passed"])
        self.assertEqual(state["fitted_parameters"], ["parent_bias", "leaf_bias"])
        self.assertEqual(state["root_bias"], 0.)
        self.assertEqual(len(state["grid_tradeoff"]), 9)
        self.assertEqual(state["grid"]["sampled_feasible_count"], 9)
        # Identical evidence cannot route three different truth types correctly.
        k, n, e = fixture()
        for r in k + n + e:
            r["log_probs"] = row("identical", "extra", [0] * 6)["log_probs"]
        state = calibrate(k, n, e, META, settings)
        self.assertTrue(state["fit_completed"])
        self.assertFalse(state["targets_passed"])
        report = state["validation_report"]["infeasibility"]
        self.assertTrue(report["no_feasible_sampled_point"])
        self.assertIn("not a proof", report["scope"])

    def test_strict_boundaries_and_inclusive_near(self):
        records = [prediction("k%d" % i, "known", "known" if i < 9 else "intra_unknown") for i in range(10)]
        records += [prediction("n%d" % i, "intra", "intra_unknown" if i < 17 else "global_unknown") for i in range(20)]
        records += [prediction("e%d" % i, "extra", "global_unknown" if i < 9 else "known") for i in range(10)]
        report = evaluate_records(records)
        self.assertEqual(report["metrics"]["known_end_to_end_leaf_accuracy"], .9)
        self.assertEqual(report["metrics"]["intra_correct_fallback_rate"], .85)
        self.assertEqual(report["metrics"]["extra_global_unknown_recall"], .9)
        self.assertEqual(report["metrics"]["open_world_accepted_leaf_precision"], .9)
        self.assertEqual(list(report["checks"].values()), [False, True, False, False])
        self.assertEqual(report["requirements"]["known_end_to_end_leaf_accuracy"]["required_correct"], 10)
        self.assertEqual(report["requirements"]["intra_correct_fallback_rate"]["required_correct"], 17)

    def test_leaf_precision_includes_every_leaf_error(self):
        records = [prediction("kc", "known", "known"), prediction("kw", "known", "known", False),
                   prediction("n", "intra", "known"), prediction("e", "extra", "known")]
        report = evaluate_records(records)
        self.assertEqual(report["counts"]["leaf_outputs"], 4)
        self.assertEqual(report["metrics"]["open_world_accepted_leaf_precision"], .25)

    def test_missing_groups_and_zero_leaf_fail(self):
        records = [prediction("k", "known", "intra_unknown"), prediction("n", "intra", "intra_unknown")]
        report = evaluate_records(records)
        self.assertIsNone(report["metrics"]["extra_global_unknown_recall"])
        self.assertIsNone(report["metrics"]["open_world_accepted_leaf_precision"])
        self.assertFalse(report["targets_passed"])
        k, n, _ = fixture()
        with self.assertRaisesRegex(ValueError, "nonempty val_extra"):
            calibrate(k, n, [], META)

    def test_duplicate_alias_does_not_change_metric_or_fit(self):
        k, n, e = fixture()
        alias = dict(k[0], path="copy.png")
        state = calibrate(k + [alias], n, e, META,
                          {"parent_bias_grid": [0], "leaf_bias_grid": [0], "source_loo": False})
        self.assertEqual(state["unique_image_count"], 3)
        self.assertEqual(state["duplicate_record_count"], 1)
        decoded = apply_router(k + [alias] + n + e, state, META)
        self.assertEqual(len(decoded), 4)  # Per-manifest output remains exhaustive.
        report = evaluate_records(decoded)
        self.assertEqual(report["unique_image_count"], 3)
        self.assertEqual(report["counts"]["known"], 1)

    def test_conflicting_duplicate_and_cross_split_rejected(self):
        k, _, _ = fixture()
        with self.assertRaisesRegex(ValueError, "cross-split"):
            unique_records(k + [dict(k[0], split="test_known")])
        with self.assertRaisesRegex(ValueError, "inconsistent model evidence"):
            unique_records(k + [dict(k[0], log_probs=[-2.] * 6)])
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            unique_records([dict(k[0], image_sha256="path-is-not-content")])

    def test_test_splits_cannot_fit_or_source_calibrate(self):
        for split in ("test_known", "test_intra", "test_extra"):
            groups = fixture()
            index = ("test_known", "test_intra", "test_extra").index(split)
            groups[index][0]["split"] = split
            for function in (calibrate, source_loo):
                with self.assertRaisesRegex(ValueError, "test fitting is prohibited"):
                    function(*groups, META)

    def test_full_tree_decode_matches_exhaustive_argmax_and_ties(self):
        rng = np.random.RandomState(4)
        records = [row(i, "extra", values) for i, values in enumerate(rng.normal(size=(40, 6)))]
        for pb, lb in ((0., 0.), (-2., .5), (4., -1.)):
            state = {"parent_bias": pb, "leaf_bias": lb}
            decoded = apply_router(records, state, META)
            for original, output in zip(records, decoded):
                scores = np.asarray(original["log_probs"]) + [0, pb, pb, lb, lb, lb]
                self.assertEqual(output["output_node"], int(scores.argmax()))
                self.assertEqual(output["image_sha256"], original["image_sha256"])
                if output["prediction_type"] == "known":
                    self.assertEqual(output["parent"], META["leaf_to_parent"][output["leaf"]])
                if output["prediction_type"] == "intra_unknown":
                    self.assertIsNone(output["leaf"])
        tied = apply_router([row("tie", "extra", [0] * 6)], {"parent_bias": 0, "leaf_bias": 0}, META)[0]
        self.assertEqual(tied["prediction_type"], "global_unknown")

    def test_source_loo_excludes_held_source_from_fit(self):
        k, n, e = fixture()
        n += [row("n2", "intra", [-8, 8, -8, -8, -8, -8], 0, source="near2")]
        e += [row("e2", "extra", [8, -8, -8, -8, -8, -8], source="extra2")]
        import taxosafe_support.calibration as module
        calls = []
        original = module._select
        def track(rows, *args, **kwargs):
            calls.append(list(rows))
            return original(rows, *args, **kwargs)
        with patch.object(module, "_select", side_effect=track):
            report = source_loo(k, n, e, META, {"parent_bias_grid": [0], "leaf_bias_grid": [0]})
        self.assertEqual(len(report["folds"]), 4)
        for fold, fitted in zip(report["folds"], calls):
            self.assertFalse(any(r["status"] == fold["status"] and r["source"] == fold["held_source"] for r in fitted))
            self.assertEqual(fold["held_metrics"]["sample_count"], 1)
            self.assertEqual(fold["held_metrics"]["correct_rate"], 1.)

    def test_raw_records_and_probabilities_validated(self):
        k, _, _ = fixture()
        result = raw_records(k, {"log_probs": np.asarray([k[0]["log_probs"]])}, [[2, 1, 0]], META)
        self.assertEqual(result[0]["global_pred_leaf"], 0)
        self.assertNotIn("global_pred_leaf", k[0])
        for bad in ([0] * 6, [float("nan")] * 6, [-1.] * 5):
            with self.assertRaises(ValueError):
                apply_router([dict(k[0], log_probs=bad)], {"parent_bias": 0, "leaf_bias": 0}, META)

    def test_metrics_adapter_preserves_strict_gates(self):
        metrics = {"known": {"sample_count": 10},
                   "intra": {"sample_count": 20, "correct_fallback_rate": .85},
                   "extra": {"sample_count": 10, "global_unknown_recall": .9},
                   "overall": {"correct_leaf_count": 9, "accepted_leaf_count": 10}}
        self.assertEqual(list(evaluate_gates(metrics)["checks"].values()), [False, True, False, False])

    def test_group_diagnostics_include_all_outputs(self):
        k, n, e = fixture()
        decoded = apply_router(k + n + e, {"parent_bias": 0, "leaf_bias": 0}, META)
        report = evaluate_records(decoded, META)
        self.assertEqual(report["per_known_leaf"]["a"]["correct_rate"], 1.)
        self.assertEqual(report["per_intra_species"]["intra"]["correct_rate"], 1.)
        self.assertEqual(report["per_extra_source"]["extra"]["prediction_type_counts"],
                         {"global_unknown": 1, "intra_unknown": 0, "known": 0})


if __name__ == "__main__":
    unittest.main()
