"""CPU regression tests: python -m unittest discover -s tests -p test_taxosafe_visual.py -v"""

import copy
import hashlib
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from taxosafe_visual import core


def toy_meta():
    return {"parent_names": ["A", "B", "C"],
            "leaf_names": ["a1", "a2", "b1", "b2", "c1"],
            "leaf_to_parent": [0, 0, 1, 1, 2]}


def toy_dataset():
    meta = toy_meta()
    rng = np.random.RandomState(812)
    centres = np.eye(12)[:5]
    bank_f, bank_y = [], []
    for c in range(5):
        bank_f.extend(centres[c] + rng.normal(0, .03, (12, 12)))
        bank_y.extend([c] * 12)
    bank = core.make_bank(np.asarray(bank_f), bank_y, meta)
    rows, features, pc, lc = [], [], [], []
    for status in ("known", "intra", "extra"):
        for group in range(5 if status == "known" else 2):
            for j in range(12):
                if status == "known":
                    leaf = group; parent = meta["leaf_to_parent"][leaf]
                    f = centres[leaf] + rng.normal(0, .05, 12)
                elif status == "intra":
                    leaf = None; parent = group
                    f = np.eye(12)[5 + group] + .2 * centres[2 * group] + rng.normal(0, .04, 12)
                else:
                    leaf, parent = None, None
                    f = np.eye(12)[8 + group] + rng.normal(0, .03, 12)
                p = rng.normal(.10, .01, 3)
                l = rng.normal(.10, .01, 5)
                if parent is not None:
                    p[parent] = .4
                if leaf is not None:
                    l[leaf] = .45
                identity = "{}-{}-{}".format(status, group, j)
                rows.append({"status": status, "true_parent": parent, "true_leaf": leaf,
                             "source": status + str(group), "path": identity + "/img.jpg",
                             "image_sha256": hashlib.sha256(identity.encode()).hexdigest()})
                features.append(f); pc.append(p); lc.append(l)
    return meta, bank, rows, np.asarray(features), np.asarray(pc), np.asarray(lc)


class VisualSupportTests(unittest.TestCase):
    def test_bank_prototypes_are_class_balanced(self):
        meta = toy_meta()
        f = np.eye(5)[[0] * 15 + [1, 2, 3, 4]]
        labels = [0] * 15 + [1, 2, 3, 4]
        bank = core.make_bank(f, labels, meta)
        np.testing.assert_allclose(bank["parent_prototypes"][0, :2], [2 ** -.5] * 2, atol=1e-6)

    def test_knn_matches_brute_force_and_k_is_capped(self):
        meta, bank, rows, f, pc, lc = toy_dataset()
        out = core.retrieve(f[:7], bank, meta, k=3, root_k=10, chunk_size=2)
        sim = core.normalise(f[:7]) @ core.normalise(bank["features"]).T
        expected = np.sort(sim, axis=1)[:, -10]
        np.testing.assert_allclose(out["global_knn"], expected)
        expected_class = np.sort(sim[:, bank["labels"] == 0], axis=1)[:, -3]
        np.testing.assert_allclose(out["class_knn"][:, 0], expected_class)
        out_large = core.retrieve(f[:7], bank, meta, k=9999)
        np.testing.assert_allclose(out_large["class_knn"][:, 0], sim[:, bank["labels"] == 0].min(axis=1))

    def test_invalid_memory_labels_fail(self):
        with self.assertRaises(ValueError):
            core.make_bank(np.eye(5), [0, 1, 2, 3, -1], toy_meta())

    def test_invalid_knn_settings_fail(self):
        meta, bank, rows, f, pc, lc = toy_dataset()
        for settings in ({"k": 0}, {"root_k": -1}, {"chunk_size": 0}):
            with self.assertRaises(ValueError):
                core.retrieve(f[:2], bank, meta, **settings)

    def test_zero_feature_rejected(self):
        with self.assertRaises(ValueError):
            core.normalise(np.zeros((2, 3)))

    def test_partition_deterministic_and_disjoint(self):
        rows = toy_dataset()[2]
        fit, cal = core.validation_partition(rows)
        f2, c2 = core.validation_partition(rows)
        np.testing.assert_array_equal(fit, f2)
        np.testing.assert_array_equal(cal, c2)
        self.assertFalse(set(fit) & set(cal))
        self.assertEqual(set(fit) | set(cal), set(range(len(rows))))

    def test_risk_threshold_controls_every_source_including_ties(self):
        scores = np.asarray([.1, .1, .5, .9, .9, .4, .7, .7, .7, .8])
        groups = np.asarray(["one"] * 5 + ["two"] * 5)
        tau = core.risk_threshold(scores, np.ones(10, bool), groups, .2)
        for g in np.unique(groups):
            self.assertLessEqual(np.mean(scores[groups == g] >= tau), .2)
        tau_zero = core.risk_threshold(scores, np.ones(10, bool), groups, 0)
        self.assertFalse(np.any(scores >= tau_zero))

    def test_risk_threshold_rejects_missing_negatives(self):
        with self.assertRaises(ValueError):
            core.risk_threshold(np.ones(3), np.zeros(3, bool), ["x"] * 3)

    def test_coverage_boundary_uses_inclusive_acceptance(self):
        x = np.arange(10, dtype=float)
        tau = core.coverage_threshold(x, np.ones(10, bool), .1)
        self.assertEqual(np.mean(x >= tau), .9)

    def test_singleton_margin_is_not_evidence(self):
        z = np.asarray([[1., -10000., 3., 4.]])
        score = core.weighted_score(z, [.25, .25, .5, 0], np.array([2]), toy_meta())
        self.assertAlmostEqual(float(score[0]), (1 * .25 + 3 * .5) / .75)

    def test_routing_always_stays_inside_parent(self):
        meta, bank, rows, f, pc, lc = toy_dataset()
        support = core.retrieve(f, bank, meta)
        for a in (0., .25, 1.):
            pp, pl = core.route(pc, lc, support, meta, 100, a, a)
            for g, leaf in zip(pp, pl):
                self.assertEqual(meta["leaf_to_parent"][leaf], g)

    def test_complete_fit_predict_and_no_truth_used_for_decision(self):
        meta, bank, rows, f, pc, lc = toy_dataset()
        fit, cal = core.validation_partition(rows)
        support = core.retrieve(f, bank, meta)
        routing, report = core.routing_fit(pc, lc, support, meta, 100, rows, fit, {})
        e = core.evidence(pc, lc, support, meta, 100, routing)
        c = core.fit_calibration(e, rows, meta, fit, cal, {})
        self.assertTrue(c["profiles"]["risk"]["empirical_constraints_satisfied"])
        # Changing ground truth after fitting must not change any decision.
        forged = copy.deepcopy(rows)
        for r in forged:
            r["true_parent"], r["true_leaf"] = 0, 0
        for profile in ("risk", "coverage"):
            a = core.predict(e, rows, c, profile)
            b = core.predict(e, forged, c, profile)
            self.assertEqual([(r["prediction_type"], r["parent"], r["leaf"]) for r in a],
                             [(r["prediction_type"], r["parent"], r["leaf"]) for r in b])
            self.assertTrue(any(r["prediction_type"] == "known" for r in a))

    def test_calibration_labels_cannot_change_selected_method_or_moments(self):
        meta, bank, rows, f, pc, lc = toy_dataset()
        fit, cal = core.validation_partition(rows)
        support = core.retrieve(f, bank, meta)
        e = core.evidence(pc, lc, support, meta, 100,
                          {"parent_alpha": 0., "child_alpha": 0., "cache_scale": 20.})
        a = core.fit_calibration(e, rows, meta, fit, cal, {})
        shuffled = copy.deepcopy(rows)
        for i in cal:
            # Keep statuses/sources and enough correct positives, but modify a
            # subset of held calibration labels. Fitted transforms stay fixed.
            if shuffled[i]["status"] == "known" and i % 3 == 0:
                shuffled[i]["true_leaf"] = (shuffled[i]["true_leaf"] + 1) % 5
        b = core.fit_calibration(e, shuffled, meta, fit, cal, {})
        for key in ("root_method", "child_method", "root_moments", "child_moments"):
            self.assertEqual(a[key], b[key])

    def test_input_overlap_rejected(self):
        from taxosafe_visual.runtime import assert_disjoint
        with self.assertRaises(ValueError):
            assert_disjoint([{"image_sha256": "a"}], {"a"}, "test")
        with self.assertRaises(ValueError):
            assert_disjoint([{"image_sha256": "b"}] * 2, set(), "val")

    def test_manifest_rejects_known_species_used_as_unknown(self):
        from taxosafe_visual.runtime import _manifest
        with tempfile.TemporaryDirectory(prefix="taxosafe_manifest_") as directory:
            root = Path(directory)
            image = root / "A" / "a1" / "one.jpg"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"fixture bytes; decoded only by GPU integration")
            manifest = root / "val_intra.txt"
            manifest.write_text("A/a1/one.jpg,0,0\n", encoding="utf-8")
            cfg = {"data": {"full_data_root": str(root), "val_intra": str(manifest)}}
            with self.assertRaisesRegex(ValueError, "known-tree species"):
                _manifest(cfg, "val_intra", toy_meta())

    def test_all_three_commands_serialize_and_keep_splits_separate(self):
        """Real pipeline/artifact execution with SYNTHETIC feature extraction.

        This exercises serialization/provenance, NOT the MaPLe GPU forward.
        """
        from taxosafe_visual import pipeline, runtime
        meta, bank, rows, f, pc, lc = toy_dataset()
        calls = []

        def fake_extract(cfg, split, model, texts, hierarchy, device):
            calls.append(split)
            if split == "train":
                selected = []
                for i, leaf in enumerate(bank["labels"]):
                    identity = "train-" + str(i)
                    selected.append({"status": "known", "true_leaf": int(leaf),
                                     "true_parent": meta["leaf_to_parent"][leaf],
                                     "source": meta["leaf_names"][leaf], "path": identity,
                                     "image_sha256": hashlib.sha256(identity.encode()).hexdigest()})
                return selected, bank["features"], np.zeros((len(selected), 3)), np.zeros((len(selected), 5))
            status = {"known": "known", "intra": "intra", "extra": "extra"}[split.split("_")[1]]
            indices = [i for i, r in enumerate(rows) if r["status"] == status]
            subset = copy.deepcopy([rows[i] for i in indices])
            for r in subset:
                r["split"] = split
                r["image_sha256"] = hashlib.sha256((split + r["image_sha256"]).encode()).hexdigest()
            return subset, f[indices], pc[indices], lc[indices]

        with tempfile.TemporaryDirectory(prefix="taxosafe_visual_test_") as directory:
            root = Path(directory)
            checkpoint = root / "best.pth"
            checkpoint.write_bytes(b"synthetic test checkpoint; not a real model")
            splits = ["train", "val_known", "val_intra", "val_extra", "test_known", "test_intra", "test_extra"]
            paths = {}
            for split in splits:
                path = root / (split + ".txt")
                path.write_text("synthetic manifest", encoding="utf-8")
                paths[split] = str(path)
            cfg = {"data": paths, "visual_support": {"primary_profile": "coverage"}}
            args = SimpleNamespace(overwrite=False, profiles="both")
            with patch.object(pipeline, "arguments", return_value=args), \
                 patch.object(pipeline, "_setup", return_value=(cfg, root, checkpoint, root)), \
                 patch.object(pipeline, "load_model", return_value=(None, None, meta, None, 100.)), \
                 patch.object(pipeline, "extract", side_effect=fake_extract), \
                 patch.object(pipeline, "extraction_signature", return_value="fixture-signature"), \
                 patch.object(runtime, "extraction_signature", return_value="fixture-signature"), \
                 contextlib.redirect_stdout(io.StringIO()):
                pipeline.build_memory()
                self.assertEqual(calls, ["train"])
                pipeline.calibrate()
                self.assertEqual(calls, ["train", "val_known", "val_intra", "val_extra"])
                pipeline.test()
                self.assertEqual(calls, splits)
                comparison = json.loads((root / "test/comparison.json").read_text())
                self.assertEqual(set(comparison["profiles"]), {"risk", "coverage"})
                for profile in ("risk", "coverage"):
                    result = json.loads((root / "test" / profile / "metrics.json").read_text())
                    self.assertEqual(result["overall"]["sample_count"], len(rows))
                    self.assertIn("open_world_accepted_leaf_precision", result["overall"])
                with self.assertRaises(FileExistsError):
                    pipeline.test()


if __name__ == "__main__":
    unittest.main()
