"""TRAIN-only local evidence: exact neighbours, sparse fallbacks and state."""
import copy
import math
import unittest
from unittest.mock import patch

import numpy as np
import torch

from taxosafe_routealign import proximity
from taxosafe_routealign.proximity import ProximityBank


META = {"leaf_names": ["a", "b", "c"], "parent_names": ["P", "Q"],
        "leaf_to_parent": [0, 0, 1]}


def fixture():
    angles = [0., 15., 30., 80., 95., 170.]
    fine = torch.tensor([[math.cos(math.radians(a)), math.sin(math.radians(a))]
                         for a in angles], dtype=torch.float64)
    parent = torch.tensor([[1., .1, 0.], [.9, .2, 0.], [1., 0., .1],
                           [1., .1, .4], [.9, .1, .5], [-1., .1, 0.]], dtype=torch.float64)
    labels = torch.tensor([0, 0, 0, 1, 1, 2])
    hashes = ["train-content-" + str(i) for i in range(len(labels))]
    return fine, parent, labels, hashes


def brute(queries, references, labels, hashes, query_hashes, k, count):
    values = torch.empty((len(queries), count), dtype=torch.float64)
    for i, query in enumerate(queries):
        for label in range(count):
            distances = [(1. - torch.dot(query, reference)).clamp(0., 2.)
                         for j, reference in enumerate(references)
                         if int(labels[j]) == label and query_hashes[i] != hashes[j]]
            if not distances:
                values[i, label] = float("inf")
            else:
                ordered = torch.stack(distances).sort().values
                values[i, label] = ordered[:min(k, len(ordered))].mean()
    return values


def aggregate(children, mapping, parents):
    result = torch.empty((len(children), parents), dtype=torch.float64)
    for i, row in enumerate(children):
        for parent in range(parents):
            values = row[torch.tensor(mapping) == parent]
            selected = values[torch.isfinite(values)].sort().values[:2]
            result[i, parent] = selected.mean() if len(selected) else float("inf")
    return result


class ProximityContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.fine, self.parent, self.labels, self.hashes = fixture()

    def fit(self, **kwargs):
        return ProximityBank.fit(self.fine, self.parent, self.labels, self.hashes, META, **kwargs)

    def test_scores_match_bruteforce_all_candidates_hash_exclusion_and_short_knn(self):
        for k in (1, 3, 99):
            with self.subTest(k=k):
                bank = self.fit(k=k)
                fine = torch.stack((self.fine[0], torch.tensor([0., 1.]), self.fine[-1]))
                parent = torch.stack((self.parent[0], torch.tensor([.95, .1, .2]), self.parent[-1]))
                query_hashes = [self.hashes[0], "unseen-content", self.hashes[-1]]
                result = bank.score(fine.requires_grad_(), parent.requires_grad_(), query_hashes)
                qf = torch.nn.functional.normalize(fine.detach(), dim=1)
                qp = torch.nn.functional.normalize(parent.detach(), dim=1)
                raw_leaf = brute(qf, bank.fine, self.labels, self.hashes, query_hashes, k, 3)
                raw_children = brute(qp, bank.parent, self.labels, self.hashes, query_hashes, k, 3)
                raw_parent = aggregate(raw_children, META["leaf_to_parent"], 2)
                for key, raw, location, scale in (("leaf_proximity", raw_leaf, bank.leaf_location, bank.leaf_scale),
                                                  ("parent_proximity", raw_parent, bank.parent_location, bank.parent_scale)):
                    raw[~torch.isfinite(raw)] = 2.
                    expected = (location - raw) / scale
                    self.assertTrue(torch.allclose(result[key], expected, atol=2e-10, rtol=1e-12), key)
                    self.assertEqual(result[key].dtype, torch.float64)
                    self.assertEqual(result[key].device.type, "cpu")
                    self.assertFalse(result[key].requires_grad)
                    self.assertTrue(bool(torch.isfinite(result[key]).all()))
                self.assertEqual(set(result), {"leaf_proximity", "parent_proximity"})

    def test_scale_statistics_use_only_true_class_leave_self_out_and_partial_pooling(self):
        bank = self.fit(k=2, shrinkage=10.)
        raw_leaf = brute(bank.fine, bank.fine, bank.labels, bank.image_hashes, bank.image_hashes, 2, 3)
        raw_parent = aggregate(brute(bank.parent, bank.parent, bank.labels,
            bank.image_hashes, bank.image_hashes, 2, 3), META["leaf_to_parent"], 2)
        parent_labels = torch.tensor(META["leaf_to_parent"])[bank.labels]
        for raw, labels, count, key in ((raw_leaf, bank.labels, 3, "leaf"), (raw_parent, parent_labels, 2, "parent")):
            own = raw[torch.arange(len(labels)), labels].numpy()
            valid = own[np.isfinite(own)]
            global_median = np.median(valid)
            global_scale = max(np.median(np.abs(valid - global_median)) * proximity.MAD_NORMALIZATION,
                               proximity.SCALE_FLOOR)
            for index in range(count):
                local = own[(labels.numpy() == index) & np.isfinite(own)]
                weight = len(local) / (len(local) + 10.) if len(local) else 0.
                median = np.median(local) if len(local) else global_median
                scale = max(np.median(np.abs(local - median)) * proximity.MAD_NORMALIZATION,
                            proximity.SCALE_FLOOR) if len(local) else global_scale
                self.assertAlmostEqual(float(getattr(bank, key + "_location")[index]),
                    weight * median + (1 - weight) * global_median, places=12)
                self.assertAlmostEqual(float(getattr(bank, key + "_scale")[index]),
                    weight * scale + (1 - weight) * global_scale, places=12)
        report = bank.fit_report
        self.assertEqual(report["fit_splits"], ["train"])
        self.assertEqual(report["leaf_train_counts"], [3, 2, 1])
        self.assertEqual(report["parent_train_counts"], [5, 1])
        self.assertEqual(report["singleton_parent_ids"], [1])
        self.assertEqual(report["single_image_leaf_ids"], [2])
        self.assertEqual(report["leaf_scales"]["groups"][2]["pooling_weight"], 0.)
        self.assertFalse(report["unknown_or_test_data_used_for_scale_fitting"])

    def test_identical_query_hashes_all_exclude_self_without_candidate_label_argument(self):
        bank = self.fit(k=1)
        qf, qp = self.fine[:1].repeat(3, 1), self.parent[:1].repeat(3, 1)
        result = bank.score(qf, qp, [self.hashes[0], self.hashes[0], "different-content"])
        self.assertTrue(torch.equal(result["leaf_proximity"][0], result["leaf_proximity"][1]))
        self.assertGreater(float(result["leaf_proximity"][2, 0]), float(result["leaf_proximity"][0, 0]))
        with self.assertRaises(TypeError):
            bank.score(qf, qp, ["a", "b", "c"], labels=torch.zeros(3))

    def test_single_image_taxonomy_and_zero_mad_are_finite_and_explicit(self):
        meta = {"leaf_names": ["a"], "parent_names": ["P"], "leaf_to_parent": [0]}
        bank = ProximityBank.fit(torch.tensor([[1., 0.]]), torch.tensor([[0., 1.]]), [0], ["only"], meta)
        self.assertTrue(bank.fit_report["leaf_scales"]["global_no_neighbour_fallback"])
        self.assertTrue(bank.fit_report["parent_scales"]["global_no_neighbour_fallback"])
        result = bank.score(torch.tensor([[1., 0.]]), torch.tensor([[0., 1.]]), ["only"])
        self.assertEqual(result["leaf_proximity"].tolist(), [[-1.]])
        self.assertEqual(result["parent_proximity"].tolist(), [[-1.]])
        repeated = ProximityBank.fit(torch.tensor([[1., 0.], [1., 0.]]),
            torch.tensor([[0., 1.], [0., 1.]]), [0, 0], ["one", "two"], meta)
        self.assertEqual(repeated.leaf_scale.tolist(), [proximity.SCALE_FLOOR])
        far = repeated.score(torch.tensor([[-1., 0.]]), torch.tensor([[0., -1.]]), ["other"])
        self.assertTrue(all(bool(torch.isfinite(value).all()) for value in far.values()))

    def test_retains_more_than_eight_rows_and_is_invariant_to_chunking_and_train_order(self):
        fine = torch.cat([self.fine + .003 * i for i in range(4)])
        parent = torch.cat([self.parent + .002 * i for i in range(4)])
        labels = self.labels.repeat(4)
        hashes = ["all-train-" + str(i) for i in range(len(labels))]
        bank = ProximityBank.fit(fine, parent, labels, hashes, META)
        self.assertEqual(len(bank.image_hashes), 24)
        self.assertEqual(bank.fit_report["leaf_train_counts"][0], 12)
        expected = bank.score(fine[:5], parent[:5], hashes[:5])
        with patch.object(proximity, "QUERY_CHUNK", 2), patch.object(proximity, "SUPPORT_CHUNK", 2):
            actual = bank.score(fine[:5], parent[:5], hashes[:5])
        order = torch.arange(len(fine) - 1, -1, -1)
        reversed_bank = ProximityBank.fit(fine[order], parent[order], labels[order],
                                          [hashes[int(i)] for i in order], META)
        reverse = reversed_bank.score(fine[:5], parent[:5], hashes[:5])
        for key in expected:
            self.assertTrue(torch.allclose(actual[key], expected[key], rtol=1e-10, atol=1e-10))
            self.assertTrue(torch.allclose(reverse[key], expected[key], rtol=1e-10, atol=1e-10))

    def test_score_never_modifies_fit_state_and_roundtrip_is_exact(self):
        bank = self.fit()
        state = bank.state_dict()
        loaded = ProximityBank.from_state_dict(state)
        expected = bank.score(self.fine, self.parent, self.hashes)
        actual = loaded.score(self.fine, self.parent, self.hashes)
        for key in expected:
            self.assertTrue(torch.equal(actual[key], expected[key]))
        for key, value in state.items():
            if torch.is_tensor(value):
                self.assertTrue(torch.equal(bank.state_dict()[key], value), key)
            else:
                self.assertEqual(bank.state_dict()[key], value, key)
        state["fine"][0, 0] = 123.
        state["fit_report"]["support_count"] = 100
        self.assertNotEqual(float(bank.fine[0, 0]), 123.)
        self.assertEqual(bank.fit_report["support_count"], 6)

    def test_state_rejects_tampered_scales_support_reports_shapes_and_schema(self):
        state = self.fit().state_dict()
        changes = [lambda s: s.update(schema_version="foreign"), lambda s: s.update(extra=True),
                   lambda s: s.pop("fine"), lambda s: s.update(k=True),
                   lambda s: s.update(shrinkage=float("nan")),
                   lambda s: s["leaf_scale"].fill_(0.),
                   lambda s: s["parent_location"].add_(.1),
                   lambda s: s.update(leaf_scale=s["leaf_scale"][:1]),
                   lambda s: s.update(fine=s["fine"].float()),
                   lambda s: s["fine"][0].zero_(),
                   lambda s: s["labels"].fill_(0),
                   lambda s: s["image_hashes"].__setitem__(0, s["image_hashes"][1]),
                   lambda s: s["fit_report"].update(support_count=99),
                   lambda s: s["meta"]["leaf_to_parent"].__setitem__(0, 1)]
        for index, change in enumerate(changes):
            with self.subTest(change=index):
                modified = copy.deepcopy(state)
                change(modified)
                with self.assertRaises(ValueError):
                    ProximityBank.from_state_dict(modified)

    def test_input_contracts_and_unfitted_use(self):
        cases = [dict(k=0), dict(k=True), dict(k=1.5), dict(shrinkage=-1.),
                 dict(shrinkage=float("inf")), dict(shrinkage=True)]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.fit(**kwargs)
        for features in (torch.zeros_like(self.fine), self.fine.long(),
                         self.fine.clone().fill_(float("nan"))):
            with self.assertRaises(ValueError):
                ProximityBank.fit(features, self.parent, self.labels, self.hashes, META)
        with self.assertRaises(ValueError):
            ProximityBank.fit(self.fine, self.parent, self.labels, [self.hashes[0]] * 6, META)
        with self.assertRaises(ValueError):
            ProximityBank.fit(self.fine[:-1], self.parent[:-1], self.labels[:-1], self.hashes[:-1], META)
        with self.assertRaises(ValueError):
            ProximityBank.fit(self.fine, self.parent, self.labels.float() + .2, self.hashes, META)
        with self.assertRaises(ValueError):
            ProximityBank().score(self.fine, self.parent, self.hashes)
        with self.assertRaises(ValueError):
            ProximityBank().state_dict()
        bank = self.fit()
        for fine, parent, hashes in ((self.fine[:, :1], self.parent, self.hashes),
                                      (self.fine, self.parent[:-1], self.hashes),
                                      (self.fine, self.parent, self.hashes[:-1])):
            with self.assertRaises(ValueError):
                bank.score(fine, parent, hashes)

    def test_empty_query_and_extreme_finite_features_are_supported(self):
        bank = self.fit()
        empty = bank.score(torch.empty((0, 2)), torch.empty((0, 3)), [])
        self.assertEqual(empty["leaf_proximity"].shape, (0, 3))
        self.assertEqual(empty["parent_proximity"].shape, (0, 2))
        large = bank.score(self.fine * 1e300, self.parent * 1e300, self.hashes)
        ordinary = bank.score(self.fine, self.parent, self.hashes)
        for key in large:
            self.assertTrue(torch.allclose(large[key], ordinary[key], rtol=1e-10, atol=1e-10))


if __name__ == "__main__":
    unittest.main()
