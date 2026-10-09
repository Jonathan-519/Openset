"""Tests of the complete-TRAIN, frozen local evidence contract."""
import copy
import unittest

import torch

from taxosafe_evidence_guard import geometry


def fixture():
    meta = {"leaf_names": ["leaf_a", "leaf_b", "leaf_c"],
            "parent_names": ["parent_a", "parent_b"], "leaf_to_parent": [0, 0, 1]}
    fine = torch.tensor([[1., 0., 0.], [.99, .12, 0.], [.8, .6, 0.],
                         [.3, .95, 0.], [.1, .99, 0.], [0., 0., 1.], [.1, 0., .99]])
    parent = torch.tensor([[1., 0., 0.], [.99, .1, 0.], [.8, .6, 0.],
                           [.9, .4, 0.], [.8, .6, 0.], [0., 0., 1.], [.1, 0., .99]])
    labels = [0, 0, 0, 1, 1, 2, 2]
    hashes = ["image_" + str(i) for i in range(len(labels))]
    records = [{"status": "known", "split": "train", "true_leaf": label,
                "true_parent": meta["leaf_to_parent"][label], "image_sha256": hashes[i]}
               for i, label in enumerate(labels)]
    group = {"records": records, "image_sha256": hashes,
             "encoded": {"fine": fine, "parent": parent}}
    return group, meta, {"neighbors": 5, "shrinkage": 5.0}


class GeometryContracts(unittest.TestCase):
    def test_complete_bank_and_hash_self_exclusion(self):
        group, meta, cfg = fixture()
        state = geometry.fit(group, meta, cfg)
        self.assertEqual(state["image_sha256"], group["image_sha256"])
        self.assertEqual(len(state["fine"]), 7)
        query = {key: value[:1] for key, value in group["encoded"].items()}
        no_self = geometry.score(query, ["image_0"], state)
        other_image = geometry.score(query, ["novel_image"], state)
        # The identical stored vector must disappear only when its SHA matches.
        self.assertLess(float(no_self["leaf"][0, 0, 1]), -1e-3)
        self.assertAlmostEqual(float(other_image["leaf"][0, 0, 1]), 0.0, places=5)
        self.assertEqual(tuple(no_self["leaf"].shape), (1, 3, 6))
        self.assertEqual(tuple(no_self["parent"].shape), (1, 2, 6))
        self.assertTrue(all(value.device.type == "cpu" for value in state.values() if isinstance(value, torch.Tensor)))
        # Independent manual mean: two nonself vectors of leaf 0, not k=5 nor n=3.
        bank = state["fine"][state["labels"] == 0]
        d = (1. - state["fine"][0] @ bank[1:].T).clamp(0., 2.).mean()
        expected = float((-d / state["fine_scale"][0]).clamp(-20., 20.))
        self.assertAlmostEqual(float(no_self["leaf"][0, 0, 0]), expected, places=5)

    def test_small_identical_class_has_finite_shrunk_scale(self):
        group, meta, cfg = fixture()
        group["encoded"]["fine"][6] = group["encoded"]["fine"][5]
        group["encoded"]["parent"][6] = group["encoded"]["parent"][5]
        state = geometry.fit(group, meta, cfg)
        self.assertAlmostEqual(float(state["fine_raw_loo_scale"][2]), 0., places=6)
        self.assertGreater(float(state["fine_scale"][2]), geometry.SCALE_FLOOR)
        self.assertAlmostEqual(float(state["fine_global_scale"]),
                               float(state["fine_raw_loo_scale"].mean()), places=6)
        result = geometry.score(group["encoded"], group["image_sha256"], state)
        for value in result.values():
            self.assertTrue(bool(torch.isfinite(value).all()))
            self.assertLessEqual(float(value.abs().max()), 20.)
        # Both images of the rare leaf are queried; k shrinks from 5 to 1.
        self.assertAlmostEqual(float(result["leaf"][5, 2, 4]), 0., places=6)

    def test_rejects_nontraining_or_unknown_statistics(self):
        for field, value in [("split", "val_known"), ("split", "test_known"),
                             ("split", "train_intra"), ("status", "intra"), ("status", "extra")]:
            group, meta, cfg = fixture()
            group["records"][0][field] = value
            with self.assertRaisesRegex(ValueError, "known TRAIN only"):
                geometry.fit(group, meta, cfg)
        group, meta, cfg = fixture()
        group["image_sha256"][1] = group["image_sha256"][0]
        with self.assertRaisesRegex(ValueError, "unique image"):
            geometry.fit(group, meta, cfg)
        group, meta, cfg = fixture()
        for key in group["encoded"]:
            group["encoded"][key] = group["encoded"][key][:-1]
        group["records"] = group["records"][:-1]
        group["image_sha256"] = group["image_sha256"][:-1]
        with self.assertRaisesRegex(ValueError, "at least two"):
            geometry.fit(group, meta, cfg)

    def test_query_truth_ignored_and_bank_frozen(self):
        group, meta, cfg = fixture()
        state = geometry.fit(group, meta, cfg)
        before = copy.deepcopy(state)
        query = copy.deepcopy(group["encoded"])
        hashes = ["unseen_" + str(i) for i in range(7)]
        expected = geometry.score(query, hashes, state)
        query["records"] = [{"status": "extra", "true_parent": 123,
                             "true_leaf": 456, "source": "test_species"}] * 7
        query["fine"].requires_grad_()
        actual = geometry.score(query, hashes, state)
        for name in expected:
            self.assertTrue(torch.equal(expected[name], actual[name]))
            self.assertFalse(actual[name].requires_grad)
        for name, value in before.items():
            if isinstance(value, torch.Tensor):
                self.assertTrue(torch.equal(value, state[name]))
            else:
                self.assertEqual(value, state[name])

    def test_batch_and_sequential_queries_agree(self):
        group, meta, cfg = fixture()
        state = geometry.fit(group, meta, cfg)
        batched = geometry.score(group["encoded"], group["image_sha256"], state)
        sequential = []
        for index, sha in enumerate(group["image_sha256"]):
            query = {key: value[index:index + 1] for key, value in group["encoded"].items()}
            sequential.append(geometry.score(query, [sha], state))
        for key in batched:
            torch.testing.assert_close(batched[key], torch.cat([value[key] for value in sequential]),
                                       atol=2e-5, rtol=2e-5)

    def test_parent_pooling_weights_leaves_equally(self):
        group, meta, cfg = fixture()
        state = geometry.fit(group, meta, cfg)
        query = {key: value[:1] for key, value in group["encoded"].items()}
        result = geometry.score(query, ["novel"], state)
        q = torch.nn.functional.normalize(query["parent"], dim=1)
        means = []
        for leaf in [0, 1]:
            points = state["parent"][state["labels"] == leaf]
            means.append(((1. - q @ points.T).clamp(0., 2.).mean() / state["parent_scale"][leaf]))
        expected = -torch.stack(means).mean()
        sample_weighted = -(means[0] * 3 + means[1] * 2) / 5
        self.assertGreater(abs(float(expected - sample_weighted)), 1e-3)
        self.assertAlmostEqual(float(result["parent"][0, 0, 0]), float(expected), places=5)
        self.assertAlmostEqual(float(result["leaf"][0, 0, 5]), float(expected), places=5)
        self.assertAlmostEqual(float(result["leaf"][0, 1, 5]), float(expected), places=5)


if __name__ == "__main__":
    unittest.main()
