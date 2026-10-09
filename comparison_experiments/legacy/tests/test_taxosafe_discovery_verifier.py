import copy
import unittest
from unittest import mock

import torch

from taxosafe_discovery.geometry import GeometryBank
from taxosafe_discovery.verifier import build_episodes, SharedVerifier
from tests.test_taxosafe_discovery_geometry import fixture


class VerifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.args = fixture()
        cls.text = {"leaf": cls.args[0] @ torch.eye(4), "parent": cls.args[1] @ torch.eye(3)}
        cls.episodes = build_episodes(*cls.args, template_scores=cls.text)

    def test_known_train_only_disjoint_and_whole_group_holdouts(self):
        report = self.episodes["report"]
        self.assertEqual(report["fit_split"], "known_train")
        self.assertFalse(report["true_unknown_images_used"])
        self.assertFalse(report["encoder_unseen_class_claim"])
        for episode in report["episodes"]:
            self.assertEqual(episode["query_support_overlap"], 0)
            self.assertFalse(set(episode["withheld_leaf_ids"]) & set(episode["active_leaf_ids"]))
            self.assertFalse(set(episode["withheld_parent_ids"]) & set(episode["active_parent_ids"]))
        self.assertEqual(report["single_child_parent_near_examples_skipped"], 12)
        self.assertEqual(report["single_child_parent_ids"], [1, 2])

    def test_rank_pairs_match_same_query_and_target_direction(self):
        for level in ("leaf", "parent"):
            data = self.episodes[level]
            pair = data["ranking_pairs"]
            self.assertTrue(torch.equal(data["query_index"][pair[:, 0]], data["query_index"][pair[:, 1]]))
            self.assertTrue(bool((data["y"][pair[:, 0]] == 1).all()))
            self.assertTrue(bool((data["y"][pair[:, 1]] == 0).all()))
            self.assertAlmostEqual(float(data["weight"][data["y"] == 1].sum()), .5, places=6)
            self.assertAlmostEqual(float(data["weight"][data["y"] == 0].sum()), .5, places=6)

    def test_bce_and_rank_are_actual_matched_budget_training(self):
        common = dict(seed=4, epochs=2, batch_size=64, lr=.002)
        bce = SharedVerifier.fit(self.episodes, loss="bce", **common)
        rank = SharedVerifier.fit(self.episodes, loss="bce_rank", **common)
        self.assertEqual(bce.fit_report["initial_parameter_sha256"], rank.fit_report["initial_parameter_sha256"])
        self.assertEqual(bce.fit_report["optimizer_steps"], rank.fit_report["optimizer_steps"])
        self.assertGreater(bce.fit_report["optimizer_steps"], 0)
        self.assertTrue(all(value > 0 for value in bce.fit_report["parameter_delta_l2"].values()))
        self.assertTrue(all(value > 0 for value in rank.fit_report["parameter_delta_l2"].values()))
        self.assertEqual(bce.fit_report["history"]["leaf"][0]["ranking"], 0.)
        self.assertGreater(rank.fit_report["history"]["leaf"][0]["ranking"], 0.)
        self.assertFalse(torch.equal(bce.heads["leaf"].layers[0].weight, rank.heads["leaf"].layers[0].weight))

    def test_scoring_roundtrip_and_candidate_count_preserved(self):
        model = SharedVerifier.fit(self.episodes, epochs=1, batch_size=128)
        bank = GeometryBank.fit(*self.args)
        geometric = bank.score(self.args[0][:2], self.args[1][:2], ["query-1", "query-2"])
        text = {k: v[:2] for k, v in self.text.items()}
        scores = model.score(geometric, text)
        loaded = SharedVerifier.from_state_dict(model.state_dict())
        other = loaded.score(geometric, text)
        self.assertEqual(scores["leaf_scores"].shape, (2, 4))
        self.assertEqual(scores["parent_scores"].shape, (2, 3))
        self.assertTrue(all(torch.equal(scores[k], other[k]) for k in scores))
        self.assertFalse(any(p.requires_grad for h in loaded.heads.values() for p in h.parameters()))
        with self.assertRaises(ValueError):
            loaded.score(geometric)

    def test_state_rejects_nonfinite_and_shape_corruption(self):
        model = SharedVerifier.fit(self.episodes, epochs=1, batch_size=256)
        state = model.state_dict()
        state["normalization"]["leaf"]["scale"][0] = 0
        with self.assertRaises(ValueError):
            SharedVerifier.from_state_dict(state)
        state = model.state_dict()
        state["heads"]["leaf"]["layers.0.weight"][0, 0] = float("nan")
        with self.assertRaises(ValueError):
            SharedVerifier.from_state_dict(state)

    def test_no_real_unknown_or_test_fit_allowed(self):
        for flag in ("true_unknown_images_used", "dev_images_used", "test_images_used"):
            episodes = copy.deepcopy(self.episodes)
            episodes["report"][flag] = True
            with self.assertRaises(ValueError):
                SharedVerifier.fit(episodes, epochs=1)

    def test_missing_class_or_duplicate_hash_not_silently_used(self):
        fine, parent, labels, hashes, meta = self.args
        with self.assertRaises(ValueError):
            build_episodes(fine[:1], parent[:1], labels[:1], hashes[:1], meta)
        with self.assertRaises(ValueError):
            build_episodes(fine, parent, labels, [hashes[0]] * len(hashes), meta)


if __name__ == "__main__":
    unittest.main()
