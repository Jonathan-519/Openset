import copy
import unittest
from unittest import mock

import torch

from taxosafe_discovery.geometry import GeometryBank
from taxosafe_discovery.verifier import SharedVerifier, build_episodes
from taxosafe_recovery.training import finetune, make_leaf_guard, RecoveryVerifier, state_hash
from tests.test_taxosafe_discovery_geometry import fixture


class RecoveryTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.args = fixture()
        fine, parent, labels, hashes, meta = cls.args
        cls.templates = {"leaf": fine @ torch.eye(4), "parent": parent @ torch.eye(3)}
        cls.episodes = build_episodes(*cls.args, template_scores=cls.templates)
        cls.source = SharedVerifier.fit(cls.episodes, loss="bce", epochs=2, batch_size=64).state_dict()
        cls.candidates = {"leaf": labels.clone(), "parent": torch.tensor(meta["leaf_to_parent"])[labels]}
        # The production candidate for this query is wrong: its true class must
        # not receive the new correct-production-candidate positive emphasis.
        cls.candidates["leaf"][0] = 1
        cls.candidates["parent"][0] = 2
        cls.options = dict(steps_per_head=4, batch_size=32)

    def train(self, mode):
        return finetune(self.source, self.episodes, self.candidates,
                        mode=mode, options=self.options, seed=9)

    def test_real_warm_start_updates_preserve_source_norm_and_rng(self):
        original = state_hash(self.source)
        torch.manual_seed(67)
        random_before = torch.get_rng_state().clone()
        state, report = self.train("hard")
        self.assertTrue(torch.equal(torch.get_rng_state(), random_before))
        self.assertEqual(state_hash(self.source), original)
        self.assertEqual(state_hash(state["source_state"]), original)
        self.assertEqual(report["source_normalization_sha256"], report["normalization_sha256"])
        self.assertEqual(report["optimizer_steps"], 8)
        self.assertTrue(report["warm_start_no_optimizer_resume"])
        self.assertFalse(report["optimizer_state_resumed"])
        self.assertTrue(report["gradients_all_finite"])
        self.assertTrue(all(value > 0 for value in report["parameter_delta_l2"].values()))
        self.assertTrue(all(value > 0 for value in report["changed_tensor_count"].values()))
        self.assertEqual(state["source_state"]["fit_report"], self.source["fit_report"])
        for level in ("leaf", "parent"):
            self.assertEqual(report["levels"][level]["focus_count"], 23)
            self.assertGreater(report["levels"][level]["score_change"]["std"], 0.)
            self.assertEqual(report["source_head_sha256"][level], state_hash(self.source["heads"][level]))

    def test_nested_losses_share_original_data_batches_and_budget(self):
        reports = {mode: self.train(mode)[1] for mode in ("bce", "hard", "anchor", "l2sp")}
        for level in ("leaf", "parent"):
            self.assertEqual(len({r["levels"][level]["base_batch_order_sha256"] for r in reports.values()}), 1)
            self.assertEqual(len({r["levels"][level]["base_weights_sha256"] for r in reports.values()}), 1)
            self.assertEqual(len({r["levels"][level]["teacher_scores_sha256"] for r in reports.values()}), 1)
            self.assertEqual(len({r["source_head_sha256"][level] for r in reports.values()}), 1)
            for mode, report in reports.items():
                history = report["history"][level]
                self.assertEqual(len(history), 4)
                self.assertTrue(all(step["base_bce"] > 0 for step in history))
                self.assertTrue(all(step["gradient_norm_before_clip"] > 0 for step in history))
                self.assertTrue(all(step["gradients_finite"] for step in history))
                if mode == "bce":
                    self.assertTrue(all(step["hard_positive_bce"] == step["negative_anchor"] == step["l2sp"] == 0. for step in history))
                else:
                    self.assertTrue(all(step["hard_positive_bce"] > 0 for step in history))
                if mode not in ("anchor", "l2sp"):
                    self.assertTrue(all(step["negative_anchor"] == 0 for step in history))
                if mode == "l2sp":
                    self.assertEqual(history[0]["l2sp"], 0.)
                    self.assertGreater(history[-1]["l2sp"], 0.)
                else:
                    self.assertTrue(all(step["l2sp"] == 0 for step in history))

    def test_inference_reload_never_calls_fit_and_is_exact(self):
        state, _ = self.train("l2sp")
        bank = GeometryBank.fit(*self.args)
        evidence = bank.score(self.args[0][:2], self.args[1][:2], ["query-a", "query-b"])
        text = {key: value[:2] for key, value in self.templates.items()}
        first = RecoveryVerifier.from_state_dict(state).score(evidence, text)
        with mock.patch.object(SharedVerifier, "fit", side_effect=AssertionError("TEST trained")), \
                mock.patch.object(GeometryBank, "fit", side_effect=AssertionError("TEST refitted geometry")), \
                mock.patch("taxosafe_recovery.training.finetune", side_effect=AssertionError("TEST fine-tuned")):
            loaded = RecoveryVerifier.from_state_dict(state)
            second = loaded.score(evidence, text)
        self.assertTrue(all(torch.equal(first[key], second[key]) for key in first))
        self.assertEqual(state_hash(loaded.state_dict()), state_hash(state))

    def test_leaf_guard_keeps_recovered_leaf_and_exact_original_parent(self):
        state, source_report = self.train("l2sp")
        guard, report = make_leaf_guard(state, self.source)
        self.assertEqual(report["optimizer_steps"], 0)
        self.assertEqual(report["reused_optimizer_steps"], source_report["optimizer_steps"])
        self.assertEqual(state_hash(guard["updated_heads"]["leaf"]), state_hash(state["updated_heads"]["leaf"]))
        self.assertEqual(state_hash(guard["updated_heads"]["parent"]), state_hash(self.source["heads"]["parent"]))
        self.assertEqual(report["parameter_delta_l2"]["parent"], 0.)
        bank = GeometryBank.fit(*self.args)
        evidence = bank.score(self.args[0][:2], self.args[1][:2], ["query-a", "query-b"])
        text = {key: value[:2] for key, value in self.templates.items()}
        guarded = RecoveryVerifier.from_state_dict(guard).score(evidence, text)
        original = SharedVerifier.from_state_dict(self.source).score(evidence, text)
        changed = RecoveryVerifier.from_state_dict(state).score(evidence, text)
        self.assertTrue(torch.equal(guarded["parent_scores"], original["parent_scores"]))
        self.assertTrue(torch.equal(guarded["leaf_scores"], changed["leaf_scores"]))
        non_l2sp, _ = self.train("hard")
        with self.assertRaises(ValueError):
            make_leaf_guard(non_l2sp, self.source)

    def test_unknown_dev_test_data_and_changed_train_identity_rejected(self):
        for key in ("true_unknown_images_used", "dev_images_used", "test_images_used"):
            episodes = copy.deepcopy(self.episodes)
            episodes["report"][key] = True
            with self.assertRaises(ValueError):
                finetune(self.source, episodes, self.candidates, mode="bce", options=self.options)
        episodes = copy.deepcopy(self.episodes)
        episodes["report"]["image_hash_digest"] = "different train"
        with self.assertRaises(ValueError):
            finetune(self.source, episodes, self.candidates, mode="hard", options=self.options)
        episodes = copy.deepcopy(self.episodes)
        episodes["leaf"]["x"][0, 0] += .01
        with self.assertRaises(ValueError):
            finetune(self.source, episodes, self.candidates, mode="hard", options=self.options)

    def test_candidate_dtype_shape_and_bounds_are_strict(self):
        for value in (self.candidates["leaf"].float(), self.candidates["leaf"][:-1],
                      torch.full_like(self.candidates["leaf"], 4)):
            candidates = dict(self.candidates, leaf=value)
            with self.assertRaises(ValueError):
                finetune(self.source, self.episodes, candidates, mode="hard", options=self.options)

    def test_tampered_norm_heads_and_original_report_rejected(self):
        state, _ = self.train("anchor")
        for kind in ("norm", "head", "source_report"):
            changed = copy.deepcopy(state)
            if kind == "norm":
                changed["source_state"]["normalization"]["leaf"]["scale"][0] *= 1.1
            elif kind == "head":
                changed["updated_heads"]["leaf"]["layers.0.weight"][0, 0] = float("nan")
            else:
                changed["source_state"]["fit_report"]["loss"] = "bce_rank"
            # Even a recomputed outer hash cannot bypass source/head binding.
            changed["state_sha256"] = state_hash({k: v for k, v in changed.items() if k != "state_sha256"})
            with self.assertRaises(ValueError):
                RecoveryVerifier.from_state_dict(changed)


if __name__ == "__main__":
    unittest.main()
