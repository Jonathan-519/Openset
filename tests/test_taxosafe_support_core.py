"""CPU regression tests for support conditioning and open-task gradient paths."""
import unittest

import torch
from torch.nn import functional as F

from taxosafe_support.support import SupportBank
from taxosafe_support.episodes import build_episodes
from taxosafe_support.evidence import HierarchicalEvidence, _masked_log_softmax
from taxosafe_support.losses import hierarchical_losses


MAPPING = [0, 0, 1, 1, 2]


def fixture(per_leaf=3, locals=True):
    generator = torch.Generator().manual_seed(42)
    centres = F.normalize(torch.randn(5, 8, generator=generator), dim=-1)
    labels = torch.arange(5).repeat_interleave(per_leaf)
    parent = centres[labels] + .04 * torch.randn(len(labels), 8, generator=generator)
    fine = centres[labels] + .02 * torch.randn(len(labels), 8, generator=generator)
    hashes = ["hash_%02d" % i for i in range(len(labels))]
    tokens = fine[:, None, :].expand(-1, 2, -1).clone() if locals else None
    bank = SupportBank(parent, fine, labels, hashes, MAPPING, tokens, tokens, max_per_leaf=per_leaf)
    encoded = {"parent": centres.clone().requires_grad_(), "fine": centres.clone().requires_grad_(),
               "parent_local": None, "fine_local": None}
    if locals:
        encoded["parent_local"] = centres[:, None, :].expand(-1, 2, -1).clone().requires_grad_()
        encoded["fine_local"] = centres[:, None, :].expand(-1, 2, -1).clone().requires_grad_()
    return bank, encoded


class SupportCoreTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_joint_tree_mass_and_inactive_candidates(self):
        bank, encoded = fixture()
        model = HierarchicalEvidence(8, MAPPING)
        episodes = build_episodes(torch.arange(5), MAPPING)
        for name, mask in episodes["masks"].items():
            out = model(encoded, bank, mask)
            self.assertTrue(torch.allclose(out["log_probs"].exp().sum(-1), torch.ones(5), atol=1e-6), name)
            self.assertTrue(torch.isneginf(out["log_probs"][:, 4:][~out["active_leaves"]]).all())
        empty = model(encoded, bank, torch.zeros(5, 5, dtype=torch.bool))
        self.assertTrue(torch.equal(empty["log_probs"][:, 0].exp(), torch.ones(5)))
        self.assertTrue(torch.isneginf(empty["log_probs"][:, 1:]).all())

    def test_extreme_active_score_keeps_probability_mass(self):
        logits = torch.tensor([[-20000., 0.], [0., 0.]], requires_grad=True)
        active = torch.tensor([[True, False], [False, False]])
        result = _masked_log_softmax(logits, active)
        self.assertEqual(float(result[0, 0].detach()), 0.)
        self.assertTrue(torch.isneginf(result[1]).all())
        result[0, 0].backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_query_hash_exclusion_precedes_all_statistics(self):
        bank, encoded = fixture()
        query_hashes = list(bank.hashes)[::3]
        a = bank.statistics(5, query_hashes=query_hashes)
        state = bank.state_dict()
        poisoned = state["parent"].clone()
        poisoned[::3] = 1000 * torch.eye(8)[:5]
        state["parent"] = poisoned
        poisoned_fine = state["fine"].clone()
        poisoned_fine[::3] = poisoned[::3]
        state["fine"] = poisoned_fine
        changed = SupportBank.from_state_dict(state)
        b = changed.statistics(5, query_hashes=query_hashes)
        for i in range(5):
            # Poisoned rows from other classes are still legitimate support;
            # compare the target leaf and its own excluded row explicitly.
            self.assertTrue(torch.allclose(a["fine_leaf"][i, i], b["fine_leaf"][i, i], atol=1e-6))
            self.assertFalse(bool(a["allowed"][i, i * 3]))
        one = SupportBank(torch.eye(2), torch.eye(2), [0, 1], ["x", "y"], [0, 1])
        self.assertFalse(bool(one.statistics(1, query_hashes=["x"])["leaf_active"][0, 0]))

    def test_removed_leaf_has_no_indirect_parent_or_radius_contribution(self):
        bank, encoded = fixture()
        mask = torch.ones(5, 5, dtype=torch.bool)
        mask[:, 0] = False
        a = bank.statistics(5, mask=mask)
        state = bank.state_dict()
        state["parent"][state["labels"] == 0] = torch.tensor([0., 0., 0., 0., 0., 0., 0., 1.])
        state["fine"][state["labels"] == 0] = -1
        b = SupportBank.from_state_dict(state).statistics(5, mask=mask)
        for key in ("parent_proto", "parent_scale", "fine_leaf", "leaf_scale"):
            self.assertTrue(torch.allclose(a[key], b[key], atol=1e-6), key)
        self.assertTrue(torch.allclose(a["parent_proto"][:, 0], a["parent_leaf"][:, 1], atol=1e-6))

    def test_parent_prototype_is_leaf_balanced(self):
        parent = torch.tensor([[1., 0.]] * 4 + [[0., 1.]])
        bank = SupportBank(parent, parent, [0, 0, 0, 0, 1], ["a", "b", "c", "d", "e"], [0, 0], max_per_leaf=4)
        proto = bank.statistics(1)["parent_proto"][0, 0]
        self.assertTrue(torch.allclose(proto, F.normalize(torch.ones(2), dim=0), atol=1e-6))

    def test_hash_dedup_conflict_and_checkpoint_roundtrip(self):
        bank, _ = fixture()
        loaded = SupportBank.from_state_dict(bank.state_dict())
        self.assertEqual(loaded.hashes, bank.hashes)
        self.assertTrue(torch.equal(loaded.labels, bank.labels))
        with self.assertRaisesRegex(ValueError, "conflicting"):
            SupportBank(torch.eye(2), torch.eye(2), [0, 1], ["same", "same"], [0, 1])
        duplicate = SupportBank(torch.eye(2).repeat_interleave(2, 0), torch.eye(2).repeat_interleave(2, 0),
                                [0, 0, 1, 1], ["a", "a", "b", "b"], [0, 1])
        self.assertEqual(len(duplicate.hashes), 2)

    def test_episode_singletons_and_matched_unrelated_controls(self):
        labels = torch.arange(5)
        a = build_episodes(labels, MAPPING, seed=8)
        b = build_episodes(labels, MAPPING, seed=8)
        self.assertFalse(bool(a["valid"]["drop_leaf"][4]))
        self.assertTrue(bool(a["masks"]["drop_leaf"][4].all()))
        for name in a["masks"]:
            self.assertTrue(torch.equal(a["masks"][name], b["masks"][name]))
        for i in range(5):
            self.assertTrue(bool(a["masks"]["control_leaf"][i, i]))
            self.assertTrue(bool(a["masks"]["control_parent"][i, i]))
            self.assertEqual(int((~a["masks"]["drop_parent"][i]).sum()), int((~a["masks"]["control_parent"][i]).sum()))
            if a["valid"]["drop_leaf"][i]:
                self.assertEqual(int((~a["masks"]["drop_leaf"][i]).sum()), int((~a["masks"]["control_leaf"][i]).sum()))
        impossible = build_episodes(torch.tensor([0]), [0, 0, 0, 1])
        self.assertFalse(bool(impossible["valid"]["control_parent"][0]))

    def test_open_episode_gradient_reaches_query_encoder(self):
        bank, encoded = fixture()
        model = HierarchicalEvidence(8, MAPPING)
        # Local head starts as a global prior; trainable residual must carry
        # local gradients after it has moved away from zero initialization.
        for matcher in (model.parent_matcher, model.fine_matcher):
            torch.nn.init.constant_(matcher.residual[-1].weight, .02)
        episodes = build_episodes(torch.arange(5), MAPPING)
        outputs = {name: model(encoded, bank, mask) for name, mask in episodes["masks"].items()}
        losses = hierarchical_losses(outputs, episodes, torch.arange(5), MAPPING,
                                     weights={"leaf": 0, "parent": 0, "paired": 0, "control": 0})
        losses["total"].backward()
        for key, value in encoded.items():
            self.assertIsNotNone(value.grad, key)
            self.assertTrue(torch.isfinite(value.grad).all(), key)
            self.assertGreater(float(value.grad.abs().sum()), 0, key)
        self.assertIsNone(bank.parent.grad)
        self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()) for p in model.parameters()))

    def test_singleton_acceptance_is_absolute_not_softmax_one(self):
        bank, _ = fixture(locals=False)
        model = HierarchicalEvidence(8, MAPPING, local_enabled=False)
        known = bank.fine[bank.labels == 4][0:1]
        encoded = {"parent": known, "fine": known}
        inside = model(encoded, bank)
        outside = model({"parent": -known, "fine": -known}, bank)
        self.assertGreater(float(inside["leaf_accept_logits"][0, 2].detach()), float(outside["leaf_accept_logits"][0, 2].detach()) + 1)

    def test_no_self_reference_and_empty_tree_loss_have_finite_gradients(self):
        bank = SupportBank(torch.eye(2), torch.eye(2), [0, 1], ["a", "b"], [0, 1])
        encoded = {"parent": torch.eye(2, requires_grad=True), "fine": torch.eye(2, requires_grad=True)}
        model = HierarchicalEvidence(2, [0, 1])
        episodes = build_episodes(torch.arange(2), [0, 1])
        outputs = {name: model(encoded, bank, mask, ["a", "b"]) for name, mask in episodes["masks"].items()}
        losses = hierarchical_losses(outputs, episodes, torch.arange(2), [0, 1])
        self.assertEqual(losses["valid_full_count"], 0)
        self.assertTrue(torch.isfinite(losses["total"]))
        losses["total"].backward()
        for value in encoded.values():
            self.assertTrue(torch.isfinite(value.grad).all())


if __name__ == "__main__":
    unittest.main()
