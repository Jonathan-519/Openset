"""Backward compatibility and candidate-specific membership contracts."""
import unittest

import torch
from torch import nn

from taxosafe_support.episodes import build_episodes
from taxosafe_support.evidence import HierarchicalEvidence
from taxosafe_support.losses import hierarchical_losses, representation_losses, _balanced_candidate_bce
from taxosafe_support.support import SupportBank
from tests.test_taxosafe_support_core import fixture, MAPPING


class FixedScores(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.register_buffer("values", torch.tensor(values, dtype=torch.float32))

    def forward(self, features):
        return self.values[None].expand(len(features), -1)


class DecoupledEvidenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_legacy_state_keys_strict_load_and_numeric_golden(self):
        bank, encoded = fixture(locals=False)
        torch.manual_seed(123)
        legacy = HierarchicalEvidence(8, MAPPING)
        expected_keys = {"root_bias", "local_bias", "leaf_to_parent"}
        expected_keys |= {"%s.residual.%s.%s" % (head, layer, parameter)
                          for head in ("parent_matcher", "fine_matcher")
                          for layer in ("0", "2") for parameter in ("weight", "bias")}
        self.assertEqual(set(legacy.state_dict()), expected_keys)
        explicit = HierarchicalEvidence(8, MAPPING, decoupled=False)
        explicit.load_state_dict(legacy.state_dict(), strict=True)
        for key, value in legacy(encoded, bank).items():
            self.assertTrue(torch.equal(value, explicit(encoded, bank)[key]), key)
        # Captured from the pre-decoupling implementation at new/1d7e4e4.
        golden = torch.tensor([
            [-2.9835500717, -5.0575685501, -7.0584712029, -9.9865207672, -.0604709312,
             -7.7980918884, -14.7668666840, -7.8545165062, -16.9084548950],
            [-2.8664548397, -5.0594763756, -8.8588666916, -9.0074701309, -7.6717944145,
             -.0661198124, -15.5077352524, -13.4887723923, -15.3314189911]])
        self.assertTrue(torch.allclose(legacy(encoded, bank)["log_probs"][:2], golden, atol=2e-6, rtol=0))
        self.assertNotIn("required_leaf_mask", bank.state_dict())

    def test_decoupled_mass_masks_and_empty_tree_backward(self):
        bank, encoded = fixture()
        model = HierarchicalEvidence(8, MAPPING, decoupled=True)
        episodes = build_episodes(torch.arange(5), MAPPING)
        for mask in list(episodes["masks"].values()) + [torch.zeros(5, 5, dtype=torch.bool)]:
            out = model(encoded, bank, mask)
            self.assertTrue(torch.allclose(out["log_probs"].exp().sum(-1), torch.ones(5), atol=1e-6))
            self.assertTrue(torch.isneginf(out["leaf_membership_logits"][~out["active_leaves"]]).all())
        empty = model(encoded, bank, torch.zeros(5, 5, dtype=torch.bool))
        self.assertTrue(torch.equal(empty["log_probs"][:, 0], torch.zeros(5)))
        self.assertTrue(torch.isneginf(empty["log_probs"][:, 1:]).all())
        (-empty["log_probs"][:, 0].mean()).backward()
        for value in encoded.values():
            self.assertTrue(torch.isfinite(value.grad).all())

    def test_candidate_membership_cannot_be_borrowed_from_other_parent(self):
        bank, encoded = fixture()
        model = HierarchicalEvidence(8, MAPPING, decoupled=True)
        model.parent_matcher = FixedScores([-20, 20, -20])
        model.parent_membership = FixedScores([20, -20, 20])
        model.fine_membership = FixedScores([20] * 5)
        out = model(encoded, bank)
        self.assertGreater(float(out["log_probs"][0, 0].exp().detach()), .9999)
        self.assertLess(float(out["log_probs"][0, 6:8].exp().sum().detach()), 1e-7)

    def test_candidate_membership_cannot_be_borrowed_from_sibling(self):
        bank, encoded = fixture()
        model = HierarchicalEvidence(8, MAPPING, decoupled=True)
        model.parent_matcher = FixedScores([20, -20, -20])
        model.parent_membership = FixedScores([20, 20, 20])
        model.fine_matcher = FixedScores([-20, 20, 0, 0, 0])
        model.fine_membership = FixedScores([20, -20, 20, 20, 20])
        out = model(encoded, bank)
        self.assertGreater(float(out["log_probs"][0, 1].exp().detach()), .9999)
        self.assertLess(float(out["log_probs"][0, 5].exp().detach()), 1e-7)

    def test_rank_parameters_do_not_change_membership_logits(self):
        bank, encoded = fixture()
        model = HierarchicalEvidence(8, MAPPING, decoupled=True)
        before = model(encoded, bank)
        with torch.no_grad():
            for matcher in (model.parent_matcher, model.fine_matcher):
                matcher.residual[-1].weight.fill_(10)
        after = model(encoded, bank)
        for key in ("parent_membership_logits", "leaf_membership_logits"):
            self.assertTrue(torch.equal(before[key], after[key]), key)
        self.assertFalse(torch.equal(before["leaf_logits"], after["leaf_logits"]))

    def test_singleton_membership_remains_absolute_and_extremes_stable(self):
        bank, encoded = fixture()
        model = HierarchicalEvidence(8, MAPPING, decoupled=True)
        model.parent_membership = FixedScores([10000, -10000, 10000])
        model.fine_membership = FixedScores([-10000, 10000, -10000, 10000, -10000])
        out = model(encoded, bank)
        self.assertTrue(torch.isfinite(out["root_logit"]).all())
        self.assertTrue(torch.isfinite(out["leaf_accept_logits"]).all())
        self.assertTrue(torch.equal(out["leaf_accept_logits"][:, 2], torch.full((5,), -10000.)))
        self.assertTrue(torch.allclose(out["log_probs"].exp().sum(-1), torch.ones(5), atol=1e-6))

    def test_all_candidate_bce_tracks_removal_and_trains_shared_biases(self):
        bank, encoded = fixture()
        model = HierarchicalEvidence(8, MAPPING, decoupled=True)
        episodes = build_episodes(torch.arange(5), MAPPING)
        outputs = {name: model(encoded, bank, mask) for name, mask in episodes["masks"].items()}
        for output in outputs.values():
            output["parent_membership_logits"].retain_grad()
            output["leaf_membership_logits"].retain_grad()
        losses = hierarchical_losses(outputs, episodes, torch.arange(5), MAPPING)
        (losses["membership_parent"] + losses["membership_leaf"]).backward()
        near, removed = outputs["drop_leaf"], outputs["drop_parent"]
        rows, parents = torch.arange(4), torch.tensor(MAPPING[:4])
        self.assertTrue((near["parent_membership_logits"].grad[rows, parents] < 0).all())
        self.assertTrue((near["leaf_membership_logits"].grad[:4][near["active_leaves"][:4]] > 0).all())
        self.assertTrue((removed["parent_membership_logits"].grad[removed["active_parents"]] > 0).all())
        self.assertEqual(float(near["leaf_membership_logits"].grad[4].abs().sum()), 0.)
        self.assertIsNotNone(model.root_bias.grad)
        self.assertIsNotNone(model.local_bias.grad)
        self.assertGreater(float(model.root_bias.grad.abs()), 0.)
        self.assertGreater(float(model.local_bias.grad.abs()), 0.)

    def test_balanced_bce_is_not_diluted_by_negative_candidate_count(self):
        a = torch.tensor([[2., -1.]], requires_grad=True)
        b = torch.tensor([[2., -1., -1., -1.]], requires_grad=True)
        loss_a = _balanced_candidate_bce(a, torch.tensor([[True, False]]), torch.ones_like(a).bool(),
                                        torch.tensor([True]), a.sum() * 0)
        loss_b = _balanced_candidate_bce(b, torch.tensor([[True, False, False, False]]), torch.ones_like(b).bool(),
                                        torch.tensor([True]), b.sum() * 0)
        self.assertTrue(torch.allclose(loss_a, loss_b))

    def test_own_hash_no_false_positive_and_gradients_reach_queries(self):
        bank = SupportBank(torch.eye(2), torch.eye(2), [0, 1], ["a", "b"], [0, 1])
        model = HierarchicalEvidence(2, [0, 1], decoupled=True)
        encoded = {"parent": torch.eye(2, requires_grad=True), "fine": torch.eye(2, requires_grad=True)}
        episodes = build_episodes(torch.arange(2), [0, 1])
        outputs = {n: model(encoded, bank, m, ["a", "b"]) for n, m in episodes["masks"].items()}
        outputs["full"]["leaf_membership_logits"].retain_grad()
        losses = hierarchical_losses(outputs, episodes, torch.arange(2), [0, 1])
        losses["total"].backward()
        self.assertEqual(losses["valid_full_count"], 0)
        self.assertTrue(torch.isfinite(losses["total"]))
        for value in encoded.values():
            self.assertTrue(torch.isfinite(value.grad).all())
        gradient = outputs["full"]["leaf_membership_logits"].grad
        self.assertEqual(float(gradient.diag().abs().sum()), 0.)
        self.assertTrue((gradient[outputs["full"]["active_leaves"]] > 0).all())

    def test_representation_pair_roles_counts_and_alias_exclusion(self):
        labels = torch.arange(5).repeat_interleave(2)
        encoded = {"parent": torch.randn(10, 8, requires_grad=True),
                   "fine": torch.randn(10, 8, requires_grad=True)}
        terms = representation_losses(encoded, labels, MAPPING, [str(i) for i in range(10)])
        self.assertEqual(terms["valid_parent_anchors"], 8)
        self.assertEqual(terms["valid_parent_pairs"], 16)
        self.assertEqual(terms["valid_leaf_anchors"], 8)
        self.assertEqual(terms["valid_leaf_pairs"], 8)
        (terms["parent_cross_species"] + terms["leaf_sibling"]).backward()
        for value in encoded.values():
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(float(value.grad.abs().sum()), 0.)
        aliases = representation_losses(encoded, labels, MAPPING, ["same", "same"] + [str(i) for i in range(2, 10)])
        self.assertEqual(aliases["valid_leaf_anchors"], 6)
        empty = representation_losses({"parent": encoded["parent"][:2], "fine": encoded["fine"][:2]},
                                      labels[:2], MAPPING, ["same", "same"])
        self.assertEqual(empty["valid_parent_pairs"], 0)
        self.assertEqual(empty["valid_leaf_pairs"], 0)
        self.assertEqual(float(empty["parent_cross_species"].detach()), 0.)
        self.assertEqual(float(empty["leaf_sibling"].detach()), 0.)

    def test_explicit_holdout_bank_rejects_forbidden_or_missing_eligible_leaf(self):
        with self.assertRaisesRegex(ValueError, "Missing training"):
            SupportBank(torch.eye(2), torch.eye(2), [0, 2], ["a", "b"], [0, 0, 1])
        bank = SupportBank(torch.eye(2), torch.eye(2), [0, 2], ["a", "b"], [0, 0, 1],
                           required_leaf_mask=[True, False, True])
        loaded = SupportBank.from_state_dict(bank.state_dict())
        self.assertTrue(torch.equal(loaded.required_leaf_mask, torch.tensor([True, False, True])))
        self.assertFalse(bool(loaded.statistics(1)["leaf_active"][0, 1]))
        with self.assertRaisesRegex(ValueError, "excluded holdout"):
            SupportBank(torch.eye(2), torch.eye(2), [0, 1], ["a", "b"], [0, 0, 1],
                        required_leaf_mask=[True, False, True])


if __name__ == "__main__":
    unittest.main()
