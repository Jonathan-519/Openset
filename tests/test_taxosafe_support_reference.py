"""Absolute reference verification, known-only pair roles, and compatibility."""
import copy
import unittest

import torch
from torch.nn import functional as F

from taxosafe_support.episodes import build_episodes
from taxosafe_support.evidence import HierarchicalEvidence
from taxosafe_support.losses import hierarchical_losses, reference_pair_losses, reference_supervision_counts
from taxosafe_support.reference import aggregate_reference_logits, bidirectional_local_coverage
from taxosafe_support.support import SupportBank
from tests.test_taxosafe_support_core import MAPPING, fixture


class ReferenceEvidenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def model(self, dimension=8, mapping=MAPPING, **kwargs):
        return HierarchicalEvidence(dimension, mapping, decoupled=True, membership_mode="reference", **kwargs)

    def test_opt_in_preserves_default_modules_and_initialization(self):
        for decoupled in (False, True):
            torch.manual_seed(56)
            default = HierarchicalEvidence(8, MAPPING, decoupled=decoupled)
            torch.manual_seed(56)
            explicit = HierarchicalEvidence(8, MAPPING, decoupled=decoupled, membership_mode="prototype")
            self.assertEqual(set(default.state_dict()), set(explicit.state_dict()))
            for key, value in default.state_dict().items():
                self.assertTrue(torch.equal(value, explicit.state_dict()[key]), key)
        reference = self.model()
        self.assertFalse(hasattr(reference, "parent_membership"))
        self.assertFalse(hasattr(reference, "fine_membership"))
        self.assertTrue(hasattr(reference, "parent_reference"))
        with self.assertRaisesRegex(ValueError, "requires decoupled"):
            HierarchicalEvidence(8, MAPPING, membership_mode="reference")

    def test_real_witnesses_reject_hollow_centroid_at_initialization(self):
        modes = F.normalize(torch.tensor([[.4, .9165], [.4, -.9165]]), dim=-1)
        refs = modes.repeat_interleave(2, 0)
        bank = SupportBank(refs, refs, [0] * 4, ["a", "b", "c", "d"], [0], max_per_leaf=4)
        q = torch.stack((torch.tensor([1., 0.]), modes[0]))
        encoded = {"parent": q, "fine": q}
        prototype = HierarchicalEvidence(2, [0], decoupled=True)(encoded, bank)
        reference = self.model(2, [0])(encoded, bank)
        p, r = prototype["leaf_membership_logits"][:, 0], reference["leaf_membership_logits"][:, 0]
        self.assertGreater(float(p[0].detach()), float(p[1].detach()))
        self.assertLess(float(r[0].detach()), 0.)
        self.assertGreater(float(r[1].detach()), 4.9)

    def test_two_local_directions_detect_missing_coverage(self):
        query = torch.tensor([[[1., 0.]]])
        references = torch.eye(2)[None]
        forward, backward = bidirectional_local_coverage(query, references, torch.zeros(1, 1))
        self.assertEqual(float(forward), 1.)
        self.assertEqual(float(backward), .5)
        reverse = bidirectional_local_coverage(references, query, torch.zeros(1, 1))
        self.assertTrue(torch.equal(forward, reverse[1]))
        self.assertTrue(torch.equal(backward, reverse[0]))
        fallback = torch.tensor([[.37]])
        self.assertTrue(all(torch.equal(x, fallback) for x in bidirectional_local_coverage(None, references, fallback)))

    def test_topk_is_masked_permutation_invariant_and_parent_leaf_balanced(self):
        labels, mapping = torch.tensor([0, 0, 0, 1, 1, 2]), torch.tensor([0, 0, 1])
        scores = torch.tensor([[100., 2., 4., -2., -2., 8.]], requires_grad=True)
        allowed = torch.tensor([[False, True, True, True, True, True]])
        parent, leaf = aggregate_reference_logits(scores, scores, allowed, labels, mapping)
        self.assertTrue(torch.equal(leaf, torch.tensor([[3., -2., 8.]])))
        self.assertTrue(torch.equal(parent, torch.tensor([[.5, 8.]])))
        permutation = torch.tensor([3, 0, 5, 2, 1, 4])
        permuted = aggregate_reference_logits(scores[:, permutation], scores[:, permutation], allowed[:, permutation],
                                              labels[permutation], mapping)
        self.assertTrue(torch.equal(parent, permuted[0]))
        self.assertTrue(torch.equal(leaf, permuted[1]))
        (parent.sum() + leaf.sum()).backward()
        self.assertEqual(float(scores.grad[0, 0]), 0.)
        self.assertTrue(torch.isfinite(scores.grad).all())

    def test_pair_scores_do_not_depend_on_other_references_or_identity_heads(self):
        bank, encoded = fixture()
        model = self.model()
        before = model(encoded, bank)
        state = bank.state_dict()
        state["parent"][bank.labels == 1] *= -1
        state["fine"][bank.labels == 1] *= -1
        changed = SupportBank.from_state_dict(state)
        after = model(encoded, changed)
        keep = bank.labels != 1
        # Reconstructing a bank renormalizes cached float features once.
        self.assertTrue(torch.allclose(before["reference_parent_logits"][:, keep], after["reference_parent_logits"][:, keep], atol=2e-6))
        self.assertTrue(torch.allclose(before["reference_leaf_logits"][:, keep], after["reference_leaf_logits"][:, keep], atol=2e-6))
        with torch.no_grad():
            for matcher in (model.parent_matcher, model.fine_matcher):
                matcher.residual[-1].weight.fill_(10.)
        ranked = model(encoded, bank)
        self.assertTrue(torch.equal(before["leaf_membership_logits"], ranked["leaf_membership_logits"]))
        self.assertTrue(torch.equal(before["parent_membership_logits"], ranked["parent_membership_logits"]))

    def test_hash_exclusion_and_singleton_roles_use_static_bank_coverage(self):
        bank, encoded = fixture()
        out = self.model()(encoded, bank, query_hashes=list(bank.hashes)[::3])
        counts = reference_supervision_counts(out, torch.arange(5), MAPPING)
        self.assertEqual(counts["leaf_positive_pairs"], 10)
        self.assertEqual(counts["parent_cross_species_positive_pairs"], 12)
        self.assertEqual(counts["parent_singleton_fallback_positive_pairs"], 2)
        self.assertEqual(counts["leaf_sibling_negative_pairs"], 12)
        self.assertEqual(counts["leaf_other_parent_negative_pairs"], 48)
        for row, col in enumerate(range(0, len(bank.labels), 3)):
            self.assertFalse(bool(out["reference_allowed"][row, col]))
            self.assertTrue(torch.isneginf(out["reference_leaf_logits"][row, col]))
        # A query mask removing every sibling must not turn a multi-leaf
        # parent into the singleton fallback supervision task.
        mask = torch.zeros(5, 5, dtype=torch.bool)
        mask[:, 0] = True
        masked = self.model()(encoded, bank, mask)
        self.assertEqual(reference_supervision_counts(masked, torch.arange(5), MAPPING)
                         ["parent_singleton_fallback_positive_pairs"], 0)

    @staticmethod
    def synthetic_output(leaf_scores, labels, mapping):
        scores = torch.tensor([leaf_scores], requires_grad=True)
        return {"reference_parent_logits": scores.clone(), "reference_leaf_logits": scores,
                "reference_allowed": torch.ones_like(scores, dtype=torch.bool),
                "reference_labels": torch.tensor(labels),
                "reference_leaf_present": torch.ones(len(mapping), dtype=torch.bool),
                "active_parents": torch.ones(1, max(mapping) + 1, dtype=torch.bool)}

    def test_sibling_negatives_are_not_diluted_and_parent_ignores_same_species(self):
        mapping = [0, 0, 1, 2]
        short = self.synthetic_output([2., 3., -5., -5.], [0, 1, 2, 3], mapping)
        long = self.synthetic_output([2., 3.] + [-5.] * 11, [0, 1] + [2] * 10 + [3], mapping)
        a, b = reference_pair_losses(short, [0], mapping), reference_pair_losses(long, [0], mapping)
        self.assertTrue(torch.allclose(a["reference_leaf"], b["reference_leaf"]))
        self.assertTrue(torch.allclose(a["reference_parent"], b["reference_parent"]))
        expected = (F.softplus(torch.tensor(-2.)) + F.softplus(torch.tensor(3.)) + F.softplus(torch.tensor(-5.))) / 3
        self.assertTrue(torch.allclose(a["reference_leaf"], expected))
        short["reference_parent_logits"].retain_grad()
        a["reference_parent"].backward()
        gradient = short["reference_parent_logits"].grad[0]
        self.assertEqual(float(gradient[0]), 0.)
        self.assertLess(float(gradient[1]), 0.)
        self.assertGreater(float(gradient[2]), 0.)

    def test_pair_objective_is_full_only_and_query_gradients_are_open(self):
        bank, encoded = fixture()
        model, labels = self.model(), torch.arange(5)
        for head in (model.parent_reference, model.fine_reference):
            torch.nn.init.constant_(head.residual[-1].weight, .04)
        episodes = build_episodes(labels, MAPPING)
        pairs = model.reference_pairs(encoded, bank)
        outputs = {name: model(encoded, bank, mask, reference_pair_logits=pairs)
                   for name, mask in episodes["masks"].items()}
        losses = hierarchical_losses(outputs, episodes, labels, MAPPING)
        only_full = hierarchical_losses({"full": outputs["full"]}, episodes, labels, MAPPING)
        for key in ("reference_parent", "reference_leaf"):
            self.assertTrue(torch.equal(losses[key], only_full[key]))
        (losses["reference_parent"] + losses["reference_leaf"]).backward()
        for key, value in encoded.items():
            self.assertTrue(torch.isfinite(value.grad).all(), key)
            self.assertGreater(float(value.grad.abs().sum()), 0., key)
        self.assertIsNone(bank.parent.grad)
        self.assertIsNone(bank.fine.grad)
        for parameter in (model.root_bias, model.local_bias, model.parent_reference.residual[-1].weight,
                          model.fine_reference.residual[-1].weight):
            self.assertGreater(float(parameter.grad.abs().sum()), 0.)

    def test_pair_reuse_matches_separate_graph_outputs_and_gradients(self):
        bank, encoded = fixture()
        other_encoded = {k: v.detach().clone().requires_grad_() for k, v in encoded.items()}
        model = self.model()
        other = copy.deepcopy(model)
        masks = build_episodes(torch.arange(5), MAPPING)["masks"]
        pairs = model.reference_pairs(encoded, bank)
        cached, separate = [], []
        for mask in masks.values():
            cached.append(model(encoded, bank, mask, reference_pair_logits=pairs))
            separate.append(other(other_encoded, bank, mask))
            self.assertTrue(torch.equal(cached[-1]["log_probs"], separate[-1]["log_probs"]))
        sum(x["log_probs"].exp().square().sum() for x in cached).backward()
        sum(x["log_probs"].exp().square().sum() for x in separate).backward()
        for key in encoded:
            self.assertTrue(torch.allclose(encoded[key].grad, other_encoded[key].grad, atol=2e-6), key)
        for (_, a), (_, b) in zip(model.named_parameters(), other.named_parameters()):
            self.assertTrue(torch.allclose(a.grad, b.grad, atol=2e-6))
        with self.assertRaisesRegex(ValueError, "same query"):
            model(other_encoded, bank, reference_pair_logits=pairs)

    def test_empty_support_singletons_and_extreme_logits_have_finite_backward(self):
        bank, encoded = fixture()
        model = self.model()
        episodes = build_episodes(torch.arange(5), MAPPING)
        for mask in list(episodes["masks"].values()) + [torch.zeros(5, 5, dtype=torch.bool)]:
            out = model(encoded, bank, mask)
            self.assertTrue(torch.allclose(out["log_probs"].exp().sum(-1), torch.ones(5), atol=1e-6))
        empty = model(encoded, bank, torch.zeros(5, 5, dtype=torch.bool))
        losses = reference_pair_losses(empty, torch.arange(5), MAPPING)
        self.assertTrue(torch.equal(empty["log_probs"][:, 0], torch.zeros(5)))
        self.assertTrue(torch.isneginf(empty["log_probs"][:, 1:]).all())
        (-empty["log_probs"][:, 0].mean() + sum(losses.values())).backward()
        for value in encoded.values():
            self.assertTrue(torch.isfinite(value.grad).all())
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        extreme_pairs = model.reference_pairs(encoded, bank)
        extreme_pairs["parent"] = torch.full_like(extreme_pairs["parent"], 10000., requires_grad=True)
        extreme_pairs["fine"] = torch.full_like(extreme_pairs["fine"], -10000., requires_grad=True)
        extreme = model(encoded, bank, reference_pair_logits=extreme_pairs)
        self.assertTrue(torch.isfinite(extreme["root_logit"]).all())
        self.assertTrue(torch.isfinite(extreme["leaf_accept_logits"]).all())
        self.assertTrue(torch.allclose(extreme["log_probs"].exp().sum(-1), torch.ones(5), atol=1e-6))
        sum(reference_pair_losses(extreme, torch.arange(5), MAPPING).values()).backward()
        self.assertTrue(torch.isfinite(extreme_pairs["parent"].grad).all())
        self.assertTrue(torch.isfinite(extreme_pairs["fine"].grad).all())
        singleton = SupportBank(torch.eye(2), torch.eye(2), [0, 1], ["a", "b"], [0, 1])
        query = {"parent": torch.eye(2, requires_grad=True), "fine": torch.eye(2, requires_grad=True)}
        singleton_model = self.model(2, [0, 1])
        out = singleton_model(query, singleton, query_hashes=["a", "b"])
        counts = reference_supervision_counts(out, [0, 1], [0, 1])
        self.assertEqual(counts["leaf_positive_pairs"], 0)
        self.assertEqual(counts["parent_singleton_fallback_positive_pairs"], 0)
        sum(reference_pair_losses(out, [0, 1], [0, 1]).values()).backward()
        self.assertTrue(all(torch.isfinite(x.grad).all() for x in query.values()))


if __name__ == "__main__":
    unittest.main()
