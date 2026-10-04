"""Per-leaf hard reference negatives and unchanged legacy loss contracts."""
import unittest

import torch
from torch.nn import functional as F

from taxosafe_support.episodes import build_episodes
from taxosafe_support.evidence import HierarchicalEvidence
from taxosafe_support.losses import (_reference_supervision_masks, hierarchical_losses,
                                    reference_pair_losses, reference_supervision_counts)
from tests.test_taxosafe_support_core import MAPPING, fixture


def output(scores, reference_labels, mapping, device="cpu", allowed=None):
    leaf = torch.tensor(scores, dtype=torch.float32, device=device, requires_grad=True)
    if leaf.ndim == 1:
        leaf = leaf[None].detach().requires_grad_()
    parent = leaf.detach().clone().requires_grad_()
    return {"reference_leaf_logits": leaf, "reference_parent_logits": parent,
            "reference_allowed": torch.ones_like(leaf, dtype=torch.bool) if allowed is None else
                                 torch.as_tensor(allowed, device=device, dtype=torch.bool),
            "reference_labels": torch.tensor(reference_labels, device=device),
            "reference_leaf_present": torch.tensor([i in reference_labels for i in range(len(mapping))],
                                                    device=device),
            "active_parents": torch.ones(len(leaf), max(mapping) + 1, device=device, dtype=torch.bool)}


def original_bce(logits, positive, groups, labels, num_leaves):
    """Frozen v3 reduction, independent of optional production selection."""
    active = torch.stack(groups).any(0)
    safe = logits.masked_fill(~active, 0.)
    element = F.binary_cross_entropy_with_logits(safe, positive.to(safe.dtype), reduction="none")
    assignment = F.one_hot(labels, num_leaves).to(safe.dtype)
    numerator, denominator = torch.zeros_like(safe[:, 0]), torch.zeros_like(safe[:, 0])
    for group in groups:
        counts = group.to(safe.dtype) @ assignment
        by_leaf = (element * group).matmul(assignment) / counts.clamp_min(1)
        leaf_count = (counts > 0).sum(-1)
        numerator = numerator + by_leaf.sum(-1) / leaf_count.clamp_min(1)
        denominator = denominator + (leaf_count > 0).to(safe.dtype)
    per_query = numerator / denominator.clamp_min(1)
    valid = denominator > 0
    return per_query.sum() / valid.sum().clamp_min(1) + safe.sum() * 0.


class RelationLossTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_default_is_bit_exact_to_original_all_pairs(self):
        mapping, refs = [0, 0, 1, 1], [0, 0, 1, 1, 1, 2, 3, 3]
        torch.manual_seed(31)
        full = output(torch.randn(3, len(refs)).tolist(), refs, mapping)
        full["reference_allowed"][0, 0] = False
        query_labels, valid = [0, 1, 2], [True, False, True]
        actual = reference_pair_losses(full, query_labels, mapping, valid)
        explicit = reference_pair_losses(full, query_labels, mapping, valid, negative_topk=None)
        masks = _reference_supervision_masks(full, query_labels, mapping, valid)
        for depth, names in (("parent", ["parent_positive", "parent_negative"]),
                             ("leaf", ["leaf_positive", "leaf_sibling_negative", "leaf_other_parent_negative"])):
            key = "reference_" + depth
            scores = full[key + "_logits"]
            expected = original_bce(scores, masks[names[0]], [masks[n] for n in names],
                                    full["reference_labels"], len(mapping))
            self.assertTrue(torch.equal(actual[key], expected))
            self.assertTrue(torch.equal(actual[key], explicit[key]))
            first = torch.autograd.grad(actual[key], scores, retain_graph=True)[0]
            second = torch.autograd.grad(expected, scores, retain_graph=True)[0]
            self.assertTrue(torch.equal(first, second))

    def test_large_k_is_bit_exact_to_all_pairs_and_gradients(self):
        mapping, refs = [0, 0, 1], [0, 0, 1, 1, 1, 2]
        full = output([[1., -1., 3., -4., 2., 0.], [2., 3., -1., 4., -3., 1.]], refs, mapping)
        full["reference_allowed"][0, 2] = False
        expected = reference_pair_losses(full, [0, 1], mapping)
        for k in (3, 6, 100):
            actual = reference_pair_losses(full, [0, 1], mapping, negative_topk=k)
            for key in actual:
                self.assertTrue(torch.equal(actual[key], expected[key]), (k, key))
                scores = full[key + "_logits"]
                a = torch.autograd.grad(actual[key], scores, retain_graph=True)[0]
                b = torch.autograd.grad(expected[key], scores, retain_graph=True)[0]
                self.assertTrue(torch.equal(a, b))

    def test_easy_negative_cannot_dilute_selected_same_leaf(self):
        mapping = [0, 0, 1]
        short = output([1., 3., 2., 4., 1.], [0, 1, 1, 2, 2], mapping)
        long = output([1., 3., 2., 4., 1., -20., -30., -40.], [0, 1, 1, 2, 2, 1, 2, 2], mapping)
        a = reference_pair_losses(short, [0], mapping, negative_topk=2)
        b = reference_pair_losses(long, [0], mapping, negative_topk=2)
        # Leaf negatives in both sibling/other-parent groups are invariant.
        self.assertTrue(torch.equal(a["reference_leaf"], b["reference_leaf"]))
        # Parent positives deliberately keep all sibling pairs, so compare
        # the other-parent negative term through its exact derivatives.
        short_grad = torch.autograd.grad(a["reference_parent"], short["reference_parent_logits"])[0]
        long_grad = torch.autograd.grad(b["reference_parent"], long["reference_parent_logits"])[0]
        self.assertTrue(torch.equal(short_grad[:, 3:5], long_grad[:, 3:5]))
        self.assertTrue(torch.equal(long_grad[:, 6:], torch.zeros_like(long_grad[:, 6:])))

    def test_reference_leaves_have_equal_weight_after_selection(self):
        mapping = [0, 0, 1, 2]
        short = output([1., 2., 3., -2.], [0, 1, 2, 3], mapping)
        repeated = output([1., 2.] + [3.] * 7 + [-2.] * 3, [0, 1] + [2] * 7 + [3] * 3, mapping)
        a = reference_pair_losses(short, [0], mapping, negative_topk=2)
        b = reference_pair_losses(repeated, [0], mapping, negative_topk=2)
        for key in a:
            self.assertTrue(torch.equal(a[key], b[key]), key)
        expected_leaf = (F.softplus(torch.tensor(-1.)) + F.softplus(torch.tensor(2.)) +
                         (F.softplus(torch.tensor(3.)) + F.softplus(torch.tensor(-2.))) / 2) / 3
        self.assertTrue(torch.equal(a["reference_leaf"], expected_leaf))

    def test_masked_extremes_do_not_enter_selection_or_gradients(self):
        mapping = [0, 0, 1]
        full = output([1., float("inf"), 3., 2., -9., float("nan"), 4., 1., -8.],
                      [0, 1, 1, 1, 1, 2, 2, 2, 2], mapping,
                      allowed=[[True, False, True, True, True, False, True, True, True]])
        losses = reference_pair_losses(full, [0], mapping, negative_topk=2)
        self.assertTrue(all(bool(torch.isfinite(x)) for x in losses.values()))
        sum(losses.values()).backward()
        leaf_gradient = full["reference_leaf_logits"].grad[0]
        parent_gradient = full["reference_parent_logits"].grad[0]
        self.assertTrue(bool(torch.isfinite(leaf_gradient).all()))
        self.assertTrue(bool(torch.isfinite(parent_gradient).all()))
        self.assertTrue(torch.equal(leaf_gradient[[1, 4, 5, 8]], torch.zeros(4)))
        self.assertTrue(bool((leaf_gradient[[2, 3, 6, 7]] > 0).all()))
        self.assertLess(float(leaf_gradient[0]), 0.)
        # Every parent-positive sibling pair still receives supervision.
        self.assertTrue(bool((parent_gradient[[2, 3, 4]] < 0).all()))
        self.assertTrue(torch.equal(parent_gradient[[0, 1, 5, 8]], torch.zeros(4)))

    def test_empty_support_or_invalid_rows_give_connected_zero(self):
        for allowed, valid in (([[False] * 4], [True]), ([[True] * 4], [False])):
            full = output([float("inf"), float("nan"), -float("inf"), 10000.], [0, 0, 1, 2], [0, 0, 1],
                          allowed=allowed)
            losses = reference_pair_losses(full, [0], [0, 0, 1], valid, negative_topk=2)
            self.assertTrue(all(float(x.detach()) == 0. for x in losses.values()))
            sum(losses.values()).backward()
            for key in ("reference_parent_logits", "reference_leaf_logits"):
                self.assertTrue(torch.equal(full[key].grad, torch.zeros_like(full[key])))

    def test_counts_cover_selected_negatives_and_all_positives(self):
        full = output([1., 2., 3., 4., -2., 1., 3.], [0, 1, 1, 1, 2, 2, 2], [0, 0, 1])
        before = reference_supervision_counts(full, [0], [0, 0, 1])
        reference_pair_losses(full, [0], [0, 0, 1], negative_topk=1)
        self.assertEqual(before, reference_supervision_counts(full, [0], [0, 0, 1]))
        after = reference_supervision_counts(full, [0], [0, 0, 1], negative_topk=1)
        self.assertEqual(set(before), set(after))
        for key in before:
            self.assertEqual(after[key], 1 if "negative" in key else before[key])
        self.assertEqual(before["leaf_sibling_negative_pairs"], 3)
        self.assertEqual(before["parent_other_parent_negative_pairs"], 3)

    def test_invalid_topk_is_rejected(self):
        full = output([1., 2.], [0, 1], [0, 1])
        for value in (0, -1, True, False, 1.5, 1.0, "2"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "positive integer"):
                reference_pair_losses(full, [0], [0, 1], negative_topk=value)

    def test_only_negative_groups_are_selected_and_query_means_are_equal(self):
        mapping = [0, 0, 1]
        full = output([[1., -2., 5., 3., 2.], [1., 0., -2., -3., 4.]], [0, 0, 1, 1, 2], mapping)
        both = reference_pair_losses(full, [0, 2], mapping, negative_topk=1)
        singles = [reference_pair_losses(full, [0, 2], mapping, valid_rows=[i == j for i in range(2)],
                                         negative_topk=1) for j in range(2)]
        for key in both:
            self.assertTrue(torch.equal(both[key], (singles[0][key] + singles[1][key]) / 2))
        grad = torch.autograd.grad(both["reference_leaf"], full["reference_leaf_logits"])[0]
        self.assertTrue(bool((grad[0, :2] < 0).all()))  # All positives, including the harder one.
        self.assertGreater(float(grad[0, 2]), 0.)
        self.assertEqual(float(grad[0, 3]), 0.)

    def _training_loss_integration(self, device):
        bank, encoded = fixture()
        bank = bank.to(device)
        encoded = {k: v.detach().to(device).requires_grad_() for k, v in encoded.items()}
        model = HierarchicalEvidence(8, MAPPING, decoupled=True, membership_mode="reference").to(device)
        for head in (model.parent_reference, model.fine_reference):
            torch.nn.init.constant_(head.residual[-1].weight, .04)
        labels = torch.arange(5, device=device)
        episodes = build_episodes(labels, MAPPING)
        pairs = model.reference_pairs(encoded, bank)
        outputs = {name: model(encoded, bank, mask, reference_pair_logits=pairs,
                               query_hashes=list(bank.hashes)[::3]) for name, mask in episodes["masks"].items()}
        default = hierarchical_losses(outputs, episodes, labels, MAPPING)
        explicit = hierarchical_losses(outputs, episodes, labels, MAPPING, reference_negative_topk=None)
        for key in default:
            if torch.is_tensor(default[key]):
                self.assertTrue(torch.equal(default[key], explicit[key]), key)
            else:
                self.assertEqual(default[key], explicit[key])
        losses = hierarchical_losses(outputs, episodes, labels, MAPPING, reference_negative_topk=2)
        expected = reference_pair_losses(outputs["full"], labels, MAPPING,
                                          episodes["valid"]["full"].to(device), negative_topk=2)
        self.assertEqual(set(losses), set(default))
        for key in expected:
            self.assertTrue(torch.equal(losses[key], expected[key]))
        sum(expected.values()).backward()
        for value in encoded.values():
            self.assertIsNotNone(value.grad)
            self.assertTrue(bool(torch.isfinite(value.grad).all()))
            self.assertGreater(float(value.grad.abs().sum()), 0.)
        for head in (model.parent_reference, model.fine_reference):
            self.assertGreater(float(head.residual[-1].weight.grad.abs().sum()), 0.)

    def test_hierarchical_loss_passes_optional_mining_and_preserves_default(self):
        self._training_loss_integration("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device required for hard-negative training regression")
    def test_cuda_hard_negative_loss_and_backward(self):
        self._training_loss_integration("cuda")


if __name__ == "__main__":
    unittest.main()
