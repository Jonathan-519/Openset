"""Coordinate-aware verification, symmetry, exclusions and gradient contracts."""
import copy
import unittest

import torch
from torch.nn import functional as F

from taxosafe_support.episodes import build_episodes
from taxosafe_support.evidence import HierarchicalEvidence
from taxosafe_support.losses import hierarchical_losses, reference_pair_losses, reference_supervision_counts
from taxosafe_support.relation import RelationMatcher
from taxosafe_support.support import SupportBank
from tests.test_taxosafe_support_core import MAPPING, fixture


class RelationEvidenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def model(self, **kwargs):
        return HierarchicalEvidence(8, MAPPING, decoupled=True, membership_mode="relation", relation_dim=4, **kwargs)

    @staticmethod
    def open_residual(model):
        # Initialization intentionally reproduces the cosine prior. Open the
        # learned branch explicitly to test its nonzero gradient paths.
        with torch.no_grad():
            for head in (model.parent_reference, model.fine_reference):
                head.residual[-1].weight.fill_(.04)

    def test_opt_in_validation_and_legacy_random_sequence(self):
        bank, encoded = fixture()
        for mode, decoupled in (("prototype", False), ("prototype", True), ("reference", True)):
            torch.manual_seed(42)
            before = HierarchicalEvidence(8, MAPPING, decoupled=decoupled, membership_mode=mode)
            rng = torch.get_rng_state().clone()
            torch.manual_seed(42)
            after = HierarchicalEvidence(8, MAPPING, decoupled=decoupled, membership_mode=mode, relation_dim=7)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()), mode)
            self.assertEqual(list(before.state_dict()), list(after.state_dict()))
            for key, value in before.state_dict().items():
                self.assertTrue(torch.equal(value, after.state_dict()[key]), key)
            a, b = before(encoded, bank), after(encoded, bank)
            for key in a:
                if isinstance(a[key], torch.Tensor):
                    self.assertTrue(torch.equal(a[key], b[key]), key)
            self.assertFalse(any("projection" in key for key in before.state_dict()))
        with self.assertRaisesRegex(ValueError, "requires decoupled"):
            HierarchicalEvidence(8, MAPPING, membership_mode="relation")
        for invalid in (0, -1, 2.5, float("nan"), float("inf"), True):
            with self.subTest(relation_dim=invalid), self.assertRaisesRegex(ValueError, "positive integer"):
                RelationMatcher(8, relation_dim=invalid)

    def test_equal_scalar_pairs_retain_separable_coordinate_evidence(self):
        matcher = RelationMatcher(3, relation_dim=3, hidden_dim=2)
        query = torch.tensor([[1., 0., 0.]])
        references = torch.tensor([[.6, .8, 0.], [.6, 0., .8]])
        with torch.no_grad():
            matcher.projection.weight.copy_(torch.eye(3))
            matcher.residual[0].weight.zero_()
            matcher.residual[0].bias.zero_()
            matcher.residual[0].weight[0, 4] = 1.  # second coordinate absolute difference
            matcher.residual[-1].weight.zero_()
            matcher.residual[-1].weight[0, 0] = 1.
        features = matcher.pair_features(query, references, query[:, None], references[:, None])
        self.assertTrue(torch.equal(features[:, 0, :3], features[:, 1, :3]))
        self.assertFalse(torch.equal(features[:, 0, 3:], features[:, 1, 3:]))
        scores = matcher(query, references, query[:, None], references[:, None])
        self.assertGreater(float(scores[0, 0].detach()), float(scores[0, 1].detach()))

    def test_pair_symmetry_and_token_reference_permutation(self):
        torch.manual_seed(71)
        matcher = RelationMatcher(8, relation_dim=5)
        matcher.residual[-1].weight.data.fill_(.1)
        query, references = torch.randn(3, 8), torch.randn(4, 8)
        qlocal, rlocal = torch.randn(3, 2, 8), torch.randn(4, 3, 8)
        a = matcher.pair_features(query, references, qlocal, rlocal)
        reverse = matcher.pair_features(references, query, rlocal, qlocal)
        self.assertTrue(torch.allclose(a, reverse.transpose(0, 1), atol=2e-6))
        permutation = torch.tensor([2, 0, 3, 1])
        b = matcher.pair_features(query, references[permutation], qlocal[:, [1, 0]], rlocal[permutation][:, [2, 0, 1]])
        self.assertTrue(torch.allclose(a[:, permutation], b, atol=2e-6))
        self.assertTrue(torch.allclose(matcher(query, references, qlocal, rlocal),
                                       matcher(references, query, rlocal, qlocal).T, atol=2e-6))

    def test_exact_local_ties_average_all_correspondences(self):
        matcher = RelationMatcher(3, relation_dim=3)
        matcher.projection.weight.data.copy_(torch.eye(3))
        query = torch.tensor([[1., 0., 0.]])
        references = torch.tensor([[1., 0., 0.]])
        qt = torch.tensor([[[1., 0., 0.]]], requires_grad=True)
        rt = torch.tensor([[[0., 1., 0.], [0., 0., 1.]]], requires_grad=True)
        features = matcher.pair_features(query, references, qt, rt)
        # Both tied token pairs get half the total bidirectional mass.
        self.assertTrue(torch.equal(features[0, 0, 9:12], torch.tensor([1., .5, .5])))
        swapped = matcher.pair_features(query, references, qt, rt[:, [1, 0]])
        self.assertTrue(torch.equal(features, swapped))
        features.sum().backward()
        self.assertTrue(torch.isfinite(qt.grad).all())
        self.assertIsNone(rt.grad)

    def test_missing_local_channels_and_chunked_graph_equivalence(self):
        torch.manual_seed(33)
        matcher = RelationMatcher(8, relation_dim=4)
        matcher.residual[-1].weight.data.fill_(.05)
        query = torch.randn(3, 8, requires_grad=True)
        references = torch.randn(7, 8, requires_grad=True)
        no_local = matcher.pair_features(query, references)
        self.assertEqual(tuple(no_local.shape), (3, 7, 19))
        self.assertTrue(torch.equal(no_local[..., 11:], torch.zeros(3, 7, 8)))
        self.assertTrue(torch.equal(no_local[..., 0], no_local[..., 1]))
        self.assertTrue(torch.equal(no_local[..., 0], no_local[..., 2]))
        qlocal = torch.randn(3, 2, 8, requires_grad=True)
        rlocal = torch.randn(7, 3, 8, requires_grad=True)
        copied = copy.deepcopy(matcher)
        copied._coordinate_budget = 24  # exactly one image pair per chunk
        a, b = matcher(query, references, qlocal, rlocal), copied(query, references, qlocal, rlocal)
        self.assertTrue(torch.allclose(a, b, atol=2e-6))
        ga = torch.autograd.grad(a.sum(), (query, qlocal, matcher.projection.weight, references, rlocal), allow_unused=True)
        gb = torch.autograd.grad(b.sum(), (query, qlocal, copied.projection.weight))
        for left, right in zip(ga[:3], gb):
            self.assertTrue(torch.allclose(left, right, atol=2e-5))
        self.assertIsNone(ga[3])
        self.assertIsNone(ga[4])

    def test_cosine_initialization_outputs_and_parameter_scope(self):
        bank, encoded = fixture()
        model = self.model()
        initial = model(encoded, bank)
        prior = HierarchicalEvidence(8, MAPPING, decoupled=True, membership_mode="reference")(encoded, bank)
        for key in ("reference_parent_logits", "reference_leaf_logits", "parent_membership_logits", "leaf_membership_logits"):
            self.assertTrue(torch.allclose(initial[key], prior[key], atol=2e-6), key)
        self.assertEqual(set(initial), set(prior))
        self.assertFalse(hasattr(model, "parent_membership"))
        self.assertEqual(tuple(model.parent_reference.projection.weight.shape), (4, 8))
        self.assertEqual(model.parent_reference.residual[0].in_features, 19)
        before = model.reference_pairs(encoded, bank)
        with torch.no_grad():
            for head in (model.parent_matcher, model.fine_matcher):
                head.residual[-1].weight.fill_(20.)
        after = model.reference_pairs(encoded, bank)
        self.assertTrue(torch.equal(before["parent"], after["parent"]))
        self.assertTrue(torch.equal(before["fine"], after["fine"]))

    def test_excluded_hash_has_no_effect_on_any_pooled_score_or_pair_loss(self):
        bank, encoded = fixture()
        query = {key: value[:1] for key, value in encoded.items()}
        model = self.model()
        self.open_residual(model)
        excluded_hash = bank.hashes[0]
        state = bank.state_dict()
        for key in ("parent", "fine", "parent_local", "fine_local"):
            state[key][0] = -state[key][0]
        changed = SupportBank.from_state_dict(state)
        a = model(query, bank, query_hashes=[excluded_hash])
        b = model(query, changed, query_hashes=[excluded_hash])
        for key in ("log_probs", "parent_logits", "leaf_logits", "parent_membership_logits", "leaf_membership_logits"):
            self.assertTrue(torch.allclose(a[key], b[key], atol=2e-6), key)
        self.assertFalse(bool(a["reference_allowed"][0, 0]))
        self.assertTrue(torch.isneginf(a["reference_parent_logits"][0, 0]))
        al, bl = reference_pair_losses(a, [0], MAPPING), reference_pair_losses(b, [0], MAPPING)
        for key in al:
            self.assertTrue(torch.allclose(al[key], bl[key], atol=2e-6), key)

    def test_cached_pairs_reuse_graph_and_reject_foreign_query_or_bank(self):
        bank, encoded = fixture()
        model = self.model()
        self.open_residual(model)
        other = copy.deepcopy(model)
        second = {key: value.detach().clone().requires_grad_() for key, value in encoded.items()}
        pairs = model.reference_pairs(encoded, bank)
        cached, separate = [], []
        for mask in build_episodes(torch.arange(5), MAPPING)["masks"].values():
            cached.append(model(encoded, bank, mask, reference_pair_logits=pairs))
            separate.append(other(second, bank, mask))
            self.assertTrue(torch.equal(cached[-1]["log_probs"], separate[-1]["log_probs"]))
        sum(out["log_probs"].exp().square().sum() for out in cached).backward()
        sum(out["log_probs"].exp().square().sum() for out in separate).backward()
        for key in encoded:
            self.assertTrue(torch.allclose(encoded[key].grad, second[key].grad, atol=2e-6), key)
        for (_, left), (_, right) in zip(model.named_parameters(), other.named_parameters()):
            self.assertTrue(torch.allclose(left.grad, right.grad, atol=2e-6))
        with self.assertRaisesRegex(ValueError, "same query"):
            model(second, bank, reference_pair_logits=pairs)
        with self.assertRaisesRegex(ValueError, "same query"):
            model(encoded, SupportBank.from_state_dict(bank.state_dict()), reference_pair_logits=pairs)

    def _training_gradient_check(self, device):
        bank, encoded = fixture()
        bank = bank.to(device)
        encoded = {key: value.detach().to(device).requires_grad_() for key, value in encoded.items()}
        model = self.model().to(device)
        self.open_residual(model)
        labels = torch.arange(5, device=device)
        episodes = build_episodes(labels, MAPPING)
        pairs = model.reference_pairs(encoded, bank)
        outputs = {name: model(encoded, bank, mask, query_hashes=list(bank.hashes)[::3], reference_pair_logits=pairs)
                   for name, mask in episodes["masks"].items()}
        counts = reference_supervision_counts(outputs["full"], labels, MAPPING)
        self.assertEqual(counts["parent_cross_species_positive_pairs"], 12)
        self.assertEqual(counts["parent_singleton_fallback_positive_pairs"], 2)
        losses = hierarchical_losses(outputs, episodes, labels, MAPPING)
        loss = losses["reference_parent"] + losses["reference_leaf"]
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        for key, value in encoded.items():
            self.assertTrue(bool(torch.isfinite(value.grad).all()), key)
            self.assertGreater(float(value.grad.abs().sum()), 0., key)
        for head in (model.parent_reference, model.fine_reference):
            for parameter in head.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
                self.assertGreater(float(parameter.grad.abs().sum()), 0.)
        for key in ("parent", "fine", "parent_local", "fine_local"):
            self.assertIsNone(getattr(bank, key).grad)

    def test_training_loss_gradients_and_detached_bank(self):
        self._training_gradient_check("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device required for relation training regression")
    def test_cuda_relation_training_loss_and_backward(self):
        self._training_gradient_check("cuda")

    def test_empty_permissions_preserve_probability_and_finite_backward(self):
        bank, encoded = fixture()
        model = self.model()
        self.open_residual(model)
        for mask in build_episodes(torch.arange(5), MAPPING)["masks"].values():
            out = model(encoded, bank, mask)
            self.assertTrue(torch.allclose(out["log_probs"].exp().sum(-1), torch.ones(5), atol=1e-6))
        empty = model(encoded, bank, torch.zeros(5, 5, dtype=torch.bool))
        self.assertTrue(torch.equal(empty["log_probs"][:, 0], torch.zeros(5)))
        self.assertTrue(torch.isneginf(empty["log_probs"][:, 1:]).all())
        loss = -empty["log_probs"][:, 0].mean() + sum(reference_pair_losses(empty, torch.arange(5), MAPPING).values())
        self.assertEqual(float(loss.detach()), 0.)
        loss.backward()
        for value in encoded.values():
            self.assertTrue(torch.isfinite(value.grad).all())
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())


if __name__ == "__main__":
    unittest.main()
