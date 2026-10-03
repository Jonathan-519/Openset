"""Frozen-feature reconstruction: formula, provenance, masks, and fitting."""
import math
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from taxosafe_refine.reconstruction import (ClassSpecificReconstruction,
                                           fit_cached_features, stratified_hash_split)


class ReconstructionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    @staticmethod
    def fixture():
        generator = torch.Generator().manual_seed(77)
        labels = torch.arange(3).repeat_interleave(10)
        features = F.normalize(torch.eye(8)[:3][labels] + .03 * torch.randn(30, 8, generator=generator), dim=-1)
        hashes = ["content_%02d" % i for i in range(len(labels))]
        return features, labels, hashes

    @staticmethod
    def axis_model(score_mode="relative", active_mask=None):
        model = ClassSpecificReconstruction(2, 2, rank=1, score_mode=score_mode, active_mask=active_mask)
        with torch.no_grad():
            for index, ae in enumerate(model.class_aes):
                ae[0].weight.zero_()
                ae[2].weight.zero_()
                ae[0].weight[0, index] = 1.
                ae[2].weight[index, 0] = 1.
        return model

    def test_bias_free_class_specific_tanh_and_error_formula(self):
        model = self.axis_model()
        x = torch.tensor([[1., 0.], [0., 2.]])
        expected = torch.tensor([[[1. - math.tanh(1.), 1.]], [[2., 2. - math.tanh(2.)]]])
        self.assertTrue(torch.allclose(model.errors(x), expected))
        self.assertEqual(tuple(model.errors(x[:, None].expand(-1, 3, -1)).shape), (2, 3, 2))
        self.assertTrue(all(layer.bias is None for ae in model.class_aes for layer in ae if isinstance(layer, nn.Linear)))
        self.assertAlmostEqual(float(model.scale.detach()), 1., places=6)

    def test_spatial_softmax_then_probability_average(self):
        model = self.axis_model()
        x = torch.tensor([[[2., 0.], [0., .2]], [[.8, .1], [0., 1.5]]])
        errors = model.errors(x)
        expected = torch.softmax(-errors * model.scale, dim=-1).mean(1)
        actual = model.log_probabilities(x).exp()
        incorrect = torch.softmax((-errors * model.scale).mean(1), dim=-1)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-7))
        self.assertFalse(torch.allclose(actual, incorrect, atol=1e-4))
        self.assertTrue(torch.allclose(actual.sum(-1), torch.ones(2)))
        loss = model.loss(x, [0, 1])
        self.assertTrue(torch.allclose(loss, -expected[[0, 1], [0, 1]].log().mean()))

    def test_candidate_relative_and_cssr_scores_do_not_rerank(self):
        x = torch.tensor([[1., 0.], [0., 2.]])
        model = self.axis_model()
        self.assertEqual(int(model(x).argmax(-1)[1]), 1)
        original_candidates = torch.tensor([0, 0])
        score = model.candidate_scores(x, original_candidates)
        self.assertTrue(torch.allclose(score, torch.tensor([math.tanh(1.) - 1., -1.])))
        self.assertTrue(torch.equal(original_candidates, torch.tensor([0, 0])))
        cssr = self.axis_model("cssr")
        self.assertTrue(torch.allclose(cssr.candidate_scores(x, original_candidates),
                                       torch.tensor([math.tanh(1.) - 1., -.5])))
        # Classification temperature is learned; absolute rejection evidence
        # must not change merely because that scale changes.
        model.raw_scale.data.fill_(20.)
        self.assertTrue(torch.equal(score, model.candidate_scores(x, original_candidates)))

    def test_spatial_score_averages_position_ratios(self):
        x = torch.tensor([[[1., 0.], [0., 2.]]])
        model = self.axis_model()
        expected = ((math.tanh(1.) - 1.) - 1.) / 2
        self.assertAlmostEqual(float(model.candidate_scores(x, [0]).detach()), expected, places=6)
        wrong = -(1. - math.tanh(1.) + 2.) / 3.
        self.assertNotAlmostEqual(expected, wrong, places=4)

    def test_masks_zero_vectors_and_small_norms_fail_closed_finitely(self):
        model = self.axis_model(active_mask=[True, False])
        x = torch.tensor([[1., 0.], [0., 0.], [1e-12, 0.], [0., 1.]])
        logp = model.log_probabilities(x)
        self.assertTrue(torch.equal(logp[:, 0], torch.zeros(4)))
        self.assertTrue(torch.isneginf(logp[:, 1]).all())
        scores = model.candidate_scores(x, [0, 0, 0, 1])
        self.assertTrue(torch.isfinite(scores).all())
        self.assertTrue(torch.equal(scores[1:], torch.full((3,), torch.finfo(torch.float32).min)))
        self.assertGreater(float(scores[0].detach()), float(scores[1].detach()))
        with self.assertRaisesRegex(ValueError, "active"):
            model.loss(x[:1], [1])
        spatial_zero = torch.tensor([[[1., 0.], [0., 0.]]])
        self.assertEqual(float(model.candidate_scores(spatial_zero, [0]).detach()), torch.finfo(torch.float32).min)

    def test_gradients_reach_only_active_autoencoders_and_positive_scale(self):
        model = ClassSpecificReconstruction(4, 3, rank=2, active_mask=[True, True, False])
        x = torch.tensor([[1., .1, .2, 0.], [0., 1., .1, .2]])
        model.loss(x, [0, 1]).backward()
        self.assertGreater(float(model.scale.detach()), 0.)
        self.assertIsNotNone(model.raw_scale.grad)
        self.assertTrue(torch.isfinite(model.raw_scale.grad))
        for index, ae in enumerate(model.class_aes):
            for parameter in ae.parameters():
                self.assertTrue(torch.isfinite(parameter.grad).all())
                if index < 2:
                    self.assertGreater(float(parameter.grad.abs().sum()), 0.)
                else:
                    self.assertEqual(float(parameter.grad.abs().sum()), 0.)

    def test_invalid_shapes_labels_configuration_fail_early(self):
        for arguments in ({"rank": 0}, {"rank": 1.5}, {"eps": 1e-30}, {"eps": 1.},
                          {"scale_init": float("inf")}, {"score_mode": "combined"},
                          {"active_mask": [False, False]}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                ClassSpecificReconstruction(2, 2, **arguments)
        model = self.axis_model()
        for features in (torch.ones(3), torch.ones(1, 0, 2), torch.ones(1, 3), torch.tensor([[float("nan"), 0.]])):
            with self.assertRaises(ValueError):
                model.errors(features)
        for labels in ([2], [-1], [.3], [True]):
            with self.assertRaises(ValueError):
                model.candidate_scores(torch.ones(1, 2), labels)

    def test_hash_split_is_stratified_deduplicated_order_independent(self):
        features, labels, hashes = self.fixture()
        split = stratified_hash_split(labels, hashes, 3, seed=17)
        permutation = torch.randperm(len(labels), generator=torch.Generator().manual_seed(8))
        other = stratified_hash_split(labels[permutation], [hashes[i] for i in permutation], 3, seed=17)
        self.assertEqual(split["fit_hashes"], other["fit_hashes"])
        self.assertEqual(split["validation_hashes"], other["validation_hashes"])
        self.assertFalse(set(split["fit_hashes"]) & set(split["validation_hashes"]))
        self.assertTrue(all(c["fit_images"] == 8 and c["validation_images"] == 2 for c in split["per_class"]))
        duplicate = stratified_hash_split(torch.cat((labels, labels[:1])), hashes + hashes[:1], 3, seed=17)
        self.assertEqual(split["fit_hashes"], duplicate["fit_hashes"])
        self.assertEqual(split["validation_hashes"], duplicate["validation_hashes"])
        self.assertEqual(duplicate["duplicate_rows_removed"], 1)
        self.assertFalse(split["strict_unseen_class_evaluation"])
        self.assertFalse(split["frozen_backbone_independent_validation"])

    def test_split_singletons_masks_and_conflicting_hashes(self):
        split = stratified_hash_split([0, 0, 1], ["a", "b", "c"], 3, active_mask=[True, True, False])
        self.assertIn("c", split["fit_hashes"])
        self.assertNotIn("c", split["validation_hashes"])
        for labels, hashes, kwargs in (([0, 1], ["a", "a"], {}),
                                       ([0, 0], ["a", "b"], {}),
                                       ([0, 1], ["a", "b"], {}),
                                       ([0, 0, 1], ["a", "b", "c"], {"active_mask": [True, False]})):
            with self.subTest(labels=labels, hashes=hashes), self.assertRaises(ValueError):
                stratified_hash_split(labels, hashes, 2, **kwargs)

    def test_fit_learns_restores_best_and_detaches_cache(self):
        features, labels, hashes = self.fixture()
        features.requires_grad_()
        model, report = fit_cached_features(features, labels, hashes, 3, rank=2, epochs=40,
                                            patience=8, learning_rate=1e-3)
        self.assertIsNone(features.grad)
        self.assertLess(report["best_validation_nll"], report["initial_validation_nll"])
        self.assertEqual(report["best_validation_nll"], min([report["initial_validation_nll"]]
                                                         + [row["validation_nll"] for row in report["history"]]))
        vi = report["split"]["validation_indices"]
        with torch.no_grad():
            observed = float(model.loss(features[vi], labels[vi]))
        self.assertAlmostEqual(observed, report["best_validation_nll"], places=6)
        self.assertGreater(report["best_epoch"], 0)
        restored = ClassSpecificReconstruction(**report["model_arguments"])
        restored.load_state_dict(model.state_dict(), strict=True)
        self.assertTrue(torch.equal(model(features), restored(features)))
        self.assertTrue(torch.equal(model.candidate_scores(features, labels), restored.candidate_scores(features, labels)))

    def test_fitting_order_rng_isolation_and_duplicate_feature_checks(self):
        features, labels, hashes = self.fixture()
        torch.manual_seed(905)
        before = torch.get_rng_state().clone()
        model, report = fit_cached_features(features, labels, hashes, 3, rank=2, epochs=4, batch_size=7)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        permutation = torch.arange(len(labels) - 1, -1, -1)
        other, second = fit_cached_features(features[permutation], labels[permutation],
                                            [hashes[i] for i in permutation], 3, rank=2, epochs=4, batch_size=7)
        self.assertEqual(report["history"], second["history"])
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, other.state_dict()[key]), key)
        duplicate_features = torch.cat((features, features[:1] + .1))
        with self.assertRaisesRegex(ValueError, "conflicting cached features"):
            fit_cached_features(duplicate_features, torch.cat((labels, labels[:1])), hashes + hashes[:1], 3)

    def test_nondefault_initial_scale_metadata_roundtrip(self):
        features, labels, hashes = self.fixture()
        model, report = fit_cached_features(features, labels, hashes, 3, rank=2,
                                            scale_init=2.5, epochs=2)
        self.assertEqual(model.constructor_arguments()["scale_init"], 2.5)
        self.assertEqual(report["model_arguments"]["scale_init"], 2.5)
        self.assertNotIn("scale_init", model.state_dict())
        restored = ClassSpecificReconstruction(**report["model_arguments"])
        self.assertAlmostEqual(float(restored.scale.detach()), 2.5, places=6)
        restored.load_state_dict(model.state_dict(), strict=True)
        self.assertTrue(torch.equal(model(features), restored(features)))
        self.assertTrue(torch.equal(model.candidate_scores(features, labels),
                                    restored.candidate_scores(features, labels)))

    def test_cached_spatial_fit_and_active_mask(self):
        features, labels, hashes = self.fixture()
        keep = labels < 2
        spatial = features[keep, None, :].repeat(1, 2, 1)
        model, report = fit_cached_features(spatial, labels[keep], hashes[:20], 3, rank=2,
                                            active_mask=[True, True, False], epochs=2)
        self.assertEqual(report["feature_shape"], [20, 2, 8])
        self.assertTrue(torch.isneginf(model(spatial)[:, 2]).all())
        scores = model.candidate_scores(spatial, labels[keep])
        self.assertTrue(torch.isfinite(scores).all())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device required for reconstruction smoke")
    def test_cuda_cached_fit_and_scores(self):
        features, labels, hashes = self.fixture()
        model, report = fit_cached_features(features, labels, hashes, 3, rank=2, epochs=2, device="cuda")
        scores = model.candidate_scores(features.to("cuda"), labels.to("cuda"))
        self.assertTrue(torch.isfinite(scores).all())
        self.assertEqual(report["epochs_completed"], 2)


if __name__ == "__main__":
    unittest.main()
