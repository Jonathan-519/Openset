import copy
import math
import unittest
from unittest import mock

import numpy as np
import torch

from taxosafe_domain import core
from taxosafe_domain.core import DomainBank


def fixture(counts=(6, 6, 6, 6), dimension=12):
    rng = np.random.RandomState(71)
    labels = torch.tensor([node for node, count in enumerate(counts) for _ in range(count)], dtype=torch.long)
    centers = rng.normal(size=(4, dimension))
    centers[1] = centers[0] + .4 * centers[1]
    features = torch.from_numpy(centers[labels.numpy()] + .12 * rng.normal(size=(len(labels), dimension)))
    features = features / features.norm(dim=1, keepdim=True)
    hashes = ["train-%03d" % i for i in range(len(labels))]
    meta = dict(leaf_names=["a", "b", "c", "d"], parent_names=["ab", "c", "d"], leaf_to_parent=[0, 0, 1, 2])
    return features, labels, hashes, meta


def resign(state):
    state["state_sha256"] = core._digest({key: value for key, value in state.items() if key != "state_sha256"})
    return state


class DomainCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_direct_covariance_density_includes_subspace_residual_and_logdet(self):
        args = fixture()
        bank = DomainBank.fit(*args, parent_rank=2, leaf_rank=1, global_rank=3)
        query = args[0][:3]
        model = bank.models["parent"][0]
        basis = model["basis"]
        covariance = model["residual_variance"] * torch.eye(query.shape[1], dtype=torch.float64)
        covariance += (basis * (model["variances"] - model["residual_variance"])[None, :]) @ basis.T
        delta = query - model["mean"]
        mahal = (delta @ torch.linalg.inv(covariance) * delta).sum(1)
        direct = -.5 * (query.shape[1] * math.log(2 * math.pi) + torch.linalg.slogdet(covariance)[1] + mahal)
        orthogonal, density = core._model_scores(query, model)
        self.assertTrue(torch.allclose(density, direct, atol=1e-8, rtol=1e-10))
        residual = delta - (delta @ basis) @ basis.T
        self.assertTrue(torch.allclose(orthogonal, -residual.square().sum(1), atol=1e-12, rtol=0.))

    def test_hierarchy_weighting_and_global_shrink_target(self):
        features, labels, hashes, meta = fixture((3, 9, 4, 8))
        bank = DomainBank.fit(features, labels, hashes, meta)
        means = torch.stack([features[labels == node].mean(0) for node in range(4)])
        expected_parent = (means[0] + means[1]) / 2.
        expected_global = (expected_parent + means[2] + means[3]) / 3.
        self.assertTrue(torch.allclose(bank.models["parent"][0]["mean"], expected_parent, atol=1e-14, rtol=0.))
        self.assertTrue(torch.allclose(bank.models["global"]["mean"], expected_global, atol=1e-14, rtol=0.))
        weights = core._weights(labels, meta, "global")
        target = float(((features - expected_global).square().sum(1) * weights).sum()) / features.shape[1]
        for model in [bank.models["global"]] + bank.models["parent"] + bank.models["leaf"]:
            self.assertAlmostEqual(model["shrink_target"], target, places=14)
            self.assertGreaterEqual(model["residual_variance"], core.VARIANCE_FLOOR)
            self.assertGreaterEqual(model["residual_variance"], .1 * target)

    def test_all_oof_statistics_exclude_entire_query_fold(self):
        args = fixture()
        calls = []
        original = core._fit_models
        def capture(features, labels, meta, settings):
            calls.append(features.clone())
            return original(features, labels, meta, settings)
        with mock.patch.object(core, "_fit_models", side_effect=capture):
            bank = DomainBank.fit(*args)
        self.assertEqual(len(calls), 4)
        seen = []
        for fold, report in enumerate(bank.fit_report["fold_reports"]):
            query = bank.oof["fold_assignment"] == fold
            self.assertTrue(torch.equal(calls[fold], args[0][~query]))
            self.assertFalse(set(report["query_hashes"]) & set(report["support_hashes"]))
            self.assertEqual(set(report["query_hashes"]) | set(report["support_hashes"]), set(args[2]))
            seen += report["query_hashes"]
        self.assertEqual(sorted(seen), sorted(args[2]))
        self.assertTrue(torch.equal(calls[-1], args[0]))
        self.assertEqual(bank.fit_report["optimizer_steps"], 0)
        self.assertFalse(bank.fit_report["dev_used_for_fit"])

    def test_true_member_oof_values_and_conditional_denominator(self):
        features, labels, hashes, meta = fixture()
        bank = DomainBank.fit(features, labels, hashes, meta)
        mapping = torch.tensor(meta["leaf_to_parent"])
        for fold in range(3):
            query = bank.oof["fold_assignment"] == fold
            models = core._fit_models(features[~query], labels[~query], meta, bank.settings)
            raw = core._raw_scores(features[query], models, meta)
            for name in core.NORMALIZER_NAMES:
                ids = labels[query] if name == "leaf_conditional" else mapping[labels[query]]
                expected = raw[name][torch.arange(int(query.sum())), ids]
                self.assertTrue(torch.equal(bank.oof["values"][name][query], expected))
            q = features[query]
            leaf_ll = core._model_scores(q, models["leaf"][0])[1]
            parent_ll = core._model_scores(q, models["parent"][0])[1]
            self.assertTrue(torch.equal(raw["leaf_conditional"][:, 0], (leaf_ll - parent_ll) / q.shape[1]))

    def test_same_parent_absolute_relative_dual_not_independent_maxima(self):
        args = fixture()
        bank = DomainBank.fit(*args)
        raw = {"parent_residual": torch.tensor([[-1., -2., -3.]], dtype=torch.float64),
               "parent_density": torch.tensor([[5., -3., -8.]], dtype=torch.float64),
               "parent_relative": torch.tensor([[-4., 7., -9.]], dtype=torch.float64),
               "leaf_conditional": torch.zeros((1, 4), dtype=torch.float64)}
        for normal in bank.normalizers.values():
            normal["location"].zero_()
            normal["scale"].fill_(1.)
        with mock.patch.object(core, "_raw_scores", return_value=raw):
            scores = bank.score(args[0][:1], ["query"])
        self.assertEqual(float(scores["root_density"][0]), 5.)
        self.assertEqual(float(scores["root_dual"][0]), -3.)
        self.assertTrue(torch.equal(scores["parent_scores"], torch.tensor([[-4., -3., -9.]], dtype=torch.float64)))

    def test_normalizer_pooling_second_moments_and_finite_floor(self):
        values = torch.tensor([1., 3., 7., 9.], dtype=torch.float64)
        normal = core._normalizer(values, torch.ones(4, dtype=torch.bool), torch.tensor([0, 0, 1, 1]), 3)
        w = 2. / 22.
        mean = w * 2. + (1 - w) * 5.
        second = w * 5. + (1 - w) * 35.
        self.assertAlmostEqual(float(normal["location"][0]), mean)
        self.assertAlmostEqual(float(normal["scale"][0]), math.sqrt(second - mean * mean))
        self.assertTrue(normal["report"]["groups"][2]["global_fallback"])
        constant = core._normalizer(torch.ones(4, dtype=torch.float64), torch.ones(4, dtype=torch.bool), torch.zeros(4, dtype=torch.long), 1)
        self.assertEqual(float(constant["scale"][0]), core.SCALE_FLOOR)

    def test_singletons_absent_oof_and_zero_variance_are_explicit(self):
        features, labels, hashes, meta = fixture((1, 3, 1, 4))
        features[:] = features[0]
        bank = DomainBank.fit(features, labels, hashes, meta, parent_rank=50, leaf_rank=50, global_rank=50)
        self.assertFalse(bool(bank.oof["valid"]["leaf_conditional"][labels == 0].any()))
        self.assertFalse(bool(bank.oof["valid"]["parent_density"][labels == 2].any()))
        for model in [bank.models["global"]] + bank.models["leaf"] + bank.models["parent"]:
            self.assertEqual(model["rank"], 0)
            self.assertGreaterEqual(model["residual_variance"], core.VARIANCE_FLOOR)
            self.assertIn("rank_capped", model["fallbacks"])
        self.assertTrue(bank.normalizers["leaf_conditional"]["report"]["groups"][0]["global_fallback"])
        scores = bank.score(features[:1], ["query"])
        self.assertTrue(all(bool(torch.isfinite(value).all()) for value in scores.values()))
        DomainBank.from_state_dict(bank.state_dict())

    def test_rank_caps_primal_dual_and_no_implicit_normalization(self):
        for dimension in (4, 40):
            args = fixture(dimension=dimension)
            features = args[0].float()
            bank = DomainBank.fit(features, *args[1:], parent_rank=99, leaf_rank=99, global_rank=99)
            self.assertTrue(torch.equal(bank.features, features.double()))
            for model in [bank.models["global"]] + bank.models["leaf"] + bank.models["parent"]:
                self.assertEqual(model["rank"], min(99, model["count"] - 1, dimension - 1, model["positive_eigenvalues"]))
            DomainBank.from_state_dict(bank.state_dict())

    def test_score_shapes_direction_and_duplicate_query_hashes(self):
        args = fixture()
        bank = DomainBank.fit(*args)
        scores = bank.score(args[0][:2], ["query", "query"])
        for name in ("root_residual", "root_density", "root_dual"):
            self.assertEqual(tuple(scores[name].shape), (2,))
        self.assertEqual(tuple(scores["parent_scores"].shape), (2, 3))
        self.assertEqual(tuple(scores["leaf_scores"].shape), (2, 4))
        self.assertTrue(torch.equal(scores["root_dual"], scores["parent_scores"].max(1).values))
        empty = bank.score(args[0][:0], [])
        self.assertEqual(tuple(empty["root_dual"].shape), (0,))

    def test_reload_and_test_score_never_fit_any_statistics(self):
        args = fixture()
        bank = DomainBank.fit(*args)
        state = bank.state_dict()
        expected = bank.score(args[0][:2], ["q1", "q2"])
        with mock.patch.object(DomainBank, "fit", side_effect=AssertionError("fit")), \
                mock.patch.object(core, "_fit_models", side_effect=AssertionError("fit models")), \
                mock.patch.object(core, "_fit_model", side_effect=AssertionError("fit model")), \
                mock.patch.object(core, "_normalizer", side_effect=AssertionError("fit normalizer")), \
                mock.patch.object(core, "_eigh", side_effect=AssertionError("eigenfit")), \
                mock.patch("torch.linalg.eigh", side_effect=AssertionError("eigenfit")):
            loaded = DomainBank.from_state_dict(state)
            actual = loaded.score(args[0][:2], ["q1", "q2"])
        for name in expected:
            self.assertTrue(torch.equal(expected[name], actual[name]))
        self.assertEqual(state["state_sha256"], loaded.state_dict()["state_sha256"])

    def test_state_digest_and_resigned_structural_tampering_rejected(self):
        bank = DomainBank.fit(*fixture())
        state = bank.state_dict()
        state["models"]["leaf"][0]["mean"][0] += .01
        with self.assertRaisesRegex(ValueError, "digest"):
            DomainBank.from_state_dict(state)
        for edit in (lambda state: state["settings"].update(variance_floor=.01),
                     lambda state: state["models"]["parent"][0].update(residual_variance=-1.),
                     lambda state: state["models"]["global"].update(logdet=0.),
                     lambda state: state["fit_report"].update(dev_used_for_fit=True),
                     lambda state: state["fit_report"]["fold_reports"][0]["support_hashes"].append("foreign"),
                     lambda state: state["normalizers"]["parent_density"]["scale"].fill_(0.)):
            state = bank.state_dict()
            edit(state)
            with self.assertRaises(ValueError):
                DomainBank.from_state_dict(resign(state))

    def test_overlap_missing_leaf_nonunit_and_invalid_settings_rejected(self):
        args = fixture()
        bank = DomainBank.fit(*args)
        with self.assertRaisesRegex(ValueError, "overlap"):
            bank.score(args[0][:1], args[2][:1])
        with self.assertRaisesRegex(ValueError, "unit normalized"):
            DomainBank.fit(args[0] * 2., *args[1:])
        with self.assertRaises(ValueError):
            DomainBank.fit(args[0], args[1], [args[2][0]] * len(args[2]), args[3])
        selected = args[1] != 1
        with self.assertRaisesRegex(ValueError, "Every known leaf"):
            DomainBank.fit(args[0][selected], args[1][selected], [h for h, keep in zip(args[2], selected) if keep], args[3])
        for options in (dict(parent_rank=-1), dict(folds=1), dict(seed=True), dict(shrinkage=float("nan"))):
            with self.assertRaises(ValueError):
                DomainBank.fit(*args, **options)

    def test_rng_preservation_determinism_and_legacy_api_independence(self):
        args = fixture()
        torch.manual_seed(987)
        before = torch.random.get_rng_state().clone()
        numpy_before = np.random.get_state()
        with mock.patch("torch.argsort", side_effect=AssertionError("legacy argsort")), \
                mock.patch.object(torch.Tensor, "argsort", side_effect=AssertionError("legacy tensor argsort")), \
                mock.patch("torch.isin", side_effect=AssertionError("legacy isin"), create=True):
            bank = DomainBank.fit(*args)
            loaded = DomainBank.from_state_dict(bank.state_dict())
            loaded.score(args[0][:1], ["query"])
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        numpy_after = np.random.get_state()
        self.assertEqual(numpy_before[0], numpy_after[0])
        self.assertTrue(np.array_equal(numpy_before[1], numpy_after[1]))
        self.assertEqual(numpy_before[2:], numpy_after[2:])
        other = DomainBank.fit(*args)
        self.assertEqual(bank.state_dict()["state_sha256"], other.state_dict()["state_sha256"])

    def test_numpy_eigh_fallback_when_legacy_linalg_api_absent(self):
        args = fixture()
        original = DomainBank.fit(*args).score(args[0][:3], ["q1", "q2", "q3"])
        with mock.patch.object(torch.linalg, "eigh", None):
            fallback = DomainBank.fit(*args)
            scores = fallback.score(args[0][:3], ["q1", "q2", "q3"])
            DomainBank.from_state_dict(fallback.state_dict())
        for name in original:
            self.assertTrue(torch.allclose(original[name], scores[name], atol=1e-6, rtol=1e-8), name)


if __name__ == "__main__":
    unittest.main()
