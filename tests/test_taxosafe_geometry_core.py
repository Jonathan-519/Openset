"""Geometry checks on synthetic hierarchies; no real-image performance claims."""
import copy
import io
import json
import unittest

import torch

from taxosafe_geometry.core import (
    HierarchicalGeometry, RobustScoreStandardizer,
    robust_standardizer_fit, robust_standardizer_transform,
)


def hierarchy_fixture(count=20, dimension=8):
    generator = torch.Generator().manual_seed(17)
    fine, parent, labels = [], [], []
    for leaf in range(6):
        genus = leaf // 2
        fine_mean, parent_mean = torch.zeros(dimension), torch.zeros(dimension)
        fine_mean[genus] = 2.
        fine_mean[3 + genus] = .6 if leaf % 2 == 0 else -.6
        parent_mean[genus] = 2.
        fine.append(fine_mean + .03 * torch.randn(count, dimension, generator=generator))
        parent.append(parent_mean + .03 * torch.randn(count, dimension, generator=generator))
        labels.extend([leaf] * count)
    return torch.cat(fine), torch.cat(parent), torch.tensor(labels), torch.tensor([0, 0, 1, 1, 2, 2])


class GeometryCoreTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_known_sibling_and_unseen_parent_evidence(self):
        fine, parent, labels, mapping = hierarchy_fixture()
        model = HierarchicalGeometry.fit(fine, parent, labels, mapping)
        true_fine, true_parent = fine[:20].mean(0), parent[:20].mean(0)
        sibling = fine[20:40].mean(0)
        between_siblings = (true_fine + sibling) / 2
        unseen_parent = torch.zeros(8)
        unseen_parent[7] = 1.
        evidence = model.score(
            torch.stack([true_fine, sibling, between_siblings, true_fine]),
            torch.stack([true_parent, true_parent, true_parent, unseen_parent]),
            torch.zeros(4, dtype=torch.long), torch.zeros(4, dtype=torch.long))
        self.assertEqual(set(evidence), {"parent_score", "leaf_score"})
        self.assertGreater(float(evidence["leaf_score"][0]), float(evidence["leaf_score"][1]))
        self.assertGreater(float(evidence["leaf_score"][0]), float(evidence["leaf_score"][2]))
        self.assertGreater(float(evidence["parent_score"][0]), float(evidence["parent_score"][3]))
        self.assertTrue(torch.equal(evidence["parent_score"][:3], evidence["parent_score"][0].expand(3)))

    def test_fit_is_deterministic_detached_and_does_not_modify_input(self):
        fine, parent, labels, mapping = hierarchy_fixture()
        fine.requires_grad_()
        parent.requires_grad_()
        saved = fine.detach().clone(), parent.detach().clone()
        first = HierarchicalGeometry.fit(fine, parent, labels, mapping)
        second = HierarchicalGeometry.fit(fine, parent, labels, mapping)
        for key, value in first.state_dict().items():
            if torch.is_tensor(value):
                self.assertTrue(torch.equal(value, second.state_dict()[key]))
                self.assertEqual(value.device.type, "cpu")
                self.assertFalse(value.requires_grad)
        score = first.score(fine, parent, mapping[labels], labels)
        self.assertTrue(all(not value.requires_grad for value in score.values()))
        self.assertIsNone(fine.grad)
        self.assertIsNone(parent.grad)
        self.assertTrue(torch.equal(saved[0], fine))
        self.assertTrue(torch.equal(saved[1], parent))
        self.assertEqual(first.diagnostics["trainable_parameters"], 0)

    def test_serialization_round_trip_and_copy_isolation(self):
        fine, parent, labels, mapping = hierarchy_fixture()
        original = HierarchicalGeometry.fit(fine, parent, labels, mapping)
        buffer = io.BytesIO()
        torch.save(original.state_dict(), buffer)
        buffer.seek(0)
        restored = HierarchicalGeometry.from_state_dict(torch.load(buffer))
        for key, value in original.score(fine, parent, mapping[labels], labels).items():
            self.assertTrue(torch.equal(value, restored.score(fine, parent, mapping[labels], labels)[key]))
        state = restored.state_dict()
        state["fine_leaf_means"].zero_()
        self.assertFalse(torch.equal(restored.state_dict()["fine_leaf_means"], state["fine_leaf_means"]))
        json.dumps(restored.diagnostics, allow_nan=False)

    def test_normalization_scale_and_large_finite_values(self):
        fine, parent, labels, mapping = hierarchy_fixture()
        model = HierarchicalGeometry.fit(fine, parent, labels, mapping)
        first = model.score(fine, parent, mapping[labels], labels)
        factors = torch.linspace(.1, 100, len(fine))[:, None]
        second = model.score(fine * factors, parent / factors, mapping[labels], labels)
        for key in first:
            self.assertTrue(torch.allclose(first[key], second[key], atol=2e-4, rtol=1e-4))
        extreme = torch.tensor([[1e300, 2e300, 3e300, 0, 0, 0, 0, 0]], dtype=torch.float64)
        result = model.score(extreme, extreme, torch.tensor([0]), torch.tensor([0]))
        self.assertTrue(all(bool(torch.isfinite(value).all()) for value in result.values()))

    def test_rank_deficient_small_classes_and_singleton_parent_fallback(self):
        # 3 leaf samples in 32 dimensions: every covariance is rank deficient
        # before fixed shrinkage/ridge. Parent 1 has no known sibling leaf.
        fine = torch.zeros(3, 32, dtype=torch.float64)
        fine[0, 0], fine[0, 2] = 1., .5
        fine[1, 0], fine[1, 2] = 1., -.5
        fine[2, 1] = 1.
        parent = fine.clone()
        mapping, labels = torch.tensor([0, 0, 1]), torch.arange(3)
        model = HierarchicalGeometry.fit(fine, parent, labels, mapping)
        self.assertEqual(model.diagnostics["singleton_parent_ids"], [1])
        state = model.state_dict()
        for name in (key for key in state if key.endswith("cholesky")):
            self.assertTrue(bool((state[name].diagonal() > 0).all()))
        result = model.score(fine, parent, mapping[labels], labels)
        residual = fine[2] / fine[2].norm() - state["fine_global_mean"]
        expected = (residual[:, None] * torch.cholesky_solve(
            residual[:, None], state["fine_global_cholesky"])).sum()
        self.assertAlmostEqual(float(result["leaf_score"][2]), float(expected), places=10)
        self.assertTrue(bool(torch.isfinite(result["parent_score"]).all()))

    def test_wrong_candidate_parent_is_rejected(self):
        fine, parent, labels, mapping = hierarchy_fixture()
        model = HierarchicalGeometry.fit(fine, parent, labels, mapping)
        with self.assertRaisesRegex(ValueError, "belong"):
            model.score(fine[:1], parent[:1], [1], [0])
        with self.assertRaises(ValueError):
            model.score(fine[:1], parent[:1], [0], [6])
        with self.assertRaises(ValueError):
            model.score(fine[:1], parent[:1], [0.5], [0])

    def test_invalid_fit_inputs_fail_before_scoring(self):
        fine, parent, labels, mapping = hierarchy_fixture()
        cases = [
            (fine[:, :, None], parent, labels, mapping),
            (fine, parent[:-1], labels, mapping),
            (fine, parent, labels[:-1], mapping),
            (fine, parent, labels.float() + .25, mapping),
            (fine[:20], parent[:20], labels[:20], mapping),
            (fine, parent, labels, torch.tensor([0, 0, 2, 2, 3, 3])),
            (fine, parent, labels, mapping.float() + .5),
            (fine.long(), parent, labels, mapping),
        ]
        for args in cases:
            with self.subTest(shapes=[tuple(item.shape) for item in args]):
                with self.assertRaises(ValueError):
                    HierarchicalGeometry.fit(*args)
        for kwargs in ({"ridge": 0}, {"ridge": float("nan")}, {"shrinkage": -1}, {"shrinkage": 2}):
            with self.assertRaises(ValueError):
                HierarchicalGeometry.fit(fine, parent, labels, mapping, **kwargs)
        for invalid in (0., float("nan"), float("inf")):
            altered = fine.clone()
            altered[0] = invalid
            with self.assertRaises(ValueError):
                HierarchicalGeometry.fit(altered, parent, labels, mapping)

    def test_scoring_shape_dtype_and_zero_vector_validation(self):
        fine, parent, labels, mapping = hierarchy_fixture()
        model = HierarchicalGeometry.fit(fine, parent, labels, mapping)
        with self.assertRaises(ValueError):
            model.score(fine[:, :-1], parent, mapping[labels], labels)
        with self.assertRaises(ValueError):
            model.score(fine, parent[:-1], mapping[labels], labels)
        with self.assertRaises(ValueError):
            model.score(torch.zeros_like(fine), parent, mapping[labels], labels)
        empty = model.score(fine[:0], parent[:0], labels[:0], labels[:0])
        self.assertEqual(empty["parent_score"].shape, (0,))
        self.assertEqual(empty["leaf_score"].shape, (0,))
        half = model.score(fine.half(), parent.half(), mapping[labels], labels)
        self.assertEqual(half["leaf_score"].dtype, torch.float32)

    def test_checkpoint_rejects_inconsistent_statistics(self):
        model = HierarchicalGeometry.fit(*hierarchy_fixture())
        for mutation in ("count", "factor", "shape", "version", "nonfinite"):
            state = model.state_dict()
            if mutation == "count":
                state["parent_counts"][0] += 1
            elif mutation == "factor":
                state["fine_leaf_cholesky"][0, 1] = .1
            elif mutation == "shape":
                state["fine_leaf_means"] = state["fine_leaf_means"][:, :-1]
            elif mutation == "version":
                state["version"] = 2
            else:
                state["parent_global_mean"][0] = float("nan")
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                HierarchicalGeometry.from_state_dict(state)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_inference_matches_cpu_and_has_no_gradients(self):
        fine, parent, labels, mapping = hierarchy_fixture()
        model = HierarchicalGeometry.fit(fine.cuda(), parent.cuda(), labels.cuda(), mapping.cuda())
        cpu = model.score(fine, parent, mapping[labels], labels)
        gpu = model.score(fine.cuda().requires_grad_(), parent.cuda().requires_grad_(), mapping[labels], labels)
        for key in cpu:
            self.assertEqual(gpu[key].device.type, "cuda")
            self.assertFalse(gpu[key].requires_grad)
            self.assertTrue(torch.allclose(cpu[key], gpu[key].cpu(), rtol=1e-4, atol=2e-4))


class RobustStandardizerTest(unittest.TestCase):
    def test_median_mad_and_json_round_trip(self):
        values = torch.tensor([-2., -1., 0., 1., 2.], requires_grad=True)
        scaler = RobustScoreStandardizer.fit(values)
        state = json.loads(json.dumps(scaler.state_dict(), allow_nan=False))
        self.assertEqual(state["method"], "median_mad")
        self.assertEqual(state["fit_rows"], 5)
        self.assertEqual(state["center"], 0.)
        self.assertAlmostEqual(state["scale"], 1.482602218505602)
        transformed = RobustScoreStandardizer.from_state_dict(state).transform(values)
        self.assertFalse(transformed.requires_grad)
        self.assertTrue(torch.allclose(transformed, values.detach() / state["scale"]))
        self.assertEqual(robust_standardizer_fit(values), state)
        self.assertTrue(torch.equal(robust_standardizer_transform(values, state), transformed))

    def test_zero_mad_uses_standard_deviation_then_unit_scale(self):
        scaler = RobustScoreStandardizer.fit(torch.tensor([0., 0., 0., 3.]))
        self.assertEqual(scaler.state_dict()["method"], "median_std_fallback")
        self.assertAlmostEqual(scaler.state_dict()["scale"], 3. * (3. ** .5) / 4.)
        constant = RobustScoreStandardizer.fit(torch.ones(1))
        self.assertEqual(constant.state_dict()["method"], "median_unit_fallback")
        self.assertEqual(constant.state_dict()["scale"], 1.)
        self.assertTrue(torch.equal(constant.transform(torch.ones(5)), torch.zeros(5)))

    def test_rejects_empty_nonfinite_and_invalid_serialized_scale(self):
        for values in (torch.tensor([]), torch.ones(2, 1), torch.tensor([float("nan")]), torch.tensor([1, 2])):
            with self.assertRaises(ValueError):
                RobustScoreStandardizer.fit(values)
        scaler = RobustScoreStandardizer.fit(torch.ones(2))
        for scale in (0., -1., float("inf")):
            state = copy.deepcopy(scaler.state_dict())
            state["scale"] = scale
            with self.assertRaises(ValueError):
                RobustScoreStandardizer.from_state_dict(state)
        with self.assertRaises(ValueError):
            scaler.transform(torch.tensor([float("inf")]))


if __name__ == "__main__":
    unittest.main()
