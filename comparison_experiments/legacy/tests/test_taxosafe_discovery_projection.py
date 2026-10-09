import copy
import unittest

import torch
from torch.nn import functional as F

from taxosafe_discovery.models import ResidualProjection, load_projection
from taxosafe_discovery.projection_training import fit_projection, supervised_contrastive, transform_projection


def fixture():
    generator = torch.Generator().manual_seed(311)
    values = F.normalize(torch.randn(12, 8, generator=generator), dim=-1)
    meta = {"leaf_names": ["a", "b", "c"], "parent_names": ["p", "q"], "leaf_to_parent": [0, 0, 1]}
    records = [{"image_sha256": format(i, "064x"), "split": "train", "status": "known",
                "true_leaf": i % 3, "true_parent": meta["leaf_to_parent"][i % 3],
                "source": "known", "path": str(i)} for i in range(12)]
    group = {"records": records, "features": {"clip": values},
             "image_sha256": [row["image_sha256"] for row in records]}
    text = F.normalize(torch.randn(3, 8, generator=generator), dim=-1)
    return group, text, meta


class ProjectionContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_identity_initialization_then_real_train_only_updates_and_state_roundtrip(self):
        group, text, meta = fixture()
        original = group["features"]["clip"].clone()
        model = ResidualProjection(8, 4)
        torch.testing.assert_close(model(original), original)
        model, report = fit_projection(group, text, meta, options={"epochs": 3, "batch_size": 6, "bottleneck": 4})
        self.assertEqual(report["optimizer_steps"], 6)
        self.assertEqual(report["selected_epoch"], 3)
        self.assertGreater(report["parameter_delta_l2"], 0)
        self.assertGreater(report["history"][-1]["valid_supcon_anchors"], 0)
        self.assertTrue(torch.equal(group["features"]["clip"], original))
        rebuilt = load_projection(model.state_dict(), 8, 4)
        self.assertTrue(torch.equal(transform_projection(model, original), transform_projection(rebuilt, original)))
        self.assertFalse(any(p.requires_grad for p in model.parameters()))

    def test_determinism_without_advancing_callers_random_state(self):
        group, text, meta = fixture()
        before = torch.random.get_rng_state()
        first, report = fit_projection(group, text, meta, seed=91, options={"epochs": 2, "batch_size": 4})
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        second, again = fit_projection(group, text, meta, seed=91, options={"epochs": 2, "batch_size": 4})
        self.assertEqual(report, again)
        for name, value in first.state_dict().items():
            self.assertTrue(torch.equal(value, second.state_dict()[name]), name)

    def test_unknown_development_or_reordered_hashes_are_rejected(self):
        for field, value in (("status", "intra"), ("split", "val_known")):
            group, text, meta = fixture()
            group["records"][-1][field] = value
            with self.assertRaisesRegex(ValueError, "TRAIN"):
                fit_projection(group, text, meta)
        group, text, meta = fixture()
        group["image_sha256"].reverse()
        with self.assertRaisesRegex(ValueError, "ordering"):
            fit_projection(group, text, meta)

    def test_supcon_excludes_same_hash_and_skips_singletons_without_nan(self):
        x = F.normalize(torch.randn(3, 4), dim=-1).requires_grad_()
        loss, count = supervised_contrastive(x, torch.tensor([0, 0, 1]), ["a", "a", "c"])
        self.assertEqual(count, 0)
        self.assertEqual(float(loss.detach()), 0.)
        loss.backward()
        self.assertTrue(torch.equal(x.grad, torch.zeros_like(x)))
        loss, count = supervised_contrastive(x, torch.tensor([0, 0, 1]), ["a", "b", "c"])
        self.assertEqual(count, 2)
        self.assertTrue(torch.isfinite(loss))

    def test_bad_state_or_unknown_options_fail_closed(self):
        group, text, meta = fixture()
        with self.assertRaisesRegex(ValueError, "Unknown"):
            fit_projection(group, text, meta, options={"dev_early_stop": True})
        state = copy.deepcopy(ResidualProjection(8).state_dict())
        state["up.bias"][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            load_projection(state, 8)


if __name__ == "__main__":
    unittest.main()
