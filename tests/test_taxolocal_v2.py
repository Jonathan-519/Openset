"""CPU unit tests for TaxoLocal-v2 boundaries (not accuracy results)."""

import unittest

import numpy as np
from taxolocal_v2_router import apply_router, calibrate_router

try:
    import torch
    from losses.taxosafe_loss import sibling_boundary_unknown_loss
except ModuleNotFoundError:  # Lightweight CI may omit the training runtime.
    torch = None
    sibling_boundary_unknown_loss = None


def _record(status, feature, parent_lse, margin, leaf, parent, true_leaf=None):
    return {
        "status": status,
        "image_feature": list(feature),
        "parent_logsumexp": float(parent_lse),
        "local_known_margin": float(margin),
        "pred_leaf": int(leaf),
        "pred_parent": int(parent),
        "global_pred_leaf": int(leaf),
        "parent_score": float(parent_lse),
        "true_leaf": true_leaf,
        "true_parent": None if status == "extra" else int(parent),
        "path": "{}/sample.jpg".format(status),
    }


class TaxoLocalV2Tests(unittest.TestCase):
    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_sibling_boundary_loss_is_differentiable(self):
        image = torch.nn.functional.normalize(
            torch.randn(6, 8, requires_grad=True), dim=-1
        )
        leaf_text = torch.nn.functional.normalize(
            torch.randn(3, 8, requires_grad=True), dim=-1
        )
        unknown_text = torch.nn.functional.normalize(
            torch.randn(2, 8, requires_grad=True), dim=-1
        )
        leaf = torch.tensor([0, 1, 0, 1, 2, 2])
        parent = torch.tensor([0, 0, 0, 0, 1, 1])
        meta = {
            "parent_names": ["A", "B"],
            "children_by_parent": [torch.tensor([0, 1]), torch.tensor([2])],
        }
        result = sibling_boundary_unknown_loss(
            image, leaf_text, unknown_text, leaf, parent, meta,
            scale=torch.tensor(10.0), pairs_per_parent=4,
        )
        self.assertGreater(result["pair_count"], 0)
        self.assertTrue(torch.isfinite(result["loss"]))
        result["loss"].backward()
        self.assertIsNotNone(image.grad_fn)

    def test_dual_router_fits_and_applies_without_test_data(self):
        train = []
        centers = np.eye(4)
        for leaf in range(4):
            for offset in (0.0, 0.02, -0.02):
                feature = centers[leaf] + offset
                train.append(_record(
                    "known", feature, 5.0, 4.0, leaf, leaf // 2,
                    true_leaf=leaf,
                ))
        known = [
            _record("known", centers[i], 6.0, 5.0, i, i // 2, i)
            for i in range(4)
        ] * 3
        novel = [
            _record("intra", [0.7, 0.7, 0.0, 0.0], 5.0, -1.0, 0, 0),
            _record("intra", [0.0, 0.0, 0.7, 0.7], 5.0, -1.0, 2, 1),
        ] * 2
        extra = [
            _record("extra", [0.5, -0.5, 0.5, -0.5], -2.0, 0.0, 0, 0),
            _record("extra", [-0.5, 0.5, -0.5, 0.5], -2.0, 0.0, 1, 0),
        ] * 2
        router = calibrate_router(
            train, known, novel, extra,
            parent_names=["A", "B"],
            leaf_names=["a", "b", "c", "d"],
            settings={
                "vim_principal_dim": 1,
                "root_known_coverage_floor": 0.9,
                "root_novel_coverage_floor": 0.5,
                "local_known_coverage_floor": 0.9,
                "per_parent_min_dev_unknown": 1,
            },
        )
        self.assertEqual(router["schema_version"], 2)
        self.assertNotIn("test", router["reference_bank"]["source"])
        output = apply_router(
            known + novel + extra, router,
            ["A", "B"], ["a", "b", "c", "d"],
        )
        self.assertEqual(len(output), len(known) + len(novel) + len(extra))
        self.assertTrue(all("vim_residual" in row for row in output))
        self.assertTrue(all("local_knownness_score" in row for row in output))
        self.assertTrue(all(
            row["prediction_type"] in {"known", "intra_unknown", "global_unknown"}
            for row in output
        ))


if __name__ == "__main__":
    unittest.main()
