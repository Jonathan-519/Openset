"""CPU protocol tests for the TaxoLocal-v2.1 router upgrade."""

import unittest

from taxolocal_v21_router import apply_router, upgrade_router


def record(status, index):
    if status == "known":
        values = (0.9, -0.05, 0.20, 0.10, 2.0)
        true_leaf, true_parent = 0, 0
    elif status == "intra":
        # Far from known leaves but still semantically localized to the parent.
        values = (0.7, -0.15, 0.95, 0.85, -2.0)
        true_leaf, true_parent = None, 0
    else:
        values = (0.05, -0.95, 1.10, 1.00, 0.0)
        true_leaf, true_parent = None, None
    margin, neg_entropy, residual, distance, local = values
    return {
        "status": status,
        "dataset_index": index,
        "path": "{}/{}.jpg".format(status, index),
        "pred_parent": 0,
        "pred_leaf": 0,
        "true_parent": true_parent,
        "true_leaf": true_leaf,
        "parent_margin": margin,
        "parent_neg_entropy": neg_entropy,
        "vim_residual": residual,
        "prototype_distance": distance,
        "local_knownness_score": local,
    }


def base_router():
    return {
        "schema_version": 2,
        "method": "TaxoLocal-v2-dual-boundary",
        "decision": "legacy",
        "parent_names": ["P"],
        "leaf_names": ["L"],
        "reference_bank": {},
        "global_fusion": {"legacy": True},
        "local_fusion": {},
        "root_threshold": -4.0,
        "pooled_local_knownness_threshold": 0.0,
        "branches": {
            "P": {
                "local_knownness_threshold": 2.0,
                "source": "parent_specific",
                "known_count": 50,
                "development_unknown_count": 20,
            }
        },
        "settings": {},
        "development_operating_point": {},
    }


class TaxoLocalV21Tests(unittest.TestCase):
    def setUp(self):
        self.development = (
            [record("known", i) for i in range(20)]
            + [record("intra", i) for i in range(20)]
            + [record("extra", i) for i in range(20)]
        )

    def test_upgrade_preserves_fitted_state_and_shrinks_local_threshold(self):
        original = base_router()
        upgraded = upgrade_router(
            original,
            self.development,
            {
                "root_known_coverage_floor": 0.95,
                "root_novel_coverage_floor": 0.85,
                "root_grid_quantiles": 11,
                "local_threshold_prior_count": 50,
            },
        )
        self.assertEqual(upgraded["schema_version"], 3)
        self.assertIn("root_intersection_gate", upgraded)
        self.assertEqual(original["branches"]["P"]["local_knownness_threshold"], 2.0)
        self.assertAlmostEqual(
            upgraded["branches"]["P"]["local_knownness_threshold"], 1.0
        )
        self.assertEqual(upgraded["reference_bank"], original["reference_bank"])

    def test_intersection_retains_near_unknown_and_rejects_far_unknown(self):
        upgraded = upgrade_router(
            base_router(), self.development,
            {"root_grid_quantiles": 11, "local_threshold_prior_count": 50},
        )
        output = apply_router(
            self.development, upgraded, ["P"], ["L"],
            scores_already_added=True,
        )
        known = [row for row in output if row["status"] == "known"]
        novel = [row for row in output if row["status"] == "intra"]
        extra = [row for row in output if row["status"] == "extra"]
        self.assertTrue(all(row["prediction_type"] == "known" for row in known))
        self.assertTrue(all(
            row["prediction_type"] == "intra_unknown" for row in novel
        ))
        self.assertTrue(all(
            row["prediction_type"] == "global_unknown" for row in extra
        ))
        self.assertTrue(all("root_semantic_margin" in row for row in output))
        self.assertTrue(all("root_manifold_margin" in row for row in output))


if __name__ == "__main__":
    unittest.main()
