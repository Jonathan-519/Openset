"""CPU protocol tests for the TaxoLocal route."""

import importlib.util
import json
from pathlib import Path
import unittest

from prepro.prepare_taxolocal_data import (
    SPECIES_ASSIGNMENT,
    declared_species,
)
from taxolocal_router import apply_router, calibrate_router
from metrics_open import _near_detection_metrics


ROOT = Path(__file__).resolve().parents[1]
HAS_TORCH = importlib.util.find_spec("torch") is not None


class DataProtocolTests(unittest.TestCase):
    def test_species_assignments_are_disjoint_and_complete(self):
        rows = declared_species()
        pairs = [(parent, species) for _, parent, species in rows]
        self.assertEqual(len(pairs), len(set(pairs)))
        self.assertEqual(len(pairs), 10)
        self.assertEqual(
            {split for split, _, _ in rows},
            {"development", "locked_test"},
        )
        for split in SPECIES_ASSIGNMENT.values():
            self.assertEqual(set(split), {"Copepoda", "Medusae"})

    def test_generated_protocol_is_species_disjoint(self):
        path = (
            ROOT
            / "prepro/data/Zooplankton_Taxonomic_Tree_taxolocal_v1"
            / "split_protocol.json"
        )
        report = json.loads(path.read_text(encoding="utf-8"))
        development = set(report["development_species"])
        locked = set(report["locked_test_species"])
        self.assertFalse(development & locked)
        self.assertFalse(report["new_data_used_for_training"])
        self.assertEqual(report["development_count"], 119)
        self.assertEqual(report["locked_test_count"], 117)


class RouterTests(unittest.TestCase):
    def records(self):
        known = [
            dict(status="known", pred_parent=0, pred_leaf=0,
                 true_parent=0, true_leaf=0, parent_score=score,
                 local_known_margin=margin)
            for score, margin in ((0.9, 1.0), (0.8, 0.8), (0.7, 0.7))
        ]
        novel = [
            dict(status="intra", pred_parent=0, pred_leaf=0,
                 true_parent=0, true_leaf=None, parent_score=0.8,
                 local_known_margin=margin)
            for margin in (-0.8, -0.5)
        ]
        extra = [
            dict(status="extra", pred_parent=0, pred_leaf=0,
                 true_parent=None, true_leaf=None, parent_score=0.1,
                 local_known_margin=-0.2)
        ]
        return known, novel, extra

    def test_router_retains_parent_for_local_unknown(self):
        known, novel, extra = self.records()
        router = calibrate_router(
            known,
            novel,
            extra,
            ["P"],
            {
                "root_known_coverage_floor": 2 / 3,
                "local_known_coverage_floor": 2 / 3,
                "per_parent_min_dev_unknown": 20,
            },
        )
        rows = apply_router(known + novel + extra, router, ["P"], ["L"])
        self.assertEqual(rows[-1]["prediction_type"], "global_unknown")
        for row in rows[len(known) : len(known) + len(novel)]:
            self.assertEqual(row["prediction_type"], "intra_unknown")
            self.assertEqual(row["parent"], row["true_parent"])
            self.assertIsNone(row["leaf"])

    def test_near_metrics_use_local_margin_and_report_oscr(self):
        known = [
            dict(local_known_margin=value, candidate_parent=0,
                 true_parent=0, candidate_leaf=0, true_leaf=0)
            for value in (1.0, 0.8, 0.6)
        ]
        novel = [
            dict(local_known_margin=value) for value in (-0.2, -0.5, -0.8)
        ]
        metrics = _near_detection_metrics(known, novel)
        self.assertEqual(metrics["auroc"], 1.0)
        self.assertGreater(metrics["oscr"], 0.9)
        self.assertEqual(metrics["score"], "local_known_margin")


@unittest.skipUnless(HAS_TORCH, "Torch is not installed")
class LossTests(unittest.TestCase):
    def test_taxonomy_weighting_penalises_close_negative_more(self):
        import torch
        from losses.taxosafe_loss import taxonomy_weighted_contrastive_loss

        features = torch.tensor([
            [1.0, 0.0], [0.9, 0.1], [1.0, 0.0], [0.0, 1.0]
        ])
        leaf = torch.tensor([0, 0, 1, 2])
        close_parent = torch.tensor([0, 0, 0, 1])
        far_parent = torch.tensor([0, 0, 1, 1])
        close = taxonomy_weighted_contrastive_loss(
            features, leaf, close_parent
        )
        far = taxonomy_weighted_contrastive_loss(
            features, leaf, far_parent
        )
        self.assertGreater(float(close), float(far))


if __name__ == "__main__":
    unittest.main()
