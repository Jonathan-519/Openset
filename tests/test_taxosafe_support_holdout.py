"""Strict held-class folds partition TRAIN before gradients, keeping global IDs."""
from contextlib import ExitStack
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from taxosafe_support import holdout, pipeline, protocol
from tests.test_taxosafe_support_pipeline import META, SIGNATURE, TinyBackbone, fixture, loader, row


class StrictHoldoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.rows = [row("train", i) for i in range(20)]

    def test_deterministic_species_and_parent_folds_without_taxonomy_reindexing(self):
        folds = holdout.build_folds(self.rows, META, seed=7)
        self.assertEqual(folds, holdout.build_folds(self.rows, META, seed=7))
        self.assertEqual(len(folds), 6)
        for fold in folds:
            groups, queries, audit = holdout.fold_rows(self.rows, fold, META)
            held = set(fold["heldout_leaf_ids"])
            self.assertTrue(all(r["true_leaf"] not in held for rows in groups.values() for r in rows))
            self.assertTrue(all(r["true_leaf"] in held for r in queries))
            self.assertFalse(set(audit["train"]["image_hashes"]) & set(audit["heldout"]["image_hashes"]))
            self.assertEqual(len(fold["active_leaf_mask"]), len(META["leaf_names"]))
        selected = holdout.select_folds(folds, seed=17, max_folds=2)
        self.assertEqual(selected, holdout.select_folds(folds, seed=17, max_folds=2))

    def test_singleton_parent_has_no_species_fold_but_can_have_parent_fold(self):
        meta = copy.deepcopy(META)
        meta["parent_names"].append("S")
        meta["leaf_names"].append("e")
        meta["leaf_to_parent"].append(2)
        rows = self.rows + [dict(row("train", i), true_leaf=4, true_parent=2, source="e") for i in range(20, 25)]
        folds = holdout.build_folds(rows, meta)
        ids = {fold["id"] for fold in folds}
        self.assertNotIn("species_004", ids)
        self.assertIn("parent_002", ids)

    def test_rejects_unknown_or_nontraining_input_before_plan(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["split"] = "val_known"
        with self.assertRaisesRegex(ValueError, "TRAIN rows only"):
            holdout.build_folds(rows, META)
        rows[0]["split"], rows[0]["status"] = "train", "intra"
        with self.assertRaisesRegex(ValueError, "TRAIN rows only"):
            holdout.build_folds(rows, META)

    def test_sparse_leaves_never_reuse_gradients_as_inner_validation(self):
        rows = [row("train", i) for i in range(8)]
        self.assertEqual(holdout.build_folds(rows, META), [])
        rows = self.rows[:-1]
        for fold in holdout.build_folds(rows, META):
            self.assertFalse(set(fold["train_indices"]) & set(fold["val_known_indices"]))

    def test_forged_overlap_and_partial_parent_holdout_rejected(self):
        fold = holdout.build_folds(self.rows, META)[0]
        forged = copy.deepcopy(fold)
        forged["train_indices"].append(forged["heldout_indices"][0])
        with self.assertRaisesRegex(ValueError, "overlap"):
            holdout.fold_rows(self.rows, forged, META)
        forged = copy.deepcopy(fold)
        forged["kind"], forged["node"] = "parent", 0
        with self.assertRaisesRegex(ValueError, "every descendant"):
            holdout.fold_rows(self.rows, forged, META)

    def test_active_episodes_have_no_inactive_controls_or_fake_siblings(self):
        active = [False, True, True, True]
        episodes = holdout.build_active_episodes(torch.tensor([1, 2, 3]), META["leaf_to_parent"], active, seed=9)
        self.assertFalse(episodes["valid"]["drop_leaf"][0])
        self.assertTrue(episodes["valid"]["drop_leaf"][1:].all())
        self.assertEqual(episodes["targets"]["full"].tolist(), [4, 5, 6])
        self.assertEqual(episodes["targets"]["drop_leaf"][1:].tolist(), [2, 2])
        for mask in episodes["masks"].values():
            self.assertFalse(mask[:, 0].any())
        for i, label in enumerate([1, 2, 3]):
            for name in ("control_leaf", "control_parent"):
                if episodes["valid"][name][i]:
                    self.assertTrue(episodes["masks"][name][i, label])
                    self.assertTrue((episodes["masks"]["full"][i] & ~episodes["masks"][name][i]).any())
        with self.assertRaisesRegex(ValueError, "held-out label"):
            holdout.build_active_episodes(torch.tensor([0]), META["leaf_to_parent"], active)

    def test_actual_shared_training_folds_use_no_heldout_gradients_or_support(self):
        cfg, _ = fixture()
        cfg["support"].update(decoupled=True)
        folds = holdout.build_folds(self.rows, META, seed=7)
        selected = [next(f for f in folds if f["id"] == name) for name in ("species_000", "parent_000")]
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            path = Path(temp)
            manifest = path / "known_train.txt"
            manifest.write_text("fixture-only\n")
            cfg["data"]["train"] = str(manifest)
            signature = lambda config: dict(SIGNATURE, config=protocol.object_hash(config))
            gradient_labels, models = [], []

            def checked_loader(rows, config, meta, training=False):
                if training:
                    gradient_labels.append((config["strict_holdout"]["id"], {r["true_leaf"] for r in rows}))
                return loader(rows, config, meta, training)

            def backbone(*args):
                model = TinyBackbone()
                models.append(model)
                return model

            stack.enter_context(patch.object(protocol, "signature", side_effect=signature))
            stack.enter_context(patch.object(pipeline, "signature", side_effect=signature))
            stack.enter_context(patch.object(pipeline, "hierarchy", return_value=META))
            stack.enter_context(patch.object(pipeline, "make_backbone", side_effect=backbone))
            stack.enter_context(patch.object(pipeline, "make_loader", side_effect=checked_loader))
            stage_reader = stack.enter_context(patch.object(pipeline, "load_stage_rows", side_effect=AssertionError("external manifests forbidden")))
            raw_reader = stack.enter_context(patch.object(protocol, "read_split", side_effect=AssertionError("external manifests forbidden")))
            stack.enter_context(patch("sys.stdout", new_callable=io.StringIO))
            report = holdout.run_validation(cfg, path, torch.device("cpu"), selected, self.rows, META)
            self.assertEqual(len(report["reports"]), 2)
            self.assertEqual(gradient_labels, [("species_000", {1, 2, 3}), ("parent_000", {2, 3})])
            self.assertEqual(len(models), 4)  # Fresh construction plus frozen reload per fold.
            stage_reader.assert_not_called()
            raw_reader.assert_not_called()
            for fold in selected:
                fold_dir = path / "holdout" / fold["id"]
                payload = pipeline._load_torch(fold_dir / "training/support.pth")
                bank = payload["bank"]
                self.assertFalse(set(bank["labels"].tolist()) & set(fold["heldout_leaf_ids"]))
                self.assertEqual(bank["required_leaf_mask"].tolist(), fold["active_leaf_mask"])
                checkpoint = pipeline._load_torch(fold_dir / "training/best.pth")
                self.assertEqual(checkpoint["meta"], META)
                receipt = protocol.read_json(fold_dir / "completed.json")
                self.assertFalse(receipt["heldout_used_for_checkpoint_selection"])
                predictions = [json.loads(line) for line in (fold_dir / "predictions.jsonl").read_text().splitlines()]
                held = [r for r in predictions if r["evaluation_role"] == "heldout"]
                for r in held:
                    expected = 1 if fold["kind"] == "species" else 0
                    self.assertEqual(r["expected_node"], expected)
                    self.assertTrue(all(r["log_probs"][3 + c] is None for c in fold["heldout_leaf_ids"]))
            with self.assertRaisesRegex(ValueError, "Stage already exists"):
                holdout.run_validation(cfg, path, torch.device("cpu"), selected, self.rows, META)


if __name__ == "__main__":
    unittest.main()
