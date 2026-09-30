"""CPU integration contracts for the new method, without CLIP or user images."""
from contextlib import ExitStack
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from taxosafe_support import pipeline, protocol

META = {"parent_names": ["P", "Q"], "leaf_names": ["a", "b", "c", "d"],
        "leaf_to_parent": [0, 0, 1, 1]}
CENTRES = torch.tensor([[1., 0., .4, 0.], [1., 0., -.4, 0.],
                        [-1., 0., 0., .4], [-1., 0., 0., -.4]])
SIGNATURE = {"config": "fixture", "hierarchy": "fixture", "code": "fixture"}


class TinyBackbone(nn.Module):
    dimension = 4

    def __init__(self):
        super().__init__()
        self.prompt_learner = nn.Parameter(torch.zeros(4))
        self.model = nn.Module()
        self.model.register_buffer("logit_scale", torch.tensor(2.0))
        self.visual_calls = 0

    def encode_image_with_spatial(self, images, normalize=True):
        self.visual_calls += 1
        global_features = F.normalize(images + self.prompt_learner, dim=-1)
        local = torch.stack((global_features, global_features.roll(1, -1) * .1 + global_features), 1)
        return global_features, F.normalize(local, dim=-1)

    def encode_text(self, names, normalize=True):
        prototypes = {name: CENTRES[c] for c, name in enumerate(META["leaf_names"])}
        prototypes.update(P=torch.tensor([1., 0., 0., 0.]), Q=torch.tensor([-1., 0., 0., 0.]))
        return F.normalize(torch.stack([prototypes[name] for name in names]), dim=-1)


class TensorRows(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        label = row["true_leaf"] if row["status"] == "known" else -1
        return torch.tensor(row["vector"]), label, i


def loader(rows, cfg, meta, training=False):
    return DataLoader(TensorRows(rows), batch_size=4, shuffle=training)


def row(split, i, status="known", leaf=None, parent=None):
    leaf = i % 4 if status == "known" and leaf is None else leaf
    parent = META["leaf_to_parent"][leaf] if status == "known" else parent
    if status == "known":
        vector = CENTRES[leaf] + torch.tensor([0., .003 * (i + 1), 0., 0.])
        source = META["leaf_names"][leaf]
    elif status == "intra":
        vector = torch.tensor([1. if parent == 0 else -1., .2, 0., 0.])
        source = split + "_species_" + str(parent)
    else:
        vector = torch.tensor([0., 1., 0., 0.])
        source = split + "_source_" + str(i)
    name = split + "_" + str(i)
    return {"path": source + "/" + name, "resolved_path": "/fixture/" + name,
            "image_sha256": hashlib.sha256(name.encode()).hexdigest(), "status": status,
            "split": split, "source": source, "true_leaf": leaf, "true_parent": parent,
            "dataset_index": i, "vector": vector.tolist()}


def fixture():
    cfg = {"seed": 7, "variant": "main", "data": {}, "model": {},
           "support": {"adapter_dim": 3, "local_tokens": 2, "max_per_leaf": 2,
                       "hidden_dim": 4, "temperature": .2, "episodes_enabled": True,
                       "loss": {"anchor": .5}},
           "training": {"epochs": 2, "warmup_epochs": 1, "min_episode_epochs": 1,
                        "episode_ramp_epochs": 1, "prompt_lr": .001, "head_lr": .003,
                        "patience": 0},
           "calibration": {"parent_bias_grid": [-1., 0., 1.], "leaf_bias_grid": [-1., 0., 1.],
                           "source_loo": False}}
    groups = {"train": [row("train", i) for i in range(8)],
              "val_known": [row("val_known", i) for i in range(4)],
              "test_known": [row("test_known", i) for i in range(4)]}
    for split in ("val_intra", "test_intra"):
        groups[split] = [row(split, i, "intra", parent=i % 2) for i in range(4)]
    for split in ("val_extra", "test_extra"):
        groups[split] = [row(split, i, "extra") for i in range(2)]
    alias = dict(groups["test_known"][0], path="a/alias", resolved_path="/fixture/alias", dataset_index=4)
    groups["test_known"].append(alias)
    return cfg, groups


class PipelineContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.cfg, self.groups = fixture()
        self.calls = []
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name) / "run"
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.object(pipeline, "hierarchy", return_value=META))
        stack.enter_context(patch.object(pipeline, "signature", return_value=SIGNATURE))
        stack.enter_context(patch.object(pipeline, "make_loader", side_effect=loader))
        stack.enter_context(patch.object(pipeline, "make_backbone", side_effect=lambda *args: TinyBackbone()))
        stack.enter_context(patch.object(pipeline, "load_stage_rows", side_effect=self.load_stage))
        stack.enter_context(patch("sys.stdout", new_callable=io.StringIO))

    def load_stage(self, cfg, stage, meta, forbidden_hashes=(), forbidden_sources=()):
        allowed = protocol.STAGE_SPLITS[stage]
        self.calls.append((stage, allowed))
        selected = {name: copy.deepcopy(self.groups[name]) for name in allowed}
        audit = protocol.audit_rows(selected, forbidden_hashes, forbidden_sources,
                                    allow_within_split=stage == "test")
        for split in audit:
            audit[split]["manifest_sha256"] = "fixture_" + split
        return selected, audit

    def train(self, debug=False):
        pipeline.train(self.cfg, self.directory, torch.device("cpu"), debug=debug)

    def test_cpu_train_calibrate_test_receipts_and_unique_metrics(self):
        self.train()
        self.assertEqual(self.calls, [("train", ("train", "val_known"))])
        training = protocol.read_json(self.directory / "training/completed.json")
        self.assertEqual(training["gradient_splits"], ["train"])
        self.assertEqual(training["best_epoch"], 2)
        self.assertGreater(training["valid_episode_queries"]["drop_leaf"], 0)
        self.assertGreater(training["valid_episode_queries"]["drop_parent"], 0)
        self.assertEqual(training["anchor"]["source"], "known_only_warmup")
        pipeline.calibrate_run(self.cfg, self.directory, torch.device("cpu"))
        calibration = protocol.read_json(self.directory / "calibration/completed.json")
        self.assertTrue(calibration["fit_completed"])
        report = protocol.read_json(self.directory / "calibration/validation_report.json")
        self.assertIn("infeasibility", report)
        # Inference must work without access to TRAIN or development manifests.
        for split in ("train", "val_known", "val_intra", "val_extra"):
            del self.groups[split]
        pipeline.test_run(self.cfg, self.directory, torch.device("cpu"))
        metrics = protocol.read_json(self.directory / "test/metrics.json")
        all_rows = protocol.read_json(self.directory / "test/metrics_all_rows.json")
        summary = protocol.read_json(self.directory / "test/summary.json")
        self.assertEqual(metrics["known"]["sample_count"], 4)
        self.assertEqual(all_rows["known"]["sample_count"], 5)
        self.assertEqual(summary["duplicate_record_count"], 1)
        predictions = [json.loads(line) for line in (self.directory / "test/predictions.jsonl").read_text().splitlines()]
        aliases = [r for r in predictions if r["image_sha256"] == self.groups["test_known"][0]["image_sha256"]]
        self.assertEqual([r["evaluation_weight"] for r in aliases], [1, 0])
        self.assertEqual(aliases[0]["log_probs"], aliases[1]["log_probs"])
        with self.assertRaisesRegex(ValueError, "Stage already exists"):
            pipeline.test_run(self.cfg, self.directory, torch.device("cpu"))

    def test_debug_checkpoint_cannot_calibrate_or_test(self):
        self.train(debug=True)
        with self.assertRaisesRegex(ValueError, "Debug runs cannot"):
            pipeline.calibrate_run(self.cfg, self.directory, torch.device("cpu"))
        self.assertFalse((self.directory / "calibration").exists())

    def test_tampered_reference_artifact_rejected_before_unknown_data_read(self):
        self.train()
        path = self.directory / "training/support.pth"
        with path.open("ab") as handle:
            handle.write(b"tampered")
        with self.assertRaisesRegex(ValueError, "Artifact hash mismatch"):
            pipeline.calibrate_run(self.cfg, self.directory, torch.device("cpu"))
        self.assertEqual(len(self.calls), 1)

    def test_training_content_cannot_reappear_in_development(self):
        self.train()
        self.groups["val_extra"][0]["image_sha256"] = self.groups["train"][0]["image_sha256"]
        with self.assertRaisesRegex(ValueError, "overlap"):
            pipeline.calibrate_run(self.cfg, self.directory, torch.device("cpu"))
        self.assertFalse((self.directory / "calibration").exists())

    def test_checkpoint_selection_validation_cannot_change_before_calibration(self):
        self.train()
        self.groups["val_known"][0]["image_sha256"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "Validation known data changed"):
            pipeline.calibrate_run(self.cfg, self.directory, torch.device("cpu"))

    def test_intervention_loss_backpropagates_to_query_encoder(self):
        encoder = pipeline.SupportEncoder(TinyBackbone(), META, self.cfg["support"])
        evidence = pipeline._make_evidence(encoder, self.cfg, META, torch.device("cpu"))
        bank, _, _ = pipeline.reference_bank(encoder, loader(self.groups["train"], self.cfg, META),
                                              self.groups["train"], self.cfg, META, torch.device("cpu"))
        query = encoder(CENTRES)
        config = copy.deepcopy(self.cfg)
        config["support"]["loss"] = {"leaf": 0., "parent": 0., "support_leaf": 0.,
                                     "support_parent": 0., "episode": 1., "paired": .25, "control": .25}
        loss, _, counts = pipeline.training_loss(query, torch.arange(4), torch.arange(4),
                           [r["image_sha256"] for r in self.groups["train"][:4]], bank,
                           evidence, config, META, 7, 1.)
        loss.backward()
        grad = encoder.backbone.prompt_learner.grad
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(float(grad.abs().sum()), 0.)
        self.assertEqual(counts["drop_leaf"], 4)
        self.assertTrue(all(not value.requires_grad for value in (bank.parent, bank.fine)))


if __name__ == "__main__":
    unittest.main()
