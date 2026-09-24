"""Run the real v11 orchestration on a small CPU encoder and filesystem data.

Only the image encoder/loader and repository taxonomy metadata are replaced.
Losses, synthesis, selection, artifact hashes, calibration, audits and reporting
are exercised together. These are software tests, not plankton benchmarks.
"""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
import yaml

from prepro.build_taxosafe_v11_known_splits import prepare
from taxosafe_dcbs import pipeline, protocol
from tests.test_taxosafe_dcbs import CENTRES, CONFIG, META


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.prompt_learner = nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            self.prompt_learner.weight.copy_(torch.eye(4))
        self.model = nn.Module()
        self.model.register_buffer("logit_scale", torch.tensor(3.))

    def encode_image(self, images, normalize=True):
        result = self.prompt_learner(images.float())
        return F.normalize(result, dim=-1) if normalize else result

    def encode_text(self, names, normalize=True):
        values = CENTRES if names == META["leaf_names"] else torch.tensor([[1., 0., 0., 0.], [-1., 0., 0., 0.], [0., 1., 0., 0.]])
        return values.to(self.prompt_learner.weight.device)


def tiny_loader(rows, cfg, meta, training=False):
    features = torch.tensor([json.loads(Path(r["resolved_path"]).read_text())["feature"] for r in rows])
    labels = torch.tensor([r["true_leaf"] if r["status"] == "known" else r["true_parent"] if r["status"] == "intra" else -1 for r in rows])
    return DataLoader(TensorDataset(features, labels, torch.arange(len(rows))), batch_size=10, shuffle=training)


def fixture(root):
    cfg = protocol.effective_config(CONFIG, "main", 12)
    cfg["data"].pop("known_preparation_audit")
    cfg["model"]["prec"] = "fp32"
    cfg["data"].update(num_known_leaves=5, n_workers=0, batch_size=10, sampler={"name": "random"})
    hierarchy = root / "tree.npy"
    hierarchy.write_bytes(b"tiny taxonomy used only for a receipt hash")
    cfg["data"]["hierarchy"] = str(hierarchy)
    cfg["training"].update(epochs=20, warmup_epochs=1, min_synthesis_epochs=2, prompt_lr=.001, head_lr=.05, patience=0)
    cfg["dcbs"].update(projection_dim=16, logit_scale=5.)
    cfg["dcbs"]["loss"]["ha"] = 0.
    cfg["dcbs"]["synthesis"].update(candidates_per_parent=32, keep_per_parent=16, radius_prior_count=0., near_noise=0.)
    cfg["calibration"].update(root_known_retention=.8, root_near_retention=.5, local_known_retention=.7, min_known_e2e=.5, max_closed_drop=.5)
    for key in ("data_root", "near_dev_root", "near_test_root", "ood_dev_root", "ood_test_root"):
        directory = root / key
        directory.mkdir()
        cfg["data"][key] = str(directory)
    generator = torch.Generator().manual_seed(1234)
    for split in ("train", "val_known", "val_intra", "val_extra", "test_known", "test_intra", "test_extra"):
        image_root = protocol.split_root(cfg, split)
        count = 60 if split == "train" else 30
        lines = []
        for i in range(count):
            if split.endswith("intra"):
                label = i % 2
                relative = "{}/near_{}_{}/{}.json".format(META["parent_names"][label], split, label, i)
                feature = torch.tensor([1. if label == 0 else -1., 0., 0., 0.])
            elif split.endswith("extra"):
                label = -1
                relative = "ood_{}_{}/{}.json".format(split, i % 2, i)
                feature = torch.tensor([0., -1., 0., .1 if i % 2 else -.1])
            else:
                label = i % 5
                relative = "{}/{}/{}_{}.json".format(META["parent_names"][META["leaf_to_parent"][label]], META["leaf_names"][label], split, i)
                feature = CENTRES[label]
            feature = F.normalize(feature + .005 * torch.randn(4, generator=generator), dim=0)
            path = image_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"feature": feature.tolist(), "id": split + str(i)}))
            lines.append("{},{},{}".format(relative, label, i))
        manifest = root / (split + ".txt")
        manifest.write_text("\n".join(lines) + "\n")
        cfg["data"][split] = str(manifest)
    return cfg


class PipelineTest(unittest.TestCase):
    def test_train_calibrate_and_frozen_test_without_fitting_manifests(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                cfg, run = fixture(root), root / "run"
                with patch.object(pipeline, "hierarchy", return_value=META), patch.object(pipeline, "make_backbone", side_effect=lambda *args: TinyEncoder()), patch.object(pipeline, "make_loader", side_effect=tiny_loader), contextlib.redirect_stdout(io.StringIO()):
                    with patch.object(protocol, "read_split", wraps=protocol.read_split) as read:
                        pipeline.train(cfg, run, torch.device("cpu"))
                    self.assertEqual([c.args[1] for c in read.call_args_list], ["train", "val_known"])
                    trained = protocol.read_json(run / "training/completed.json")
                    self.assertEqual(trained["gradient_splits"], ["train"])
                    self.assertFalse(trained["test_used_for_fitting"])
                    self.assertGreaterEqual(trained["best_epoch"], 3)
                    self.assertTrue(all(v >= 2 for v in trained["synthesis_epochs"].values()))
                    with patch.object(protocol, "read_split", wraps=protocol.read_split) as read:
                        pipeline.calibrate_run(cfg, run, torch.device("cpu"))
                    self.assertEqual([c.args[1] for c in read.call_args_list], ["val_known", "val_intra", "val_extra"])
                    frozen = protocol.file_hash(run / "calibration/router.json")
                    for split in ("train", "val_known", "val_intra", "val_extra"):
                        Path(cfg["data"][split]).unlink()
                    with patch.object(protocol, "read_split", wraps=protocol.read_split) as read:
                        pipeline.test_run(cfg, run, torch.device("cpu"))
                    self.assertEqual([c.args[1] for c in read.call_args_list], ["test_known", "test_intra", "test_extra"])
                    self.assertEqual(protocol.file_hash(run / "calibration/router.json"), frozen)
                    metrics = protocol.read_json(run / "test/metrics.json")
                    self.assertEqual([metrics[s]["sample_count"] for s in ("known", "intra", "extra")], [30, 30, 30])
                    self.assertTrue(metrics["per_intra_species"])
                    self.assertTrue(metrics["per_extra_source"])
                    rows = [json.loads(line) for line in (run / "test/predictions.jsonl").read_text().splitlines()]
                    self.assertEqual(len(rows), 90)
                    self.assertTrue(all(r["evaluation_weight"] == 1 for r in rows))
                    self.assertEqual(len(protocol.read_json(run / "test/gates.json")["checks"]), 6)
                    changed = copy.deepcopy(cfg)
                    changed["calibration"]["min_known_e2e"] = .1
                    with self.assertRaisesRegex(ValueError, "changed since training"):
                        pipeline.load_trained(changed, run, torch.device("cpu"))
        finally:
            torch.set_num_threads(old_threads)

    def test_known_preparation_preserves_locked_test_and_original_manifests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = fixture(root)
            def image(split, index):
                relative = Path(cfg["data"][split]).read_text().splitlines()[index].rsplit(",", 2)[0]
                return protocol.split_root(cfg, split) / relative
            image("val_known", 0).write_bytes(image("train", 0).read_bytes())
            image("val_known", 1).write_bytes(image("test_known", 1).read_bytes())
            splits = ("train", "val_known", "test_known")
            hashes = {s: protocol.file_hash(cfg["data"][s]) for s in splits}
            path = root / "config.yml"
            path.write_text(yaml.safe_dump(cfg))
            report = prepare(path, root / "prepared", root)
            self.assertEqual(report["retained_counts"], {"train": 59, "val_known": 29, "test_known": 30})
            self.assertFalse(report["unknown_images_read"])
            self.assertTrue(report["test_known_used_only_for_identity_audit"])
            self.assertEqual(hashes, {s: protocol.file_hash(cfg["data"][s]) for s in splits})
            self.assertEqual(report, prepare(path, root / "prepared", root))


if __name__ == "__main__":
    unittest.main()
