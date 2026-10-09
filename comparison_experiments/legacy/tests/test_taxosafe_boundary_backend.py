"""An internally rehashed child cache still cannot replace frozen D05 evidence."""
import copy
from pathlib import Path
import tempfile
import unittest

import torch

from taxosafe_boundary import backend, protocol
from taxosafe_support import pipeline as support


class BoundaryFrozenCache(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.suite = self.root / "boundary"
        self.parent = self.root / "discovery"
        self.output = self.suite / "cache/train"
        self.output.mkdir(parents=True)
        parent = self.parent / "cache/train"
        parent.mkdir(parents=True)
        self.cfg = copy.deepcopy(protocol.DEFAULTS)
        meta = dict(leaf_names=["a", "b"], parent_names=["p"], leaf_to_parent=[0, 0])
        hashes = ["1" * 64, "2" * 64]
        feature = torch.eye(2)
        rows = [dict(image_sha256=h, split="train", status="known", true_leaf=i,
                     true_parent=0, global_pred_leaf=i, support_evidence={"candidate": i})
                for i, h in enumerate(hashes)]
        rows.append(dict(rows[0], image_path="alias.png"))
        self.cache = dict(meta=meta, groups={"train": dict(records=rows, image_sha256=hashes,
            record_feature_indices=[0, 1, 0], features={key: feature.clone() for key in
                ("clip", "source_fine", "source_parent")})},
            text={"single_leaf":feature.clone(), "ensemble_leaf":feature.clone(),
                  "single_parent":feature[:1].clone(), "ensemble_parent":feature[:1].clone()},
            provenance=dict(source_binding={"reference": "immutable"}, preprocessing={},
                clip_core_sha256="core", clip_initialization="inherited", templates_sha256="templates"),
            timings={"image_forward_count":2})
        support._save_torch(parent / "features.pth", dict(cache=self.cache))
        protocol.write_json(parent / "completed.json", {"source": "signed"})
        self.info = dict(directory=self.parent, binding={"directory":str(self.parent)}, meta=meta,
            reference=dict(meta=meta, binding=self.cache["provenance"]["source_binding"],
                           config={"data":{}}, training={"audit":{"train":{"image_hashes":hashes}}}),
            training={"inference_spec_sha256":backend.legacy._text_contract(self.cache)})
        self.header = backend._header(self.cfg, self.info, "train")
        self.binding = dict(parent_cache_sha256=protocol.file_hash(parent / "features.pth"),
            parent_cache_receipt_sha256=protocol.file_hash(parent / "completed.json"),
            inference_spec_sha256=self.info["training"]["inference_spec_sha256"], audit={"train":hashes})

    def _write_child(self, cache):
        payload = dict(self.header, **self.binding, cache=cache)
        support._save_torch(self.output / "features.pth", payload)
        receipt = dict(self.header, **self.binding, artifacts={"features":{
            "path":"features.pth", "sha256":protocol.file_hash(self.output / "features.pth")}})
        protocol.write_json(self.output / "completed.json", receipt)

    def test_exact_cache_copy_loads_without_image_forward(self):
        self._write_child(copy.deepcopy(self.cache))
        loaded, _ = backend._load_cache(self.suite, "train", self.cfg, self.info)
        self.assertEqual(backend._semantic(loaded), backend._semantic(self.cache))

    def test_coherently_rehashed_features_labels_candidates_and_aliases_are_rejected(self):
        def feature(cache):
            cache["groups"]["train"]["features"]["clip"][0, 0] += .1

        def labels(cache):
            for row in cache["groups"]["train"]["records"]:
                row["true_leaf"] = 1 - row["true_leaf"]

        def candidates(cache):
            for row in cache["groups"]["train"]["records"]:
                row["global_pred_leaf"] = 1 - row["global_pred_leaf"]

        def aliases(cache):
            cache["groups"]["train"]["records"][-1]["image_path"] = "changed_alias.png"

        parent_digest = protocol.file_hash(self.parent / "cache/train/features.pth")
        for change in (feature, labels, candidates, aliases):
            with self.subTest(change=change.__name__):
                cache = copy.deepcopy(self.cache)
                change(cache)
                self._write_child(cache)
                with self.assertRaisesRegex(ValueError, "frozen D05 cache contents"):
                    backend._load_cache(self.suite, "train", self.cfg, self.info)
                self.assertEqual(protocol.file_hash(self.parent / "cache/train/features.pth"), parent_digest)


if __name__ == "__main__":
    unittest.main()
