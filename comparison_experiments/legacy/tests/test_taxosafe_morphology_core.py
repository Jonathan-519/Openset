"""Numerical matching and leakage boundaries, independent of server accuracy."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from PIL import Image
from torch.nn import functional as F

from taxosafe_morphology import features, matching, episodes, protocol, training
from taxosafe_morphology.verifier import SpatialVerifier
from taxosafe_discovery.geometry import GeometryBank
from taxosafe_discovery.verifier import build_episodes, SharedVerifier


def fixture():
    generator = torch.Generator().manual_seed(13)
    meta = dict(leaf_names=["a", "b", "c"], parent_names=["P", "Q"], leaf_to_parent=[0, 0, 1])
    labels = torch.arange(12) % 3
    vectors = F.normalize(torch.randn(12, 8, generator=generator), dim=-1)
    hashes = [hashlib.sha256(str(i).encode()).hexdigest() for i in range(12)]
    text = {"ensemble_" + level: F.normalize(torch.randn(count, 8, generator=generator), dim=-1)
            for level, count in (("leaf", 3), ("parent", 2))}
    template = {level:vectors @ text["ensemble_" + level].T for level in ("leaf", "parent")}
    old_episodes = build_episodes(vectors, vectors, labels, hashes, meta, template, folds=2)
    old = SharedVerifier.fit(old_episodes, epochs=2, batch_size=64, hidden=4)
    bank = GeometryBank.fit(vectors, vectors, labels, hashes, meta)
    payload = dict(text=text, geometry=bank.state_dict(), verifier=old.state_dict())
    view = dict(raw_clip=vectors, labels=labels, image_sha256=hashes,
        records=[dict(split="train", status="known", image_sha256=h, true_leaf=int(c), true_parent=meta["leaf_to_parent"][c])
                 for h,c in zip(hashes,labels)])
    spatial = dict(image_sha256=hashes, positions=matching.grid_positions(4),
                   tokens=F.normalize(torch.randn(12,4,8,generator=generator), dim=-1))
    return view, spatial, payload, meta


class SpatialCore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads(); torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_many_to_one_borrowing_is_impossible(self):
        cost = torch.ones(1,4,4); cost[:,:,0] = 0.
        plan, error = matching.transport(cost, .1, 5)
        self.assertLess(float(error.detach().max()), 1e-6)
        self.assertAlmostEqual(float((plan*cost).sum()), .75, places=5)
        self.assertTrue(torch.allclose(plan.sum(1), torch.full((1,4),.25), atol=1e-6))

    def test_full_mass_and_finite_gradients_on_real_grid_sizes(self):
        for size in (4,49,196):
            cost = (torch.rand(2,size,size)*2).requires_grad_()
            plan, error = matching.transport(cost,.1,8)
            (plan*cost).sum().backward()
            self.assertLess(float(error.detach().max()),2e-5)
            self.assertTrue(torch.isfinite(cost.grad).all())
            self.assertTrue(torch.allclose(plan.sum((1,2)),torch.ones(2),atol=1e-5))

    def test_zero_control_and_serialized_signed_correction(self):
        model = SpatialVerifier(8,3,4,8)
        tokens = F.normalize(torch.randn(3,4,8),dim=-1)
        delta,_,detail = model.evidence(tokens[0],tokens[1:],torch.tensor([0,1]),matching.grid_positions(4),protocol.DEFAULTS["matching"])
        self.assertEqual(float(delta.detach()),0.)
        self.assertEqual(len(detail["descriptors"]),2)
        for bias in (-2.,2.):
            model.delta[-1].bias.data.fill_(bias)
            restored = SpatialVerifier.restore(model.export_state())
            result = restored.evidence(tokens[0],tokens[1:],torch.tensor([0,1]),matching.grid_positions(4),protocol.DEFAULTS["matching"])[0]
            self.assertEqual(float(result),bias)

    def test_support_exclusion_single_child_and_unrelated_controls(self):
        view,spatial,payload,meta = fixture()
        data = episodes.build_episodes(view,payload,meta,dict(references_per_leaf=2,folds=2),1)
        labels = view["labels"]
        for item in data["report"]["episodes"]:
            self.assertFalse(set(item["support_image_sha256"]) & set(item["query_image_sha256"]))
            self.assertTrue(set(item["reference_indices"]) <= set(item["support_indices"]))
            if item["kind"] == "drop_leaf":
                self.assertFalse(bool((labels[item["support_indices"]] == item["identity"]).any()))
            if item["kind"] == "drop_parent":
                self.assertFalse(any(meta["leaf_to_parent"][labels[i]] == item["identity"] for i in item["support_indices"]))
        self.assertGreater(data["report"]["single_child_parent_near_examples_skipped"],0)
        self.assertEqual(data["report"]["single_child_parent_ids"],[1])
        self.assertTrue(any(e["kind"]=="unrelated_leaf_removal" for e in data["report"]["episodes"]))
        self.assertTrue(any(e["kind"]=="unrelated_parent_removal" for e in data["report"]["episodes"]))

    def test_unknown_training_rejected(self):
        view,spatial,payload,meta = fixture()
        view["records"][0]["status"]="extra"
        with self.assertRaises(ValueError):
            episodes.build_episodes(view,payload,meta,dict(references_per_leaf=2,folds=2),1)

    def test_actual_adapter_training_and_pure_inference(self):
        view,spatial,payload,meta = fixture()
        cfg=copy.deepcopy(protocol.DEFAULTS)
        cfg["support"]["folds"]=2
        cfg["training"].update(steps=4,batch_size=2,adapter_dim=4,hidden=8,log_every=4)
        data=episodes.build_episodes(view,payload,meta,cfg["support"],cfg["seed"])
        with patch("builtins.print"):
            model,report=training.fit_spatial(view,spatial,data,meta,cfg,"leaf")
        self.assertEqual(report["optimizer_steps"],4)
        self.assertEqual(report["initial_control_max_abs_delta"],0.)
        self.assertGreater(report["parameter_delta_l2"],0.)
        self.assertGreater(report["gradient_l2_sum"]["adapter"],0.)
        query=dict(spatial,tokens=spatial["tokens"][:2],image_sha256=["e"*64,"f"*64])
        with patch.object(GeometryBank,"fit",side_effect=AssertionError("inference fit")):
            score,details=training.score_spatial(model,query,[0,1],meta)
        self.assertTrue(torch.isfinite(score).all())
        self.assertTrue(all(d["discarded_patch_mass"]==0 for d in details))
        with self.assertRaises(ValueError):
            training.score_spatial(model,spatial,[0]*12,meta)

    def test_missing_changed_and_bad_raw_images_all_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); valid=root/"valid.png"; bad=root/"bad.png"
            Image.new("RGB",(7,11)).save(valid);bad.write_bytes(b"not an image")
            values=[(valid,protocol.file_hash(valid)),(bad,protocol.file_hash(bad)),(root/"missing.png","a"*64),(valid,"b"*64)]
            rows=[dict(resolved_path=str(p),path=p.name,image_sha256=h,status="known",split="train",source="a",true_leaf=0,true_parent=0)
                  for p,h in values]
            cache=dict(groups=dict(train=dict(records=rows,image_sha256=[h for _,h in values])))
            info=dict(reference=dict(config=dict(data=dict(data_root=str(root)))))
            _,audit=features.audited_rows(cache,info)
            self.assertFalse(audit["valid"])
            self.assertEqual(len(audit["problems"]),3)
            self.assertFalse(audit["test_images_opened"])


if __name__ == "__main__":
    unittest.main()
