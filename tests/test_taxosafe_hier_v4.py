"""Mechanism and artifact tests. Synthetic features are NOT biological results."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
import yaml

from taxosafe_hier.model import (HierEvidence, sample_episode, episode_loss,
                                 train_adapter, score_adapter, save_adapter, load_adapter)
from taxosafe_hier.extraction import spatial_tokens
from taxosafe_hier.calibration import protected_thresholds
from taxosafe_hier import pipeline
from taxosafe_visual import core, runtime
from tools import run_taxosafe_hier_v4 as runner


def settings():
    value = yaml.safe_load((Path(__file__).resolve().parents[1] / runner.CONFIG).read_text())
    value["adapter"].update(rank=4, epochs=2, episodes_per_epoch=8)
    value["query_chunk_size"] = 7
    return value


def fixture():
    rng = np.random.RandomState(21)
    meta = {"parent_names": ["A", "B"], "leaf_names": ["a", "b", "c"], "leaf_to_parent": [0, 0, 1]}
    parent = np.eye(10, dtype=np.float32)[:2]
    leaf = np.eye(10, dtype=np.float32)[2:5] + parent[np.asarray(meta["leaf_to_parent"])]
    leaf = core.normalise(leaf).astype(np.float32)
    def make(split):
        rows, vectors = [], []
        statuses = ("known",) if split == "train" else ("known", "intra", "extra")
        for status in statuses:
            count = 3 if status == "known" else 2
            for group in range(count):
                for i in range(8):
                    tl = group if status == "known" else None
                    tp = meta["leaf_to_parent"][tl] if tl is not None else group if status == "intra" else None
                    center = leaf[tl] if tl is not None else (
                        parent[tp] + np.eye(10)[5 + group] if status == "intra" else np.eye(10)[8 + group])
                    vectors.append(center + rng.normal(0, .05, 10))
                    tag = "{}_{}_{}_{}".format(split, status, group, i)
                    rows.append({"split": "train" if split == "train" else split + "_" + status,
                                 "status": status, "true_leaf": tl, "true_parent": tp,
                                 "image_sha256": hashlib.sha256(tag.encode()).hexdigest(),
                                 "source": status + str(group), "path": tag})
        g = core.normalise(vectors).astype(np.float32)
        patches = np.stack([core.normalise(g + rng.normal(0, .03, g.shape)) for _ in range(4)], axis=1).astype(np.float32)
        return {"global": g, "patches": patches, "parent_cosine": g @ parent.T, "leaf_cosine": g @ leaf.T,
                "parent_text": parent, "leaf_text": leaf}, rows
    return meta, make


class HierEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_identity_and_orthogonal_increment_not_projecting_global(self):
        meta, make = fixture(); d, _ = make("train")
        model = HierEvidence(d["parent_text"], d["leaf_text"], meta["leaf_to_parent"], settings()["adapter"])
        g, p = torch.tensor(d["global"]), torch.tensor(d["patches"])
        torch.testing.assert_close(model(g, p, 0), g)
        with torch.no_grad():
            model.up.normal_()
        z, delta, _ = model(g, p, 0, details=True)
        torch.testing.assert_close(delta @ model.basis, torch.zeros(len(g), 2), atol=1e-6, rtol=0)
        self.assertTrue(torch.all(delta.norm(dim=-1) < .5))
        torch.testing.assert_close((g + delta) @ model.basis, g @ model.basis, atol=1e-6, rtol=0)
        self.assertFalse(torch.allclose(z, g))

    def test_held_text_is_absent_from_attention(self):
        meta, make = fixture(); d, _ = make("train")
        model = HierEvidence(d["parent_text"], d["leaf_text"], meta["leaf_to_parent"], settings()["adapter"])
        with torch.no_grad(): model.up.normal_()
        g, p = torch.tensor(d["global"]), torch.tensor(d["patches"])
        a = model(g, p, 0, active_leaves=[0])
        with torch.no_grad(): model.leaf_text[1].fill_(999.)
        b = model(g, p, 0, active_leaves=[0])
        torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_sample_disjoint_and_open_loss_has_finite_gradient(self):
        meta, make = fixture(); data, rows = make("train")
        y = np.array([r["true_leaf"] for r in rows])
        s, q = sample_episode(y, [0, 1], np.random.RandomState(1), 4, 4)
        self.assertFalse(set(s) & set(q))
        model = HierEvidence(data["parent_text"], data["leaf_text"], meta["leaf_to_parent"], settings()["adapter"])
        config = settings()["adapter"]
        config["open_margin"] = 2.0  # Force a nontrivial hard-negative hinge.
        loss, logs = episode_loss(model, torch.tensor(data["global"]), torch.tensor(data["patches"]),
                                  torch.tensor(y), 0, s, q, 1, config)
        loss.backward()
        self.assertTrue(torch.isfinite(model.up.grad).all())
        self.assertGreater(float(model.up.grad.abs().sum()), 0)
        self.assertTrue(np.isfinite(logs["open"]))
        self.assertGreater(logs["open"], 0)
        modified = torch.tensor(data["global"])
        modified[s[y[s] == 1]] = -modified[s[y[s] == 1]]
        _, altered = episode_loss(model, modified, torch.tensor(data["patches"]), torch.tensor(y),
                                  0, s, q, 1, config)
        self.assertAlmostEqual(logs["open"], altered["open"], places=6)
        with self.assertRaisesRegex(ValueError, "disjoint"):
            episode_loss(model, torch.tensor(data["global"]), torch.tensor(data["patches"]),
                         torch.tensor(y), 0, s, s, 1, settings()["adapter"])

    def test_training_is_deterministic_train_only_and_updates_parameters(self):
        meta, make = fixture(); data, rows = make("train")
        args = settings()["adapter"]
        a, history = train_adapter(data, rows, meta, args, "full", 2)
        b, _ = train_adapter(data, rows, meta, args, "full", 2)
        self.assertGreater(float(a.up.abs().sum()), 0)
        self.assertEqual(len(history), 2)
        for key, value in a.state_dict().items():
            torch.testing.assert_close(value, b.state_dict()[key], atol=0, rtol=0)
        forged = copy.deepcopy(rows); forged[0]["split"] = "test_known"
        with self.assertRaisesRegex(ValueError, "TRAIN"):
            train_adapter(data, forged, meta, args, "full", 2)

    def test_knn_chunk_invariance_singleton_and_serialization(self):
        meta, make = fixture(); data, rows = make("train"); query, _ = make("val")
        model, _ = train_adapter(data, rows, meta, settings()["adapter"], "full", 1)
        y = np.asarray([r["true_leaf"] for r in rows]); parent = np.arange(len(query["global"])) % 2
        a = score_adapter(model, query, data, y, parent, 1)
        b = score_adapter(model, query, data, y, parent, 30)
        np.testing.assert_allclose(a["score"], b["score"], rtol=5e-5, atol=5e-5)
        np.testing.assert_array_equal(a["leaf"], b["leaf"])
        self.assertTrue(np.all(np.asarray(meta["leaf_to_parent"])[a["leaf"]] == parent))
        self.assertGreater(np.ptp(a["score"][parent == 1]), 0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.npz"; save_adapter(path, model)
            other = load_adapter(path, data, meta, settings()["adapter"], "full")
            c = score_adapter(other, query, data, y, parent, 1)
            np.testing.assert_array_equal(a["score"], c["score"])

    def test_patch_extractor_excludes_cls_and_prompt_tokens(self):
        visual = SimpleNamespace(positional_embedding=torch.zeros(5, 3), ln_post=torch.nn.Identity(), proj=None)
        tokens = torch.arange(7 * 2 * 3).reshape(7, 2, 3).float() + 1
        a = spatial_tokens([tokens, [], 0], visual, 2)
        tokens[0].fill_(9999); tokens[5:].fill_(-9999)
        b = spatial_tokens([tokens, [], 0], visual, 2)
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        self.assertEqual(tuple(a.shape), (2, 4, 3))

    def test_real_maple_transformer_hook_preserves_global_forward(self):
        # Real repository visual transformer at small random dimensions; no
        # pretrained checkpoint or biological accuracy is implied by this test.
        source = Path(__file__).resolve().parents[1] / "models/maple_model.py"
        spec = importlib.util.spec_from_file_location("tiny_maple", source)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        model = module.VisionTransformer_MaPLe(8, 4, 8, 2, 2, 6,
                {"trainer": "MaPLe", "maple_length": 2}).eval()
        x = torch.randn(2, 3, 8, 8); prompt = torch.randn(2, 8)
        deep = [torch.randn(2, 8)]
        with torch.no_grad(): expected = model(x, prompt, deep)
        captured = []
        handle = model.transformer.register_forward_hook(lambda m, args, out: captured.append(spatial_tokens(out, model, 2)))
        with torch.no_grad(): actual = model(x, prompt, deep)
        handle.remove()
        torch.testing.assert_close(expected, actual, atol=0, rtol=0)
        self.assertEqual(tuple(captured[0].shape), (2, 4, 6))

    def test_protected_calibration_preserves_known_and_reports_infeasible(self):
        rows = [dict(status="known" if i < 2 else "intra", true_parent=0, true_leaf=i if i < 2 else None,
                     split="val_known" if i < 2 else "val_intra", source="x") for i in range(4)]
        base = [{**r, "prediction_type": "known", "candidate_parent": 0, "candidate_leaf": i if i < 2 else 0} for i, r in enumerate(rows)]
        out = {"score": np.array([1., 1., .9, .9]), "leaf": np.array([0, 1, 0, 0]), "parent": np.zeros(4, int)}
        meta = {"parent_names": ["P"]}
        a = protected_thresholds(out, rows, base, base, meta, [0, 1, 2, 3])
        self.assertTrue(a["all_known_floors_feasible"])
        self.assertEqual(a["thresholds"]["0"], 1.)
        out["leaf"][1] = 0
        b = protected_thresholds(out, rows, base, base, meta, [0, 1, 2, 3])
        self.assertFalse(b["all_known_floors_feasible"])

    def test_protected_calibration_ignores_nonheld_scores_and_rejects_test(self):
        rows = [dict(status="known" if i == 0 else "intra", true_parent=0, true_leaf=0 if i == 0 else None,
                     split="val_known" if i == 0 else "val_intra", source="x") for i in range(3)]
        base = [{**r, "prediction_type": "known", "candidate_parent": 0, "candidate_leaf": 0} for r in rows]
        out = {"score": np.array([1., .2, 9.]), "leaf": np.zeros(3, int), "parent": np.zeros(3, int)}
        a = protected_thresholds(out, rows, base, base, {"parent_names": ["P"]}, [0, 1])
        out["score"][2] = -1e10; rows[2]["true_parent"] = 999
        b = protected_thresholds(out, rows, base, base, {"parent_names": ["P"]}, [0, 1])
        self.assertEqual(a, b)
        rows[0]["split"] = "test_known"
        with self.assertRaisesRegex(ValueError, "validation"):
            protected_thresholds(out, rows, base, base, {"parent_names": ["P"]}, [0, 1])

    def test_balanced_v3_runtime_accepts_only_enabled_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / "ckpt").mkdir(); (root / "ckpt/best.pth").write_bytes(b"test")
            archive = root / "training.yml"; archive.write_text("data: {}\n")
            extension = {"base_config": str(archive), "visual_support": {"primary_profile": "balanced",
                         "residual": {"enabled": True, "balanced_profile": True}}}
            config = root / "extension.yml"; config.write_text(yaml.safe_dump(extension))
            cfg, _, _ = runtime.load_configuration(str(config), "1", str(root))
            self.assertEqual(cfg["visual_support"]["primary_profile"], "balanced")
            extension["visual_support"]["residual"]["balanced_profile"] = False
            config.write_text(yaml.safe_dump(extension))
            with self.assertRaisesRegex(ValueError, "not enabled"):
                runtime.load_configuration(str(config), "1", str(root))

    def test_pipeline_real_training_calibration_and_test_artifacts(self):
        meta, make = fixture(); train, train_rows = make("train"); val, val_rows = make("val"); test, test_rows = make("test")
        labels = np.asarray([r["true_leaf"] for r in train_rows])
        bank = core.make_bank(train["global"], labels, meta)
        bank["image_hashes"] = np.asarray([r["image_sha256"] for r in train_rows])
        fit, held = core.validation_partition(val_rows)
        support = core.retrieve(val["global"], bank, meta)
        routing, _ = core.routing_fit(val["parent_cosine"], val["leaf_cosine"], support, meta, 30, val_rows, fit, {})
        e = core.evidence(val["parent_cosine"], val["leaf_cosine"], support, meta, 30, routing)
        calibration = core.fit_calibration(e, val_rows, meta, fit, held, {})
        calibration["routing"] = routing
        calibration["metadata"] = {"settings": {}, "logit_scale": 30,
                 "validation_image_hashes": [r["image_sha256"] for r in val_rows],
                 "fit_image_hashes": [val_rows[i]["image_sha256"] for i in fit],
                 "threshold_image_hashes": [val_rows[i]["image_sha256"] for i in held]}
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp); (folder / "cache").mkdir()
            for name, data, rows in (("train", train, train_rows), ("val", val, val_rows), ("test", test, test_rows)):
                np.savez_compressed(str(folder / "cache" / (name + ".npz")), **data)
                runtime.write_records(folder / "cache" / (name + ".jsonl"), rows)
            np.savez_compressed(str(folder / "root.npz"), **bank)
            runtime.write_json(folder / "root.json", calibration)
            plan = {"suite": str(folder), "seed": 1, "root_memory": str(folder / "root.npz"),
                    "root_calibration": str(folder / "root.json"), "taxonomy": meta, "settings": settings(), "inputs_sha256": {}}
            runtime.write_json(folder / "plan.json", plan)
            pipeline.train(plan, "cpu")
            pipeline.calibrate(plan, "cpu")
            with patch.object(pipeline.residual, "calibrate", side_effect=AssertionError("No fitting at test")):
                pipeline.test(plan, "cpu")
            report = runtime.read_json(folder / "summary.json")
            self.assertEqual(len(report["rows"]), 18)
            for row in report["rows"]:
                self.assertGreaterEqual(row["near_oscr_before_root_gate"], 0)
                self.assertLessEqual(row["near_oscr_before_root_gate"], 1)
            def prediction(v, p): return pipeline.read_rows(folder / "artifacts" / v / "test" / p / "predictions.jsonl")
            identity, full = prediction("identity", "balanced"), prediction("full", "protected")
            for a, b in zip(identity, full):
                for key in ("root_knownness_score", "root_gate_margin", "candidate_parent"):
                    self.assertEqual(a[key], b[key])
            # OSCR treats tied known/unknown scores as one threshold step.
            perfect = copy.deepcopy(identity)
            for r in perfect:
                if r["status"] == "known":
                    r["candidate_parent"], r["candidate_leaf"] = r["true_parent"], r["true_leaf"]
                r["child_knownness_score"] = 1. if r["status"] == "known" else 0.
            self.assertAlmostEqual(pipeline.audit_metrics(perfect)["near_oscr_before_root_gate"], 1.)
            for r in perfect: r["child_knownness_score"] = 0.
            self.assertAlmostEqual(pipeline.audit_metrics(perfect)["near_oscr_before_root_gate"], .5)
            # Real resume integrity: final receipt rejects a changed metric byte.
            runtime.write_json(folder / "receipts/test.json", {"plan_sha256": runtime.sha256(folder / "plan.json"),
                               "outputs": {p: runtime.sha256(folder / p) for p in runner.outputs(plan, "test")}})
            self.assertTrue(runner.verified(plan, "test"))
            (folder / "summary.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "changed"):
                runner.verified(plan, "test")


if __name__ == "__main__":
    unittest.main()
