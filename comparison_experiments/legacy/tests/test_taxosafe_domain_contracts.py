"""Preregistered domain controls and immutable input/signature contracts."""
import ast
import copy
import unittest
from pathlib import Path
import tempfile

import torch
from taxosafe_domain import backend
from taxosafe_support import pipeline as support
import numpy as np
from unittest.mock import patch

from taxosafe_domain import protocol
from taxosafe_boundary import protocol as previous


class DomainContracts(unittest.TestCase):
    def test_twelve_controls_isolate_evidence_routing_and_joint_selection(self):
        cfg = protocol.effective_config(protocol.PROJECT_ROOT / protocol.DEFAULT_CONFIG)
        self.assertEqual(cfg, protocol.DEFAULTS)
        self.assertEqual([a["id"].split("_")[0] for a in cfg["arms"]],
                         ["H{:02d}".format(i) for i in range(12)])
        self.assertEqual([a["id"] for a in cfg["arms"] if a["kind"] == "fit"],
                         ["H04_subspace_root", "H11_dual_rank16"])
        self.assertNotIn("training", cfg)
        for arm in cfg["arms"][5:11]:
            self.assertEqual(arm["kind"], "reuse")
            self.assertEqual(arm["weight_source"], "H04_subspace_root")
            self.assertEqual(arm["bank_variant"], "main")
        self.assertEqual([a["id"] for a in cfg["arms"] if a["policy"] == "joint"],
                         ["H10_dual_joint"])
        self.assertEqual([a["id"] for a in cfg["arms"]
                          if a["candidate_policy"] == "domain_parent_reference_child"],
                         ["H09_dual_reroute"])
        self.assertEqual([(cfg["arms"][i]["root"], cfg["arms"][i]["policy"])
                          for i in (6, 8, 9)], [("dual", "staged")] * 3)
        self.assertEqual(cfg["arms"][7]["root"], cfg["arms"][2]["root"])

    def test_bank_settings_are_fixed_and_detached_for_each_variant(self):
        cfg = protocol.validate_config(protocol.DEFAULTS)
        for index, name in ((4, "bank"), (11, "wide_bank")):
            result = protocol.bank_settings(cfg, cfg["arms"][index])
            self.assertEqual(result, dict(cfg[name], seed=cfg["seed"]))
            result["parent_rank"] = 99
            self.assertEqual(cfg[name], protocol.DEFAULTS[name])
        with self.assertRaises(ValueError):
            protocol.bank_settings(cfg, cfg["arms"][1])
        for key in ("parent_rank", "leaf_rank", "global_rank"):
            self.assertEqual(cfg["wide_bank"][key], 2 * cfg["bank"][key])

    def test_mutated_controls_and_hidden_training_or_test_options_rejected(self):
        mutations = [
            lambda c: c["arms"].pop(),
            lambda c: c["arms"].reverse(),
            lambda c: c["arms"][9].update(candidate_policy="reference_path"),
            lambda c: c["arms"][6].update(weight_source="H11_dual_rank16"),
            lambda c: c["arms"][8].update(policy="joint"),
            lambda c: c.update(skip_failed_gates=True),
            lambda c: c.update(training={"epochs": 1}),
            lambda c: c.update(seed=True),
            lambda c: c["bank"].update(parent_rank=0),
            lambda c: c["bank"].update(parent_rank=8.0),
            lambda c: c["bank"].update(folds=True),
            lambda c: c["bank"].update(shrinkage=float("nan")),
            lambda c: c["wide_bank"].update(global_rank=16),
            lambda c: c["calibration"].update(root_known_target=.9),
            lambda c: c["calibration"].update(root_near_target=1),
            lambda c: c["calibration"].update(seed=2),
            lambda c: c["calibration"].update(use_test_labels=True),
        ]
        for index, change in enumerate(mutations):
            cfg = copy.deepcopy(protocol.DEFAULTS)
            change(cfg)
            with self.subTest(index=index), self.assertRaises(ValueError):
                protocol.validate_config(cfg)

    def test_config_source_and_runtime_code_are_all_signed(self):
        old = set(previous.code_files())
        current = set(protocol.code_files())
        self.assertTrue(old < current)
        self.assertFalse(any("taxosafe_domain" in str(path) for path in old))
        self.assertIn(protocol.PROJECT_ROOT / "taxosafe_domain/protocol.py", current)
        for path in (protocol.PROJECT_ROOT / "taxosafe_domain").glob("*.py"):
            self.assertIn(path, current)
        baseline = protocol.signature(protocol.DEFAULTS, {"D05": "frozen-first"})
        cfg = copy.deepcopy(protocol.DEFAULTS)
        cfg["seed"] = cfg["calibration"]["seed"] = 2
        self.assertNotEqual(baseline, protocol.signature(protocol.validate_config(cfg), {"D05": "frozen-first"}))
        self.assertNotEqual(baseline, protocol.signature(protocol.DEFAULTS, {"D05": "changed"}))
        original_hash = protocol.file_hash
        changed_path = protocol.PROJECT_ROOT / "taxosafe_domain/protocol.py"
        with patch.object(protocol, "file_hash", side_effect=lambda p: "changed-runtime" if p == changed_path else original_hash(p)):
            self.assertNotEqual(baseline, protocol.signature(protocol.DEFAULTS, {"D05": "frozen-first"}))

    def test_validation_does_not_mutate_global_defaults(self):
        result = protocol.validate_config(protocol.DEFAULTS)
        result["bank"]["parent_rank"] = 99
        result["arms"][0]["id"] = "changed"
        self.assertEqual(protocol.DEFAULTS["bank"]["parent_rank"], 8)
        self.assertEqual(protocol.DEFAULTS["arms"][0]["id"], "H00_reference")

    def test_runtime_python38_syntax_and_no_observed_unsupported_torch_calls(self):
        # This is a syntax/API scan, not a claim of running Python 3.8.
        for path in (protocol.PROJECT_ROOT / "taxosafe_domain").glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 8))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                    continue
                if node.func.attr == "argsort":
                    self.assertFalse(any(k.arg == "stable" for k in node.keywords), str(path))
                if node.func.attr == "isin" and isinstance(node.func.value, ast.Name):
                    self.assertNotEqual(node.func.value.id, "torch", str(path))

    def test_test_export_keeps_frozen_dev_status_separate_from_current_metrics(self):
        from tests.test_taxosafe_domain_calibration import as_domain, fit, fixture, META
        source_groups = as_domain(fixture())
        router, _ = fit(source_groups)
        with tempfile.TemporaryDirectory() as directory:
            for current_passed in (True, False):
                groups = copy.deepcopy(source_groups)
                for rows in groups:
                    for row in rows:
                        row["split"] = "test_" + row["status"]
                        if not current_passed:
                            row["domain"]["root_score"] = router["root_threshold"] - 1.
                output = Path(directory) / str(current_passed)
                output.mkdir()
                receipt = dict(stage="test", inherited_dev_targets_passed=not current_passed)
                backend._export(output, dict(zip(("test_known", "test_intra", "test_extra"), groups)),
                                router, {"meta": META}, protocol.ARMS[2], None, None,
                                receipt, {"seconds": 0.}, "0" * 64)
                report = protocol.read_json(output / "report.json")
                self.assertEqual(report["targets_passed"], current_passed)
                self.assertEqual(report["evaluation_status"], "passed" if current_passed else "best_effort")
                self.assertEqual(report["calibration_status"], "best_effort" if current_passed else "passed")
                self.assertEqual(report["calibration_status_origin"], "frozen_development")
                self.assertFalse(report["calibration_gate_is_execution_gate"])
                self.assertTrue(report["test_allowed_after_failed_gates"])

    def test_reference_anchor_repair_uses_leaf_node_offset_without_mutating_source(self):
        from taxosafe_domain import calibration
        from tests.test_taxosafe_discovery_calibration import META, row
        record = row("anchor-path", "known")
        # Tree order is root, p, q, a, b, c. The large parent p probability
        # must never be confused with either leaf's probability.
        record["log_probs"] = np.log([.01, .70, .02, .05, .20, .02]).tolist()
        before = copy.deepcopy(record)
        inconsistent = dict(parent=np.array([0]), leaf=np.array([2]))
        with patch.object(calibration.membership, "candidate_scores", return_value=inconsistent):
            p, leaf, original, tree, mapping = calibration.reference_path([record], META)
            bp, bl, bo, bt, bm = backend._reference_path([record], META)
        self.assertEqual((p.tolist(), leaf.tolist(), original.tolist()), ([0], [1], [2]))
        for left, right in ((p, bp), (leaf, bl), (original, bo), (tree, bt), (mapping, bm)):
            self.assertTrue(np.array_equal(left, right))
        self.assertEqual(record, before)
        consistent = dict(parent=np.array([0]), leaf=np.array([0]))
        with patch.object(calibration.membership, "candidate_scores", return_value=consistent):
            _, unchanged, original, _, _ = calibration.reference_path([record], META)
        self.assertEqual(unchanged.tolist(), [0])
        self.assertEqual(original.tolist(), [0])


class DomainFrozenCache(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.suite = self.root / "domain"
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
