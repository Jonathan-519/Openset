"""Opt-in reference evidence through shared train, frozen routing, and folds."""
import copy
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch
import yaml

from taxosafe_support import holdout, pipeline, protocol
from tests import test_taxosafe_support_pipeline as baseline
from tests import test_taxosafe_support_holdout as holdout_baseline


def reference_config(cfg):
    cfg = copy.deepcopy(cfg)
    cfg["support"].update(decoupled=True, membership="reference", reference_topk=2)
    cfg["support"]["loss"].update(pair_parent=.25, pair_leaf=.25,
        membership_parent=1., membership_leaf=1., parent_cross_species=.1, leaf_sibling=.1)
    cfg["calibration"] = {"decoder": "membership", "policy": "known_first",
        "threshold_grid": "quantile", "membership_grid_points": 5, "source_loo": False}
    return cfg


class ReferencePipelineContracts(baseline.PipelineContracts):
    """Every original stage, immutability, content, and gradient check applies."""

    def setUp(self):
        super().setUp()
        self.cfg = reference_config(self.cfg)

    def test_pair_loss_uses_one_query_graph_and_logs_only_effective_exposure(self):
        encoder = pipeline.SupportEncoder(baseline.TinyBackbone(), baseline.META, self.cfg["support"])
        evidence = pipeline._make_evidence(encoder, self.cfg, baseline.META, torch.device("cpu"))
        bank, _, _ = pipeline.reference_bank(encoder,
            baseline.loader(self.groups["train"], self.cfg, baseline.META),
            self.groups["train"], self.cfg, baseline.META, torch.device("cpu"))
        config = copy.deepcopy(self.cfg)
        config["support"]["loss"] = {key: 0. for key in (
            "leaf", "parent", "support_leaf", "support_parent", "episode", "paired", "control",
            "anchor", "parent_cross_species", "leaf_sibling", "membership_parent", "membership_leaf")}
        config["support"]["loss"].update(pair_parent=.25, pair_leaf=.25)
        before = encoder.backbone.visual_calls
        encoded = encoder(baseline.CENTRES)
        args = (encoded, torch.arange(4), torch.arange(4),
            [r["image_sha256"] for r in self.groups["train"][:4]], bank,
            evidence, config, baseline.META, 9)
        with patch.object(evidence, "reference_pairs", wraps=evidence.reference_pairs) as pairs:
            _, _, warmup_counts = pipeline.training_loss(*args, episode_weight=0.)
            loss, terms, counts = pipeline.training_loss(*args, episode_weight=1.)
            self.assertEqual(pairs.call_count, 2)  # Once per call, not once per intervention.
        self.assertAlmostEqual(float(loss.detach()), .25 * float(
            (terms["reference_parent"] + terms["reference_leaf"]).detach()), places=6)
        loss.backward()
        self.assertEqual(encoder.backbone.visual_calls, before + 1)
        for parameter in (encoder.backbone.prompt_learner,
                          evidence.parent_reference.residual[-1].weight,
                          evidence.fine_reference.residual[-1].weight):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0.)
        for key in ("parent_cross_species_positive_pairs", "leaf_positive_pairs",
                    "leaf_sibling_negative_pairs", "leaf_other_parent_negative_pairs"):
            self.assertGreater(counts[key], 0)
            self.assertEqual(warmup_counts[key], 0)
        self.assertEqual(counts["parent_singleton_fallback_positive_pairs"], 0)
        self.assertFalse(bank.parent.requires_grad)
        self.assertFalse(bank.fine.requires_grad)

    def test_reference_scores_router_and_review_pack_preserve_provenance(self):
        from tools.pack_taxosafe_new_review import pack
        self.train()
        logs = [json.loads(line) for line in (self.directory / "training/train.jsonl").read_text().splitlines()]
        self.assertIn("reference_parent", logs[-1]["loss"])
        self.assertIn("reference_leaf", logs[-1]["loss"])
        self.assertGreater(logs[-1]["valid_episode_queries"]["parent_cross_species_positive_pairs"], 0)
        self.assertEqual(logs[0]["valid_episode_queries"]["parent_cross_species_positive_pairs"], 0)
        cost = protocol.read_json(self.directory / "training/model_cost.json")
        self.assertEqual(cost["query_backbone_passes_per_step"], 1)
        self.assertEqual(cost["membership"], "reference")
        self.assertEqual(cost["reference_topk"], 2)
        pipeline.calibrate_run(self.cfg, self.directory, torch.device("cpu"))
        pipeline.test_run(self.cfg, self.directory, torch.device("cpu"))
        router = protocol.read_json(self.directory / "calibration/router.json")
        self.assertEqual(router["schema_version"], "support_membership_v1")
        self.assertEqual(router["decoder"], "membership")
        predictions = [json.loads(line) for line in (self.directory / "test/predictions.jsonl").read_text().splitlines()]
        for record in predictions:
            heads = record["support_evidence"]
            self.assertEqual(len(heads["parent_membership_logits"]), 2)
            self.assertEqual(len(heads["leaf_membership_logits"]), 4)
            self.assertTrue(all(torch.isfinite(torch.tensor(v)).all() for v in heads.values()))
        archive_path = Path(self.tmp.name) / "review.tar.gz"
        pack(self.directory, archive_path)
        with tarfile.open(archive_path) as archive:
            names = archive.getnames()
            self.assertNotIn("run/training/best.pth", names)
            saved = json.loads(archive.extractfile("run/calibration/router.json").read())
            self.assertEqual(saved, router)
            manifest = json.loads(archive.extractfile("archive_manifest.json").read())
            self.assertTrue(any(item["path"] == "run/test/predictions.jsonl" for item in manifest["files"]))


class ReferenceStrictHoldoutContract(unittest.TestCase):
    def test_fold_exports_active_support_candidates_without_applying_thresholds(self):
        cfg, _ = baseline.fixture()
        cfg = reference_config(cfg)
        fold = {"id": "species_000", "kind": "species", "target_parent": 0,
                "active_leaf_mask": [False, True, True, True]}
        encoder = Mock()
        # Text candidate, global support candidate and membership maximizers
        # deliberately disagree with the required parent-then-child identity.
        encoder.encode.return_value = {"leaf_logits": torch.tensor([[-torch.inf, 0., 1., 9.]])}
        output = {
            "log_probs": torch.tensor([[.1, .05, .05, 0., .1, .6, .1]]).log(),
            "root_logit": torch.tensor([0.]),
            "parent_logits": torch.tensor([[2., 1.]]),
            "leaf_logits": torch.tensor([[-torch.inf, 1., 99., 98.]]),
            "parent_membership_logits": torch.tensor([[-9., 9.]]),
            "leaf_membership_logits": torch.tensor([[-torch.inf, -8., 8., 9.]])}
        evidence = Mock(return_value=output)
        groups = {"val_known": [baseline.row("train", 1)], "heldout": [baseline.row("train", 0)]}
        with patch.object(pipeline, "make_loader", side_effect=baseline.loader):
            records, report = holdout.score_fold(encoder, evidence, None, cfg, baseline.META,
                                                  groups, fold, torch.device("cpu"))
        self.assertEqual(report["decoder"], "uncalibrated_joint_argmax")
        self.assertFalse(report["calibration_used"])
        self.assertFalse(report["support_candidate_diagnostics"]["membership_thresholds_applied"])
        for record in records:
            self.assertEqual(record["predicted_node"], 5)
            self.assertEqual(record["closed_pred_leaf"], 3)
            self.assertEqual(record["support_candidate_parent"], 0)
            self.assertEqual(record["support_candidate_leaf"], 1)
            self.assertEqual(record["candidate_parent"], 0)
            self.assertEqual(record["candidate_leaf"], 1)
            self.assertEqual(record["support_candidate_parent_membership_logit"], -9.)
            self.assertEqual(record["support_candidate_leaf_membership_logit"], -8.)
            self.assertIsNone(record["support_evidence"]["leaf_logits"][0])
            self.assertIsNone(record["support_evidence"]["leaf_membership_logits"][0])
            self.assertFalse(record["membership_thresholds_applied"])
        fold.update(id="parent_000", kind="parent", active_leaf_mask=[False, False, True, True])
        output.update(parent_logits=torch.tensor([[-torch.inf, 1.]]),
                      parent_membership_logits=torch.tensor([[-torch.inf, 9.]]),
                      leaf_logits=torch.tensor([[-torch.inf, -torch.inf, 99., 98.]]),
                      leaf_membership_logits=torch.tensor([[-torch.inf, -torch.inf, 8., 9.]]),
                      log_probs=torch.tensor([[.1, 0., .1, 0., 0., .7, .1]]).log())
        groups["val_known"] = [baseline.row("train", 2)]
        with patch.object(pipeline, "make_loader", side_effect=baseline.loader):
            parent_records, _ = holdout.score_fold(encoder, evidence, None, cfg, baseline.META,
                                                   groups, fold, torch.device("cpu"))
        for record in parent_records:
            self.assertIsNone(record["support_evidence"]["parent_logits"][0])
            self.assertIsNone(record["support_evidence"]["parent_membership_logits"][0])
            self.assertEqual(record["support_candidate_parent"], 1)
            self.assertEqual(record["support_candidate_leaf"], 2)
            self.assertEqual(record["support_candidate_leaf_membership_logit"], 8.)

    def test_reference_mode_runs_both_strict_fold_kinds_with_original_taxonomy(self):
        self.rows = [baseline.row("train", i) for i in range(20)]
        original = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, original)
        torch.set_num_threads(1)

        def fixture():
            cfg, groups = baseline.fixture()
            return reference_config(cfg), groups

        with patch.object(holdout_baseline, "fixture", side_effect=fixture):
            holdout_baseline.StrictHoldoutTests.test_actual_shared_training_folds_use_no_heldout_gradients_or_support(self)


class ReferenceConfigurationContracts(unittest.TestCase):
    def test_opt_in_config_retains_permissions_budget_schedule_and_old_defaults(self):
        folder = protocol.PROJECT_ROOT / "configs/Zooplankton_Taxonomic_Tree"
        old = protocol.effective_config(folder / "TaxoSafe_support_decoupled.yml")
        new = protocol.effective_config(folder / "TaxoSafe_support_reference.yml")
        for key in ("data", "training", "evaluation_gates"):
            self.assertEqual(new[key], old[key])
        self.assertEqual(new["output_root"], "runs/taxosafe_new/reference")
        self.assertEqual(new["support"]["max_per_leaf"], old["support"]["max_per_leaf"])
        self.assertEqual(new["support"]["loss"]["pair_parent"], .25)
        self.assertEqual(new["support"]["loss"]["pair_leaf"], .25)
        self.assertEqual(protocol.signature(new)["method"], "support_reference_v3")
        self.assertEqual(protocol.signature(old)["method"], "support_decoupled_v2")
        self.assertEqual(new["calibration"]["decoder"], "membership")
        self.assertNotIn("membership", old["support"])
        self.assertNotIn("decoder", old["calibration"])
        self.assertEqual(pipeline.DEFAULT_CONFIG, "configs/Zooplankton_Taxonomic_Tree/TaxoSafe_support_new.yml")

    def test_invalid_reference_options_fail_closed(self):
        source = protocol.PROJECT_ROOT / "configs/Zooplankton_Taxonomic_Tree/TaxoSafe_support_reference.yml"
        base = yaml.safe_load(source.read_text())
        changes = [("support", "decoupled", False), ("support", "membership", "nearest"),
            ("support", "reference_topk", True), ("support", "reference_topk", 0),
            ("support", "reference_topk", 2.5), ("support", "reference_topk", 9),
            ("calibration", "decoder", "fallback"), ("calibration", "threshold_grid", "bias"),
            ("calibration", "membership_grid_points", True), ("calibration", "membership_grid_points", 1),
            ("calibration", "membership_grid_points", 402),
            ("calibration", "parent_threshold_grid", []),
            ("calibration", "leaf_threshold_grid", [float("nan")])]
        with tempfile.TemporaryDirectory() as folder:
            for section, key, value in changes:
                config = copy.deepcopy(base)
                config[section][key] = value
                path = Path(folder) / "config.yml"
                path.write_text(yaml.safe_dump(config))
                with self.subTest(section=section, key=key, value=value), self.assertRaises(ValueError):
                    protocol.effective_config(path)


if __name__ == "__main__":
    unittest.main()
