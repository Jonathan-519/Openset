"""End-to-end compatibility and new-mode gradient contracts on tiny CPU data."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch
import yaml

from taxosafe_support import pipeline, protocol
from tests import test_taxosafe_support_pipeline as baseline
from tests.test_taxosafe_support_pipeline import (
    CENTRES, META, TinyBackbone, loader,
)


class DecoupledPipelineContracts(baseline.PipelineContracts):
    """Run every existing integration contract through the opt-in model too."""
    def setUp(self):
        super().setUp()
        self.cfg["support"]["decoupled"] = True
        self.cfg["support"]["loss"].update(
            membership_parent=1., membership_leaf=1., parent_cross_species=.1, leaf_sibling=.1)
        self.cfg["calibration"]["policy"] = "known_first"

    def test_candidate_evidence_survives_calibration_and_test_exports(self):
        self.train()
        logs = [json.loads(line) for line in (self.directory / "training/train.jsonl").read_text().splitlines()]
        for key in ("membership_parent", "membership_leaf", "parent_cross_species", "leaf_sibling"):
            self.assertIn(key, logs[-1]["loss"])
        self.assertIn("valid_parent_pairs", logs[-1]["valid_episode_queries"])
        pipeline.calibrate_run(self.cfg, self.directory, torch.device("cpu"))
        pipeline.test_run(self.cfg, self.directory, torch.device("cpu"))
        router = protocol.read_json(self.directory / "calibration/router.json")
        self.assertEqual(router["selection_policy"], "known_first")
        records = [json.loads(line) for line in (self.directory / "test/predictions.jsonl").read_text().splitlines()]
        for record in records:
            evidence = record["support_evidence"]
            self.assertEqual(len(evidence["parent_membership_logits"]), len(META["parent_names"]))
            self.assertEqual(len(evidence["leaf_membership_logits"]), len(META["leaf_names"]))

    def test_membership_loss_alone_reaches_encoder_and_both_acceptance_heads(self):
        encoder = pipeline.SupportEncoder(TinyBackbone(), META, self.cfg["support"])
        evidence = pipeline._make_evidence(encoder, self.cfg, META, torch.device("cpu"))
        bank, _, _ = pipeline.reference_bank(encoder, loader(self.groups["train"], self.cfg, META),
                                             self.groups["train"], self.cfg, META, torch.device("cpu"))
        config = copy.deepcopy(self.cfg)
        config["support"]["loss"] = {key: 0. for key in (
            "leaf", "parent", "support_leaf", "support_parent", "episode", "paired", "control",
            "anchor", "parent_cross_species", "leaf_sibling")}
        config["support"]["loss"].update(membership_parent=1., membership_leaf=1.)
        calls = encoder.backbone.visual_calls
        encoded = encoder(CENTRES)
        loss, _, _ = pipeline.training_loss(encoded, torch.arange(4), torch.arange(4),
            [r["image_sha256"] for r in self.groups["train"][:4]], bank,
            evidence, config, META, 9, 1.)
        loss.backward()
        self.assertEqual(encoder.backbone.visual_calls, calls + 1)
        for parameter in (encoder.backbone.prompt_learner,
                          evidence.parent_membership.residual[-1].weight,
                          evidence.fine_membership.residual[-1].weight):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0.)


class DecoupledConfigurationContracts(unittest.TestCase):
    def test_new_configuration_keeps_manifests_gates_and_legacy_default(self):
        directory = protocol.PROJECT_ROOT / "configs/Zooplankton_Taxonomic_Tree"
        old = protocol.effective_config(directory / "TaxoSafe_support_new.yml")
        new = protocol.effective_config(directory / "TaxoSafe_support_decoupled.yml")
        self.assertEqual(new["data"], old["data"])
        self.assertEqual(new["evaluation_gates"], old["evaluation_gates"])
        self.assertFalse(old["support"].get("decoupled", False))
        self.assertTrue(new["support"]["decoupled"])
        self.assertEqual(new["calibration"]["policy"], "known_first")
        self.assertEqual(protocol.signature(new)["method"], "support_decoupled_v2")
        self.assertEqual(pipeline.DEFAULT_CONFIG, "configs/Zooplankton_Taxonomic_Tree/TaxoSafe_support_new.yml")

    def test_invalid_new_options_fail_before_training(self):
        path = protocol.PROJECT_ROOT / "configs/Zooplankton_Taxonomic_Tree/TaxoSafe_support_decoupled.yml"
        original = yaml.safe_load(path.read_text())
        invalid = []
        for key, value in (("decoupled", "false"), ("margins", {"representation": float("nan")})):
            cfg = copy.deepcopy(original)
            cfg["support"][key] = value
            invalid.append(cfg)
        cfg = copy.deepcopy(original)
        cfg["calibration"]["policy"] = "relax_gates"
        invalid.append(cfg)
        with tempfile.TemporaryDirectory() as folder:
            for cfg in invalid:
                target = Path(folder) / "config.yml"
                target.write_text(yaml.safe_dump(cfg))
                with self.assertRaises(ValueError):
                    protocol.effective_config(target)


if __name__ == "__main__":
    unittest.main()
