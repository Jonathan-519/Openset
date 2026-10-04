"""Direction-aware reference training, candidate selection, and old contracts."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch
import yaml

from taxosafe_support import pipeline, protocol
from tests import test_taxosafe_support_pipeline as baseline
from tests import test_taxosafe_support_holdout as holdout_baseline
from tests.test_taxosafe_support_reference_pipeline import reference_config


def relation_config(cfg, negative_topk=2, selection="candidate"):
    cfg = reference_config(cfg)
    cfg["support"].update(membership="relation", relation_dim=3)
    if negative_topk is not None:
        cfg["support"]["pair_negative_topk"] = negative_topk
    cfg["training"]["selection"] = selection
    return cfg


class RelationPipelineContracts(baseline.PipelineContracts):
    """Reuse all original receipt, isolation, and full-stage integration tests."""

    def setUp(self):
        super().setUp()
        self.cfg = relation_config(self.cfg)

    def test_relation_heads_train_once_per_query_and_diagnostics_do_not_change_loss(self):
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
            _, _, warmup = pipeline.training_loss(*args, episode_weight=0.)
            loss, terms, counts = pipeline.training_loss(*args, episode_weight=1.)
            self.assertEqual(pairs.call_count, 2)
        expected = .25 * (terms["reference_parent"] + terms["reference_leaf"])
        self.assertTrue(torch.equal(loss, expected))
        diagnostics = {k: v for k, v in terms.items() if k.startswith("diagnostic_")}
        self.assertEqual(len(diagnostics), 3)
        for value in diagnostics.values():
            self.assertFalse(value.requires_grad)
            self.assertTrue(bool(torch.isfinite(value)))
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
            self.assertEqual(warmup[key], 0)
        self.assertFalse(bank.parent.requires_grad)
        self.assertFalse(bank.fine.requires_grad)

    def test_relation_logs_candidate_selection_and_audit_separately(self):
        self.train()
        logs = [json.loads(line) for line in (self.directory / "training/train.jsonl").read_text().splitlines()]
        for record in logs:
            validation = record["known_validation"]
            self.assertIn("candidate_leaf_accuracy", validation)
            self.assertFalse(validation["membership_thresholds_applied"])
            self.assertFalse(validation["unknown_data_used"])
            self.assertEqual(validation["selection_splits"], ["val_known"])
            self.assertEqual(len(record["diagnostics"]), 3)
            self.assertFalse(any(key.startswith("diagnostic_") for key in record["loss"]))
        cost = protocol.read_json(self.directory / "training/model_cost.json")
        self.assertEqual(cost["membership"], "relation")
        self.assertEqual(cost["query_backbone_passes_per_step"], 1)
        self.assertEqual(cost["checkpoint_selection"], "candidate")
        self.assertEqual(cost["reference_pair_selection"], "per_leaf_topk_negative/all_positive")
        self.assertEqual(self.calls, [("train", ("train", "val_known"))])

    def test_relation_only_ablation_keeps_text_selection_and_all_pair_supervision(self):
        self.cfg = relation_config(baseline.fixture()[0], negative_topk=None, selection="text")
        self.train()
        receipt = protocol.read_json(self.directory / "training/completed.json")
        self.assertNotIn("candidate_leaf_accuracy", receipt["known_validation"])
        cost = protocol.read_json(self.directory / "training/model_cost.json")
        self.assertEqual(cost["checkpoint_selection"], "text")
        self.assertIsNone(cost["pair_negative_topk"])
        self.assertEqual(cost["reference_pair_selection"], "all_allowed_pairs")


class CandidateSelectionContracts(unittest.TestCase):
    def test_candidate_validation_uses_parent_then_child_ignores_membership_and_masks_inactive(self):
        labels = torch.tensor([0, 2, 3])
        encoder = Mock()
        encoder.encode.return_value = {"leaf_logits": torch.nn.functional.one_hot(labels, 4).float() * 10}
        output = {"log_probs": torch.full((3, 7), 1 / 7).log(),
                  "parent_logits": torch.tensor([[1., 2.], [-torch.inf, 2.], [2., 2.]]),
                  "leaf_logits": torch.tensor([[99., 98., 3., 2.], [-torch.inf, -torch.inf, 4., 1.], [5., 5., 99., 98.]]),
                  "parent_membership_logits": torch.tensor([[100., -100.]] * 3),
                  "leaf_membership_logits": torch.tensor([[100., -100., -100., 100.]] * 3)}
        evidence = Mock(return_value=output)
        loader = [(torch.zeros(3, 4), labels, torch.arange(3))]
        old = pipeline.known_validation(encoder, evidence, None, loader, baseline.META, "cpu")
        self.assertEqual(old["leaf_accuracy"], 1.)
        self.assertEqual(set(old), {"leaf_accuracy", "uncalibrated_e2e", "structured_nll", "count",
                                   "selection_splits", "unknown_data_used"})
        new = pipeline.known_validation(encoder, evidence, None, loader, baseline.META, "cpu",
                                        selection="candidate", selection_splits=["known_train_inner_validation"])
        self.assertEqual(new["candidate_leaf_accuracy"], 1 / 3)
        self.assertFalse(new["membership_thresholds_applied"])
        self.assertEqual(new["selection_splits"], ["known_train_inner_validation"])
        self.assertEqual({key: new[key] for key in old if key != "selection_splits"},
                         {key: old[key] for key in old if key != "selection_splits"})

    def test_selection_primary_is_opt_in_and_nll_remains_only_tiebreaker(self):
        text_best = {"leaf_accuracy": 1., "candidate_leaf_accuracy": .5, "structured_nll": .1}
        candidate_best = {"leaf_accuracy": .5, "candidate_leaf_accuracy": 1., "structured_nll": 2.}
        self.assertGreater(pipeline._selection_key(text_best), pipeline._selection_key(candidate_best))
        self.assertLess(pipeline._selection_key(text_best, "candidate"),
                        pipeline._selection_key(candidate_best, "candidate"))
        better_nll = dict(candidate_best, structured_nll=1.)
        self.assertGreater(pipeline._selection_key(better_nll, "candidate"),
                           pipeline._selection_key(candidate_best, "candidate"))

    def test_relation_diagnostics_handle_singleton_or_absent_local_evidence(self):
        result = pipeline._relation_diagnostics({"fine_local": None, "parent_local": None},
            {"leaf_membership_logits": torch.tensor([[1., -torch.inf]]),
             "active_leaves": torch.tensor([[True, False]])}, torch.tensor([0]), torch.tensor([0, 1]))
        self.assertEqual(result, {"diagnostic_leaf_membership_sibling_margin": torch.tensor(0.)})


class RelationConfigurationContracts(unittest.TestCase):
    def test_new_configs_preserve_data_schedule_loss_and_gates(self):
        folder = protocol.PROJECT_ROOT / "configs/Zooplankton_Taxonomic_Tree"
        reference = protocol.effective_config(folder / "TaxoSafe_support_reference.yml")
        main = protocol.effective_config(folder / "TaxoSafe_support_relation.yml")
        control = protocol.effective_config(folder / "TaxoSafe_support_relation_only.yml")
        for cfg in (main, control):
            for section in ("data", "calibration", "evaluation_gates"):
                self.assertEqual(cfg[section], reference[section])
            self.assertEqual({k: v for k, v in cfg["training"].items() if k != "selection"}, reference["training"])
            self.assertEqual(cfg["support"]["loss"], reference["support"]["loss"])
            self.assertEqual(cfg["support"]["max_per_leaf"], reference["support"]["max_per_leaf"])
            self.assertEqual(cfg["support"]["relation_dim"], 32)
            self.assertEqual(protocol.signature(cfg)["method"], "support_relation_v4")
        self.assertEqual(main["output_root"], "runs/taxosafe_new/relation")
        self.assertEqual(main["training"]["selection"], "candidate")
        self.assertEqual(main["support"]["pair_negative_topk"], 2)
        self.assertEqual(control["output_root"], "runs/taxosafe_new/relation_only")
        self.assertEqual(control["training"]["selection"], "text")
        self.assertNotIn("pair_negative_topk", control["support"])
        self.assertNotIn("selection", reference["training"])
        self.assertEqual(protocol.signature(reference)["method"], "support_reference_v3")

    def test_invalid_relation_options_fail_closed(self):
        source = protocol.PROJECT_ROOT / "configs/Zooplankton_Taxonomic_Tree/TaxoSafe_support_relation.yml"
        base = yaml.safe_load(source.read_text())
        changes = [("support", "relation_dim", value) for value in (True, 0, 1.5, 513)]
        changes += [("support", "pair_negative_topk", value) for value in (True, 0, 1.5, 9, None)]
        changes += [("training", "selection", "unknown"), ("calibration", "decoder", "joint"),
                    ("support", "decoupled", False), ("support", "membership", "reference")]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.yml"
            for section, key, value in changes:
                config = copy.deepcopy(base)
                config[section][key] = value
                path.write_text(yaml.safe_dump(config))
                with self.subTest(section=section, key=key, value=value), self.assertRaises(ValueError):
                    protocol.effective_config(path)


class RelationStrictHoldoutContract(unittest.TestCase):
    def test_relation_mode_runs_strict_species_and_parent_folds_with_candidate_selection(self):
        self.rows = [baseline.row("train", i) for i in range(20)]
        original = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, original)
        torch.set_num_threads(1)

        def fixture():
            cfg, groups = baseline.fixture()
            return relation_config(cfg), groups

        score_fold = holdout_baseline.holdout.score_fold
        checked_kinds = []

        def checked_score(*args, **kwargs):
            records, report = score_fold(*args, **kwargs)
            checked_kinds.append(report["kind"])
            self.assertFalse(report["support_candidate_diagnostics"]["membership_thresholds_applied"])
            self.assertFalse(report["calibration_used"])
            self.assertEqual(report["decoder"], "uncalibrated_joint_argmax")
            for record in records:
                self.assertFalse(record["membership_thresholds_applied"])
                self.assertEqual(record["candidate_leaf"], record["support_candidate_leaf"])
                self.assertEqual(record["candidate_parent"], record["support_candidate_parent"])
                self.assertEqual(set(record["support_evidence"]), {
                    "parent_logits", "leaf_logits", "parent_membership_logits", "leaf_membership_logits"})
            return records, report

        with patch.object(holdout_baseline, "fixture", side_effect=fixture), \
                patch.object(holdout_baseline.holdout, "score_fold", side_effect=checked_score):
            holdout_baseline.StrictHoldoutTests.test_actual_shared_training_folds_use_no_heldout_gradients_or_support(self)
        self.assertEqual(checked_kinds, ["species", "parent"])


class RelationCudaContract(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_relation_training_loss_and_backward(self):
        from tests.test_taxosafe_support_core import MAPPING, fixture
        bank, encoded = fixture()
        bank = bank.to("cuda")
        encoded = {key: value.detach().to("cuda").requires_grad_() for key, value in encoded.items()}
        encoded["leaf_logits"] = torch.randn(5, 5, device="cuda", requires_grad=True)
        encoded["parent_logits"] = torch.randn(5, 3, device="cuda", requires_grad=True)
        cfg = relation_config(baseline.fixture()[0])
        meta = {"leaf_names": list("abcde"), "parent_names": list("PQR"), "leaf_to_parent": MAPPING}
        evidence = pipeline.HierarchicalEvidence(8, MAPPING, decoupled=True,
            membership_mode="relation", relation_dim=3).cuda()
        for head in (evidence.parent_reference, evidence.fine_reference):
            torch.nn.init.constant_(head.residual[-1].weight, .04)
        labels = torch.arange(5, device="cuda")
        loss, terms, counts = pipeline.training_loss(encoded, labels, labels,
            list(bank.hashes)[::3], bank, evidence, cfg, meta, 9, 1.)
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        for value in encoded.values():
            self.assertIsNotNone(value.grad)
            self.assertTrue(bool(torch.isfinite(value.grad).all()))
        for head in (evidence.parent_reference, evidence.fine_reference):
            self.assertGreater(float(head.residual[-1].weight.grad.abs().sum()), 0.)
        self.assertEqual(counts["parent_singleton_fallback_positive_pairs"], 2)
        self.assertTrue(all(not value.requires_grad for key, value in terms.items() if key.startswith("diagnostic_")))


if __name__ == "__main__":
    unittest.main()
