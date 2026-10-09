"""Decision-head anchoring and frozen-router checkpoint selection contracts."""
import copy
import unittest
from unittest.mock import patch

import torch

from taxosafe_routealign.losses import evidence_anchor_losses
from taxosafe_routealign import training
from tests import test_taxosafe_refine_pipeline as source_fixture


def outputs(rows=3, requires_grad=False):
    return {"parent_logits": torch.randn(rows, 2, requires_grad=requires_grad),
            "leaf_logits": torch.randn(rows, 4, requires_grad=requires_grad),
            "parent_membership_logits": torch.randn(rows, 2, requires_grad=requires_grad),
            "leaf_membership_logits": torch.randn(rows, 4, requires_grad=requires_grad),
            "active_parents": torch.ones(rows, 2, dtype=torch.bool),
            "active_leaves": torch.ones(rows, 4, dtype=torch.bool)}


class EvidenceAnchorTests(unittest.TestCase):
    def test_actual_decision_heads_receive_gradients_and_teacher_is_detached(self):
        torch.manual_seed(9)
        student, teacher = outputs(requires_grad=True), outputs(requires_grad=True)
        membership, ranking, audit = evidence_anchor_losses(student, teacher, [0, 1, 2], [0, 0, 1, 1])
        (membership + .5 * ranking).backward()
        for field in student:
            if field.startswith("active_"):
                continue
            self.assertGreater(float(student[field].grad.abs().sum()), 0., field)
            self.assertIsNone(teacher[field].grad, field)
        self.assertEqual(audit["parent_active_candidates"], 6)
        self.assertEqual(audit["leaf_query_classes"], 3)

    def test_absolute_logit_shift_is_visible_when_relative_rank_kl_is_zero(self):
        torch.manual_seed(10)
        teacher = outputs()
        student = {k: (v.clone() if k.startswith("active_") else (v + 3.).requires_grad_())
                   for k, v in teacher.items()}
        membership, ranking, _ = evidence_anchor_losses(student, teacher, [0, 1, 2], [0, 0, 1, 1])
        self.assertAlmostEqual(float(ranking.detach()), 0., places=6)
        self.assertAlmostEqual(float(membership.detach()), 2.5, places=6)
        membership.backward()
        self.assertGreater(float(student["parent_membership_logits"].grad.abs().sum()), 0.)

    def test_identical_outputs_and_empty_candidates_remain_finite(self):
        torch.manual_seed(11)
        teacher = outputs()
        teacher["active_parents"][0] = False
        teacher["active_leaves"][0] = False
        teacher["active_leaves"][1, 2:] = False
        for field in list(teacher):
            if field.startswith("active_"):
                continue
            mask = teacher["active_parents" if field.startswith("parent_") else "active_leaves"]
            teacher[field] = teacher[field].masked_fill(~mask, -torch.inf)
        student = {k: (v.clone() if k.startswith("active_") else v.clone().requires_grad_())
                   for k, v in teacher.items()}
        membership, ranking, audit = evidence_anchor_losses(student, teacher, [0, 1, 2], [0, 0, 1, 1])
        self.assertAlmostEqual(float(membership.detach()), 0., places=7)
        self.assertAlmostEqual(float(ranking.detach()), 0., places=7)
        (membership + ranking).backward()
        for field in student:
            if not field.startswith("active_"):
                self.assertTrue(torch.isfinite(student[field].grad).all(), field)
        self.assertEqual(audit["parent_valid_queries"], 2)

    def test_query_class_repetition_does_not_change_class_balanced_loss(self):
        torch.manual_seed(12)
        student, teacher = outputs(rows=2), outputs(rows=2)
        first = evidence_anchor_losses(student, teacher, [0, 2], [0, 0, 1, 1])
        indices = torch.tensor([0, 0, 0, 1])
        repeated = evidence_anchor_losses({k: v[indices] for k, v in student.items()},
                                          {k: v[indices] for k, v in teacher.items()},
                                          [0, 0, 0, 2], [0, 0, 1, 1])
        for a, b in zip(first[:2], repeated[:2]):
            self.assertTrue(torch.allclose(a, b, atol=1e-7))

    def test_nonfinite_active_head_fails_and_unavailable_candidate_is_excluded(self):
        student, teacher = outputs(), outputs()
        student["parent_membership_logits"][0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "Active evidence"):
            evidence_anchor_losses(student, teacher, [0, 1, 2], [0, 0, 1, 1])
        student["active_parents"][0, 0] = False
        a, b, audit = evidence_anchor_losses(student, teacher, [0, 1, 2], [0, 0, 1, 1])
        self.assertTrue(torch.isfinite(a) and torch.isfinite(b))
        self.assertEqual(audit["parent_active_candidates"], 5)


class FrozenRouterSelectionTests(unittest.TestCase):
    def test_router_acceptance_precedes_candidate_accuracy_and_nll(self):
        preserved = {"source_router_known_correct": 198, "candidate_leaf_accuracy": .98, "structured_nll": .2}
        rejected = {"source_router_known_correct": 197, "candidate_leaf_accuracy": 1., "structured_nll": .1}
        self.assertGreater(training._selection_key(preserved), training._selection_key(rejected))

    def test_validation_uses_source_thresholds_with_production_boundary_semantics(self):
        class Encoder(torch.nn.Module):
            def text_features(self):
                return torch.ones(1)

            def encode(self, images, text_features=None):
                return {"leaf_logits": torch.tensor([[3., 0., 0., 0.], [0., 3., 0., 0.], [0., 0., 3., 0.]])}

        class Evidence(torch.nn.Module):
            def forward(self, encoded, bank):
                return {"parent_logits": torch.tensor([[3., 0.], [3., 0.], [0., 3.]]),
                        "leaf_logits": encoded["leaf_logits"],
                        "parent_membership_logits": torch.tensor([[1., 0.], [2., 0.], [0., 2.]]),
                        "leaf_membership_logits": torch.tensor([[2., 0., 0., 0.], [0., .75, 0., 0.], [0., 0., 2., 0.]]),
                        "log_probs": torch.zeros(3, 7).log_softmax(-1)}

        meta = {"leaf_names": ["a", "b", "c", "d"], "parent_names": ["p", "q"], "leaf_to_parent": [0, 0, 1, 1]}
        router = {"schema_version": "support_membership_v1", "decoder": "membership",
                  "parent_threshold": 1.00000004, "leaf_threshold": .75}
        before = copy.deepcopy(router)
        batch = [(torch.zeros(3, 1), torch.tensor([0, 1, 3]), torch.arange(3))]
        result = training.known_validation(Encoder(), Evidence(), None, batch, meta, torch.device("cpu"), router)
        self.assertEqual(result["source_router_known_correct"], 1)
        self.assertEqual(result["candidate_leaf_accuracy"], 2 / 3)
        self.assertEqual(result["source_router_root_rejections"], 1)
        self.assertEqual(result["source_router_leaf_outputs"], 2)
        self.assertEqual(router, before)
        self.assertFalse(result["unknown_data_used"])


class ActualTrainingTests(unittest.TestCase):
    setUpClass = classmethod(source_fixture.FrozenPipelineContracts.setUpClass.__func__)
    tearDownClass = classmethod(source_fixture.FrozenPipelineContracts.tearDownClass.__func__)
    setUp = source_fixture.FrozenPipelineContracts.setUp
    make_source = source_fixture.FrozenPipelineContracts.make_source
    load_stage = source_fixture.FrozenPipelineContracts.load_stage
    read_split = source_fixture.FrozenPipelineContracts.read_split
    assert_frozen = source_fixture.FrozenPipelineContracts.assert_frozen

    def test_real_update_own_bank_and_teacher_exclusion_without_unknown_data(self):
        from taxosafe_refine.importer import load_reference
        from taxosafe_support import pipeline as support
        from tests.test_taxosafe_sweep_integration import TinyFrozenCoreBackbone
        from tests import test_taxosafe_support_pipeline as toy

        self.stack.enter_context(patch.object(support, "make_backbone",
                                              side_effect=lambda *args: TinyFrozenCoreBackbone()))
        self.make_source()
        source = load_reference(self.source, self.device)
        before_files = source_fixture.artifact_snapshot(self.source)
        before_encoder = source_fixture.tensor_snapshot(source.encoder)
        before_evidence = source_fixture.tensor_snapshot(source.evidence)
        # Any attempt to load real unknown or TEST images now fails the fixture.
        for split in list(self.groups):
            if split not in ("train", "val_known"):
                del self.groups[split]
        teacher_calls = []
        original_forward = type(source.evidence).forward

        def checked_forward(module, encoded, bank, *args, **kwargs):
            output = original_forward(module, encoded, bank, *args, **kwargs)
            hashes = kwargs.get("query_hashes")
            if hashes is not None:
                for i, digest in enumerate(hashes):
                    for j, support_digest in enumerate(bank.hashes):
                        if digest == support_digest:
                            self.assertFalse(bool(output["reference_allowed"][i, j]))
                if module is source.evidence:
                    teacher_calls.append(tuple(hashes))
            return output

        directory = self.root / "new_training"
        budget = dict(training.DEFAULT_BUDGET, epochs=2, min_epochs=1, patience=2, batches_per_epoch=2)
        with patch.object(type(source.evidence), "forward", new=checked_forward):
            receipt = training.train_arm(source, copy.deepcopy(training.ARM), copy.deepcopy(source.config),
                                         directory, self.device, 1, budget)
        encoder, evidence, bank, loaded = training.load_arm_model(source, directory, self.device)
        self.assertEqual(receipt, loaded)
        self.assertEqual(receipt["optimizer_steps"], 4)
        self.assertEqual(len(teacher_calls), 4)
        self.assertGreater(receipt["parameter_delta"]["l2"], 0.)
        self.assertGreaterEqual(receipt["best_epoch"], 1)
        self.assertLessEqual(receipt["selected_checkpoint_optimizer_steps"], 4)
        self.assertFalse(receipt["baseline_selection"]["used_as_primary"])
        self.assertEqual(receipt["gradient_splits"], ["train"])
        self.assertFalse(receipt["unknown_images_used_for_gradients"])
        self.assertEqual(receipt["known_validation"]["threshold_origin"], "immutable_reference_router")
        self.assertTrue(any(not torch.equal(value, before_evidence[key])
                            for key, value in evidence.state_dict().items()))
        for key, value in encoder.state_dict().items():
            if key.startswith("backbone."):
                self.assertTrue(torch.equal(value, before_encoder[key]), key)
        self.assert_frozen(source.encoder, before_encoder)
        self.assert_frozen(source.evidence, before_evidence)
        self.assertEqual(source_fixture.artifact_snapshot(self.source), before_files)
        rebuilt, _, _ = support.reference_bank(encoder,
            toy.loader(self.groups["train"], source.config, source.meta), self.groups["train"],
            source.config, source.meta, self.device)
        saved = support._load_torch(directory / "support.pth")["bank"]
        for key, value in rebuilt.state_dict().items():
            if torch.is_tensor(value):
                self.assertTrue(torch.equal(saved[key], value), key)
        self.assertTrue(set(bank.hashes) <= set(source.training["audit"]["train"]["image_hashes"]))
        with (directory / "train.jsonl").open("a") as handle:
            handle.write(" ")
        with self.assertRaisesRegex(ValueError, "artifact hash"):
            training.load_arm_model(source, directory, self.device)


if __name__ == "__main__":
    unittest.main()
