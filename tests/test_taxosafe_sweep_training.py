"""Fine-tuning gradient masks and trained-reference anchoring contracts."""
import copy
import unittest

import torch

from tests import test_taxosafe_refine_pipeline as fixture
from taxosafe_sweep import training


class SweepStudentTests(unittest.TestCase):
    setUpClass = classmethod(fixture.FrozenPipelineContracts.setUpClass.__func__)
    tearDownClass = classmethod(fixture.FrozenPipelineContracts.tearDownClass.__func__)
    setUp = fixture.FrozenPipelineContracts.setUp
    make_source = fixture.FrozenPipelineContracts.make_source
    load_stage = fixture.FrozenPipelineContracts.load_stage
    read_split = fixture.FrozenPipelineContracts.read_split
    assert_frozen = fixture.FrozenPipelineContracts.assert_frozen

    def test_heads_clone_exact_source_without_prompt_gradients_or_shared_storage(self):
        from taxosafe_refine.importer import load_reference
        self.make_source()
        source = load_reference(self.source, self.device)
        original = fixture.artifact_snapshot(self.source)
        encoder, evidence, names = training.build_student(source, self.device, "heads")
        self.assertFalse(any(n.startswith("encoder.backbone.") for n in names))
        self.assertTrue(any(n.startswith("encoder.parent_branch.") for n in names))
        self.assertTrue(any(n.startswith("encoder.fine_branch.") for n in names))
        self.assertTrue(any(n.startswith("evidence.") for n in names))
        for student, teacher in ((encoder, source.encoder), (evidence, source.evidence)):
            expected = dict(teacher.named_parameters())
            for name, value in student.named_parameters():
                self.assertTrue(torch.equal(value, expected[name]), name)
                self.assertNotEqual(value.data_ptr(), expected[name].data_ptr(), name)
                self.assertFalse(expected[name].requires_grad, name)
        self.assertEqual(fixture.artifact_snapshot(self.source), original)

    def test_prompt_scope_only_enables_named_prompts_inside_backbone(self):
        from taxosafe_refine.importer import load_reference
        self.make_source()
        source = load_reference(self.source, self.device)
        # Add an unused ordinary backbone parameter to the tiny model so this
        # test catches blanket requires_grad_(True), not just missing prompts.
        source.encoder.backbone.frozen_probe = torch.nn.Parameter(torch.ones(2), requires_grad=False)
        encoder, evidence, names = training.build_student(source, self.device, "prompts_heads")
        self.assertIn("encoder.backbone.prompt_learner", names)
        self.assertNotIn("encoder.backbone.frozen_probe", names)
        self.assertFalse(encoder.backbone.frozen_probe.requires_grad)
        self.assertTrue(encoder.backbone.prompt_learner.requires_grad)
        self.assertFalse(any(p.requires_grad for p in source.encoder.parameters()))
        with self.assertRaises(ValueError):
            training.build_student(source, self.device, "full_backbone")

    def test_loaded_update_must_match_scope_and_actual_tensor_delta(self):
        from taxosafe_refine.importer import load_reference
        self.make_source()
        source = load_reference(self.source, self.device)
        source.encoder.backbone.frozen_probe = torch.nn.Parameter(torch.ones(2), requires_grad=False)
        encoder, evidence, names = training.build_student(source, self.device, "heads")
        delta = training._delta(encoder, evidence, source, names)
        with self.assertRaisesRegex(ValueError, "nonzero"):
            training._verify_update_scope(encoder, evidence, source, names, delta)
        parameter = next(p for p in encoder.parameters() if p.requires_grad)
        with torch.no_grad():
            parameter.add_(.01)
        delta = training._delta(encoder, evidence, source, names)
        self.assertEqual(training._verify_update_scope(encoder, evidence, source, names, delta), delta)
        with self.assertRaisesRegex(ValueError, "disagrees"):
            training._verify_update_scope(encoder, evidence, source, names, dict(delta, l2=99.))
        with torch.no_grad():
            encoder.backbone.frozen_probe.add_(.1)
        with self.assertRaisesRegex(ValueError, "frozen parameter"):
            training._verify_update_scope(encoder, evidence, source, names, delta)


class SweepAnchorTests(unittest.TestCase):
    def test_reference_targets_are_detached_and_both_student_branches_receive_gradients(self):
        torch.manual_seed(3)
        student = {"leaf_logits": torch.randn(3, 4, requires_grad=True),
                   "parent_logits": torch.randn(3, 2, requires_grad=True),
                   "fine": torch.randn(3, 5, requires_grad=True),
                   "parent": torch.randn(3, 5, requires_grad=True)}
        teacher = {key: torch.randn_like(value, requires_grad=True) for key, value in student.items()}
        kd, feature = training.anchor_losses(student, teacher, 2.)
        (.5 * kd + .5 * feature).backward()
        self.assertTrue(torch.isfinite(kd) and torch.isfinite(feature))
        for key in student:
            self.assertIsNotNone(student[key].grad, key)
            self.assertGreater(float(student[key].grad.abs().sum()), 0, key)
            self.assertIsNone(teacher[key].grad, key)

    def test_identical_reference_targets_have_zero_anchor_penalty(self):
        torch.manual_seed(4)
        student = {"leaf_logits": torch.randn(3, 4), "parent_logits": torch.randn(3, 2),
                   "fine": torch.randn(3, 5), "parent": torch.randn(3, 5)}
        kd, feature = training.anchor_losses(student, copy.deepcopy(student), 2.)
        self.assertAlmostEqual(float(kd), 0., places=6)
        self.assertAlmostEqual(float(feature), 0., places=6)
        with self.assertRaises(ValueError):
            training.anchor_losses(student, student, 0.)

    def test_invalid_budget_cannot_silently_turn_training_off(self):
        for override in ({"epochs": 0}, {"min_epochs": 30}, {"prompt_lr": float("nan")},
                         {"head_lr": True}, {"unknown_gradient": 1}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                training._budget(override)


if __name__ == "__main__":
    unittest.main()
