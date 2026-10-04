"""Real CPU sweep on deterministic toy vectors, never research measurements.

Only model/image I/O and process isolation use fixture seams. Historical
reference import, gradient updates, support rebuilding, calibration, every
arm's TEST, receipts and comparison exports run through production code.
"""
import copy
import unittest
from unittest.mock import patch

import torch

from taxosafe_support import pipeline as support
from taxosafe_support import protocol as source_protocol
from tests import test_taxosafe_refine_pipeline as source_fixture
from tests import test_taxosafe_support_pipeline as toy


class TinyFrozenCoreBackbone(toy.TinyBackbone):
    """Include a genuine frozen core parameter in the query graph."""

    def __init__(self):
        super().__init__()
        self.visual_gain = torch.nn.Parameter(torch.ones(4), requires_grad=False)

    def encode_image_with_spatial(self, images, normalize=True):
        return super().encode_image_with_spatial(images * self.visual_gain, normalize)


class SweepIntegrationTests(unittest.TestCase):
    setUpClass = classmethod(source_fixture.FrozenPipelineContracts.setUpClass.__func__)
    tearDownClass = classmethod(source_fixture.FrozenPipelineContracts.tearDownClass.__func__)
    setUp = source_fixture.FrozenPipelineContracts.setUp
    make_source = source_fixture.FrozenPipelineContracts.make_source
    load_stage = source_fixture.FrozenPipelineContracts.load_stage
    read_split = source_fixture.FrozenPipelineContracts.read_split

    def test_actual_six_arm_sweep_exports_failed_gate_test_without_source_mutation(self):
        from taxosafe_refine import importer
        from taxosafe_sweep import evaluation, protocol, runner, training

        self.stack.enter_context(patch.object(support, "make_backbone",
                                              side_effect=lambda *args: TinyFrozenCoreBackbone()))
        # Distinct image hashes but exactly indistinguishable known/near inputs:
        # the toy DEV cannot satisfy both known and near scientific gates.
        for known, near in zip(self.groups["val_known"], self.groups["val_intra"]):
            near["vector"] = copy.deepcopy(known["vector"])
        self.make_source()
        original = source_fixture.artifact_snapshot(self.source)
        source = importer.load_reference(self.source, self.device)
        original_encoder = source_fixture.tensor_snapshot(source.encoder)
        original_evidence = source_fixture.tensor_snapshot(source.evidence)
        train_rows = copy.deepcopy(self.groups["train"])
        train_hashes = {row["image_sha256"] for row in train_rows}
        cfg = protocol.effective_config(protocol.PROJECT_ROOT / protocol.DEFAULT_CONFIG)
        cfg["budget"].update(epochs=2, min_epochs=1, patience=2, batches_per_epoch=2)
        suite = self.root / "sweep"
        stages, selection_bytes = [], []

        def launch_in_process(directory, arm_id, stage, device):
            stages.append((arm_id, stage))
            if stage == "test":
                # Every DEV result and the immutable selection must exist
                # before the first TEST forward. Also make all fit rows
                # inaccessible to catch an accidental TEST-time refit.
                for arm in cfg["arms"]:
                    self.assertTrue((suite / "arms" / arm["id"] / "calibration/completed.json").is_file())
                if not selection_bytes:
                    self.stack.enter_context(patch.object(evaluation.membership, "calibrate",
                        side_effect=AssertionError("TEST cannot calibrate even from cached DEV")))
                    self.stack.enter_context(patch.object(support, "training_loss",
                        side_effect=AssertionError("TEST cannot perform neural training")))
                selection_bytes.append((suite / "dev_selection.json").read_bytes())
                for split in ("train", "val_known", "val_intra", "val_extra"):
                    self.groups.pop(split, None)
            logs = suite / "logs" / arm_id
            logs.mkdir(parents=True, exist_ok=True)
            out, err = logs / (stage + ".stdout.log"), logs / (stage + ".stderr.log")
            out.write_text("Toy integration runs the real worker in this process.\n", encoding="utf-8")
            err.write_text("", encoding="utf-8")
            runner.worker(directory, arm_id, stage, device)
            return 0, out, err

        with patch.object(runner, "_launch_stage", side_effect=launch_in_process):
            runner.execute_suite(cfg, self.source, suite, device="cpu")

        self.assertEqual(source_fixture.artifact_snapshot(self.source), original)
        self.assertEqual(len(selection_bytes), 6)
        self.assertTrue(all(value == selection_bytes[0] for value in selection_bytes))
        self.assertEqual((suite / "dev_selection.json").read_bytes(), selection_bytes[0])
        first_test = next(i for i, item in enumerate(stages) if item[1] == "test")
        self.assertEqual([stage for _, stage in stages[first_test:]], ["test"] * 6)
        self.assertEqual(sum(stage == "training" for _, stage in stages), 5)
        completed = source_protocol.read_json(suite / "suite_completed.json")
        self.assertTrue(completed["workflow_completed"])
        self.assertFalse(completed["test_used_for_selection"])
        self.assertEqual(completed["technical_failure_arms"], [])

        loaded = {}
        for arm in cfg["arms"]:
            directory = suite / "arms" / arm["id"]
            calibration = source_protocol.read_json(directory / "calibration/completed.json")
            test = source_protocol.read_json(directory / "test/completed.json")
            self.assertFalse(calibration["targets_passed"], arm["id"])
            self.assertEqual(test["status"], "completed")
            self.assertEqual(test["calibration_status"], calibration["calibration_status"])
            self.assertFalse(test["test_used_for_fitting"])
            report = source_protocol.read_json(directory / "test/report.json")
            self.assertFalse(report["arm_predictions_replaced_by_reference"])
            self.assertFalse(report["calibration_gate_is_execution_gate"])
            self.assertTrue((directory / "test/per_species.csv").is_file())
            metrics = source_protocol.read_json(directory / "test/metrics.json")
            self.assertEqual(metrics["known"]["sample_count"], 4)
            predictions = source_fixture.read_records(directory / "test/predictions.jsonl")
            aliases = [row for row in predictions if row["image_sha256"] == self.groups["test_known"][0]["image_sha256"]]
            self.assertEqual([row["evaluation_weight"] for row in aliases], [1, 0])
            if arm["kind"] == "baseline":
                self.assertFalse((directory / "training").exists())
                continue
            encoder, evidence, bank, receipt = training.load_arm_model(source, directory / "training", self.device)
            loaded[arm["id"]] = encoder, evidence
            self.assertFalse(encoder.training)
            self.assertFalse(evidence.training)
            self.assertTrue(all(not parameter.requires_grad for parameter in encoder.parameters()))
            self.assertTrue(all(not parameter.requires_grad for parameter in evidence.parameters()))
            self.assertGreater(receipt["parameter_delta"]["l2"], 0)
            self.assertGreater(receipt["parameter_delta"]["changed_parameter_tensors"], 0)
            self.assertEqual(receipt["support_splits"], ["train"])
            self.assertFalse(receipt["unknown_images_used_for_gradients"])
            self.assertTrue(torch.equal(encoder.backbone.visual_gain,
                                        source.encoder.backbone.visual_gain))
            self.assertNotIn("encoder.backbone.visual_gain", receipt["trainable_parameter_names"])
            self.assertTrue(set(bank.hashes) <= train_hashes)
            # Match the saved support tensors to a fresh extraction from this
            # selected model, exposing stale source-bank reuse.
            rebuilt, _, _ = support.reference_bank(encoder,
                toy.loader(train_rows, source.config, source.meta), train_rows,
                source.config, source.meta, self.device)
            # Compare the serialized bank: its loader re-normalizes float32
            # vectors and can add one rounding step to the in-memory bank.
            saved_bank = support._load_torch(directory / "training/support.pth")["bank"]
            for key, value in saved_bank.items():
                if torch.is_tensor(value):
                    self.assertTrue(torch.equal(value, rebuilt.state_dict()[key]), (arm["id"], key))
                else:
                    self.assertEqual(value, rebuilt.state_dict()[key], (arm["id"], key))
            if arm["kind"] == "finetune":
                self.assertGreater(receipt["optimizer_steps"], 0)
                self.assertGreaterEqual(receipt["best_epoch"], 1)
                self.assertEqual(receipt["gradient_splits"], ["train"])
                self.assertFalse(receipt["baseline_selection"]["used_as_primary"])
                self.assertEqual(receipt["source_binding"], source.binding)
                if arm["scope"] == "heads":
                    for name, value in encoder.state_dict().items():
                        if name.startswith("backbone."):
                            self.assertTrue(torch.equal(value, original_encoder[name]), (arm["id"], name))
                else:
                    self.assertFalse(torch.equal(encoder.backbone.prompt_learner,
                                                  source.encoder.backbone.prompt_learner))
            else:
                self.assertEqual(receipt["optimizer_steps"], 0)
                self.assertEqual(receipt["gradient_splits"], [])

        # E05 is its declared fixed interpolation, never a hidden fifth
        # gradient optimization or a TEST-selected blend coefficient.
        for index, initial in enumerate((original_encoder, original_evidence)):
            updated = loaded["E04_hierarchy_anchor"][index].state_dict()
            blended = loaded["E05_hierarchy_blend"][index].state_dict()
            for name, old in initial.items():
                expected = (old.float() * .5 + updated[name].float() * .5).to(old.dtype) if old.is_floating_point() else old
                self.assertTrue(torch.equal(blended[name], expected), name)
        for model, initial in ((source.encoder, original_encoder), (source.evidence, original_evidence)):
            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, initial[name]), name)
        self.assertEqual(source_fixture.artifact_snapshot(self.source), original)


if __name__ == "__main__":
    unittest.main()
