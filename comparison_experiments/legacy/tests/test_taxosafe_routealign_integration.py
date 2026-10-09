"""Real five-arm lifecycle on toy vectors; not research performance results.

Only model/image I/O and subprocess isolation are substituted. Training,
TRAIN proximity, DEV calibration/cross-fit, frozen selection, every TEST,
artifact bindings and comparison exports execute their production code.
"""
import copy
import csv
import unittest
from unittest.mock import patch

import torch

from taxosafe_support import pipeline as support
from taxosafe_support import protocol as source_protocol
from tests import test_taxosafe_refine_pipeline as fixture
from tests.test_taxosafe_sweep_integration import TinyFrozenCoreBackbone


class RouteAlignIntegrationTests(unittest.TestCase):
    setUpClass = classmethod(fixture.FrozenPipelineContracts.setUpClass.__func__)
    tearDownClass = classmethod(fixture.FrozenPipelineContracts.tearDownClass.__func__)
    setUp = fixture.FrozenPipelineContracts.setUp
    make_source = fixture.FrozenPipelineContracts.make_source
    load_stage = fixture.FrozenPipelineContracts.load_stage
    read_split = fixture.FrozenPipelineContracts.read_split

    def test_actual_five_arm_suite_keeps_failed_gates_testable_and_binds_reused_models(self):
        from taxosafe_refine import importer
        from taxosafe_routealign import calibration, evaluation, protocol, runner, training
        from taxosafe_routealign.proximity import ProximityBank

        self.stack.enter_context(patch.object(support, "make_backbone",
                                              side_effect=lambda *args: TinyFrozenCoreBackbone()))
        # Known and near images have distinct hashes but identical inputs.
        # No router can accept all four known leaves and correctly reject
        # their near twins, so passing all four DEV gates is impossible.
        for known, near in zip(self.groups["val_known"], self.groups["val_intra"]):
            near["vector"] = copy.deepcopy(known["vector"])
        self.make_source()
        original_files = fixture.artifact_snapshot(self.source)
        source = importer.load_reference(self.source, self.device)
        source_encoder = fixture.tensor_snapshot(source.encoder)
        source_evidence = fixture.tensor_snapshot(source.evidence)
        train_hashes = [row["image_sha256"] for row in self.groups["train"]]
        cfg = protocol.effective_config(protocol.PROJECT_ROOT / protocol.DEFAULT_CONFIG)
        cfg["budget"].update(epochs=2, min_epochs=1, patience=2, batches_per_epoch=2)
        cfg["router"]["grid_points"] = 3
        suite = self.root / "routealign"
        stages, frozen_selection, fit_calls, source_instances = [], [], [], []
        test_started = False
        real_load_reference, real_fit = importer.load_reference, ProximityBank.fit

        def load_reference(*args, **kwargs):
            reference = real_load_reference(*args, **kwargs)
            source_instances.append((reference, fixture.tensor_snapshot(reference.encoder),
                                     fixture.tensor_snapshot(reference.evidence),
                                     copy.deepcopy(reference.bank.state_dict())))
            return reference

        def fit_proximity(*args, **kwargs):
            self.assertFalse(test_started, "TEST cannot call ProximityBank.fit")
            # Both statistical fits must use all and only the known TRAIN
            # rows, including their original manifest order and content IDs.
            self.assertEqual(list(args[3]), train_hashes)
            self.assertEqual(len(args[0]), len(train_hashes))
            fit_calls.append(list(args[3]))
            return real_fit(*args, **kwargs)

        def launch_in_process(directory, arm_id, stage, device):
            nonlocal test_started
            stages.append((arm_id, stage))
            if stage == "test":
                for arm in cfg["arms"]:
                    self.assertTrue((suite / "arms" / arm["id"] / "calibration/completed.json").is_file())
                if not test_started:
                    test_started = True
                    for owner, name in ((calibration, "fit_router"), (evaluation.membership, "calibrate"),
                                        (evaluation, "load_training_rows"), (support, "training_loss")):
                        self.stack.enter_context(patch.object(owner, name,
                            side_effect=AssertionError("TEST cannot fit or read known TRAIN: " + name)))
                    # Cached DEV predictions may be audited for immutable
                    # selection, but no TRAIN/DEV image rows remain available.
                    for split in ("train", "val_known", "val_intra", "val_extra"):
                        del self.groups[split]
                    self.calls.clear()
                frozen_selection.append((suite / "dev_selection.json").read_bytes())
            logs = suite / "logs" / arm_id
            logs.mkdir(parents=True, exist_ok=True)
            stdout, stderr = logs / (stage + ".stdout.log"), logs / (stage + ".stderr.log")
            stdout.write_text("Toy integration runs the actual stage worker in process.\n", encoding="utf-8")
            stderr.write_text("", encoding="utf-8")
            runner.worker(directory, arm_id, stage, device)
            return 0, stdout, stderr

        with patch.object(importer, "load_reference", side_effect=load_reference), \
                patch.object(ProximityBank, "fit", side_effect=fit_proximity), \
                patch.object(runner, "_launch_stage", side_effect=launch_in_process):
            runner.execute_suite(cfg, self.source, suite, device="cpu")

        ids = [arm["id"] for arm in cfg["arms"]]
        self.assertEqual([item for item in stages if item[1] == "training"], [("A01_evidence_anchor", "training")])
        first_test = next(i for i, item in enumerate(stages) if item[1] == "test")
        self.assertEqual(stages[first_test:], [(arm_id, "test") for arm_id in ids])
        self.assertEqual(self.calls, [("test", source_protocol.STAGE_SPLITS["test"])] * 5)
        self.assertEqual(len(fit_calls), 2, "A04 must reuse A03 TRAIN proximity without fitting again")
        self.assertEqual(len(frozen_selection), 5)
        self.assertTrue(all(value == frozen_selection[0] for value in frozen_selection))
        self.assertEqual((suite / "dev_selection.json").read_bytes(), frozen_selection[0])
        selection = source_protocol.read_json(suite / "dev_selection.json")
        self.assertFalse(selection["selection_uses_test"])
        self.assertFalse(selection["test_predictions_read"])
        completed = source_protocol.read_json(suite / "suite_completed.json")
        self.assertTrue(completed["workflow_completed"])
        self.assertTrue(completed["all_valid_calibrations_test_attempted_regardless_of_gate"])
        self.assertFalse(completed["test_used_for_selection"])
        self.assertEqual(completed["technical_failure_arms"], [])
        self.assertEqual(completed["completed_test_arms"], ids)

        calibrations, bindings = {}, {}
        for arm_id in ids:
            directory = suite / "arms" / arm_id
            calibrated = source_protocol.read_json(directory / "calibration/completed.json")
            tested = source_protocol.read_json(directory / "test/completed.json")
            calibrations[arm_id], bindings[arm_id] = calibrated, calibrated["binding"]
            self.assertFalse(calibrated["targets_passed"], arm_id)
            self.assertEqual(tested["status"], "completed")
            self.assertEqual(tested["binding"], calibrated["binding"])
            self.assertEqual(tested["calibration_status"], calibrated["calibration_status"])
            self.assertFalse(tested["test_used_for_fitting"])
            self.assertEqual(tested["calibration_router_sha256"],
                             source_protocol.file_hash(directory / "calibration/router.json"))
            self.assertFalse(tested["summary"]["calibration_gate_is_execution_gate"])
            self.assertFalse(tested["summary"]["arm_predictions_replaced_by_reference"])
            self.assertTrue((directory / "test/per_species.csv").is_file())
            self.assertEqual(source_protocol.read_json(directory / "test/metrics.json")["known"]["sample_count"], 4)
            predictions = fixture.read_records(directory / "test/predictions.jsonl")
            # The toy has 11 manifest rows but 10 unique contents: the same
            # alias semantics as the production TEST manifest's 927/926.
            self.assertEqual(len(predictions), 11)
            self.assertEqual(sum(row["evaluation_weight"] for row in predictions), 10)
            aliases = [row for row in predictions if row["image_sha256"] == self.groups["test_known"][0]["image_sha256"]]
            self.assertEqual([row["evaluation_weight"] for row in aliases], [1, 0])
            for key in ("prediction_type", "parent", "leaf", "log_probs"):
                self.assertEqual(aliases[0][key], aliases[1][key], (arm_id, key))
            if arm_id != "A01_evidence_anchor":
                self.assertFalse((directory / "training").exists(), arm_id)

        for key in ("checkpoint", "support"):
            self.assertEqual(bindings["A00_reference"][key], bindings["A02_proximity"][key])
            self.assertEqual(bindings["A01_evidence_anchor"][key], bindings["A03_combined"][key])
            self.assertEqual(bindings["A01_evidence_anchor"][key], bindings["A04_parent_rerank"][key])
            self.assertEqual(bindings["A00_reference"][key]["sha256"], source.training[key]["sha256"])
        updated_dir = suite / "arms/A01_evidence_anchor/training"
        encoder, evidence, bank, receipt = training.load_arm_model(source, updated_dir, self.device)
        self.assertGreater(receipt["optimizer_steps"], 0)
        self.assertGreater(receipt["selected_checkpoint_optimizer_steps"], 0)
        self.assertGreaterEqual(receipt["best_epoch"], 1)
        self.assertGreater(receipt["parameter_delta"]["l2"], 0)
        self.assertTrue(any(not torch.equal(value, source_evidence[name]) for name, value in evidence.state_dict().items()))
        for name, value in encoder.state_dict().items():
            if name.startswith("backbone."):
                self.assertTrue(torch.equal(value, source_encoder[name]), name)

        a03, a04 = suite / "arms/A03_combined/calibration", suite / "arms/A04_parent_rerank/calibration"
        reuse = source_protocol.read_json(a04 / "proximity_fit.json")
        self.assertEqual(reuse["origin"]["operation"], "reuse_same_checkpoint_train_bank")
        self.assertEqual(reuse["origin"]["source_receipt_sha256"], source_protocol.file_hash(a03 / "completed.json"))
        self.assertEqual(calibrations["A03_combined"]["artifacts"]["proximity"]["sha256"],
                         source_protocol.file_hash(a03 / "proximity.pth"))
        self.assertEqual(calibrations["A04_parent_rerank"]["artifacts"]["proximity"]["sha256"],
                         source_protocol.file_hash(a04 / "proximity.pth"))
        before = support._load_torch(a03 / "proximity.pth")
        after = support._load_torch(a04 / "proximity.pth")
        for key, value in before.items():
            if key == "bank":
                for name, state in value.items():
                    if torch.is_tensor(state):
                        self.assertTrue(torch.equal(state, after[key][name]), name)
                    else:
                        self.assertEqual(state, after[key][name], name)
            else:
                self.assertEqual(value, after[key], key)
        # Reuse requires the same actual weights/support, not just the arm's
        # name or a compatible tensor shape.
        for key in ("checkpoint_sha256", "support_sha256"):
            altered = copy.deepcopy(after)
            altered[key] = "0" * 64
            bad = self.root / ("changed_" + key + ".pth")
            support._save_torch(bad, altered)
            with self.assertRaisesRegex(ValueError, "binding changed"):
                evaluation._load_proximity(bad, source, bindings["A04_parent_rerank"], cfg["proximity"])

        with (suite / "comparison_all.csv").open(encoding="utf-8-sig", newline="") as handle:
            comparison = {row["arm_id"]: row for row in csv.DictReader(handle)}
        self.assertEqual(set(comparison), set(ids))
        self.assertGreater(int(comparison["A01_evidence_anchor"]["optimizer_steps"]), 0)
        for arm_id in ids:
            self.assertEqual(comparison[arm_id]["test_execution"], "completed")
            if arm_id != "A01_evidence_anchor":
                self.assertEqual(int(comparison[arm_id]["optimizer_steps"]), 0)

        self.assertEqual(len(source_instances), len(stages))
        for reference, old_encoder, old_evidence, old_bank in source_instances:
            for model, old in ((reference.encoder, old_encoder), (reference.evidence, old_evidence)):
                self.assertFalse(model.training)
                self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.parameters()))
                for name, value in model.state_dict().items():
                    self.assertTrue(torch.equal(value, old[name]), name)
            for name, value in reference.bank.state_dict().items():
                if torch.is_tensor(value):
                    self.assertTrue(torch.equal(value, old_bank[name]), name)
                else:
                    self.assertEqual(value, old_bank[name], name)
        self.assertEqual(fixture.artifact_snapshot(self.source), original_files)


if __name__ == "__main__":
    unittest.main()
