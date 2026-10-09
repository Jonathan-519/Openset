"""Eleven-arm CPU lifecycle with real CLIP cores, prompts and SGD updates.

Only source filesystem/data I/O is synthetic. The image cache, vanilla CLIP
forward, template tokenization, episode models, calibration and TEST inference
execute their production implementations. No uploaded experiment is needed.
"""
import copy
import json
from pathlib import Path
from unittest.mock import patch
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from tests import test_taxosafe_refine_pipeline as reference_fixture
from tests import test_taxosafe_support_pipeline as data_fixture
from taxosafe_support import pipeline as support
from taxosafe_refine import importer
from taxosafe_discovery import backend, features, protocol, reporting, runner


class TinyCoreBackbone(data_fixture.TinyBackbone):
    """Keep the source's easy synthetic task, with a genuine frozen CLIP core."""
    dimension = 32

    def __init__(self):
        from models.model import CLIP
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1229)
            core = CLIP(32, 8, 1, 64, 4, 32, 49408, 64, 1, 1).float().eval().requires_grad_(False)
        self.model.image_encoder = core.visual
        self.model.text_encoder = nn.Module()
        for key in ("transformer", "positional_embedding", "ln_final", "text_projection"):
            setattr(self.model.text_encoder, key, getattr(core, key))
        self.token_embedding = core.token_embedding

    def encode_image_with_spatial(self, images, normalize=True):
        # Cache inputs embed exactly the original vector into four pixels.
        vector = images[:, 0, 0, :4] if images.ndim == 4 else images
        global_features, patches = super().encode_image_with_spatial(vector, normalize)
        return F.pad(global_features, (0, 28)), F.pad(patches, (0, 28))

    def encode_text(self, names, normalize=True):
        return F.pad(super().encode_text(names, normalize), (0, 28))


def image_loader(rows, cfg, meta, training=False):
    """Synthetic real-shaped images; only this data-input seam is replaced."""
    for vectors, labels, indices in data_fixture.loader(rows, cfg, meta, training):
        grid = torch.arange(3 * 8 * 8, dtype=torch.float32).reshape(1, 3, 8, 8)
        images = torch.sin(grid * .13 + vectors[:, 0, None, None, None])
        images = images + vectors[:, 1, None, None, None] * torch.cos(grid * .07)
        images = images + vectors[:, 2, None, None, None] * torch.sin(grid * .19)
        images = images + vectors[:, 3, None, None, None] * torch.cos(grid * .23)
        images[:, 0, 0, :4] = vectors
        yield images, labels, indices


class DiscoveryLifecycle(reference_fixture.FrozenPipelineContracts):
    def setUp(self):
        super().setUp()
        self.stack.enter_context(patch.object(support, "make_backbone", side_effect=lambda *args: TinyCoreBackbone()))
        # Identical inputs with incompatible known/unknown decisions ensure
        # that every arm fails the joint gates without mocking gate outcomes.
        for prefix in ("val", "test"):
            known = self.groups[prefix + "_known"]
            for i, row in enumerate(self.groups[prefix + "_intra"]):
                row["vector"] = list(known[i % 4]["vector"])
                row["true_parent"] = known[i % 4]["true_parent"]
            for i, row in enumerate(self.groups[prefix + "_extra"]):
                row["vector"] = list(known[i % 4]["vector"])

    def _prepare(self):
        self.make_source()
        self.reference = importer.load_reference(self.source, self.device)
        self.info = importer.inspect_reference(self.source)
        self.source_files = reference_fixture.artifact_snapshot(self.source)
        self.source_states = [reference_fixture.tensor_snapshot(module)
                              for module in (self.reference.encoder, self.reference.evidence)]
        self.source_bank = self.reference.bank.state_dict()
        self.stack.enter_context(patch.object(backend, "inspect_reference", return_value=self.info))
        self.stack.enter_context(patch.object(backend, "load_reference", return_value=self.reference))
        self.cfg = copy.deepcopy(protocol.DEFAULTS)
        self.cfg["verifier"].update(folds=2, epochs=2, batch_size=64)
        self.cfg["projection"].update(epochs=2, batch_size=4, bottleneck=4)
        self.cfg["prompt"].update(epochs=2, batch_size=4)
        self.cfg["calibration"].update(min_parent_known=1, shrinkage=1.)
        self.cfg = protocol.validate_config(self.cfg)
        self.suite = self.root / "discovery"
        self.snapshot = runner._initialize(self.suite, self.cfg, self.info, "cpu")
        for stage in ("train", "development"):
            with patch.object(support, "make_loader", side_effect=image_loader):
                backend.prepare_cache(self.suite, stage, "cpu")
            runner._complete_stage(self.suite, None, "cache_" + stage, self.snapshot)

    def _model(self, arm_id):
        return support._load_torch(self.suite / "arms" / arm_id / "training/model.pth")

    def _json(self, arm_id, stage, filename="completed.json"):
        return protocol.read_json(self.suite / "arms" / arm_id / stage / filename)

    def _assert_source_unchanged(self):
        self.assertEqual(reference_fixture.artifact_snapshot(self.source), self.source_files)
        for module, state in zip((self.reference.encoder, self.reference.evidence), self.source_states):
            self.assert_frozen(module, state)
        for key, value in self.reference.bank.state_dict().items():
            if torch.is_tensor(value):
                self.assertTrue(torch.equal(value, self.source_bank[key]), key)
            else:
                self.assertEqual(value, self.source_bank[key], key)

    def test_eleven_real_arms_failed_gates_still_test_without_any_refitting(self):
        self._prepare()
        train_cache, _ = backend._load_cache(self.suite, "train", self.cfg, self.info)
        self.assertNotEqual(features.tensor_hash(train_cache["text"]["single_leaf"]),
                            features.tensor_hash(train_cache["text"]["ensemble_leaf"]))
        for arm in self.cfg["arms"]:
            arm_id = arm["id"]
            backend.fit_arm(self.suite, arm_id, "cpu")
            runner._complete_stage(self.suite, arm_id, "training", self.snapshot)
            backend.calibrate_arm(self.suite, arm_id, "cpu")
            runner._complete_stage(self.suite, arm_id, "calibration", self.snapshot)
            self.assertFalse(self._json(arm_id, "calibration")["targets_passed"], arm_id)
        for arm_id in ("D05_episode_bce", "D06_episode_rank", "D08_residual", "D09_coop", "D10_forced_prompt"):
            self.assertGreater(self._json(arm_id, "training")["optimizer_steps"], 0, arm_id)
        projection = self._model("D08_residual")["fit_report"]["projection"]
        self.assertGreater(projection["parameter_delta_l2"], 0)
        self.assertEqual(projection["gradient_splits"], ["train"])
        d06 = self._json("D06_episode_rank", "training")
        d07 = self._json("D07_parentwise", "training")
        self.assertEqual(d06["model"]["sha256"], d07["model"]["sha256"])
        self.assertEqual(d07["optimizer_steps"], 0)
        global_router = self._json("D06_episode_rank", "calibration", "router.json")
        local_router = self._json("D07_parentwise", "calibration", "router.json")
        self.assertEqual(global_router["variant"], "global")
        self.assertEqual(local_router["variant"], "parentwise")
        self.assertNotEqual(global_router, local_router)
        coop, fa = (self._model(arm_id)["fit_report"]["prompt"]
                    for arm_id in ("D09_coop", "D10_forced_prompt"))
        for report in (coop, fa):
            self.assertGreater(report["parameter_delta_l2"], 0)
            self.assertEqual(report["changed_tensor_count"], 2)
            self.assertGreater(report["history"][0]["context_gradient_l2_sum"], 0)
            self.assertEqual(report["frozen_core_sha256_before"], report["frozen_core_sha256_after"])
            self.assertTrue(all(value < 1e-6 for value in report["initial_reference_max_abs_gap"].values()))
        self.assertEqual(coop["batch_order_sha256"], fa["batch_order_sha256"])
        self.assertEqual(coop["provenance"]["initial_context"], fa["provenance"]["initial_context"])
        self.assertEqual(coop["provenance"]["reference_text_sha256"], fa["provenance"]["reference_text_sha256"])
        self.assertNotEqual(self._model("D09_coop")["prompt"]["context"]["leaf"].tolist(),
                            self._model("D10_forced_prompt")["prompt"]["context"]["leaf"].tolist())
        selection = reporting.freeze_dev_selection(self.suite)
        self.assertIsNone(selection["qualified_candidate_arm_id"])
        self.assertEqual(selection["recommendation_arm_id"], "D00_reference")
        selection_hash = protocol.file_hash(self.suite / "dev_selection.json")
        # Remove every fitting/development dataset. Then make fitting functions
        # explode, so TEST can only succeed through serialized inference states.
        self.groups = {key: value for key, value in self.groups.items() if key.startswith("test_")}
        with patch("taxosafe_discovery.geometry.GeometryBank.fit", side_effect=AssertionError("TEST geometry fit")), \
                patch("taxosafe_discovery.verifier.SharedVerifier.fit", side_effect=AssertionError("TEST verifier fit")), \
                patch("taxosafe_discovery.verifier.build_episodes", side_effect=AssertionError("TEST episodes")), \
                patch("taxosafe_discovery.projection_training.fit_projection", side_effect=AssertionError("TEST projection fit")), \
                patch("taxosafe_discovery.prompt_training.fit_prompt", side_effect=AssertionError("TEST prompt fit")), \
                patch("taxosafe_discovery.calibration.fit_router", side_effect=AssertionError("TEST calibration fit")), \
                patch("taxosafe_discovery.calibration.crossfit_audit", side_effect=AssertionError("TEST crossfit")), \
                patch.object(support, "training_loss", side_effect=AssertionError("TEST reference gradient")), \
                patch.object(support, "reference_bank", side_effect=AssertionError("TEST support fit")):
            with patch.object(support, "make_loader", side_effect=image_loader):
                backend.prepare_cache(self.suite, "test", "cpu")
            runner._complete_stage(self.suite, None, "cache_test", self.snapshot)
            for arm in self.cfg["arms"]:
                arm_id = arm["id"]
                backend.test_arm(self.suite, arm_id, "cpu")
                runner._complete_stage(self.suite, arm_id, "test", self.snapshot)
                receipt = self._json(arm_id, "test")
                self.assertFalse(receipt["calibration_gate_is_execution_gate"])
                self.assertTrue(receipt["test_allowed_after_failed_gates"])
                self.assertEqual(receipt["summary"]["counts"]["known"], 4)
                predictions = reference_fixture.read_records(self.suite / "arms" / arm_id / "test/predictions.jsonl")
                self.assertEqual(len(predictions), 11)
                self.assertEqual(sum(row["evaluation_weight"] for row in predictions), 10)
        summary = reporting.summarize_suite(self.suite)
        self.assertEqual(protocol.file_hash(self.suite / "dev_selection.json"), selection_hash)
        self.assertEqual(len(list(self.suite.glob("arms/*/test/completed.json"))), 11)
        self._assert_source_unchanged()
        self._assert_tamper_rejected()

    def _assert_tamper_rejected(self):
        arm = next(arm for arm in self.cfg["arms"] if arm["id"] == "D03_single_prompt")
        directory = self.suite / "arms" / arm["id"] / "training"
        model_path, receipt_path = directory / "model.pth", directory / "completed.json"
        model_bytes, receipt_bytes = model_path.read_bytes(), receipt_path.read_bytes()
        try:
            model_path.write_bytes(model_bytes + b"tampered")
            with self.assertRaisesRegex(ValueError, "artifact"):
                backend._load_model(self.suite, arm, self.cfg, self.info)
            model_path.write_bytes(model_bytes)
            payload = support._load_torch(model_path)
            payload["text"]["single_leaf"][0, 0] += .1
            support._save_torch(model_path, payload)
            receipt = protocol.read_json(receipt_path)
            receipt["model"]["sha256"] = protocol.file_hash(model_path)
            receipt["artifacts"]["model"]["sha256"] = protocol.file_hash(model_path)
            protocol.write_json(receipt_path, receipt)
            with self.assertRaisesRegex(ValueError, "text|prompt|specification"):
                backend._load_model(self.suite, arm, self.cfg, self.info)
        finally:
            model_path.write_bytes(model_bytes)
            receipt_path.write_bytes(receipt_bytes)
        cache_path = self.suite / "cache/test/features.pth"
        cache_bytes = cache_path.read_bytes()
        try:
            cache_path.write_bytes(cache_bytes + b"tampered")
            with self.assertRaisesRegex(ValueError, "artifact"):
                backend._load_cache(self.suite, "test", self.cfg, self.info)
        finally:
            cache_path.write_bytes(cache_bytes)
        runner._verify_stage(self.suite, arm["id"], "training", self.snapshot)
        runner._verify_stage(self.suite, None, "cache_test", self.snapshot)


for _name in dir(reference_fixture.FrozenPipelineContracts):
    if _name.startswith("test_") and _name not in DiscoveryLifecycle.__dict__:
        setattr(DiscoveryLifecycle, _name, None)


if __name__ == "__main__":
    unittest.main()
