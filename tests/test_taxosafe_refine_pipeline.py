"""Adversarial CPU integration checks for a frozen reference refinement run.

The source run uses the real artifact/receipt formats and deterministic toy
vectors. No CLIP downloads, user images, or external historical runs are needed.
"""
from contextlib import ExitStack
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from taxosafe_support import pipeline as reference_pipeline
from taxosafe_support import protocol as reference_protocol
from tests import test_taxosafe_support_pipeline as fixture
from tests.test_taxosafe_support_reference_pipeline import reference_config


def tensor_snapshot(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def artifact_snapshot(directory):
    return {str(path.relative_to(directory)): reference_protocol.file_hash(path)
            for path in sorted(Path(directory).rglob("*")) if path.is_file()}


def read_records(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


class FrozenPipelineContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "reference"
        self.directory = self.root / "refinement"
        self.device = torch.device("cpu")
        self.source_cfg, self.groups = fixture.fixture()
        self.source_cfg = reference_config(self.source_cfg)
        hierarchy = self.root / "hierarchy.json"
        hierarchy.write_text(json.dumps(fixture.META), encoding="utf-8")
        train_manifest = self.root / "train.txt"
        train_manifest.write_text("fixture_train", encoding="utf-8")
        self.source_cfg["model"]["arch"] = "maple"
        self.source_cfg["data"].update(hierarchy=str(hierarchy), num_known_leaves=4,
                                       train=str(train_manifest))
        self.source_cfg["evaluation_gates"] = {
            "known_e2e": {"operator": ">", "target": .9},
            "near_correct_fallback": {"operator": ">=", "target": .85},
            "extra_root_rejection": {"operator": ">", "target": .9},
            "open_world_leaf_precision": {"operator": ">", "target": .9}}
        self.calls = []
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch("sys.stdout", new_callable=io.StringIO))
        self.stack.enter_context(patch.object(reference_pipeline, "hierarchy", return_value=fixture.META))
        self.stack.enter_context(patch.object(reference_pipeline, "make_loader", side_effect=fixture.loader))
        self.stack.enter_context(patch.object(reference_pipeline, "make_backbone",
                                              side_effect=lambda *args: fixture.TinyBackbone()))
        self.stack.enter_context(patch.object(reference_pipeline, "load_stage_rows", side_effect=self.load_stage))
        self.stack.enter_context(patch.object(reference_protocol, "read_split", side_effect=self.read_split))

    def make_source(self, with_test=False):
        # The configuration and hierarchy digests are genuine. Only the source
        # code version is historical: the tiny source exercises the supported
        # reference architecture without recreating a checkout of that commit.
        historical = reference_protocol.signature(self.source_cfg)
        historical["support_code"] = "ef5526b8390b838473272eaaf56dcdb128e37e9e2c3a543896d8cd5a30e9aed4"
        with patch.object(reference_pipeline, "signature", return_value=historical):
            reference_pipeline.train(self.source_cfg, self.source, self.device)
            reference_pipeline.calibrate_run(self.source_cfg, self.source, self.device)
            if with_test:
                reference_pipeline.test_run(self.source_cfg, self.source, self.device)
        self.calls.clear()

    def configure_refinement(self):
        from taxosafe_refine import protocol
        self.cfg = protocol.effective_config(reference_protocol.PROJECT_ROOT / protocol.DEFAULT_CONFIG)
        self.cfg["reconstruction"]["rank"] = 2
        self.cfg["training"].update(epochs=4, batch_size=4, patience=3, learning_rate=.02)
        self.cfg["calibration"].update(grid_points=5, source_loo=False)

    def load_stage(self, cfg, stage, meta, forbidden_hashes=(), forbidden_sources=()):
        allowed = reference_protocol.STAGE_SPLITS[stage]
        self.calls.append((stage, allowed))
        selected = {name: copy.deepcopy(self.groups[name]) for name in allowed}
        audit = reference_protocol.audit_rows(selected, forbidden_hashes, forbidden_sources,
                                             allow_within_split=stage == "test")
        for split in audit:
            audit[split]["manifest_sha256"] = hashlib.sha256(("fixture_" + split).encode()).hexdigest()
        return selected, audit

    def read_split(self, cfg, split, meta):
        self.assertEqual(split, "train", "Refinement fitting cannot read external development or test images")
        self.calls.append(("train", ("train",)))
        return copy.deepcopy(self.groups[split])

    def assert_frozen(self, module, snapshot):
        self.assertFalse(module.training)
        self.assertEqual(set(module.state_dict()), set(snapshot))
        for key, value in module.state_dict().items():
            self.assertTrue(torch.equal(value.detach().cpu(), snapshot[key]), key)
        for parameter in module.parameters():
            self.assertFalse(parameter.requires_grad)
            self.assertIsNone(parameter.grad)

    def test_fallback_reproduces_source_decisions_for_all_record_types(self):
        from taxosafe_refine import calibration
        self.make_source()
        baseline = reference_protocol.read_json(self.source / "calibration/router.json")
        rows = read_records(self.source / "calibration/development_scores.jsonl")
        for i, row in enumerate(rows):
            row["reconstruction_score"] = -1e6 if i % 2 else 1e6
        groups = [[row for row in rows if row["status"] == status]
                  for status in ("known", "intra", "extra")]
        router = calibration.calibrate(*groups, baseline, fixture.META,
                                       {"grid_points": 5, "source_loo": False})
        router.update(reconstruction_gate_enabled=False,
                      leaf_threshold=baseline["leaf_threshold"])
        expected = reference_pipeline.apply_router(rows, baseline, fixture.META)
        actual = calibration.apply_router(rows, router, fixture.META)
        exact_fields = ("prediction_type", "output_node", "parent", "leaf",
                        "candidate_parent", "candidate_leaf", "root_knownness_score",
                        "parent_membership_score", "leaf_membership_score", "parent_threshold",
                        "support_evidence", "log_probs")
        for old, new in zip(expected, actual):
            for field in exact_fields:
                self.assertEqual(new[field], old[field], field)

    def test_reviewed_historical_import_matches_original_reference_forward(self):
        from taxosafe_refine import importer
        self.make_source()
        source_bytes = artifact_snapshot(self.source)
        imported = importer.load_reference(self.source, self.device)
        self.assertEqual(imported.binding["source_commit"], "54dff0ed81771e4cbc14b844f22ebea06e3f9c8d")
        self.assertEqual(imported.binding["runtime_commit"], "4394be543badc2a7def2fa5a60b45b531b5a861d")
        self.assert_frozen(imported.encoder, tensor_snapshot(imported.encoder))
        self.assert_frozen(imported.evidence, tensor_snapshot(imported.evidence))
        groups = {name: self.groups[name] for name in reference_protocol.STAGE_SPLITS["calibrate"]}
        imported_rows, _ = reference_pipeline.collect(groups, imported.config, imported.meta,
                imported.encoder, imported.evidence, imported.bank, self.device)
        original = read_records(self.source / "calibration/development_scores.jsonl")
        actual = [row for rows in imported_rows.values() for row in rows]
        self.assertEqual(len(actual), len(original))
        for old, new in zip(original, actual):
            for field in ("image_sha256", "support_evidence", "log_probs", "global_pred_leaf"):
                self.assertEqual(new[field], old[field], field)
        self.assertEqual(artifact_snapshot(self.source), source_bytes)

    def test_source_binary_or_router_changes_are_rejected_before_image_loading(self):
        from taxosafe_refine import importer
        self.make_source()
        for relative in ("training/best.pth", "training/support.pth", "calibration/router.json"):
            path = self.source / relative
            original = path.read_bytes()
            try:
                path.write_bytes(original + b"\nchanged")
                with self.subTest(artifact=relative), self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
                    importer.load_reference(self.source, self.device)
                self.assertEqual(self.calls, [])
            finally:
                path.write_bytes(original)

    def test_fitting_changes_only_the_new_verifier_and_caches_known_train(self):
        from taxosafe_refine import importer, pipeline
        self.make_source()
        self.configure_refinement()
        baseline = importer.load_reference(self.source, self.device)
        old_encoder, old_evidence = tensor_snapshot(baseline.encoder), tensor_snapshot(baseline.evidence)
        old_bank = baseline.bank.state_dict()
        source_bytes = artifact_snapshot(self.source)
        with patch.object(pipeline, "load_reference", return_value=baseline), \
                patch.object(pipeline, "fit_cached_features", wraps=pipeline.fit_cached_features) as fit_cache:
            pipeline.fit(self.cfg, self.source, self.directory, self.device)
        cached_features, cached_labels, cached_hashes = fit_cache.call_args.args[:3]
        self.assertFalse(cached_features.requires_grad)
        self.assertEqual(cached_hashes, [row["image_sha256"] for row in self.groups["train"]])
        self.assertEqual(cached_labels.tolist(), [row["true_leaf"] for row in self.groups["train"]])
        self.assert_frozen(baseline.encoder, old_encoder)
        self.assert_frozen(baseline.evidence, old_evidence)
        for key, before in old_bank.items():
            after = baseline.bank.state_dict()[key]
            if torch.is_tensor(before):
                self.assertTrue(torch.equal(before, after), key)
                self.assertFalse(after.requires_grad)
            else:
                self.assertEqual(before, after, key)
        self.assertEqual(artifact_snapshot(self.source), source_bytes)
        self.assertTrue(self.calls)
        self.assertTrue(all(stage == "train" for stage, _ in self.calls))
        receipt = reference_protocol.read_json(self.directory / "refinement/completed.json")
        self.assertEqual(receipt["fit_splits"], ["train"])
        self.assertEqual(receipt["frozen_base_trainable"], 0)
        self.assertIs(receipt["test_used_for_fitting"], False)

    def test_completed_fit_rejects_a_changed_but_internally_valid_source_binding(self):
        from taxosafe_refine import importer, pipeline
        self.make_source()
        self.configure_refinement()
        pipeline.fit(self.cfg, self.source, self.directory, self.device)
        path = self.source / "training/completed.json"
        receipt = reference_protocol.read_json(path)
        receipt["review_note"] = "A different source receipt after refinement completed"
        reference_protocol.write_json(path, receipt)
        # This source remains internally self-consistent. The refinement must
        # compare its previously recorded binding, not merely revalidate it.
        importer.inspect_reference(self.source)
        self.calls.clear()
        with self.assertRaises(ValueError):
            pipeline.calibrate_run(self.cfg, self.source, self.directory, self.device)
        self.assertEqual(self.calls, [])
        self.assertFalse((self.directory / "calibration").exists())

    def test_full_cycle_preserves_parent_decisions_and_test_never_refits(self):
        from taxosafe_refine import calibration, importer, pipeline
        self.make_source(with_test=True)
        self.configure_refinement()
        baseline = importer.load_reference(self.source, self.device)
        source_bytes = artifact_snapshot(self.source)
        test_groups = {name: self.groups[name] for name in reference_protocol.STAGE_SPLITS["test"]}
        baseline_scores, _ = reference_pipeline.collect(test_groups, baseline.config, baseline.meta,
                baseline.encoder, baseline.evidence, baseline.bank, self.device)
        baseline_predictions = [row for rows in baseline_scores.values()
                                for row in reference_pipeline.apply_router(rows, baseline.router, fixture.META)]
        pipeline.fit(self.cfg, self.source, self.directory, self.device)
        pipeline.calibrate_run(self.cfg, self.source, self.directory, self.device)
        router = reference_protocol.read_json(self.directory / "calibration/router.json")
        self.assertEqual(router["fixed_parent_threshold"], baseline.router["parent_threshold"])
        self.assertEqual(router["baseline_router"], baseline.router)
        fit_hashes = set(router["fit_image_sha256"])
        test_hashes = {row["image_sha256"] for rows in test_groups.values() for row in rows}
        self.assertFalse(fit_hashes & test_hashes)
        for split in ("train", "val_known", "val_intra", "val_extra"):
            del self.groups[split]
        self.calls.clear()
        with patch.object(calibration, "calibrate", side_effect=AssertionError("TEST cannot fit a router")), \
                patch.object(pipeline, "fit_cached_features", side_effect=AssertionError("TEST cannot fit model weights")):
            pipeline.test_run(self.cfg, self.source, self.directory, self.device)
        self.assertEqual(self.calls, [("test", reference_protocol.STAGE_SPLITS["test"])])
        actual = read_records(self.directory / "test/predictions.jsonl")
        self.assertEqual(len(actual), len(baseline_predictions))
        for old, new in zip(baseline_predictions, actual):
            for field in ("image_sha256", "support_evidence", "log_probs", "candidate_parent",
                          "candidate_leaf", "parent_membership_score", "parent_threshold"):
                self.assertEqual(new[field], old[field], field)
            self.assertEqual(new["prediction_type"] == "global_unknown", old["prediction_type"] == "global_unknown")
        aliases = [row for row in actual if row["image_sha256"] == self.groups["test_known"][0]["image_sha256"]]
        self.assertEqual([row["evaluation_weight"] for row in aliases], [1, 0])
        self.assertEqual(aliases[0]["reconstruction_score"], aliases[1]["reconstruction_score"])
        self.assertEqual(artifact_snapshot(self.source), source_bytes)

    def test_spatial_option_keeps_one_visual_pass_and_nondefault_scale_roundtrips(self):
        from taxosafe_refine import importer, pipeline
        self.make_source()
        self.configure_refinement()
        self.cfg["features"] = "raw_spatial"
        self.cfg["reconstruction"]["scale_init"] = 2.5
        baseline = importer.load_reference(self.source, self.device)
        group = {"test_known": self.groups["test_known"]}
        before = baseline.encoder.backbone.visual_calls
        records, features, _ = pipeline.collect_features(group, self.cfg, baseline, self.device)
        self.assertEqual(baseline.encoder.backbone.visual_calls - before, 1)
        self.assertEqual(features["test_known"].shape, (4, 2, 4))
        original, _ = reference_pipeline.collect(group, baseline.config, baseline.meta,
                baseline.encoder, baseline.evidence, baseline.bank, self.device)
        for old, new in zip(original["test_known"], records["test_known"]):
            self.assertEqual(new["support_evidence"], old["support_evidence"])
            self.assertEqual(new["log_probs"], old["log_probs"])
        pipeline.fit(self.cfg, self.source, self.directory, self.device)
        model, _ = pipeline._load_fit(self.cfg, baseline, self.directory, self.device)
        self.assertFalse(model.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))
        scored, _, _ = pipeline.collect_features(group, self.cfg, baseline, self.device,
                                                  model=model, keep_features=False)
        self.assertTrue(torch.isfinite(torch.tensor([r["reconstruction_score"]
                                                    for r in scored["test_known"]])).all())


if __name__ == "__main__":
    unittest.main()
