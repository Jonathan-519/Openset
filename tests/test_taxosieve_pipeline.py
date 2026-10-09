"""Native TaxoSieve integration with real CPU D05 training and artifact contracts.

Only the image boundary and inspection of an external C00 source are mocked.
The temporary run uses the real initializer, frozen source file, tensor caches,
BCE optimizer, C00 reproduction, both routers, conditional OOF, TEST and inspect.
The production configuration is never written or changed by this test suite.
"""
import contextlib
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F
import yaml

from taxosieve import calibration, d05, pipeline, protocol, source
from taxosafe_support import calibration as base
from taxosafe_support import membership_calibration as membership
from taxosafe_support import protocol as support_protocol


def _digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _row(identity, split, status, leaf, meta):
    parent = meta["leaf_to_parent"][leaf]
    parent_logits, leaf_logits = [-1., -1.], [-1., -1., -1.]
    parent_logits[parent], leaf_logits[leaf] = 2., 2.
    return dict(image_sha256=_digest(identity), path=identity + ".png",
        split=split, status=status,
        source=(identity.split("-")[0] if status != "known" else meta["leaf_names"][leaf]),
        true_leaf=leaf if status == "known" else None,
        true_parent=None if status == "extra" else parent,
        log_probs=[-math.log(6.)] * 6, global_pred_leaf=leaf,
        support_evidence=dict(parent_logits=parent_logits, leaf_logits=leaf_logits,
            parent_membership_logits=[-2. if status == "extra" else 2.] * 2,
            leaf_membership_logits=[2. if status == "known" else -2.] * 3))


def _audit(groups):
    return {split: dict(count=len(group["records"]), unique_image_count=len(group["image_sha256"]),
        image_hashes=sorted(group["image_sha256"]),
        sources=sorted({r["source"] for r in group["records"]}),
        manifest_sha256=_digest("manifest:" + split)) for split, group in groups.items()}


def _fixture(config, binding):
    meta = dict(parent_names=["p", "q"], leaf_names=["a", "b", "c"], leaf_to_parent=[0, 0, 1])
    generator = torch.Generator().manual_seed(71)
    text = {prefix + "_" + level: F.normalize(torch.randn(count, 8, generator=generator), dim=1)
            for prefix in ("single", "ensemble") for level, count in (("leaf", 3), ("parent", 2))}
    provenance = dict(source_binding=copy.deepcopy(binding), preprocessing=copy.deepcopy(config["data"]),
        clip_core_sha256=_digest("fixture core"), templates_sha256=_digest("fixture templates"),
        clip_initialization="source_frozen_pretrained_core_without_prompts_or_adapters")

    def group(rows):
        features = F.normalize(torch.randn(len(rows), 8, generator=generator), dim=1)
        return dict(records=rows, image_sha256=[r["image_sha256"] for r in rows],
            record_feature_indices=list(range(len(rows))),
            features={key: features.clone() for key in d05.FEATURE_KEYS})

    caches = {}
    for stage in ("train", "development", "test"):
        if stage == "train":
            rows = [_row("train-" + str(i), "train", "known", i // 4, meta) for i in range(12)]
            groups = {"train": group(rows)}
        else:
            prefix = "val_" if stage == "development" else "test_"
            groups = {}
            for status in base.STATUSES:
                count = 6 if status == "known" else 4
                split = prefix + status
                rows = [_row(stage + status + str(i % 2) + "-" + str(i), split,
                             status, i % 3, meta) for i in range(count)]
                groups[split] = group(rows)
            if stage == "test":
                # One physical image, two manifest aliases: inference keeps the
                # rows, while metric denominators and evaluation weights dedup.
                duplicate = copy.deepcopy(groups["test_known"]["records"][0])
                duplicate["path"] = "second_alias.png"
                groups["test_known"]["records"].append(duplicate)
                groups["test_known"]["record_feature_indices"].append(0)
        caches[stage] = dict(meta=copy.deepcopy(meta), text=copy.deepcopy(text),
            provenance=copy.deepcopy(provenance), groups=groups, timings={})
    return caches


class TaxoSievePipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stack = contextlib.ExitStack()
        cls.addClassCleanup(cls.stack.close)
        cls.temp = Path(cls.stack.enter_context(tempfile.TemporaryDirectory(prefix="taxosieve_native_integration_")))
        cls.reference_dir = cls.temp / "external_reference"
        (cls.reference_dir / "calibration").mkdir(parents=True)
        cls.reference_bytes = cls.reference_dir / "frozen_reference_artifact.bin"
        cls.reference_bytes.write_bytes(b"synthetic external C00 artifact; no image model")
        cls.cfg = copy.deepcopy(protocol.DEFAULTS)
        cls.cfg["d05"].update(epochs=2, batch_size=32)
        cls.config_path = cls.temp / "fixture_recipe.yml"
        cls.config_path.write_text(yaml.safe_dump(cls.cfg))
        # Only the per-test temporary recipe and its expected constant change.
        # effective_config, initialize and inspect_run themselves remain real.
        cls.stack.enter_context(patch.object(protocol, "DEFAULTS", cls.cfg))
        cls.ref_config = support_protocol.effective_config(protocol.resolve(cls.cfg["reference_config"]), seed=1)
        binding = cls._binding()
        cls.caches = _fixture(cls.ref_config, binding)
        cls.audits = {stage: _audit(cache["groups"]) for stage, cache in cls.caches.items()}
        raw = {split: group["records"] for split, group in cls.caches["development"]["groups"].items()}
        cls.meta = cls.caches["train"]["meta"]
        cls.reference_router = membership.calibrate(*[raw["val_" + status] for status in base.STATUSES],
            cls.meta, cls.ref_config["calibration"])
        protocol.write_records(cls.reference_dir / "calibration/development_scores.jsonl",
                               [r for rows in raw.values() for r in rows])
        cls.stack.enter_context(patch.object(source, "inspect_reference", side_effect=cls._inspect_reference))
        cls.stack.enter_context(patch.object(source, "load_reference",
            side_effect=lambda directory, device: SimpleNamespace(**cls._inspect_reference(directory))))
        cls.stack.enter_context(patch.object(source, "load_training_rows",
            side_effect=lambda reference: (copy.deepcopy(cls.caches["train"]["groups"]["train"]["records"]),
                                            copy.deepcopy(cls.audits["train"]["train"]))))
        cls.stack.enter_context(patch.object(source, "stage_rows", side_effect=cls._stage_rows))
        cls.stack.enter_context(patch.object(pipeline, "_require_image_device", return_value=torch.device("cpu")))
        cls.collect = cls.stack.enter_context(patch.object(d05, "collect_cache", side_effect=cls._collect))
        old_threads = torch.get_num_threads()
        cls.stack.callback(torch.set_num_threads, old_threads)
        torch.set_num_threads(1)
        cls.complete = cls.temp / "complete"
        pipeline.train(cls.complete, cls.reference_dir, device="cpu", config=cls.config_path)
        pipeline.calibrate(cls.complete, save_scores=True)
        pipeline.test(cls.complete, save_scores=True)

    @classmethod
    def _binding(cls):
        return dict(directory=str(cls.reference_dir.resolve()),
                    checkpoint_sha256=protocol.file_hash(cls.reference_bytes))

    @classmethod
    def _inspect_reference(cls, directory):
        if Path(directory).resolve() != cls.reference_dir.resolve():
            raise AssertionError("Unexpected external source directory")
        return dict(directory=cls.reference_dir, config=copy.deepcopy(cls.ref_config),
            meta=copy.deepcopy(cls.meta), binding=cls._binding(), router=copy.deepcopy(cls.reference_router),
            audit={split: copy.deepcopy(audit) for group in cls.audits.values() for split, audit in group.items()})

    @classmethod
    def _stage_rows(cls, reference, stage):
        key = "development" if stage == "calibrate" else "test"
        return ({split: copy.deepcopy(group["records"]) for split, group in cls.caches[key]["groups"].items()},
                copy.deepcopy(cls.audits[key]))

    @classmethod
    def _collect(cls, reference, groups, device):
        stage = "train" if set(groups) == {"train"} else "development" if "val_known" in groups else "test"
        expected = {split: group["records"] for split, group in cls.caches[stage]["groups"].items()}
        if groups != expected or reference.binding != cls._binding():
            raise AssertionError("Image boundary received another split or source")
        return copy.deepcopy(cls.caches[stage])

    def setUp(self):
        self.case = Path(self.stack.enter_context(tempfile.TemporaryDirectory(prefix="case_", dir=self.temp)))

    def _copy_run(self, through="test"):
        directory = self.case / (through + "_run")
        directory.mkdir()
        for name in ("run.json", "source.json"):
            shutil.copyfile(self.complete / name, directory / name)
        stages = ["cache/train", "training", "cache/development", "calibration", "cache/test", "test"]
        for stage in stages[:stages.index(through) + 1]:
            shutil.copytree(self.complete / stage, directory / stage)
        return directory

    def _inspect(self, directory):
        output = io.StringIO()
        with patch.object(sys, "argv", ["run_taxosieve.py", "inspect", "--run-dir", str(directory)]), \
             contextlib.redirect_stdout(output):
            pipeline.main()
        return json.loads(output.getvalue())

    def test_native_lifecycle_has_real_optimizer_routers_folds_and_bound_artifacts(self):
        run = protocol.inspect_run(self.complete)
        self.assertEqual(run["mode"], "train")
        self.assertEqual(run["schema_version"], "taxosieve_v1")
        self.assertEqual(run["version"], "TaxoSieve_v1")
        self.assertEqual(run["config"], self.cfg)
        self.assertFalse(run["test_used_for_fitting"])
        stages = self._inspect(self.complete)
        self.assertEqual(set(stages), {"cache/train", "training", "cache/development", "calibration", "cache/test", "test"})
        training, calibrated, tested = (stages[key] for key in ("training", "calibration", "test"))
        self.assertGreater(training["optimizer_steps"], 0)
        self.assertEqual(training["optimizer_steps"], training["optimizer_steps_in_this_run"])
        self.assertEqual(training["gradient_splits"], ["train"])
        self.assertEqual(calibrated["model_sha256"], training["artifacts"]["model"]["sha256"])
        self.assertEqual(calibrated["training_receipt_sha256"], protocol.file_hash(self.complete / "training/completed.json"))
        self.assertEqual(tested["frozen_development_sha256"], protocol.file_hash(self.complete / "calibration/completed.json"))
        self.assertEqual(tested["test_cache_sha256"], stages["cache/test"]["artifacts"]["features"]["sha256"])
        for stage in stages.values():
            self.assertEqual(stage["source_sha256"], protocol.file_hash(self.complete / "source.json"))
            self.assertFalse(stage["test_used_for_fitting"])
            self.assertFalse(stage["unknown_images_used_for_gradients"])
        audit = protocol.read_json(self.complete / "calibration/audit.json")
        self.assertTrue(audit["reference_reproduction"]["decisions_match"])
        self.assertTrue(audit["reference_reproduction"]["raw_scores_match"])
        self.assertTrue(audit["crossfit"]["complete"])
        self.assertEqual(len(audit["crossfit"]["folds"]), 7)
        self.assertEqual(audit["crossfit"]["evaluated_image_count"], 14)
        for fold in audit["crossfit"]["folds"]:
            self.assertFalse(set(fold["fit_image_sha256"]) & set(fold["held_image_sha256"]))

    def test_external_reference_mismatch_leaves_no_new_run(self):
        directory = self.case / "incompatible_reference"
        info = self._inspect_reference(self.reference_dir)
        info["config"]["data"]["name"] = "another_dataset_version"
        with patch.object(source, "inspect_reference", return_value=info):
            with self.assertRaisesRegex(ValueError, "locked TaxoSieve reference recipe"):
                pipeline.train(directory, self.reference_dir, device="cpu", config=self.config_path)
        self.assertFalse(directory.exists())

    def test_invalid_external_reference_leaves_no_new_run(self):
        directory = self.case / "invalid_reference"
        with patch.object(source, "inspect_reference", side_effect=ValueError("Reference artifact hash mismatch")):
            with self.assertRaisesRegex(ValueError, "Reference artifact hash mismatch"):
                pipeline.train(directory, self.reference_dir, device="cpu", config=self.config_path)
        self.assertFalse(directory.exists())

    def test_imported_reference_mismatch_leaves_no_new_run(self):
        directory = self.case / "incompatible_d05"
        info = self._inspect_reference(self.reference_dir)
        info["config"]["data"]["name"] = "historical_dataset_version"
        packet = dict(reference_directory=str(self.reference_dir), meta=info["meta"],
                      reference_config=copy.deepcopy(info["config"]))
        with patch.object(pipeline, "_historical_export", return_value=packet), \
             patch.object(source, "inspect_reference", return_value=info):
            with self.assertRaisesRegex(ValueError, "Imported source uses another reference recipe"):
                pipeline.import_d05(self.case / "historical_d05", directory,
                                    device="cpu", config=self.config_path)
        self.assertFalse(directory.exists())

    def test_test_is_blocked_before_calibration_without_reading_test_images(self):
        directory = self._copy_run("training")
        before = self.collect.call_count
        with self.assertRaises(ValueError):
            pipeline.test(directory)
        self.assertEqual(self.collect.call_count, before)
        self.assertFalse((directory / "cache/test").exists())
        self.assertFalse((directory / "test").exists())

    def test_completed_entry_points_revalidate_model_bytes(self):
        directory = self._copy_run()
        model = directory / "training/model.pth"
        model.write_bytes(model.read_bytes() + b"changed")
        for operation in (pipeline.calibrate, pipeline.test, self._inspect):
            with self.subTest(operation=operation.__name__), self.assertRaisesRegex(ValueError, "digest mismatch"):
                operation(directory)

    def test_completed_entry_points_revalidate_training_cache_bytes(self):
        directory = self._copy_run()
        cache = directory / "cache/train/features.pth"
        cache.write_bytes(cache.read_bytes() + b"changed")
        for operation in (pipeline.calibrate, pipeline.test, self._inspect):
            with self.subTest(operation=operation.__name__), self.assertRaisesRegex(ValueError, "digest mismatch"):
                operation(directory)

    def test_router_byte_mutation_is_rejected_even_when_json_is_equivalent(self):
        directory = self._copy_run()
        router = directory / "calibration/router.json"
        router.write_bytes(router.read_bytes() + b"\n")
        for operation in (pipeline.calibrate, pipeline.test, self._inspect):
            with self.subTest(operation=operation.__name__), self.assertRaisesRegex(ValueError, "digest mismatch"):
                operation(directory)

    def test_source_byte_mutation_is_rejected_even_when_json_is_equivalent(self):
        directory = self._copy_run()
        binding = directory / "source.json"
        binding.write_bytes(binding.read_bytes() + b"\n")
        for operation in (pipeline.calibrate, pipeline.test, self._inspect):
            with self.subTest(operation=operation.__name__), self.assertRaisesRegex(ValueError, "source/run/permission"):
                operation(directory)

    def test_changed_external_reference_binding_blocks_completed_test(self):
        directory = self._copy_run()
        original = self.reference_bytes.read_bytes()
        try:
            self.reference_bytes.write_bytes(original + b"changed")
            with self.assertRaisesRegex(ValueError, "Reference artifacts"):
                pipeline.test(directory)
        finally:
            self.reference_bytes.write_bytes(original)

    def test_valid_but_different_dev_text_contract_is_rejected_before_router_fit(self):
        directory = self._copy_run("training")
        info, bound = protocol.inspect_source(directory)
        altered = copy.deepcopy(self.caches["development"])
        altered["text"]["ensemble_leaf"][0, 0] += .125
        pipeline._store_cache(directory, "development", altered, self.audits["development"], info, bound)
        with patch.object(calibration, "fit_router", wraps=calibration.fit_router) as fit:
            with self.assertRaisesRegex(ValueError, "TRAIN and DEV inference representations differ"):
                pipeline.calibrate(directory)
            fit.assert_not_called()
        self.assertFalse((directory / "calibration").exists())

    def test_valid_test_cache_must_bind_the_actual_frozen_development(self):
        directory = self._copy_run("calibration")
        info, bound = protocol.inspect_source(directory)
        pipeline._store_cache(directory, "test", copy.deepcopy(self.caches["test"]), self.audits["test"],
                              info, bound, frozen_development_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "TEST representation or development binding"):
            pipeline.test(directory)
        self.assertFalse((directory / "test").exists())

    def test_duplicate_aliases_have_unique_metrics_and_explicit_terminal_weights(self):
        rows = source.read_records(self.complete / "test/predictions.jsonl")
        result = protocol.verify_stage(self.complete, "test")["summary"]
        self.assertEqual(len(rows), 15)
        self.assertEqual(result["input_record_count"], 15)
        self.assertEqual(result["unique_image_count"], 14)
        self.assertEqual(result["duplicate_record_count"], 1)
        self.assertEqual(result["counts"]["known"], 6)
        self.assertEqual(sum(r["evaluation_weight"] for r in rows), 14)
        alias = next(r for r in rows if r["path"] == "second_alias.png")
        original = next(r for r in rows if r["image_sha256"] == alias["image_sha256"] and r is not alias)
        self.assertEqual((original["evaluation_weight"], alias["evaluation_weight"]), (1, 0))
        for key in pipeline.TERMINAL_FIELDS:
            self.assertEqual(alias[key], original[key])
        for row in rows:
            self.assertTrue({"selected_root_score", "selected_parent_score", "selected_leaf_score",
                             "root_threshold", "leaf_threshold", "root_pass"} <= set(row))
            self.assertNotIn("support_evidence", row)

    def test_completed_train_resume_does_not_refit_or_reextract(self):
        directory = self._copy_run()
        before = self.collect.call_count
        with patch.object(d05, "fit_payload", side_effect=AssertionError("Unexpected second fit")):
            receipt = pipeline.train(directory, self.reference_dir, device="cpu", config=self.config_path, resume=True)
        self.assertEqual(receipt, protocol.verify_stage(directory, "training"))
        self.assertEqual(before, self.collect.call_count)

    def test_partial_stage_is_reported_by_inspect_and_never_overwritten(self):
        directory = self._copy_run("calibration")
        (directory / "test").mkdir()
        with self.assertRaises(ValueError):
            self._inspect(directory)
        with self.assertRaises(ValueError):
            pipeline.test(directory)
        self.assertEqual(list((directory / "test").iterdir()), [])


if __name__ == "__main__":
    unittest.main()
