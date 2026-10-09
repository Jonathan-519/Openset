"""Real eight-arm CPU lifecycle, including actual source-held-out refits.

Only images/backbone and subprocess launch use the existing tiny C00 fixture.
The production cache, optimizer, calibration, audits, TEST, and resume paths run.
These are functional contracts, never estimates of the user's Trial 1 accuracy.
"""
import copy
from contextlib import ExitStack, redirect_stdout
import hashlib
import json
from pathlib import Path
import traceback
import unittest
from unittest.mock import patch

import torch

from tests import test_taxosafe_refine_pipeline as fixture
from tests.test_taxosafe_boundary_legacy_torch import legacy_torch_apis
from taxosafe_support import protocol as base_protocol
from taxosafe_support import calibration as base
from taxosafe_dcbs.protocol import normalized_name
from taxosafe_evidence_guard import backend, calibration, features, geometry, protocol, reporting, runner, training


class EvidenceGuardLifecycle(fixture.FrozenPipelineContracts):
    def configure_guard(self, steps=3):
        # Repeated feature vectors with independently audited image identities
        # intentionally prevent perfect quality. One far vector exercises ROOT.
        for prefix in ("val", "test"):
            known = self.groups[prefix + "_known"]
            for index, row in enumerate(self.groups[prefix + "_intra"]):
                row["vector"] = list(known[index % len(known)]["vector"])
            self.groups[prefix + "_extra"][0]["vector"] = list(known[0]["vector"])
        self.make_source(with_test=True)
        cfg = copy.deepcopy(protocol.DEFAULTS)
        cfg["training"].update(steps=steps, batch_size=8, adapter_dim=4, log_every=1)
        cfg["geometry"].update(neighbors=1)
        self.unknown_train = {}
        for split, dev_split in (("train_intra", "val_intra"), ("oe_train", "val_extra")):
            selected = {}
            for row in self.groups[dev_split]:
                selected.setdefault(row["source"], row)
            rows = []
            for source, template in sorted(selected.items()):
                for repeat in range(2):
                    identity = split + "/" + source + "/" + str(repeat)
                    row = copy.deepcopy(template)
                    row.update(path=identity, resolved_path="/fixture/" + identity,
                               image_sha256=hashlib.sha256(identity.encode()).hexdigest(),
                               dataset_index=len(rows), manifest_index=len(rows), split=split)
                    rows.append(row)
            self.unknown_train[split] = rows
            manifest = self.root / (split + ".txt")
            manifest.write_text("fixture_" + split, encoding="utf-8")
            cfg["data"][split] = str(manifest)
        return cfg

    def fixture_source_rows(self, info, stage, cfg=None):
        if stage != "train":
            return self.original_source_rows(info, stage, cfg)
        from types import SimpleNamespace
        rows, audit = features.load_training_rows(SimpleNamespace(**info))
        added = copy.deepcopy(self.unknown_train)
        forbidden = {h for value in info["audit"].values() for h in value["image_hashes"]}
        sources = {source for name in ("test_intra", "test_extra") for source in info["audit"][name]["sources"]}
        extra = base_protocol.audit_rows(added, forbidden_hashes=forbidden, forbidden_sources=sources)
        for name, value in extra.items():
            value.update(manifest_sha256=base_protocol.file_hash(cfg["data"][name]),
                         expanded_training_data=True, real_unknown_images=True,
                         full_C00_support_retained=True, development_source_overlap=value["sources"],
                         test_source_overlap=[], test_isolation={"source": "validated_C00_TEST_receipt",
                             "test_image_hashes_read": False, "test_features_extracted": False})
        return dict(train=rows, **added), dict(train=audit, **extra)

    def patches(self, suite, calls, injected_arm=None):
        stack = ExitStack()
        self.original_source_rows = features.source_rows
        stack.enter_context(patch.object(features, "source_rows", side_effect=self.fixture_source_rows))
        stack.enter_context(patch.object(features, "audit_images", return_value=dict(valid=True, problems=[], image_count=0)))

        def launch(directory, arm, stage, device):
            calls.append((arm, stage))
            logs = directory / "logs" / (arm or "cache")
            logs.mkdir(parents=True, exist_ok=True)
            out, err = logs / (stage + ".stdout.log"), logs / (stage + ".stderr.log")
            try:
                with out.open("w") as handle, redirect_stdout(handle), ExitStack() as guards:
                    if injected_arm == arm and stage == "training":
                        raise RuntimeError("injected EvidenceGuard training failure")
                    if stage in ("cache_test", "test"):
                        self.assertTrue((suite / "dev_selection.json").is_file())
                        for target in ("taxosafe_evidence_guard.training.fit", "taxosafe_evidence_guard.geometry.fit",
                                       "taxosafe_evidence_guard.calibration.fit", "taxosafe_evidence_guard.calibration.source_crossfit"):
                            guards.enter_context(patch(target, side_effect=AssertionError("TEST fitting forbidden: " + target)))
                    runner.worker(directory, arm, stage, device)
                err.write_text("")
                return 0, out, err
            except Exception:
                err.write_text(traceback.format_exc())
                return 2, out, err
        stack.enter_context(patch.object(runner, "_launch_stage", side_effect=launch))
        stack.enter_context(legacy_torch_apis(isin_mode="missing"))
        return stack

    def assert_no_failures(self, result, suite):
        if result["technical_failures"]:
            self.fail(json.dumps(result["technical_failures"], indent=2) + "\n" +
                      "\n".join(str(path) + "\n" + path.read_text() for path in suite.rglob("failure.json")))

    def test_all_eight_arms_and_actual_source_refits_failed_quality_still_TEST(self):
        cfg = self.configure_guard()
        original_bytes = fixture.artifact_snapshot(self.source)
        suite = self.root / "guard_suite"
        calls, fit_calls = [], []
        original_fit = training.fit

        def record_fit(*args, **kwargs):
            payload, report = original_fit(*args, **kwargs)
            fit_calls.append(dict(report=copy.deepcopy(report), kwargs=copy.deepcopy(kwargs)))
            return payload, report

        with self.patches(suite, calls), patch.object(training, "fit", side_effect=record_fit):
            preflight = runner.preflight(cfg, self.source, suite)
            self.assertFalse(preflight["test_cache_opened"])
            self.assertFalse(suite.exists())
            result = runner.execute_suite(cfg, self.source, suite)
            self.assert_no_failures(result, suite)
            self.assertEqual(result["completed_calibration_count"], 8)
            self.assertEqual(result["completed_test_count"], 8)
            self.assertTrue(all(not item["targets_passed"] for item in result["all_arms"]))
            self.assertEqual(result["recommendation_arm_id"], "R00_reference")
            first_test = calls.index((None, "cache_test"))
            self.assertEqual(sum(stage == "calibration" for _, stage in calls[:first_test]), 8)
            self.assertGreater(len(fit_calls), 6, "OOF must perform genuine additional adapter optimization")

            source_train_hashes = {row["image_sha256"] for rows in self.unknown_train.values() for row in rows}
            for arm in cfg["arms"]:
                directory = suite / "arms" / arm["id"]
                trained = protocol.read_json(directory / "training/training_report.json")
                inherited = arm["id"] in ("R00_reference", "R07_root_guard")
                self.assertEqual(trained["optimizer_steps"], 0 if inherited else 3)
                self.assertFalse(trained["frozen_encoder_updated"])
                if not inherited:
                    self.assertGreater(trained["parameter_delta_l2"], 0.)
                tested = protocol.read_json(directory / "test/completed.json")
                self.assertTrue(tested["test_allowed_after_failed_gates"])
                self.assertEqual(tested["summary"]["unique_image_count"], 10)
                self.assertEqual(tested["summary"]["input_record_count"], 11)

            for stage in ("calibration", "test"):
                filename = "development_predictions.jsonl" if stage == "calibration" else "predictions.jsonl"
                expected = fixture.read_records(self.source / stage / filename)
                actual = fixture.read_records(suite / "arms/R00_reference" / stage / "predictions.jsonl")
                fields = ("image_sha256", "prediction_type", "parent", "leaf", "output_node")
                self.assertEqual([{key: row[key] for key in fields} for row in expected],
                                 [{key: row[key] for key in fields} for row in actual])

            # Explicit source-holdout receipts, not merely held DEV images.
            crossfit = protocol.read_json(suite / "arms/R04_OE_both/calibration/crossfit_audit.json")
            self.assertTrue(crossfit["complete"])
            self.assert_source_exclusion(crossfit, source_train_hashes)
            self.assertEqual(len(fit_calls), 6 * (1 + len(crossfit["folds"])))
            guard_oof = protocol.read_json(suite / "arms/R07_root_guard/calibration/crossfit_audit.json")
            self.assertTrue(guard_oof["complete"])
            self.assertEqual([fold["model_state_sha256"] for fold in crossfit["folds"]],
                             [fold["model_state_sha256"] for fold in guard_oof["folds"]])

            info = runner._source(self.source)
            cached, _ = backend._load_cache(suite, "test", cfg, info)
            both = next(arm for arm in cfg["arms"] if arm["id"] == "R04_OE_both")
            guard = next(arm for arm in cfg["arms"] if arm["id"] == "R07_root_guard")
            payload, _ = backend._load_model(suite, both, cfg, info)
            reused, _ = backend._load_model(suite, guard, cfg, info)
            self.assert_model_equal(payload, reused)
            changed = copy.deepcopy(cached)
            for group in changed["groups"].values():
                for row in group["records"]:
                    row.update(status="extra", true_leaf=None, true_parent=None, source="never_seen_label")
            before, after = training.score(cached, payload), training.score(changed, payload)
            for split in before:
                self.assertEqual([row["support_evidence"] for row in before[split]],
                                 [row["support_evidence"] for row in after[split]])

            baseline = base.unique_records(fixture.read_records(suite / "arms/R00_reference/test/predictions.jsonl"))
            guarded = {row["image_sha256"]: row for row in fixture.read_records(suite / "arms/R07_root_guard/test/predictions.jsonl")}
            baseline_roots = [row for row in baseline if row["prediction_type"] == "global_unknown"]
            self.assertTrue(baseline_roots, "ROOT preservation assertion must not be vacuous")
            for row in baseline_roots:
                self.assertEqual(guarded[row["image_sha256"]]["prediction_type"], "global_unknown")

            calls_before, fits_before = list(calls), len(fit_calls)
            runner.execute_suite(cfg, self.source, suite, resume=True)
            self.assertEqual(calls, calls_before)
            self.assertEqual(len(fit_calls), fits_before)
            frozen = suite / "dev_selection.json"
            frozen_bytes = frozen.read_bytes()
            frozen.unlink()
            with self.assertRaises(ValueError):
                runner._resume_audit(suite, cfg, runner._source(self.source))
            frozen.write_bytes(frozen_bytes)
            router = suite / "arms/R04_OE_both/calibration/router.json"
            router_bytes = router.read_bytes()
            router.write_bytes(router_bytes + b" ")
            with self.assertRaises(ValueError):
                reporting.freeze_dev_selection(suite)
            router.write_bytes(router_bytes)
        self.assertEqual(fixture.artifact_snapshot(self.source), original_bytes)

    def test_one_optimizer_failure_does_not_block_other_seven_tests(self):
        cfg = self.configure_guard(steps=2)
        original_bytes = fixture.artifact_snapshot(self.source)
        suite, calls = self.root / "one_failed_arm", []
        with self.patches(suite, calls, injected_arm="R02_OE_fine"):
            result = runner.execute_suite(cfg, self.source, suite)
        self.assertEqual(set(result["technical_failures"]), {"R02_OE_fine"})
        self.assertEqual(result["completed_test_count"], 7)
        self.assertNotIn(("R02_OE_fine", "test"), calls)
        failed = [row for row in result["all_arms"] if row["arm_id"] == "R02_OE_fine"]
        self.assertTrue(failed)
        self.assertTrue(all(row["execution"] == "unavailable" and
                            row["known_end_to_end_leaf_accuracy"] is None for row in failed))
        self.assertEqual(fixture.artifact_snapshot(self.source), original_bytes)

    def assert_model_equal(self, left, right):
        # The model payload key is part of the public training cache contract.
        self.assertEqual(set(left["state"]), set(right["state"]))
        for name in left["state"]:
            self.assertTrue(torch.equal(left["state"][name], right["state"][name]), name)

    def assert_source_exclusion(self, audit, unknown_train_hashes):
        folds = audit["folds"]
        found = 0
        for fold in folds:
            if fold.get("kind") != "unknown_source":
                continue
            held = fold["source"]
            found += 1
            held_hashes = {row["image_sha256"] for rows in self.unknown_train.values()
                           for row in rows if normalized_name(row["source"]) == normalized_name(held)}
            self.assertTrue(held_hashes)
            trained = fold["training_report"]
            self.assertFalse(set(trained["fit_image_sha256"]) & held_hashes)
            self.assertFalse(set(trained["used_image_sha256"]) & held_hashes)
            self.assertEqual(set(fold["excluded_train_image_sha256"]), held_hashes)
            self.assertNotIn(normalized_name(held), {normalized_name(source) for source in trained["fit_unknown_source_counts"]})
            self.assertEqual(trained["optimizer_steps"], 3)
            self.assertEqual(trained["training_seed"], fold["seed"])
            centered = fold["router"]["reference_router"]
            self.assertEqual(trained["loss_reference_parent_threshold"], centered["parent_threshold"])
            self.assertEqual(trained["loss_reference_leaf_threshold"], centered["leaf_threshold"])
            self.assertFalse(fold["full_data_adapter_used"])
            self.assertEqual(len(fold["model_state_sha256"]), 64)
            self.assertTrue(set(trained["fit_image_sha256"]) & unknown_train_hashes)
        self.assertEqual(found, len({row["source"] for rows in self.unknown_train.values() for row in rows}))


for name in dir(fixture.FrozenPipelineContracts):
    if name.startswith("test_") and name not in EvidenceGuardLifecycle.__dict__:
        setattr(EvidenceGuardLifecycle, name, None)


if __name__ == "__main__":
    unittest.main()
