"""Configuration, historical provenance, read-only CLI and review boundaries."""
import copy
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import inspect
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from taxosafe_discovery import protocol, runner
from taxosafe_refine import importer
from taxosafe_routealign import protocol as previous_protocol
from taxosafe_support import protocol as support_protocol
from tools.pack_taxosafe_discovery_review import pack


def snapshot(directory):
    return {str(p.relative_to(directory)): p.read_bytes()
            for p in Path(directory).rglob("*") if p.is_file() and not p.is_symlink()}


class DiscoveryProtocolContracts(unittest.TestCase):
    def setUp(self):
        self.cfg = protocol.effective_config(protocol.PROJECT_ROOT / protocol.DEFAULT_CONFIG)

    def test_declared_configuration_and_separate_source_code_signatures(self):
        self.assertEqual(self.cfg, protocol.DEFAULTS)
        self.assertEqual(len(self.cfg["arms"]), 11)
        self.assertEqual(self.cfg["arms"][0]["id"], "D00_reference")
        self.assertEqual(self.cfg["arms"][7]["weight_source"], "D06_episode_rank")
        self.assertEqual([a["candidate"] for a in self.cfg["arms"][:9]], ["reference"] * 9)
        first = protocol.signature(self.cfg, {"source": "one"})
        self.assertEqual(first["method"], protocol.SCHEMA_VERSION)
        changed = copy.deepcopy(self.cfg)
        changed["prompt"]["epochs"] += 1
        self.assertNotEqual(first, protocol.signature(changed, {"source": "one"}))
        self.assertNotEqual(first, protocol.signature(self.cfg, {"source": "two"}))
        with patch.object(protocol, "code_signature", return_value="changed code"):
            self.assertNotEqual(first, protocol.signature(self.cfg, {"source": "one"}))
        validated = protocol.validate_config(self.cfg)
        validated["verifier"]["epochs"] = 9
        self.assertNotEqual(validated, self.cfg)

    def test_arms_and_research_switches_cannot_silently_change(self):
        cases = []
        for mutation in (lambda cfg: cfg["arms"].pop(),
                         lambda cfg: cfg["arms"].reverse(),
                         lambda cfg: cfg["arms"].append(copy.deepcopy(cfg["arms"][0])),
                         lambda cfg: cfg["arms"][1].update(candidate="text"),
                         lambda cfg: cfg.update(test_tuning=True),
                         lambda cfg: cfg["calibration"].update(test_tuning=True),
                         lambda cfg: cfg["calibration"].update(skip_failed_gates=True),
                         lambda cfg: cfg["geometry"].pop("shrinkage")):
            cfg = copy.deepcopy(self.cfg)
            mutation(cfg)
            cases.append(cfg)
        for cfg in cases:
            with self.subTest(configuration=cfg), self.assertRaises(ValueError):
                protocol.validate_config(cfg)

    def test_invalid_numeric_budgets_and_seeds_fail_closed(self):
        cases = [("geometry", "shrinkage", 0), ("geometry", "shrinkage", 1.1),
                 ("geometry", "shrinkage", True), ("geometry", "shrinkage", float("nan")),
                 ("verifier", "folds", 1), ("verifier", "folds", 11),
                 ("verifier", "epochs", 1.5), ("verifier", "lr", -1),
                 ("projection", "epochs", False), ("projection", "temperature", float("inf")),
                 ("projection", "bottleneck", 0), ("prompt", "n_ctx", 17),
                 ("prompt", "lr", .011), ("prompt", "reference_weight", "3"),
                 ("calibration", "seed", 2), ("calibration", "seed", True),
                 ("calibration", "min_parent_known", 0)]
        for section, key, value in cases:
            cfg = copy.deepcopy(self.cfg)
            cfg[section][key] = value
            with self.subTest(section=section, key=key, value=value), self.assertRaises(ValueError):
                protocol.validate_config(cfg)
        for value in (True, -1, 2 ** 31, "1"):
            cfg = copy.deepcopy(self.cfg)
            cfg["seed"] = value
            with self.subTest(seed=value), self.assertRaises(ValueError):
                protocol.validate_config(cfg)

    def test_new_code_never_enters_historical_signature_or_importer_interfaces(self):
        old = {str(p.relative_to(protocol.PROJECT_ROOT)) for p in previous_protocol.code_files()}
        new = {str(p.relative_to(protocol.PROJECT_ROOT)) for p in protocol.code_files()}
        self.assertFalse(any(p.startswith("taxosafe_discovery/") for p in old))
        self.assertTrue(old < new)
        for name in ("protocol", "runner", "backend", "reporting", "features", "models", "geometry",
                     "verifier", "calibration", "projection_training", "prompt_training", "__main__"):
            self.assertIn("taxosafe_discovery/" + name + ".py", new)
        self.assertEqual(str(inspect.signature(importer.inspect_reference)), "(directory)")
        self.assertEqual(str(inspect.signature(importer.load_reference)), "(directory, device)")
        self.assertEqual(str(inspect.signature(support_protocol.signature)), "(cfg)")
        # The migration allowlist remains separate from strict old checks.
        from tests.test_taxosafe_refine_importer import approved_fixture_signature
        source = approved_fixture_signature()
        runtime = approved_fixture_signature(historic=False)
        with self.assertRaises(ValueError):
            support_protocol.require_signature(source, runtime)
        accepted = importer.verify_source_signature(source, runtime)
        self.assertEqual(accepted["source_signature"], source)
        self.assertEqual(accepted["runtime_signature"], runtime)
        with self.assertRaises(ValueError):
            importer.verify_source_signature(source, dict(runtime, code="0" * 64))


class DiscoveryCommandContracts(unittest.TestCase):
    def test_real_module_help_exposes_user_entrypoint_without_source(self):
        result = subprocess.run([sys.executable, "-m", "taxosafe_discovery", "--help"],
            cwd=protocol.PROJECT_ROOT, capture_output=True, text=True, timeout=60, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        for flag in ("--config", "--reference-run-dir", "--run-dir", "--suite-dir", "--device", "--resume", "--preflight"):
            self.assertIn(flag, result.stdout)

    def test_cli_preflight_reads_locked_audits_without_loading_models_or_writing_suite(self):
        from taxosafe_support import pipeline as support
        from taxosafe_refine import pipeline as refine
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, destination = root / "source", root / "fresh"
            source.mkdir()
            (source / "immutable.txt").write_text("source remains intact", encoding="utf-8")
            audits = {split: {"unique_image_count": count} for split, count in (("train", 8), ("val_known", 4))}
            info = {"binding": {"directory": str(source)}, "config": {}, "meta": {}, "training": {"audit": audits}}
            stages = []
            def stage_rows(reference, stage):
                stages.append(stage)
                prefix = "val_" if stage == "calibrate" else "test_"
                return {}, {prefix + status: {"unique_image_count": 4} for status in ("known", "intra", "extra")}
            args = ["taxosafe_discovery", "--reference-run-dir", str(source), "--run-dir", str(destination),
                    "--device", "cpu", "--preflight"]
            before, stream = snapshot(root), io.StringIO()
            with patch("sys.argv", args), patch.object(runner, "_source", return_value=info), \
                    patch.object(support, "load_stage_rows", return_value=({}, audits)), \
                    patch.object(refine, "_stage_rows", side_effect=stage_rows), \
                    patch.object(importer, "load_reference", side_effect=AssertionError("preflight cannot load tensors")), \
                    patch.object(support, "make_loader", side_effect=AssertionError("preflight cannot infer images")), \
                    redirect_stdout(stream):
                runner.main()
            result = json.loads(stream.getvalue())
            self.assertEqual(stages, ["calibrate", "test"])
            self.assertEqual(result["locked_split_counts"]["train"], 8)
            for key in ("checkpoint_tensors_loaded", "model_forward_performed", "test_predictions_read", "destination_created"):
                self.assertFalse(result[key])
            self.assertFalse(destination.exists())
            self.assertEqual(snapshot(root), before)

    def test_missing_explicit_reference_fails_before_any_inspection(self):
        with patch("sys.argv", ["taxosafe_discovery", "--run-dir", "unused"]), \
                patch.object(runner, "_source") as read, redirect_stderr(io.StringIO()), \
                self.assertRaises(SystemExit) as raised:
            runner.main()
        self.assertEqual(raised.exception.code, 2)
        read.assert_not_called()


class DiscoveryArchiveContracts(unittest.TestCase):
    def test_review_keeps_cache_receipts_all_failure_logs_and_hash_manifest_without_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            suite = root / "suite"
            suite.mkdir()
            protocol.write_json(suite / "snapshot.json", {"schema_version": protocol.SCHEMA_VERSION})
            expected = {"snapshot.json"}
            for stage in ("train", "development", "test"):
                cache = suite / "cache" / stage
                cache.mkdir(parents=True)
                for name in ("completed.json", "failure.json", "stage_binding.json"):
                    protocol.write_json(cache / name, {"stage": stage, "fixture": name})
                    expected.add("cache/" + stage + "/" + name)
                (cache / "features.pth").write_bytes(b"private cached tensors")
                (cache / "vectors.jsonl").write_text('{"private_feature_vector":[1,2,3]}\n', encoding="utf-8")
            for arm in protocol.ARMS:
                arm_id = arm["id"]
                protocol.write_json(suite / "arms" / arm_id / "failure.json", {"stage": "training", "error": "fixture"})
                expected.add("arms/" + arm_id + "/failure.json")
                for stage in ("training", "calibration", "test"):
                    logs = suite / "logs" / arm_id
                    logs.mkdir(parents=True, exist_ok=True)
                    for kind in ("stdout", "stderr"):
                        name = stage + "." + kind + ".log"
                        (logs / name).write_text("fixture log\n", encoding="utf-8")
                        expected.add("logs/" + arm_id + "/" + name)
            cache_logs = suite / "logs" / "cache"
            cache_logs.mkdir(parents=True)
            (cache_logs / "cache_train.stderr.log").write_text("cache failed\n", encoding="utf-8")
            expected.add("logs/cache/cache_train.stderr.log")
            (suite / "model.pth").write_bytes(b"private model")
            outside = root / "external.json"
            outside.write_text('{"private":true}', encoding="utf-8")
            (suite / "external_link.json").symlink_to(outside)
            before = snapshot(suite)
            with redirect_stdout(io.StringIO()):
                result = pack(suite, root / "review.tar.gz")
            with tarfile.open(result) as archive:
                names = set(archive.getnames())
                self.assertEqual(names, {"suite/" + name for name in expected} | {"archive_manifest.json"})
                manifest = json.load(archive.extractfile("archive_manifest.json"))
                self.assertFalse(manifest["includes_feature_caches"])
                self.assertFalse(manifest["includes_checkpoints"])
                self.assertEqual({row["path"] for row in manifest["files"]}, names - {"archive_manifest.json"})
                for item in manifest["files"]:
                    value = archive.extractfile(item["path"]).read()
                    self.assertEqual(item["sha256"], hashlib.sha256(value).hexdigest())
                    self.assertEqual(item["bytes"], len(value))
            self.assertEqual(snapshot(suite), before)

    def test_archive_cannot_overwrite_existing_file_or_live_suite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            suite = root / "suite"
            suite.mkdir()
            protocol.write_json(suite / "snapshot.json", {"schema_version": protocol.SCHEMA_VERSION})
            existing = root / "existing.tar.gz"
            existing.write_bytes(b"preserve")
            for destination in (existing, suite / "nested.tar.gz"):
                with self.subTest(destination=destination), self.assertRaises(ValueError):
                    pack(suite, destination)
            self.assertEqual(existing.read_bytes(), b"preserve")


if __name__ == "__main__":
    unittest.main()
