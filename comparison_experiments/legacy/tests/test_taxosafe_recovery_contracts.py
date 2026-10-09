"""Recovery configuration, historical provenance and export boundaries."""
import copy
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest

from taxosafe_recovery import protocol
from taxosafe_discovery import protocol as previous
from tools.pack_taxosafe_recovery_review import pack


class RecoveryProtocolContracts(unittest.TestCase):
    def test_config_and_four_independent_finetunes_are_predeclared(self):
        cfg = protocol.effective_config(protocol.PROJECT_ROOT / protocol.DEFAULT_CONFIG)
        self.assertEqual(cfg, protocol.DEFAULTS)
        self.assertEqual([a["id"] for a in cfg["arms"][:2]], ["F00_reference", "F01_d05"])
        self.assertEqual([a["mode"] for a in cfg["arms"] if a["kind"] == "finetune"],
                         ["bce", "hard", "anchor", "l2sp"])
        self.assertTrue(all("weight_source" not in a for a in cfg["arms"] if a["kind"] == "finetune"))
        self.assertEqual(cfg["arms"][2]["weight_source"], "F01_d05")
        self.assertEqual(cfg["arms"][7]["weight_source"], "F06_l2sp")
        self.assertEqual(cfg["arms"][8]["policy"], "fixed_parent")
        result = protocol.validate_config(cfg)
        result["training"]["lr"] = .0002
        self.assertNotEqual(result, cfg)

    def test_source_config_and_runtime_code_are_all_bound(self):
        cfg, binding = copy.deepcopy(protocol.DEFAULTS), {"reference": "one", "d05": "two"}
        sig = protocol.signature(cfg, binding)
        cfg["training"]["steps_per_head"] += 1
        self.assertNotEqual(sig, protocol.signature(cfg, binding))
        self.assertNotEqual(sig, protocol.signature(protocol.DEFAULTS, dict(binding, d05="three")))
        old = set(previous.code_files())
        current = set(protocol.code_files())
        self.assertTrue(old < current)
        self.assertFalse(any("taxosafe_recovery" in str(p) for p in old))
        self.assertIn(protocol.PROJECT_ROOT / "taxosafe_recovery/protocol.py", current)

    def test_changed_controls_and_test_selection_flags_are_rejected(self):
        for mutation in (
            lambda c: c["arms"].pop(), lambda c: c["arms"].reverse(),
            lambda c: c["arms"][3].update(weight_source="F04_hard_positive"),
            lambda c: c["arms"][8].update(policy="standard"),
            lambda c: c.update(skip_failed_gates=True),
            lambda c: c["training"].update(use_test_errors=True),
            lambda c: c["calibration"].update(known_target=.94),
        ):
            cfg = copy.deepcopy(protocol.DEFAULTS)
            mutation(cfg)
            with self.assertRaises(ValueError):
                protocol.validate_config(cfg)

    def test_invalid_budget_and_numeric_values_are_rejected(self):
        for section, key, value in (
            ("training", "steps_per_head", 0), ("training", "steps_per_head", 1.5),
            ("training", "batch_size", True), ("training", "lr", .01),
            ("training", "lr", 0), ("training", "hard_weight", float("nan")),
            ("training", "l2sp_weight", float("inf")),
            ("training", "hardness_eta", "2"), ("calibration", "seed", 2),
        ):
            cfg = copy.deepcopy(protocol.DEFAULTS)
            cfg[section][key] = value
            with self.subTest(section=section, key=key, value=value), self.assertRaises(ValueError):
                protocol.validate_config(cfg)

    def test_real_help_names_the_saved_d05_suite(self):
        result = subprocess.run([sys.executable, "-m", "taxosafe_recovery", "--help"],
            cwd=protocol.PROJECT_ROOT, text=True, capture_output=True, check=False, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        for flag in ("--discovery-run-dir", "--run-dir", "--preflight", "--resume", "--device"):
            self.assertIn(flag, result.stdout)


class RecoveryReviewContracts(unittest.TestCase):
    def test_failures_and_diagnostics_survive_while_models_and_vectors_do_not(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            suite = root / "suite"
            protocol.write_json(suite / "snapshot.json", {"schema_version": protocol.SCHEMA_VERSION})
            expected = {"snapshot.json"}
            for arm in protocol.ARMS:
                relative = "arms/" + arm["id"] + "/failure.json"
                protocol.write_json(suite / relative, {"technical_failure": True, "error": "test fixture"})
                expected.add(relative)
            for stage in ("train", "development", "test"):
                for name in ("completed.json", "stage_binding.json", "failure.json"):
                    relative = "cache/" + stage + "/" + name
                    protocol.write_json(suite / relative, {"fixture": True})
                    expected.add(relative)
                (suite / "cache" / stage / "features.pth").write_bytes(b"excluded tensor cache")
                (suite / "cache" / stage / "vectors.json").write_text("[1, 2, 3]", encoding="utf-8")
            relative = "logs/cache/cache_test.stderr.log"
            (suite / relative).parent.mkdir(parents=True)
            (suite / relative).write_text("test cache failure retained", encoding="utf-8")
            expected.add(relative)
            (suite / "model.pth").write_bytes(b"excluded model")
            (root / "external.json").write_text("{}", encoding="utf-8")
            (suite / "external.json").symlink_to(root / "external.json")
            with redirect_stdout(io.StringIO()):
                output = pack(suite, root / "review.tar.gz")
            with tarfile.open(output) as tar:
                names = set(tar.getnames())
                self.assertEqual(names, {"suite/" + p for p in expected} | {"archive_manifest.json"})
                manifest = json.load(tar.extractfile("archive_manifest.json"))
                self.assertEqual(manifest["schema_version"], "taxosafe_recovery_review_v1")
                for item in manifest["files"]:
                    content = tar.extractfile(item["path"]).read()
                    self.assertEqual(item["sha256"], hashlib.sha256(content).hexdigest())
                    self.assertEqual(item["bytes"], len(content))

    def test_export_refuses_to_overwrite_or_write_inside_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            suite = root / "suite"
            protocol.write_json(suite / "snapshot.json", {"schema_version": protocol.SCHEMA_VERSION})
            existing = root / "existing.tar.gz"
            existing.write_bytes(b"keep")
            for target in (existing, suite / "nested.tar.gz"):
                with self.assertRaises(ValueError):
                    pack(suite, target)
            self.assertEqual(existing.read_bytes(), b"keep")


if __name__ == "__main__":
    unittest.main()
