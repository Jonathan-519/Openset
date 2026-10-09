"""Domain CLI and failure-review export contracts."""
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
from unittest.mock import patch

from taxosafe_domain import protocol, runner
from tools.pack_taxosafe_domain_review import pack


class DomainCLIContracts(unittest.TestCase):
    def test_real_module_and_shell_help(self):
        for command in ([sys.executable, "-m", "taxosafe_domain", "--help"],
                        ["bash", "tools/run_taxosafe_domain.sh", "--help"]):
            environment = dict(__import__("os").environ)
            environment["PATH"] = str(Path(sys.executable).parent) + ":" + environment.get("PATH", "")
            result = subprocess.run(command, cwd=protocol.PROJECT_ROOT, text=True,
                                    capture_output=True, check=False, timeout=60, env=environment)
            self.assertEqual(result.returncode, 0, result.stderr)
            for flag in ("--discovery-run-dir", "--run-dir", "--preflight", "--resume", "--device"):
                self.assertIn(flag, result.stdout)

    def test_preflight_has_no_output_directory_or_forward(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_dir, destination = root / "discovery", root / "fresh_domain"
            source_dir.mkdir()
            source = dict(directory=source_dir, reference=dict(directory=root/"reference"),
                          binding=dict(directory=str(source_dir)), caches={
                              "train": {"audit": {"known_train": {"unique_image_count": 10}}}})
            arguments = ["domain", "--discovery-run-dir", str(source_dir), "--run-dir", str(destination), "--preflight"]
            output = io.StringIO()
            with patch.object(sys, "argv", arguments), patch.object(runner, "_source", return_value=source), redirect_stdout(output):
                runner.main()
            report = json.loads(output.getvalue())
            self.assertFalse(destination.exists())
            for key in ("checkpoint_tensors_loaded", "model_forward_performed", "test_predictions_read", "test_cache_opened", "destination_created"):
                self.assertFalse(report[key])

    def test_default_source_and_fresh_domain_destination(self):
        captured = {}
        def preflight(cfg, source, suite, device, resume):
            captured.update(source=source, suite=suite, device=device, resume=resume)
            return {"destination_created": False}
        with patch.object(sys, "argv", ["domain", "--preflight"]), patch.object(runner, "preflight", side_effect=preflight), redirect_stdout(io.StringIO()):
            runner.main()
        self.assertEqual(captured["source"], protocol.PROJECT_ROOT / "runs/taxosafe_new/discovery/trial_1_20261005_175231")
        self.assertEqual(captured["suite"].parent, protocol.PROJECT_ROOT / "runs/taxosafe_new/domain")
        self.assertTrue(captured["suite"].name.startswith("trial_1_"))
        self.assertFalse(captured["suite"].exists())

    def test_resume_cannot_silently_select_a_new_timestamp(self):
        with patch.object(sys, "argv", ["domain", "--resume"]), redirect_stdout(io.StringIO()), patch("sys.stderr", io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                runner.main()
        self.assertEqual(error.exception.code, 2)


class DomainReviewContracts(unittest.TestCase):
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
                self.assertEqual(manifest["schema_version"], "taxosafe_domain_review_v1")
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
