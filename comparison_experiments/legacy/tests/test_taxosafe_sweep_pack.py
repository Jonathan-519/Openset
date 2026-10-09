import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest

from tools.pack_taxosafe_sweep_review import pack


class SweepPackTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.suite = self.root / "suite"
        self.suite.mkdir()
        self.write("snapshot.json", json.dumps({"schema_version": "taxosafe_sweep_v1"}))

    def write(self, relative, payload):
        path = self.suite / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload if isinstance(payload, bytes) else payload.encode("utf-8"))
        return path

    def test_all_arms_failures_and_manifest_are_included_without_mutating_source(self):
        for index, arm in enumerate(("reference", "parent_ce", "parent_bce",
                                     "leaf_margin", "leaf_reconstruction", "hierarchy_blend")):
            base = "arms/E%02d_%s/" % (index, arm)
            self.write(base + "training/stdout.log", "training\n")
            self.write(base + "calibration/metrics.json", '{"accuracy": 0.9}\n')
            self.write(base + "test/predictions.jsonl", '{"node": 3}\n')
            self.write(base + "training/failure.stderr", "Traceback: simulated failure\n")
        for name in ("config.json", "source_binding.json", "dev_selection.json", "summary.json"):
            self.write(name, "{}\n")
        self.write("runner.stdout", "running\n")
        self.write("stderr", "interrupted\n")
        before = {p.relative_to(self.suite): (p.read_bytes(), p.stat().st_mtime_ns)
                  for p in self.suite.rglob("*") if p.is_file()}
        output = pack(self.suite)
        self.assertEqual(output.parent, self.suite.parent)
        self.assertRegex(output.name, r"^suite_review_\d{8}T\d{12}Z\.tar\.gz$")
        with tarfile.open(output) as archive:
            manifest = json.load(archive.extractfile("archive_manifest.json"))
            names = set(archive.getnames())
            self.assertEqual(names, {"suite/" + p.as_posix() for p in before} | {"archive_manifest.json"})
            self.assertFalse(manifest["is_model_backup"])
            for item in manifest["files"]:
                payload = archive.extractfile(item["path"]).read()
                self.assertEqual(item["bytes"], len(payload))
                self.assertEqual(item["sha256"], hashlib.sha256(payload).hexdigest())
            self.assertTrue(all(member.isfile() for member in archive.getmembers()))
        after = {p.relative_to(self.suite): (p.read_bytes(), p.stat().st_mtime_ns)
                 for p in self.suite.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_incomplete_suite_requires_only_valid_snapshot(self):
        self.write("arms/E00_reference/training/stderr", "CUDA failure\n")
        self.write("arms/E01_parent_ce/failed.json", '{"state":"failed"}\n')
        output = pack(self.suite, self.root / "incomplete.tar.gz")
        with tarfile.open(output) as archive:
            self.assertIn("suite/arms/E00_reference/training/stderr", archive.getnames())
            self.assertIn("suite/arms/E01_parent_ce/failed.json", archive.getnames())
            self.assertNotIn("suite/dev_selection.json", archive.getnames())

    def test_images_models_caches_and_binary_disguised_as_text_are_excluded(self):
        for name in ("input.png", "model.pt", "features.npy", "features.npz",
                     "images/labels.json", "checkpoints/config.json", "cache/log.txt",
                     "feature_caches/metadata.json", "training_features_cache/state.json"):
            self.write(name, b"excluded")
        self.write("binary.log", b"bad\0binary")
        self.write("binary.json", b"\xff\xfe")
        self.write("arms/E00_reference/training/stderr.log", "failure\n")
        output = pack(self.suite, self.root / "review.tar.gz")
        with tarfile.open(output) as archive:
            self.assertEqual(set(archive.getnames()), {
                "suite/snapshot.json", "suite/arms/E00_reference/training/stderr.log",
                "archive_manifest.json",
            })
            manifest = json.load(archive.extractfile("archive_manifest.json"))
            self.assertEqual(set(manifest["skipped_non_text_files"]),
                             {"suite/binary.log", "suite/binary.json"})

    def test_links_and_special_files_are_not_followed(self):
        external = self.root / "external"
        external.mkdir()
        (external / "secret.txt").write_text("private", encoding="utf-8")
        (self.suite / "linked_directory").symlink_to(external, target_is_directory=True)
        (self.suite / "linked.txt").symlink_to(external / "secret.txt")
        (self.suite / "loop").symlink_to(self.suite, target_is_directory=True)
        os.mkfifo(self.suite / "fifo.log")
        output = pack(self.suite, self.root / "review.tar.gz")
        with tarfile.open(output) as archive:
            self.assertEqual(set(archive.getnames()), {"suite/snapshot.json", "archive_manifest.json"})

    def test_source_and_source_ancestor_links_are_rejected(self):
        alias = self.root / "alias"
        alias.symlink_to(self.suite, target_is_directory=True)
        parent_alias = self.root / "parent_alias"
        parent_alias.symlink_to(self.root, target_is_directory=True)
        for source in (alias, parent_alias / "suite"):
            with self.subTest(source=source), self.assertRaises(ValueError):
                pack(source, self.root / "review.tar.gz")
        self.assertFalse((self.root / "review.tar.gz").exists())

    def test_snapshot_symlink_and_wrong_or_missing_schema_are_rejected(self):
        snapshot = self.suite / "snapshot.json"
        for payload in ("{}", "[]", "null", "not JSON", '{"schema":"taxosafe_sweep_v1"}',
                        '{"schema_version":"other"}'):
            snapshot.write_text(payload, encoding="utf-8")
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                pack(self.suite, self.root / "review.tar.gz")
        snapshot.unlink()
        with self.assertRaises(ValueError):
            pack(self.suite, self.root / "review.tar.gz")
        external = self.root / "external.json"
        external.write_text('{"schema_version":"taxosafe_sweep_v1"}', encoding="utf-8")
        snapshot.symlink_to(external)
        with self.assertRaises(ValueError):
            pack(self.suite, self.root / "review.tar.gz")
        self.assertFalse((self.root / "review.tar.gz").exists())

    def test_destination_must_be_new_outside_suite_and_without_ancestor_links(self):
        existing = self.root / "existing.tar.gz"
        existing.write_bytes(b"original")
        linked = self.root / "linked.tar.gz"
        linked.symlink_to(existing)
        alias = self.root / "alias"
        alias.symlink_to(self.root, target_is_directory=True)
        for output in (existing, linked, alias / "new.tar.gz", self.suite / "nested/review.tar.gz"):
            with self.subTest(output=output), self.assertRaises(ValueError):
                pack(self.suite, output)
        self.assertEqual(existing.read_bytes(), b"original")
        self.assertFalse((self.root / "new.tar.gz").exists())
        self.assertFalse((self.suite / "nested").exists())
        output = pack(self.suite, self.root / "new_parent/review.tar.gz")
        self.assertTrue(output.is_file())

    def test_cli_requires_explicit_suite_directory(self):
        script = Path(__file__).resolve().parents[1] / "tools/pack_taxosafe_sweep_review.py"
        result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--suite-dir", result.stderr)
        output = self.root / "cli.tar.gz"
        result = subprocess.run([sys.executable, str(script), "--suite-dir", str(self.suite),
                                 "--output", str(output)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(output.is_file())


if __name__ == "__main__":
    unittest.main()
