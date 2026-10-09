import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from tools.pack_taxosafe_parentrisk_review import pack


class ParentRiskPackTest(unittest.TestCase):
    def test_text_audit_includes_oof_excludes_models_and_symlinks(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            run = root / "run"
            (run / "parentrisk").mkdir(parents=True)
            (run / "calibration").mkdir()
            (run / "parentrisk/config.json").write_text(json.dumps({"geometry": {}, "calibration": {}}))
            (run / "calibration/oof_predictions.jsonl").write_text('{}\n')
            (run / "parentrisk/frozen_evidence.pth").write_bytes(b'model')
            (root / "secret.txt").write_text("external")
            (run / "link.txt").symlink_to(root / "secret.txt")
            (run / "linked").symlink_to(root, target_is_directory=True)
            target = root / "review.tar.gz"
            pack(run, target)
            with tarfile.open(target) as archive:
                self.assertEqual(set(archive.getnames()), {
                    "run/parentrisk/config.json", "run/calibration/oof_predictions.jsonl", "archive_manifest.json"})
                manifest = json.load(archive.extractfile("archive_manifest.json"))
                self.assertFalse(manifest["is_model_backup"])
                for item in manifest["files"]:
                    content = archive.extractfile(item["path"]).read()
                    self.assertEqual(len(content), item["bytes"])
                    self.assertEqual(hashlib.sha256(content).hexdigest(), item["sha256"])
            with self.assertRaises(ValueError):
                pack(run, target)
            alias = root / "alias"
            alias.symlink_to(run, target_is_directory=True)
            with self.assertRaises(ValueError):
                pack(alias, root / "invalid.tar.gz")


if __name__ == "__main__":
    unittest.main()
