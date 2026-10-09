import contextlib
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from tools import install_taxosafe_dcbs_v11 as installer
from tools.pack_taxosafe_dcbs_review import pack


class DeliveryTest(unittest.TestCase):
    def test_manifest_rejects_escaping_and_non_code_payloads(self):
        for path in ("../secret.py", "/absolute.py", "a/../../escape.py", "runs/trial/config.json", "prepro/raw/a.jpg", "models/model.pth"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                installer.parse_manifest(("0" * 64 + "  " + path + "\n").encode())

    def test_install_stages_hashes_before_writes_and_preserves_backup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("train_taxosafe.py", "models/maple.py", "loader/taxosafe_data.py"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("baseline")
            (root / "new.py").write_bytes(b"old")
            (root / installer.MANIFEST).write_bytes(b"old manifest")
            payload = b"new code\n"
            manifest = (hashlib.sha256(payload).hexdigest() + "  new.py\n").encode()
            commit = "a" * 40
            with patch.object(installer, "download", side_effect=[manifest, b"corrupt"]):
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    installer.install(root, commit)
            self.assertEqual((root / "new.py").read_bytes(), b"old")
            self.assertFalse((root / "dcbs_v11_backups").exists())
            with patch.object(installer, "download", side_effect=[manifest, payload]), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(installer.install(root, commit), commit)
            self.assertEqual((root / "new.py").read_bytes(), payload)
            backup = next((root / "dcbs_v11_backups").iterdir())
            self.assertEqual((backup / "new.py").read_bytes(), b"old")
            self.assertEqual((backup / installer.MANIFEST).read_bytes(), b"old manifest")
            self.assertEqual((root / "train_taxosafe.py").read_text(), "baseline")

    def test_review_pack_excludes_weights_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run"
            (run / "training").mkdir(parents=True)
            manifest = root / "test.txt"
            manifest.write_text("some/image.jpg,0,0\n")
            (run / "training/config.json").write_text(json.dumps({"data": {"test_known": str(manifest)}}))
            (run / "training/train.jsonl").write_text('{"epoch": 1}\n')
            (run / "training/best.pth").write_bytes(b"large weights")
            archive = root / "review.tar.gz"
            with contextlib.redirect_stdout(io.StringIO()):
                pack(run, archive)
            with tarfile.open(archive) as result:
                names = result.getnames()
            self.assertIn("run/training/train.jsonl", names)
            self.assertIn("manifests/test_known.txt", names)
            self.assertFalse(any(n.endswith(".pth") for n in names))
            with self.assertRaisesRegex(ValueError, "already exists"):
                pack(run, archive)


if __name__ == "__main__":
    unittest.main()
