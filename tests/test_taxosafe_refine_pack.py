"""The existing packer also exports refinement text without following sources."""
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from tools.pack_taxosafe_new_review import pack


class RefinementReviewPackTests(unittest.TestCase):
    def test_source_bindings_are_text_not_instructions_to_copy_models(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            run = root / "refine"
            stage = run / "refinement"
            stage.mkdir(parents=True)
            config = {"features": "fine", "reconstruction": {"rank": 16}}
            (stage / "config.json").write_text(json.dumps(config))
            (stage / "source.json").write_text(json.dumps({"directory": str(root / "source")}))
            (stage / "model.pth").write_bytes(b"weights must stay local")
            source = root / "source"
            source.mkdir()
            (source / "private.txt").write_text("not in selected run")
            (stage / "outside.txt").symlink_to(source / "private.txt")
            target = root / "review.tar.gz"
            pack(run, target)
            with tarfile.open(target) as archive:
                names = archive.getnames()
                self.assertNotIn("run/refinement/model.pth", names)
                self.assertNotIn("run/refinement/outside.txt", names)
                manifest = json.loads(archive.extractfile("archive_manifest.json").read())
                self.assertEqual(manifest["schema"], "taxosafe_refine_review_v1")
                for item in manifest["files"]:
                    payload = archive.extractfile(item["path"]).read()
                    self.assertEqual(item["sha256"], hashlib.sha256(payload).hexdigest())
                    self.assertEqual(item["bytes"], len(payload))
            with self.assertRaises(ValueError):
                pack(run, target)


if __name__ == "__main__":
    unittest.main()
