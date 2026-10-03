"""Geometry diagnostics remain text-only and preserve existing pack formats."""
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from tools.pack_taxosafe_new_review import pack


class GeometryPackTests(unittest.TestCase):
    def test_geometry_diagnostics_do_not_follow_source_or_include_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            run = root / "geometry_run"
            stage = run / "geometry"
            stage.mkdir(parents=True)
            (stage / "config.json").write_text(json.dumps({
                "geometry": {"shrinkage": 0.1, "ridge": 0.0001},
                "calibration": {"source_loo": True}}))
            (stage / "source_binding.json").write_text(json.dumps({"directory": str(root / "source")}))
            (stage / "features.pth").write_bytes(b"private feature cache")
            source = root / "source"
            source.mkdir()
            (source / "private.txt").write_text("not selected")
            (stage / "outside.txt").symlink_to(source / "private.txt")
            output = pack(run, root / "review.tar.gz")
            with tarfile.open(output) as archive:
                manifest = json.loads(archive.extractfile("archive_manifest.json").read())
                self.assertEqual(manifest["schema"], "taxosafe_geometry_review_v1")
                self.assertFalse(manifest["includes_checkpoints"])
                self.assertNotIn("run/geometry/features.pth", archive.getnames())
                self.assertNotIn("run/geometry/outside.txt", archive.getnames())
                for item in manifest["files"]:
                    payload = archive.extractfile(item["path"]).read()
                    self.assertEqual(hashlib.sha256(payload).hexdigest(), item["sha256"])
            with self.assertRaises(ValueError):
                pack(run, output)


if __name__ == "__main__":
    unittest.main()
