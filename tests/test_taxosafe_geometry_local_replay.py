"""Saved-score replay must freeze DEV selection before reading TEST."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from taxosafe_geometry import calibration as geometry, protocol
from taxosafe_support import calibration as base
from tests.test_taxosafe_geometry_local import OPTIONS, fixture
from tests.test_taxosafe_geometry_calibration import BASELINE, META
from tools import replay_taxosafe_local as replay


class LocalReplayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.source, self.output = root / "source", root / "output"
        self.reference_config, self.config = root / "reference.json", root / "local.yml"
        protocol.write_json(self.reference_config, dict(calibration=dict(decoder="membership", membership_grid_points=3)))
        protocol.write_json(self.source / "geometry/source_binding.json",
                            dict(receipt_sha256={"training/config.json": protocol.file_hash(self.reference_config)}))
        protocol.write_json(self.source / "geometry/fit_report.json", dict(fit_splits=["train"],
                            test_used_for_fitting=False, unknown_images_used_for_fitting=False))
        protocol.write_json(self.source / "calibration/router.json", dict(baseline_router=BASELINE, meta=META,
                            baseline_router_sha256=geometry._hash(BASELINE)))
        self.rows = sum(fixture(), [])
        protocol.write_records(self.source / "calibration/development_scores.jsonl", self.rows)
        cfg = dict(name="fixture-local", seed=1, geometry=dict(shrinkage=.1, ridge=.001), calibration=OPTIONS)
        self.config.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    def run_replay(self, evaluate=False):
        return replay.replay(self.source, self.reference_config, self.output, self.config, evaluate)

    def test_default_never_opens_test_and_keeps_source_intact(self):
        before = {str(p): p.read_bytes() for p in self.source.rglob("*") if p.is_file()}
        self.assertFalse((self.source / "test").exists())
        result = self.run_replay()
        self.assertNotIn("test_selected", result)
        self.assertFalse(result["test_used_for_fitting"])
        frozen = protocol.read_json(self.output / "selection_frozen.json")
        self.assertEqual(frozen["router_sha256"], protocol.file_hash(self.output / "router.json"))
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.source.rglob("*") if p.is_file()})

    def make_test(self, overlap=False):
        (self.source / "test").mkdir()
        rows = copy.deepcopy(self.rows)
        for i, row in enumerate(rows):
            row["split"] = "test_" + row["status"]
            if not overlap:
                row["image_sha256"] = str(i + 5) * 64
        protocol.write_records(self.source / "test/predictions.jsonl", rows)
        protocol.write_records(self.source / "test/baseline_predictions.jsonl", base.apply_router(rows, BASELINE, META))

    def test_test_is_read_only_after_freeze_and_decisions_reproduce(self):
        self.make_test()
        read = replay._records
        def spy(path):
            if "test" in Path(path).parts:
                frozen = protocol.read_json(self.output / "selection_frozen.json")
                self.assertEqual(frozen["router_sha256"], protocol.file_hash(self.output / "router.json"))
            return read(path)
        with patch.object(replay, "_records", side_effect=spy):
            result = self.run_replay(True)
        self.assertTrue(result["test_selected"]["targets_passed"])
        self.assertEqual(result["test_known_protection"]["lost_baseline_correct"], 0)

    def test_overlap_tampered_config_and_reused_output_fail_closed(self):
        self.make_test(overlap=True)
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.run_replay(True)
        with self.assertRaisesRegex(ValueError, "new output"):
            self.run_replay()
        self.output = self.output.with_name("other")
        self.reference_config.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "source binding"):
            self.run_replay()


if __name__ == "__main__":
    unittest.main()
