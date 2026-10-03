"""Configuration and provenance checks for the independent geometry stage."""
import copy
from pathlib import Path
import tempfile
import unittest

import yaml

from taxosafe_geometry import protocol


class GeometryProtocolTest(unittest.TestCase):
    def setUp(self):
        self.cfg = protocol.effective_config(protocol.PROJECT_ROOT / protocol.DEFAULT_CONFIG)

    def parse(self, cfg):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yml"
            path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
            return protocol.effective_config(path)

    def test_default_config_and_separate_signature(self):
        self.assertTrue(self.cfg["calibration"]["source_loo_safeguard"])
        signature = protocol.signature(self.cfg, {"source": "fixture"})
        self.assertEqual(signature["method"], "frozen_reference_geometry_v1")
        changed = copy.deepcopy(self.cfg)
        changed["geometry"]["ridge"] *= 2
        self.assertNotEqual(signature, protocol.signature(changed, {"source": "fixture"}))
        self.assertNotEqual(signature, protocol.signature(self.cfg, {"source": "different"}))

    def test_invalid_numeric_values_and_unknown_fields_fail_closed(self):
        cases = [("geometry", "shrinkage", -1), ("geometry", "ridge", 0),
                 ("geometry", "ridge", float("nan")), ("geometry", "ridge", True),
                 ("calibration", "grid_points", True), ("calibration", "weights", [0, 0]),
                 ("calibration", "weights", [0, 1.1]), ("calibration", "weights", [False]),
                 ("calibration", "source_loo", "true"), ("calibration", "test_tuning", True)]
        for section, key, value in cases:
            cfg = copy.deepcopy(self.cfg)
            cfg[section][key] = value
            with self.subTest(section=section, key=key, value=value), self.assertRaises(ValueError):
                self.parse(cfg)

    def test_safeguard_cannot_be_enabled_without_source_loo(self):
        self.cfg["calibration"]["source_loo"] = False
        with self.assertRaisesRegex(ValueError, "requires source_loo"):
            self.parse(self.cfg)
        self.cfg["calibration"]["source_loo_safeguard"] = False
        self.assertEqual(self.parse(self.cfg), self.cfg)


if __name__ == "__main__":
    unittest.main()
