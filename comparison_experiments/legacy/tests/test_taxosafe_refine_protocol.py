"""Fail closed on configuration errors before loading a source checkpoint."""
import copy
from pathlib import Path
import tempfile
import unittest

import yaml

from taxosafe_refine import protocol


class RefinementConfigurationTests(unittest.TestCase):
    def test_default_and_spatial_ablation_are_explicit(self):
        path = protocol.PROJECT_ROOT / protocol.DEFAULT_CONFIG
        cfg = protocol.effective_config(path)
        self.assertEqual(cfg["features"], "fine")
        self.assertEqual(cfg["reconstruction"]["score_mode"], "relative")
        self.assertEqual(cfg["reconstruction"]["rank"], 16)
        self.assertNotIn("data", cfg)
        self.assertNotIn("parent_threshold", cfg)
        self.assertEqual(protocol.effective_config(path, seed=5)["seed"], 5)

    def test_bad_values_and_foreign_model_settings_are_rejected(self):
        cfg = protocol.effective_config(protocol.PROJECT_ROOT / protocol.DEFAULT_CONFIG)
        changes = [
            (None, "features", "learned_local"), (None, "seed", True),
            (None, "init_checkpoint", "unchecked.pth"),
            ("reconstruction", "rank", 0), ("reconstruction", "rank", 1.5),
            ("reconstruction", "score_mode", "automatic"),
            ("reconstruction", "scale_init", float("inf")),
            ("reconstruction", "eps", 0),
            ("training", "learning_rate", float("nan")),
            ("training", "validation_fraction", 0),
            ("training", "validation_fraction", .6),
            ("training", "patience", False),
            ("calibration", "grid_points", 1),
            ("calibration", "source_loo", "true"),
            ("calibration", "test_threshold", .1),
        ]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.yml"
            for section, name, value in changes:
                changed = copy.deepcopy(cfg)
                target = changed if section is None else changed[section]
                target[name] = value
                path.write_text(yaml.safe_dump(changed))
                with self.subTest(section=section, name=name, value=value), self.assertRaises(ValueError):
                    protocol.effective_config(path)

    def test_signature_binds_source_and_refinement_separately(self):
        cfg = protocol.effective_config(protocol.PROJECT_ROOT / protocol.DEFAULT_CONFIG)
        first = protocol.signature(cfg, {"checkpoint_sha256": "first"})
        second = protocol.signature(cfg, {"checkpoint_sha256": "second"})
        self.assertNotEqual(first["reference"], second["reference"])
        self.assertEqual(first["code"], second["code"])
        changed = copy.deepcopy(cfg)
        changed["reconstruction"]["rank"] += 1
        self.assertNotEqual(first["config"], protocol.signature(changed, {})["config"])


if __name__ == "__main__":
    unittest.main()
