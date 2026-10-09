"""Boundary controls cannot silently change historical D05 or test policy."""
import copy
import unittest

from taxosafe_boundary import protocol
from taxosafe_recovery import protocol as previous


class BoundaryContracts(unittest.TestCase):
    def test_predeclared_factorial_and_threshold_controls(self):
        cfg = protocol.effective_config(protocol.PROJECT_ROOT / protocol.DEFAULT_CONFIG)
        self.assertEqual(cfg, protocol.DEFAULTS)
        trained = [a for a in cfg["arms"] if a["kind"] == "train"]
        self.assertEqual([a["mode"] for a in trained], ["bce8", "rank8", "bce9", "rank9"])
        self.assertTrue(all(a["policy"] == "kp_frontier" for a in trained))
        self.assertTrue(all("weight_source" not in a for a in trained))
        self.assertEqual(cfg["arms"][2]["weight_source"], "G01_d05")
        self.assertEqual(cfg["arms"][8]["weight_source"], "G07_rank9")
        self.assertEqual(cfg["arms"][9]["policy"], "fixed_parent")

    def test_all_imported_runtime_is_bound_without_changing_old_scope(self):
        old, current = set(previous.code_files()), set(protocol.code_files())
        self.assertTrue(old < current)
        self.assertFalse(any("taxosafe_boundary" in str(p) for p in old))
        self.assertIn(protocol.PROJECT_ROOT / "taxosafe_boundary/protocol.py", current)
        before = protocol.signature(protocol.DEFAULTS, {"d05": "first"})
        cfg = copy.deepcopy(protocol.DEFAULTS)
        cfg["training"]["steps"] += 1
        self.assertNotEqual(before, protocol.signature(cfg, {"d05": "first"}))
        self.assertNotEqual(before, protocol.signature(protocol.DEFAULTS, {"d05": "second"}))

    def test_changed_controls_unknown_fields_and_invalid_options_rejected(self):
        mutations = [
            lambda c: c["arms"].pop(),
            lambda c: c["arms"].reverse(),
            lambda c: c["arms"][4].update(weight_source="G05_rank8"),
            lambda c: c.update(skip_failed_gates=True),
            lambda c: c["training"].update(use_test_errors=True),
            lambda c: c["training"].update(steps=0),
            lambda c: c["training"].update(batch_size=True),
            lambda c: c["training"].update(lr=.01),
            lambda c: c["training"].update(ranking_weight=float("nan")),
            lambda c: c["boundary"].update(tail_size=16),
            lambda c: c["calibration"].update(known_target=.94),
            lambda c: c["calibration"].update(seed=2),
        ]
        for index, change in enumerate(mutations):
            cfg = copy.deepcopy(protocol.DEFAULTS)
            change(cfg)
            with self.subTest(index=index), self.assertRaises(ValueError):
                protocol.validate_config(cfg)

    def test_validation_returns_detached_config(self):
        result = protocol.validate_config(protocol.DEFAULTS)
        result["training"]["steps"] = 1
        self.assertEqual(protocol.DEFAULTS["training"]["steps"], 300)


if __name__ == "__main__":
    unittest.main()
