"""Independent experiment configuration and inherited-artifact boundaries."""
import copy
import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from taxosafe_frontier import protocol
from taxosafe_parentrisk import protocol as previous_protocol


class FrontierProtocolTests(unittest.TestCase):
    def setUp(self):
        self.cfg = protocol.effective_config(protocol.PROJECT_ROOT / protocol.DEFAULT_CONFIG)

    def parse(self, cfg):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.yml"
            path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
            return protocol.effective_config(path)

    def test_configuration_and_source_have_separate_semantic_bindings(self):
        self.assertEqual(self.parse(self.cfg), self.cfg)
        sig = protocol.signature(self.cfg, {"source": "one"})
        self.assertEqual(sig["method"], "frozen_reference_frontier_v1")
        changed = copy.deepcopy(self.cfg)
        changed["geometry"]["ridge"] *= 2
        self.assertNotEqual(sig, protocol.signature(changed, {"source": "one"}))
        self.assertNotEqual(sig, protocol.signature(self.cfg, {"source": "two"}))

    def test_grid_modes_parent_repairs_and_baseline_overrides_are_not_exposed(self):
        for key, value in (("grid_points", 7), ("mode", "parent_only"), ("parent_weights", [[1, 0, 0]]),
                           ("baseline_calibration", {"decoder": "membership"}), ("test_tuning", True)):
            cfg = copy.deepcopy(self.cfg)
            cfg["calibration"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.parse(cfg)

    def test_numeric_configuration_fails_closed(self):
        for section, key, value in (("geometry", "ridge", 0), ("geometry", "ridge", float("nan")),
                                    ("geometry", "shrinkage", True), ("geometry", "shrinkage", -1),
                                    ("calibration", "outer_folds", 1), ("calibration", "inner_folds", True),
                                    ("calibration", "min_rule_sources", 1)):
            cfg = copy.deepcopy(self.cfg)
            cfg[section][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.parse(cfg)
        for seed in (-1, True, "1", 2 ** 31):
            cfg = copy.deepcopy(self.cfg)
            cfg["seed"] = seed
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                self.parse(cfg)

    def test_new_entrypoint_and_package_do_not_enter_old_signature_file_set(self):
        old = {str(p.relative_to(protocol.PROJECT_ROOT)) for p in previous_protocol.code_files()}
        new = {str(p.relative_to(protocol.PROJECT_ROOT)) for p in protocol.code_files()}
        self.assertFalse(any(p.startswith("taxosafe_frontier/") for p in old))
        self.assertTrue(old < new)
        for path in ("taxosafe_frontier/protocol.py", "taxosafe_frontier/pipeline.py",
                     "taxosafe_frontier/__main__.py"):
            self.assertIn(path, new)
        self.assertFalse((protocol.PROJECT_ROOT / "refine_taxosafe_frontier.py").exists())

    def test_receipts_reject_mutated_escaping_or_symlink_artifacts(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "artifact.json"
            path.write_text("{}", encoding="utf-8")
            names = {"model": "artifact.json"}
            receipt = protocol.artifacts(root, names)
            protocol.verify_artifacts(root, receipt, names)
            invalid = copy.deepcopy(receipt)
            invalid["model"]["path"] = "../artifact.json"
            with self.assertRaises(ValueError):
                protocol.verify_artifacts(root, invalid, names)
            path.write_text("changed", encoding="utf-8")
            with self.assertRaises(ValueError):
                protocol.verify_artifacts(root, receipt, names)
            outside = root / "external.json"
            outside.write_text("{}", encoding="utf-8")
            path.unlink()
            path.symlink_to(outside)
            with self.assertRaises(ValueError):
                protocol.verify_artifacts(root, receipt, names)

    def test_fit_preflight_is_read_only_and_never_loads_tensors(self):
        from taxosafe_frontier import pipeline
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, destination = root / "source", root / "fresh"
            source.mkdir()
            args = ["taxosafe_frontier", "fit", "--reference-run-dir", str(source),
                    "--run-dir", str(destination), "--preflight"]
            stream = io.StringIO()
            with patch("sys.argv", args), patch.object(pipeline, "inspect_reference",
                    return_value={"binding": {"source": "test_fixture"}}), \
                    patch.object(pipeline, "load_reference", side_effect=AssertionError("no tensors in preflight")), \
                    redirect_stdout(stream):
                pipeline.run()
            report = json.loads(stream.getvalue())
            self.assertFalse(report["dataset_images_opened"])
            self.assertFalse(report["checkpoint_tensors_loaded"])
            self.assertFalse(destination.exists())

    def test_test_requires_explicit_opt_in_before_reading_a_source(self):
        from taxosafe_frontier import pipeline
        args = ["taxosafe_frontier", "test", "--reference-run-dir", "unused", "--run-dir", "unused-new"]
        with patch("sys.argv", args), patch.object(pipeline, "inspect_reference") as inspect, \
                redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            pipeline.run()
        self.assertEqual(raised.exception.code, 2)
        inspect.assert_not_called()


if __name__ == "__main__":
    unittest.main()
