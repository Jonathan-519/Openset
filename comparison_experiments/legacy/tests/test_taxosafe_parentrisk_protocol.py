"""Provenance, configuration and filesystem protection contracts."""
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

import yaml

from taxosafe_parentrisk import protocol


class ParentRiskProtocolTests(unittest.TestCase):
    def setUp(self):
        self.cfg = protocol.effective_config(protocol.PROJECT_ROOT / protocol.DEFAULT_CONFIG)

    def parse(self, cfg):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yml"
            path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
            return protocol.effective_config(path)

    def test_explicit_modes_have_separate_semantic_signatures(self):
        signatures = set()
        for suffix, mode in (("", "combined"), ("_audit", "audit"), ("_parent_only", "parent_only")):
            path = protocol.PROJECT_ROOT / "configs/Zooplankton_Taxonomic_Tree" / (
                "TaxoSafe_reference_parentrisk" + suffix + ".yml")
            cfg = protocol.effective_config(path)
            self.assertEqual(cfg["calibration"]["mode"], mode)
            self.assertEqual(self.parse(cfg), cfg)
            signatures.add(protocol.object_hash(protocol.signature(cfg, {"fixture": "source"})))
        self.assertEqual(len(signatures), 3)
        self.assertNotEqual(protocol.signature(self.cfg, {"source": 1}),
                            protocol.signature(self.cfg, {"source": 2}))

    def test_forbidden_overrides_and_invalid_values_fail_closed(self):
        cases = [("geometry", "ridge", 0), ("geometry", "ridge", float("nan")),
                 ("geometry", "shrinkage", True), ("geometry", "shrinkage", -0.1),
                 ("calibration", "mode", "legacy"), ("calibration", "outer_folds", 1),
                 ("calibration", "inner_folds", True), ("calibration", "grid_points", 99),
                 ("calibration", "min_rule_sources", 1),
                 ("calibration", "baseline_calibration", {"policy": "balanced"}),
                 ("calibration", "test_tuning", True)]
        for section, key, value in cases:
            cfg = copy.deepcopy(self.cfg)
            cfg[section][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.parse(cfg)
        for seed in (-1, True, "1", 2 ** 31):
            cfg = copy.deepcopy(self.cfg)
            cfg["seed"] = seed
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                self.parse(cfg)

    def test_code_signature_binds_new_and_frozen_dependencies(self):
        paths = {str(p.relative_to(protocol.PROJECT_ROOT)) for p in protocol.code_files()}
        for name in ("taxosafe_parentrisk/pipeline.py", "taxosafe_parentrisk/evidence.py",
                     "taxosafe_parentrisk/calibration.py", "taxosafe_parentrisk/decoder.py",
                     "taxosafe_parentrisk/folds.py", "taxosafe_refine/importer.py",
                     "taxosafe_geometry/core.py", "refine_taxosafe_parentrisk.py",
                     "taxosafe_support/reference.py", "metrics_open.py"):
            self.assertIn(name, paths)
        self.assertFalse(any(p.startswith(("runs/", "tests/")) for p in paths))

    def test_fresh_fit_preserves_partial_and_completed_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol.ensure_fresh_fit(root)
            (root / ".dcbs.lock").write_text("fixture", encoding="utf-8")
            protocol.ensure_fresh_fit(root)
            marker = root / "old_result.json"
            marker.write_text("preserve", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fresh"):
                protocol.ensure_fresh_fit(root)
            self.assertEqual(marker.read_text(), "preserve")

    def test_artifact_verification_rejects_mutation_and_path_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "artifact.json"
            path.write_text("{}", encoding="utf-8")
            names = {"artifact": "artifact.json"}
            receipt = protocol.artifacts(root, names)
            protocol.verify_artifacts(root, receipt, names)
            changed = copy.deepcopy(receipt)
            changed["artifact"]["path"] = "../artifact.json"
            with self.assertRaises(ValueError):
                protocol.verify_artifacts(root, changed, names)
            path.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                protocol.verify_artifacts(root, receipt, names)

    def test_fit_preflight_does_not_load_tensors_or_create_destination(self):
        from taxosafe_parentrisk import pipeline
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "fresh"
            source = root / "source"
            source.mkdir()
            argv = ["refine_taxosafe_parentrisk.py", "fit", "--reference-run-dir", str(source),
                    "--run-dir", str(destination), "--preflight"]
            output = io.StringIO()
            with patch("sys.argv", argv), patch.object(pipeline, "inspect_reference",
                    return_value={"binding": {"source": "receipt_fixture"}}) as inspect, \
                    patch.object(pipeline, "load_reference", side_effect=AssertionError("preflight cannot load tensors")), \
                    redirect_stdout(output):
                pipeline.run()
            inspect.assert_called_once_with(source)
            report = json.loads(output.getvalue())
            self.assertFalse(report["dataset_images_opened"])
            self.assertFalse(report["checkpoint_tensors_loaded"])
            self.assertFalse(destination.exists())

    def test_test_stage_requires_explicit_opt_in_before_any_source_read(self):
        from taxosafe_parentrisk import pipeline
        argv = ["refine_taxosafe_parentrisk.py", "test", "--reference-run-dir", "unused_reference",
                "--run-dir", "unused_destination"]
        with patch("sys.argv", argv), patch.object(pipeline, "inspect_reference") as inspect, \
                redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            pipeline.run()
        self.assertEqual(raised.exception.code, 2)
        inspect.assert_not_called()


if __name__ == "__main__":
    unittest.main()
