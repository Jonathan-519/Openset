"""Compatibility checks for shared I/O and read-only code inventory."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import taxosafe_io
from tools import audit_code_inventory, run_unit_tests


ROOT = Path(__file__).resolve().parents[1]
HELPERS = ("load_yaml", "resolve_run_dir", "sha256_file", "write_json", "write_jsonl")


def legacy_wrappers(filename):
    """Exercise actual legacy wrappers without loading an image model/tokenizer."""
    tree = ast.parse((ROOT / filename).read_text(encoding="utf-8"))
    module = ast.Module(body=[node for node in tree.body
                             if isinstance(node, ast.FunctionDef) and node.name in HELPERS],
                        type_ignores=[])
    namespace = {"_io": taxosafe_io}
    exec(compile(module, filename, "exec"), namespace)
    return types.SimpleNamespace(**{name: namespace[name] for name in HELPERS})


class IOCompatibilityTests(unittest.TestCase):
    def test_both_legacy_modules_keep_identical_utf8_bytes_and_nonmutating_jsonl(self):
        APIs = [taxosafe_io, legacy_wrappers("taxosafe_eval_utils.py"),
                legacy_wrappers("calibrate_taxolocal_v21.py")]
        value = {"z": "浮游动物", "a": [1, None]}
        row = {"source": "桡足类", "parent_cosine": [1], "leaf_cosine": [2],
               "image_feature": [3], "keep": [4]}
        with tempfile.TemporaryDirectory() as folder:
            for i, api in enumerate(APIs):
                target = Path(folder) / str(i) / "out.json"
                self.assertIsNone(api.write_json(target, value))
                expected = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
                self.assertEqual(target.read_bytes(), expected)
                self.assertEqual(api.sha256_file(target, chunk_size=3), hashlib.sha256(expected).hexdigest())
                target = target.with_suffix(".jsonl")
                api.write_jsonl(target, iter([row]), drop_vector_fields=True)
                self.assertEqual(target.read_text(encoding="utf-8"), '{"source": "桡足类", "keep": [4]}\n')
                api.write_jsonl(target, [row])
                self.assertEqual(json.loads(target.read_text(encoding="utf-8")), row)
        self.assertEqual(row["parent_cosine"], [1])

    def test_yaml_and_run_path_semantics_remain_compatible(self):
        cfg = {"data": {"name": "dataset"}, "model": {"arch": "maple"}, "exp": "trial"}
        with tempfile.TemporaryDirectory() as folder:
            good, bad = Path(folder) / "good.yml", Path(folder) / "bad.yml"
            good.write_text("name: 浮游动物\n", encoding="utf-8")
            bad.write_text("- not-a-mapping\n", encoding="utf-8")
            for api in (taxosafe_io, legacy_wrappers("taxosafe_eval_utils.py"),
                        legacy_wrappers("calibrate_taxolocal_v21.py")):
                self.assertEqual(api.load_yaml(good), ({"name": "浮游动物"}, os.path.abspath(good)))
                with self.assertRaisesRegex(ValueError, "YAML root must be a mapping"):
                    api.load_yaml(bad)
                self.assertEqual(api.resolve_run_dir(cfg, 2, folder),
                                 os.path.join(folder, "runs", "dataset", "maple", "trial", "trial_2"))
                self.assertEqual(api.resolve_run_dir(cfg, "2", folder, "relative-run"),
                                 os.path.abspath("relative-run"))

    def test_shared_io_import_does_not_load_gpu_or_model_modules(self):
        subprocess.run([sys.executable, "-c", "import sys, taxosafe_io; "
                        "assert 'torch' not in sys.modules; assert 'models' not in sys.modules"],
                       cwd=ROOT, check=True, capture_output=True, text=True)


class InventoryTests(unittest.TestCase):
    def test_static_scan_never_executes_modules_and_accepts_new_config_schema(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "test_cli.py").write_text("raise RuntimeError('must not execute')\n"
                                             "if __name__ == '__main__':\n    pass\n")
            (root / "broken.py").write_text("def broken(:\n")
            (root / "configs").mkdir()
            (root / "configs/new.yml").write_text(
                "data:\n  near_dev_root: prepro/raw/near\n  val_intra: missing/manifest.txt\n")
            paths = ["test_cli.py", "broken.py", "not_checked_out.py", "configs/new.yml"]
            with patch.object(audit_code_inventory, "repository_paths", return_value=(paths, "git")), \
                    patch.object(audit_code_inventory.subprocess, "check_output", return_value="base\n"):
                report = audit_code_inventory.audit(root)
            self.assertEqual(report["python_sources_parsed"], 1)
            self.assertEqual(report["python_not_checked_out"], ["not_checked_out.py"])
            self.assertEqual(report["syntax_errors"][0]["file"], "broken.py")
            refs = report["config_references"]["configs/new.yml"]
            self.assertEqual(refs["status"], "ok")
            self.assertFalse(any(row["exists_locally"] for row in refs["references"]))
            self.assertEqual(report["test_discovery"]["root_inference_cli_modules"], ["test_cli.py"])

    def test_plain_export_directory_needs_no_git_and_prunes_data_and_environments(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "safe.py").write_text("value = 1\n", encoding="utf-8")
            for name in ("prepro/raw", "runs", "weights", ".git", "venv", ".venv",
                         "openset-venv", "__pycache__"):
                directory = root / name
                directory.mkdir(parents=True, exist_ok=True)
                (directory / "unread.py").write_text("not valid python!\n", encoding="utf-8")
            report = audit_code_inventory.audit(root)
            self.assertEqual(report["inventory_mode"], "filesystem")
            self.assertIsNone(report["base_commit"])
            self.assertEqual(report["python_sources_parsed"], 1)
            self.assertFalse(report["syntax_errors"])

    def test_unit_test_discovery_uses_explicit_package_root(self):
        with patch.object(unittest.TestLoader, "discover", return_value="suite") as collect:
            self.assertEqual(run_unit_tests.discover("test_repository_cleanup.py"), "suite")
        collect.assert_called_once_with(start_dir=str(ROOT / "tests"),
                                        pattern="test_repository_cleanup.py", top_level_dir=str(ROOT))


if __name__ == "__main__":
    unittest.main()
