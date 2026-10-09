"""Exercise preparation tools only against temporary synthetic directories."""

import argparse
import ast
import contextlib
import hashlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

import numpy as np
import yaml

from loader.treelibs import Tree
from prepro.legacy_tools import audit_taxosafe_splits as audit_module
from prepro.legacy_tools import build_taxosafe_v10_dev_splits as v10
from prepro.legacy_tools import build_taxosafe_v11_known_splits as v11
from prepro.legacy_tools import make_taxosafe_clean_v1 as clean_v1
from prepro.legacy_tools import me_npy
from prepro.legacy_tools import prepro as exporter
from prepro.legacy_tools import rebuild_taxosafe_v9_dataset as v9
from prepro.legacy_tools import regenerate_open_lists as open_lists


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = Path(os.environ.get(
    "H02_ORIGINAL_SOURCE", str(ROOT / "comparison_experiments/legacy")
))
ENTRIES = (
    "build_known_view", "build_taxosafe_v10_dev_splits",
    "build_taxosafe_v11_known_splits", "prepro",
    "rebuild_taxosafe_v9_dataset", "prepare_taxolocal_data",
    "audit_taxosafe_splits", "make_taxosafe_clean_v1",
    "regenerate_open_lists", "me_npy", "validate_taxosafe_setup",
)


def legacy(relative):
    path = ORIGINAL / relative
    namespace = {"__name__": "_legacy_preparation_fixture", "__file__": str(path)}
    with mock.patch.object(sys, "path", list(sys.path)):
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), namespace)
    return namespace


def frozen_assets():
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for directory in ("prepro/data", "prepro/splits")
        for path in sorted((ROOT / directory).rglob("*")) if path.is_file()
    }


def known_fixture(root):
    definitions = {
        "train": [("train-test.jpg", 0, b"A"), ("train-val.jpg", 0, b"B"),
                  ("train0.jpg", 0, b"C"), ("train1.jpg", 1, b"D"),
                  ("train1-copy.jpg", 1, b"D")],
        "val_known": [("val-test.jpg", 0, b"A"), ("val0.jpg", 0, b"B"),
                      ("val1.jpg", 1, b"E")],
        "test_known": [("test0.jpg", 0, b"A"), ("test1.jpg", 1, b"F"),
                       ("test0-copy.jpg", 0, b"A")],
    }
    images = root / "images"
    images.mkdir(parents=True)
    data = {"data_root": "images", "full_data_root": ".", "ood_root": "."}
    for split, records in definitions.items():
        rows = []
        for index, (name, label, content) in enumerate(records):
            (images / name).write_bytes(content)
            rows.append("{},{},{}".format(name, label, 20 + index))
        filename = "{}.txt".format(split)
        (root / filename).write_text("\n".join(rows) + "\n", encoding="utf-8")
        data[split] = filename
    config = root / "source.yml"
    config.write_text(yaml.safe_dump({"data": data}), encoding="utf-8")
    return config


class DataPreparationTests(unittest.TestCase):
    def test_all_help_entrypoints_and_imports_leave_locked_assets_unchanged(self):
        before = frozen_assets()
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        with tempfile.TemporaryDirectory() as temporary:
            tasks = []
            for entry in ENTRIES:
                tasks.append(([sys.executable, "-m", "prepro.legacy_tools." + entry, "--help"], ROOT))
                tasks.append(([sys.executable, str(ROOT / "prepro/legacy_tools" / (entry + ".py")), "--help"], temporary))

            def check(task):
                command, cwd = task
                result = subprocess.run(command, cwd=cwd, env=environment,
                                        capture_output=True, text=True, timeout=60)
                return command, result

            with ThreadPoolExecutor(max_workers=4) as executor:
                for command, result in executor.map(check, tasks):
                    self.assertEqual(result.returncode, 0, "{}\n{}".format(command, result.stderr))
                    self.assertIn("usage:", result.stdout)
            result = subprocess.run(
                [sys.executable, "-c", "; ".join("import prepro.legacy_tools." + item for item in ENTRIES)],
                cwd=ROOT, env=environment, capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "")
            self.assertEqual(list(Path(temporary).iterdir()), [])
        self.assertEqual(frozen_assets(), before)

    def test_preparation_config_is_exact_original_v10_data_mapping(self):
        original = yaml.safe_load((ORIGINAL / "configs/Zooplankton_Taxonomic_Tree/Zooplankton_Taxonomic_Tree_v10_perf.yml").read_text())
        current = yaml.safe_load(v11.DEFAULT_CONFIG.read_text())
        self.assertEqual(current, {"data": original["data"]})
        self.assertIn("Zooplankton_TT_v9_rebuild", current["data"]["train"])
        self.assertIn("Zooplankton_TT_v9_rebuild", current["data"]["val_known"])

    def test_original_preparation_sources_are_independent_files(self):
        for entry in ENTRIES[:6]:
            current = ROOT / "prepro/legacy_tools" / (entry + ".py")
            original = ORIGINAL / "prepro" / (entry + ".py")
            self.assertFalse(os.path.samefile(current, original), str(original))

    def test_v9_small_species_fixture_matches_original_rows_and_seed(self):
        old = legacy("prepro/rebuild_taxosafe_v9_dataset.py")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            known = root / "known"
            for parent, species, count in (("Copepoda", "Species_A", 11), ("Medusae", "Species_B", 7)):
                folder = known / parent / species
                folder.mkdir(parents=True)
                for index in range(count):
                    (folder / ("image{}.jpg".format(index))).write_bytes(bytes([index]))
            old["KNOWN_ROOT"], old["DATA_OUT"] = known, root / "original"
            with contextlib.redirect_stdout(io.StringIO()):
                expected = old["split_known_species_images"]({"Species_A": 0, "Species_B": 1})
                with mock.patch.multiple(v9, KNOWN_ROOT=known, DATA_OUT=root / "current"):
                    actual = v9.split_known_species_images({"Species_A": 0, "Species_B": 1})
            self.assertEqual(actual, expected)
            for path in (root / "original").iterdir():
                self.assertEqual(path.read_bytes(), (root / "current" / path.name).read_bytes())

    def test_v10_group_selection_and_output_order_match_original(self):
        old = legacy("prepro/build_taxosafe_v10_dev_splits.py")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows_by_mode = {
                "intra": ["{}/{}/{}.jpg,0,{}".format(parent, species, i, i)
                          for parent, species in (("Medusae", "B"), ("Copepoda", "A"))
                          for i in range(7)],
                "extra": ["{}/{}.jpg,-1,{}".format(source, i, i)
                          for source in ("Z", "A") for i in range(5)],
            }
            expected = {}
            for mode, lines in rows_by_mode.items():
                expected[mode] = old["stratified_split"](lines, mode)
                self.assertEqual(v10.stratified_split(lines, mode), expected[mode])
                (root / ("gt_val_{}.txt".format(mode))).write_text("\n".join(lines) + "\n")
            with contextlib.redirect_stdout(io.StringIO()):
                v10.main(["--root", str(root)])
            for mode, filenames in {
                "intra": ("gt_train_intra_v10.txt", "gt_val_intra_v10.txt"),
                "extra": ("gt_oe_train_v10.txt", "gt_val_extra_v10.txt"),
            }.items():
                for filename, lines in zip(filenames, expected[mode]):
                    self.assertEqual((root / filename).read_text(), "\n".join(lines) + "\n")

    def test_v11_identity_audit_is_byte_equivalent_to_original(self):
        old = legacy("prepro/build_taxosafe_v11_known_splits.py")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = known_fixture(root)
            inputs = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
            expected = old["prepare"](config, root / "original", root)
            actual = v11.prepare(config, root / "current", root)
            self.assertEqual(actual, expected)
            self.assertEqual(actual["retained_counts"], {"train": 2, "val_known": 2, "test_known": 3})
            self.assertEqual(actual["unique_test_known_images"], 2)
            self.assertFalse(actual["unknown_images_read"])
            self.assertFalse(actual["locked_test_manifest_changed"])
            for path in (root / "original").iterdir():
                self.assertEqual(path.read_bytes(), (root / "current" / path.name).read_bytes())
            self.assertFalse((root / "current/gt_test_known.txt").exists())
            for path, content in inputs.items():
                self.assertEqual(path.read_bytes(), content)
            self.assertEqual(v11.prepare(config, root / "current", root), actual)
            (root / "current/gt_train_known.txt").write_text("existing-different\n")
            with self.assertRaisesRegex(ValueError, "Existing preparation differs"):
                v11.prepare(config, root / "current", root)
            self.assertEqual((root / "current/gt_train_known.txt").read_text(), "existing-different\n")

    def test_v11_rejects_label_conflict_without_writing_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = known_fixture(root)
            path = root / "val_known.txt"
            path.write_text(path.read_text().replace("val-test.jpg,0,", "val-test.jpg,1,"))
            with self.assertRaisesRegex(ValueError, "conflicting known labels"):
                v11.prepare(config, root / "out", root)
            self.assertFalse((root / "out").exists())

    def test_legacy_audit_and_cleaner_preserve_original_duplicate_policy(self):
        old = legacy("audit_taxosafe_splits.py")
        original_audit_module = types.ModuleType("audit_taxosafe_splits")
        original_audit_module.__dict__.update(old)
        with mock.patch.dict(sys.modules, {"audit_taxosafe_splits": original_audit_module}):
            old_clean = legacy("make_taxosafe_clean_v1.py")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = known_fixture(root)
            extension = root / "visual.yml"
            extension.write_text(yaml.safe_dump({"base_config": config.name, "visual_support": {"seed": 41}}))
            expected = old["audit"](root, extension)
            actual = audit_module.audit(root, extension)
            self.assertEqual(actual, expected)
            self.assertEqual(clean_v1.select_records_to_drop(actual), old_clean["select_records_to_drop"](expected))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(clean_v1.main(["--config", str(extension), "--project-root", str(root)]), 0)
            output = root / "prepro/data/Zooplankton_Taxonomic_Tree_clean_v1"
            after = audit_module.audit(root, output / "TaxoSafe_visual_clean_v1.yml")
            self.assertEqual(after["totals"]["cross_split_duplicate_groups"], 0)
            self.assertEqual(after["totals"]["within_split_duplicate_groups"], 0)
            for split in ("train", "val_known", "test_known"):
                rows = (output / ("gt_{}.txt".format(split))).read_text().splitlines()
                self.assertEqual(len(rows), 2)
                self.assertEqual([int(row.rsplit(",", 1)[1]) for row in rows], [0, 1])

    def test_audit_resolves_role_roots_without_changing_legacy_fallback(self):
        root = Path("/synthetic/project")
        data = {"data_root": "known", "full_data_root": "legacy-near", "ood_root": "legacy-ood"}
        self.assertEqual(audit_module.split_root(root, data, "val_intra"), root / "legacy-near")
        self.assertEqual(audit_module.split_root(root, data, "test_extra"), root / "legacy-ood")
        data.update({"near_dev_root": "near-dev", "near_test_root": "near-test", "oe_train_root": "oe",
                     "ood_dev_root": "ood-dev", "ood_test_root": "ood-test"})
        for split, path in (("val_intra", "near-dev"), ("test_intra", "near-test"), ("oe_train", "oe"),
                            ("val_extra", "ood-dev"), ("test_extra", "ood-test")):
            self.assertEqual(audit_module.split_root(root, data, split), root / path)

    def test_generic_export_retains_original_rows_and_stable_pickle_class(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "images"
            for parent, leaf in (("Parent_A", "Leaf_A"), ("Parent_B", "Leaf_B")):
                folder = images / parent / leaf
                folder.mkdir(parents=True)
                for name in ("10.jpg", "2.jpg"):
                    (folder / name).write_bytes(name.encode())
            args = argparse.Namespace(data=str(images), display=False, subsample=None)
            source = ast.parse((ORIGINAL / "prepro/prepro.py").read_text())
            nodes = [node for node in source.body if isinstance(node, ast.FunctionDef)]
            namespace = {"np": np, "os": os, "json": __import__("json"), "Tree": Tree, "args": args}
            exec(compile(ast.Module(body=nodes, type_ignores=[]), "original-prepro-functions", "exec"), namespace)
            (root / "original").mkdir()
            (root / "current").mkdir()
            with contextlib.redirect_stdout(io.StringIO()):
                namespace["main"](str(root / "original"))
                exporter.main(str(root / "current"), args)
            self.assertEqual((root / "original/gt_all.txt").read_bytes(), (root / "current/gt_all.txt").read_bytes())
            tree = np.load(root / "current/tree.npy", allow_pickle=True).item()
            self.assertEqual(type(tree).__module__, "loader.treelibs")
            self.assertEqual(tree.leaf_nodes, {0: "Leaf_A", 1: "Leaf_B"})

    def test_historical_open_lists_keep_count_gate_before_writes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepro = root / "prepro"
            taxonomy = prepro / "raw/Zooplankton_Taxonomic_Tree"
            for _, parent, species in open_lists.val_intra_specification + open_lists.test_intra_specification:
                folder = taxonomy / parent / species
                folder.mkdir(parents=True)
                for name in ("10.jpg", "2.jpg"):
                    (folder / name).write_bytes(b"synthetic")
            ood = prepro / "raw/Zooplankton_OOD"
            for split in ("oe_train", "val_extra", "test_extra"):
                (ood / split).mkdir(parents=True)
                (ood / split / "1.jpg").write_bytes(b"synthetic")
            output = prepro / "data/Zooplankton_Taxonomic_Tree"
            with mock.patch.multiple(open_lists, PROJECT_ROOT=root, PREPRO_ROOT=prepro,
                                     TAXONOMY_ROOT=taxonomy, OOD_ROOT=ood, OUTPUT_ROOT=output):
                rows = open_lists.build_intra_rows(open_lists.val_intra_specification)
                self.assertTrue(rows[0][0].endswith("/2.jpg"))
                self.assertTrue(rows[1][0].endswith("/10.jpg"))
                with self.assertRaisesRegex(ValueError, "数据数量不符合预期"):
                    open_lists.regenerate()
            self.assertFalse(output.exists())

    def test_historical_tree_wrapper_retains_exact_seventeen_leaf_ids(self):
        original = ast.parse((ORIGINAL / "me_npy.py").read_text(encoding="utf-8"))
        expected = next(ast.literal_eval(node.value) for node in original.body
                        if isinstance(node, ast.Assign) and any(
                            isinstance(target, ast.Name) and target.id == "txt_leaf_order"
                            for target in node.targets))
        fold = __import__("json").loads((ROOT / "prepro/splits/Zooplankton_Taxonomic_Tree/fold1.json").read_text())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for parent, roles in fold.items():
                for species in roles["known"]:
                    (root / "known" / parent / species).mkdir(parents=True)
            with contextlib.redirect_stdout(io.StringIO()):
                tree, leaf_map = me_npy.build_tree(root / "known", root / "out")
            self.assertEqual(tree.leaf_nodes, dict(enumerate(expected)))
            self.assertEqual(leaf_map, {name: index for index, name in enumerate(expected)})
            self.assertEqual(len(tree.root.children), 7)
            self.assertEqual(type(np.load(root / "out/tree.npy", allow_pickle=True).item()).__module__,
                             "loader.treelibs")


if __name__ == "__main__":
    unittest.main()
