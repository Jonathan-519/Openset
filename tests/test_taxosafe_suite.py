"""Protocol tests. End-to-end artifact integration uses explicitly synthetic features."""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import yaml

from prepro.build_known_view import PARENTS, build_known_view
from tools import prepare_taxosafe as prepare
from tools import run_taxosafe_suite as suite
from taxosafe_visual import pipeline
from test_taxosafe_residual import fixture


class PreparationTests(unittest.TestCase):
    def test_wrong_view_target_is_preserved_and_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fold = {p: {"known": ["species_" + p]} for p in PARENTS}
            for p in PARENTS:
                (root / "raw" / p / ("species_" + p)).mkdir(parents=True)
            path = root / "fold.json"
            path.write_text(json.dumps(fold))
            self.assertEqual(build_known_view(root / "raw", path, root / "view"), 7)
            self.assertEqual(build_known_view(root / "raw", path, root / "view"), 0)
            target = root / "view" / PARENTS[0] / ("species_" + PARENTS[0])
            target.unlink()
            target.symlink_to(root / "missing")
            with self.assertRaises(FileExistsError):
                build_known_view(root / "raw", path, root / "view")
            self.assertTrue(target.is_symlink())
            self.assertEqual(target.resolve(strict=False), root / "missing")

    def test_missing_or_wrong_vocabulary_never_silently_downloads_or_overwrites(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(prepare.urllib.request, "urlopen") as fetch:
            path = Path(directory) / "vocab.gz"
            with self.assertRaises(FileNotFoundError):
                prepare.ensure_bpe(path)
            path.write_bytes(b"user-provided-file")
            with self.assertRaises(ValueError):
                prepare.ensure_bpe(path, download=True)
            self.assertEqual(path.read_bytes(), b"user-provided-file")
            fetch.assert_not_called()

    def test_bad_download_leaves_no_vocabulary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vocab.gz"
            with patch.object(prepare.urllib.request, "urlopen", return_value=io.BytesIO(b"bad response")):
                with self.assertRaises(ValueError):
                    prepare.ensure_bpe(path, download=True)
            self.assertFalse(path.exists())
            self.assertEqual(list(Path(directory).iterdir()), [])


class SuiteTests(unittest.TestCase):
    def new_plan(self, root, reuse=False, seed=None):
        cfg = yaml.safe_load(suite.resolve(suite.DEFAULT_CONFIG).read_text())
        cfg["data"]["name"] = "SYNTHETIC_PROTOCOL_TEST"
        cfg["exp"] = "unit-test"
        source = root / "source.yml"
        run = None
        if reuse:
            run = root / "run"
            (run / "ckpt").mkdir(parents=True)
            (run / "ckpt/best.pth").write_bytes(b"SYNTHETIC CHECKPOINT; NOT A TORCH MODEL")
            source = run / "training.yml"
        source.write_text(yaml.safe_dump(cfg))
        return suite.make_plan(source, root / "suite", seed=seed, run_dir=run)

    def receipt(self, plan, step):
        suite.dump(suite.receipt_path(plan, step), {
            "plan_sha256": suite.file_hash(suite.resolve(plan["suite"]) / "plan.json"),
            "outputs_sha256": {p: suite.file_hash(suite.resolve(p)) for p in step["outputs"]}})

    def test_seed_updates_model_sampler_and_holdout_and_variants_keep_root_fixed(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = self.new_plan(Path(directory), seed=3)
            cfg = yaml.safe_load(suite.resolve(plan["training_config"]).read_text())
            self.assertEqual([cfg["seed"], cfg["data"]["seed"], cfg["data"]["sampler"]["seed"],
                              cfg["data"]["sampler"]["holdout_seed"], cfg["open_treecut"]["holdout_seed"]], [3] * 5)
            self.assertEqual(plan["trial"], "3")
            self.assertFalse(cfg["loss"].get("mask_hidden_in_consistency", False))
            roots = []
            for name, item in plan["variants"].items():
                extension = yaml.safe_load(suite.resolve(item["config"]).read_text())
                self.assertEqual(extension["base_config"], plan["archived_training_config"])
                settings = extension["visual_support"]
                variant = settings.pop("residual")
                for key, value in suite.VARIANTS[name].items():
                    self.assertEqual(variant[key], value)
                roots.append(settings)
            self.assertTrue(all(r == roots[0] for r in roots))
            self.assertEqual(len(suite.steps(plan)), 13)

    def test_changed_frozen_config_is_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = self.new_plan(Path(directory))
            path = suite.resolve(plan["variants"]["full"]["config"])
            path.write_text(path.read_text() + "# changed after planning\n")
            with self.assertRaisesRegex(ValueError, "Frozen input changed"):
                suite.execute(plan, "all", dry_run=True)

    def test_reused_training_archive_preserves_bom_and_windows_newlines(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run"
            (run / "ckpt").mkdir(parents=True)
            (run / "ckpt/best.pth").write_bytes(b"SYNTHETIC CHECKPOINT")
            content = suite.resolve(suite.DEFAULT_CONFIG).read_text()
            original = b"\xef\xbb\xbf" + content.replace("\n", "\r\n").encode("utf-8")
            archive = run / "training.yml"
            archive.write_bytes(original)
            plan = suite.make_plan(archive, root / "suite", run_dir=run)
            self.assertEqual(suite.resolve(plan["training_config"]).read_bytes(), original)
            self.assertNotIn("train", [step["stage"] for step in suite.steps(plan)])

    def test_partial_outputs_are_not_treated_as_completed(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = self.new_plan(Path(directory), reuse=True)
            path = suite.resolve(suite.steps(plan, "memory")[0]["outputs"][0])
            path.parent.mkdir(parents=True)
            path.write_bytes(b"partial")
            with self.assertRaisesRegex(FileExistsError, "Existing outputs"):
                suite.execute(plan, "memory", resume=True, dry_run=True)

    def test_resume_checks_completed_output_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = self.new_plan(Path(directory), reuse=True)
            for step in suite.steps(plan, "memory"):
                for output in step["outputs"]:
                    path = suite.resolve(output)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"SYNTHETIC RECEIPT TEST")
                self.receipt(plan, step)
            with contextlib.redirect_stdout(io.StringIO()):
                suite.execute(plan, "memory", resume=True)
            changed = suite.resolve(suite.steps(plan, "memory")[0]["outputs"][0])
            changed.write_bytes(b"modified")
            with self.assertRaisesRegex(ValueError, "Completed output changed"):
                suite.execute(plan, "memory", resume=True)

    def test_all_variants_produce_twelve_rows_and_summary_checks_root_invariance(self):
        meta, bank, rows, f, pc, lc = fixture()
        calls = []

        def synthetic_extract(cfg, split, model, texts, hierarchy, device):
            calls.append(split)
            if split == "train":
                records = [{"status": "known", "true_leaf": int(c), "true_parent": meta["leaf_to_parent"][c],
                            "source": meta["leaf_names"][c], "path": "train-" + str(i),
                            "image_sha256": str(bank["image_hashes"][i])} for i, c in enumerate(bank["labels"])]
                return records, bank["features"], np.zeros((len(records), 3)), np.zeros((len(records), 5))
            indices = [i for i, row in enumerate(rows) if row["status"] == split.split("_")[1]]
            records = copy.deepcopy([rows[i] for i in indices])
            for row in records:
                row["image_sha256"] = hashlib.sha256((split + row["image_sha256"]).encode()).hexdigest()
            return records, f[indices], pc[indices], lc[indices]

        with tempfile.TemporaryDirectory() as directory:
            plan = self.new_plan(Path(directory), reuse=True)
            for stage, fn in (("memory", pipeline.build_memory), ("calibrate", pipeline.calibrate), ("test", pipeline.test)):
                for step in suite.steps(plan, stage):
                    name = step["id"].rsplit("_", 1)[0]
                    item = plan["variants"][name]
                    args = SimpleNamespace(config=item["config"], trial=plan["trial"], run_dir=plan["run_dir"],
                                           artifact_dir=item["artifacts"], overwrite=False, profiles="both")
                    suite.verify_prerequisites(plan, step)
                    with patch.object(pipeline, "arguments", return_value=args), \
                         patch.object(pipeline, "load_model", return_value=(None, None, meta, None, 100.)), \
                         patch.object(pipeline, "extract", side_effect=synthetic_extract), \
                         contextlib.redirect_stdout(io.StringIO()):
                        fn()
                    self.receipt(plan, step)
            self.assertEqual(calls, ["train"] * 4 + ["val_known", "val_intra", "val_extra"] * 4
                             + ["test_known", "test_intra", "test_extra"] * 4)
            summary = suite.summarize(plan)
            self.assertEqual(len(summary["rows"]), 12)
            self.assertEqual(summary["primary_profile"], "coverage")
            self.assertTrue(summary["root_predictions_identical_across_variants"])
            item = plan["variants"]["fixed_metric"]
            path = suite.resolve(item["artifacts"]) / "test/coverage/predictions.jsonl"
            predictions = [json.loads(line) for line in path.read_text().splitlines()]
            predictions[0]["candidate_parent"] += 1
            path.write_text("\n".join(json.dumps(row) for row in predictions) + "\n")
            step = next(s for s in suite.steps(plan, "test") if s["id"] == "fixed_metric_test")
            self.receipt(plan, step)
            with self.assertRaisesRegex(ValueError, "Frozen-root comparison failed"):
                suite.summarize(plan)


if __name__ == "__main__":
    unittest.main()
