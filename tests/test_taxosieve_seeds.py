"""Regression checks for isolated seed snapshots and their frozen lifecycle."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

from tools import run_taxosieve_seeds as driver


ROOT = Path(__file__).resolve().parents[1]


class TaxoSieveSeedTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="taxosieve_seeds_")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.project = self.directory / "project"
        self.project.mkdir()
        for package in driver.PACKAGES:
            shutil.copytree(ROOT / package, self.project / package,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for name in driver.ROOT_FILES:
            shutil.copy2(ROOT / name, self.project / name)
        shutil.copytree(ROOT / "configs", self.project / "configs")
        self.reference = yaml.safe_load((self.project / "configs/taxosieve_reference.yml").read_text())
        # Only metadata is needed for prepare/verify; image execution is tested
        # separately by the pipeline and dataset suites.
        for key in driver.INPUT_KEYS:
            path = self.project / self.reference["data"][key]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture input for " + key + "\n", encoding="utf-8")

    def prepare(self, batch=None, seeds=(2, 5)):
        return driver.prepare_suite(self.project, seeds=seeds,
            output=self.directory / "suite", device="cpu", train_batch_size=batch)

    def run_python(self, script, cwd, *arguments):
        env = dict(os.environ, PYTHONPATH=str(cwd), PYTHONDONTWRITEBYTECODE="1")
        return subprocess.run([sys.executable, "-c", script, *map(str, arguments)],
            cwd=cwd, env=env, capture_output=True, text=True, timeout=60)

    def test_prepare_baseline_preserves_source_and_freezes_standalone_driver(self):
        source = {str(p.relative_to(self.project)): driver.digest(p)
                  for p in self.project.rglob("*") if p.is_file()}
        suite = self.prepare()
        plan = driver.verify_suite(suite)
        self.assertEqual(source, {str(p.relative_to(self.project)): driver.digest(p)
                         for p in self.project.rglob("*") if p.is_file()})
        self.assertEqual(plan["reference_recipe"]["train_batch_size"], 12)
        self.assertTrue(plan["reference_recipe"]["same_optimizer_steps_as_baseline"])
        self.assertEqual(driver.digest(suite / "driver.py"), driver.digest(Path(driver.__file__)))
        self.assertEqual(driver.digest(suite / "runtime/configs/taxosieve_reference.yml"),
                         driver.digest(self.project / "configs/taxosieve_reference.yml"))
        # No tools package exists under this fresh project's runtime or suite.
        # Importing the copied driver from elsewhere must remain self-contained.
        script = """
import importlib.util, sys
spec = importlib.util.spec_from_file_location('frozen', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert module.verify_suite(sys.argv[2])['seeds'] == [2, 5]
"""
        result = self.run_python(script, self.directory, suite / "driver.py", suite)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_batch24_changes_only_three_reference_settings(self):
        suite = self.prepare(24)
        plan = driver.verify_suite(suite)
        recipe = plan["reference_recipe"]
        self.assertEqual((recipe["images_per_species"], recipe["batches_per_epoch"],
                          recipe["sampled_images_per_epoch"]), (4, 120, 2880))
        self.assertFalse(recipe["same_optimizer_steps_as_baseline"])
        expected = copy.deepcopy(self.reference)
        expected["data"]["batch_size"] = 24
        expected["data"]["sampler"].update(images_per_species=4, batches_per_epoch=120)
        actual = yaml.safe_load((suite / "runtime/configs/taxosieve_reference.yml").read_text())
        self.assertEqual(actual, expected)

    def test_real_sampler_matches_recorded_batch_budget_and_seed(self):
        from loader.hierarchical_episode_sampler import HierarchicalEpisodeBatchSampler

        class Dataset:
            target = [leaf for leaf in range(6) for _ in range(8)]

            def __len__(self):
                return len(self.target)

        dataset = Dataset()
        for seed in (8, 18, 28, 38, 48):
            for batch in (12, 24):
                recipe = driver.batch_recipe(self.reference, batch)
                settings = {key: recipe[key] for key in ("parents_per_batch", "species_per_parent",
                    "images_per_species", "batches_per_epoch")}
                sampler = HierarchicalEpisodeBatchSampler(dataset, [0, 0, 0, 1, 1, 1],
                    seed=seed, holdout_seed=seed, **settings)
                sampler.set_epoch(3)
                draws = list(sampler)
                self.assertEqual(sampler.seed, sampler.holdout_seed)
                self.assertEqual(len(draws), recipe["batches_per_epoch"])
                self.assertEqual({len(draw) for draw in draws}, {batch})
                self.assertEqual(sum(map(len, draws)), 2880)
                sampler.set_epoch(3)
                self.assertEqual(draws, list(sampler))

    def test_default_main_recipe_stays_locked_while_snapshot_accepts_one_common_seed(self):
        suite = self.prepare()
        script = """
import copy, sys
from taxosieve import protocol as p
cfg = p.effective_config()
assert [cfg[k]['seed'] for k in ('d05', 'd05_calibration', 'calibration')] == [1, 1, 1]
if sys.argv[1] == 'snapshot':
    for seed in (2, 5, 8, 18, 28, 38, 48):
        cfg = p.effective_config(seed=seed)
        assert all(cfg[k]['seed'] == seed for k in ('d05', 'd05_calibration', 'calibration'))
    for bad in (-1, True, 2**31):
        try:
            p.effective_config(seed=bad)
        except ValueError:
            pass
        else:
            raise AssertionError('accepted invalid seed')
    for key, value in (('seed', 99), ('epochs', 101)):
        changed = copy.deepcopy(cfg)
        changed['d05'][key] = value
        try:
            p.validate_seed_config(changed)
        except ValueError:
            pass
        else:
            raise AssertionError('accepted mixed seeds or modified recipe')
else:
    assert not hasattr(p, 'validate_seed_config')
    try:
        p.effective_config(seed=5)
    except TypeError:
        pass
    else:
        raise AssertionError('main recipe is no longer fixed')
"""
        for cwd, mode in ((self.project, "source"), (suite / "runtime", "snapshot")):
            with self.subTest(mode=mode):
                result = self.run_python(script, cwd, mode)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_snapshot_patch_ignores_comments_but_rejects_algorithm_changes(self):
        protocol = self.project / "taxosieve/protocol.py"
        original = protocol.read_text()
        protocol.write_text(original.replace("    cfg = yaml.safe_load(resolve(path).read_text(encoding=\"utf-8\"))",
            "    # The comment does not affect the locked recipe.\n    cfg = yaml.safe_load(resolve(path).read_text(encoding=\"utf-8\"))"))
        self.prepare()
        altered = original.replace("return copy.deepcopy(cfg)", "return cfg", 1)
        with self.assertRaisesRegex(ValueError, "function changed"):
            driver.replace_function(altered, "effective_config",
                "def effective_config(path=DEFAULT_CONFIG):\n"
                "    cfg = yaml.safe_load(resolve(path).read_text(encoding='utf-8'))\n"
                "    if object_hash(cfg) != object_hash(DEFAULTS):\n"
                "        raise ValueError('TaxoSieve settings differ from the locked TaxoSieve_v1 recipe')\n"
                "    return copy.deepcopy(cfg)\n", "")

    def test_runtime_tamper_is_rejected(self):
        suite = self.prepare()
        file = suite / "runtime/taxosieve/pipeline.py"
        file.chmod(0o644)
        file.write_text(file.read_text() + "\n# changed\n")
        with self.assertRaisesRegex(ValueError, "Frozen runtime changed"):
            driver.verify_suite(suite)

    def test_input_tamper_is_rejected(self):
        suite = self.prepare()
        file = self.project / self.reference["data"]["train"]
        file.write_text("changed\n")
        with self.assertRaisesRegex(ValueError, "Frozen data/config input changed"):
            driver.verify_suite(suite)

    def test_driver_tamper_is_rejected(self):
        suite = self.prepare()
        file = suite / "driver.py"
        file.chmod(0o644)
        file.write_text(file.read_text() + "\n# changed\n")
        with self.assertRaisesRegex(ValueError, "Sweep driver changed"):
            driver.verify_suite(suite)

    def test_declared_batch_recipe_tamper_is_rejected(self):
        suite = self.prepare(24)
        file = suite / "plan.json"
        plan = driver.read_json(file)
        plan["reference_recipe"]["batches_per_epoch"] = 240
        file.chmod(0o644)
        file.write_text(json.dumps(plan))
        with self.assertRaisesRegex(ValueError, "Reference recipe metadata changed"):
            driver.verify_suite(suite)

    def test_invalid_seeds_and_batches_leave_no_suite(self):
        for seeds in ((), (2, 2), (True,), (-1,), (2**31,)):
            with self.subTest(seeds=seeds), self.assertRaises(ValueError):
                self.prepare(seeds=seeds)
        for batch in (0, 6, 13, 42, True):
            with self.subTest(batch=batch), self.assertRaises(ValueError):
                self.prepare(batch=batch)
        self.assertFalse((self.directory / "suite").exists())

    def test_existing_run_cannot_be_overwritten_or_automatically_resumed(self):
        suite = self.prepare()
        (suite / "seed_2").mkdir()
        with patch.object(driver, "execute") as execute:
            with self.assertRaisesRegex(ValueError, "never overwrite"):
                driver.run_suite(suite)
            execute.assert_not_called()
        self.assertFalse((suite / "started.json").exists())

    def test_all_dev_precede_frozen_selection_and_all_test(self):
        suite = self.prepare()
        stages = []

        def execute(directory, label, arguments, seed):
            phase = arguments[0]
            stages.append((seed, phase))
            if phase == "calibrate":
                path = suite / ("seed_" + str(seed)) / "calibration"
                path.mkdir(parents=True)
                values = {key: (0.95 if seed == 2 else 0.9) for key in driver.METRICS.values()}
                driver.write_json(path / "audit.json", dict(crossfit=dict(complete=True,
                    passed=False, report=dict(targets_passed=False, metrics=values))))
            if phase == "test":
                self.assertEqual(driver.read_json(suite / "selection.json")["recommended_seed"], 2)

        summary = dict(rows=[], selection=dict(recommended_seed=2, recommended_targets_passed=False),
            test_statistics={key: dict(mean=None, sample_std=None) for key in driver.METRICS})
        with patch.object(driver, "execute", side_effect=execute), \
             patch.object(driver, "summarize_suite", return_value=summary):
            driver.run_suite(suite)
        self.assertEqual(stages, [(2, "preflight"), (2, "train"), (2, "calibrate"),
            (5, "preflight"), (5, "train"), (5, "calibrate"), (2, "test"), (5, "test")])
        self.assertFalse(driver.read_json(suite / "selection.json")["test_used_for_selection"])

    def test_legacy_cli_wrappers_and_canonical_entry_share_all_options(self):
        for name in ("run_taxosieve_seeds.py", "run_taxosieve_seeds_batch.py", "tools/run_taxosieve_seeds.py"):
            result = subprocess.run([sys.executable, str(ROOT / name), "--help"],
                cwd=self.directory, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--train-batch-size", result.stdout)
            self.assertIn("--suite", result.stdout)

    def test_snapshot_preserves_native_cache_training_calibration_test_lifecycle(self):
        suite = self.prepare(seeds=(3,))
        # Reuse the native pipeline's synthetic-image-boundary integration
        # checks against the physical snapshot, including real CPU D05 fitting,
        # OOF calibration, TEST, receipt validation and duplicate accounting.
        # Only this temporary fixture's already-shortened recipe uses seed=3.
        script = """
from pathlib import Path
import sys
path = Path(sys.argv[1])
source = path.read_text()
before = '        cls.cfg["d05"].update(epochs=2, batch_size=32)'
after = before + '\\n        for name in ("d05", "d05_calibration", "calibration"):\\n            cls.cfg[name]["seed"] = 3'
assert source.count(before) == 1
source = source.replace(before, after)
before = 'seed=1)'
assert source.count(before) == 1
source = source.replace(before, 'seed=3)')
sys.argv = [str(path)]
__file__ = str(path)
exec(compile(source, str(path), 'exec'), globals())
"""
        result = self.run_python(script, suite / "runtime", ROOT / "tests/test_taxosieve_pipeline.py")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Ran ", result.stderr)
        self.assertNotIn("Ran 0 tests", result.stderr)
        self.assertIn("OK", result.stderr)


if __name__ == "__main__":
    unittest.main()
