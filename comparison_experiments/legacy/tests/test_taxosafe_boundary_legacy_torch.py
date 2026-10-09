"""Emulate the unavailable Torch APIs observed in the actual Boundary run.

This checks API independence and numerical equivalence on modern Torch with the
specific old interfaces removed. It does not claim a real old-Torch execution.
"""
import ast
import contextlib
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest import mock

import torch

from taxosafe_boundary import core
from taxosafe_discovery.verifier import SharedVerifier, build_episodes as old_episodes
from tests.test_taxosafe_discovery_geometry import fixture


@contextlib.contextmanager
def legacy_torch_apis(isin_mode="missing"):
    """Reject stable argsort keywords and remove or poison torch.isin.

    The full lifecycle tests reuse this context. Patches are always reversed,
    including after exceptions, and other argsort signatures remain available.
    """
    if isin_mode not in ("missing", "raises"):
        raise ValueError("Unknown legacy isin simulation")
    original_function = torch.argsort
    original_method = torch.Tensor.argsort
    sentinel = object()
    original_isin = getattr(torch, "isin", sentinel)

    def argsort_function(*args, **kwargs):
        if "stable" in kwargs:
            raise TypeError("argsort() received an unexpected keyword argument 'stable'")
        return original_function(*args, **kwargs)

    def argsort_method(value, *args, **kwargs):
        if "stable" in kwargs:
            raise TypeError("argsort() received an unexpected keyword argument 'stable'")
        return original_method(value, *args, **kwargs)

    def unavailable_isin(*args, **kwargs):
        raise AssertionError("torch.isin is unavailable on the simulated server")

    try:
        if isin_mode == "missing":
            if original_isin is not sentinel:
                delattr(torch, "isin")
        else:
            torch.isin = unavailable_isin
        with mock.patch.object(torch, "argsort", new=argsort_function), \
                mock.patch.object(torch.Tensor, "argsort", new=argsort_method):
            yield
    finally:
        if original_isin is sentinel:
            if hasattr(torch, "isin"):
                delattr(torch, "isin")
        else:
            torch.isin = original_isin


class LegacyTorchBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # These tests compare the compatibility implementation against native
        # modern APIs. On an actual old Torch install that oracle is absent;
        # the separate lifecycle tests can still import legacy_torch_apis.
        probe = torch.tensor([1., 0., 1.])
        if not callable(getattr(torch, "isin", None)):
            raise unittest.SkipTest("Native torch.isin reference unavailable; compatibility lifecycle tests remain usable")
        try:
            torch.argsort(probe, stable=True)
            probe.argsort(stable=True)
            torch.isin(probe, probe)
        except (AttributeError, TypeError, NotImplementedError, RuntimeError) as error:
            raise unittest.SkipTest("Native stable argsort/isin reference unavailable: {}".format(error))
        torch.set_num_threads(1)
        cls.args = fixture()
        cls.text = {"leaf": cls.args[0] @ torch.eye(4), "parent": cls.args[1] @ torch.eye(3)}
        original = old_episodes(*cls.args, template_scores=cls.text)
        cls.source = SharedVerifier.fit(original, loss="bce", epochs=1, batch_size=128).state_dict()
        cls.episodes = core.build_episodes(*cls.args, template_scores=cls.text, seed=1, folds=3)
        cls.modes = ("bce8", "rank8", "bce9", "rank9")
        cls.options = dict(seed=19, steps=3, batch_size=64, lr=.001)
        # Independent native stable reference, not the compatibility helper
        # under a modern environment. Equal state hashes therefore check ties,
        # rank samples, optimizer trajectory, normalization, and audits together.
        def native_order(values, descending=False):
            return torch.argsort(values, descending=descending, stable=True)
        def native_intersection(left, right):
            return bool(torch.isin(left, right).any())
        with mock.patch.object(core, "_stable_argsort", new=native_order), \
                mock.patch.object(core, "_has_common_query", new=native_intersection):
            cls.native = {mode: core.BoundaryVerifier.fit(cls.source, cls.episodes,
                mode=mode, **cls.options).state_dict() for mode in cls.modes}

    def test_simulation_blocks_both_stable_calls_and_both_isin_failure_modes(self):
        x = torch.tensor([2., 1.])
        original_isin, original_argsort = torch.isin, torch.argsort
        with legacy_torch_apis():
            self.assertFalse(hasattr(torch, "isin"))
            for call in (lambda: torch.argsort(x, stable=True), lambda: x.argsort(stable=True)):
                with self.assertRaises(TypeError): call()
            self.assertEqual(torch.argsort(x).tolist(), [1, 0])
            self.assertEqual(x.argsort().tolist(), [1, 0])
        self.assertIs(torch.isin, original_isin)
        self.assertIs(torch.argsort, original_argsort)
        with self.assertRaisesRegex(RuntimeError, "deliberate"):
            with legacy_torch_apis("raises"):
                with self.assertRaises(AssertionError): torch.isin(x, x)
                raise RuntimeError("deliberate")
        self.assertIs(torch.isin, original_isin)
        self.assertIs(torch.argsort, original_argsort)

    def test_stable_ties_match_native_in_both_directions_without_new_apis(self):
        cases = [
            torch.tensor([3., -2., 3., 0., -2., 3., 0.], dtype=torch.float32),
            torch.tensor([0., -0., 1., -1., -0., 1.], dtype=torch.float64),
            torch.tensor([4, 2, 4, 2, -3, -3], dtype=torch.int32),
            torch.tensor([2**63 - 1, -(2**63), 2**63 - 1, 0, -(2**63)], dtype=torch.int64),
            torch.empty(0, dtype=torch.float64),
            torch.tensor([7], dtype=torch.int64),
        ]
        expected = [(x, direction, torch.argsort(x, descending=direction, stable=True))
                    for x in cases for direction in (False, True)]
        with legacy_torch_apis():
            for values, descending, reference in expected:
                with self.subTest(values=values.tolist(), descending=descending):
                    result = core._stable_argsort(values, descending=descending)
                    self.assertEqual(result.dtype, torch.long)
                    self.assertEqual(result.device.type, "cpu")
                    self.assertTrue(torch.equal(result, reference))
        # Non-contiguous input is still sorted by its input order, not storage order.
        values = torch.tensor([3., 999., 1., 999., 3., 999.])[::2]
        with legacy_torch_apis("raises"):
            self.assertEqual(core._stable_argsort(values).tolist(), [1, 0, 2])
            self.assertEqual(core._stable_argsort(values, descending=True).tolist(), [0, 2, 1])

    def test_sort_input_validation_and_cpu_query_intersections(self):
        bad_values = [torch.tensor([[1., 2.]]), torch.tensor([float("nan")]),
                      torch.tensor([float("inf")]), torch.tensor([True, False]),
                      torch.tensor([1., 2.], dtype=torch.float16)]
        with legacy_torch_apis():
            for values in bad_values:
                with self.subTest(dtype=values.dtype, shape=tuple(values.shape)), self.assertRaises(ValueError):
                    core._stable_argsort(values)
            with self.assertRaises(ValueError):
                core._stable_argsort(torch.tensor([1.]), descending=1)
            self.assertTrue(core._has_common_query(torch.tensor([9, 1, 9]), torch.tensor([2, 9])))
            self.assertFalse(core._has_common_query(torch.tensor([1, 2]), torch.tensor([3, 4])))
            self.assertFalse(core._has_common_query(torch.empty(0, dtype=torch.long), torch.tensor([3])))
            with self.assertRaises(ValueError):
                core._has_common_query(torch.tensor([1.]), torch.tensor([1]))
            with self.assertRaises(ValueError):
                core._has_common_query(torch.tensor([[1]]), torch.tensor([1]))

    def test_all_four_modes_train_under_missing_isin_with_exact_native_trajectories(self):
        with legacy_torch_apis("missing"):
            rebuilt = core.build_episodes(*self.args, template_scores=self.text, seed=1, folds=3)
            self.assertEqual(core._digest(rebuilt), core._digest(self.episodes))
            for mode in self.modes:
                with self.subTest(mode=mode):
                    model = core.BoundaryVerifier.fit(self.source, rebuilt, mode=mode, **self.options)
                    self.assertEqual(model.state_dict()["state_sha256"], self.native[mode]["state_sha256"])
                    self.assertEqual(model.fit_report["optimizer_steps"], 6)
                    self.assertEqual(model.fit_report["initial_teacher_max_abs_difference"], 0.)
                    self.assertTrue(all(v > 0 for v in model.fit_report["parameter_delta_l2"].values()))
                    for level in ("leaf", "parent"):
                        self.assertTrue(torch.equal(model.normalization[level]["mean"][:8], self.source["normalization"][level]["mean"]))
                        self.assertTrue(torch.equal(model.normalization[level]["scale"][:8], self.source["normalization"][level]["scale"]))
                        history = model.fit_report["history"][level]
                        self.assertTrue(all(r["gradnorm"] > 0 for r in history))
                        self.assertTrue(all(r["rank"] > 0 if mode.startswith("rank") else r["rank"] == 0 for r in history))
                        if mode.endswith("9"):
                            self.assertTrue(bool((model.heads[level].layers[0].weight[:, 8] != 0).any()))
                    restored = core.BoundaryVerifier.from_state_dict(model.state_dict())
                    self.assertEqual(restored.state_dict()["state_sha256"], model.state_dict()["state_sha256"])

    def test_poisoned_isin_does_not_change_rank_pairs_or_training(self):
        source = SharedVerifier.from_state_dict(self.source)
        for level in ("leaf", "parent"):
            d = self.episodes[level]
            x = ((d["x"][:, :8] - source.normalization[level]["mean"]) / source.normalization[level]["scale"]).contiguous()
            groups = core._ranking_groups(d)
            expected = core._tail_pairs(source.heads[level], x, groups, torch.Generator().manual_seed(47))
            with legacy_torch_apis("raises"):
                actual_groups = core._ranking_groups(d)
                actual = core._tail_pairs(source.heads[level], x, actual_groups, torch.Generator().manual_seed(47))
            self.assertTrue(torch.equal(expected, actual))
            positive, negative = actual[:, 0], actual[:, 1]
            self.assertTrue(torch.equal(d["bank_id"][positive], d["bank_id"][negative]))
            self.assertTrue(torch.equal(d["candidate"][positive], d["candidate"][negative]))
            self.assertTrue(bool((d["query_index"][positive] != d["query_index"][negative]).all()))
            self.assertTrue(bool((d["y"][positive] == 1).all() & (d["y"][negative] == 0).all()))
        with legacy_torch_apis("raises"):
            for mode in self.modes:
                with self.subTest(mode=mode):
                    model = core.BoundaryVerifier.fit(self.source, self.episodes, mode=mode, **self.options)
                    self.assertEqual(model.state_dict()["state_sha256"], self.native[mode]["state_sha256"])

    def test_guard_composition_keeps_trained_leaf_and_exact_d05_parent(self):
        state = self.native["rank9"]
        with legacy_torch_apis():
            guard, report = core.make_leaf_guard(state, self.source)
            loaded = core.BoundaryVerifier.from_state_dict(guard)
        self.assertEqual(report["optimizer_steps"], 0)
        for name, value in self.source["heads"]["parent"].items():
            self.assertTrue(torch.equal(guard["heads"]["parent"][name], value))
        for name, value in state["heads"]["leaf"].items():
            self.assertTrue(torch.equal(guard["heads"]["leaf"][name], value))
        self.assertEqual(loaded.state_dict()["state_sha256"], guard["state_sha256"])

    def test_standalone_file_import_does_not_require_new_torch_symbols(self):
        # Execute the production file as its own module while both unavailable
        # APIs are disabled. Relative package import caches cannot hide a new
        # top-level dependency on argsort(stable=...) or torch.isin.
        with legacy_torch_apis():
            spec = importlib.util.spec_from_file_location("boundary_core_legacy_contract", core.__file__)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.assertFalse(hasattr(torch, "isin"))
            self.assertEqual(module._stable_argsort(torch.tensor([2., 1., 2.]), descending=True).tolist(), [0, 2, 1])
            bank = module.BoundaryBank.fit(*self.args)
            restored = module.BoundaryBank.from_state_dict(bank.state_dict())
            scores = restored.score(self.args[0][:2], self.args[1][:2], ["import-query-1", "import-query-2"])
            self.assertEqual(scores["leaf_scores"].shape, (2, 4))
            for dimension in (8, 9):
                for level in ("leaf", "parent"):
                    teacher = SharedVerifier.from_state_dict(self.source).heads[level]
                    head = module._head_from_source(self.source["heads"][level], dimension, self.source["hidden"])
                    x8 = torch.arange(56., dtype=torch.float32).reshape(7, 8) / 19.
                    values = x8 if dimension == 8 else torch.cat((x8, torch.ones(7, 1) * 99.), 1)
                    self.assertTrue(torch.equal(head(values), teacher(x8)))

    def test_new_family_syntax_and_unsupported_api_calls_are_absent(self):
        # Python 3.8 syntax checking is distinct from an actual Python 3.8 run.
        family = Path(core.__file__).parent
        for path in family.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 8))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                function = node.func
                if isinstance(function, ast.Attribute) and function.attr == "argsort":
                    self.assertFalse(any(k.arg == "stable" for k in node.keywords), str(path))
                if isinstance(function, ast.Attribute) and function.attr == "isin" and isinstance(function.value, ast.Name):
                    self.assertNotEqual(function.value.id, "torch", str(path))


if __name__ == "__main__":
    unittest.main()
