"""D05 extraction checks, including exact comparisons with archived source.

Set TAXOSIEVE_ORIGINAL_SOURCE (or H02_ORIGINAL_SOURCE) to an external intact
snapshot to enable the optional archived comparison when auditing a release.
The self-contained D05 mathematical and lifecycle checks always run.
No images, historical result directories, or trained user weights are needed.
"""
import ast
import copy
import hashlib
import importlib.util
import math
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from taxosieve import d05
from taxosieve.d05_geometry import GeometryBank
from taxosieve.d05_verifier import SharedVerifier, build_episodes
from taxosafe_support import membership_calibration as membership


def _digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _row(identity, split, status, leaf, meta, candidate_parent=1):
    parent_count = len(meta["parent_names"])
    leaf_count = len(meta["leaf_names"])
    parent_logits = [0.] * parent_count
    parent_logits[candidate_parent] = 2.
    leaf_logits = [float(i) / 10. for i in range(leaf_count)]
    parent = None if status == "extra" else meta["leaf_to_parent"][leaf]
    return dict(image_sha256=_digest(identity), path=identity + ".png",
        split=split, status=status, source=identity.split("-")[0],
        species=meta["leaf_names"][leaf] if status == "known" else "held-source",
        true_leaf=leaf if status == "known" else None, true_parent=parent,
        log_probs=[-math.log(1 + parent_count + leaf_count)] * (1 + parent_count + leaf_count),
        global_pred_leaf=leaf_count - 1,
        support_evidence=dict(parent_logits=parent_logits, leaf_logits=leaf_logits,
            parent_membership_logits=[.1] * parent_count,
            leaf_membership_logits=[.2] * leaf_count))


def _group(rows, features):
    return dict(records=rows,
        features=dict(source_fine=features.clone(), source_parent=features.clone(), clip=features.clone()),
        image_sha256=[r["image_sha256"] for r in rows],
        record_feature_indices=list(range(len(rows))))


def _fixture():
    # One two-child parent and one singleton parent exercise both holdout cases.
    meta = dict(leaf_names=["a", "b", "c"], parent_names=["p", "q"], leaf_to_parent=[0, 0, 1])
    generator = torch.Generator().manual_seed(137)
    features = F.normalize(torch.randn(12, 8, generator=generator), dim=1)
    text = {prefix + "_" + level: F.normalize(torch.randn(n, 8, generator=generator), dim=1)
            for prefix in ("single", "ensemble") for level, n in (("leaf", 3), ("parent", 2))}
    provenance = dict(source_binding={"fixture": "frozen_reference"},
        clip_core_sha256=_digest("core"),
        clip_initialization="source_frozen_pretrained_core_without_prompts_or_adapters",
        templates_sha256=_digest("templates"), preprocessing={"image_size": 8})
    rows = [_row("train-" + str(i), "train", "known", i // 4, meta) for i in range(12)]
    train = dict(meta=meta, text=text, provenance=provenance,
                 groups={"train": _group(rows, features)}, timings={})
    groups = {}
    for status in ("known", "intra", "extra"):
        split = "val_" + status
        query = F.normalize(torch.randn(2, 8, generator=generator), dim=1)
        rows = [_row(status + "-" + str(i), split, status, i, meta) for i in range(2)]
        groups[split] = _group(rows, query)
    development = dict(meta=copy.deepcopy(meta), text=copy.deepcopy(text),
        provenance=copy.deepcopy(provenance), groups=groups, timings={})
    return train, development


def _load_original():
    source = os.environ.get("TAXOSIEVE_ORIGINAL_SOURCE") or os.environ.get("H02_ORIGINAL_SOURCE")
    if not source:
        return None
    original = Path(source).expanduser().resolve()
    required = [original / "taxosafe_discovery" / name
                for name in ("geometry.py", "verifier.py", "backend.py")]
    if not all(path.is_file() for path in required):
        raise FileNotFoundError("Configured archived snapshot is incomplete: " + str(original))
    if str(original) not in sys.path:
        sys.path.append(str(original))
    package_name = "_h02_d05_original_snapshot"
    package = types.ModuleType(package_name)
    package.__path__ = [str(original / "taxosafe_discovery")]
    sys.modules[package_name] = package
    modules = {}
    for name in ("geometry", "verifier"):
        fullname = package_name + "." + name
        spec = importlib.util.spec_from_file_location(fullname, original / "taxosafe_discovery" / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[fullname] = module
        spec.loader.exec_module(module)
        modules[name] = module
    # Load just the archived pure scoring functions, not its 11-arm runtime.
    source = (original / "taxosafe_discovery" / "backend.py").read_text(encoding="utf-8")
    lines = source.splitlines(keepends=True)
    selected = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)
                and node.name in ("_representations", "_templates", "score_groups")]
    namespace = dict(__package__=package_name, copy=copy, torch=torch, membership=membership)
    exec("\n\n".join("".join(lines[node.lineno-1:node.end_lineno]) for node in selected), namespace)
    modules["score_groups"] = namespace["score_groups"]
    return modules


def _assert_equal(test, left, right, path="root"):
    if torch.is_tensor(left):
        test.assertTrue(torch.is_tensor(right), path)
        test.assertEqual(left.dtype, right.dtype, path)
        test.assertEqual(left.shape, right.shape, path)
        test.assertTrue(torch.equal(left, right), path)
    elif isinstance(left, dict):
        test.assertEqual(set(left), set(right), path)
        for key in left:
            _assert_equal(test, left[key], right[key], path + "." + str(key))
    elif isinstance(left, (list, tuple)):
        test.assertEqual(len(left), len(right), path)
        for i, (a, b) in enumerate(zip(left, right)):
            _assert_equal(test, a, b, path + "[" + str(i) + "]")
    else:
        test.assertEqual(left, right, path)


class D05CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.train, cls.development = _fixture()
        cls.payload, cls.report = d05.fit_payload(cls.train, seed=1, folds=3,
            epochs=2, batch_size=32, lr=.001, hidden=32)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_real_bce_updates_and_singleton_holdout(self):
        report = self.report["verifier"]
        episode = report["episode_report"]
        self.assertEqual(report["loss"], "bce")
        self.assertEqual(report["evidence_dimension"], 8)
        self.assertTrue(all(v > 0 for v in report["parameter_delta_l2"].values()))
        self.assertEqual(episode["single_child_parent_ids"], [1])
        self.assertEqual(episode["single_child_parent_near_examples_skipped"], 4)
        for detail in episode["episodes"]:
            self.assertEqual(detail["query_support_overlap"], 0)
        expected = 2 * sum(math.ceil(v["rows"] / 32) for v in episode["examples"].values())
        self.assertEqual(self.report["optimizer_steps"], expected)

    def test_state_and_inference_never_refit(self):
        before = d05.score_groups(self.development, self.payload, self.train["meta"])
        with patch.object(GeometryBank, "fit", side_effect=AssertionError("Unexpected fit")), \
             patch("taxosieve.d05_geometry._statistics", side_effect=AssertionError("Unexpected statistics")), \
             patch.object(SharedVerifier, "fit", side_effect=AssertionError("Unexpected optimization")):
            geometry, verifier = d05.validate_payload(self.payload, self.train["meta"], self.train)
            after = d05.score_groups(self.development, self.payload, self.train["meta"])
            _assert_equal(self, geometry.state_dict(), self.payload["geometry"])
            _assert_equal(self, verifier.state_dict(), self.payload["verifier"])
        self.assertEqual(before, after)

    def test_frozen_reference_candidates_and_alias_mapping(self):
        cache = copy.deepcopy(self.development)
        group = cache["groups"]["val_intra"]
        alias = dict(group["records"][0], path="an-alias.png")
        group["records"].append(alias)
        group["record_feature_indices"].append(0)
        scored = d05.score_groups(cache, self.payload, self.train["meta"])
        rows = scored["val_intra"]
        self.assertEqual(rows[0]["discovery"], rows[-1]["discovery"])
        for split, rows in scored.items():
            candidates = membership.candidate_scores(cache["groups"][split]["records"], self.train["meta"])
            for i, row in enumerate(rows):
                self.assertEqual(row["discovery"]["candidate_parent"], int(candidates["parent"][i]))
                self.assertEqual(row["discovery"]["candidate_leaf"], int(candidates["leaf"][i]))

    def test_provenance_and_conflicting_aliases_fail(self):
        moved = copy.deepcopy(self.train)
        moved["groups"]["train"]["records"][0]["status"] = "intra"
        with self.assertRaises(ValueError):
            d05.fit_payload(moved, epochs=1)
        duplicate = copy.deepcopy(self.development)
        group = duplicate["groups"]["val_known"]
        bad = copy.deepcopy(group["records"][0])
        bad["support_evidence"]["parent_logits"][0] += 1.
        group["records"].append(bad)
        group["record_feature_indices"].append(0)
        with self.assertRaises(ValueError):
            d05.score_groups(duplicate, self.payload, self.train["meta"])
        altered = copy.deepcopy(self.payload)
        altered["geometry"]["statistics"]["leaf"]["within_variance"][0] *= 1.1
        with self.assertRaises(ValueError):
            d05.validate_payload(altered, self.train["meta"])

    def test_query_support_overlap_is_rejected(self):
        geometry, _ = d05.validate_payload(self.payload, self.train["meta"])
        group = self.train["groups"]["train"]
        with self.assertRaises(ValueError):
            geometry.score(group["features"]["clip"], group["features"]["clip"], group["image_sha256"])

    def test_other_experiment_payloads_are_rejected(self):
        altered = copy.deepcopy(self.payload)
        altered["verifier"]["fit_report"]["loss"] = "bce_rank"
        with self.assertRaises(ValueError):
            d05.validate_payload(altered, self.train["meta"])
        altered = copy.deepcopy(self.payload)
        altered["projection"] = {"state": {}}
        with self.assertRaises(ValueError):
            d05.validate_payload(altered, self.train["meta"])


class ArchivedD05EquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = _load_original()
        if cls.original is None:
            raise unittest.SkipTest("Archived snapshot unavailable; set TAXOSIEVE_ORIGINAL_SOURCE for exact comparison")
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.train, cls.development = _fixture()
        group = cls.train["groups"]["train"]
        features = group["features"]["clip"]
        labels = torch.tensor([r["true_leaf"] for r in group["records"]], dtype=torch.long)
        args = (features, features, labels, group["image_sha256"], cls.train["meta"])
        options = dict(template_scores=d05._templates(features, cls.train["text"]), folds=3, seed=1, shrinkage=.1)
        cls.old_episodes = cls.original["verifier"].build_episodes(*args, **options)
        cls.new_episodes = build_episodes(*args, **options)
        cls.old_geometry = cls.original["geometry"].GeometryBank.fit(*args, shrinkage=.1)
        cls.old_verifier = cls.original["verifier"].SharedVerifier.fit(cls.old_episodes,
            loss="bce", seed=1, epochs=2, batch_size=32, lr=.001, hidden=32)
        cls.old_payload = dict(meta=copy.deepcopy(cls.train["meta"]), model_spec=copy.deepcopy(d05.MODEL_SPEC),
            text=copy.deepcopy(cls.train["text"]), provenance=copy.deepcopy(cls.train["provenance"]),
            projection=None, inference_spec_sha256=d05._text_contract(cls.train),
            geometry=cls.old_geometry.state_dict(), verifier=cls.old_verifier.state_dict(),
            fit_report=dict(training_execution="completed",
                optimizer_steps=cls.old_verifier.fit_report["optimizer_steps"],
                geometry=copy.deepcopy(cls.old_geometry.fit_report), verifier=copy.deepcopy(cls.old_verifier.fit_report)))
        cls.new_payload, _ = d05.fit_payload(cls.train, seed=1, folds=3,
            epochs=2, batch_size=32, lr=.001, hidden=32)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "threads"):
            torch.set_num_threads(cls.threads)

    def test_episode_values_weights_and_order_are_exact(self):
        for level in ("leaf", "parent"):
            for key in ("x", "y", "weight", "source_leaf", "kind", "query_index", "candidate"):
                _assert_equal(self, self.old_episodes[level][key], self.new_episodes[level][key], level + "." + key)
        for key in ("episodes", "train_count", "image_hash_digest", "single_child_parent_near_examples_skipped"):
            self.assertEqual(self.old_episodes["report"][key], self.new_episodes["report"][key])

    def test_bce_parameters_normalization_and_scores_are_exact(self):
        _assert_equal(self, self.old_payload["geometry"], self.new_payload["geometry"])
        for key in ("heads", "normalization"):
            _assert_equal(self, self.old_payload["verifier"][key], self.new_payload["verifier"][key])
        old_report = self.old_payload["verifier"]["fit_report"]
        new_report = self.new_payload["verifier"]["fit_report"]
        for key in ("optimizer_steps", "optimizer_steps_by_level", "initial_parameter_sha256", "parameter_delta_l2"):
            self.assertEqual(old_report[key], new_report[key])
        expected = self.original["score_groups"](self.development, d05.MODEL_SPEC,
                                                  self.old_payload, self.train["meta"])
        actual = d05.score_groups(self.development, self.new_payload, self.train["meta"])
        self.assertEqual(expected, actual)

    def test_original_bce_payload_still_loads_without_fit(self):
        with patch.object(GeometryBank, "fit", side_effect=AssertionError("Unexpected fit")), \
             patch.object(SharedVerifier, "fit", side_effect=AssertionError("Unexpected fit")):
            geometry, verifier = d05.validate_payload(self.old_payload, self.train["meta"], self.train)
        _assert_equal(self, geometry.state_dict(), self.old_payload["geometry"])
        _assert_equal(self, verifier.state_dict(), self.old_payload["verifier"])


if __name__ == "__main__":
    unittest.main()
