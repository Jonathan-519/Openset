"""CPU unit + real-artifact pipeline tests using SYNTHETIC features, not a GPU run."""
import contextlib
import copy
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from taxosafe_visual import core, pipeline, residual, runtime
from test_taxosafe_visual import toy_dataset


def fixture():
    meta, bank, rows, f, pc, lc = toy_dataset()
    bank["image_hashes"] = np.asarray([hashlib.sha256(("memory-" + str(i)).encode()).hexdigest() for i in range(len(bank["labels"]))])
    return meta, bank, rows, f, pc, lc


class ResidualTests(unittest.TestCase):
    def test_project_removes_only_parent_direction_and_preserves_magnitude(self):
        x = np.asarray([[3., 4., 0.], [1., 0., 0.]])
        np.testing.assert_allclose(residual.project(x, np.array([1., 0., 0.]), 1), [[0, 4, 0], [0, 0, 0]])
        np.testing.assert_allclose(residual.project(x, np.array([1., 0., 0.]), 0), x)

    def test_self_neighbours_excluded_from_radii(self):
        meta, bank, *_ = fixture()
        state = residual.build_state(bank["features"], bank["labels"], meta, 0, 1, 0)
        cls = state["classes"]["0"]
        f = core.normalise(bank["features"])[cls["indices"]]
        d = residual.distances(f, f); np.fill_diagonal(d, np.inf)
        np.testing.assert_allclose(cls["radii"], d.min(1))
        self.assertGreater(min(cls["radii"]), 1e-4)

    def test_support_removal_rebuilds_parent_axis_and_removes_held_leaf(self):
        meta, bank, *_ = fixture()
        y, x = bank["labels"], bank["features"]
        full = residual.build_state(x, y, meta, .5, 3)
        keep = y != 0
        held = residual.build_state(x[keep], y[keep], meta, .5, 3)
        self.assertNotIn("0", held["classes"])
        self.assertEqual(held["parents"]["0"]["leaves"], [1])
        self.assertFalse(np.allclose(full["parents"]["0"]["direction"], held["parents"]["0"]["direction"]))

    def test_parent_axis_is_leaf_balanced(self):
        meta, bank, *_ = fixture()
        state = residual.build_state(bank["features"], bank["labels"], meta, 1, 3)
        x, y = core.normalise(bank["features"]), bank["labels"]
        means = core.normalise(np.array([x[y == c].mean(0) for c in [0, 1]]))
        expected = core.normalise(means.mean(0)[None])[0]
        np.testing.assert_allclose(state["parents"]["0"]["direction"], expected)

    def test_training_partition_deterministic_disjoint_and_rejects_hash_duplicates(self):
        _, bank, *_ = fixture()
        a = residual.training_folds(bank); b = residual.training_folds(bank)
        np.testing.assert_array_equal(a, b)
        for c in np.unique(bank["labels"]):
            self.assertEqual(set(a[bank["labels"] == c]), {0, 1})
        bank["image_hashes"][0] = bank["image_hashes"][1]
        with self.assertRaises(ValueError):
            residual.training_folds(bank)

    def test_rare_classes_fail_loudly_instead_of_reusing_query_as_support(self):
        meta, bank, *_ = fixture()
        bank = {k: v[:3] for k, v in bank.items() if k in ("features", "labels", "image_hashes")}
        with self.assertRaisesRegex(ValueError, ">=4"):
            residual.training_folds(bank)

    def test_selection_is_train_only_and_deterministic(self):
        meta, bank, *_ = fixture()
        settings = {"alphas": [0, 1], "ks": [1], "folds": 2}
        a = residual.fit(bank, meta, settings)
        b = residual.fit(bank, meta, settings)
        self.assertEqual(a, b)
        self.assertEqual(a["selection"]["source"], "train_only_crossfit_leave_one_species_out")
        self.assertEqual(a["selection"]["chosen"]["episode_count"], 8)
        json.dumps(a, allow_nan=False)

    def test_scores_are_chunk_invariant_and_remain_inside_parent(self):
        meta, bank, rows, f, *_ = fixture()
        state = residual.build_state(bank["features"], bank["labels"], meta, .5, 3)
        parents = np.arange(len(f)) % 3
        a = residual.score(f, bank, state, parents, 1)
        b = residual.score(f, bank, state, parents, 128)
        np.testing.assert_allclose(a["score"], b["score"], atol=1e-10)
        np.testing.assert_array_equal(a["leaf"], b["leaf"])
        self.assertTrue(np.all(np.asarray(meta["leaf_to_parent"])[a["leaf"]] == parents))

    def test_singleton_has_absolute_support_score(self):
        meta, bank, *_ = fixture()
        state = residual.build_state(bank["features"], bank["labels"], meta, 1, 3)
        q = np.stack([bank["features"][bank["labels"] == 4][0], np.eye(12)[11]])
        out = residual.score(q, bank, state, [2, 2])
        self.assertEqual(out["leaf"].tolist(), [4, 4])
        self.assertGreater(out["score"][0], out["score"][1])

    def test_root_is_unchanged_and_all_child_output_fields_consistent(self):
        meta, bank, rows, f, pc, lc = fixture()
        support = core.retrieve(f, bank, meta)
        fit, cal = core.validation_partition(rows)
        routing, _ = core.routing_fit(pc, lc, support, meta, 100, rows, fit, {})
        e = core.evidence(pc, lc, support, meta, 100, routing)
        base_cal = core.fit_calibration(e, rows, meta, fit, cal, {})
        state = residual.build_state(bank["features"], bank["labels"], meta, .5, 3)
        out = residual.score(f, bank, state, e["pred_parent"])
        rs_cal = residual.calibrate(out, rows, meta, cal, {})
        for profile in ("risk", "coverage"):
            baseline = core.predict(e, rows, base_cal, profile)
            new = pipeline._refine(baseline, out, rs_cal, meta, profile)
            for a, b in zip(baseline, new):
                self.assertEqual(a["root_knownness_score"], b["root_knownness_score"])
                self.assertEqual(a["prediction_type"] == "global_unknown", b["prediction_type"] == "global_unknown")
                self.assertEqual(b["pred_leaf"], b["candidate_leaf"])
                self.assertEqual(b["leaf_name"], None if b["leaf"] is None else meta["leaf_names"][b["leaf"]])
            forged = copy.deepcopy(baseline)
            for row in forged:
                row.update(status="extra", true_parent=None, true_leaf=None, source="forged")
            forged = residual.apply(forged, out, rs_cal, meta, profile)
            self.assertEqual([(r["prediction_type"], r["leaf"]) for r in new], [(r["prediction_type"], r["leaf"]) for r in forged])

    def test_calibration_reads_only_held_partition_and_flags_missing_branch(self):
        meta, bank, rows, f, pc, lc = fixture()
        state = residual.build_state(bank["features"], bank["labels"], meta, 0, 1)
        parents = np.array([r["true_parent"] if r["true_parent"] is not None else 0 for r in rows])
        out = residual.score(f, bank, state, parents)
        fit, cal = core.validation_partition(rows)
        a = residual.calibrate(out, rows, meta, cal, {})
        b_rows = copy.deepcopy(rows)
        for i in fit:
            b_rows[i].update(status="extra", true_parent=None, true_leaf=None, source="changed")
        b = residual.calibrate(out, b_rows, meta, cal, {})
        self.assertEqual(a, b)
        self.assertEqual(a["branches"]["C"]["risk_threshold_source"], "pooled_no_branch_negatives")
        self.assertIsNone(a["branches"]["C"]["coverage_and_risk_feasible_on_calibration"])

    def test_infeasible_risk_and_coverage_are_reported(self):
        meta = {"parent_names": ["A"], "leaf_names": ["a"], "leaf_to_parent": [0]}
        rows = [{"status": "known", "true_leaf": 0, "true_parent": 0, "source": "a"}] * 10
        rows += [{"status": "intra", "true_leaf": None, "true_parent": 0, "source": "u"}] * 10
        out = {"score": np.zeros(20), "parent": np.zeros(20, int)}
        cal = residual.calibrate(out, rows, meta, np.arange(20), {})
        self.assertFalse(cal["branches"]["A"]["coverage_and_risk_feasible_on_calibration"])

    def test_source_balanced_threshold_is_group_balanced_and_deterministic(self):
        scores = np.asarray([.90, .80, .70, .60, .10, .20, .55, .65])
        positive = np.asarray([1, 1, 1, 1, 0, 0, 0, 0], dtype=bool)
        negative = ~positive
        leaves = np.asarray([0, 0, 1, 1, -1, -1, -1, -1])
        sources = np.asarray(["a", "a", "b", "b", "u0", "u0", "u1", "u1"])
        tau, report = residual.source_balanced_threshold(
            scores, positive, leaves, negative, sources)
        # Replicating only the easy and already larger source must not give it
        # more weight than the hard source.
        extra = np.asarray([4, 5] * 20)
        tau_repeated, repeated = residual.source_balanced_threshold(
            np.r_[scores, scores[extra]], np.r_[positive, positive[extra]],
            np.r_[leaves, leaves[extra]], np.r_[negative, negative[extra]],
            np.r_[sources, sources[extra]])
        self.assertEqual(tau, tau_repeated)
        self.assertEqual(report["utility"], repeated["utility"])
        self.assertEqual(report["known_leaf_groups"], 2)
        self.assertEqual(report["intra_source_groups"], 2)

    def test_balanced_profile_uses_correctly_routed_intra_and_coverage_backoff(self):
        meta = {"parent_names": ["A", "B"], "leaf_names": ["a", "b"],
                "leaf_to_parent": [0, 1]}
        rows = ([{"status": "known", "true_leaf": 0, "true_parent": 0, "source": "a"}] * 3 +
                [{"status": "known", "true_leaf": 1, "true_parent": 1, "source": "b"}] * 3 +
                [{"status": "intra", "true_leaf": None, "true_parent": 0, "source": "u"}] * 3 +
                [{"status": "intra", "true_leaf": None, "true_parent": 0, "source": "wrong"}])
        out = {"score": np.asarray([.9, .8, .7, .9, .8, .7, .1, .2, .3, .99]),
               "parent": np.asarray([0, 0, 0, 1, 1, 1, 0, 0, 0, 1]),
               "leaf": np.asarray([0, 0, 0, 1, 1, 1, 0, 0, 0, 1])}
        settings = {"residual": {"threshold_mode": "hierarchical_shrinkage",
                                   "threshold_shrinkage": "auto",
                                   "balanced_profile": True}}
        cal = residual.calibrate(out, rows, meta, np.arange(len(rows)), settings)
        self.assertIn("balanced", cal["profiles"])
        self.assertEqual(cal["branches"]["A"]["intra_count_correctly_routed_here"], 3)
        self.assertEqual(cal["branches"]["A"]["balanced_threshold_source"],
                         "parent_leaf_and_source_macro_utility")
        self.assertEqual(cal["branches"]["B"]["balanced_threshold_source"],
                         "coverage_backoff_no_routed_intra_source")
        self.assertIsNotNone(cal["branches"]["A"]["balanced_calibration"])
        self.assertIsNone(cal["branches"]["B"]["balanced_calibration"])

    def test_hierarchical_thresholds_shrink_rare_leaf_to_parent_and_use_candidate_leaf(self):
        meta = {"parent_names": ["A"], "leaf_names": ["common", "rare"],
                "leaf_to_parent": [0, 0]}
        common = [.10, .20, .30, .40, .50, .60, .70, .80, .90, 1.00]
        rare = [.05, .15]
        rows = ([{"status": "known", "true_leaf": 0, "true_parent": 0, "source": "common"}
                 for _ in common] +
                [{"status": "known", "true_leaf": 1, "true_parent": 0, "source": "rare"}
                 for _ in rare] +
                [{"status": "known", "true_leaf": 1, "true_parent": 0, "source": "rare"}] +
                [{"status": "intra", "true_leaf": None, "true_parent": 0, "source": "novel"}
                 for _ in range(10)])
        scores = np.asarray(common + rare + [-5.] + [.01] * 10)
        # The -5 known item is routed to the wrong candidate leaf. Lowering a
        # threshold cannot make it correct, so leaf-aware calibration excludes it.
        leaves = np.asarray([0] * len(common) + [1] * len(rare) + [0] + [0] * 10)
        out = {"score": scores, "parent": np.zeros(len(rows), int), "leaf": leaves}
        settings = {"residual": {"threshold_mode": "hierarchical_shrinkage",
                                  "threshold_shrinkage": "auto"}}
        cal = residual.calibrate(out, rows, meta, np.arange(len(rows)), settings)
        self.assertEqual(cal["threshold_mode"], "hierarchical_shrinkage")
        self.assertEqual(cal["threshold_shrinkage"], 6.0)  # median of counts 10 and 2
        self.assertEqual(cal["branches"]["A"]["known_counts_correct_parent"]["rare"], 3)
        self.assertEqual(cal["branches"]["A"]["known_counts_correct_leaf"]["rare"], 2)
        rare_tau = residual.threshold(cal, "coverage", 0, 1)
        parent_tau = core.coverage_threshold(scores, np.asarray(
            [True] * 12 + [False] * 11), .10)
        self.assertGreater(rare_tau, .05)
        self.assertLess(rare_tau, parent_tau)
        self.assertNotEqual(residual.threshold(cal, "coverage", 0, 0), rare_tau)

    def test_leaf_aware_apply_is_truth_independent_and_preserves_root(self):
        meta, bank, rows, f, pc, lc = fixture()
        support = core.retrieve(f, bank, meta)
        fit, held = core.validation_partition(rows)
        routing, _ = core.routing_fit(pc, lc, support, meta, 100, rows, fit, {})
        evidence = core.evidence(pc, lc, support, meta, 100, routing)
        base_cal = core.fit_calibration(evidence, rows, meta, fit, held, {})
        state = residual.build_state(bank["features"], bank["labels"], meta, 0, 1,
                                     local_scaling=False)
        out = residual.score(f, bank, state, evidence["pred_parent"])
        settings = {"residual": {"threshold_mode": "hierarchical_shrinkage",
                                  "threshold_shrinkage": "auto"}}
        cal = residual.calibrate(out, rows, meta, held, settings)
        baseline = core.predict(evidence, rows, base_cal, "coverage")
        expected = residual.apply(baseline, out, cal, meta, "coverage")
        forged = copy.deepcopy(baseline)
        for row in forged:
            row.update(status="extra", true_parent=None, true_leaf=None, source="forged")
        actual = residual.apply(forged, out, cal, meta, "coverage")
        for a, b, original in zip(actual, expected, baseline):
            self.assertEqual((a["prediction_type"], a["leaf"], a["residual_child_threshold"]),
                             (b["prediction_type"], b["leaf"], b["residual_child_threshold"]))
            self.assertEqual(a["root_knownness_score"], original["root_knownness_score"])

    def test_invalid_threshold_settings_fail_loudly(self):
        meta = {"parent_names": ["A"], "leaf_names": ["a"], "leaf_to_parent": [0]}
        rows = ([{"status": "known", "true_leaf": 0, "true_parent": 0, "source": "a"}] * 2 +
                [{"status": "intra", "true_leaf": None, "true_parent": 0, "source": "u"}] * 2)
        out = {"score": np.zeros(4), "parent": np.zeros(4, int), "leaf": np.zeros(4, int)}
        for value in ({"threshold_mode": "mystery"},
                      {"threshold_mode": "hierarchical_shrinkage", "threshold_shrinkage": -1}):
            with self.assertRaises(ValueError):
                residual.calibrate(out, rows, meta, np.arange(4), {"residual": value})

    def test_diagnostics_handle_no_unknowns(self):
        r = [{"status": "known", "true_parent": 0, "candidate_parent": 0,
              "prediction_type": "known", "child_knownness_score": 1}]
        self.assertIsNone(residual.diagnostics(r)["all"]["auroc"])

    def test_three_stage_partial_pooling_pipeline_provenance_and_paired_baseline(self):
        meta, bank, rows, f, pc, lc = fixture()
        calls = []

        def fake_extract(cfg, split, model, texts, hierarchy, device):
            calls.append(split)
            if split == "train":
                train = [{"status": "known", "true_leaf": int(c), "true_parent": meta["leaf_to_parent"][c],
                          "source": meta["leaf_names"][c], "path": "train-" + str(i),
                          "image_sha256": str(bank["image_hashes"][i])} for i, c in enumerate(bank["labels"])]
                return train, bank["features"], np.zeros((len(train), 3)), np.zeros((len(train), 5))
            ids = [i for i, r in enumerate(rows) if r["status"] == split.split("_")[1]]
            subset = copy.deepcopy([rows[i] for i in ids])
            for row in subset:
                row["image_sha256"] = hashlib.sha256((split + row["image_sha256"]).encode()).hexdigest()
            return subset, f[ids], pc[ids], lc[ids]

        with tempfile.TemporaryDirectory(prefix="taxosafe_rs_test_") as directory:
            root = Path(directory)
            ckpt = root / "best.pth"; ckpt.write_bytes(b"SYNTHETIC CHECKPOINT")
            splits = ["train", "val_known", "val_intra", "val_extra", "test_known", "test_intra", "test_extra"]
            paths = {}
            for split in splits:
                path = root / (split + ".txt"); path.write_text("synthetic manifest")
                paths[split] = str(path)
            cfg = {"data": paths, "visual_support": {
                "primary_profile": "coverage", "residual": {
                    "enabled": True, "alphas": [0], "ks": [1], "local_scaling": False,
                    "threshold_mode": "hierarchical_shrinkage", "threshold_shrinkage": "auto",
                    "balanced_profile": True}}}
            args = SimpleNamespace(overwrite=False, profiles="all")
            with patch.object(pipeline, "arguments", return_value=args), \
                 patch.object(pipeline, "_setup", return_value=(cfg, root, ckpt, root)), \
                 patch.object(pipeline, "load_model", return_value=(None, None, meta, None, 100.)), \
                 patch.object(pipeline, "extract", side_effect=fake_extract), \
                 patch.object(pipeline, "extraction_signature", return_value="fixture"), \
                 patch.object(runtime, "extraction_signature", return_value="fixture"), \
                 contextlib.redirect_stdout(io.StringIO()):
                pipeline.build_memory()
                self.assertEqual(calls, ["train"])
                state = json.loads((root / "residual_state.json").read_text())
                pipeline.calibrate()
                self.assertEqual(calls, splits[:4])
                cal = json.loads((root / "calibration.json").read_text())
                self.assertEqual(cal["schema_version"], 7)
                self.assertEqual(cal["residual_calibration"]["threshold_mode"],
                                 "hierarchical_shrinkage")
                self.assertIn("balanced", cal["residual_calibration"]["profiles"])
                pipeline.test()
                self.assertEqual(calls, splits)
                report = json.loads((root / "test/comparison.json").read_text())
                self.assertTrue(report["root_decisions_identical"])
                for profile in ("risk", "balanced", "coverage"):
                    self.assertEqual(report["profiles"][profile]["extra_far"], report["paired_baseline_v4"][profile]["extra_far"])
                    self.assertTrue((root / "test" / profile / "baseline_v4_predictions.jsonl").is_file())
                    self.assertTrue((root / "test" / profile / "matched_v4_predictions.jsonl").is_file())
                    self.assertEqual(report["profiles"][profile]["extra_far"], report["matched_calibration_v4"][profile]["extra_far"])
                with self.assertRaises(FileExistsError):
                    pipeline.test()
                state["state"]["alpha"] = .123
                (root / "residual_state.json").write_text(json.dumps(state))
                with self.assertRaisesRegex(ValueError, "changed after calibration"):
                    pipeline.test()
                self.assertEqual(calls, splits)  # Failure before loading test images.

    def test_archived_config_must_match_run_directory(self):
        import yaml
        with tempfile.TemporaryDirectory(prefix="taxosafe_archive_") as directory:
            root = Path(directory)
            run = root / "run"; (run / "ckpt").mkdir(parents=True)
            (run / "ckpt/best.pth").write_bytes(b"fixture")
            cfg = {"data": {"name": "data"}, "model": {"arch": "maple"}, "exp": "test"}
            wrong = root / "training.yml"; wrong.write_text(yaml.safe_dump(cfg))
            extension = root / "extension.yml"
            settings = {"base_config": str(wrong), "require_archived_training_config": True}
            extension.write_text(yaml.safe_dump(settings))
            with self.assertRaisesRegex(ValueError, "archived YAML"):
                runtime.load_configuration(str(extension), "1", str(run))
            archived = run / "training.yml"; archived.write_text(yaml.safe_dump(cfg))
            settings["base_config"] = str(archived); extension.write_text(yaml.safe_dump(settings))
            loaded, _, _ = runtime.load_configuration(str(extension), "1", str(run))
            self.assertEqual(loaded["visual_support"]["archived_training_config_sha256"], runtime.sha256(archived))


if __name__ == "__main__":
    unittest.main()
