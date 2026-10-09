"""Independent semantic tests for boundary evidence and matched-support learning.

The tiny tensors exercise actual optimization; they are not performance claims
for the deployed data. No historical source module is changed by these tests.
"""
import copy
import math
import unittest
from unittest import mock

import torch
from torch.nn import functional as F

from taxosafe_discovery.geometry import GeometryBank, evidence_features
from taxosafe_discovery.verifier import SharedVerifier, build_episodes as old_episodes
from taxosafe_boundary import core
from tests.test_taxosafe_discovery_geometry import fixture


LEVELS = ("leaf", "parent")
MODES = ("bce8", "rank8", "bce9", "rank9")


def tensor_leaves(value):
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from tensor_leaves(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from tensor_leaves(child)


class BoundaryBankTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.args = fixture()

    def test_class_scores_are_finite_and_witnesses_are_class_specific(self):
        fine, parent, labels, hashes, meta = self.args
        bank = core.BoundaryBank.fit(*self.args, tail_size=32)
        scores = bank.score(fine, parent, ["new-" + h for h in hashes])
        self.assertEqual(scores["leaf_scores"].shape, (24, 4))
        self.assertEqual(scores["parent_scores"].shape, (24, 3))
        for level, width in (("leaf", 4), ("parent", 3)):
            self.assertEqual(scores[level + "_active"].shape, (width,))
            self.assertTrue(bool(scores[level + "_active"].all()))
            self.assertTrue(bool(torch.isfinite(scores[level + "_scores"]).all()))
        # A query at a stored witness's coordinates has maximal class inclusion;
        # hashes differ, so this is a geometric identity test, not TRAIN fitting.
        self.assertTrue(torch.equal(scores["leaf_scores"].argmax(1), labels))
        query_fine, query_parent = fine.double() + .03, parent.double() + .02
        ordinary = bank.score(query_fine, query_parent, ["offset-" + h for h in hashes])
        scaled = bank.score(3. * query_fine, 7. * query_parent, ["scaled-" + h for h in hashes])
        self.assertTrue(torch.allclose(ordinary["leaf_scores"], scaled["leaf_scores"], atol=2e-5, rtol=0.))

    def test_weibull_tail_mle_and_max_witness_score_match_independent_calculation(self):
        fine, parent, labels, hashes, meta = self.args
        bank = core.BoundaryBank.fit(*self.args, tail_size=32)
        query_fine, query_parent = fine[:3].double() + .17, parent[:3].double() + .11
        scored = bank.score(query_fine, query_parent, ["formula-a", "formula-b", "formula-c"])
        for level, query, support, y in (
                ("leaf", query_fine, bank.fine, labels),
                ("parent", query_parent, bank.parent, torch.tensor(meta["leaf_to_parent"])[labels])):
            shape, scale = bank.parameters[level]["shape"], bank.parameters[level]["scale"]
            self.assertTrue(bool(((shape >= .2) & (shape <= 10.)).all()))
            self.assertTrue(bool((scale > 0).all()))
            # This computes pair distances directly in D-space, independently of
            # core's norm/dot-product implementation and vectorized MLE groups.
            for witness in range(len(support)):
                negatives = support[y != y[witness]]
                tail = ((negatives - support[witness]).norm(dim=1) * .5).sort().values[:32]
                k = float(shape[witness]); lam = float(scale[witness])
                if bool((tail <= 1e-12).any()) or float(tail.max() - tail.min()) <= 1e-12 or len(tail) < 2:
                    self.assertEqual(k, 1.)
                    self.assertAlmostEqual(lam, float(tail.clamp_min(1e-12).mean()), places=10)
                else:
                    logx = tail.log()
                    score_equation = 1. / k + float(logx.mean()) - float((torch.softmax(k * logx, dim=0) * logx).sum())
                    if .200001 < k < 9.999999:
                        self.assertAlmostEqual(score_equation, 0., places=8)
                    elif k >= 9.999999:
                        self.assertGreaterEqual(score_equation, -1e-9)
                    else:
                        self.assertLessEqual(score_equation, 1e-9)
                    expected_scale = float(tail.pow(k).mean().pow(1. / k))
                    self.assertAlmostEqual(lam, expected_scale, places=10)
            unit = query / query.norm(dim=1, keepdim=True)
            distances = (unit[:, None, :] - support[None, :, :]).norm(dim=2).clamp_min(1e-12)
            per_witness = -shape[None, :] * torch.log(distances / scale[None, :])
            expected = torch.stack([per_witness[:, y == c].max(dim=1).values
                                    for c in range(len(meta[level + "_names"]))], dim=1)
            self.assertTrue(torch.allclose(scored[level + "_scores"], expected, atol=1e-9, rtol=0.))

    def test_query_exclusion_and_absent_candidate_masks(self):
        fine, parent, labels, hashes, meta = self.args
        bank = core.BoundaryBank.fit(*self.args)
        with self.assertRaisesRegex(ValueError, "overlap"):
            bank.score(fine[:1], parent[:1], hashes[:1])
        mask = labels != 1
        kept_hashes = [h for h, keep in zip(hashes, mask) if keep]
        bank = core.BoundaryBank.fit(fine[mask], parent[mask], labels[mask], kept_hashes, meta)
        scores = bank.score(fine[~mask], parent[~mask], [h for h, keep in zip(hashes, mask) if not keep])
        self.assertFalse(bool(scores["leaf_active"][1]))
        self.assertTrue(bool(scores["parent_active"][0]))
        self.assertTrue(bool((scores["leaf_scores"][:, 1] == -1e6).all()))

    def test_loading_is_exact_and_never_estimates_new_statistics(self):
        bank = core.BoundaryBank.fit(*self.args)
        state = bank.state_dict()
        expected = bank.score(self.args[0][:3], self.args[1][:3], ["query-1", "query-2", "query-3"])
        with mock.patch.object(core.BoundaryBank, "fit", side_effect=AssertionError("TEST refits boundary")), \
                mock.patch.object(core.BoundaryBank, "_fit_precomputed", side_effect=AssertionError("TEST estimates tail")), \
                mock.patch.object(core, "_fit_weibull", side_effect=AssertionError("TEST estimates Weibull")), \
                mock.patch.object(GeometryBank, "fit", side_effect=AssertionError("TEST refits geometry")):
            restored = core.BoundaryBank.from_state_dict(state)
            actual = restored.score(self.args[0][:3], self.args[1][:3], ["query-1", "query-2", "query-3"])
        self.assertEqual(expected.keys(), actual.keys())
        self.assertTrue(all(torch.equal(expected[k], actual[k]) for k in expected))
        changed = copy.deepcopy(state)
        value = next(x for x in tensor_leaves(changed) if x.is_floating_point() and x.numel())
        value.reshape(-1)[0] += .01
        with self.assertRaises(ValueError):
            core.BoundaryBank.from_state_dict(changed)

    def test_bad_inputs_and_conflicting_content_are_rejected(self):
        fine, parent, labels, hashes, meta = self.args
        invalid = [dict(tail_size=0), dict(tail_size=True), dict(tail_size=1.5)]
        for options in invalid:
            with self.subTest(options=options), self.assertRaises(ValueError):
                core.BoundaryBank.fit(*self.args, **options)
        for args in ((fine, parent, labels, [hashes[0]] * len(hashes), meta),
                     (fine * float("nan"), parent, labels, hashes, meta),
                     (fine, parent[:-1], labels, hashes, meta),
                     (fine, parent, labels.float() + .2, hashes, meta)):
            with self.assertRaises(ValueError):
                core.BoundaryBank.fit(*args)

    def test_coincident_interclass_vectors_have_finite_fitted_scores(self):
        fine, parent, labels, hashes, meta = self.args
        fine = fine.clone(); parent = parent.clone()
        fine[6] = fine[0]; parent[6] = parent[0]
        bank = core.BoundaryBank.fit(fine, parent, labels, hashes, meta, tail_size=100)
        scores = bank.score(fine[:2], parent[:2], ["coincident-1", "coincident-2"])
        self.assertTrue(all(bool(torch.isfinite(v).all()) for v in tensor_leaves(bank.state_dict()) if v.is_floating_point()))
        self.assertTrue(all(bool(torch.isfinite(scores[level + "_scores"]).all()) for level in LEVELS))


class BoundaryEpisodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.args = fixture()
        cls.text = {"leaf": cls.args[0] @ torch.eye(4), "parent": cls.args[1] @ torch.eye(3)}
        cls.episodes = core.build_episodes(*cls.args, template_scores=cls.text, folds=3, seed=1)

    def test_every_bank_excludes_all_queries_and_withheld_support(self):
        fine, parent, labels, hashes, meta = self.args
        report = self.episodes["report"]
        self.assertEqual(report["fit_split"], "known_train")
        self.assertEqual(len(report["episodes"]), 3 * (1 + 4 + 3))
        for flag in ("true_unknown_images_used", "dev_images_used", "test_images_used"):
            self.assertIs(report[flag], False)
        mapping = torch.tensor(meta["leaf_to_parent"])
        full_queries = {}
        for e in report["episodes"]:
            support, query = set(e["support_indices"]), set(e["query_indices"])
            self.assertFalse(support & query)
            self.assertEqual(e["query_support_overlap"], 0)
            self.assertGreater(len(support), 0)
            self.assertGreater(len(query), 0)
            self.assertFalse(set(e["active_leaf_ids"]) & set(e["withheld_leaf_ids"]))
            self.assertFalse(set(e["active_parent_ids"]) & set(e["withheld_parent_ids"]))
            self.assertFalse({int(labels[i]) for i in support} & set(e["withheld_leaf_ids"]))
            self.assertFalse({int(mapping[labels[i]]) for i in support} & set(e["withheld_parent_ids"]))
            self.assertEqual(set(e["active_leaf_ids"]), {int(labels[i]) for i in support})
            self.assertEqual(set(e["active_parent_ids"]), {int(mapping[labels[i]]) for i in support})
            full_queries.setdefault(e["fold"], query)
            self.assertEqual(query, full_queries[e["fold"]])
        self.assertEqual(set.union(*full_queries.values()), set(range(len(labels))))
        self.assertEqual(sum(map(len, full_queries.values())), len(labels))

    def test_labels_match_candidate_truth_and_singleton_parent_has_no_false_positive(self):
        labels, meta = self.args[2], self.args[4]
        mapping = torch.tensor(meta["leaf_to_parent"])
        banks = {e["bank_id"]: e for e in self.episodes["report"]["episodes"]}
        near_parent_positives = 0
        singleton_withheld_negatives = 0
        for level in LEVELS:
            data = self.episodes[level]
            self.assertEqual(data["x"].shape[1], 9)
            self.assertTrue(bool(torch.isfinite(data["x"]).all()))
            self.assertAlmostEqual(float(data["weight"][data["y"] == 1].sum()), .5, places=6)
            self.assertAlmostEqual(float(data["weight"][data["y"] == 0].sum()), .5, places=6)
            for i in range(len(data["y"])):
                query = int(data["query_index"][i]); candidate = int(data["candidate"][i])
                bank = banks[int(data["bank_id"][i])]
                truth = int(labels[query] if level == "leaf" else mapping[labels[query]])
                self.assertEqual(float(data["y"][i]), float(candidate == truth))
                self.assertIn(candidate, bank["active_" + level + "_ids"])
                self.assertEqual(int(data["source_leaf"][i]), int(labels[query]))
                self.assertIn(query, bank["query_indices"])
                self.assertEqual(bool(data["withheld"][i]), truth not in bank["active_" + level + "_ids"])
                if level == "parent" and bank["kind"] == "drop_leaf" and int(labels[query]) in bank["withheld_leaf_ids"]:
                    if truth == 0 and candidate == 0:
                        near_parent_positives += 1
                        self.assertEqual(float(data["y"][i]), 1.)
                    elif truth in (1, 2):
                        singleton_withheld_negatives += 1
                        self.assertEqual(float(data["y"][i]), 0.)
        self.assertGreater(near_parent_positives, 0)
        self.assertGreater(singleton_withheld_negatives, 0)

    def test_fits_receive_exact_reported_support_and_scores_receive_full_query_fold(self):
        fitted, scored = [], []
        original_fit, original_score = core.BoundaryBank._fit_precomputed, core.BoundaryBank.score
        def fit(*args, **kwargs):
            fitted.append(list(args[3]))
            return original_fit(*args, **kwargs)
        def score(bank, fine, parent, hashes):
            scored.append(list(hashes))
            return original_score(bank, fine, parent, hashes)
        with mock.patch.object(core.BoundaryBank, "_fit_precomputed", side_effect=fit), \
                mock.patch.object(core.BoundaryBank, "score", new=score):
            episodes = core.build_episodes(*self.args, template_scores=self.text, folds=3, seed=1)
        reported = episodes["report"]["episodes"]
        self.assertEqual(len(fitted), len(reported))
        self.assertEqual(len(scored), len(reported))
        for i, e in enumerate(reported):
            self.assertEqual(fitted[i], [self.args[3][j] for j in e["support_indices"]])
            self.assertEqual(scored[i], [self.args[3][j] for j in e["query_indices"]])
            self.assertFalse(set(fitted[i]) & set(scored[i]))


class BoundaryVerifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.args = fixture()
        cls.text = {"leaf": cls.args[0] @ torch.eye(4), "parent": cls.args[1] @ torch.eye(3)}
        legacy = old_episodes(*cls.args, template_scores=cls.text)
        cls.source = SharedVerifier.fit(legacy, loss="bce", epochs=2, batch_size=128).state_dict()
        cls.episodes = core.build_episodes(*cls.args, template_scores=cls.text, folds=3, seed=1)
        cls.models = {mode: core.BoundaryVerifier.fit(cls.source, cls.episodes, mode=mode,
            seed=7, steps=4, batch_size=64, lr=.001) for mode in MODES}

    def test_zero_added_column_preserves_source_function_before_optimization(self):
        source = SharedVerifier.from_state_dict(self.source)
        for level in LEVELS:
            old = source.heads[level]
            x8 = torch.linspace(-3., 3., 19 * 8).reshape(19, 8)
            x9 = torch.cat((x8, torch.linspace(-100., 100., len(x8))[:, None]), dim=1)
            head = core._head_from_source(self.source["heads"][level], 9, self.source["hidden"])
            self.assertTrue(torch.equal(head.layers[0].weight[:, :8], old.layers[0].weight))
            self.assertTrue(bool((head.layers[0].weight[:, 8] == 0).all()))
            self.assertTrue(torch.equal(head(x9), old(x8)))

    def test_four_arms_have_matched_source_norm_batches_budget_and_nonzero_updates(self):
        reports = [m.fit_report for m in self.models.values()]
        self.assertTrue(all(r["source_normalization_sha256"] == reports[0]["source_normalization_sha256"] for r in reports))
        self.assertTrue(all(r["source_head_sha256"] == reports[0]["source_head_sha256"] for r in reports))
        self.assertTrue(all(r["base_batch_sha256"] == reports[0]["base_batch_sha256"] for r in reports))
        for mode, model in self.models.items():
            report = model.fit_report
            self.assertEqual(report["optimizer_steps"], 8)
            self.assertEqual(report["optimizer_steps_by_level"], {"leaf": 4, "parent": 4})
            self.assertTrue(report["normalization_first_eight_unchanged"])
            self.assertEqual(report["initial_teacher_max_abs_difference"], 0.)
            for level in LEVELS:
                self.assertTrue(torch.equal(model.normalization[level]["mean"][:8], self.source["normalization"][level]["mean"]))
                self.assertTrue(torch.equal(model.normalization[level]["scale"][:8], self.source["normalization"][level]["scale"]))
                self.assertGreater(report["parameter_delta_l2"][level], 0.)
                self.assertEqual(len(report["history"][level]), 4)
                self.assertTrue(all(r["gradnorm"] > 0 and math.isfinite(r["gradnorm"]) for r in report["history"][level]))
                ranks = [r["rank"] for r in report["history"][level]]
                self.assertTrue(all(r > 0 if mode.startswith("rank") else r == 0 for r in ranks))
                if mode.endswith("9"):
                    self.assertTrue(bool((model.heads[level].layers[0].weight[:, 8] != 0).any()))
        for dimension in (8, 9):
            self.assertFalse(torch.equal(self.models["bce" + str(dimension)].heads["leaf"].layers[0].weight,
                                         self.models["rank" + str(dimension)].heads["leaf"].layers[0].weight))

    def test_tail_rank_pairs_have_live_gradient_and_correct_cross_query_semantics(self):
        source = SharedVerifier.from_state_dict(self.source)
        for level in LEVELS:
            data = self.episodes[level]
            groups = core._ranking_groups(data)
            head = core._head_from_source(self.source["heads"][level], 8, self.source["hidden"])
            head.requires_grad_(True)
            x = (data["x"][:, :8] - source.normalization[level]["mean"]) / source.normalization[level]["scale"]
            pairs = core._tail_pairs(head, x, groups, torch.Generator().manual_seed(3), pair_budget=256, tail_fraction=.2)
            self.assertGreater(len(pairs), 0)
            a, b = pairs[:, 0], pairs[:, 1]
            self.assertTrue(bool((data["y"][a] == 1).all() & (data["y"][b] == 0).all()))
            self.assertTrue(torch.equal(data["bank_id"][a], data["bank_id"][b]))
            self.assertTrue(torch.equal(data["candidate"][a], data["candidate"][b]))
            self.assertTrue(bool((data["query_index"][a] != data["query_index"][b]).all()))
            loss = F.softplus(.2 - head(x[a]) + head(x[b])).mean()
            gradient = torch.autograd.grad(loss, head.layers[0].weight)[0]
            self.assertTrue(bool(torch.isfinite(gradient).all()))
            self.assertGreater(float(gradient.abs().sum()), 0.)

    def test_tail_pair_mining_uses_weak_positive_and_hard_withheld_negative(self):
        # Ten positive and ten negative queries share one support bank/candidate.
        # The final two negatives are held out; other wrong-class negatives have
        # deliberately higher scores but must not displace that preferred pool.
        y = torch.cat((torch.ones(10), torch.zeros(10)))
        data = dict(y=y, bank_id=torch.zeros(20, dtype=torch.long),
            candidate=torch.zeros(20, dtype=torch.long), query_index=torch.arange(20),
            source_leaf=torch.cat((torch.zeros(10, dtype=torch.long), torch.ones(10, dtype=torch.long))),
            withheld=torch.tensor([False] * 18 + [True, True]))
        x = torch.zeros(20, 8)
        x[:10, 0] = torch.arange(10.)
        x[10:18, 0] = 100.
        x[18:, 0] = torch.tensor([3., 4.])
        head = torch.nn.Linear(8, 1, bias=False)
        with torch.no_grad():
            head.weight.zero_(); head.weight[0, 0] = 1.
        class ScalarHead(torch.nn.Module):
            def forward(self, values):
                return head(values).squeeze(-1)
        groups = core._ranking_groups(data)
        pairs = core._tail_pairs(ScalarHead(), x, groups, torch.Generator().manual_seed(5),
                                 pair_budget=16, tail_fraction=.2)
        self.assertGreater(len(pairs), 0)
        self.assertTrue(set(pairs[:, 0].tolist()) <= {0, 1})
        self.assertEqual(set(pairs[:, 1].tolist()), {19})
        data["query_index"][18] = 0
        with self.assertRaises(ValueError):
            core._ranking_groups(data)

    def test_train_and_reload_preserve_cpu_rng_and_source_bytes(self):
        source = copy.deepcopy(self.source)
        torch.manual_seed(97); before = torch.get_rng_state().clone()
        model = core.BoundaryVerifier.fit(source, self.episodes, mode="rank9", seed=3, steps=2, batch_size=64)
        self.assertTrue(torch.equal(torch.get_rng_state(), before))
        for level in LEVELS:
            for name in self.source["heads"][level]:
                self.assertTrue(torch.equal(source["heads"][level][name], self.source["heads"][level][name]))
        core.BoundaryVerifier.from_state_dict(model.state_dict())
        self.assertTrue(torch.equal(torch.get_rng_state(), before))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA RNG preservation requires CUDA")
    def test_initialized_cuda_rng_is_preserved(self):
        torch.cuda.manual_seed_all(123)
        before = [value.clone() for value in torch.cuda.get_rng_state_all()]
        model = core.BoundaryVerifier.fit(self.source, self.episodes, mode="rank9", seed=9,
                                           steps=1, batch_size=64)
        core.BoundaryVerifier.from_state_dict(model.state_dict())
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, torch.cuda.get_rng_state_all())))

    def test_frozen_inference_roundtrip_does_not_fit_or_train(self):
        geometry = GeometryBank.fit(*self.args)
        boundary = core.BoundaryBank.fit(*self.args)
        query_hashes = ["unseen-1", "unseen-2"]
        args = (self.args[0][:2], self.args[1][:2], query_hashes)
        geom, extra = geometry.score(*args), boundary.score(*args)
        text = {k: v[:2] for k, v in self.text.items()}
        for mode, model in self.models.items():
            expected = model.score(geom, text, extra)
            with mock.patch.object(core.BoundaryVerifier, "fit", side_effect=AssertionError("TEST trains verifier")), \
                    mock.patch.object(core.BoundaryBank, "fit", side_effect=AssertionError("TEST fits boundary")), \
                    mock.patch.object(GeometryBank, "fit", side_effect=AssertionError("TEST fits geometry")), \
                    mock.patch.object(core, "build_episodes", side_effect=AssertionError("TEST builds episodes")):
                loaded = core.BoundaryVerifier.from_state_dict(model.state_dict())
                actual = loaded.score(geom, text, extra)
            self.assertTrue(all(torch.equal(expected[k], actual[k]) for k in expected))
            self.assertFalse(any(p.requires_grad for h in loaded.heads.values() for p in h.parameters()))
            if mode.endswith("9"):
                with self.assertRaises(ValueError):
                    loaded.score(geom, text)

    def test_unknown_dev_test_and_corrupt_training_examples_are_rejected(self):
        for key in ("true_unknown_images_used", "dev_images_used", "test_images_used"):
            episodes = copy.deepcopy(self.episodes); episodes["report"][key] = True
            with self.assertRaises(ValueError):
                core.BoundaryVerifier.fit(self.source, episodes, mode="bce8", steps=1)
        for change in ("feature", "label", "bank", "candidate"):
            episodes = copy.deepcopy(self.episodes)
            if change == "feature": episodes["leaf"]["x"][0, 0] = float("nan")
            elif change == "label": episodes["leaf"]["y"][0] = .5
            elif change == "bank": episodes["leaf"]["bank_id"][0] = -1
            else: episodes["leaf"]["candidate"][0] = -1
            with self.subTest(change=change), self.assertRaises(ValueError):
                core.BoundaryVerifier.fit(self.source, episodes, mode="bce9", steps=1)

    def test_tampered_model_heads_norm_and_source_binding_are_rejected(self):
        for change in ("head", "norm", "source"):
            state = self.models["rank9"].state_dict()
            if change == "head": state["heads"]["leaf"]["layers.0.weight"][0, 0] += .1
            elif change == "norm": state["normalization"]["leaf"]["scale"][0] = 0.
            else: state["source_state"]["heads"]["leaf"]["layers.0.weight"][0, 0] += .1
            with self.subTest(change=change), self.assertRaises(ValueError):
                core.BoundaryVerifier.from_state_dict(state)


if __name__ == "__main__":
    unittest.main()
