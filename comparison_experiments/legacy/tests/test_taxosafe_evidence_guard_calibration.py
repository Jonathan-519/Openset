import copy
import hashlib
import math
import unittest
from unittest import mock

import torch

from taxosafe_evidence_guard import calibration as cal
from taxosafe_support import membership_calibration as membership
from taxosafe_dcbs.protocol import normalized_name


META = dict(leaf_names=["a", "b", "c", "d"], parent_names=["p", "q"], leaf_to_parent=[0, 0, 1, 1])
SETTINGS = dict(seed=1, offset_grid=[-1., 0., 1.])
REFERENCE_SETTINGS = dict(policy="known_first", membership_grid_points=3, source_loo=False)
REFERENCE = dict(schema_version=membership.SCHEMA_VERSION, decoder="membership", meta=META,
                 parent_threshold=0., leaf_threshold=0.)
ARM = dict(id="R04_OE_both", kind="fit", adapt_parent=True, adapt_fine=True,
           use_oe=True, anchor=True, geometry=False, root_guard=False)
CFG = dict(seed=1, calibration=SETTINGS, training=dict(steps=3), geometry={})


def row(identity, split, status, source, leaf=None, parent=None):
    p = parent if parent is not None else 0
    c = leaf if leaf is not None else (0 if p == 0 else 2)
    ranking_parent = [0., 0.]
    ranking_parent[p] = 4.
    ranking_leaf = [0.] * 4
    ranking_leaf[c] = 4.
    evidence = dict(parent_logits=ranking_parent, leaf_logits=ranking_leaf,
                    parent_membership_logits=[-3. if status == "extra" else 3.] * 2,
                    leaf_membership_logits=[3. if status == "known" else -3.] * 4)
    return dict(image_sha256=hashlib.sha256(identity.encode()).hexdigest(), split=split,
                status=status, source=source, true_leaf=leaf, true_parent=parent,
                support_evidence=evidence, log_probs=[-math.log(7.)] * 7)


def fixtures():
    dev, train = [], []
    for leaf in range(4):
        for i in range(3):
            dev.append(row("dev-known-{}-{}".format(leaf, i), "val_known", "known", META["leaf_names"][leaf],
                           leaf, META["leaf_to_parent"][leaf]))
            train.append(row("train-known-{}-{}".format(leaf, i), "train", "known", META["leaf_names"][leaf],
                             leaf, META["leaf_to_parent"][leaf]))
    for status, split, train_split in (("intra", "val_intra", "train_intra"), ("extra", "val_extra", "oe_train")):
        for source in range(4):
            for i in range(2):
                name = "{}_source_{}".format(status, source)
                parent = source % 2 if status == "intra" else None
                dev.append(row("dev-{}-{}".format(name, i), split, status, name, parent=parent))
                train.append(row("train-{}-{}".format(name, i), train_split, status, name, parent=parent))
    def cache(rows):
        return dict(groups={split: dict(records=[r for r in rows if r["split"] == split])
                            for split in {r["split"] for r in rows}}, meta=META,
                    source_binding={"frozen_C00": "fixture"})
    return cache(train), cache(dev), dev, train


class EvidenceGuardCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.traincache, self.devcache, self.dev, self.train = fixtures()
        self.folds = cal.reference_folds(self.dev, META, SETTINGS, REFERENCE_SETTINGS)
        self.calls = []

    def fake_fit(self, cache, meta, arm, cfg, device="cpu", exclude_sources=(), reference_router=None):
        excluded = {normalized_name(name) for name in exclude_sources}
        selected = [r for r in self.train if r["status"] == "known" or
                    arm.get("use_oe") and normalized_name(r["source"]) not in excluded]
        matching = [fold for fold in self.folds if fold.get("reference_router") == reference_router]
        self.assertEqual(len(matching), 1)
        fold = matching[0]
        self.assertFalse(set(reference_router["fit_image_sha256"]) & set(fold["held_image_sha256"]))
        # No DEV labels/records are arguments to adapter fitting. Loss centers
        # come only from this fold's fit-only reference operating point.
        self.assertIs(cache, self.traincache)
        self.assertEqual(excluded, {fold["source"]} if fold["kind"] == "unknown_source" else set())
        self.calls.append(dict(fold_id=fold["fold_id"], excluded=excluded,
                               fit_hashes=[r["image_sha256"] for r in selected]))
        report = dict(fit_image_sha256=[r["image_sha256"] for r in selected],
                      used_image_sha256=[r["image_sha256"] for r in selected],
                      used_sources=sorted({normalized_name(r["source"]) for r in selected if r["status"] != "known"}),
                      training_seed=cfg["seed"], excluded_sources=sorted(excluded),
                      loss_reference_parent_threshold=float(reference_router["parent_threshold"]),
                      loss_reference_leaf_threshold=float(reference_router["leaf_threshold"]))
        return dict(state={"adapter": torch.tensor([float(cfg["seed"])])}, report=report), report

    def fake_score(self, cache, payload, device="cpu"):
        return {split: [dict(copy.deepcopy(r), source_support_evidence=copy.deepcopy(r["support_evidence"]))
                        for r in group["records"]] for split, group in cache["groups"].items()}

    def test_zero_offset_is_exact_original_and_hash_protected(self):
        before = membership.decode_records(self.dev, REFERENCE, META)
        router, diagnostics = cal.fit(self.dev, META, SETTINGS, REFERENCE,
                                     dict(ARM, id="R00_reference", kind="reference"))
        self.assertEqual((router["root_offset"], router["leaf_offset"]), (0., 0.))
        self.assertEqual(len(diagnostics["operating_point_grid"]), 1)
        after = cal.decode(self.dev, router, META)
        fields = ("output_node", "parent", "leaf", "candidate_parent", "candidate_leaf",
                  "parent_membership_score", "leaf_membership_score", "root_threshold", "local_threshold")
        self.assertEqual([[r[f] for f in fields] for r in before], [[r[f] for f in fields] for r in after])
        router["root_offset"] = 1.
        with self.assertRaisesRegex(ValueError, "identity"):
            cal.decode(self.dev, router, META)

    def test_ROOT_guard_retains_original_roots_independent_of_truth(self):
        original = row("guard", "val_extra", "extra", "fish")
        changed = copy.deepcopy(original)
        changed["source_support_evidence"] = copy.deepcopy(original["support_evidence"])
        changed["support_evidence"]["parent_membership_logits"] = [5., 5.]
        changed["support_evidence"]["leaf_membership_logits"] = [5.] * 4
        router = cal._state(META, 0., 0., [], REFERENCE, dict(ARM, root_guard=True))
        out = cal.decode([changed], router, META)[0]
        self.assertEqual(out["prediction_type"], "global_unknown")
        self.assertTrue(out["root_guard_applied"])
        changed.update(status="known", source="different", true_parent=1, true_leaf=3)
        same = cal.decode([changed], router, META)[0]
        self.assertEqual((out["output_node"], out["parent"], out["leaf"]),
                         (same["output_node"], same["parent"], same["leaf"]))
        self.assertEqual(original["support_evidence"]["parent_membership_logits"], [-3., -3.])

    def test_test_fit_rejected_and_disabled_branch_threshold_locked(self):
        rows = copy.deepcopy(self.dev)
        rows[0]["split"] = "test_known"
        with self.assertRaisesRegex(ValueError, "TEST"):
            cal.fit(rows, META, SETTINGS, REFERENCE, ARM)
        _, fine = cal.fit(self.dev, META, SETTINGS, REFERENCE, dict(ARM, id="R02_OE_fine", adapt_parent=False))
        self.assertEqual({item["root_offset"] for item in fine["operating_point_grid"]}, {0.})
        _, parent = cal.fit(self.dev, META, SETTINGS, REFERENCE, dict(ARM, id="R03_OE_parent", adapt_fine=False))
        self.assertEqual({item["leaf_offset"] for item in parent["operating_point_grid"]}, {0.})

    def test_source_crossfit_refits_all_11_folds_and_excludes_TRAIN_source(self):
        self.assertEqual(len(self.folds), 11)
        self.assertEqual({f["status"] for f in self.folds}, {"completed"})
        with mock.patch("taxosafe_evidence_guard.training.fit", side_effect=self.fake_fit), \
             mock.patch("taxosafe_evidence_guard.training.score", side_effect=self.fake_score):
            oof, scores = cal.source_crossfit(self.traincache, self.devcache, META, ARM, CFG, self.folds)
        self.assertTrue(oof["complete"], oof["folds"])
        self.assertTrue(oof["passed"])
        self.assertEqual(len(self.calls), 11)
        self.assertTrue(oof["adapter_refitted_in_folds"])
        self.assertFalse(oof["independent_full_pipeline_validation"])
        self.assertEqual(len(oof["predictions"]), len(self.dev))
        self.assertEqual(len(scores["folds"]), 11)
        for audit in oof["folds"]:
            self.assertEqual(audit["excluded_train_image_count"], 2 if audit["kind"] == "unknown_source" else 0)
            self.assertFalse(set(audit["router"]["fit_image_sha256"]) & set(audit["held_image_sha256"]))

    def test_R07_reuses_only_matching_R04_fold_models_without_training(self):
        with mock.patch("taxosafe_evidence_guard.training.fit", side_effect=self.fake_fit), \
             mock.patch("taxosafe_evidence_guard.training.score", side_effect=self.fake_score):
            _, scores = cal.source_crossfit(self.traincache, self.devcache, META, ARM, CFG, self.folds)
        with mock.patch("taxosafe_evidence_guard.training.fit", side_effect=AssertionError("no retraining")), \
             mock.patch("taxosafe_evidence_guard.training.score", side_effect=AssertionError("no full-data scores")):
            guarded, _ = cal.source_crossfit(self.traincache, self.devcache, META,
                dict(ARM, id="R07_root_guard", root_guard=True), CFG, self.folds, reused=scores)
            self.assertTrue(guarded["complete"], guarded["folds"])
            broken = copy.deepcopy(scores)
            broken["folds"].pop()
            incomplete, _ = cal.source_crossfit(self.traincache, self.devcache, META,
                dict(ARM, id="R07_root_guard", root_guard=True), CFG, self.folds, reused=broken)
            self.assertFalse(incomplete["complete"])
            self.assertFalse(incomplete["passed"])
            self.assertEqual(sum(f["status"] == "not_evaluable" for f in incomplete["folds"]), 1)
            invalid, _ = cal.source_crossfit(self.traincache, self.devcache, META,
                dict(ARM, id="R07_root_guard", root_guard=True), CFG, self.folds, reused=None)
            self.assertFalse(invalid["complete"])
            self.assertFalse(invalid["passed"])
            self.assertEqual(invalid["held_image_count"], 0)

    def test_leaked_TRAIN_source_is_not_evaluable_not_baseline_substitution(self):
        def leaking_fit(*args, **kwargs):
            payload, report = self.fake_fit(*args, **kwargs)
            if kwargs["exclude_sources"]:
                source = kwargs["exclude_sources"][0]
                leaked = next(r for r in self.train if normalized_name(r["source"]) == source)
                report["fit_image_sha256"].append(leaked["image_sha256"])
            return payload, report
        with mock.patch("taxosafe_evidence_guard.training.fit", side_effect=leaking_fit), \
             mock.patch("taxosafe_evidence_guard.training.score", side_effect=self.fake_score):
            oof, scores = cal.source_crossfit(self.traincache, self.devcache, META, ARM, CFG, self.folds)
        self.assertFalse(oof["complete"])
        self.assertFalse(oof["passed"])
        self.assertEqual(sum(f["status"] == "not_evaluable" for f in oof["folds"]), 8)
        self.assertEqual(len(oof["predictions"]), 12)
        self.assertTrue(all("Held unknown source" in f["error"] for f in oof["folds"] if f["status"] == "not_evaluable"))

    def test_real_adapter_source_fold_refits_without_mock_training(self):
        from tests.test_taxosafe_evidence_guard_core import EvidenceGuardCoreTests
        fixture = EvidenceGuardCoreTests(methodName="test_source_holdout_removes_train_unknowns_and_never_knowns")
        fixture.setUp()
        try:
            groups = {}
            for old_split, split, status in (("train", "val_known", "known"),
                                              ("train_intra", "val_intra", "intra"),
                                              ("oe_train", "val_extra", "extra")):
                old = fixture.cache["groups"][old_split]
                hashes = [hashlib.sha256(("independent-dev-" + h).encode()).hexdigest() for h in old["image_sha256"]]
                groups[split] = fixture._group(split, status, old["encoded"], hashes)
            devcache = dict(fixture.cache, groups=groups)
            rows = [r for group in groups.values() for r in group["records"]]
            settings = dict(seed=1, offset_grid=[0.])
            folds = cal.reference_folds(rows, META, settings, REFERENCE_SETTINGS)
            cfg = dict(fixture.cfg, calibration=settings)
            with mock.patch("builtins.print"):
                oof, scores = cal.source_crossfit(fixture.cache, devcache, META, ARM, cfg, folds)
            self.assertTrue(oof["complete"], [(f["fold_id"], f.get("error")) for f in oof["folds"]])
            self.assertEqual(len(folds), 7)
            for fold in oof["folds"]:
                self.assertEqual(fold["training_report"]["optimizer_steps"], 3)
                self.assertGreater(fold["training_report"]["parameter_delta_l2"], 0.)
                self.assertFalse(set(fold["training_report"]["fit_image_sha256"]) & set(fold["excluded_train_image_sha256"]))
                self.assertFalse(set(fold["training_report"]["used_image_sha256"]) & set(fold["excluded_train_image_sha256"]))
            self.assertTrue(all(item["model_state_sha256"] for item in scores["folds"]))
        finally:
            fixture.tearDown()

    def test_reference_OOF_never_trains(self):
        with mock.patch("taxosafe_evidence_guard.training.fit", side_effect=AssertionError("reference immutable")):
            oof, _ = cal.source_crossfit(self.traincache, self.devcache, META,
                dict(ARM, id="R00_reference", kind="reference"), CFG, self.folds)
        self.assertTrue(oof["complete"])
        self.assertFalse(oof["adapter_refitted_in_folds"])


if __name__ == "__main__":
    unittest.main()
