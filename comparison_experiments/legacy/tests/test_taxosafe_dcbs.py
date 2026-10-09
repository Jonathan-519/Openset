"""CPU contracts for depth targets, feature supports and frozen calibration."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F
import yaml

from taxosafe_dcbs import calibration as cal
from taxosafe_dcbs.model import DCBSHeads, dcbs_loss, sibling_margin_loss
from taxosafe_dcbs import protocol
from taxosafe_dcbs.synthesis import empty_synthetic, fit_support, support_membership, synthesize

META = {"parent_names": ["P", "Q", "S"], "leaf_names": ["a", "b", "c", "d", "e"], "leaf_to_parent": [0, 0, 1, 1, 2]}
CENTRES = F.normalize(torch.tensor([[1., 0., .3, 0.], [1., 0., -.3, 0.], [-1., 0., .3, 0.], [-1., 0., -.3, 0.], [0., 1., 0., 0.]]), dim=1)
CONFIG = protocol.PROJECT_ROOT / "configs/Zooplankton_Taxonomic_Tree/TaxoSafe_dcbs_v11.yml"


def settings():
    return {"projection_dim": 16, "logit_scale": 5., "loss": {"ha": .02, "margin": .1},
            "synthesis": {"candidates_per_parent": 64, "keep_per_parent": 16, "radius_prior_count": 0., "near_noise": 0., "extra_noise": .2}}


def known_features(n=12):
    labels = torch.arange(5).repeat_interleave(n)
    noise = torch.randn(n * 5, 4, generator=torch.Generator().manual_seed(4)) * .003
    return F.normalize(CENTRES[labels] + noise, dim=1), labels


def score_fixture():
    known, near, extra = [], [], []
    for i in range(75):
        c, q = i % 5, i / 1000
        p = META["leaf_to_parent"][c]
        known.append({"status": "known", "split": "val_known", "source": META["leaf_names"][c], "path": str(i),
                      "true_leaf": c, "true_parent": p, "pred_leaf": c, "pred_parent": p, "global_pred_leaf": c,
                      "path_consistent": True, "root_ratio": 2 + q, "root_support": .8 + q,
                      "local_ratio": 1 + q, "local_support": .8 + q, "sibling_margin": None if p == 2 else .5 + q})
    for i in range(24):
        p, q = i % 2, i / 1000
        near.append({"status": "intra", "split": "val_intra", "source": "near_" + str(p), "path": "n" + str(i),
                     "true_leaf": None, "true_parent": p, "pred_leaf": p * 2, "pred_parent": p, "global_pred_leaf": p * 2,
                     "path_consistent": True, "root_ratio": 2.01 + q, "root_support": .84 + q,
                     "local_ratio": -.4 + q, "local_support": .3 + q, "sibling_margin": .02})
        extra.append(dict(near[-1], status="extra", split="val_extra", source="ood_" + str(i % 2), path="e" + str(i),
                          true_parent=None, root_ratio=-3 + i / 100, root_support=.1 + q))
    return known, near, extra


class ModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_depth_supervision_is_learnable(self):
        torch.manual_seed(5)
        cfg = settings()
        cfg["loss"].update(ha=0, margin=0)
        heads = DCBSHeads(4, META["leaf_to_parent"], 3, cfg)
        h, labels = known_features(4)
        near = torch.tensor([[1., 0., 0., 0.], [-1., 0., 0., 0.]])
        extra = torch.tensor([[0., -1., 0., .1], [0., -1., 0., -.1]])
        synth = {"near": near, "near_parent": torch.tensor([0, 1]), "extra": extra}
        optimizer = torch.optim.Adam(heads.parameters(), lr=.025)
        for _ in range(90):
            loss, _ = dcbs_loss(heads, h, labels, h @ CENTRES.T * 5, h @ torch.tensor([[1., 0., 0., 0.], [-1., 0., 0., 0.], [0., 1., 0., 0.]]).T * 5, synth, cfg)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            out = heads(near)
            self.assertEqual(out["root"].argmax(1).tolist(), [0, 1])
            self.assertEqual([int(heads.local_logits(out, p)[p].argmax()) for p in range(2)], [2, 2])
            self.assertEqual(heads(extra)["root"].argmax(1).tolist(), [3, 3])

    def test_adversary_cannot_erase_backbone_or_taxonomy(self):
        heads = DCBSHeads(4, META["leaf_to_parent"], 3, settings())
        h = CENTRES.clone().requires_grad_()
        F.cross_entropy(heads.adversarial_logits(h), torch.tensor(META["leaf_to_parent"])).backward()
        self.assertIsNone(h.grad)
        self.assertTrue(all(p.grad is None for p in heads.taxonomy.parameters()))
        self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()) for p in heads.fine.parameters()))

    def test_singleton_margin_is_connected_zero(self):
        heads = DCBSHeads(4, META["leaf_to_parent"], 3, settings())
        h = CENTRES[4:].clone().requires_grad_()
        loss = sibling_margin_loss(heads, heads(h), torch.tensor([4]))
        loss.backward()
        self.assertEqual(float(loss), 0.)
        self.assertTrue(bool(torch.isfinite(h.grad).all()))

    def test_lite_shares_projection_and_disallows_ha(self):
        cfg = dict(settings(), shared_projection=True)
        heads = DCBSHeads(4, META["leaf_to_parent"], 3, cfg)
        t, f = heads.embeddings(CENTRES)
        self.assertIs(t, f)
        bank = fit_support(*known_features(), META["leaf_to_parent"], cfg["synthesis"])
        with self.assertRaisesRegex(ValueError, "HA requires"):
            dcbs_loss(heads, CENTRES, torch.arange(5), torch.eye(5), torch.zeros(5, 3), empty_synthetic(bank), cfg)


class SynthesisTest(unittest.TestCase):
    def test_depth_support_and_determinism(self):
        cfg = settings()["synthesis"]
        bank = fit_support(*known_features(), META["leaf_to_parent"], cfg)
        a = synthesize(bank, CENTRES, 5, cfg, torch.Generator().manual_seed(7))
        b = synthesize(bank, CENTRES, 5, cfg, torch.Generator().manual_seed(7))
        self.assertGreater(len(a["near"]), 0)
        self.assertGreater(len(a["extra"]), 0)
        self.assertTrue(torch.equal(a["near"], b["near"]))
        leaf, parent = support_membership(a["near"], bank)
        self.assertFalse(bool(leaf.any()))
        self.assertTrue(bool(parent[torch.arange(len(parent)), a["near_parent"]].all()))
        leaf, parent = support_membership(a["extra"], bank)
        self.assertFalse(bool(leaf.any() or parent.any()))
        self.assertEqual(a["stats"]["near_by_parent"]["2"], 0)

    def test_no_fake_fallback_when_support_covers_candidates(self):
        cfg = settings()["synthesis"]
        bank = fit_support(*known_features(), META["leaf_to_parent"], cfg)
        bank["leaf_radius"].fill_(2.)
        bank["parent_radius"].fill_(2.)
        synth = synthesize(bank, CENTRES, 5, cfg)
        self.assertEqual(len(synth["near"]) + len(synth["extra"]), 0)

    def test_missing_training_leaf_is_rejected(self):
        h, y = known_features()
        with self.assertRaisesRegex(ValueError, "Missing training"):
            fit_support(h[y != 4], y[y != 4], META["leaf_to_parent"], settings()["synthesis"])


class CalibrationTest(unittest.TestCase):
    def test_three_depths_and_post_shrink_known_floor(self):
        k, n, e = score_fixture()
        router = cal.calibrate(k, n, e, META, {})
        rk, rn, re = (cal.apply_router(rows, router, META) for rows in (k, n, e))
        self.assertGreaterEqual(sum(r["prediction_type"] == "known" for r in rk) / len(k), .92)
        self.assertTrue(all(r["prediction_type"] == "intra_unknown" and r["parent"] == r["true_parent"] for r in rn))
        self.assertTrue(all(r["prediction_type"] == "global_unknown" for r in re))
        self.assertEqual(router["branches"]["2"]["source"], "known_quantile")
        self.assertTrue(any(r["prediction_type"] == "known" for r in rk if r["true_parent"] == 2))
        self.assertGreaterEqual(router["known_guard"]["final_e2e"], router["known_guard"]["required_e2e"])

    def test_source_loo_excludes_held_source_from_fit(self):
        k, n, e = score_fixture()
        state = cal.fit_cdf(k, 3)
        k, n, e = (cal.score_records(rows, state) for rows in (k, n, e))
        with patch.object(cal, "_root_select", wraps=cal._root_select) as fitted:
            report = cal.source_loo(k, n, e, {})
        for call, fold in zip(fitted.call_args_list, report["folds"]):
            self.assertNotIn(fold["held_source"], {r["source"] for r in call.args[2]})
            self.assertNotIn(fold["held_source"], fold["fit_sources"])

    def test_calibration_rejects_test_records(self):
        k, n, e = score_fixture()
        k[0]["split"] = "test_known"
        with self.assertRaisesRegex(ValueError, "val_known"):
            cal.calibrate(k, n, e, META, {})

    def test_infeasible_path_guard_fails_instead_of_relaxing_floor(self):
        k, n, e = score_fixture()
        for row in k:
            row["path_consistent"] = False
        with self.assertRaisesRegex(ValueError, "cannot satisfy known E2E"):
            cal.calibrate(k, n, e, META, {})

    def test_decisions_ignore_true_labels_and_respect_path_conflicts(self):
        k, n, e = score_fixture()
        router = cal.calibrate(k, n, e, META, {})
        original = cal.apply_router(k, router, META)
        changed = [dict(r, true_leaf=4, true_parent=2) for r in k]
        routed = cal.apply_router(changed, router, META)
        self.assertEqual([(r["prediction_type"], r["leaf"], r["parent"]) for r in original], [(r["prediction_type"], r["leaf"], r["parent"]) for r in routed])
        accepted = next(r for r in original if r["prediction_type"] == "known")
        self.assertEqual(cal.apply_router([dict(accepted, path_consistent=False)], router, META)[0]["prediction_type"], "intra_unknown")

    def test_no_pooling_is_a_single_local_threshold(self):
        router = cal.calibrate(*score_fixture(), META, {"partial_pooling": False})
        self.assertEqual(len(set(router["local_thresholds"].values())), 1)

    def test_extreme_logits_are_finite_and_singletons_omit_margin(self):
        output = {"root": np.array([[0., 0., 10000., -10000.]]), "leaf": np.full((1, 5), 10000.),
                  "stop": np.full((1, 3), -10000.), "taxonomy": np.array([[0., 1., 0., 0.]]), "fine": np.array([[0., 1., 0., 0.]])}
        support = {"parent": [[1., 0., 0., 0.], [-1., 0., 0., 0.], [0., 1., 0., 0.]], "leaf": CENTRES.tolist()}
        row = cal.raw_records([{}], output, np.array([[0., 0., 0., 0., 10000.]]), support, META, {})[0]
        self.assertIsNone(row["sibling_margin"])
        self.assertTrue(np.isfinite([row["root_ratio"], row["local_ratio"]]).all())
        self.assertEqual(row["pred_leaf"], 4)

    def test_constant_evidence_is_not_silently_accepted(self):
        k, n, e = score_fixture()
        for r in k:
            r.update(root_ratio=1., root_support=1.)
        with self.assertRaisesRegex(ValueError, "degenerate"):
            cal.calibrate(k, n, e, META, {})


def audit_row(digest="abc", status="known", source="a", split="train"):
    return {"image_sha256": digest, "resolved_path": "/images/" + split + "/" + digest,
            "status": status, "source": source, "true_leaf": 0 if status == "known" else None,
            "true_parent": 0 if status != "extra" else None}


class ProtocolTest(unittest.TestCase):
    def test_hash_and_source_overlap_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "across splits"):
            protocol.audit_rows({"train": [audit_row()], "val_known": [audit_row(split="val_known")]})
        with self.assertRaisesRegex(ValueError, "overlap"):
            protocol.audit_rows({"test_extra": [audit_row("new", "extra", "OOD_A")]}, forbidden_sources=["ood a"])

    def test_stage_allowlist_never_loads_test_for_training(self):
        cfg = {"data": {"train": "train", "val_known": "val"}}
        audit = {s: {} for s in ("train", "val_known")}
        with patch.object(protocol, "read_split", return_value=[audit_row()]) as read, patch.object(protocol, "audit_rows", return_value=audit), patch.object(protocol, "file_hash", return_value="sha"):
            protocol.load_stage_rows(cfg, "train", META)
        self.assertEqual([c.args[1] for c in read.call_args_list], ["train", "val_known"])

    def test_run_lock_and_stage_refuse_concurrent_or_repeated_writers(self):
        with tempfile.TemporaryDirectory() as directory:
            with protocol.run_lock(directory):
                with self.assertRaisesRegex(RuntimeError, "Another DCBS"):
                    with protocol.run_lock(directory):
                        pass
            protocol.claim_stage(directory, "training")
            with self.assertRaisesRegex(ValueError, "already exists"):
                protocol.claim_stage(directory, "training")

    def test_artifact_hash_is_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "artifact.json").write_text("{}")
            protocol.write_json(root / "completed.json", {"router": {"path": "artifact.json", "sha256": protocol.file_hash(root / "artifact.json")}})
            protocol.verify_artifact(root, "completed.json", "router")
            (root / "artifact.json").write_text("{\"changed\":true}")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                protocol.verify_artifact(root, "completed.json", "router")

    def test_variants_and_legacy_oe_checkpoint_guard(self):
        lite = protocol.effective_config(CONFIG, "lite", 7)
        self.assertTrue(lite["dcbs"]["shared_projection"])
        self.assertEqual(lite["dcbs"]["loss"]["ha"], 0)
        self.assertEqual(lite["data"]["sampler"]["seed"], 7)
        self.assertFalse(protocol.effective_config(CONFIG, "heads_only")["dcbs"]["synthesis"]["near_enabled"])
        with tempfile.TemporaryDirectory() as directory:
            bad = copy.deepcopy(lite)
            bad["init_checkpoint"] = "v10_oe.pth"
            path = Path(directory) / "bad.yml"
            path.write_text(yaml.safe_dump(bad))
            with self.assertRaisesRegex(ValueError, "fresh prompts"):
                protocol.effective_config(path)

    def test_locked_test_duplicates_keep_rows_but_reject_conflicts(self):
        a = audit_row(split="test_known")
        b = dict(a, resolved_path="/images/copy")
        audit = protocol.audit_rows({"test_known": [a, b]}, allow_within_split=True)["test_known"]
        self.assertEqual((audit["count"], audit["unique_image_count"]), (2, 1))
        with self.assertRaisesRegex(ValueError, "conflicting"):
            protocol.audit_rows({"test_known": [a, dict(b, true_leaf=1)]}, allow_within_split=True)
        with self.assertRaisesRegex(ValueError, "across splits"):
            protocol.audit_rows({"test_known": [a], "test_intra": [b]}, allow_within_split=True)


if __name__ == "__main__":
    unittest.main()
