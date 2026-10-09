"""Meaningful contracts for preserved C00 evidence and real-unknown training."""
import copy
import hashlib
import unittest

import torch
from torch.nn import functional as F

from taxosafe_support.evidence import HierarchicalEvidence
from taxosafe_support.support import SupportBank
from taxosafe_evidence_guard.model import EvidenceGuard, RAW_FIELDS, restore_bank
from taxosafe_evidence_guard import training


META = dict(leaf_names=["a", "b", "c", "d"], parent_names=["p", "q"], leaf_to_parent=[0, 0, 1, 1])
SETTINGS = dict(steps=3, batch_size=6, learning_rate=.005, adapter_dim=4, weight_decay=.0001,
                feature_bound=.25, classification=.5, membership=1., distillation=2., residual=.1, log_every=3)


def arm(name="R04_OE_both", parent=True, fine=True, oe=True, anchor=True, geometry=False):
    return dict(id=name, kind="fit", adapt_parent=parent, adapt_fine=fine, use_oe=oe,
                geometry=geometry, anchor=anchor, root_guard=False, weight_source=None)


class EvidenceGuardCoreTests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        torch.manual_seed(170)
        self.kwargs = dict(hidden_dim=4, temperature=.3, local_enabled=True, decoupled=True,
                           membership_mode="reference", reference_topk=2)
        self.verifier = HierarchicalEvidence(6, META["leaf_to_parent"], **self.kwargs).eval()
        # Emulate a trained verifier whose pair head actually uses local views.
        with torch.no_grad():
            for matcher in (self.verifier.parent_matcher, self.verifier.fine_matcher,
                            self.verifier.parent_reference, self.verifier.fine_reference):
                matcher.residual[-1].weight.normal_(0., .1)
        self.features = dict(parent=F.normalize(torch.randn(12, 6), dim=-1),
                             fine=F.normalize(torch.randn(12, 6), dim=-1),
                             parent_local=F.normalize(torch.randn(12, 2, 6), dim=-1),
                             fine_local=F.normalize(torch.randn(12, 2, 6), dim=-1))
        self.hashes = [hashlib.sha256(("known" + str(i)).encode()).hexdigest() for i in range(12)]
        self.labels = torch.arange(4).repeat_interleave(3)
        self.bank = SupportBank(**self.features, labels=self.labels, hashes=self.hashes,
                                 leaf_to_parent=META["leaf_to_parent"], max_per_leaf=3)
        self.context = dict(dimension=6, evidence_kwargs=self.kwargs,
                            evidence_state=copy.deepcopy(self.verifier.state_dict()), bank_state=self.bank.state_dict(),
                            source_router=dict(parent_threshold=.2, leaf_threshold=.3), meta=META)
        groups = {"train": self._group("train", "known", self.features, self.hashes)}
        for split, status in (("train_intra", "intra"), ("oe_train", "extra")):
            encoded = {key: F.normalize(torch.randn(4, *value.shape[1:]), dim=-1)
                       for key, value in self.features.items()}
            hashes = [hashlib.sha256((split + str(i)).encode()).hexdigest() for i in range(4)]
            groups[split] = self._group(split, status, encoded, hashes)
        self.cache = dict(groups=groups, context=self.context, meta=META, reference_eval_batch_size=4)
        self.cfg = dict(seed=1, training=SETTINGS,
                        geometry=dict(neighbors=2, shrinkage=5., residual_bound=2., hidden=4))

    def tearDown(self):
        torch.set_num_threads(self.threads)

    def _group(self, split, status, encoded, hashes):
        with torch.no_grad():
            output = self.verifier(encoded, self.bank, query_hashes=hashes)
        rows = []
        for i, h in enumerate(hashes):
            leaf = int(self.labels[i]) if status == "known" else None
            parent = META["leaf_to_parent"][leaf] if leaf is not None else (i % 2 if status == "intra" else None)
            source = META["leaf_names"][leaf] if leaf is not None else (status + "_source_" + str(i % 2))
            rows.append(dict(image_sha256=h, split=split, status=status, true_leaf=leaf, true_parent=parent,
                             source=source, species=source, global_pred_leaf=0,
                             log_probs=output["log_probs"][i].tolist(),
                             support_evidence={key: output[key][i].tolist() for key in RAW_FIELDS}))
        return dict(records=rows, image_sha256=hashes, encoded=encoded)

    def test_zero_residual_preserves_all_four_heads_local_views_and_bank_bits(self):
        model = EvidenceGuard(self.context, arm(), SETTINGS)
        baseline = self.verifier(self.features, self.bank, query_hashes=self.hashes)
        output = model(self.features, self.hashes)
        for key in RAW_FIELDS + ("log_probs",):
            self.assertTrue(torch.equal(baseline[key], output[key]), key)
        for key, value in self.context["bank_state"].items():
            if torch.is_tensor(value):
                self.assertTrue(torch.equal(getattr(model.bank, key), value), key)
        no_local = dict(self.features)
        no_local.pop("parent_local")
        no_local.pop("fine_local")
        self.assertFalse(torch.equal(output["leaf_membership_logits"], model(no_local, self.hashes)["leaf_membership_logits"]))

    def test_train_hash_is_excluded_before_original_statistics_and_pair_pooling(self):
        model = EvidenceGuard(self.context, arm(), SETTINGS)
        encoded = {key: value[:1] for key, value in self.features.items()}
        before = model(encoded, self.hashes[:1])
        index = model.bank.hashes.index(self.hashes[0])
        for key in ("parent", "fine", "parent_local", "fine_local"):
            getattr(model.bank, key)[index] = torch.randn_like(getattr(model.bank, key)[index]) * 20.
        after = model(encoded, self.hashes[:1])
        for key in RAW_FIELDS:
            self.assertTrue(torch.equal(before[key], after[key]), key)

    def test_source_holdout_removes_train_unknowns_and_never_knowns(self):
        _, rows, hashes = training.training_inputs(self.cache, True, ["intra_source_0"])
        self.assertEqual(sum(row["status"] == "known" for row in rows), 12)
        self.assertEqual(sum(row["status"] == "intra" for row in rows), 2)
        self.assertFalse(any(row["source"] == "intra_source_0" for row in rows))
        bad = copy.deepcopy(self.cache)
        bad["groups"]["train_intra"]["records"][0]["split"] = "val_intra"
        with self.assertRaises(ValueError):
            training.training_inputs(bad, True)

    def test_known_control_has_no_unknown_data_dependency(self):
        known = copy.deepcopy(self.cache)
        del known["groups"]["train_intra"]
        del known["groups"]["oe_train"]
        payload, report = training.fit(known, META, arm("R01_known_control", oe=False), self.cfg)
        self.assertFalse(report["unknown_images_used_for_gradients"])
        self.assertEqual(report["gradient_splits"], ["train"])
        self.assertGreater(report["parameter_delta_l2"], 0.)
        for key, value in self.context["evidence_state"].items():
            self.assertTrue(torch.equal(value, payload["state"]["verifier." + key]))

    def test_real_unknown_fit_and_branch_controls_really_update_only_allowed_query_views(self):
        for spec, fixed_prefix in ((arm("R02_OE_fine", parent=False), "parent"),
                                   (arm("R03_OE_parent", fine=False), "leaf"),
                                   (arm(), None),
                                   (arm("R06_no_anchor", anchor=False), None)):
            with self.subTest(arm=spec["id"]):
                payload, report = training.fit(self.cache, META, spec, self.cfg, exclude_sources=["intra_source_0"])
                self.assertEqual(report["optimizer_steps"], 3)
                self.assertTrue(report["unknown_images_used_for_gradients"])
                self.assertNotIn("intra_source_0", report["used_sources"])
                self.assertEqual(report["synthetic_feature_count"], 0)
                self.assertFalse(report["class_or_parent_support_deletion"])
                model = EvidenceGuard(payload["context"], spec, SETTINGS)
                model.load_state_dict(payload["state"], strict=True)
                output, teacher = model(self.features, self.hashes), model.teacher(self.features, self.hashes)
                if fixed_prefix:
                    for suffix in ("_logits", "_membership_logits"):
                        self.assertTrue(torch.equal(output[fixed_prefix + suffix], teacher[fixed_prefix + suffix]))
                if not spec["anchor"]:
                    self.assertTrue(all(item["distillation"] == 0. for item in report["history"]))

    def test_scoring_is_label_blind_and_preserves_source_evidence_for_review(self):
        payload, _ = training.fit(self.cache, META, arm(), self.cfg)
        changed = copy.deepcopy(self.cache)
        for group in changed["groups"].values():
            for row in group["records"]:
                row.update(source="unrelated", true_leaf=None, true_parent=None, status="extra")
        first, second = training.score(self.cache, payload), training.score(changed, payload)
        for split in first:
            for a, b, old in zip(first[split], second[split], self.cache["groups"][split]["records"]):
                self.assertEqual(a["support_evidence"], b["support_evidence"])
                self.assertEqual(a["log_probs"], b["log_probs"])
                self.assertEqual(a["source_support_evidence"], old["support_evidence"])
                self.assertAlmostEqual(float(torch.tensor(a["log_probs"]).exp().sum()), 1., places=5)

    def test_four_head_anchor_detects_membership_boundary_change_not_just_leaf_classification(self):
        teacher = self.verifier(self.features, self.bank, query_hashes=self.hashes)
        student = {key: value.clone() for key, value in teacher.items() if torch.is_tensor(value)}
        initial = training.four_head_anchor(student, teacher, self.context["source_router"])
        student["parent_membership_logits"] = student["parent_membership_logits"] + 1.
        changed = training.four_head_anchor(student, teacher, self.context["source_router"])
        self.assertTrue(torch.allclose(initial, torch.zeros_like(initial), atol=1e-6))
        self.assertGreater(float(changed.detach().mean()), 0.)

    def test_geometry_zero_head_preserves_original_evidence_without_replacing_rank(self):
        model = EvidenceGuard(self.context, arm(geometry=True), SETTINGS, self.cfg["geometry"], {})
        geom = dict(parent=torch.randn(12, 2, 6), leaf=torch.randn(12, 4, 6))
        original = self.verifier(self.features, self.bank, query_hashes=self.hashes)
        output = model(self.features, self.hashes, geometry=geom)
        for key in RAW_FIELDS + ("log_probs",):
            self.assertTrue(torch.equal(output[key], original[key]), key)

    def test_geometry_arm_runs_real_gradient_fit_on_fixed_train_geometry(self):
        spec = arm("R05_OE_knn", geometry=True)
        payload, report = training.fit(self.cache, META, spec, self.cfg)
        self.assertEqual(set(payload["geometry_state"]["image_sha256"]), set(self.hashes))
        self.assertTrue(report["geometry_known_TRAIN_only"])
        self.assertGreater(sum(value for key, value in report["tensor_delta_l2"].items()
                               if key.startswith("geometry_heads.")), 0.)
        scored = training.score(self.cache, payload)
        self.assertEqual(set(scored), set(self.cache["groups"]))


if __name__ == "__main__":
    unittest.main()
