"""Real self-exclusion, all-parent RMD, and TRAIN-only evidence contracts."""
import copy
import unittest
from unittest.mock import patch

import torch

from taxosafe_parentrisk import evidence
from taxosafe_support.evidence import HierarchicalEvidence
from taxosafe_support.support import SupportBank
from tests import test_taxosafe_refine_pipeline as fixture
from tests.test_taxosafe_geometry_core import hierarchy_fixture


class ParentEvidenceGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_all_parent_scores_match_each_own_candidate_not_first_parent(self):
        fine, parent, labels, mapping = hierarchy_fixture(count=3)
        model = evidence.ParentEvidenceGeometry.fit(fine, parent, labels, mapping)
        all_scores = model.score_parents(parent)
        self.assertEqual(tuple(all_scores.shape), (len(parent), 3))
        for p in range(3):
            leaf = int(torch.where(mapping == p)[0][0])
            selected = model.score(fine, parent, [p] * len(fine), [leaf] * len(fine))
            self.assertTrue(torch.equal(all_scores[:, p], selected["parent_score"]))
        self.assertFalse(torch.equal(all_scores[:, 0], all_scores[:, 1]))
        restored = evidence.ParentEvidenceGeometry.from_state_dict(model.state_dict())
        self.assertTrue(torch.equal(all_scores, restored.score_parents(parent)))
        self.assertEqual(tuple(restored.score_parents(parent[:0]).shape), (0, 3))
        self.assertFalse(model.score_parents(parent.requires_grad_()).requires_grad)

    def test_singleton_parent_taxonomy_is_not_recomputed_after_self_exclusion(self):
        mapping = [0, 0, 1]
        features = torch.tensor([[1., .1], [1., .2], [1., -.1], [1., -.2], [-1., .1], [-1., .2]])
        bank = SupportBank(features, features, [0, 0, 1, 1, 2, 2],
                           ["a", "b", "c", "d", "e", "f"], mapping)
        model = HierarchicalEvidence(2, mapping, decoupled=True, membership_mode="reference")
        encoded = {"fine": features[-1:], "parent": features[-1:]}
        output = model(encoded, bank, query_hashes=["f"])
        audit = evidence._support_audits(bank, output, ["f"], {
            "parent_names": ["p", "q"], "leaf_names": ["a", "b", "c"], "leaf_to_parent": mapping})[0]
        self.assertEqual(audit["known_leaves_per_parent"], [2, 1])
        self.assertEqual(audit["effective_references_per_leaf"], [2, 2, 1])
        self.assertEqual(audit["effective_references_per_parent"], [4, 1])
        self.assertEqual(audit["excluded_self_references"], 1)
        self.assertFalse(bool(output["reference_allowed"][0, bank.hashes.index("f")]))

    def test_exhausted_support_fails_without_fabricated_scores(self):
        features = torch.tensor([[1., .1], [-1., .1]])
        bank = SupportBank(features, features, [0, 1], ["a", "b"], [0, 1])
        model = HierarchicalEvidence(2, [0, 1], decoupled=True, membership_mode="reference")
        output = model({"fine": features[:1], "parent": features[:1]}, bank, query_hashes=["a"])
        with self.assertRaisesRegex(ValueError, "without references"):
            evidence._support_audits(bank, output, ["a"], {
                "parent_names": ["p", "q"], "leaf_names": ["a", "b"], "leaf_to_parent": [0, 1]})

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_all_parent_and_long_count_regression(self):
        fine, parent, labels, mapping = hierarchy_fixture(count=3)
        model = evidence.ParentEvidenceGeometry.fit(fine, parent, labels, mapping)
        self.assertTrue(torch.allclose(model.score_parents(parent),
                                       model.score_parents(parent.cuda()).cpu(), atol=2e-3, rtol=1e-4))
        bank = SupportBank(parent, fine, labels, [str(i) for i in range(len(fine))], mapping).to("cuda")
        reference = HierarchicalEvidence(fine.shape[1], mapping, decoupled=True,
                                         membership_mode="reference").cuda()
        query = {"fine": fine[:1].cuda(), "parent": parent[:1].cuda()}
        output = reference(query, bank, query_hashes=["0"])
        audits = evidence._support_audits(bank, output, ["0"], {
            "parent_names": ["p", "q", "r"], "leaf_names": list("abcdef"),
            "leaf_to_parent": mapping.tolist()})
        self.assertEqual(audits[0]["excluded_self_references"], 1)


class ParentEvidenceCollectionTests(unittest.TestCase):
    setUpClass = classmethod(fixture.FrozenPipelineContracts.setUpClass.__func__)
    tearDownClass = classmethod(fixture.FrozenPipelineContracts.tearDownClass.__func__)
    setUp = fixture.FrozenPipelineContracts.setUp
    make_source = fixture.FrozenPipelineContracts.make_source
    load_stage = fixture.FrozenPipelineContracts.load_stage
    read_split = fixture.FrozenPipelineContracts.read_split

    def collect_train(self):
        from taxosafe_refine.importer import load_reference
        self.make_source()
        reference = load_reference(self.source, self.device)
        records, cached, timing = evidence.collect_features(
            {"train": self.groups["train"]}, reference, self.device)
        return reference, records["train"], cached["train"], timing["train"]

    def test_real_collector_excludes_self_and_separates_parent_text_from_support(self):
        reference, records, cached, audit = self.collect_train()
        self.assertEqual(audit["support_overlap_unique_images"], len(records))
        self.assertEqual(audit["excluded_self_references"], len(records))
        self.assertEqual(audit["effective_references_min"], len(reference.bank.hashes) - 1)
        images, _, indices = next(iter(evidence.support_pipeline.make_loader(
            self.groups["train"], reference.config, reference.meta)))
        encoded = evidence.encode_baseline(reference.encoder, images)
        hashes = [self.groups["train"][i]["image_sha256"] for i in indices.tolist()]
        expected = reference.evidence(encoded, reference.bank, query_hashes=hashes)
        leaky = reference.evidence(encoded, reference.bank)
        self.assertFalse(torch.equal(expected["leaf_membership_logits"], leaky["leaf_membership_logits"]))
        for i in range(len(indices)):
            self.assertEqual(records[i]["support_evidence"]["parent_logits"], expected["parent_logits"][i].tolist())
            self.assertEqual(records[i]["encoder_evidence"]["parent_text_logits"], encoded["parent_logits"][i].tolist())
            self.assertEqual(records[i]["support_audit"]["excluded_self_references"], 1)
        self.assertEqual(cached["image_sha256"], [r["image_sha256"] for r in records])

    def test_fit_is_train_only_and_inference_ignores_truth(self):
        reference, records, cached, _ = self.collect_train()
        geometry, scales, audit = evidence.fit_evidence(records, cached, reference.meta,
                                                        {"shrinkage": .1, "ridge": .0001})
        self.assertEqual(set(scales), set(evidence.SCORE_NAMES))
        for key, value in audit.items():
            self.assertEqual(value["source_split"], "train")
            self.assertEqual(value["count"], len(value["image_sha256"]))
            self.assertEqual(value["count"], scales[key].state_dict()["fit_rows"])
        for key in evidence.PARENT_SCORE_NAMES:
            self.assertEqual(audit[key]["selection"], "true_parent_per_known_train_image")
            self.assertEqual(audit[key]["count"], len(records))
        altered = copy.deepcopy(records)
        for row in altered:
            row.update(true_leaf=None, true_parent=None, status="extra", split="test_extra", source="arbitrary")
        with patch.object(evidence.RobustScoreStandardizer, "fit", side_effect=AssertionError("no inference fitting")):
            evidence.add_evidence(altered, cached["fine"], cached["parent"], reference.meta, geometry, scales)
        for before, after in zip(records, altered):
            for key in ("parent_evidence", "geometry_parent_score", "geometry_leaf_score",
                        "baseline_parent_z", "baseline_leaf_z"):
                self.assertEqual(before[key], after[key])
        with self.assertRaisesRegex(ValueError, "known TRAIN"):
            evidence.fit_evidence(altered, cached, reference.meta, {})
        changed = dict(cached, image_sha256=list(reversed(cached["image_sha256"])))
        with self.assertRaisesRegex(ValueError, "hashes differ"):
            evidence.fit_evidence(records, changed, reference.meta, {})
        alias = copy.deepcopy(records[0])
        alias["encoder_evidence"]["parent_text_logits"][0] += 1.
        with self.assertRaisesRegex(ValueError, "inconsistent encoder_evidence"):
            evidence.fit_evidence(records + [alias], cached, reference.meta, {})
        bad = copy.deepcopy(records)
        bad[0]["encoder_evidence"]["parent_text_logits"][0] = None
        with self.assertRaisesRegex(ValueError, "parent evidence"):
            evidence.add_evidence(bad, cached["fine"], cached["parent"], reference.meta, geometry, scales)

    def test_dev_collection_passes_hashes_and_aliases_share_single_visual_pass(self):
        reference, records, cached, _ = self.collect_train()
        geometry, scales, _ = evidence.fit_evidence(records, cached, reference.meta, {})
        original = fixture.artifact_snapshot(self.source)
        with patch.object(reference.evidence, "forward", wraps=reference.evidence.forward) as forward:
            collected, features, timings = evidence.collect_features(
                {"test_known": self.groups["test_known"]}, reference, self.device,
                geometry=geometry, scales=scales)
        calls = forward.call_args_list
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].kwargs["query_hashes"], features["test_known"]["image_sha256"])
        rows = collected["test_known"]
        self.assertEqual(len(rows), 5)
        self.assertEqual(timings["test_known"]["unique_images"], 4)
        self.assertEqual(timings["test_known"]["support_overlap_unique_images"], 0)
        self.assertEqual(rows[0]["parent_evidence"], rows[-1]["parent_evidence"])
        self.assertEqual(original, fixture.artifact_snapshot(self.source))


if __name__ == "__main__":
    unittest.main()
