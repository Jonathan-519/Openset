import copy
import unittest
from unittest import mock

import torch

from taxosafe_discovery.geometry import GeometryBank, evidence_features


def fixture():
    torch.manual_seed(3)
    labels = torch.arange(4).repeat_interleave(6)
    centers = torch.eye(4)
    fine = centers[labels] + .08 * torch.randn(24, 4)
    parent = fine[:, :3] + .01
    hashes = ["image-%02d" % i for i in range(24)]
    meta = {"leaf_names": ["a", "b", "c", "d"], "parent_names": ["ab", "c", "d"],
            "leaf_to_parent": [0, 0, 1, 2]}
    return fine, parent, labels, hashes, meta


class GeometryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_relative_distance_formula_and_independent_levels(self):
        fine, parent, labels, hashes, meta = fixture()
        bank = GeometryBank.fit(fine, parent, labels, hashes, meta)
        query = fine[:2] + .01
        scores = bank.score(query, parent[:2], ["new-a", "new-b"])
        q = query.double() / query.double().norm(dim=1, keepdim=True)
        stat = bank._stats["leaf"]
        direct = ((q - stat["global_mean"]).square() / stat["global_variance"]).sum(1)[:, None]
        direct = direct - ((q[:, None, :] - stat["means"][None, :, :]).square() / stat["within_variance"]).sum(2)
        self.assertTrue(torch.allclose(scores["leaf_rmd"], direct, atol=1e-10, rtol=0))
        self.assertEqual(scores["leaf_rmd"].shape, (2, 4))
        self.assertEqual(scores["parent_rmd"].shape, (2, 3))

    def test_self_exclusion_applies_to_whole_bank(self):
        args = fixture()
        bank = GeometryBank.fit(*args)
        with self.assertRaisesRegex(ValueError, "overlap"):
            bank.score(args[0][:1], args[1][:1], [args[3][0]])

    def test_missing_class_masks_geometry_and_template_rivals(self):
        fine, parent, labels, hashes, meta = fixture()
        mask = labels != 1
        bank = GeometryBank.fit(fine[mask], parent[mask], labels[mask], [h for h, keep in zip(hashes, mask) if keep], meta)
        scores = bank.score(fine[~mask], parent[~mask], [h for h, keep in zip(hashes, mask) if not keep])
        self.assertFalse(bool(scores["leaf_active"][1]))
        self.assertTrue(bool(scores["parent_active"][0]))
        text = {"leaf": torch.zeros(6, 4), "parent": torch.zeros(6, 3)}
        before = evidence_features(scores, "leaf", text)
        text["leaf"][:, 1] = 1e5
        after = evidence_features(scores, "leaf", text)
        self.assertTrue(torch.equal(before[:, [0, 2, 3]], after[:, [0, 2, 3]]))
        self.assertTrue(bool((scores["leaf_rmd"][:, 1] == -1e6).all()))

    def test_state_roundtrip_and_inconsistent_report_rejected(self):
        fine, parent, labels, hashes, meta = fixture()
        bank = GeometryBank.fit(fine, parent, labels, hashes, meta)
        state = bank.state_dict()
        # TEST loads the frozen numerical estimates; even invoking either
        # fitting entry point during load is forbidden by the protocol.
        with mock.patch.object(GeometryBank, "fit", side_effect=AssertionError("TEST fit called")), \
                mock.patch("taxosafe_discovery.geometry._statistics", side_effect=AssertionError("TEST statistics refit called")):
            loaded = GeometryBank.from_state_dict(state)
        score = bank.score(fine[:1], parent[:1], ["new"])
        other = loaded.score(fine[:1], parent[:1], ["new"])
        for key in score:
            self.assertTrue(torch.equal(score[key], other[key]))
        state["fit_report"]["support_count"] += 1
        with self.assertRaisesRegex(ValueError, "report"):
            GeometryBank.from_state_dict(state)

    def test_saved_statistic_tampering_is_rejected_without_recalculation(self):
        bank = GeometryBank.fit(*fixture())
        state = bank.state_dict()
        state["statistics"]["leaf"]["means"][0, 0] *= .5
        with self.assertRaisesRegex(ValueError, "digest"):
            GeometryBank.from_state_dict(state)
        state = bank.state_dict()
        state["statistics"]["leaf"]["within_variance"][0] = -1.
        with self.assertRaisesRegex(ValueError, "covariance"):
            GeometryBank.from_state_dict(state)
        state = bank.state_dict()
        state["statistics"]["parent"]["counts"][0] += 1
        with self.assertRaisesRegex(ValueError, "counts"):
            GeometryBank.from_state_dict(state)

    def test_invalid_train_inputs_rejected(self):
        fine, parent, labels, hashes, meta = fixture()
        with self.assertRaises(ValueError):
            GeometryBank.fit(fine, parent, labels, [hashes[0]] * len(hashes), meta)
        with self.assertRaises(ValueError):
            GeometryBank.fit(fine * float("nan"), parent, labels, hashes, meta)
        with self.assertRaises(ValueError):
            GeometryBank.fit(fine, parent, labels.float() + .1, hashes, meta)


if __name__ == "__main__":
    unittest.main()
