import copy
import io
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from taxosafe_discovery import prompt_training as prompts
from taxosafe_discovery.features import core_tensor_hash
from tests.test_taxosafe_discovery_features import core_source, tensor_snapshot


def word_tokens(texts, context_length=8):
    """Small vocabulary with the real CLIP SOT/context/class/EOT convention."""
    result = torch.zeros(len(texts), context_length, dtype=torch.long)
    fixed = {"a": 2, "photo": 3, "of": 4, ".": 5, "X": 6}
    for i, text in enumerate(texts):
        words = text.replace(".", " .").split()
        values = [1] + [fixed.get(word, 10 + sum(word.encode()) % 110) for word in words] + [127]
        if len(values) > context_length:
            raise ValueError("MiniCLIP text exceeds context length")
        result[i, :len(values)] = torch.tensor(values)
    return result


def fixture():
    core, source = core_source()
    meta = dict(leaf_names=["alpha", "beta", "gamma"],
                parent_names=["parentA", "parentB"], leaf_to_parent=[0, 0, 1])
    source.meta = copy.deepcopy(meta)
    source.binding = {"fixture": "frozen_core"}
    generator = torch.Generator().manual_seed(817)
    values = F.normalize(torch.randn(9, 16, generator=generator), dim=-1)
    records = [dict(image_sha256=format(i, "064x"), split="train", status="known",
                    source="known", path=str(i), true_leaf=i % 3,
                    true_parent=meta["leaf_to_parent"][i % 3]) for i in range(len(values))]
    group = dict(records=records, features={"clip": values},
                 image_sha256=[row["image_sha256"] for row in records],
                 record_feature_indices=list(range(len(records))))
    return core, source, group, meta


class PromptTrainingContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def fit(self, source, group, meta, loss="coop", **kwargs):
        with patch("models.clip.tokenize", side_effect=word_tokens):
            return prompts.fit_prompt(source, group, meta, loss=loss,
                                      options={"epochs": 2, "batch_size": 4}, **kwargs)

    def test_real_context_gradients_frozen_core_and_cpu_artifact(self):
        core, source, group, meta = fixture()
        before = tensor_snapshot(source.encoder)
        features = group["features"]["clip"].clone()
        artifact, report = self.fit(source, group, meta, "fa")
        self.assertEqual(report["optimizer_steps"], 6)
        self.assertEqual(report["trainable_parameters"], 2 * 4 * 64)
        self.assertEqual(report["selected_epoch"], 2)
        self.assertEqual(report["frozen_core_sha256_before"], core_tensor_hash(core))
        self.assertEqual(report["frozen_core_sha256_before"], report["frozen_core_sha256_after"])
        self.assertGreater(report["parameter_delta_l2"], 0.)
        self.assertTrue(all(delta > 0 for delta in report["context_delta_l2"].values()))
        self.assertTrue(all(row["context_gradient_l2_sum"] > 0 for row in report["history"]))
        self.assertTrue(all(gap < 1e-6 for gap in report["initial_reference_max_abs_gap"].values()))
        self.assertFalse(report["development_used_for_selection"])
        self.assertEqual(report["gradient_splits"], ["train"])
        for key, value in source.encoder.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]), key)
        self.assertTrue(torch.equal(features, group["features"]["clip"]))
        for value in (*artifact["context"].values(), *artifact["text_features"].values()):
            self.assertEqual(value.device.type, "cpu")
            self.assertFalse(value.requires_grad)
            self.assertTrue(torch.isfinite(value).all())
        for level in ("leaf", "parent"):
            torch.testing.assert_close(artifact["text_features"][level].norm(dim=1),
                                       torch.ones(len(meta[level + "_names"])))
        buffer = io.BytesIO()
        torch.save(artifact, buffer)
        buffer.seek(0)
        restored = torch.load(buffer, weights_only=True)
        self.assertTrue(torch.equal(restored["context"]["leaf"], artifact["context"]["leaf"]))

    def test_deterministic_coop_and_fa_share_budget_order_and_initialization(self):
        _, source, group, meta = fixture()
        rng = torch.random.get_rng_state().clone()
        first, first_report = self.fit(source, group, meta, seed=17)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        repeated, repeated_report = self.fit(source, group, meta, seed=17)
        other, other_report = self.fit(source, group, meta, "fa", seed=17)
        self.assertEqual(first_report, repeated_report)
        self.assertTrue(torch.equal(first["context"]["leaf"], repeated["context"]["leaf"]))
        self.assertEqual(first_report["batch_order_sha256"], other_report["batch_order_sha256"])
        self.assertEqual(first_report["optimizer_steps"], other_report["optimizer_steps"])
        self.assertEqual(first_report["initial_reference_max_abs_gap"], other_report["initial_reference_max_abs_gap"])
        self.assertFalse(torch.equal(first["context"]["leaf"], other["context"]["leaf"]))

    def test_objective_matches_weighted_reference_denominator_and_detaches_teacher(self):
        logits = torch.tensor([[.7, -.1, .2], [-.4, .2, .9]], requires_grad=True)
        reference = torch.tensor([[.3, .2, .1], [.1, -.2, .5]], requires_grad=True)
        labels = torch.tensor([0, 2])
        loss = prompts.prompt_objective(logits, labels, reference, "fa", reference_weight=3., temperature=2.)
        denominator = torch.cat((logits / 2., reference.detach() / 2. + torch.tensor(3.).log()), dim=1).logsumexp(1)
        expected = (denominator - (logits / 2.)[torch.arange(2), labels]).mean()
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertIsNone(reference.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())
        torch.testing.assert_close(prompts.prompt_objective(logits, labels, reference, "coop", temperature=2.),
                                   F.cross_entropy(logits / 2., labels))

    def test_only_shallow_context_is_trainable_no_visual_module(self):
        core, _, _, meta = fixture()
        with patch("models.clip.tokenize", side_effect=word_tokens):
            model = prompts.ShallowPrompts(core, meta)
        self.assertEqual(sorted(name for name, p in model.named_parameters() if p.requires_grad),
                         ["context.leaf", "context.parent"])
        self.assertFalse(any(name.startswith("visual") for name in model.state_dict()))
        self.assertEqual(tuple(model.context["leaf"].shape), (4, 64))

    def test_nontrain_unknown_bad_labels_and_cache_aliases_rejected_before_core(self):
        for field, value in (("split", "val_known"), ("status", "intra"),
                             ("true_leaf", True), ("true_parent", 99)):
            _, source, group, meta = fixture()
            group["records"][0][field] = value
            with patch.object(prompts, "build_clip_core") as build:
                with self.assertRaises(ValueError):
                    self.fit(source, group, meta)
                build.assert_not_called()
        _, source, group, meta = fixture()
        group["image_sha256"].reverse()
        with self.assertRaisesRegex(ValueError, "ordering"):
            self.fit(source, group, meta)
        _, source, group, meta = fixture()
        group["record_feature_indices"][0] = 1
        with self.assertRaisesRegex(ValueError, "mapping"):
            self.fit(source, group, meta)
        _, source, group, meta = fixture()
        group["features"]["clip"][0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "cache"):
            self.fit(source, group, meta)

    def test_known_train_labels_actually_affect_optimization(self):
        _, source, group, meta = fixture()
        first, _ = self.fit(source, group, meta)
        modified = copy.deepcopy(group)
        for row in modified["records"]:
            row["true_leaf"] = (row["true_leaf"] + 1) % 3
            row["true_parent"] = meta["leaf_to_parent"][row["true_leaf"]]
        changed, _ = self.fit(source, modified, meta)
        self.assertFalse(torch.equal(first["context"]["leaf"], changed["context"]["leaf"]))


if __name__ == "__main__":
    unittest.main()
