import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from taxosafe_discovery import features
from taxosafe_discovery.models import clip_image_features
from tests import test_taxosafe_refine_pipeline as reference_fixture

tensor_snapshot = reference_fixture.tensor_snapshot


def fake_tokens(texts, context_length=8):
    values = torch.zeros(len(texts), context_length, dtype=torch.long)
    for i, text in enumerate(texts):
        values[i, 0] = 1
        values[i, 1:5] = torch.tensor([2 + sum(text.encode()[j::4]) % 110 for j in range(4)])
        values[i, 5] = 127
    return values


def core_source():
    from models.model import CLIP
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(61)
        vanilla = CLIP(16, 8, 1, 64, 4, 8, 128, 64, 1, 1).float().eval().requires_grad_(False)
    backbone = nn.Module()
    backbone.model = nn.Module()
    backbone.model.image_encoder = copy.deepcopy(vanilla.visual)
    backbone.model.text_encoder = nn.Module()
    for key in ("transformer", "positional_embedding", "ln_final", "text_projection"):
        setattr(backbone.model.text_encoder, key, copy.deepcopy(getattr(vanilla, key)))
    backbone.model.logit_scale = copy.deepcopy(vanilla.logit_scale)
    backbone.token_embedding = copy.deepcopy(vanilla.token_embedding)
    backbone.model.prompt_learner = nn.Linear(16, 16)
    encoder = nn.Module()
    encoder.backbone = backbone
    encoder.eval().requires_grad_(False)
    evidence = nn.Linear(2, 2).eval().requires_grad_(False)
    return vanilla, SimpleNamespace(encoder=encoder, evidence=evidence)


class FrozenCoreContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_strict_core_matches_original_and_ignores_prompt_tensors(self):
        original, source = core_source()
        snapshot = tensor_snapshot(source.encoder)
        rng = torch.random.get_rng_state()
        rebuilt = features.build_clip_core(source)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertEqual(features.core_tensor_hash(original), features.core_tensor_hash(rebuilt))
        images = torch.randn(3, 3, 8, 8)
        tokens = fake_tokens(["alpha", "beta", "gamma"])
        self.assertTrue(torch.equal(original.encode_image(images), rebuilt.encode_image(images)))
        self.assertTrue(torch.equal(original.encode_text(tokens), rebuilt.encode_text(tokens)))
        for key, value in source.encoder.state_dict().items():
            self.assertTrue(torch.equal(value, snapshot[key]), key)
        with torch.no_grad():
            source.encoder.backbone.model.prompt_learner.weight.add_(999.)
        again = features.build_clip_core(source)
        self.assertEqual(features.core_tensor_hash(rebuilt), features.core_tensor_hash(again))
        self.assertFalse(any(p.requires_grad for p in rebuilt.parameters()))

    def test_spatial_path_matches_vanilla_global_exactly(self):
        core, _ = core_source()
        images = torch.randn(2, 3, 8, 8)
        result = clip_image_features(core, images, return_patches=True)
        self.assertTrue(torch.equal(result["global"], F.normalize(core.encode_image(images), dim=-1)))
        self.assertEqual(tuple(result["patches"].shape), (2, 4, 16))
        torch.testing.assert_close(result["patches"].norm(dim=-1), torch.ones(2, 4))
        with self.assertRaisesRegex(ValueError, "resolution"):
            clip_image_features(core, torch.randn(2, 3, 12, 12))

    def test_template_ensemble_changes_features_and_has_correct_pooling(self):
        core, _ = core_source()
        names = ["leaf_alpha", "leaf_beta"]
        with patch("models.clip.tokenize", side_effect=fake_tokens):
            single = features.encode_templates(core, names, features.SINGLE_TEMPLATES)
            ensemble = features.encode_templates(core, names, features.ENSEMBLE_TEMPLATES)
            pieces = [features.encode_templates(core, names, [template]) for template in features.ENSEMBLE_TEMPLATES]
        torch.testing.assert_close(ensemble, F.normalize(torch.stack(pieces).mean(0), dim=-1))
        self.assertNotEqual(features.tensor_hash(single), features.tensor_hash(ensemble))
        self.assertFalse(single.requires_grad)

    def test_rejects_unfrozen_source_and_wrong_core_shape(self):
        _, source = core_source()
        source.encoder.train()
        with self.assertRaisesRegex(ValueError, "frozen"):
            features.build_clip_core(source)
        source.encoder.eval()
        source.encoder.backbone.model.image_encoder.extra = nn.Parameter(torch.ones(1), requires_grad=False)
        with self.assertRaises(RuntimeError):
            features.build_clip_core(source)


class CacheContracts(reference_fixture.FrozenPipelineContracts):
    # Reuse receipt-compatible synthetic source without inheriting its test suite.
    def test_cache_preserves_exact_baseline_aliases_and_source(self):
        from taxosafe_refine.importer import load_reference
        from taxosafe_support import pipeline as support
        self.make_source()
        source = load_reference(self.source, self.device)
        snapshots = [tensor_snapshot(source.encoder), tensor_snapshot(source.evidence)]
        groups = {"val_known": copy.deepcopy(self.groups["val_known"])}
        alias = dict(groups["val_known"][0], path="alias-path")
        groups["val_known"].append(alias)
        expected, _ = support.collect(groups, source.config, source.meta, source.encoder,
                                      source.evidence, source.bank, self.device)
        dimension = source.encoder.dimension
        dummy = nn.Linear(dimension, dimension).eval().requires_grad_(False)
        def image_features(core, images):
            value = images.flatten(1)[:, :dimension]
            return {"global": F.normalize(value, dim=-1)}
        def text_features(core, names, templates):
            return F.normalize(torch.arange(len(names) * dimension).reshape(len(names), dimension).float() + 1, dim=-1)
        with patch.object(features, "build_clip_core", return_value=dummy), \
                patch.object(features, "clip_image_features", side_effect=image_features), \
                patch.object(features, "encode_templates", side_effect=text_features):
            cache = features.collect_cache(source, groups, self.device)
        group = cache["groups"]["val_known"]
        self.assertEqual(group["records"], expected["val_known"])
        self.assertEqual(group["record_feature_indices"][-1], 0)
        self.assertEqual(len(group["features"]["clip"]), len(groups["val_known"]) - 1)
        for module, snapshot in zip((source.encoder, source.evidence), snapshots):
            self.assert_frozen(module, snapshot)
        self.assertFalse(cache["provenance"]["new_weights_downloaded"])
        self.assertFalse(any(name.startswith("test") for name in cache["groups"]))
        overlapping = dict(groups, val_extra=[dict(groups["val_known"][0], split="val_extra")])
        with self.assertRaisesRegex(ValueError, "overlaps"):
            features.collect_cache(source, overlapping, self.device)

    def test_cache_restores_source_on_clip_failure(self):
        from taxosafe_refine.importer import load_reference
        self.make_source()
        source = load_reference(self.source, self.device)
        snapshot = tensor_snapshot(source.encoder)
        with patch.object(features, "build_clip_core", side_effect=RuntimeError("synthetic core failure")):
            with self.assertRaisesRegex(RuntimeError, "synthetic"):
                features.collect_cache(source, {"train": self.groups["train"]}, self.device)
        self.assert_frozen(source.encoder, snapshot)


# Inherited tests belong to the original suite, not this module's test budget.
for _name in dir(reference_fixture.FrozenPipelineContracts):
    if _name.startswith("test_") and _name not in CacheContracts.__dict__:
        setattr(CacheContracts, _name, None)


if __name__ == "__main__":
    unittest.main()
