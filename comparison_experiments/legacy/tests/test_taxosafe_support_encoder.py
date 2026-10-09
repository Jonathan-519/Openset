"""Cheap CPU contracts for the real query gradient and preprocessing paths."""
import unittest

import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from taxosafe_support.encoders import SupportEncoder
from taxosafe_support.preprocessing import Letterbox, get_transform


META = {"leaf_names": ["a", "b", "c"], "parent_names": ["p", "q"],
        "leaf_to_parent": [0, 0, 1]}


class TinySpatialBackbone(nn.Module):
    dimension = 8

    def __init__(self):
        super().__init__()
        self.prompt_learner = nn.Linear(8, 8)
        self.model = nn.Module()
        self.model.register_buffer("logit_scale", torch.tensor(1.0))
        self.register_buffer("texts", F.normalize(torch.randn(5, 8), dim=-1))
        self.image_calls = 0
        self.text_calls = 0

    def encode_image_with_spatial(self, images, normalize=True):
        self.image_calls += 1
        tokens = self.prompt_learner(images)
        return F.normalize(tokens.mean(1), dim=-1), F.normalize(tokens, dim=-1)

    def encode_text(self, names, normalize=True):
        self.text_calls += 1
        if names != META["leaf_names"] + META["parent_names"]:
            raise AssertionError("Text classes/order changed")
        return self.texts


class EncoderTest(unittest.TestCase):
    def test_one_visual_forward_and_real_query_gradients(self):
        torch.manual_seed(3)
        backbone = TinySpatialBackbone()
        model = SupportEncoder(backbone, META, {"adapter_dim": 4, "local_tokens": 2})
        output = model(torch.randn(4, 6, 8))
        self.assertEqual(backbone.image_calls, 1)
        self.assertEqual(backbone.text_calls, 1)
        self.assertEqual(output["fine_local"].shape, (4, 2, 8))
        labels = torch.tensor([0, 1, 2, 0])
        loss = F.cross_entropy(output["leaf_logits"], labels)
        loss += F.cross_entropy(output["parent_logits"], torch.tensor([0, 0, 1, 0]))
        loss += output["fine_local"][:, :, 0].sum() + output["parent_local"][:, :, 1].sum()
        loss.backward()
        for parameter in (backbone.prompt_learner.weight, model.fine_branch.up.weight,
                          model.parent_branch.up.weight, model.fine_branch.queries,
                          model.parent_branch.queries):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0)

    def test_reference_encoding_avoids_text_pass(self):
        backbone = TinySpatialBackbone()
        model = SupportEncoder(backbone, META, {"adapter_dim": 4, "local_tokens": 2})
        output = model.encode(torch.randn(2, 6, 8), classify=False)
        self.assertEqual(backbone.text_calls, 0)
        self.assertNotIn("leaf_logits", output)

    def test_shared_and_no_local_ablations(self):
        model = SupportEncoder(TinySpatialBackbone(), META,
                               {"adapter_dim": 4, "shared_encoder": True, "local_enabled": False})
        output = model(torch.randn(2, 6, 8))
        self.assertTrue(torch.equal(output["fine"], output["parent"]))
        self.assertIsNone(output["fine_local"])
        self.assertIsNone(model.fine_branch)

    def test_letterbox_retains_both_ends(self):
        image = Image.new("RGB", (400, 100), "white")
        for x in range(20):
            for y in range(100):
                image.putpixel((x, y), (255, 0, 0))
                image.putpixel((399-x, y), (0, 0, 255))
        transformed = Letterbox()(image)
        self.assertEqual(transformed.size, (224, 224))
        self.assertEqual(transformed.getpixel((2, 112)), (255, 0, 0))
        self.assertEqual(transformed.getpixel((221, 112)), (0, 0, 255))

    def test_only_train_augments_and_eval_is_deterministic(self):
        train = get_transform({"resize_mode": "letterbox", "augment": True}, True)
        evaluate = get_transform({"resize_mode": "letterbox", "augment": True}, False)
        self.assertTrue(any(type(op).__name__ == "RandomHorizontalFlip" for op in train.transforms))
        self.assertFalse(any(type(op).__name__.startswith("Random") for op in evaluate.transforms))
        image = Image.new("RGB", (330, 80), (10, 30, 90))
        self.assertTrue(torch.equal(evaluate(image), evaluate(image)))


if __name__ == "__main__":
    unittest.main()
