"""Frozen CLIP token access and a separately trained identity-initialized adapter."""
import torch
from torch import nn
from torch.nn import functional as F


@torch.no_grad()
def clip_image_features(core, images, return_patches=False):
    """The vanilla ViT arithmetic, with optional patches from the same forward."""
    visual = core.visual
    x = visual.conv1(images.to(dtype=core.dtype))
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
    cls = visual.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1)
    x = torch.cat((cls, x), dim=1)
    if x.shape[1:] != visual.positional_embedding.shape:
        raise ValueError("Image resolution differs from the frozen CLIP positional grid")
    x = visual.ln_pre(x + visual.positional_embedding.to(x.dtype))
    x = visual.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
    global_features = visual.ln_post(x[:, 0]) @ visual.proj
    result = {"global": F.normalize(global_features.float(), dim=-1)}
    if return_patches:
        result["patches"] = F.normalize((visual.ln_post(x[:, 1:]) @ visual.proj).float(), dim=-1)
    return result


class ResidualProjection(nn.Module):
    """Small feature-space residual; frozen CLIP/text weights are not children."""
    def __init__(self, dimension, bottleneck=64):
        super().__init__()
        if type(dimension) is not int or dimension < 2 or type(bottleneck) is not int or bottleneck < 1:
            raise ValueError("Invalid residual projection dimensions")
        self.dimension, self.bottleneck = dimension, bottleneck
        self.norm = nn.LayerNorm(dimension)
        self.down = nn.Linear(dimension, bottleneck)
        self.up = nn.Linear(bottleneck, dimension)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, features):
        if features.ndim != 2 or features.shape[1] != self.dimension:
            raise ValueError("Projection expects [N,D] image features")
        features = features.float()
        return F.normalize(features + self.up(F.gelu(self.down(self.norm(features)))), dim=-1)


def load_projection(state, dimension, bottleneck=64, device="cpu"):
    model = ResidualProjection(dimension, bottleneck)
    model.load_state_dict(state, strict=True)
    if any(not bool(torch.isfinite(value).all()) for value in model.state_dict().values()):
        raise ValueError("Projection state contains non-finite tensors")
    return model.to(device).eval().requires_grad_(False)
