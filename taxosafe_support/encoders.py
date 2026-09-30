"""Two token-level adaptations with one shared MaPLe visual forward.

The parent and fine branches adapt spatial tokens BEFORE their own learned
pooling, rather than applying two MLPs to the same pooled image embedding.
The expensive visual transformer and its trainable prompts remain shared to
bound runtime. This is a speed-conscious adaptation, not two separate CLIPs.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class TokenBranch(nn.Module):
    def __init__(self, dimension, bottleneck=64, local_tokens=4):
        super().__init__()
        if dimension < 2 or bottleneck < 1 or local_tokens < 1:
            raise ValueError("Invalid token branch dimensions")
        self.norm = nn.LayerNorm(dimension)
        self.down = nn.Linear(dimension, bottleneck)
        self.up = nn.Linear(bottleneck, dimension)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        self.queries = nn.Parameter(torch.randn(local_tokens, dimension) / math.sqrt(dimension))
        self.pool_gate = nn.Parameter(torch.tensor(0.01))

    def forward(self, global_features, spatial_features):
        tokens = torch.cat((global_features[:, None], spatial_features), dim=1).float()
        tokens = tokens + self.up(F.gelu(self.down(self.norm(tokens))))
        cls, patches = tokens[:, 0], tokens[:, 1:]
        # All patches retain differentiable weights; no hard top-k selection
        # silently disconnects the spatial prompt path.
        attention = torch.einsum("btd,kd->bkt", self.norm(patches), self.queries)
        attention = (attention / math.sqrt(tokens.shape[-1])).softmax(-1)
        local = attention @ patches
        pooled = cls + self.pool_gate.tanh() * local.mean(1)
        return F.normalize(pooled, dim=-1), F.normalize(local, dim=-1)


class SupportEncoder(nn.Module):
    def __init__(self, backbone, meta, settings):
        super().__init__()
        self.backbone = backbone
        self.meta = meta
        visual = getattr(getattr(backbone, "model", None), "image_encoder", None)
        if visual is not None and getattr(visual, "proj", None) is not None:
            dimension = int(visual.proj.shape[-1])
        else:
            # Tiny encoders can expose dimension for CPU integration tests.
            dimension = int(getattr(backbone, "dimension", settings.get("dimension", 512)))
        self.dimension = dimension
        kwargs = dict(bottleneck=int(settings.get("adapter_dim", 64)),
                      local_tokens=int(settings.get("local_tokens", 4)))
        self.parent_branch = TokenBranch(dimension, **kwargs)
        self.shared_encoder = bool(settings.get("shared_encoder", False))
        self.fine_branch = None if self.shared_encoder else TokenBranch(dimension, **kwargs)
        self.local_enabled = bool(settings.get("local_enabled", True))

    def text_features(self):
        names = self.meta["leaf_names"] + self.meta["parent_names"]
        return self.backbone.encode_text(names, normalize=True).float()

    def encode(self, images, text_features=None, classify=True):
        # The only visual forward: no detach on a real query's graph.
        global_features, spatial_features = self.backbone.encode_image_with_spatial(images, normalize=True)
        parent, parent_local = self.parent_branch(global_features, spatial_features)
        if self.shared_encoder:
            fine, fine_local = parent, parent_local
        else:
            fine, fine_local = self.fine_branch(global_features, spatial_features)
        result = {"parent": parent, "fine": fine,
                  "parent_local": parent_local if self.local_enabled else None,
                  "fine_local": fine_local if self.local_enabled else None}
        if classify:
            text = self.text_features() if text_features is None else text_features
            count = len(self.meta["leaf_names"])
            if text.shape != (count + len(self.meta["parent_names"]), self.dimension):
                raise ValueError("Text feature shape differs from the locked hierarchy")
            scale = self.backbone.model.logit_scale.exp().float()
            result.update(leaf_logits=scale * fine @ text[:count].T,
                          parent_logits=scale * parent @ text[count:].T)
        return result

    def forward(self, images):
        return self.encode(images)
