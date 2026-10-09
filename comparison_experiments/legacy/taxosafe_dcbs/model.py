"""Factorized stop evidence; the original MaPLe leaf classifier is retained."""
import torch
from torch import nn
from torch.nn import functional as F


class _Reverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, strength):
        ctx.strength = strength
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.strength * grad, None


def reverse_gradient(x, strength=1.0):
    return _Reverse.apply(x, float(strength))


def projection(input_dim, output_dim):
    return nn.Sequential(nn.Linear(input_dim, output_dim), nn.LayerNorm(output_dim),
                         nn.GELU(), nn.Linear(output_dim, output_dim))


class DCBSHeads(nn.Module):
    """P + root STOP, and a known-child/STOP classifier for each parent.

    Synthetic inputs live in the SAME backbone space as real inputs, before
    either projection. A shared leaf weight matrix implements local heads.
    """
    def __init__(self, input_dim, leaf_to_parent, num_parents, settings):
        super().__init__()
        mapping = torch.as_tensor(leaf_to_parent, dtype=torch.long)
        if mapping.ndim != 1 or not len(mapping):
            raise ValueError("leaf_to_parent must be a nonempty vector")
        if set(mapping.tolist()) != set(range(num_parents)):
            raise ValueError("Every parent must have at least one known child")
        self.register_buffer("leaf_to_parent", mapping)
        self.num_parents, self.num_leaves = num_parents, len(mapping)
        dim = int(settings.get("projection_dim", 128))
        self.shared = bool(settings.get("shared_projection", False))
        self.taxonomy = projection(input_dim, dim)
        self.fine = None if self.shared else projection(input_dim, dim)
        self.root = nn.Linear(dim, num_parents + 1, bias=False)
        self.leaf_head = nn.Linear(dim, self.num_leaves, bias=False)
        self.local_stop = nn.Linear(dim, num_parents, bias=False)
        self.parent_adversary = nn.Linear(dim, num_parents)
        self.scale = float(settings.get("logit_scale", 10.0))
        if dim < 2 or self.scale <= 0:
            raise ValueError("Invalid projection dimension/logit scale")

    def embeddings(self, h):
        h = h.float()
        zt = F.normalize(self.taxonomy(h), dim=-1)
        zf = zt if self.shared else F.normalize(self.fine(h), dim=-1)
        return zt, zf

    def forward(self, h):
        zt, zf = self.embeddings(h)
        cosine = lambda z, layer: self.scale * F.linear(z, F.normalize(layer.weight, dim=-1))
        return {"taxonomy": zt, "fine": zf, "root": cosine(zt, self.root),
                "leaf": cosine(zf, self.leaf_head), "stop": cosine(zf, self.local_stop)}

    def local_logits(self, output, parent):
        children = torch.where(self.leaf_to_parent == parent)[0]
        return torch.cat((output["leaf"][:, children], output["stop"][:, parent:parent + 1]), 1)

    def adversarial_logits(self, h):
        # HA updates fine projection/adversary only. Detached h protects the
        # shared backbone and taxonomy evidence from the reversed gradient.
        _, zf = self.embeddings(h.detach())
        return self.parent_adversary(reverse_gradient(zf))


def _local_ce(heads, output, labels, unknown_parents=None):
    total = output["leaf"].sum() * 0.0
    count = 0
    parents = heads.leaf_to_parent[labels] if unknown_parents is None else unknown_parents
    for p in range(heads.num_parents):
        mask = parents == p
        n = int(mask.sum())
        if not n:
            continue
        child = torch.where(heads.leaf_to_parent == p)[0]
        if unknown_parents is None:
            target = (labels[mask, None] == child[None, :]).long().argmax(1)
        else:
            target = torch.full((n,), len(child), device=parents.device, dtype=torch.long)
        total = total + F.cross_entropy(heads.local_logits(output, p)[mask], target, reduction="sum")
        count += n
    return total / max(count, 1)


def sibling_margin_loss(heads, output, labels, margin=0.1):
    cosine = output["leaf"] / heads.scale
    rows = torch.arange(len(labels), device=labels.device)
    positive = cosine[rows, labels]
    siblings = heads.leaf_to_parent[None, :] == heads.leaf_to_parent[labels, None]
    siblings[rows, labels] = False
    rankable = siblings.any(1)
    if not bool(rankable.any()):
        return cosine.sum() * 0.0
    negative = cosine.masked_fill(~siblings, -2.0).max(1).values
    return F.relu(float(margin) + negative[rankable] - positive[rankable]).mean()


def dcbs_loss(heads, h, labels, leaf_logits, parent_logits, synthetic, settings, novelty_weight=1.0):
    """known=(p,c), near=(p,STOP), extra=(root STOP); type-balanced means."""
    output = heads(h)
    parents = heads.leaf_to_parent[labels]
    zero = output["root"].sum() * 0.0
    losses = {"leaf": F.cross_entropy(leaf_logits.float(), labels),
              "parent": F.cross_entropy(parent_logits.float(), parents),
              "root_known": F.cross_entropy(output["root"], parents),
              "local_known": _local_ce(heads, output, labels),
              "near_root": zero, "near_stop": zero, "extra_stop": zero,
              "ha": zero, "margin": zero}
    if len(synthetic["near"]):
        near_output = heads(synthetic["near"].detach())
        losses["near_root"] = F.cross_entropy(near_output["root"], synthetic["near_parent"])
        losses["near_stop"] = _local_ce(heads, near_output, None, synthetic["near_parent"])
    if len(synthetic["extra"]):
        extra_output = heads(synthetic["extra"].detach())
        target = torch.full((len(synthetic["extra"]),), heads.num_parents, dtype=torch.long, device=h.device)
        losses["extra_stop"] = F.cross_entropy(extra_output["root"], target)
    weights = settings.get("loss", {})
    if float(weights.get("ha", 0.0)) > 0:
        if heads.shared:
            raise ValueError("HA requires separate projections; disable HA for lite")
        losses["ha"] = F.cross_entropy(heads.adversarial_logits(h), parents)
    if float(weights.get("margin", 0.0)) > 0:
        losses["margin"] = sibling_margin_loss(heads, output, labels, settings.get("sibling_margin", 0.1))
    defaults = {"leaf": 1.0, "parent": 0.25, "root_known": 0.5, "local_known": 0.5,
                "near_root": 0.5, "near_stop": 0.5, "extra_stop": 0.5, "ha": 0.0, "margin": 0.0}
    total = zero
    for key, value in losses.items():
        ramp = novelty_weight if key in {"near_root", "near_stop", "extra_stop", "ha", "margin"} else 1.0
        total = total + float(weights.get(key, defaults[key])) * ramp * value
    return total, losses
