"""Absolute image-to-reference evidence without centroid or candidate-count inputs.

Each cached image is a reference mode. Node membership pools a bounded number
of real witnesses, while parent membership gives every supported child leaf the
same weight. References are TRAIN-only detached features owned by SupportBank.
"""
import torch
from torch.nn import functional as F


def bidirectional_local_coverage(query, references, fallback):
    """Return query->reference and reference->query mean token coverage.

    A high match for one small shared part cannot silently stand for coverage
    of both token sets. Missing local features preserve a global-only ablation.
    """
    if query is None or references is None:
        return fallback, fallback
    query = F.normalize(query.float(), dim=-1)
    references = F.normalize(references.float(), dim=-1)
    pairs = torch.einsum("bkd,njd->bnkj", query, references)
    return pairs.max(-1).values.mean(-1), pairs.max(-2).values.mean(-1)


def pair_features(encoded, bank, local_enabled=True):
    """Three independent pair measurements; no node statistics are consulted."""
    result = {}
    for branch in ("parent", "fine"):
        query = F.normalize(encoded[branch].float(), dim=-1)
        references = getattr(bank, branch)
        global_similarity = query @ references.T
        forward, backward = bidirectional_local_coverage(
            encoded.get(branch + "_local") if local_enabled else None,
            getattr(bank, branch + "_local") if local_enabled else None,
            global_similarity)
        result[branch] = torch.stack((global_similarity, forward, backward), -1)
    return result


def _topk_per_leaf(logits, allowed, labels, num_leaves, topk):
    """Mask before selection; empty leaves have exactly zero probability mass."""
    # One bounded [B,C,N] operation avoids a GPU synchronization per leaf.
    belongs = F.one_hot(labels, num_leaves).T.bool()
    permitted = allowed[:, None, :] & belongs[None, :, :]
    values = logits[:, None, :].expand(-1, num_leaves, -1).masked_fill(~permitted, -torch.inf)
    best = values.topk(min(int(topk), logits.shape[1]), dim=-1).values
    finite = torch.isfinite(best)
    count = finite.sum(-1)
    pooled = best.masked_fill(~finite, 0.).sum(-1) / count.clamp_min(1)
    return pooled.masked_fill(count == 0, -torch.inf), count > 0


def aggregate_reference_logits(parent_logits, leaf_logits, allowed, labels,
                               leaf_to_parent, topk=2, num_parents=None):
    """Leaf top-k mean, then equal-leaf averaging for every parent.

    A reference may contribute only after query-content and intervention masks
    are applied. More images in one species cannot increase its parent weight.
    """
    if int(topk) < 1:
        raise ValueError("reference_topk must be positive")
    if parent_logits.shape != leaf_logits.shape or allowed.shape != leaf_logits.shape:
        raise ValueError("Reference logits and permissions must have matching [B,N] shapes")
    labels = torch.as_tensor(labels, dtype=torch.long, device=leaf_logits.device)
    mapping = torch.as_tensor(leaf_to_parent, dtype=torch.long, device=leaf_logits.device)
    if labels.shape != (leaf_logits.shape[1],):
        raise ValueError("One leaf label per reference is required")
    leaf, active = _topk_per_leaf(leaf_logits, allowed, labels, len(mapping), topk)
    parent_by_leaf, _ = _topk_per_leaf(parent_logits, allowed, labels, len(mapping), topk)
    if num_parents is None:
        num_parents = int(mapping.max()) + 1
    assignment = F.one_hot(mapping, int(num_parents)).to(parent_logits.dtype)
    count = active.to(parent_logits.dtype) @ assignment
    parent = parent_by_leaf.masked_fill(~active, 0.) @ assignment / count.clamp_min(1)
    return parent.masked_fill(count == 0, -torch.inf), leaf
