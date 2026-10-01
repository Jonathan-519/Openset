"""Small, content-deduplicated support banks built exclusively from known TRAIN.

The caller must enforce split provenance before construction. This object stores
no query data and recomputes statistics after content exclusion and interventions.
"""
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from torch.nn import functional as F


def validate_mapping(leaf_to_parent):
    mapping = torch.as_tensor(leaf_to_parent, dtype=torch.long)
    if mapping.ndim != 1 or not len(mapping):
        raise ValueError("leaf_to_parent must be a nonempty vector")
    if set(mapping.tolist()) != set(range(int(mapping.max()) + 1)):
        raise ValueError("Parent ids must be contiguous and nonnegative")
    return mapping


@dataclass(init=False)
class SupportBank:
    """Detached cache with a deterministic, bounded number of images per leaf.

    Hashes are image-content hashes, never paths. Repeated content is retained
    once; conflicting labels fail closed. Local token tensors are optional.
    """
    parent: torch.Tensor
    fine: torch.Tensor
    labels: torch.Tensor
    hashes: tuple
    leaf_to_parent: torch.Tensor
    parent_local: Optional[torch.Tensor]
    fine_local: Optional[torch.Tensor]
    max_per_leaf: int

    def __init__(self, parent, fine, labels, hashes: Sequence[str], leaf_to_parent,
                 parent_local=None, fine_local=None, max_per_leaf=8, required_leaf_mask=None):
        parent, fine = torch.as_tensor(parent), torch.as_tensor(fine)
        labels = torch.as_tensor(labels, dtype=torch.long, device=parent.device)
        mapping = validate_mapping(leaf_to_parent).to(parent.device)
        n = len(labels)
        if parent.ndim != 2 or fine.ndim != 2 or parent.shape != fine.shape or len(parent) != n:
            raise ValueError("Parent/fine supports must have matching [N,D] shapes")
        if not n or len(hashes) != n or any(not isinstance(h, str) or not h for h in hashes):
            raise ValueError("Every support needs a nonempty content hash")
        if labels.ndim != 1 or bool(((labels < 0) | (labels >= len(mapping))).any()):
            raise ValueError("Invalid support labels")
        if not torch.isfinite(parent).all() or not torch.isfinite(fine).all():
            raise ValueError("Support features must be finite")
        if int(max_per_leaf) < 1:
            raise ValueError("max_per_leaf must be positive")
        required = torch.ones(len(mapping), dtype=torch.bool, device=parent.device)
        if required_leaf_mask is not None:
            required = torch.as_tensor(required_leaf_mask, dtype=torch.bool, device=parent.device)
            if required.shape != mapping.shape or not bool(required.any()):
                raise ValueError("required_leaf_mask must select at least one known leaf")
            if bool((~required[labels]).any()):
                raise ValueError("Support contains an excluded holdout leaf")
            # Optional state is absent for legacy banks. Loading a strict
            # holdout cache revalidates both eligible and forbidden classes.
            self.required_leaf_mask = required.detach().clone()
        unique = {}
        for i, (h, c) in enumerate(zip(hashes, labels.tolist())):
            if h in unique and int(labels[unique[h]]) != c:
                raise ValueError("Identical content has conflicting leaf labels")
            unique.setdefault(h, i)
        chosen = []
        for c in range(len(mapping)):
            rows = [i for h, i in sorted(unique.items()) if int(labels[i]) == c]
            if not rows and bool(required[c]):
                raise ValueError("Missing training support for leaf %d" % c)
            chosen.extend(rows[:int(max_per_leaf)])
        index = torch.tensor(chosen, device=parent.device)
        self.parent = F.normalize(parent[index].detach().float(), dim=-1)
        self.fine = F.normalize(fine[index].detach().float(), dim=-1)
        self.labels, self.leaf_to_parent = labels[index].detach(), mapping.detach()
        self.hashes = tuple(hashes[i] for i in chosen)
        self.max_per_leaf = int(max_per_leaf)
        for name, value in (("parent_local", parent_local), ("fine_local", fine_local)):
            if value is None:
                setattr(self, name, None)
                continue
            value = torch.as_tensor(value, device=parent.device)
            if value.ndim != 3 or len(value) != n or value.shape[-1] != parent.shape[-1]:
                raise ValueError(name + " must have shape [N,K,D]")
            if value.shape[1] < 1 or not torch.isfinite(value).all():
                raise ValueError("Invalid local support tokens")
            setattr(self, name, F.normalize(value[index].detach().float(), dim=-1))

    @property
    def num_leaves(self):
        return len(self.leaf_to_parent)

    @property
    def num_parents(self):
        return int(self.leaf_to_parent.max()) + 1

    def to(self, device):
        # No resampling or reconstruction, so checkpoint ordering is preserved.
        for name in ("parent", "fine", "labels", "leaf_to_parent", "parent_local", "fine_local"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, value.to(device))
        if hasattr(self, "required_leaf_mask"):
            self.required_leaf_mask = self.required_leaf_mask.to(device)
        return self

    def state_dict(self):
        return {name: (value.detach().cpu().clone() if torch.is_tensor(value) else value)
                for name, value in vars(self).items()}

    @classmethod
    def from_state_dict(cls, state):
        return cls(**state)

    def permitted(self, batch_size, query_hashes=None, mask=None):
        """[B,N] support permissions; exclusions precede every statistic."""
        device = self.parent.device
        active = torch.ones(batch_size, self.num_leaves, dtype=torch.bool, device=device)
        if mask is not None:
            mask = torch.as_tensor(mask, dtype=torch.bool, device=device)
            if mask.shape != active.shape:
                raise ValueError("Support mask must have shape [B,C]")
            active &= mask
        allowed = active[:, self.labels]
        if query_hashes is not None:
            if len(query_hashes) != batch_size:
                raise ValueError("One content hash per query is required")
            excluded = torch.tensor([[q == h for h in self.hashes] for q in query_hashes],
                                    dtype=torch.bool, device=device)
            allowed &= ~excluded
        return allowed

    def statistics(self, batch_size, query_hashes=None, mask=None, scale_floor=0.025):
        """Leaf means and leaf-balanced parent means, with recomputed radii.

        All scales use only the currently permitted TRAIN support. No removed
        leaf, excluded query, or validation image contributes indirectly.
        """
        allowed = self.permitted(batch_size, query_hashes, mask)
        one_hot = F.one_hot(self.labels, self.num_leaves).to(self.parent.dtype)
        weights = allowed.to(self.parent.dtype).unsqueeze(-1) * one_hot.unsqueeze(0)
        counts = weights.sum(1)
        leaf_active = counts > 0
        leaf_weights = weights / counts.clamp_min(1).unsqueeze(1)
        parent_mean = torch.einsum("bnc,nd->bcd", leaf_weights, self.parent)
        fine_mean = torch.einsum("bnc,nd->bcd", leaf_weights, self.fine)
        parent_leaf = F.normalize(parent_mean, dim=-1)
        fine_leaf = F.normalize(fine_mean, dim=-1)
        pmap = F.one_hot(self.leaf_to_parent, self.num_parents).to(self.parent.dtype)
        leaf_to_parent = leaf_active.to(self.parent.dtype).unsqueeze(-1) * pmap.unsqueeze(0)
        children = leaf_to_parent.sum(1)
        parent_active = children > 0
        leaf_to_parent = leaf_to_parent / children.clamp_min(1).unsqueeze(1)
        parent_proto = F.normalize(torch.einsum("bcp,bcd->bpd", leaf_to_parent, parent_leaf), dim=-1)
        # E[1-cos(reference,prototype)] can be computed from the mean directly;
        # avoid a [B,N,C,D] pairwise expansion in each training intervention.
        leaf_scale = (1 - (fine_mean * fine_leaf).sum(-1)).clamp_min(float(scale_floor))
        # Each leaf contributes equal mass, irrespective of image count.
        parent_weights = torch.einsum("bnc,bcp->bnp", leaf_weights, leaf_to_parent)
        balanced_parent_mean = torch.einsum("bcp,bcd->bpd", leaf_to_parent, parent_mean)
        parent_scale = (1 - (balanced_parent_mean * parent_proto).sum(-1)).clamp_min(float(scale_floor))
        return {"allowed": allowed, "leaf_active": leaf_active, "parent_active": parent_active,
                "leaf_weights": leaf_weights, "parent_weights": parent_weights,
                "leaf_to_parent_weights": leaf_to_parent, "fine_leaf": fine_leaf,
                "parent_leaf": parent_leaf, "parent_proto": parent_proto,
                "leaf_scale": leaf_scale, "parent_scale": parent_scale}
