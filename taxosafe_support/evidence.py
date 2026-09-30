"""Candidate-aligned support evidence and normalized joint tree probabilities."""
import torch
from torch import nn
from torch.nn import functional as F
from .support import validate_mapping


def _masked_log_softmax(logits, active):
    # Rows with no candidates have no mass; avoid softmax(-inf,...,-inf).
    masked = logits.masked_fill(~active, -torch.inf)
    safe = torch.where(active.any(-1, keepdim=True), masked, torch.zeros_like(masked))
    return F.log_softmax(safe, dim=-1).masked_fill(~active, -torch.inf)


class CandidateMatcher(nn.Module):
    """Depth-shared absolute evidence: global, local, normalized distance.

    No leaf identity, parent identity, candidate count, or support count is an
    input. The cosine prior keeps initial scores useful before episode training.
    """
    def __init__(self, hidden_dim, temperature):
        super().__init__()
        self.residual = nn.Sequential(nn.Linear(3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        self.temperature = float(temperature)

    def forward(self, features):
        return (features[..., 0] - 0.5) / self.temperature + self.residual(features).squeeze(-1)


class HierarchicalEvidence(nn.Module):
    def __init__(self, dimension, leaf_to_parent, hidden_dim=32, temperature=0.1, local_enabled=True):
        super().__init__()
        mapping = validate_mapping(leaf_to_parent)
        if int(dimension) < 1 or int(hidden_dim) < 1 or float(temperature) <= 0:
            raise ValueError("Invalid evidence dimensions/temperature")
        self.dimension = int(dimension)
        self.num_leaves, self.num_parents = len(mapping), int(mapping.max()) + 1
        self.register_buffer("leaf_to_parent", mapping)
        self.parent_matcher = CandidateMatcher(int(hidden_dim), temperature)
        self.fine_matcher = CandidateMatcher(int(hidden_dim), temperature)
        self.root_bias = nn.Parameter(torch.zeros(()))
        self.local_bias = nn.Parameter(torch.zeros(()))
        self.local_enabled = bool(local_enabled)

    def _local(self, query, references, weights, fallback):
        if not self.local_enabled or query is None or references is None:
            return fallback
        query = F.normalize(query.float(), dim=-1)
        # Each query token matches its best token within each support image.
        # [B,N,Kquery,Ksupport] is bounded by the per-leaf bank cap.
        pair = torch.einsum("bkd,njd->bnkj", query, references)
        image_scores = pair.max(-1).values.mean(-1)
        return torch.einsum("bn,bnv->bv", image_scores, weights)

    def forward(self, encoded, bank, mask=None, query_hashes=None):
        parent, fine = encoded["parent"].float(), encoded["fine"].float()
        if parent.ndim != 2 or parent.shape != fine.shape or parent.shape[1] != self.dimension:
            raise ValueError("Encoded parent/fine features must match [B,D]")
        if not torch.equal(bank.leaf_to_parent, self.leaf_to_parent):
            raise ValueError("Bank and model taxonomy differ")
        parent, fine = F.normalize(parent, dim=-1), F.normalize(fine, dim=-1)
        stats = bank.statistics(len(parent), query_hashes=query_hashes, mask=mask)
        pg = torch.einsum("bd,bpd->bp", parent, stats["parent_proto"])
        fg = torch.einsum("bd,bcd->bc", fine, stats["fine_leaf"])
        pl = self._local(encoded.get("parent_local"), bank.parent_local, stats["parent_weights"], pg)
        fl = self._local(encoded.get("fine_local"), bank.fine_local, stats["leaf_weights"], fg)
        # Bounded TRAIN-normalized distances avoid exploding tiny-class radii.
        pd = -((1 - pg).clamp_min(0) / stats["parent_scale"]).clamp_max(40)
        fd = -((1 - fg).clamp_min(0) / stats["leaf_scale"]).clamp_max(40)
        parent_features, leaf_features = torch.stack((pg, pl, pd), -1), torch.stack((fg, fl, fd), -1)
        parent_logits = self.parent_matcher(parent_features).masked_fill(~stats["parent_active"], -torch.inf)
        leaf_logits = self.fine_matcher(leaf_features).masked_fill(~stats["leaf_active"], -torch.inf)
        root_logit = parent_logits.max(-1).values + self.root_bias
        local_logits = torch.stack([leaf_logits[:, self.leaf_to_parent == p].max(-1).values
                                    for p in range(self.num_parents)], -1) + self.local_bias
        log_q = _masked_log_softmax(parent_logits, stats["parent_active"])
        log_t = torch.full_like(leaf_logits, -torch.inf)
        for p in range(self.num_parents):
            children = self.leaf_to_parent == p
            log_t[:, children] = _masked_log_softmax(leaf_logits[:, children], stats["leaf_active"][:, children])
        log_root = F.logsigmoid(-root_logit)
        # Stable finite surrogates only for inactive branches; those branches
        # are then exactly masked. Prevent -inf + inf and NaN gradients.
        safe_root = torch.where(stats["parent_active"].any(-1), root_logit, torch.zeros_like(root_logit))
        safe_local = torch.where(stats["parent_active"], local_logits, torch.zeros_like(local_logits))
        log_parent = F.logsigmoid(safe_root)[:, None] + log_q + F.logsigmoid(-safe_local)
        log_leaf = (F.logsigmoid(safe_root)[:, None] + log_q[:, self.leaf_to_parent]
                    + F.logsigmoid(safe_local)[:, self.leaf_to_parent] + log_t)
        log_probs = torch.cat((log_root[:, None], log_parent, log_leaf), -1)
        return {"log_probs": log_probs, "root_logit": root_logit,
                "leaf_accept_logits": local_logits, "parent_logits": parent_logits,
                "leaf_logits": leaf_logits, "active_leaves": stats["leaf_active"],
                "active_parents": stats["parent_active"], "parent_features": parent_features,
                "leaf_features": leaf_features}
