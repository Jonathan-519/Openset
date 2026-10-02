"""Candidate-aligned support evidence and normalized joint tree probabilities."""
import torch
from torch import nn
from torch.nn import functional as F
from .support import validate_mapping
from .reference import pair_features, aggregate_reference_logits


def _masked_log_softmax(logits, active):
    # Rows with no candidates have no mass; avoid softmax(-inf,...,-inf).
    masked = logits.masked_fill(~active, -torch.inf)
    safe = torch.where(active.any(-1, keepdim=True), masked, torch.zeros_like(masked))
    return F.log_softmax(safe, dim=-1).masked_fill(~active, -torch.inf)


def _masked_logsumexp(values, active, dim=-1):
    """Exact absent mass with a finite backward path for empty rows."""
    has_values = active.any(dim=dim, keepdim=True)
    masked = values.masked_fill(~active, -torch.inf)
    safe = torch.where(has_values, masked, torch.zeros_like(masked))
    return torch.logsumexp(safe, dim=dim).masked_fill(~has_values.squeeze(dim), -torch.inf)


class CandidateMatcher(nn.Module):
    """Depth-shared candidate score: global, local, normalized distance.

    No leaf identity, parent identity, candidate count, or support count is an
    input. The cosine prior keeps initial scores useful before episode training.
    Decoupled mode gives ranking and membership independent instances.
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
    def __init__(self, dimension, leaf_to_parent, hidden_dim=32, temperature=0.1,
                 local_enabled=True, decoupled=False, membership_mode="prototype", reference_topk=2):
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
        self.decoupled = bool(decoupled)
        if membership_mode not in ("prototype", "reference"):
            raise ValueError("membership_mode must be prototype or reference")
        if membership_mode == "reference" and not self.decoupled:
            raise ValueError("Reference membership requires decoupled evidence")
        if int(reference_topk) < 1:
            raise ValueError("reference_topk must be positive")
        self.membership_mode, self.reference_topk = membership_mode, int(reference_topk)
        # Do not construct new modules in legacy mode: its state keys, random
        # initialization sequence, and forward arithmetic remain unchanged.
        if self.decoupled:
            if self.membership_mode == "reference":
                self.parent_reference = CandidateMatcher(int(hidden_dim), temperature)
                self.fine_reference = CandidateMatcher(int(hidden_dim), temperature)
            else:
                self.parent_membership = CandidateMatcher(int(hidden_dim), temperature)
                self.fine_membership = CandidateMatcher(int(hidden_dim), temperature)

    def _local(self, query, references, weights, fallback):
        if not self.local_enabled or query is None or references is None:
            return fallback
        query = F.normalize(query.float(), dim=-1)
        # Each query token matches its best token within each support image.
        # [B,N,Kquery,Ksupport] is bounded by the per-leaf bank cap.
        pair = torch.einsum("bkd,njd->bnkj", query, references)
        image_scores = pair.max(-1).values.mean(-1)
        return torch.einsum("bn,bnv->bv", image_scores, weights)

    def reference_pairs(self, encoded, bank):
        """Compute one query batch's pair graph for explicit episode reuse.

        This is an ephemeral caller-owned value, never a persistent model cache.
        It must not be retained across query encodings, support refreshes, or
        optimizer steps. Permissions are applied separately in each forward.
        """
        if self.membership_mode != "reference":
            raise ValueError("Pair evidence is only available in reference mode")
        features = pair_features(encoded, bank, local_enabled=self.local_enabled)
        return {"parent": self.parent_reference(features["parent"]) + self.root_bias,
                "fine": self.fine_reference(features["fine"]) + self.local_bias,
                "_encoded_id": id(encoded), "_bank_id": id(bank)}

    def forward(self, encoded, bank, mask=None, query_hashes=None, reference_pair_logits=None):
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
        if self.decoupled:
            if self.membership_mode == "reference":
                pairs = (self.reference_pairs(encoded, bank) if reference_pair_logits is None
                         else reference_pair_logits)
                if pairs.get("_encoded_id") != id(encoded) or pairs.get("_bank_id") != id(bank):
                    raise ValueError("Pair evidence must come from the same query encoding and support bank")
                allowed = stats["allowed"]
                if pairs["parent"].shape != allowed.shape or pairs["fine"].shape != allowed.shape:
                    raise ValueError("Cached pair evidence has an invalid shape")
                # Biases are included once, before pooling. They train through
                # both direct pair BCE and the unchanged joint output objective.
                parent_pairs = pairs["parent"].masked_fill(~allowed, -torch.inf)
                leaf_pairs = pairs["fine"].masked_fill(~allowed, -torch.inf)
                pm, lm = aggregate_reference_logits(parent_pairs, leaf_pairs, allowed, bank.labels,
                                                     self.leaf_to_parent, self.reference_topk,
                                                     num_parents=self.num_parents)
                output = self._decoupled_output(parent_features, leaf_features, parent_logits,
                                                leaf_logits, stats, membership_logits=(pm, lm))
                output.update(reference_parent_logits=parent_pairs, reference_leaf_logits=leaf_pairs,
                              reference_allowed=allowed, reference_labels=bank.labels,
                              reference_leaf_present=F.one_hot(bank.labels, self.num_leaves).bool().any(0))
                return output
            return self._decoupled_output(parent_features, leaf_features, parent_logits, leaf_logits, stats)
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

    def _decoupled_output(self, parent_features, leaf_features, parent_logits, leaf_logits, stats,
                          membership_logits=None):
        """Route mass using the membership of the SAME identity candidate.

        q and t only rank identities. Independent sigmoid gates r_p and a_c
        express membership: leaf=q_p*r_p*t_c*a_c, parent=q_p*r_p*sum(t_c*(1-a_c)),
        root=sum(q_p*(1-r_p)). Thus a convincing sibling cannot lend its
        membership evidence to a different predicted leaf or parent.
        """
        pa, la = stats["parent_active"], stats["leaf_active"]
        # Existing scalar parameters remain useful: they are shared membership
        # intercepts, supervised by BCE as well as the joint episode likelihood.
        if membership_logits is None:
            pm = (self.parent_membership(parent_features) + self.root_bias).masked_fill(~pa, -torch.inf)
            lm = (self.fine_membership(leaf_features) + self.local_bias).masked_fill(~la, -torch.inf)
        else:
            pm, lm = membership_logits
        log_q = _masked_log_softmax(parent_logits, pa)
        log_t = torch.full_like(leaf_logits, -torch.inf)
        local_accept, local_reject = [], []
        for p in range(self.num_parents):
            children = self.leaf_to_parent == p
            conditional = _masked_log_softmax(leaf_logits[:, children], la[:, children])
            log_t[:, children] = conditional
            local_accept.append(_masked_logsumexp(conditional + F.logsigmoid(lm[:, children]), la[:, children]))
            local_reject.append(_masked_logsumexp(conditional + F.logsigmoid(-lm[:, children]), la[:, children]))
        log_local_accept, log_local_reject = torch.stack(local_accept, -1), torch.stack(local_reject, -1)
        log_root_accept = _masked_logsumexp(log_q + F.logsigmoid(pm), pa)
        log_root_reject = _masked_logsumexp(log_q + F.logsigmoid(-pm), pa)
        has_parent = pa.any(-1)
        log_root_reject = torch.where(has_parent, log_root_reject, torch.zeros_like(log_root_reject))
        root_logit = log_root_accept - log_root_reject
        # Never subtract -inf from -inf for an inactive parent, even if the
        # resulting slot would subsequently be masked.
        local_logits = (torch.where(pa, log_local_accept, torch.zeros_like(log_local_accept))
                        - torch.where(pa, log_local_reject, torch.zeros_like(log_local_reject)))
        local_logits = local_logits.masked_fill(~pa, -torch.inf)
        log_parent = log_q + F.logsigmoid(pm) + log_local_reject
        log_leaf = (log_q[:, self.leaf_to_parent] + F.logsigmoid(pm)[:, self.leaf_to_parent]
                    + log_t + F.logsigmoid(lm))
        return {"log_probs": torch.cat((log_root_reject[:, None], log_parent, log_leaf), -1),
                "root_logit": root_logit, "leaf_accept_logits": local_logits,
                "parent_logits": parent_logits, "leaf_logits": leaf_logits,
                "parent_membership_logits": pm, "leaf_membership_logits": lm,
                "active_leaves": la, "active_parents": pa,
                "parent_features": parent_features, "leaf_features": leaf_features,
                "decoupled": True}
