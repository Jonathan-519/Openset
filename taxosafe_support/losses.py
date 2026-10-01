"""Known classification, intervention NLL, paired depth and control losses."""
import torch
from torch.nn import functional as F
from .support import validate_mapping


def _mean(values, valid, zero):
    valid = valid & torch.isfinite(values)
    return values[valid].mean() if bool(valid.any()) else zero


def _balanced_candidate_bce(logits, positive, active, valid_rows, zero):
    """Balance positive/negative candidate means within each query.

    A pseudo-unknown query has only negatives and is still trained. Empty
    support has no target at all, rather than an invented positive or negative.
    """
    active = active & valid_rows[:, None]
    safe = logits.masked_fill(~active, 0.)
    element = F.binary_cross_entropy_with_logits(safe, positive.to(safe.dtype), reduction="none")
    pos, neg = active & positive, active & ~positive
    pos_count, neg_count = pos.sum(-1), neg.sum(-1)
    pos_loss = (element * pos).sum(-1) / pos_count.clamp_min(1)
    neg_loss = (element * neg).sum(-1) / neg_count.clamp_min(1)
    groups = (pos_count > 0).to(safe.dtype) + (neg_count > 0).to(safe.dtype)
    values = (pos_loss + neg_loss) / groups.clamp_min(1)
    return values[groups > 0].mean() if bool((groups > 0).any()) else zero


def representation_losses(encoded, labels, leaf_to_parent, query_hashes=None,
                          margin=0.2, temperature=0.1):
    """Known-only batch contrastive supervision on the existing image graph.

    Parent positives are DIFFERENT leaves of the same parent, with other
    parents as negatives. Fine positives are the same leaf, with its siblings
    as negatives. Content aliases are excluded from both sides. Singletons or
    batches without both a positive and a negative give a connected zero loss.
    Counts refer to directed positive pairs for eligible anchors.
    """
    parent, fine = encoded["parent"].float(), encoded["fine"].float()
    if parent.ndim != 2 or parent.shape != fine.shape:
        raise ValueError("Encoded parent/fine features must have matching [B,D] shapes")
    if float(temperature) <= 0 or float(margin) < 0:
        raise ValueError("Contrastive temperature must be positive and margin nonnegative")
    labels = torch.as_tensor(labels, dtype=torch.long, device=parent.device)
    mapping = validate_mapping(leaf_to_parent).to(parent.device)
    if labels.shape != (len(parent),) or bool(((labels < 0) | (labels >= len(mapping))).any()):
        raise ValueError("Invalid representation labels")
    parents = mapping[labels]
    allowed = ~torch.eye(len(labels), dtype=torch.bool, device=parent.device)
    if query_hashes is not None:
        if len(query_hashes) != len(labels) or any(not isinstance(h, str) or not h for h in query_hashes):
            raise ValueError("One nonempty content hash per representation query is required")
        aliases = torch.tensor([[a == b for b in query_hashes] for a in query_hashes],
                               dtype=torch.bool, device=parent.device)
        allowed &= ~aliases
    same_leaf = labels[:, None] == labels[None, :]
    same_parent = parents[:, None] == parents[None, :]

    def contrastive(features, positive, negative):
        z = F.normalize(features, dim=-1)
        similarities = z @ z.T
        positive, negative = positive & allowed, negative & allowed
        valid = positive.any(-1) & negative.any(-1)
        if not bool(valid.any()):
            return features.sum() * 0., 0, 0
        positive, negative = positive[valid], negative[valid]
        # Additive positive margin asks for cross-species/within-species
        # separation without increasing the number of image forward passes.
        logits = (similarities[valid] - float(margin) * positive) / float(temperature)
        denominator = torch.logsumexp(logits.masked_fill(~(positive | negative), -torch.inf), -1)
        positive_mean = (logits * positive).sum(-1) / positive.sum(-1)
        return (denominator - positive_mean).mean(), int(valid.sum()), int(positive.sum())

    parent_loss, parent_anchors, parent_pairs = contrastive(parent, same_parent & ~same_leaf, ~same_parent)
    leaf_loss, leaf_anchors, leaf_pairs = contrastive(fine, same_leaf, same_parent & ~same_leaf)
    return {"parent_cross_species": parent_loss, "leaf_sibling": leaf_loss,
            "valid_parent_anchors": parent_anchors, "valid_leaf_anchors": leaf_anchors,
            "valid_parent_pairs": parent_pairs, "valid_leaf_pairs": leaf_pairs}


def hierarchical_losses(outputs, episodes, labels, leaf_to_parent, weights=None, margins=None):
    """Return differentiable loss components and valid full-support row count.

    Singleton images whose own hash is their only support cannot supply a full
    support training target; their known classifier loss belongs to the encoder
    path, supplied separately by the pipeline. They are not self-matched here.
    """
    weights, margins = weights or {}, margins or {}
    full = outputs["full"]
    device = full["log_probs"].device
    labels = torch.as_tensor(labels, dtype=torch.long, device=device)
    mapping = validate_mapping(leaf_to_parent).to(device)
    parents, rows = mapping[labels], torch.arange(len(labels), device=device)
    zero = full["parent_features"].sum() * 0
    full_valid = episodes["valid"]["full"].to(device) & full["active_leaves"][rows, labels]
    # Slice first: CE with an inactive target (-inf) must never reach backward.
    leaf = F.cross_entropy(full["leaf_logits"][full_valid], labels[full_valid]) if bool(full_valid.any()) else zero
    parent_valid = full["active_parents"][rows, parents]
    parent = F.cross_entropy(full["parent_logits"][parent_valid], parents[parent_valid]) if bool(parent_valid.any()) else zero
    nlls = []
    valid_by_name = {}
    for name, output in outputs.items():
        valid = episodes["valid"][name].to(device).clone()
        targets = episodes["targets"][name].to(device)
        selected = output["log_probs"][rows, targets]
        valid &= torch.isfinite(selected)
        valid_by_name[name] = valid
        if bool(valid.any()):
            nlls.append(-selected[valid].mean())
    episode = torch.stack(nlls).mean() if nlls else zero
    paired_parts, control_parts = [], []
    if "drop_leaf" in outputs:
        near = outputs["drop_leaf"]
        valid = valid_by_name["drop_leaf"] & full_valid
        lf, ln = full["leaf_accept_logits"][rows, parents], near["leaf_accept_logits"][rows, parents]
        rank = F.relu(float(margins.get("leaf", 1.0)) - lf + ln)
        paired_parts.append(_mean(rank, valid, zero))
        tolerance = float(margins.get("stability", 0.05))
        root_stability = F.relu((torch.sigmoid(full["root_logit"]) - torch.sigmoid(near["root_logit"])).abs() - tolerance)
        parent_evidence = "parent_membership_logits" if "parent_membership_logits" in full else "parent_logits"
        parent_stability = F.relu((torch.sigmoid(full[parent_evidence][rows, parents])
                                  - torch.sigmoid(near[parent_evidence][rows, parents])).abs() - tolerance)
        # Match the true parent itself: a high score for another parent must not
        # conceal a collapse in the parent's evidence after its leaf is removed.
        paired_parts.append(_mean((root_stability + parent_stability) / 2, valid, zero))
    if "drop_parent" in outputs:
        removed = outputs["drop_parent"]
        # Sigmoid scores stay finite when the entire reference tree is empty.
        before = outputs.get("drop_leaf", full)
        valid = valid_by_name["drop_parent"] & full_valid
        if "drop_leaf" in valid_by_name:
            # For singleton parents use full vs missing-parent, never fabricate near.
            use_near = valid_by_name["drop_leaf"]
            score = torch.where(use_near, torch.sigmoid(before["root_logit"]), torch.sigmoid(full["root_logit"]))
        else:
            score = torch.sigmoid(full["root_logit"])
        rank = F.relu(float(margins.get("root", 0.2)) - score + torch.sigmoid(removed["root_logit"]))
        paired_parts.append(_mean(rank, valid, zero))
    for name in ("control_leaf", "control_parent"):
        if name not in outputs:
            continue
        ctrl = outputs[name]
        valid = valid_by_name[name] & full_valid
        # Unrelated removals should preserve absolute acceptance evidence. Joint
        # class probability may change because its denominator genuinely changed.
        diff = ((torch.sigmoid(full["root_logit"]) - torch.sigmoid(ctrl["root_logit"])).abs()
                + (torch.sigmoid(full["leaf_accept_logits"][rows, parents])
                   - torch.sigmoid(ctrl["leaf_accept_logits"][rows, parents])).abs())
        control_parts.append(_mean(F.relu(diff - float(margins.get("stability", .05))), valid, zero))
    paired = torch.stack(paired_parts).mean() if paired_parts else zero
    control = torch.stack(control_parts).mean() if control_parts else zero
    losses = {"leaf": leaf, "parent": parent, "episode": episode, "paired": paired, "control": control}
    defaults = {"leaf": 1.0, "parent": .25, "episode": 1.0, "paired": .25, "control": .25}
    if "parent_membership_logits" in full:
        parent_bce, leaf_bce = [], []
        parent_positive = F.one_hot(parents, int(mapping.max()) + 1).bool()
        leaf_positive = F.one_hot(labels, len(mapping)).bool()
        for name, output in outputs.items():
            if "parent_membership_logits" not in output or "leaf_membership_logits" not in output:
                raise ValueError("Cannot mix legacy and decoupled episode outputs")
            # Use episode validity, not target likelihood: a query with no own
            # leaf reference is a legitimate negative for remaining leaves.
            valid = episodes["valid"][name].to(device)
            if bool((output["active_parents"] & valid[:, None]).any()):
                parent_bce.append(_balanced_candidate_bce(output["parent_membership_logits"],
                                  parent_positive, output["active_parents"], valid, zero))
            if bool((output["active_leaves"] & valid[:, None]).any()):
                leaf_bce.append(_balanced_candidate_bce(output["leaf_membership_logits"],
                                leaf_positive, output["active_leaves"], valid, zero))
        losses["membership_parent"] = torch.stack(parent_bce).mean() if parent_bce else zero
        losses["membership_leaf"] = torch.stack(leaf_bce).mean() if leaf_bce else zero
        defaults.update(membership_parent=1.0, membership_leaf=1.0)
    losses["total"] = sum(float(weights.get(key, defaults[key])) * value for key, value in losses.items())
    losses["valid_full_count"] = int(full_valid.sum())
    return losses
