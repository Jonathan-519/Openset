"""Known classification, intervention NLL, paired depth and control losses."""
import torch
from torch.nn import functional as F
from .support import validate_mapping


def _mean(values, valid, zero):
    valid = valid & torch.isfinite(values)
    return values[valid].mean() if bool(valid.any()) else zero


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
        parent_stability = F.relu((torch.sigmoid(full["parent_logits"][rows, parents])
                                  - torch.sigmoid(near["parent_logits"][rows, parents])).abs() - tolerance)
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
    losses["total"] = sum(float(weights.get(key, defaults[key])) * value for key, value in losses.items())
    losses["valid_full_count"] = int(full_valid.sum())
    return losses
