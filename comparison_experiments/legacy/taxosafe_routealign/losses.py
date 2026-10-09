"""TRAIN-only distillation of the raw heads used by membership routing."""
import math

import torch
from torch.nn import functional as F


def _class_mean(values, valid, labels):
    """Equal weight per observed query leaf, independent of batch frequency."""
    parts = [values[valid & (labels == label)].mean() for label in labels[valid].unique()]
    return torch.stack(parts).mean() if parts else values.sum() * 0.


def _masked_kl(student, teacher, active, temperature):
    valid = active.any(-1)
    student = student.masked_fill(~active, -torch.inf)
    teacher = teacher.masked_fill(~active, -torch.inf)
    student = torch.where(valid[:, None], student, torch.zeros_like(student))
    teacher = torch.where(valid[:, None], teacher, torch.zeros_like(teacher))
    log_student = F.log_softmax(student / temperature, dim=-1).masked_fill(~active, 0.)
    log_teacher = F.log_softmax(teacher / temperature, dim=-1).masked_fill(~active, 0.)
    probability = F.softmax(teacher / temperature, dim=-1).masked_fill(~active, 0.)
    return (probability * (log_teacher - log_student)).sum(-1) * temperature ** 2, valid


def evidence_anchor_losses(student, teacher, labels, leaf_to_parent, temperature=2.):
    """Return absolute membership and relative rank anchors, plus exposure counts.

    Both outputs must use each side's own TRAIN bank and the same query-content
    exclusions. Only jointly active candidates enter the objective. Membership
    balances the true candidate and remaining candidates within each query;
    every observed query leaf then receives equal weight. Rank KL balances
    parents when averaging conditional leaf distributions. Teacher tensors are
    always detached here, even if the caller forgot its no-grad context.
    """
    if not math.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError("Evidence distillation temperature must be finite and positive")
    device = student["parent_logits"].device
    labels = torch.as_tensor(labels, dtype=torch.long, device=device)
    mapping = torch.as_tensor(leaf_to_parent, dtype=torch.long, device=device)
    if (labels.ndim != 1 or mapping.ndim != 1 or not len(mapping)
            or bool(((labels < 0) | (labels >= len(mapping))).any())):
        raise ValueError("Invalid query labels or hierarchy")
    families = (("parent", mapping[labels], int(mapping.max()) + 1), ("leaf", labels, len(mapping)))
    losses, ranks, audit = [], [], {}
    for family, target, width in families:
        mask_name = "active_parents" if family == "parent" else "active_leaves"
        active = student[mask_name].bool() & teacher[mask_name].to(device).bool()
        expected = (len(labels), width)
        if active.shape != expected:
            raise ValueError("Evidence activity mask shape differs from the hierarchy")
        values = {}
        for kind in ("membership_logits", "logits"):
            field = family + "_" + kind
            a, b = student[field], teacher[field].detach().to(device)
            if a.shape != expected or b.shape != expected:
                raise ValueError("Evidence head shape differs from the hierarchy: " + field)
            if not bool((torch.isfinite(a) | ~active).all() & (torch.isfinite(b) | ~active).all()):
                raise ValueError("Active evidence logits must be finite: " + field)
            values[kind] = a.masked_fill(~active, 0.), b.masked_fill(~active, 0.)
        a, b = values["membership_logits"]
        element = F.smooth_l1_loss(a, b, reduction="none", beta=1.)
        positive = F.one_hot(target, width).bool() & active
        negative = active & ~positive
        pc, nc = positive.sum(-1), negative.sum(-1)
        groups = (pc > 0).to(a.dtype) + (nc > 0).to(a.dtype)
        per_query = ((element * positive).sum(-1) / pc.clamp_min(1)
                     + (element * negative).sum(-1) / nc.clamp_min(1)) / groups.clamp_min(1)
        losses.append(_class_mean(per_query, groups > 0, labels))
        a, b = values["logits"]
        if family == "parent":
            divergence, valid = _masked_kl(a, b, active, temperature)
        else:
            parts = [_masked_kl(a[:, mapping == p], b[:, mapping == p], active[:, mapping == p], temperature)
                     for p in range(int(mapping.max()) + 1)]
            divergences = torch.stack([part[0] for part in parts], -1)
            valid_parents = torch.stack([part[1] for part in parts], -1)
            divergence = (divergences * valid_parents).sum(-1) / valid_parents.sum(-1).clamp_min(1)
            valid = valid_parents.any(-1)
        ranks.append(_class_mean(divergence, valid, labels))
        audit[family + "_active_candidates"] = int(active.sum())
        audit[family + "_valid_queries"] = int((groups > 0).sum())
        audit[family + "_query_classes"] = int(labels[groups > 0].unique().numel())
    return torch.stack(losses).mean(), torch.stack(ranks).mean(), audit
