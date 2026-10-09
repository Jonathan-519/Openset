"""Full-mass entropic transport between two REAL spatial token grids.

Uniform row/column capacities prohibit many query patches borrowing a single
reference patch. Marginal rounding enforces capacities after finite Sinkhorn
iterations. Nothing is removed as background, invisible or difficult to match.
This is an OT-inspired verifier, not a reproduction of DeepEMD.
"""
import math

import torch
from torch.nn import functional as F

DESCRIPTORS = ("mean_similarity", "query_worst_quarter_similarity",
               "reference_worst_quarter_similarity", "query_soft_coverage",
               "reference_soft_coverage", "high_cost_mass", "transport_entropy",
               "neighbour_distance_distortion")


def grid_positions(count):
    side = int(math.sqrt(count))
    if side * side != count or side < 2:
        raise ValueError("Complete square spatial grid with at least 2x2 tokens required")
    return torch.tensor([((x + .5) / side, (y + .5) / side)
                         for y in range(side) for x in range(side)], dtype=torch.float32)


def transport(cost, epsilon=.1, iterations=40):
    if cost.ndim != 3 or min(cost.shape) < 1 or not bool(torch.isfinite(cost).all()):
        raise ValueError("Transport expects finite [pairs,query_tokens,reference_tokens] costs")
    if not epsilon > 0 or type(iterations) is not int or iterations < 1:
        raise ValueError("Invalid transport settings")
    n, m = cost.shape[-2:]
    kernel = -cost / epsilon
    u, v = torch.zeros_like(cost[:, :, 0]), torch.zeros_like(cost[:, 0, :])
    for _ in range(iterations):
        u = -math.log(n) - torch.logsumexp(kernel + v[:, None, :], dim=2)
        v = -math.log(m) - torch.logsumexp(kernel + u[:, :, None], dim=1)
    plan = torch.exp(kernel + u[:, :, None] + v[:, None, :])
    # Differentiable nonnegative marginal rounding: shrink surplus rows/cols,
    # then fill the two deficits with a rank-one transport. No mass discarded.
    tiny = torch.finfo(plan.dtype).tiny
    plan = plan * ((1. / n) / plan.sum(2).clamp_min(tiny)).clamp(max=1.)[:, :, None]
    plan = plan * ((1. / m) / plan.sum(1).clamp_min(tiny)).clamp(max=1.)[:, None, :]
    row_deficit = (1. / n - plan.sum(2)).clamp_min(0.)
    col_deficit = (1. / m - plan.sum(1)).clamp_min(0.)
    denominator = row_deficit.sum(1).clamp_min(1e-12)
    plan = plan + row_deficit[:, :, None] * col_deficit[:, None, :] / denominator[:, None, None]
    error = torch.maximum((plan.sum(2) - 1. / n).abs().amax(1),
                          (plan.sum(1) - 1. / m).abs().amax(1))
    if not bool(torch.isfinite(plan).all()) or bool((error > 2e-5).any()):
        raise ValueError("Full matching failed its marginal-capacity check")
    return plan, error


def pair_evidence(query, reference, positions, settings):
    """One descriptor/vector per complete reference, never patchwise max fusion."""
    if query.ndim != 3 or query.shape != reference.shape or query.shape[1] != len(positions):
        raise ValueError("Matching needs aligned complete square grids")
    query, reference = F.normalize(query.float(), dim=-1), F.normalize(reference.float(), dim=-1)
    cost = (1. - query @ reference.transpose(1, 2)).clamp(0., 2.)
    plan, capacity_error = transport(cost, settings["epsilon"], settings["iterations"])
    count = query.shape[1]
    qcost, rcost = (plan * cost).sum(2) * count, (plan * cost).sum(1) * count
    tail = max(1, (count + 3) // 4)
    scale = settings["epsilon"]
    coverage_q = torch.sigmoid((settings["coverage_cost"] - qcost) / scale).mean(1)
    coverage_r = torch.sigmoid((settings["coverage_cost"] - rcost) / scale).mean(1)
    high_cost = (plan * (cost > settings["coverage_cost"]).to(cost.dtype)).sum((1, 2))
    entropy = -(plan * plan.clamp_min(1e-12).log()).sum((1, 2)) / math.log(count * count)
    positions = positions.to(query)
    correspondence = (plan * count) @ positions
    side = int(math.sqrt(count))
    edges = [(i, j) for i in range(count) for j in (i + 1, i + side)
             if j < count and (j == i + side or i // side == j // side)]
    left = torch.tensor([x for x, _ in edges], device=query.device)
    right = torch.tensor([x for _, x in edges], device=query.device)
    # Neighbour lengths are invariant to a common rotation/translation. This
    # diagnostic does not impose an upright animal or reject an orientation.
    distortion = ((correspondence[:, left] - correspondence[:, right]).norm(dim=-1) - 1. / side).abs().mean(1)
    desc = torch.stack((1. - (plan * cost).sum((1, 2)),
                        1. - qcost.topk(tail, dim=1).values.mean(1),
                        1. - rcost.topk(tail, dim=1).values.mean(1),
                        coverage_q, coverage_r, high_cost, entropy, distortion), dim=1)
    aligned = (plan * count) @ reference
    vector = torch.cat(((query - aligned).abs().mean(1), (query * aligned).mean(1)), dim=1)
    diagnostics = dict(capacity_error=capacity_error, query_patch_cost=qcost,
                       reference_patch_cost=rcost, target_patch=plan.argmax(2),
                       matched_mass=plan.sum((1, 2)))
    return torch.cat((vector, desc), dim=1), desc, diagnostics
