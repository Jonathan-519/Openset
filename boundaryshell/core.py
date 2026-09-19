"""Pure PyTorch geometry; no CLIP, dataset, or evaluation dependencies."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def normalize(x):
    return F.normalize(x.float(), dim=-1, eps=1e-8)


class OpenProjectionHead(nn.Module):
    """Fixed orthogonal projection plus a small trainable residual."""
    def __init__(self, input_dim, output_dim=128, hidden_dim=128):
        super().__init__()
        if not 2 <= output_dim <= input_dim:
            raise ValueError("output_dim must be in [2, input_dim]")
        q, _ = torch.linalg.qr(torch.randn(input_dim, output_dim))
        self.register_buffer("base", q)
        self.residual = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(),
                                      nn.Linear(hidden_dim, output_dim))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, x):
        x = normalize(x)
        return normalize(x @ self.base + 0.25 * self.residual(x))


def distances(z, centers):
    # Squared chord distance avoids acos' singular derivative at +/-1.
    return (2 - 2 * normalize(z) @ normalize(centers).T).clamp_min(1e-8).sqrt()


@torch.no_grad()
def fit_support(z, labels, leaf_to_parent, quantile=0.95, shrinkage=20., min_radius=0.03):
    if not 0 < quantile < 1 or shrinkage < 0 or min_radius <= 0:
        raise ValueError("Invalid support settings")
    z = normalize(z)
    labels = labels.long()
    mapping = leaf_to_parent.to(z.device).long()
    n_classes = len(mapping)
    if len(z) != len(labels) or not len(z) or not torch.isfinite(z).all():
        raise ValueError("Empty, inconsistent, or non-finite reference bank")
    if labels.min() < 0 or labels.max() >= n_classes:
        raise ValueError("Reference labels outside known classes")
    if mapping.min() < 0 or sorted(mapping.unique().tolist()) != list(range(int(mapping.max()) + 1)):
        raise ValueError("Parent IDs must be contiguous")
    counts = torch.bincount(labels, minlength=n_classes)
    if torch.any(counts < 2):
        raise ValueError("Every known class needs at least two train references")
    centers = torch.stack([normalize(z[labels == c].mean(0)) for c in range(n_classes)])
    own_distance = (z - centers[labels]).norm(dim=-1)
    raw = torch.stack([torch.quantile(own_distance[labels == c], quantile) for c in range(n_classes)])
    pooled = torch.stack([torch.quantile(own_distance[mapping[labels] == p], quantile)
                          for p in range(int(mapping.max()) + 1)])
    w = counts.float() / (counts.float() + shrinkage)
    radius = (w * raw + (1 - w) * pooled[mapping]).clamp(min=min_radius, max=1.95)
    parent_centers = torch.stack([normalize(z[mapping[labels] == p].mean(0))
                                  for p in range(len(pooled))])
    parent_radius = torch.stack([torch.quantile(
        (z[mapping[labels] == p] - parent_centers[p]).norm(dim=-1), 0.99)
        for p in range(len(pooled))]).clamp_min(min_radius)
    return dict(centers=centers, radius=radius, raw_radius=raw, counts=counts,
                parent_centers=parent_centers, parent_radius=parent_radius,
                leaf_to_parent=mapping)


@torch.no_grad()
def synthesize_shell(x, labels, support, mode="taxonomy", k=3, margin=0.08,
                     noise=0.15, attempts=4):
    """Generate in frozen classifier space, outside EVERY known support.

    Generating in the trainable output space and then scoring with the same
    supports makes the reject loss tautological. A frozen input-space teacher
    defines shell examples; the projection must learn their separation.
    """
    if mode not in {"taxonomy", "knn", "random"} or margin <= 0 or k < 1 or attempts < 1:
        raise ValueError("Invalid generator settings")
    centers, radii = support["centers"], support["radius"]
    mapping = support["leaf_to_parent"]
    n = len(centers)
    if n < 2:
        raise ValueError("At least two classes required")
    pair = distances(centers, centers)
    pair.fill_diagonal_(float("inf"))
    if mode == "taxonomy":
        pair.masked_fill_(mapping[:, None] != mapping[None, :], float("inf"))
    neighbors = pair.argsort(dim=1)[:, :min(k, n - 1)]
    outputs, parents, sources = [], [], []
    proposed = 0
    for _ in range(attempts):
        source = labels.long()
        if mode == "random":
            target = (source + torch.randint(1, n, source.shape, device=x.device)) % n
        else:
            rank = torch.randint(neighbors.shape[1], source.shape, device=x.device)
            target = neighbors[source, rank]
        eligible = torch.isfinite(pair[source, target]) if mode == "taxonomy" else torch.ones_like(source, dtype=torch.bool)
        center = centers[source]
        direction = centers[target] - (centers[target] * center).sum(-1, keepdim=True) * center
        residual = normalize(x) - (normalize(x) * center).sum(-1, keepdim=True) * center
        direction = normalize(direction + noise * residual)
        chord = (radii[source] + margin * (0.5 + torch.rand(len(source), device=x.device))).clamp(max=1.98)
        theta = 2 * torch.asin(chord / 2)
        candidate = normalize(theta.cos()[:, None] * center + theta.sin()[:, None] * direction)
        outside = (distances(candidate, centers) > radii[None, :] + 1e-5).all(dim=1)
        parent = mapping[source]
        # Parent preservation is asserted only for actual same-parent pairs.
        same_parent = mapping[target] == parent
        in_parent = (candidate - support["parent_centers"][parent]).norm(dim=-1) <= (
            support["parent_radius"][parent] + margin)
        good = eligible & outside
        if mode == "taxonomy":
            good &= in_parent
        proposed += int(eligible.sum())
        outputs.append(candidate[good])
        parents.append(torch.where(same_parent & in_parent, parent, -1)[good])
        sources.append(source[good])
    return (torch.cat(outputs), torch.cat(parents), torch.cat(sources),
            {"proposed": proposed, "accepted": sum(len(o) for o in outputs)})


def boundary_loss(head, x, y, synthetic, synthetic_parent, support, reject_margin=0.15,
                  parent_weight=0.2):
    z = head(x)
    ratio = distances(z, support["centers"]) / support["radius"][None, :]
    own = ratio.gather(1, y[:, None]).squeeze(1)
    inside = F.relu(own - 0.95).square().mean()
    classification = F.cross_entropy(-ratio / 0.2, y)
    reject = z.sum() * 0
    parent_loss = z.sum() * 0
    if len(synthetic):
        u = head(synthetic)
        nearest = (distances(u, support["centers"]) / support["radius"][None, :]).min(dim=1).values
        reject = F.relu(1 + reject_margin - nearest).square().mean()
        valid = synthetic_parent >= 0
        if valid.any():
            parent_loss = F.cross_entropy(-distances(u[valid], support["parent_centers"]) / 0.2,
                                           synthetic_parent[valid])
    total = inside + 0.2 * classification + reject + parent_weight * parent_loss
    return total, dict(inside=float(inside.detach()), separation=float(classification.detach()),
                       reject=float(reject.detach()), parent=float(parent_loss.detach()))


def calibrate_known(local, semantic, manifold, correct, target=0.90, max_drop=0.02,
                    root_budget=0.002):
    """Choose the most selective observed threshold meeting the full E2E gate.

    Only known calibration labels are used. This is an empirical calibration
    constraint, NOT a guarantee on unseen test data or a conformal claim.
    """
    local, semantic, manifold = [v.detach().double().cpu() for v in (local, semantic, manifold)]
    correct = correct.detach().bool().cpu()
    n = len(correct)
    if not n or any(len(v) != n or not torch.isfinite(v).all() for v in (local, semantic, manifold)):
        raise ValueError("Invalid calibration scores")
    if not 0 < target < 1 or not 0 <= max_drop < 1 or not 0 <= root_budget < 1:
        raise ValueError("Invalid calibration constraint")
    needed = max(math.floor(target * n) + 1, math.ceil(int(correct.sum()) - max_drop * n - 1e-10))
    if int(correct.sum()) < needed:
        raise ValueError("Known closed accuracy cannot satisfy strict E2E > {:.1%}; no deployable router".format(target))
    sem_t = float(torch.quantile(semantic, root_budget))
    man_t = float(torch.quantile(manifold, root_budget))
    root_reject = (semantic < sem_t) & (manifold < man_t)
    if int((correct & ~root_reject).sum()) < needed:
        # Explicitly disable root rejection if even its tiny budget is too big.
        sem_t, man_t = float(semantic.min()) - 1., float(manifold.min()) - 1.
        root_reject = torch.zeros(n, dtype=torch.bool)
    eligible = local[correct & ~root_reject].sort(descending=True).values
    local_t = float(eligible[needed - 1])
    retained = ~root_reject & (local >= local_t)
    e2e = float((retained & correct).double().mean())
    return dict(local_threshold=local_t, semantic_threshold=sem_t, manifold_threshold=man_t,
                target=target, max_drop=max_drop, root_budget=root_budget,
                calibration_count=n, required_correct=needed,
                closed_accuracy=float(correct.double().mean()), known_e2e=e2e,
                coverage=float(retained.double().mean()), gate_passed=e2e > target)
