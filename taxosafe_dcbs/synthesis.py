"""TRAIN-only cosine supports and depth-conditioned feature sampling."""
import torch
from torch.nn import functional as F


def _centres(features, labels, count):
    centres = []
    for label in range(count):
        subset = features[labels == label]
        if not len(subset):
            raise ValueError("Missing training support for label {}".format(label))
        centres.append(F.normalize(subset.mean(0), dim=0))
    return torch.stack(centres)


@torch.no_grad()
def fit_support(features, labels, leaf_to_parent, settings):
    """Complete unaugmented TRAIN pass; scarce radii shrink toward pooled ones."""
    h = F.normalize(features.detach().float(), dim=-1)
    labels = labels.long()
    mapping = torch.as_tensor(leaf_to_parent, dtype=torch.long, device=h.device)
    if h.ndim != 2 or len(h) != len(labels) or not bool(torch.isfinite(h).all()):
        raise ValueError("Invalid training support matrix")
    if bool(((labels < 0) | (labels >= len(mapping))).any()):
        raise ValueError("Support fitting accepts only known training leaf labels")
    leaves = _centres(h, labels, len(mapping))
    parents = _centres(leaves, mapping, int(mapping.max()) + 1)
    prior = float(settings.get("radius_prior_count", 10.0))
    if prior < 0:
        raise ValueError("radius_prior_count must be non-negative")

    def radii(centres, target, quantile, minimum):
        if not 0 < quantile < 1:
            raise ValueError("Support quantiles must be strictly between zero and one")
        distances = (1.0 - (h * centres[target]).sum(1)).clamp(0, 2)
        pooled = torch.quantile(distances, quantile)
        values = []
        for c in range(len(centres)):
            d = distances[target == c]
            w = len(d) / (len(d) + prior)
            values.append((w * torch.quantile(d, quantile) + (1 - w) * pooled).clamp(min=minimum, max=1.99))
        return torch.stack(values)
    return {"leaf": leaves, "parent": parents,
            "leaf_radius": radii(leaves, labels, float(settings.get("leaf_support_quantile", 0.90)), 1e-4),
            "parent_radius": radii(parents, mapping[labels], float(settings.get("parent_support_quantile", 0.95)), 1e-4),
            "features": h, "labels": labels, "leaf_to_parent": mapping}


@torch.no_grad()
def support_membership(points, bank):
    points = F.normalize(points.float(), dim=-1)
    leaf_distance = (1 - points @ bank["leaf"].T).clamp(0, 2)
    parent_distance = (1 - points @ bank["parent"].T).clamp(0, 2)
    return leaf_distance <= bank["leaf_radius"], parent_distance <= bank["parent_radius"]


def empty_synthetic(bank):
    return {"near": bank["features"][:0], "near_parent": bank["labels"][:0],
            "extra": bank["features"][:0], "stats": {"near_candidates": 0, "near_kept": 0,
            "extra_candidates": 0, "extra_kept": 0, "near_by_parent": {}}}


@torch.no_grad()
def synthesize(bank, leaf_text, scale, settings, generator=None):
    """Depth is determined by support membership, not by the mixing recipe.

    Invalid candidates are skipped. Singleton parents cannot supply siblings.
    Among eligible candidates, prioritize confidently absorbed hard negatives.
    """
    out = empty_synthetic(bank)
    h, labels, mapping = bank["features"], bank["labels"], bank["leaf_to_parent"]
    device, dim = h.device, h.shape[1]
    count, keep = int(settings.get("candidates_per_parent", 128)), int(settings.get("keep_per_parent", 16))
    noise = float(settings.get("near_noise", 0.02))
    if count < 1 or keep < 1 or noise < 0:
        raise ValueError("Invalid synthesis candidate/keep count or noise")
    rand = lambda *shape: torch.rand(*shape, device=device, generator=generator)
    randint = lambda n, shape: torch.randint(n, shape, device=device, generator=generator)

    def samples(classes):
        chosen = torch.empty_like(classes)
        for c in classes.unique().tolist():
            ids = torch.where(labels == c)[0]
            mask = classes == c
            chosen[mask] = ids[randint(len(ids), (int(mask.sum()),))]
        return h[chosen]
    if settings.get("near_enabled", True):
        for p in range(len(bank["parent"])):
            children = torch.where(mapping == p)[0]
            out["stats"]["near_by_parent"][str(p)] = 0
            if len(children) < 2:
                continue
            ia = randint(len(children), (count,))
            ib = (ia + 1 + randint(len(children) - 1, (count,))) % len(children)
            a, b = samples(children[ia]), samples(children[ib])
            mix = 0.25 + 0.5 * rand(count, 1)
            perturb = torch.randn(count, dim, device=device, generator=generator)
            points = F.normalize(mix * a + (1 - mix) * b + noise * F.normalize(perturb, dim=-1), dim=-1)
            in_leaf, in_parent = support_membership(points, bank)
            points = points[~in_leaf.any(1) & in_parent[:, p]]
            out["stats"]["near_candidates"] += count
            if len(points):
                hard = (float(scale) * points @ leaf_text[children].T).max(1).values
                points = points[torch.argsort(hard, descending=True)[:keep]]
                out["near"] = torch.cat((out["near"], points))
                out["near_parent"] = torch.cat((out["near_parent"], torch.full((len(points),), p, dtype=torch.long, device=device)))
                out["stats"]["near_by_parent"][str(p)] = len(points)
    if settings.get("extra_enabled", True) and len(bank["parent"]) > 1:
        count_extra = count * len(bank["parent"])
        ca = randint(len(mapping), (count_extra,))
        cb = torch.empty_like(ca)
        for p in range(len(bank["parent"])):
            mask = mapping[ca] == p
            others = torch.where(mapping != p)[0]
            cb[mask] = others[randint(len(others), (int(mask.sum()),))]
        a, b = samples(ca), samples(cb)
        t = torch.where(rand(count_extra, 1) < 0.5, 0.25 + 0.5 * rand(count_extra, 1), 1.25 + 0.75 * rand(count_extra, 1))
        direction = F.normalize(torch.randn(count_extra, dim, device=device, generator=generator), dim=-1)
        points = F.normalize(t * a + (1 - t) * b + float(settings.get("extra_noise", 0.2)) * direction, dim=-1)
        in_leaf, in_parent = support_membership(points, bank)
        points = points[~in_parent.any(1) & ~in_leaf.any(1)]
        out["stats"]["extra_candidates"] = count_extra
        if len(points):
            hard = (float(scale) * points @ leaf_text.T).max(1).values
            out["extra"] = points[torch.argsort(hard, descending=True)[:keep * len(bank["parent"])]]
    out["stats"]["near_kept"], out["stats"]["extra_kept"] = len(out["near"]), len(out["extra"])
    return out
