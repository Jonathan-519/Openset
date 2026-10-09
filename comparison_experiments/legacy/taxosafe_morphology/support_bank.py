"""Real TRAIN individuals, selected per species after exclusion, never centroids."""
import torch
from torch.nn import functional as F


def select_references(features, labels, hashes, eligible, per_leaf):
    """Deterministic medoid then farthest-point diversity; TRAIN information only."""
    unit = F.normalize(features.detach().cpu().float(), dim=1)
    labels = labels.detach().cpu().long()
    eligible = torch.as_tensor(eligible, dtype=torch.bool)
    if eligible.shape != labels.shape or not bool(eligible.any()):
        raise ValueError("Reference selection requires nonempty eligible TRAIN support")
    selected = []
    for leaf in sorted(set(labels[eligible].tolist())):
        indices = sorted((eligible & (labels == leaf)).nonzero(as_tuple=True)[0].tolist(), key=lambda i: hashes[i])
        current = unit[indices]
        centre = F.normalize(current.mean(0), dim=0)
        first = int((current @ centre).argmax())
        chosen = [first]
        while len(chosen) < min(per_leaf, len(indices)):
            distance = 1. - (current @ current[chosen].T).max(1).values
            distance[chosen] = -1.
            chosen.append(int(distance.argmax()))
        selected.extend(indices[i] for i in chosen)
    return torch.tensor(selected, dtype=torch.long)


def make_bank(tokens, positions, features, labels, hashes, meta, per_leaf):
    indices = select_references(features, labels, hashes, torch.ones(len(labels), dtype=torch.bool), per_leaf)
    return dict(schema_version="morphology_real_support_v1", fit_split="known_train",
                meta=meta, tokens=tokens[indices].detach().cpu().clone(), positions=positions.detach().cpu().clone(),
                labels=labels[indices].detach().cpu().clone(), image_sha256=[hashes[i] for i in indices.tolist()],
                references_per_leaf=per_leaf, unique_individuals=len(indices),
                selected_train_indices=indices.tolist())


def candidate_indices(labels, meta, level, candidate):
    if level == "leaf":
        return (labels == candidate).nonzero(as_tuple=True)[0]
    if level != "parent":
        raise ValueError("Unknown verification level")
    mapping = torch.tensor(meta["leaf_to_parent"], device=labels.device)
    return (mapping[labels] == candidate).nonzero(as_tuple=True)[0]
