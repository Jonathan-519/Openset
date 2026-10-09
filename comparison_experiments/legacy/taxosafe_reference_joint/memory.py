"""Multimodal C00 TRAIN support, cross-fitted by image and masked by class."""
import torch
from torch.nn import functional as F
from .protocol import object_hash


def modes(values, count):
    """Deterministic farthest-first real exemplars; never fabricate an average animal."""
    values = F.normalize(values.float(), dim=-1)
    selected = [int((values @ F.normalize(values.mean(0), dim=0)).argmax())]
    while len(selected) < min(count, len(values)):
        distance = 1.0 - (values @ values[selected].T).max(1).values
        distance[selected] = -1.0
        selected.append(int(distance.argmax()))
    return selected


def build_memory(fine, parent, labels, hashes, meta, settings, seed):
    if len(set(hashes)) != len(hashes) or len(labels) != len(hashes):
        raise ValueError("Memory must contain unique audited TRAIN images")
    c, count = len(meta["leaf_names"]), settings["modes"]
    if fine.ndim != 2 or parent.shape != fine.shape or not bool(torch.isfinite(fine).all() and torch.isfinite(parent).all()):
        raise ValueError("Invalid reference embedding matrices")
    fold_ids = torch.empty(len(labels), dtype=torch.long)
    for leaf in range(c):
        ids = (labels == leaf).nonzero(as_tuple=False).flatten().tolist()
        if len(ids) < 2:
            raise ValueError("Each TRAIN leaf needs at least two unique images for held-image support: " + str(leaf))
        ids.sort(key=lambda i: object_hash([seed, hashes[i]]))
        for index, i in enumerate(ids):
            fold_ids[i] = index % settings["folds"]
    banks = []
    for held in [-1] + list(range(settings["folds"])):
        bank = dict(fine=torch.zeros(c, count, fine.shape[1]), parent=torch.zeros(c, count, parent.shape[1]),
                    valid=torch.zeros(c, count, dtype=torch.bool), image_sha256=[], held_fold=held)
        for leaf in range(c):
            keep = labels == leaf
            if held >= 0:
                keep = keep & (fold_ids != held)
            ids = keep.nonzero(as_tuple=False).flatten().tolist()
            if not ids:
                raise ValueError("A held-image fold removed every support example")
            chosen = [ids[i] for i in modes(fine[ids], count)]
            n = len(chosen)
            bank["fine"][leaf, :n] = fine[chosen]
            bank["parent"][leaf, :n] = parent[chosen]
            bank["valid"][leaf, :n] = True
            bank["image_sha256"].append([hashes[i] for i in chosen])
        banks.append(bank)
    return dict(banks=banks, fold_ids=fold_ids, train_image_sha256=list(hashes), labels=labels,
                schema_version="reference_joint_memory_v1", settings=dict(settings))


def to_device(bank, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in bank.items()}
