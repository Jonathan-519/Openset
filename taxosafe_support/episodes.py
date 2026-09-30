"""Known-only paired support interventions with size-matched controls."""
import torch
from .support import validate_mapping


def build_episodes(labels, leaf_to_parent, seed=0):
    labels = torch.as_tensor(labels, dtype=torch.long)
    mapping = validate_mapping(leaf_to_parent).to(labels.device)
    if labels.ndim != 1 or bool(((labels < 0) | (labels >= len(mapping))).any()):
        raise ValueError("Invalid episode labels")
    b, c, p = len(labels), len(mapping), int(mapping.max()) + 1
    names = ("full", "drop_leaf", "drop_parent", "control_leaf", "control_parent")
    masks = {name: torch.ones(b, c, dtype=torch.bool, device=labels.device) for name in names}
    valid = {name: torch.ones(b, dtype=torch.bool, device=labels.device) for name in names}
    targets = {name: 1 + p + labels.clone() for name in names}
    generator = torch.Generator().manual_seed(int(seed))
    mapping_cpu = mapping.cpu()
    for i, label in enumerate(labels.cpu().tolist()):
        parent = int(mapping_cpu[label])
        siblings = torch.where(mapping_cpu == parent)[0].tolist()
        unrelated = torch.where(mapping_cpu != parent)[0].tolist()
        if len(siblings) > 1:
            masks["drop_leaf"][i, label] = False
            targets["drop_leaf"][i] = 1 + parent
        else:
            # No invented near-unknown task for a single-known-leaf parent.
            valid["drop_leaf"][i] = False
        masks["drop_parent"][i, siblings] = False
        targets["drop_parent"][i] = 0
        if unrelated:
            j = int(torch.randint(len(unrelated), (), generator=generator))
            masks["control_leaf"][i, unrelated[j]] = False
        else:
            valid["control_leaf"][i] = False
        # Prefer another whole parent of identical size. Otherwise remove the
        # exact same number of leaves from unrelated parents. This deliberately
        # does not use candidate counts as a model input or rejection label.
        same_size = [q for q in range(p) if q != parent and int((mapping_cpu == q).sum()) == len(siblings)]
        if same_size:
            other = same_size[int(torch.randint(len(same_size), (), generator=generator))]
            chosen = torch.where(mapping_cpu == other)[0].tolist()
        elif len(unrelated) >= len(siblings):
            order = torch.randperm(len(unrelated), generator=generator).tolist()
            chosen = [unrelated[j] for j in order[:len(siblings)]]
        else:
            chosen = []
            valid["control_parent"][i] = False
        masks["control_parent"][i, chosen] = False
    return {"masks": masks, "valid": valid, "targets": targets}
