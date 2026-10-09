"""Sibling-conditioned local evidence and a parent-orthogonal metric increment.

Only the increment is projected; the pretrained global feature is retained.
Root routing must use the original features, NOT these normalized child features.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


VARIANTS = ("identity", "full", "no_local", "no_orthogonal", "no_open", "shared")


class HierEvidence(nn.Module):
    def __init__(self, parent_text, leaf_text, mapping, settings, variant="full"):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError("Unknown variant: " + variant)
        self.variant = variant
        parent = F.normalize(torch.as_tensor(parent_text, dtype=torch.float32), dim=-1)
        leaf = F.normalize(torch.as_tensor(leaf_text, dtype=torch.float32), dim=-1)
        mapping = torch.as_tensor(mapping, dtype=torch.long)
        if parent.ndim != 2 or leaf.ndim != 2 or parent.shape[1] != leaf.shape[1]:
            raise ValueError("Text features must be compatible matrices")
        if mapping.shape != (len(leaf),) or set(mapping.tolist()) != set(range(len(parent))):
            raise ValueError("Every parent needs a known leaf")
        if not torch.isfinite(parent).all() or not torch.isfinite(leaf).all():
            raise ValueError("Nonfinite text features")
        # SVD handles dependent parent descriptors; QR would add arbitrary axes.
        _, singular, vh = torch.linalg.svd(parent, full_matrices=False)
        basis = vh[singular > singular.max() * 1e-6].T.contiguous()
        self.register_buffer("basis", basis)
        self.register_buffer("leaf_text", leaf)
        self.register_buffer("mapping", mapping)
        self.register_buffer("parent_text", parent)
        dim, rank = parent.shape[1], int(settings["rank"])
        if not 0 < rank <= dim:
            raise ValueError("Adapter rank must be in [1, feature dimension]")
        if variant == "shared":
            # Full has (2+P)*D*r parameters; shared has 3*D*r_shared.
            # For this dataset P=7,r=32 -> r_shared=96, exact parameter match.
            rank = (rank * (len(parent) + 2) + 2) // 3
        self.attention_temperature = float(settings["attention_temperature"])
        self.increment_bound = float(settings["increment_bound"])
        if self.attention_temperature <= 0 or not 0 < self.increment_bound <= 1:
            raise ValueError("Invalid temperature or increment bound")
        self.down = nn.Linear(2 * dim, rank, bias=False)
        self.up = nn.Parameter(torch.zeros(1 if variant == "shared" else len(parent), rank, dim))

    def forward(self, global_features, patches, parent, active_leaves=None, details=False):
        g = F.normalize(global_features.float(), dim=-1)
        if patches.ndim != 3 or patches.shape[0] != len(g) or patches.shape[-1] != g.shape[-1]:
            raise ValueError("Expected [images, patches, feature dimension]")
        if not 0 <= int(parent) < len(self.parent_text):
            raise ValueError("Invalid parent")
        if active_leaves is None:
            active = torch.where(self.mapping == int(parent))[0]
        else:
            active = torch.as_tensor(active_leaves, dtype=torch.long, device=g.device)
        if not len(active) or torch.any(self.mapping[active] != int(parent)):
            raise ValueError("Active leaves must be nonempty and inside parent")
        tokens = F.normalize(patches.float(), dim=-1)
        # Contrast descriptors contain only active siblings. Held names are
        # removed from pseudo-unknown attention as well as from support.
        text = self.leaf_text[active]
        contrast = text - text.mean(0, keepdim=True)
        salience = (tokens @ contrast.T).square().mean(-1)
        weights = F.softmax(salience / self.attention_temperature, dim=1)
        local = (weights.unsqueeze(-1) * tokens).sum(1)
        if self.variant == "no_local":
            local = torch.zeros_like(g)
        delta = F.gelu(self.down(torch.cat([g, local], dim=-1))) @ self.up[
            0 if self.variant == "shared" else int(parent)]
        if self.variant != "no_orthogonal":
            delta = delta - (delta @ self.basis) @ self.basis.T
        # Scaling preserves orthogonality and bounds the increment norm.
        delta = delta * self.increment_bound / (1 + delta.norm(dim=-1, keepdim=True))
        if self.variant == "identity":
            delta = delta * 0
        result = F.normalize(g + delta, dim=-1)
        return (result, delta, weights) if details else result


def class_distances(query, support, labels, active):
    """Absolute squared nearest-support distance for each active species."""
    distance = (query.square().sum(1, keepdim=True)
                + support.square().sum(1).unsqueeze(0) - 2 * query @ support.T).clamp_min(0)
    values = []
    for leaf in active:
        mask = labels == int(leaf)
        if not mask.any():
            raise ValueError("Active leaf has no support")
        values.append(distance[:, mask].min(dim=1).values)
    return torch.stack(values, dim=1)


def sample_episode(labels, children, rng, support_per_leaf, query_per_leaf):
    support, query = [], []
    for leaf in children:
        ids = rng.permutation(np.flatnonzero(labels == leaf))
        if len(ids) < 2:
            raise ValueError("Each leaf needs two distinct images; never duplicate support/query")
        ns = min(int(support_per_leaf), len(ids) // 2)
        nq = min(int(query_per_leaf), len(ids) - ns)
        support.extend(ids[:ns].tolist())
        query.extend(ids[ns:ns + nq].tolist())
    return np.asarray(support, dtype=np.int64), np.asarray(query, dtype=np.int64)


def episode_loss(model, g, patches, labels, parent, support_ids, query_ids, held_leaf, settings):
    if set(map(int, support_ids)) & set(map(int, query_ids)):
        raise ValueError("Support and query must be disjoint")
    children = torch.where(model.mapping == parent)[0]
    support_ids = torch.as_tensor(support_ids, device=g.device, dtype=torch.long)
    query_ids = torch.as_tensor(query_ids, device=g.device, dtype=torch.long)
    sy, qy = labels[support_ids], labels[query_ids]
    lookup = torch.full_like(model.mapping, -1)
    lookup[children] = torch.arange(len(children), device=g.device)
    z, delta, _ = model(g[query_ids], patches[query_ids], parent, details=True)
    zs = model(g[support_ids], patches[support_ids], parent)
    distance = class_distances(z, zs, sy, children)
    targets = lookup[qy]
    if (targets < 0).any():
        raise ValueError("Query outside episode parent")
    ce = F.cross_entropy(-distance / float(settings["distance_temperature"]), targets)
    compact = distance.gather(1, targets[:, None]).mean()
    anchor = delta.square().sum(-1).mean()
    opened = ce * 0
    if held_leaf is not None and model.variant != "no_open":
        if len(children) < 2 or not (children == held_leaf).any():
            raise ValueError("Pseudo-unknown must be a child of a non-singleton branch")
        active = children[children != held_leaf]
        keep = sy != held_leaf
        if not (qy == held_leaf).any() or not (qy != held_leaf).any():
            raise ValueError("Open episode requires both active and held queries")
        # Re-encode all compared queries/support with the SAME active semantic set.
        za = model(g[query_ids], patches[query_ids], parent, active)
        sa = model(g[support_ids[keep]], patches[support_ids[keep]], parent, active)
        da = class_distances(za, sa, sy[keep], active)
        active_lookup = torch.full_like(model.mapping, -1)
        active_lookup[active] = torch.arange(len(active), device=g.device)
        known = qy != held_leaf
        own = da[known].gather(1, active_lookup[qy[known]][:, None]).squeeze(1)
        novel = da[~known].min(1).values
        # Hard within-parent pseudo novelty: true active support must be closer
        # than the strongest remaining support to a held query.
        opened = F.relu(float(settings["open_margin"]) + own[:, None] - novel[None, :]).mean()
    loss = (ce + float(settings["compact_weight"]) * compact
            + float(settings["open_weight"]) * opened
            + float(settings["anchor_weight"]) * anchor)
    return loss, {"ce": float(ce.detach()), "compact": float(compact.detach()),
                  "open": float(opened.detach()), "anchor": float(anchor.detach())}


def train_adapter(data, rows, meta, settings, variant, seed, device="cpu"):
    if any(r["split"] != "train" or r["status"] != "known" for r in rows):
        raise ValueError("Adapter training accepts only known TRAIN records")
    hashes = [r["image_sha256"] for r in rows]
    if len(set(hashes)) != len(hashes):
        raise ValueError("Duplicate training image contents")
    if int(settings["epochs"]) < 1 or int(settings["episodes_per_epoch"]) < 1:
        raise ValueError("Positive training budget required")
    for key in ("support_per_leaf", "query_per_leaf"):
        if int(settings[key]) < 1:
            raise ValueError("Positive episode sample counts required")
    if float(settings["distance_temperature"]) <= 0:
        raise ValueError("Positive distance temperature required")
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    rng = np.random.RandomState(int(seed))
    model = HierEvidence(data["parent_text"], data["leaf_text"], meta["leaf_to_parent"], settings, variant).to(device)
    if variant == "identity":
        return model, []
    g = torch.as_tensor(data["global"], dtype=torch.float32, device=device)
    patches = torch.as_tensor(data["patches"], dtype=torch.float32, device=device)
    y = np.asarray([r["true_leaf"] for r in rows], dtype=np.int64)
    if set(y.tolist()) != set(range(len(meta["leaf_names"]))):
        raise ValueError("Training must contain every known leaf")
    yt = torch.as_tensor(y, device=device)
    branches = [np.flatnonzero(np.asarray(meta["leaf_to_parent"]) == p) for p in range(len(meta["parent_names"]))]
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings["lr"]), weight_decay=float(settings["weight_decay"]))
    history = []
    for epoch in range(int(settings["epochs"])):
        totals = {k: 0. for k in ("loss", "ce", "compact", "open", "anchor")}
        for step in range(int(settings["episodes_per_epoch"])):
            parent = int(rng.randint(len(branches)))
            support, query = sample_episode(y, branches[parent], rng, settings["support_per_leaf"], settings["query_per_leaf"])
            held = int(rng.choice(branches[parent])) if len(branches[parent]) >= 2 else None
            optimizer.zero_grad()
            loss, parts = episode_loss(model, g, patches, yt, parent, support, query, held, settings)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite adapter loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            totals["loss"] += float(loss.detach())
            for k, value in parts.items():
                totals[k] += value
        row = {k: v / int(settings["episodes_per_epoch"]) for k, v in totals.items()}
        row["epoch"] = epoch + 1
        history.append(row)
        print("[{}] epoch {}/{} loss={:.5f} open={:.5f}".format(variant, epoch + 1, settings["epochs"], row["loss"], row["open"]), flush=True)
    return model.eval(), history


@torch.no_grad()
def score_adapter(model, query, support, support_labels, parent_ids, chunk_size=64):
    """Query truth is absent from this API; routing is supplied by frozen root."""
    if int(chunk_size) < 1:
        raise ValueError("Positive chunk size required")
    device = next(model.parameters()).device
    parents = np.asarray(parent_ids, dtype=np.int64)
    if parents.shape != (len(query["global"]),) or (parents < 0).any() or (parents >= len(model.parent_text)).any():
        raise ValueError("Invalid parent routes")
    labels = np.asarray(support_labels, dtype=np.int64)
    out = {"score": np.empty(len(parents)), "leaf": np.empty(len(parents), dtype=np.int64),
           "neighbour": np.empty(len(parents), dtype=np.int64), "parent": parents.copy()}
    for parent in np.unique(parents):
        children = torch.where(model.mapping == int(parent))[0]
        ref_ids = np.flatnonzero(np.isin(labels, children.cpu().numpy()))
        if not len(ref_ids):
            raise ValueError("Parent has no reference images")
        tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
        reference = torch.cat([model(tensor(support["global"][ids]), tensor(support["patches"][ids]), int(parent))
                               for ids in np.array_split(ref_ids, max(1, int(np.ceil(len(ref_ids) / chunk_size))))])
        ids = np.flatnonzero(parents == parent)
        for start in range(0, len(ids), chunk_size):
            idx = ids[start:start + chunk_size]
            z = model(tensor(query["global"][idx]), tensor(query["patches"][idx]), int(parent))
            d = (z.square().sum(1, keepdim=True) + reference.square().sum(1).unsqueeze(0) - 2 * z @ reference.T).clamp_min(0)
            distance, neighbor = d.min(1)
            nearest = ref_ids[neighbor.cpu().numpy()]
            out["score"][idx] = (-.5 * torch.log(distance.clamp_min(1e-12))).cpu().numpy()
            out["leaf"][idx] = labels[nearest]
            out["neighbour"][idx] = nearest
    if not np.isfinite(out["score"]).all():
        raise FloatingPointError("Invalid support scores")
    return out


def save_adapter(path, model):
    np.savez_compressed(str(path), **{k: v.detach().cpu().numpy() for k, v in model.state_dict().items()})


def load_adapter(path, data, meta, settings, variant, device="cpu"):
    model = HierEvidence(data["parent_text"], data["leaf_text"], meta["leaf_to_parent"], settings, variant).to(device)
    with np.load(str(path), allow_pickle=False) as archive:
        model.load_state_dict({k: torch.as_tensor(archive[k], device=device) for k in archive.files}, strict=True)
    return model.eval()
