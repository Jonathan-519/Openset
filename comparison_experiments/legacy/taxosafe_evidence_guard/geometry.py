"""Full-TRAIN local geometry for the frozen C00 feature branches.

This module fits no unknown detector and returns neither probabilities nor
p-values. Every known TRAIN image is retained. Cosine distance is 1 - cosine,
and each leaf/branch scale is its mean leave-one-image-out kNN distance,
shrunk towards the *leaf-balanced* global mean by n / (n + shrinkage).
Scales have a fixed numerical floor of 1e-4; dimensionless features are clipped
to [-20, 20]. The floor and clipping are numerical guards, not DEV thresholds.
Query hashes exclude the same image from every distance calculation. k is
reduced separately per query to the available number of nonself images.

For leaf c, the six features are:
  0: negative scaled mean fine-branch kNN distance;
  1: negative scaled nearest fine-branch distance;
  2: best sibling scaled mean distance minus c's (zero without siblings);
  3: best leaf outside c's parent distance minus c's (zero without rivals);
  4: fine second-nearest minus nearest distance within the selected k neighbors,
     divided by c's scale (zero when only one neighbor is selected);
  5: negative mean scaled parent-branch distance across its parent's leaves.

For parent p, the six features are:
  0: negative mean scaled parent-branch distance across p's leaves;
  1: negative best child scaled parent-branch distance;
  2: best other parent's leaf-balanced mean distance minus p's mean;
  3: negative mean scaled fine-branch distance across p's leaves;
  4: gap between best and second-best child parent-branch mean distances;
  5: negative mean scaled nearest parent-branch distance across p's leaves.

All parent means weight leaves equally, regardless of their image counts.
Support counts and labels never become features. Statistics use known TRAIN
only. score reads query fine/parent tensors and hashes, never query labels,
status, source names, or other entries in the encoded mapping. The geometry
is frozen: tensors are detached from the query adapter's computation graph.
"""
import math

import torch
from torch.nn import functional as F


SCHEMA = "taxosafe_evidence_guard_geometry_v1"
FEATURE_COUNT = 6
SCALE_FLOOR = 1e-4
FEATURE_BOUND = 20.0


def _features(value, name, rows=None, device=None):
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise ValueError(name + " must be a two-dimensional tensor")
    if rows is not None and value.shape[0] != rows:
        raise ValueError(name + " row count does not match the image hashes")
    if value.shape[1] < 1:
        raise ValueError(name + " has no feature dimensions")
    result = value.detach().to(device=device or value.device, dtype=torch.float32)
    if not bool(torch.isfinite(result).all()):
        raise ValueError(name + " contains nonfinite values")
    if len(result) and not bool((result.norm(dim=1) > 1e-12).all()):
        raise ValueError(name + " contains zero-norm feature vectors")
    return F.normalize(result, dim=1)


def _hashes(values, rows, unique=False):
    values = list(values)
    if len(values) != rows or any(not isinstance(v, str) or not v for v in values):
        raise ValueError("Image hashes must be nonempty strings aligned to features")
    if unique and len(set(values)) != len(values):
        raise ValueError("TRAIN geometry requires one row per unique image hash")
    return values


def _meta(meta):
    c, p = len(meta["leaf_names"]), len(meta["parent_names"])
    values = list(meta["leaf_to_parent"])
    if not c or not p or len(values) != c:
        raise ValueError("Invalid geometry hierarchy dimensions")
    if any(isinstance(v, bool) or int(v) != v or not 0 <= int(v) < p for v in values):
        raise ValueError("Invalid leaf-to-parent mapping")
    if set(int(v) for v in values) != set(range(p)):
        raise ValueError("Every parent must contain at least one leaf")
    return c, p, torch.tensor(values, dtype=torch.long)


def _loo_scale(bank, labels, classes, neighbors, shrinkage):
    raw, counts = [], []
    for leaf in range(classes):
        points = bank[labels == leaf]
        count = len(points)
        if count < 2:
            raise ValueError("Every geometry leaf needs at least two unique TRAIN images")
        distance = (1.0 - points @ points.T).clamp(0.0, 2.0)
        distance.fill_diagonal_(float("inf"))
        closest = distance.topk(min(neighbors, count - 1), dim=1, largest=False).values
        raw.append(closest.mean())
        counts.append(count)
    raw = torch.stack(raw)
    count_tensor = torch.tensor(counts, dtype=torch.float32)
    # A large class does not dominate the target scale for a rare class.
    global_scale = raw.mean().clamp(min=SCALE_FLOOR)
    weight = count_tensor / (count_tensor + shrinkage)
    scale = (weight * raw + (1.0 - weight) * global_scale).clamp(min=SCALE_FLOOR)
    if not bool(torch.isfinite(scale).all()):
        raise ValueError("Nonfinite TRAIN geometry scale")
    return scale, raw, global_scale


def fit(group, meta, settings):
    """Fit immutable CPU geometry from a complete, unique known-TRAIN group.

    ``group`` must contain ``records``, ``image_sha256`` and ``encoded`` with
    C00 ``fine``/``parent`` matrices. All records must be status=known and
    split=train; DEV, TEST and real/synthetic unknowns are rejected. Labels are
    used solely here to construct the known TRAIN bank and hierarchy.
    """
    records = list(group["records"])
    if not records:
        raise ValueError("Cannot fit geometry without known TRAIN images")
    hashes = _hashes(group["image_sha256"], len(records), unique=True)
    c, p, mapping = _meta(meta)
    labels = []
    for index, record in enumerate(records):
        if record.get("split") != "train" or record.get("status") != "known":
            raise ValueError("Geometry statistics may use known TRAIN only")
        leaf = record.get("true_leaf")
        if leaf is None or isinstance(leaf, bool) or int(leaf) != leaf or not 0 <= int(leaf) < c:
            raise ValueError("Invalid known TRAIN leaf label for geometry")
        leaf = int(leaf)
        if record.get("true_parent") != int(mapping[leaf]):
            raise ValueError("Known TRAIN parent label disagrees with hierarchy")
        if record.get("image_sha256", hashes[index]) != hashes[index]:
            raise ValueError("Known TRAIN record hash disagrees with feature row")
        labels.append(leaf)
    labels = torch.tensor(labels, dtype=torch.long)
    raw_neighbors = settings.get("neighbors", 5)
    if (isinstance(raw_neighbors, bool) or int(raw_neighbors) != raw_neighbors
            or int(raw_neighbors) < 1):
        raise ValueError("geometry.neighbors must be a positive integer")
    neighbors = int(raw_neighbors)
    shrinkage = float(settings.get("shrinkage", 5.0))
    if not math.isfinite(shrinkage) or shrinkage < 0:
        raise ValueError("geometry.shrinkage must be finite and nonnegative")
    encoded = group["encoded"]
    fine = _features(encoded["fine"], "TRAIN fine", len(hashes), torch.device("cpu")).clone()
    parent = _features(encoded["parent"], "TRAIN parent", len(hashes), torch.device("cpu")).clone()
    fs, fr, fg = _loo_scale(fine, labels, c, neighbors, shrinkage)
    ps, pr, pg = _loo_scale(parent, labels, c, neighbors, shrinkage)
    return {
        "schema_version": SCHEMA,
        "fit_splits": ["train"],
        "fit_status": ["known"],
        "test_used_for_fitting": False,
        "neighbors": neighbors,
        "shrinkage": shrinkage,
        "image_sha256": hashes,
        "fine": fine,
        "parent": parent,
        "labels": labels,
        "leaf_to_parent": mapping,
        "leaf_count": c,
        "parent_count": p,
        "fine_scale": fs,
        "parent_scale": ps,
        "fine_raw_loo_scale": fr,
        "parent_raw_loo_scale": pr,
        "fine_global_scale": fg,
        "parent_global_scale": pg,
        "feature_count": FEATURE_COUNT,
        "statistics": "leaf-balanced shrinkage of known-TRAIN leave-one-image-out cosine kNN distances",
    }


def _distances(query, bank, query_hashes, bank_hashes):
    distance = (1.0 - query @ bank.T).clamp(0.0, 2.0)
    positions = {value: index for index, value in enumerate(bank_hashes)}
    rows, columns = [], []
    for row, value in enumerate(query_hashes):
        if value in positions:
            rows.append(row)
            columns.append(positions[value])
    if rows:
        distance[torch.tensor(rows, device=distance.device),
                 torch.tensor(columns, device=distance.device)] = float("inf")
    return distance


def _leaf_distances(distance, labels, classes, scales, neighbors):
    means, nearest, gaps = [], [], []
    for leaf in range(classes):
        values = distance[:, labels == leaf]
        closest = values.topk(min(neighbors, values.shape[1]), dim=1, largest=False).values
        finite = torch.isfinite(closest)
        count = finite.sum(dim=1)
        if not bool((count > 0).all()):
            raise ValueError("A query has no nonself known TRAIN support for a leaf")
        mean = torch.where(finite, closest, torch.zeros_like(closest)).sum(1) / count.to(distance.dtype)
        first = closest[:, 0]
        # With k=1 there is no second selected neighbor; its gap is defined as 0.
        second = closest[:, 1] if closest.shape[1] > 1 else first
        second = torch.where(torch.isfinite(second), second, first)
        means.append(mean / scales[leaf])
        nearest.append(first / scales[leaf])
        gaps.append((second - first).clamp(min=0.0) / scales[leaf])
    return torch.stack(means, 1), torch.stack(nearest, 1), torch.stack(gaps, 1)


def score(encoded, hashes, state, device=None):
    """Return frozen 6-D ``leaf`` and ``parent`` evidence for each query.

    Hashes are used only for exact-image self exclusion. Unknown hashes are
    ordinary queries; labels and image source names are neither required nor
    inspected. ``device`` defaults to the device of ``encoded['fine']``.
    """
    if state.get("schema_version") != SCHEMA:
        raise ValueError("Unsupported frozen geometry state")
    target = torch.device(device) if device is not None else encoded["fine"].device
    fine = _features(encoded["fine"], "query fine", device=target)
    query_hashes = _hashes(hashes, len(fine))
    parent = _features(encoded["parent"], "query parent", len(fine), target)
    c, p = int(state["leaf_count"]), int(state["parent_count"])
    if len(fine) == 0:
        return {"leaf": fine.new_empty((0, c, FEATURE_COUNT)),
                "parent": fine.new_empty((0, p, FEATURE_COUNT))}
    bf, bp = state["fine"].to(target), state["parent"].to(target)
    if fine.shape[1] != bf.shape[1] or parent.shape[1] != bp.shape[1]:
        raise ValueError("Query feature dimensions disagree with frozen TRAIN bank")
    labels, mapping = state["labels"].to(target), state["leaf_to_parent"].to(target)
    fd = _distances(fine, bf, query_hashes, state["image_sha256"])
    pd = _distances(parent, bp, query_hashes, state["image_sha256"])
    fm, fn, gap = _leaf_distances(fd, labels, c, state["fine_scale"].to(target), state["neighbors"])
    pm, pn, _ = _leaf_distances(pd, labels, c, state["parent_scale"].to(target), state["neighbors"])
    parent_mean, parent_best, parent_fine, parent_gap, parent_nearest = [], [], [], [], []
    for j in range(p):
        children = mapping == j
        support = pm[:, children]
        ordered = support.topk(min(2, support.shape[1]), dim=1, largest=False).values
        parent_mean.append(support.mean(1))
        parent_best.append(ordered[:, 0])
        parent_fine.append(fm[:, children].mean(1))
        parent_gap.append(ordered[:, -1] - ordered[:, 0])
        parent_nearest.append(pn[:, children].mean(1))
    parent_mean, parent_best, parent_fine, parent_gap, parent_nearest = [
        torch.stack(value, 1) for value in
        (parent_mean, parent_best, parent_fine, parent_gap, parent_nearest)]
    sibling_advantage, outside_advantage = [], []
    ids = torch.arange(c, device=target)
    zeros = fine.new_zeros(len(fine))
    for leaf in range(c):
        siblings = (mapping == mapping[leaf]) & (ids != leaf)
        outside = mapping != mapping[leaf]
        sibling_advantage.append(fm[:, siblings].min(1).values - fm[:, leaf] if bool(siblings.any()) else zeros)
        outside_advantage.append(fm[:, outside].min(1).values - fm[:, leaf] if bool(outside.any()) else zeros)
    rivals = []
    for j in range(p):
        others = torch.arange(p, device=target) != j
        rivals.append(parent_mean[:, others].min(1).values - parent_mean[:, j] if bool(others.any()) else zeros)
    leaf = torch.stack((-fm, -fn, torch.stack(sibling_advantage, 1),
                        torch.stack(outside_advantage, 1), gap, -parent_mean[:, mapping]), 2)
    parent = torch.stack((-parent_mean, -parent_best, torch.stack(rivals, 1),
                          -parent_fine, parent_gap, -parent_nearest), 2)
    result = {"leaf": leaf.clamp(-FEATURE_BOUND, FEATURE_BOUND),
              "parent": parent.clamp(-FEATURE_BOUND, FEATURE_BOUND)}
    if not all(bool(torch.isfinite(value).all()) for value in result.values()):
        raise ValueError("Nonfinite local geometry evidence")
    return result
