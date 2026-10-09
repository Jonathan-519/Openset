"""Known-TRAIN local-neighbour evidence, inspired by kNN/proximity methods.

This is not a reproduction of NNGuide or a learned classifier. Every known
TRAIN feature is retained. A leaf uses mean cosine distance to up to ``k``
neighbours, with matching content hashes excluded. A parent first computes
that distance for each of its known children in the parent feature space,
then averages the best two available children (one for a singleton).

TRAIN leave-self-out distances supply median/MAD standardization, partially
pooled toward global TRAIN statistics. Larger standardized proximity means
closer to known support. The caller must enforce TRAIN manifest provenance;
this module accepts neither candidate labels nor DEV/TEST calibration data.
"""
import copy
import math

import torch


SCHEMA_VERSION = "routealign_proximity_v1"
QUERY_CHUNK = 128
SUPPORT_CHUNK = 1024
SCALE_FLOOR = 1e-6
MAD_NORMALIZATION = 1.4826
_STATISTICS = ("leaf_location", "leaf_scale", "parent_location", "parent_scale")
_STATE_KEYS = {"schema_version", "k", "shrinkage", "fine", "parent", "labels",
               "image_hashes", "meta", "fit_report", *_STATISTICS}


def _ids(value, name, count=None, limit=None):
    raw = torch.as_tensor(value)
    if raw.ndim != 1 or (count is not None and len(raw) != count):
        raise ValueError(name + " must be a vector with one ID per row")
    if raw.dtype == torch.bool or raw.is_complex():
        raise ValueError(name + " must contain integer IDs")
    if raw.is_floating_point() and (not bool(torch.isfinite(raw).all())
                                  or not bool((raw == raw.round()).all())):
        raise ValueError(name + " must contain finite integer IDs")
    result = raw.detach().to(device="cpu", dtype=torch.long).clone()
    if bool((result < 0).any()) or (limit is not None and bool((result >= limit).any())):
        raise ValueError(name + " contains an out-of-range ID")
    return result


def _meta(value):
    if not isinstance(value, dict) or set(value) != {"leaf_names", "parent_names", "leaf_to_parent"}:
        raise ValueError("meta must contain the complete locked leaf/parent taxonomy")
    names = {}
    for key in ("leaf_names", "parent_names"):
        items = value[key]
        if (not isinstance(items, (list, tuple)) or not items
                or any(not isinstance(item, str) or not item.strip() for item in items)
                or len(set(items)) != len(items)):
            raise ValueError(key + " must contain nonempty unique names")
        names[key] = list(items)
    mapping = _ids(value["leaf_to_parent"], "leaf_to_parent", len(names["leaf_names"]), len(names["parent_names"]))
    if set(mapping.tolist()) != set(range(len(names["parent_names"]))):
        raise ValueError("Every declared parent must have at least one known child")
    return dict(names, leaf_to_parent=mapping.tolist())


def _hashes(value, count, unique):
    if (not isinstance(value, (list, tuple)) or len(value) != count
            or any(not isinstance(item, str) or not item.strip() for item in value)):
        raise ValueError("image/query hashes must be nonempty strings aligned with features")
    if unique and len(set(value)) != len(value):
        raise ValueError("Known TRAIN image hashes must be unique; deduplicate content before fitting")
    return tuple(value)


def _features(value, name, count=None, dimension=None, normalized=False):
    raw = torch.as_tensor(value)
    if raw.ndim != 2 or raw.shape[1] < 1 or raw.dtype == torch.bool or raw.is_complex() or not raw.is_floating_point():
        raise ValueError(name + " must contain real floating-point [N,D] features with D > 0")
    if (count is not None and len(raw) != count) or (dimension is not None and raw.shape[1] != dimension):
        raise ValueError(name + " dimensions differ from the fitted bank")
    result = raw.detach().to(device="cpu", dtype=torch.float64).clone()
    if not bool(torch.isfinite(result).all()):
        raise ValueError(name + " features must be finite")
    if normalized:
        if not torch.allclose(result.norm(dim=1), torch.ones(len(result), dtype=torch.float64), rtol=0., atol=1e-12):
            raise ValueError(name + " stored feature vectors must be unit normalized")
        return result
    largest = result.abs().amax(dim=1, keepdim=True)
    if bool((largest == 0).any()):
        raise ValueError(name + " contains zero-norm features")
    # Scaling first prevents overflow when a finite feature has huge entries.
    scaled = result / largest
    return scaled / scaled.norm(dim=1, keepdim=True)


def _median(values):
    ordered = values.sort().values
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2.


def _robust(values):
    median = float(_median(values))
    mad = float(_median((values - median).abs()))
    return median, mad, max(MAD_NORMALIZATION * mad, SCALE_FLOOR)


def _scales(distances, labels, count, shrinkage):
    valid = torch.isfinite(distances)
    pooled = distances[valid]
    global_fallback = not len(pooled)
    global_median, global_mad, global_scale = (1., None, 1.) if global_fallback else _robust(pooled)
    locations, scales, details = [], [], []
    for index in range(count):
        selected = distances[(labels == index) & valid]
        n = len(selected)
        weight = n / (n + shrinkage) if n else 0.
        median, mad, scale = _robust(selected) if n else (global_median, None, global_scale)
        location = weight * median + (1. - weight) * global_median
        pooled_scale = max(weight * scale + (1. - weight) * global_scale, SCALE_FLOOR)
        locations.append(location)
        scales.append(pooled_scale)
        details.append({"valid_leave_self_out_count": n, "pooling_weight": weight,
                        "local_median": None if not n else median, "local_mad": mad,
                        "location": location, "scale": pooled_scale,
                        "global_only_fallback": not bool(n)})
    report = {"global_valid_leave_self_out_count": len(pooled), "global_median": global_median,
              "global_mad": global_mad, "global_scale": global_scale,
              "global_no_neighbour_fallback": global_fallback, "groups": details}
    return torch.tensor(locations, dtype=torch.float64), torch.tensor(scales, dtype=torch.float64), report


class ProximityBank:
    """Detached float64 CPU evidence with bounded distance working memory.

    ``fit`` requires the complete known TRAIN feature cache and its locked
    taxonomy. Saved state is self-contained and fully revalidated on reload.
    A singleton known leaf cannot establish within-leaf novelty separation;
    a singleton parent has no cross-child support evidence.
    """

    @classmethod
    @torch.no_grad()
    def fit(cls, fine, parent, labels, image_hashes, meta, k=3, shrinkage=10.0):
        fine = _features(fine, "fine")
        parent = _features(parent, "parent", count=len(fine))
        return cls._build(fine, parent, labels, image_hashes, meta, k, shrinkage)

    @classmethod
    def _build(cls, fine, parent, labels, image_hashes, meta, k, shrinkage):
        if type(k) is not int or k < 1:
            raise ValueError("k must be a positive integer")
        if (isinstance(shrinkage, bool) or not isinstance(shrinkage, (float, int))
                or not math.isfinite(shrinkage) or shrinkage < 0):
            raise ValueError("shrinkage must be finite and nonnegative")
        if not len(fine):
            raise ValueError("Known TRAIN cannot be empty")
        result = cls()
        result.meta = _meta(meta)
        result.k, result.shrinkage = k, float(shrinkage)
        result.fine, result.parent = fine.clone(), parent.clone()
        leaves, parents = len(result.meta["leaf_names"]), len(result.meta["parent_names"])
        result.labels = _ids(labels, "labels", len(fine), leaves)
        result.image_hashes = _hashes(image_hashes, len(fine), unique=True)
        result._hash_index = {value: i for i, value in enumerate(result.image_hashes)}
        result._leaf_indices = [(result.labels == leaf).nonzero(as_tuple=True)[0] for leaf in range(leaves)]
        if any(not len(indices) for indices in result._leaf_indices):
            raise ValueError("Every known leaf must have at least one TRAIN image")
        mapping = torch.tensor(result.meta["leaf_to_parent"], dtype=torch.long)
        result._children = [(mapping == parent).nonzero(as_tuple=True)[0] for parent in range(parents)]
        raw_leaf, raw_parent = result._raw(fine, parent, result.image_hashes)
        row_indices = torch.arange(len(fine))
        parent_labels = mapping[result.labels]
        result.leaf_location, result.leaf_scale, leaf_report = _scales(
            raw_leaf[row_indices, result.labels], result.labels, leaves, result.shrinkage)
        result.parent_location, result.parent_scale, parent_report = _scales(
            raw_parent[row_indices, parent_labels], parent_labels, parents, result.shrinkage)
        leaf_counts = torch.bincount(result.labels, minlength=leaves).tolist()
        parent_counts = torch.bincount(parent_labels, minlength=parents).tolist()
        result.fit_report = {
            "schema_version": SCHEMA_VERSION, "fit_splits": ["train"],
            "provenance_requirement": "caller_verifies_locked_known_TRAIN_manifest",
            "unknown_or_test_data_used_for_scale_fitting": False,
            "support_count": len(fine), "all_train_rows_retained": True,
            "leaf_train_counts": leaf_counts, "parent_train_counts": parent_counts,
            "fine_dimension": fine.shape[1], "parent_dimension": parent.shape[1],
            "k": k, "shrinkage": result.shrinkage, "distance": "mean_k_nearest_cosine_distance",
            "scale_fit": "true_label_TRAIN_leave_content_hash_out_only",
            "partial_pooling": "valid_loo_count/(valid_loo_count+shrinkage)",
            "mad_normalization": MAD_NORMALIZATION, "scale_floor": SCALE_FLOOR,
            "leaf_scales": leaf_report, "parent_scales": parent_report,
            "parent_aggregation": "mean_of_best_up_to_two_available_child_distances",
            "singleton_parent_ids": [i for i, children in enumerate(result._children) if len(children) == 1],
            "singleton_parent_fallback": "one_child_distance; no_cross_child_evidence",
            "single_image_leaf_ids": [i for i, n in enumerate(leaf_counts) if n == 1],
            "single_image_leaf_limitation": "no_own_leaf_leave_self_out_sample; use_global_TRAIN_scale",
            "no_available_neighbour_score_distance": 2.0,
            "no_available_neighbour_scale_fit": "excluded; global_fallback_if_no_valid_LOO_samples",
            "inspiration": "kNN/proximity; not_an_NNGuide_formula_reproduction",
        }
        return result

    def _leaf_distances(self, queries, references, query_hashes):
        distances = torch.empty((len(queries), len(self._leaf_indices)), dtype=torch.float64)
        query_indices = torch.tensor([self._hash_index.get(value, -1) for value in query_hashes], dtype=torch.long)
        for start in range(0, len(queries), QUERY_CHUNK):
            stop = min(start + QUERY_CHUNK, len(queries))
            for leaf, support_indices in enumerate(self._leaf_indices):
                effective_k = min(self.k, len(support_indices))
                nearest = torch.full((stop - start, effective_k), float("inf"), dtype=torch.float64)
                for offset in range(0, len(support_indices), SUPPORT_CHUNK):
                    indices = support_indices[offset:offset + SUPPORT_CHUNK]
                    block = (1. - queries[start:stop] @ references[indices].T).clamp(0., 2.)
                    block.masked_fill_(query_indices[start:stop, None] == indices[None, :], float("inf"))
                    nearest = torch.topk(torch.cat((nearest, block), dim=1), effective_k,
                                         dim=1, largest=False, sorted=False).values
                available = torch.isfinite(nearest)
                count = available.sum(dim=1)
                average = nearest.masked_fill(~available, 0.).sum(dim=1) / count.clamp_min(1)
                distances[start:stop, leaf] = average.masked_fill(count == 0, float("inf"))
        return distances

    def _raw(self, fine, parent, query_hashes):
        leaf = self._leaf_distances(fine, self.fine, query_hashes)
        child = self._leaf_distances(parent, self.parent, query_hashes)
        parent_distance = torch.empty((len(parent), len(self._children)), dtype=torch.float64)
        for index, children in enumerate(self._children):
            best = torch.topk(child[:, children], min(2, len(children)), dim=1, largest=False).values
            available = torch.isfinite(best)
            count = available.sum(dim=1)
            average = best.masked_fill(~available, 0.).sum(dim=1) / count.clamp_min(1)
            parent_distance[:, index] = average.masked_fill(count == 0, float("inf"))
        return leaf, parent_distance

    @torch.no_grad()
    def score(self, fine, parent, query_hashes):
        if not hasattr(self, "fit_report"):
            raise ValueError("Fit the ProximityBank before scoring")
        fine = _features(fine, "fine", dimension=self.fine.shape[1])
        parent = _features(parent, "parent", count=len(fine), dimension=self.parent.shape[1])
        hashes = _hashes(query_hashes, len(fine), unique=False)
        leaf, parent = self._raw(fine, parent, hashes)
        leaf.masked_fill_(~torch.isfinite(leaf), 2.)
        parent.masked_fill_(~torch.isfinite(parent), 2.)
        return {"leaf_proximity": (self.leaf_location - leaf) / self.leaf_scale,
                "parent_proximity": (self.parent_location - parent) / self.parent_scale}

    def state_dict(self):
        if not hasattr(self, "fit_report"):
            raise ValueError("Fit the ProximityBank before saving")
        state = {"schema_version": SCHEMA_VERSION, "k": self.k, "shrinkage": self.shrinkage,
                 "image_hashes": list(self.image_hashes), "meta": copy.deepcopy(self.meta),
                 "fit_report": copy.deepcopy(self.fit_report)}
        for name in ("fine", "parent", "labels", *_STATISTICS):
            state[name] = getattr(self, name).detach().cpu().clone()
        return state

    @classmethod
    @torch.no_grad()
    def from_state_dict(cls, state):
        if not isinstance(state, dict) or set(state) != _STATE_KEYS or state.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Invalid ProximityBank state schema")
        for name in ("fine", "parent", "labels", *_STATISTICS):
            tensor = state[name]
            dtype = torch.long if name == "labels" else torch.float64
            if (not torch.is_tensor(tensor) or tensor.device.type != "cpu" or tensor.dtype != dtype
                    or tensor.requires_grad or not bool(torch.isfinite(tensor).all())):
                raise ValueError("Invalid CPU tensor in ProximityBank state: " + name)
        fine = _features(state["fine"], "fine", normalized=True)
        parent = _features(state["parent"], "parent", count=len(fine), normalized=True)
        result = cls._build(fine, parent, state["labels"], state["image_hashes"],
                            state["meta"], state["k"], state["shrinkage"])
        for name in _STATISTICS:
            expected = getattr(result, name)
            if state[name].shape != expected.shape or not torch.equal(state[name], expected):
                raise ValueError("Stored TRAIN proximity scales disagree with support features: " + name)
        if state["fit_report"] != result.fit_report:
            raise ValueError("Stored proximity fit report disagrees with known TRAIN support")
        return result
