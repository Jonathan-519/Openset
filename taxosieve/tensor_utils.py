"""Exact tensor and taxonomy validation used by the frozen D05 evidence model.

Extracted from the historical proximity module. No proximity experiment,
calibration, fitting, or prediction policy is retained here.
"""
import torch


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
