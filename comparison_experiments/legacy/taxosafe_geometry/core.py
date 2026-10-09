"""Frozen hierarchical relative Mahalanobis evidence fitted on known TRAIN.

The evidence verifies the caller's existing candidates; it never ranks classes.
All features are L2 normalized, and all sufficient statistics are fitted in
float64 on CPU without gradients. Means and covariances are sample weighted
(maximum-likelihood denominator N), as in ordinary Gaussian RMD. Covariance
shrinkage is essential when feature dimension exceeds the per-class count.

Parent evidence = global-background distance - selected-parent distance.
Leaf evidence = selected-parent-background distance - selected-leaf distance.
Within-parent and within-leaf covariances are shared over all known TRAIN rows.
A parent containing just one known leaf cannot define sibling geometry; for
that parent the fine background explicitly falls back to the global fine
distribution. This fallback is reported, not presented as sibling evidence.
"""
import math

import torch


_VERSION = 1
_FACTORS = (
    "parent_global_cholesky", "parent_within_cholesky",
    "fine_global_cholesky", "fine_parent_cholesky", "fine_leaf_cholesky",
)
_MEANS = ("parent_global_mean", "parent_means", "fine_global_mean",
          "fine_parent_means", "fine_leaf_means")


def _integer_vector(value, name, length=None, limit=None, device="cpu"):
    raw = torch.as_tensor(value, device=device)
    if raw.ndim != 1 or (length is not None and len(raw) != length):
        raise ValueError(name + " must have one entry per row")
    if raw.dtype == torch.bool or raw.is_complex():
        raise ValueError(name + " must contain integer IDs")
    if raw.is_floating_point() and (not bool(torch.isfinite(raw).all()) or
                                   not bool((raw == raw.round()).all())):
        raise ValueError(name + " must contain finite integer IDs")
    result = raw.detach().to(dtype=torch.long)
    if bool((result < 0).any()) or (limit is not None and bool((result >= limit).any())):
        raise ValueError(name + " contains an out-of-range ID")
    return result


def _normalized(value, name, device, dtype, dimension=None, rows=None):
    raw = torch.as_tensor(value)
    if raw.is_complex() or raw.dtype == torch.bool or not raw.is_floating_point():
        raise ValueError(name + " must contain real floating-point features")
    result = raw.detach().to(device=device, dtype=dtype)
    if result.ndim != 2 or result.shape[1] < 1:
        raise ValueError(name + " must have shape [N,D] with D > 0")
    if dimension is not None and result.shape[1] != dimension:
        raise ValueError(name + " has a different feature dimension from TRAIN")
    if rows is not None and len(result) != rows:
        raise ValueError(name + " must have one feature vector per image")
    if not bool(torch.isfinite(result).all()):
        raise ValueError(name + " must be finite")
    # Divide by the largest component first: vector_norm of very large finite
    # inputs may overflow even though their unit-normalized vector is valid.
    largest = result.abs().amax(dim=1, keepdim=True)
    if bool((largest == 0).any()):
        raise ValueError(name + " contains a zero feature vector")
    scaled = result / largest
    return scaled / torch.norm(scaled, p=2, dim=1, keepdim=True)


def _covariance_factor(residual, shrinkage, ridge):
    dimension = residual.shape[1]
    covariance = residual.T @ residual / len(residual)
    target = covariance.diagonal().mean()
    covariance = (1. - shrinkage) * covariance
    covariance.diagonal().add_(shrinkage * target + ridge)
    # Symmetrizing removes BLAS last-bit asymmetry before the factorization.
    covariance = (covariance + covariance.T) * .5
    if hasattr(torch.linalg, "cholesky_ex"):
        factor, status = torch.linalg.cholesky_ex(covariance)
    else:  # Older ProTeCt PyTorch environments.
        try:
            factor, status = torch.cholesky(covariance), 0
        except RuntimeError as error:
            raise ValueError("Covariance factorization failed; increase the positive ridge") from error
    if int(status) != 0 or not bool(torch.isfinite(factor).all()):
        raise ValueError("Covariance factorization failed; increase the positive ridge")
    if factor.shape != (dimension, dimension):
        raise ValueError("Unexpected covariance factor shape")
    return factor


def _group_means(features, labels, count):
    # Row order is fixed by the TRAIN cache. This explicit loop avoids CUDA
    # scatter nondeterminism and handles a class with exactly one TRAIN image.
    return torch.stack([features[labels == index].mean(0) for index in range(count)])


def _distance(features, means, cholesky):
    if hasattr(torch.linalg, "solve_triangular"):
        whitened = torch.linalg.solve_triangular(
            cholesky, (features - means).T, upper=False)
    else:  # torch.linalg.solve_triangular was introduced after torch 1.8.
        whitened = torch.triangular_solve(
            (features - means).T, cholesky, upper=False)[0]
    return whitened.square().sum(0)


class HierarchicalGeometry:
    """Closed-form candidate verifier with no trainable model parameters.

    Use ``fit`` to construct it or ``from_state_dict`` to reload its CPU tensor
    checkpoint. ``score`` returns detached tensors on the feature device, using
    float64 for double inputs and float32 otherwise. Device/dtype copies of the
    sufficient statistics are cached; no image features enter that cache.
    """

    @classmethod
    @torch.no_grad()
    def fit(cls, fine_features, parent_features, labels, leaf_to_parent,
            shrinkage=.1, ridge=1e-4):
        shrinkage, ridge = float(shrinkage), float(ridge)
        if not math.isfinite(shrinkage) or not 0 <= shrinkage <= 1:
            raise ValueError("shrinkage must be finite and in [0,1]")
        if not math.isfinite(ridge) or ridge < torch.finfo(torch.float32).tiny:
            raise ValueError("ridge must be positive and representable in float32")
        fine = _normalized(fine_features, "fine_features", "cpu", torch.float64)
        if not len(fine):
            raise ValueError("Known TRAIN must contain at least one image")
        parent = _normalized(parent_features, "parent_features", "cpu", torch.float64,
                             rows=len(fine))
        mapping = _integer_vector(leaf_to_parent, "leaf_to_parent")
        if not len(mapping):
            raise ValueError("leaf_to_parent must be nonempty")
        n_leaf, n_parent = len(mapping), int(mapping.max()) + 1
        if torch.unique(mapping).tolist() != list(range(n_parent)):
            raise ValueError("Parent IDs must be contiguous starting at zero")
        y = _integer_vector(labels, "labels", len(fine), n_leaf)
        leaf_counts = torch.bincount(y, minlength=n_leaf)
        if bool((leaf_counts == 0).any()):
            raise ValueError("Every mapped leaf must have known TRAIN observations")
        py = mapping[y]
        parent_counts = torch.bincount(py, minlength=n_parent)
        fine_leaf_means = _group_means(fine, y, n_leaf)
        fine_parent_means = _group_means(fine, py, n_parent)
        parent_means = _group_means(parent, py, n_parent)
        fine_global_mean, parent_global_mean = fine.mean(0), parent.mean(0)
        state = {
            "version": _VERSION, "shrinkage": shrinkage, "ridge": ridge,
            "leaf_to_parent": mapping.clone(), "leaf_counts": leaf_counts,
            "parent_counts": parent_counts,
            "parent_global_mean": parent_global_mean, "parent_means": parent_means,
            "fine_global_mean": fine_global_mean, "fine_parent_means": fine_parent_means,
            "fine_leaf_means": fine_leaf_means,
            "parent_global_cholesky": _covariance_factor(
                parent - parent_global_mean, shrinkage, ridge),
            "parent_within_cholesky": _covariance_factor(
                parent - parent_means[py], shrinkage, ridge),
            "fine_global_cholesky": _covariance_factor(
                fine - fine_global_mean, shrinkage, ridge),
            "fine_parent_cholesky": _covariance_factor(
                fine - fine_parent_means[py], shrinkage, ridge),
            "fine_leaf_cholesky": _covariance_factor(
                fine - fine_leaf_means[y], shrinkage, ridge),
        }
        return cls.from_state_dict(state)

    @classmethod
    def from_state_dict(cls, state):
        required = set(_FACTORS + _MEANS + (
            "version", "shrinkage", "ridge", "leaf_to_parent", "leaf_counts", "parent_counts"))
        if not isinstance(state, dict) or set(state) != required or state["version"] != _VERSION:
            raise ValueError("Invalid hierarchical geometry state schema/version")
        shrinkage, ridge = float(state["shrinkage"]), float(state["ridge"])
        if not math.isfinite(shrinkage) or not 0 <= shrinkage <= 1:
            raise ValueError("Invalid covariance shrinkage in state")
        if not math.isfinite(ridge) or ridge < torch.finfo(torch.float32).tiny:
            raise ValueError("Invalid positive covariance ridge in state")
        mapping = _integer_vector(state["leaf_to_parent"], "leaf_to_parent")
        if not len(mapping):
            raise ValueError("Empty taxonomy in state")
        n_leaf, n_parent = len(mapping), int(mapping.max()) + 1
        if torch.unique(mapping).tolist() != list(range(n_parent)):
            raise ValueError("Noncontiguous parent taxonomy in state")
        leaf_counts = _integer_vector(state["leaf_counts"], "leaf_counts", n_leaf)
        parent_counts = _integer_vector(state["parent_counts"], "parent_counts", n_parent)
        expected = torch.zeros(n_parent, dtype=torch.long).index_add_(0, mapping, leaf_counts)
        if bool((leaf_counts == 0).any()) or not torch.equal(expected, parent_counts):
            raise ValueError("Inconsistent or empty TRAIN counts in state")
        copied = {"version": _VERSION, "shrinkage": shrinkage, "ridge": ridge,
                  "leaf_to_parent": mapping.clone(), "leaf_counts": leaf_counts.clone(),
                  "parent_counts": parent_counts.clone()}
        for key in _FACTORS + _MEANS:
            raw = torch.as_tensor(state[key])
            if raw.is_complex() or not raw.is_floating_point() or not bool(torch.isfinite(raw).all()):
                raise ValueError("Nonfinite or non-real statistic: " + key)
            copied[key] = raw.detach().to(device="cpu", dtype=torch.float64).clone()
        fine_dim = copied["fine_global_mean"].numel()
        parent_dim = copied["parent_global_mean"].numel()
        if min(fine_dim, parent_dim) < 1:
            raise ValueError("Empty geometry feature dimension")
        shapes = {"fine_global_mean": (fine_dim,), "parent_global_mean": (parent_dim,),
                  "fine_leaf_means": (n_leaf, fine_dim),
                  "fine_parent_means": (n_parent, fine_dim),
                  "parent_means": (n_parent, parent_dim)}
        for key in _FACTORS:
            dimension = parent_dim if key.startswith("parent_") else fine_dim
            shapes[key] = (dimension, dimension)
        for key, shape in shapes.items():
            if tuple(copied[key].shape) != shape:
                raise ValueError("Invalid statistic shape: " + key)
        for key in _FACTORS:
            value = copied[key]
            if not torch.equal(value, value.tril()) or bool((value.diagonal() <= 0).any()):
                raise ValueError("Invalid lower Cholesky factor: " + key)
        model = cls()
        model._state = copied
        model._device_cache = {}
        model.fine_dimension, model.parent_dimension = fine_dim, parent_dim
        model.num_leaves, model.num_parents = n_leaf, n_parent
        return model

    def state_dict(self):
        """Return independent CPU tensors, suitable for ``torch.save``."""
        return {key: value.clone() if torch.is_tensor(value) else value
                for key, value in self._state.items()}

    @property
    def diagnostics(self):
        mapping = self._state["leaf_to_parent"]
        siblings = torch.bincount(mapping, minlength=self.num_parents)
        return {
            "method": "hierarchical_relative_mahalanobis",
            "fit_split": "known_train", "train_rows": int(self._state["leaf_counts"].sum()),
            "fine_dimension": self.fine_dimension, "parent_dimension": self.parent_dimension,
            "leaf_counts": self._state["leaf_counts"].tolist(),
            "parent_counts": self._state["parent_counts"].tolist(),
            "known_leaves_per_parent": siblings.tolist(),
            "singleton_parent_ids": torch.where(siblings == 1)[0].tolist(),
            "singleton_fine_background": "global_fine",
            "fine_background_covariance": "shared_pooled_within_parent",
            "covariance_weighting": "sample_weighted_mle",
            "shrinkage": self._state["shrinkage"], "ridge": self._state["ridge"],
            "covariance_count": len(_FACTORS), "fit_dtype": "float64_cpu",
            "trainable_parameters": 0,
        }

    def _on_device(self, device, dtype):
        key = (str(device), dtype)
        if key not in self._device_cache:
            self._device_cache[key] = {
                name: value.to(device=device, dtype=dtype if value.is_floating_point() else value.dtype)
                for name, value in self._state.items() if torch.is_tensor(value)}
        return self._device_cache[key]

    @torch.no_grad()
    def score(self, fine, parent, candidate_parent, candidate_leaf):
        fine_input, parent_input = torch.as_tensor(fine), torch.as_tensor(parent)
        if fine_input.device != parent_input.device:
            raise ValueError("Fine and parent features must be on the same device")
        device = fine_input.device
        dtype = torch.float64 if (fine_input.dtype == torch.float64 or
                                 parent_input.dtype == torch.float64) else torch.float32
        fine = _normalized(fine_input, "fine", device, dtype, dimension=self.fine_dimension)
        parent = _normalized(parent_input, "parent", device, dtype,
                             dimension=self.parent_dimension, rows=len(fine))
        cp = _integer_vector(candidate_parent, "candidate_parent", len(fine), self.num_parents, device)
        cl = _integer_vector(candidate_leaf, "candidate_leaf", len(fine), self.num_leaves, device)
        state = self._on_device(device, dtype)
        if not torch.equal(state["leaf_to_parent"][cl], cp):
            raise ValueError("Candidate leaf must belong to candidate parent")
        parent_score = (_distance(parent, state["parent_global_mean"], state["parent_global_cholesky"])
                        - _distance(parent, state["parent_means"][cp], state["parent_within_cholesky"]))
        fine_background = _distance(fine, state["fine_parent_means"][cp], state["fine_parent_cholesky"])
        sibling_counts = torch.bincount(state["leaf_to_parent"], minlength=self.num_parents)
        singleton = sibling_counts[cp] == 1
        if bool(singleton.any()):
            fine_background[singleton] = _distance(
                fine[singleton], state["fine_global_mean"], state["fine_global_cholesky"])
        leaf_score = fine_background - _distance(
            fine, state["fine_leaf_means"][cl], state["fine_leaf_cholesky"])
        if not bool(torch.isfinite(parent_score).all() and torch.isfinite(leaf_score).all()):
            raise ValueError("Nonfinite geometry evidence; check feature precision and ridge")
        return {"parent_score": parent_score, "leaf_score": leaf_score}


class RobustScoreStandardizer:
    """TRAIN-only median/MAD scale for one channel of candidate evidence.

    Pass only the known TRAIN rows with correct corresponding candidates to
    ``fit``. Median absolute deviation uses the normal-consistency constant.
    If MAD is zero, fall back to population standard deviation and then one.
    The serialized parameters are JSON compatible. This is a score transform,
    not a probability estimate or a distribution-free error guarantee.
    """

    @classmethod
    def fit(cls, scores):
        raw = torch.as_tensor(scores)
        if raw.is_complex() or not raw.is_floating_point():
            raise ValueError("Standardizer scores must be real floating-point values")
        values = raw.detach().to(device="cpu", dtype=torch.float64)
        if values.ndim != 1 or not len(values) or not bool(torch.isfinite(values).all()):
            raise ValueError("Standardizer requires nonempty finite one-dimensional TRAIN scores")
        center = float(torch.quantile(values, .5))
        scale = float(torch.quantile((values - center).abs(), .5)) * 1.482602218505602
        method = "median_mad"
        if not math.isfinite(scale) or scale <= 1e-12:
            scale, method = float(values.std(unbiased=False)), "median_std_fallback"
        if not math.isfinite(scale) or scale <= 1e-12:
            scale, method = 1., "median_unit_fallback"
        return cls.from_state_dict({"version": 1, "center": center, "scale": scale,
                                    "method": method, "fit_rows": len(values)})

    @classmethod
    def from_state_dict(cls, state):
        if not isinstance(state, dict) or set(state) != {"version", "center", "scale", "method", "fit_rows"}:
            raise ValueError("Invalid standardizer state schema")
        if state["version"] != 1 or state["method"] not in (
                "median_mad", "median_std_fallback", "median_unit_fallback"):
            raise ValueError("Invalid standardizer state version/method")
        center, scale = float(state["center"]), float(state["scale"])
        rows = state["fit_rows"]
        if not math.isfinite(center) or not math.isfinite(scale) or scale <= 0:
            raise ValueError("Standardizer center and positive scale must be finite")
        if isinstance(rows, bool) or not isinstance(rows, int) or rows < 1:
            raise ValueError("Standardizer fit_rows must be a positive integer")
        model = cls()
        model._state = dict(state, center=center, scale=scale)
        return model

    def state_dict(self):
        return dict(self._state)

    @torch.no_grad()
    def transform(self, scores):
        raw = torch.as_tensor(scores)
        if raw.is_complex() or not raw.is_floating_point() or not bool(torch.isfinite(raw).all()):
            raise ValueError("Scores to standardize must be real finite floating-point values")
        dtype = torch.float64 if raw.dtype == torch.float64 else torch.float32
        result = (raw.detach().to(dtype=dtype) - self._state["center"]) / self._state["scale"]
        if not bool(torch.isfinite(result).all()):
            raise ValueError("Standardized evidence overflowed; use float64 input")
        return result


def robust_standardizer_fit(scores):
    """Fit one channel and return JSON-compatible parameters."""
    return RobustScoreStandardizer.fit(scores).state_dict()


def robust_standardizer_transform(scores, parameters):
    """Apply previously fitted TRAIN parameters without refitting."""
    return RobustScoreStandardizer.from_state_dict(parameters).transform(scores)
