"""TRAIN-only hierarchy-balanced PPCA evidence for a root-first decoder.

This is a statistical adaptation of low-rank Gaussian reconstruction, not a
reproduction of ViM, KPCA or a calibrated probability. Parent domains are fit
independently of leaf acceptance. Absolute and relative evidence are combined
within the SAME parent before taking the union over parents. Frozen CLIP
features are accepted already unit-normalized and are never normalized again.

All three-fold normalizer observations exclude the entire query fold from
means, covariance, shrinkage targets and subspaces. A support-only holdout is
not a claim that the source reference classifier never used those species.
"""
import copy
import hashlib
import json
import math

import numpy as np
import torch

from taxosafe_routealign.proximity import _hashes, _ids, _meta


SCHEMA_VERSION = "taxosafe_domain_ppca_v1"
VARIANCE_FLOOR = 1e-8
SCALE_FLOOR = 1e-6
NORMALIZER_STRENGTH = 20.0
UNIT_NORM_TOLERANCE = 2e-6
ABSENT_SCORE = -1e6
NORMALIZER_NAMES = ("parent_residual", "parent_density", "parent_relative", "leaf_conditional")
MODEL_KEYS = {"mean", "basis", "variances", "residual_variance", "logdet", "count",
              "rank", "requested_rank", "positive_eigenvalues", "raw_total_variance",
              "raw_residual_variance", "shrink_target", "fallbacks"}


def _digest(value):
    """A deterministic, typed recursive hash; tensors are bound byte-for-byte."""
    digest = hashlib.sha256()
    def visit(item):
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            digest.update(json.dumps(["tensor", str(tensor.dtype), list(tensor.shape)]).encode())
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict{")
            for key in sorted(item):
                visit(key)
                visit(item[key])
            digest.update(b"}")
        elif isinstance(item, (tuple, list)):
            digest.update(b"list[")
            for child in item:
                visit(child)
            digest.update(b"]")
        else:
            digest.update(json.dumps(item, sort_keys=True, allow_nan=False, separators=(",", ":")).encode())
            digest.update(b";")
    visit(value)
    return digest.hexdigest()


def _features(value, name, dimension=None):
    raw = torch.as_tensor(value)
    if (raw.ndim != 2 or raw.shape[1] < 1 or not raw.is_floating_point()
            or raw.is_complex() or (dimension is not None and raw.shape[1] != dimension)):
        raise ValueError(name + " must be a real floating [N,D] matrix with the fitted dimension")
    value = raw.detach().to(device="cpu", dtype=torch.float64).clone()
    if not bool(torch.isfinite(value).all()):
        raise ValueError(name + " must be finite")
    if not torch.allclose(value.norm(dim=1), torch.ones(len(value), dtype=torch.float64),
                          atol=UNIT_NORM_TOLERANCE, rtol=0.):
        raise ValueError(name + " must already be unit normalized; no implicit normalization is performed")
    return value


def _settings(parent_rank, leaf_rank, global_rank, shrinkage, folds, seed):
    for name, value in (("parent_rank", parent_rank), ("leaf_rank", leaf_rank), ("global_rank", global_rank)):
        if type(value) is not int or value < 0:
            raise ValueError(name + " must be a nonnegative integer")
    if (isinstance(shrinkage, bool) or not isinstance(shrinkage, (int, float))
            or not math.isfinite(shrinkage) or not 0. <= shrinkage <= 1.):
        raise ValueError("shrinkage must be finite in [0,1]")
    if type(folds) is not int or folds < 2:
        raise ValueError("folds must be an integer >= 2")
    if type(seed) is not int or not 0 <= seed < 2 ** 32:
        raise ValueError("seed must be an integer in [0,2**32)")
    return dict(parent_rank=parent_rank, leaf_rank=leaf_rank, global_rank=global_rank,
                shrinkage=float(shrinkage), folds=folds, seed=seed,
                variance_floor=VARIANCE_FLOOR, normalizer_strength=NORMALIZER_STRENGTH,
                normalizer_scale_floor=SCALE_FLOOR, unit_norm_tolerance=UNIT_NORM_TOLERANCE)


def _weights(labels, meta, level):
    """Population covariance weights: parent -> leaf -> image equal mass."""
    mapping = torch.tensor(meta["leaf_to_parent"], dtype=torch.long)
    counts = torch.bincount(labels, minlength=len(mapping)).double()
    active_leaves = (counts > 0).nonzero(as_tuple=True)[0]
    if not len(labels):
        return torch.empty(0, dtype=torch.float64)
    if level == "leaf":
        return torch.full((len(labels),), 1. / len(labels), dtype=torch.float64)
    if level == "parent":
        return counts[labels].reciprocal() / len(active_leaves)
    active_parents = sorted(set(mapping[active_leaves].tolist()))
    child_counts = torch.bincount(mapping[active_leaves], minlength=len(meta["parent_names"])).double()
    return 1. / (len(active_parents) * child_counts[mapping[labels]] * counts[labels])


def _eigh(matrix):
    """Old Torch without the linalg namespace uses the same CPU LAPACK task."""
    implementation = getattr(getattr(torch, "linalg", None), "eigh", None)
    if callable(implementation):
        return implementation(matrix)
    values, vectors = np.linalg.eigh(matrix.numpy())
    return torch.from_numpy(values.copy()), torch.from_numpy(vectors.copy())


def _fit_model(features, weights, requested_rank, target, shrinkage):
    """Exact weighted population spectrum via the smaller primal/dual matrix."""
    n, dimension = features.shape
    fallbacks = []
    if not n:
        mean = torch.zeros(dimension, dtype=torch.float64)
        centered = features
        eigenvalues = torch.empty(0, dtype=torch.float64)
        vectors = torch.empty((dimension, 0), dtype=torch.float64)
        total = 0.
        fallbacks.append("absent_support")
    else:
        mean = (features * weights[:, None]).sum(0)
        centered = features - mean
        weighted = centered * weights.sqrt()[:, None]
        total = float(weighted.square().sum())
        if n < 2 or total <= VARIANCE_FLOOR:
            eigenvalues = torch.empty(0, dtype=torch.float64)
            vectors = torch.empty((dimension, 0), dtype=torch.float64)
            fallbacks.append("singleton_support" if n < 2 else "zero_or_tiny_variance")
        else:
            matrix = weighted @ weighted.T if n < dimension else weighted.T @ weighted
            eigenvalues, vectors = _eigh(matrix)
            # eigh is ascending; flip is legal here (no candidate/tie ordering).
            eigenvalues = eigenvalues.flip(0).clamp_min(0.)
            vectors = vectors.flip(1)
            positive = eigenvalues > max(float(eigenvalues[0]) * 1e-10, 1e-12)
            eigenvalues, vectors = eigenvalues[positive], vectors[:, positive]
            if n < dimension:
                vectors = weighted.T @ vectors / eigenvalues.sqrt()[None, :]
    positive_count = len(eigenvalues)
    rank = min(requested_rank, max(n - 1, 0), max(dimension - 1, 0), positive_count)
    if rank < requested_rank:
        fallbacks.append("rank_capped")
    basis = vectors[:, :rank].contiguous()
    top = eigenvalues[:rank]
    raw_residual = max(0., (total - float(top.sum())) / (dimension - rank))
    residual_variance = max((1. - shrinkage) * raw_residual + shrinkage * target, VARIANCE_FLOOR)
    variances = ((1. - shrinkage) * top + shrinkage * target).clamp_min(VARIANCE_FLOOR)
    if residual_variance == VARIANCE_FLOOR:
        fallbacks.append("residual_variance_floor")
    if rank and bool((variances == VARIANCE_FLOOR).any()):
        fallbacks.append("principal_variance_floor")
    return dict(mean=mean, basis=basis, variances=variances, residual_variance=residual_variance,
                logdet=float(variances.log().sum()) + (dimension - rank) * math.log(residual_variance),
                count=n, rank=rank, requested_rank=requested_rank, positive_eigenvalues=positive_count,
                raw_total_variance=total, raw_residual_variance=raw_residual,
                shrink_target=target, fallbacks=fallbacks)


def _fit_models(features, labels, meta, settings):
    weights = _weights(labels, meta, "global")
    mean = (features * weights[:, None]).sum(0)
    target_raw = float(((features - mean).square().sum(1) * weights).sum()) / features.shape[1]
    target = max(target_raw, VARIANCE_FLOOR)
    global_model = _fit_model(features, weights, settings["global_rank"], target, settings["shrinkage"])
    mapping = torch.tensor(meta["leaf_to_parent"], dtype=torch.long)
    models = {"global": global_model, "parent": [], "leaf": []}
    for level, count, ids in (("parent", len(meta["parent_names"]), mapping[labels]),
                              ("leaf", len(meta["leaf_names"]), labels)):
        for node in range(count):
            mask = ids == node
            models[level].append(_fit_model(features[mask], _weights(labels[mask], meta, level),
                                 settings[level + "_rank"], target, settings["shrinkage"]))
    return models


def _model_scores(features, model):
    delta = features - model["mean"]
    projection = delta @ model["basis"]
    orthogonal = (delta.square().sum(1) - projection.square().sum(1)).clamp_min(0.)
    mahalanobis = (projection.square() / model["variances"]).sum(1) + orthogonal / model["residual_variance"]
    density = -.5 * (features.shape[1] * math.log(2. * math.pi) + model["logdet"] + mahalanobis)
    return -orthogonal, density


def _raw_scores(features, models, meta):
    dimension = features.shape[1]
    _, global_ll = _model_scores(features, models["global"])
    parent = [_model_scores(features, model) for model in models["parent"]]
    parent_residual = torch.stack([item[0] for item in parent], dim=1)
    parent_ll = torch.stack([item[1] for item in parent], dim=1)
    leaf_ll = torch.stack([_model_scores(features, model)[1] for model in models["leaf"]], dim=1)
    mapping = torch.tensor(meta["leaf_to_parent"], dtype=torch.long)
    return dict(parent_residual=parent_residual, parent_density=parent_ll / dimension,
                parent_relative=(parent_ll - global_ll[:, None]) / dimension,
                leaf_conditional=(leaf_ll - parent_ll[:, mapping]) / dimension)


def _assign_folds(labels, folds, seed):
    rng = np.random.RandomState(seed)
    result = torch.empty(len(labels), dtype=torch.long)
    offset = 0
    for node in sorted(set(labels.tolist())):
        indices = (labels == node).nonzero(as_tuple=True)[0].numpy()
        indices = rng.permutation(indices)
        assignments = (np.arange(len(indices)) + offset) % folds
        result[torch.from_numpy(indices.copy())] = torch.from_numpy(assignments.astype(np.int64))
        offset = (offset + len(indices)) % folds
    return result


def _normalizer(values, valid, ids, count):
    pooled = values[valid]
    global_mean = float(pooled.mean()) if len(pooled) else 0.
    global_second = float(pooled.square().mean()) if len(pooled) else 1.
    locations, scales, details = [], [], []
    for node in range(count):
        selected = values[valid & (ids == node)]
        n = len(selected)
        weight = n / (n + NORMALIZER_STRENGTH)
        local_mean = float(selected.mean()) if n else global_mean
        local_second = float(selected.square().mean()) if n else global_second
        location = weight * local_mean + (1. - weight) * global_mean
        second = weight * local_second + (1. - weight) * global_second
        scale = max(math.sqrt(max(0., second - location * location)), SCALE_FLOOR)
        locations.append(location)
        scales.append(scale)
        details.append(dict(valid_oof_count=n, pooling_weight=weight, location=location, scale=scale,
                            global_fallback=n == 0, scale_at_floor=scale == SCALE_FLOOR))
    return dict(location=torch.tensor(locations, dtype=torch.float64), scale=torch.tensor(scales, dtype=torch.float64),
                report=dict(global_valid_oof_count=len(pooled), global_mean=global_mean,
                            global_second_moment=global_second, global_fallback=not bool(len(pooled)),
                            pooling_strength=NORMALIZER_STRENGTH, groups=details))


def _model_report(model):
    return {key: copy.deepcopy(value) for key, value in model.items() if key not in ("mean", "basis", "variances")}


class DomainBank:
    """Immutable statistical support and TRAIN-only cross-fitted score scales."""

    @classmethod
    @torch.no_grad()
    def fit(cls, features, labels, image_hashes, meta, parent_rank=8, leaf_rank=4,
            global_rank=16, shrinkage=.1, folds=3, seed=1):
        result = cls()
        result.settings = _settings(parent_rank, leaf_rank, global_rank, shrinkage, folds, seed)
        result.features = _features(features, "TRAIN features")
        result.meta = _meta(meta)
        result.labels = _ids(labels, "TRAIN labels", len(result.features), len(result.meta["leaf_names"]))
        result.image_hashes = _hashes(image_hashes, len(result.features), unique=True)
        result._hashes = set(result.image_hashes)
        if len(result.features) < folds:
            raise ValueError("TRAIN needs at least folds images")
        counts = torch.bincount(result.labels, minlength=len(result.meta["leaf_names"]))
        if bool((counts == 0).any()):
            raise ValueError("Every known leaf must have full-TRAIN support")
        mapping = torch.tensor(result.meta["leaf_to_parent"], dtype=torch.long)
        parent_labels = mapping[result.labels]
        assignment = _assign_folds(result.labels, folds, seed)
        oof = {"fold_assignment": assignment, "values": {}, "valid": {}}
        for name in NORMALIZER_NAMES:
            oof["values"][name] = torch.zeros(len(result.labels), dtype=torch.float64)
            oof["valid"][name] = torch.zeros(len(result.labels), dtype=torch.bool)
        fold_reports = []
        for fold in range(folds):
            support, query = assignment != fold, assignment == fold
            models = _fit_models(result.features[support], result.labels[support], result.meta, result.settings)
            raw = _raw_scores(result.features[query], models, result.meta)
            parent_active = torch.tensor([model["count"] > 0 for model in models["parent"]])
            leaf_active = torch.tensor([model["count"] > 0 for model in models["leaf"]])
            query_parents, query_leaves = parent_labels[query], result.labels[query]
            for name in NORMALIZER_NAMES:
                ids = query_leaves if name == "leaf_conditional" else query_parents
                valid = leaf_active[query_leaves] & parent_active[query_parents] if name == "leaf_conditional" else parent_active[query_parents]
                selected = raw[name][torch.arange(int(query.sum())), ids]
                oof["values"][name][query] = torch.where(valid, selected, torch.zeros_like(selected))
                oof["valid"][name][query] = valid
            fold_reports.append(dict(fold=fold, support_count=int(support.sum()), query_count=int(query.sum()),
                support_hashes=[h for h, keep in zip(result.image_hashes, support.tolist()) if keep],
                query_hashes=[h for h, keep in zip(result.image_hashes, query.tolist()) if keep],
                parent_counts=[model["count"] for model in models["parent"]],
                leaf_counts=[model["count"] for model in models["leaf"]],
                models={level: [_model_report(model) for model in models[level]] for level in ("parent", "leaf")},
                global_model=_model_report(models["global"])))
        result.models = _fit_models(result.features, result.labels, result.meta, result.settings)
        result.oof = oof
        result.normalizers = {}
        for name in NORMALIZER_NAMES:
            ids = result.labels if name == "leaf_conditional" else parent_labels
            count = len(result.meta["leaf_names"] if name == "leaf_conditional" else result.meta["parent_names"])
            result.normalizers[name] = _normalizer(oof["values"][name], oof["valid"][name], ids, count)
        result.fit_report = dict(schema_version=SCHEMA_VERSION, fit_split="known_train", optimizer_steps=0,
            support_count=len(result.features), feature_dimension=result.features.shape[1], settings=copy.deepcopy(result.settings),
            source_feature_sha256=_digest(result.features), source_label_sha256=_digest(result.labels),
            train_identity_sha256=_digest(list(result.image_hashes)), taxonomy_sha256=_digest(result.meta),
            true_unknown_images_used=False, dev_used_for_fit=False, test_used_for_fit=False,
            input_features_renormalized=False, self_support_allowed=False, inference_recomputes_statistics=False,
            covariance="weighted_population_low_rank_plus_isotropic", weighting="global:equal_parent_equal_child_equal_image;parent:equal_child_equal_image;leaf:equal_image",
            shrink_target="same_bank_hierarchy_balanced_global_total_covariance_trace_divided_by_dimension",
            normalizer_provenance="true_class_scores_from_whole_fold_excluded_known_TRAIN_only",
            normalizer_method="n/(n+20) pooling of group/global mean and second moment; std floor 1e-6",
            score_direction="higher_is_more_supported;not_probability", relative_requires_same_parent_absolute=True,
            fitted_banks=folds + 1, fitted_models=(folds + 1) * (1 + len(result.meta["parent_names"]) + len(result.meta["leaf_names"])),
            oof_sha256=_digest(oof), fold_reports=fold_reports,
            full_models={level: [_model_report(model) for model in result.models[level]] for level in ("parent", "leaf")},
            full_global_model=_model_report(result.models["global"]),
            normalizers={name: copy.deepcopy(value["report"]) for name, value in result.normalizers.items()})
        return result

    @torch.no_grad()
    def score(self, features, image_hashes):
        features = _features(features, "query features", self.features.shape[1])
        hashes = _hashes(image_hashes, len(features), unique=False)
        if self._hashes.intersection(hashes):
            raise ValueError("Query/TRAIN support hash overlap; inference cannot score its fitted support")
        raw = _raw_scores(features, self.models, self.meta)
        z = {name: (values - self.normalizers[name]["location"]) / self.normalizers[name]["scale"]
             for name, values in raw.items()}
        parent_scores = torch.minimum(z["parent_density"], z["parent_relative"])
        output = dict(root_residual=z["parent_residual"].max(1).values,
                      root_density=z["parent_density"].max(1).values,
                      root_dual=parent_scores.max(1).values, parent_scores=parent_scores,
                      leaf_scores=z["leaf_conditional"],
                      parent_active=torch.ones(len(self.meta["parent_names"]), dtype=torch.bool),
                      leaf_active=torch.ones(len(self.meta["leaf_names"]), dtype=torch.bool))
        output.update({"raw_" + name: value for name, value in raw.items()})
        if any(not bool(torch.isfinite(value).all()) for value in output.values()):
            raise ValueError("Domain evidence is nonfinite")
        return output

    def state_dict(self):
        state = copy.deepcopy(dict(schema_version=SCHEMA_VERSION, features=self.features, labels=self.labels,
            image_hashes=list(self.image_hashes), meta=self.meta, settings=self.settings, models=self.models,
            normalizers=self.normalizers, oof=self.oof, fit_report=self.fit_report))
        state["state_sha256"] = _digest(state)
        return state

    @classmethod
    def from_state_dict(cls, state):
        expected = {"schema_version", "features", "labels", "image_hashes", "meta", "settings", "models",
                    "normalizers", "oof", "fit_report", "state_sha256"}
        if not isinstance(state, dict) or set(state) != expected or state["schema_version"] != SCHEMA_VERSION:
            raise ValueError("Invalid DomainBank state schema")
        payload = {key: value for key, value in state.items() if key != "state_sha256"}
        if not isinstance(state["state_sha256"], str) or _digest(payload) != state["state_sha256"]:
            raise ValueError("DomainBank state digest mismatch")
        result = cls()
        if not isinstance(state["settings"], dict):
            raise ValueError("Invalid saved settings")
        setting = state["settings"]
        try:
            result.settings = _settings(*(setting[key] for key in ("parent_rank", "leaf_rank", "global_rank", "shrinkage", "folds", "seed")))
        except KeyError as error:
            raise ValueError("Missing saved settings") from error
        if result.settings != setting:
            raise ValueError("Saved numerical settings differ from this schema")
        result.meta = _meta(state["meta"])
        if not torch.is_tensor(state["features"]) or state["features"].dtype != torch.float64:
            raise ValueError("Saved features must be float64")
        result.features = _features(state["features"], "saved TRAIN features")
        n, dimension = result.features.shape
        if n < result.settings["folds"]:
            raise ValueError("Saved support is too small")
        if not torch.is_tensor(state["labels"]) or state["labels"].dtype != torch.long:
            raise ValueError("Saved labels must be long")
        result.labels = _ids(state["labels"], "saved labels", n, len(result.meta["leaf_names"]))
        result.image_hashes = _hashes(state["image_hashes"], n, unique=True)
        result._hashes = set(result.image_hashes)
        leaf_counts = torch.bincount(result.labels, minlength=len(result.meta["leaf_names"]))
        if bool((leaf_counts == 0).any()):
            raise ValueError("Saved full TRAIN lacks a leaf")
        mapping = torch.tensor(result.meta["leaf_to_parent"], dtype=torch.long)
        parent_counts = torch.bincount(mapping[result.labels], minlength=len(result.meta["parent_names"]))
        models = state["models"]
        if not isinstance(models, dict) or set(models) != {"global", "parent", "leaf"}:
            raise ValueError("Saved model hierarchy differs")
        for level, counts in (("global", [n]), ("parent", parent_counts.tolist()), ("leaf", leaf_counts.tolist())):
            items = [models[level]] if level == "global" else models[level]
            if not isinstance(items, list) or len(items) != len(counts):
                raise ValueError("Saved model count differs")
            for model, count in zip(items, counts):
                _validate_model(model, count, dimension, result.settings[level + "_rank"], result.settings["shrinkage"])
                if model["shrink_target"] != models["global"]["shrink_target"]:
                    raise ValueError("Saved model shrink target is not the same bank global target")
        result.models = copy.deepcopy(models)
        _validate_oof(state["oof"], result.labels, result.meta, result.settings, result.image_hashes, state["fit_report"])
        result.oof = copy.deepcopy(state["oof"])
        normals = state["normalizers"]
        if not isinstance(normals, dict) or set(normals) != set(NORMALIZER_NAMES):
            raise ValueError("Saved normalizer keys differ")
        for name in NORMALIZER_NAMES:
            count = len(result.meta["leaf_names"] if name == "leaf_conditional" else result.meta["parent_names"])
            normal = normals[name]
            if not isinstance(normal, dict) or set(normal) != {"location", "scale", "report"}:
                raise ValueError("Saved normalizer schema differs")
            for key in ("location", "scale"):
                _tensor(normal[key], (count,), torch.float64, "normalizer " + key)
            if bool((normal["scale"] < SCALE_FLOOR).any()):
                raise ValueError("Saved normalizer scale violates floor")
            details = normal["report"].get("groups", [])
            if (len(details) != count or any(details[i]["location"] != float(normal["location"][i])
                    or details[i]["scale"] != float(normal["scale"][i]) for i in range(count))):
                raise ValueError("Saved normalizer report differs from statistics")
        result.normalizers = copy.deepcopy(normals)
        report = state["fit_report"]
        required = dict(schema_version=SCHEMA_VERSION, fit_split="known_train", optimizer_steps=0,
            support_count=n, feature_dimension=dimension, settings=result.settings,
            source_feature_sha256=_digest(result.features), source_label_sha256=_digest(result.labels),
            train_identity_sha256=_digest(list(result.image_hashes)), taxonomy_sha256=_digest(result.meta),
            true_unknown_images_used=False, dev_used_for_fit=False, test_used_for_fit=False,
            input_features_renormalized=False, self_support_allowed=False, inference_recomputes_statistics=False,
            relative_requires_same_parent_absolute=True, fitted_banks=result.settings["folds"] + 1,
            fitted_models=(result.settings["folds"] + 1) * (1 + len(parent_counts) + len(leaf_counts)),
            oof_sha256=_digest(result.oof), full_global_model=_model_report(result.models["global"]),
            full_models={level: [_model_report(model) for model in result.models[level]] for level in ("parent", "leaf")},
            normalizers={name: value["report"] for name, value in result.normalizers.items()})
        if not isinstance(report, dict) or any(report.get(key) != value for key, value in required.items()):
            raise ValueError("Saved fit report/provenance differs from frozen artifacts")
        result.fit_report = copy.deepcopy(report)
        return result


def _tensor(value, shape, dtype, name):
    if (not torch.is_tensor(value) or value.dtype != dtype or tuple(value.shape) != tuple(shape)
            or value.device.type != "cpu"
            or not bool(torch.isfinite(value).all())):
        raise ValueError("Invalid saved " + name)


def _validate_model(model, count, dimension, requested_rank, shrinkage):
    if not isinstance(model, dict) or set(model) != MODEL_KEYS:
        raise ValueError("Saved PPCA model schema differs")
    rank = model["rank"]
    if (type(rank) is not int or type(model["positive_eigenvalues"]) is not int
            or not 0 <= model["positive_eigenvalues"] <= min(count - 1, dimension)
            or rank != min(requested_rank, count - 1, dimension - 1, model["positive_eigenvalues"])
            or model["count"] != count or model["requested_rank"] != requested_rank):
        raise ValueError("Saved PPCA rank/count constraints differ")
    _tensor(model["mean"], (dimension,), torch.float64, "PPCA mean")
    _tensor(model["basis"], (dimension, rank), torch.float64, "PPCA basis")
    _tensor(model["variances"], (rank,), torch.float64, "PPCA variances")
    if float(model["mean"].norm()) > 1. + UNIT_NORM_TOLERANCE:
        raise ValueError("Saved PPCA mean exceeds normalized feature bounds")
    if rank and not torch.allclose(model["basis"].T @ model["basis"], torch.eye(rank, dtype=torch.float64), atol=2e-6, rtol=0.):
        raise ValueError("Saved PPCA basis is not orthonormal")
    for key in ("residual_variance", "logdet", "raw_total_variance", "raw_residual_variance", "shrink_target"):
        if isinstance(model[key], bool) or not isinstance(model[key], (float, int)) or not math.isfinite(model[key]):
            raise ValueError("Invalid saved PPCA scalar " + key)
    rho = model["residual_variance"]
    target = model["shrink_target"]
    if (rho < VARIANCE_FLOOR or target < VARIANCE_FLOOR or model["raw_residual_variance"] < 0
            or model["raw_total_variance"] < 0 or bool((model["variances"] < VARIANCE_FLOOR).any())
            or rho != max((1. - shrinkage) * model["raw_residual_variance"] + shrinkage * target, VARIANCE_FLOOR)):
        raise ValueError("Saved PPCA variance/floor constraints differ")
    if rank and (bool((model["variances"][1:] > model["variances"][:-1] + 1e-12).any())
                 or float(model["variances"].min()) + 1e-10 < rho):
        raise ValueError("Saved PPCA principal variances are inconsistent")
    expected = float(model["variances"].log().sum()) + (dimension - rank) * math.log(rho)
    if not math.isclose(model["logdet"], expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("Saved PPCA logdet differs")
    allowed = {"absent_support", "singleton_support", "zero_or_tiny_variance", "rank_capped",
               "residual_variance_floor", "principal_variance_floor"}
    if not isinstance(model["fallbacks"], list) or any(item not in allowed for item in model["fallbacks"]):
        raise ValueError("Saved PPCA fallback audit differs")


def _validate_oof(oof, labels, meta, settings, hashes, report):
    if not isinstance(oof, dict) or set(oof) != {"fold_assignment", "values", "valid"}:
        raise ValueError("Saved OOF schema differs")
    n = len(labels)
    _tensor(oof["fold_assignment"], (n,), torch.long, "OOF assignment")
    assignment = oof["fold_assignment"]
    if not torch.equal(assignment, _assign_folds(labels, settings["folds"], settings["seed"])):
        raise ValueError("Saved OOF assignments differ from stratified seed protocol")
    if set(oof["values"]) != set(NORMALIZER_NAMES) or set(oof["valid"]) != set(NORMALIZER_NAMES):
        raise ValueError("Saved OOF score fields differ")
    for name in NORMALIZER_NAMES:
        _tensor(oof["values"][name], (n,), torch.float64, "OOF scores")
        _tensor(oof["valid"][name], (n,), torch.bool, "OOF validity")
        if bool((oof["values"][name][~oof["valid"][name]] != 0).any()):
            raise ValueError("Absent-support OOF scores must use zero with an explicit invalid mask")
    folds = report.get("fold_reports", [])
    if len(folds) != settings["folds"]:
        raise ValueError("Saved OOF fold reports differ")
    mapping = torch.tensor(meta["leaf_to_parent"], dtype=torch.long)
    for fold, item in enumerate(folds):
        support, query = assignment != fold, assignment == fold
        lc = torch.bincount(labels[support], minlength=len(meta["leaf_names"]))
        pc = torch.bincount(mapping[labels[support]], minlength=len(meta["parent_names"]))
        expected = dict(fold=fold, support_count=int(support.sum()), query_count=int(query.sum()),
            support_hashes=[h for h, keep in zip(hashes, support.tolist()) if keep],
            query_hashes=[h for h, keep in zip(hashes, query.tolist()) if keep],
            leaf_counts=lc.tolist(), parent_counts=pc.tolist())
        if any(item.get(key) != value for key, value in expected.items()):
            raise ValueError("Saved OOF support/query provenance differs")
        for name in NORMALIZER_NAMES:
            valid = pc[mapping[labels[query]]] > 0
            if name == "leaf_conditional":
                valid &= lc[labels[query]] > 0
            if not torch.equal(oof["valid"][name][query], valid):
                raise ValueError("Saved OOF support validity differs")
