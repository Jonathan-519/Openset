"""Frozen known-TRAIN geometry required by D05 and TaxoSieve.

Relative Mahalanobis distance is an existing method, not a new contribution.
Here a shared diagonal covariance is shrunk toward its isotropic average.
Class-independent background distance minus class distance is high for members.
All fitting and scoring is float64 CPU. Query hashes must be disjoint from
support, including when computing means/covariances, not only nearest neighbours.
"""
import copy
import hashlib
import json
import math

import torch

from .tensor_utils import _features, _hashes, _ids, _meta


SCHEMA_VERSION = "discovery_geometry_v1"
ABSENT_SCORE = -1e6
FEATURE_NAMES = ("rmd", "distance", "cosine", "margin", "nearest", "nearest_margin")
STATISTICS = ("counts", "means", "global_mean", "within_variance", "global_variance")


def _parameters(shrinkage, ridge):
    if (isinstance(shrinkage, bool) or not isinstance(shrinkage, (int, float))
            or not math.isfinite(shrinkage) or not 0 <= shrinkage <= 1):
        raise ValueError("shrinkage must be finite and between zero and one")
    if (isinstance(ridge, bool) or not isinstance(ridge, (int, float))
            or not math.isfinite(ridge) or ridge < torch.finfo(torch.float32).tiny):
        raise ValueError("ridge must be finite, positive and representable in float32")
    return float(shrinkage), float(ridge)


def _state_digest(state):
    """Bind saved sufficient statistics and support without recomputing a fit."""
    digest = hashlib.sha256()
    header = {key: state[key] for key in ("schema_version", "image_hashes", "meta", "shrinkage", "ridge", "fit_report")}
    digest.update(json.dumps(header, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))
    tensors = [(key, state[key]) for key in ("fine", "parent", "labels")]
    tensors += [(level + "." + key, state["statistics"][level][key])
                for level in ("leaf", "parent") for key in STATISTICS]
    for name, value in tensors:
        value = value.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _variance(residual, shrinkage, ridge):
    value = residual.square().mean(0)
    return ((1. - shrinkage) * value + shrinkage * value.mean() + ridge)


def _statistics(features, labels, count, shrinkage, ridge):
    counts = torch.bincount(labels, minlength=count)
    means = torch.zeros((count, features.shape[1]), dtype=torch.float64)
    for c in range(count):
        if counts[c]:
            means[c] = features[labels == c].mean(0)
    global_mean = features.mean(0)
    return dict(counts=counts, means=means, global_mean=global_mean,
                within_variance=_variance(features - means[labels], shrinkage, ridge),
                global_variance=_variance(features - global_mean, shrinkage, ridge))


def _rival_margin(values, active):
    if int(active.sum()) < 2:
        return torch.zeros_like(values)
    masked = values.masked_fill(~active[None, :], ABSENT_SCORE)
    top = masked.topk(2, dim=1)
    rival = top.values[:, :1].expand_as(values).clone()
    rival.scatter_(1, top.indices[:, :1], top.values[:, 1:2])
    return values - rival


class GeometryBank:
    """A self-contained, strictly disjoint support bank; absent classes are masked."""

    @classmethod
    @torch.no_grad()
    def fit(cls, fine, parent, labels, image_hashes, meta, shrinkage=.1, ridge=1e-4):
        shrinkage, ridge = _parameters(shrinkage, ridge)
        result = cls()
        result.meta = _meta(meta)
        result.fine = _features(fine, "fine")
        if not len(result.fine):
            raise ValueError("TRAIN support cannot be empty")
        result.parent = _features(parent, "parent", count=len(result.fine))
        result.labels = _ids(labels, "labels", len(result.fine), len(result.meta["leaf_names"]))
        result.image_hashes = _hashes(image_hashes, len(result.fine), unique=True)
        result._hashes = set(result.image_hashes)
        result.shrinkage, result.ridge = float(shrinkage), float(ridge)
        result.parent_labels = torch.tensor(result.meta["leaf_to_parent"])[result.labels]
        result._stats = {
            "leaf": _statistics(result.fine, result.labels, len(result.meta["leaf_names"]), shrinkage, ridge),
            "parent": _statistics(result.parent, result.parent_labels, len(result.meta["parent_names"]), shrinkage, ridge),
        }
        result.fit_report = result._report()
        return result

    def _report(self):
        return dict(schema_version=SCHEMA_VERSION, fit_split="known_train",
            support_count=len(self.fine), leaf_counts=self._stats["leaf"]["counts"].tolist(),
            parent_counts=self._stats["parent"]["counts"].tolist(),
            covariance="shared_diagonal_shrunk_toward_isotropic", shrinkage=self.shrinkage,
            ridge=self.ridge, self_support_allowed=False, missing_support_score=ABSENT_SCORE,
            feature_dimensions={"leaf": self.fine.shape[1], "parent": self.parent.shape[1]},
            true_unknown_images_used=False, inference_recomputes_statistics=False)

    @torch.no_grad()
    def score(self, fine, parent, query_hashes):
        fine = _features(fine, "fine", dimension=self.fine.shape[1])
        parent = _features(parent, "parent", count=len(fine), dimension=self.parent.shape[1])
        hashes = _hashes(query_hashes, len(fine), unique=False)
        if self._hashes.intersection(hashes):
            raise ValueError("Query/support image hashes overlap; exclude query before all geometry fitting")
        output = {}
        for level, query, support, labels in (("leaf", fine, self.fine, self.labels),
                                               ("parent", parent, self.parent, self.parent_labels)):
            stat = self._stats[level]
            means, variance = stat["means"], stat["within_variance"]
            active = stat["counts"] > 0
            precision = variance.reciprocal()
            distance = (query.square() @ precision[:, None]
                        + (means.square() * precision).sum(1)[None, :]
                        - 2. * (query * precision) @ means.T).clamp_min(0.)
            background = ((query - stat["global_mean"]).square() / stat["global_variance"]).sum(1)
            normalized_mean = means / means.norm(dim=1, keepdim=True).clamp_min(1e-12)
            cosine = query @ normalized_mean.T
            nearest = torch.full_like(cosine, ABSENT_SCORE)
            for start in range(0, len(query), 128):
                block = query[start:start + 128] @ support.T
                for c in active.nonzero(as_tuple=True)[0].tolist():
                    nearest[start:start + 128, c] = block[:, labels == c].amax(1)
            values = dict(rmd=background[:, None] - distance, distance=-distance,
                          cosine=cosine, margin=_rival_margin(cosine, active),
                          nearest=nearest, nearest_margin=_rival_margin(nearest, active))
            for name, value in values.items():
                value = value.masked_fill(~active[None, :], ABSENT_SCORE)
                if not bool(torch.isfinite(value).all()):
                    raise ValueError("Geometry produced nonfinite evidence")
                output[level + "_" + name] = value
            output[level + "_active"] = active.clone()
        return output

    def state_dict(self):
        state = dict(schema_version=SCHEMA_VERSION, fine=self.fine.clone(), parent=self.parent.clone(),
            labels=self.labels.clone(), image_hashes=list(self.image_hashes), meta=copy.deepcopy(self.meta),
            shrinkage=self.shrinkage, ridge=self.ridge, fit_report=copy.deepcopy(self.fit_report),
            statistics={level: {key: value.clone() for key, value in stat.items()} for level, stat in self._stats.items()})
        state["state_sha256"] = _state_digest(state)
        return state

    @classmethod
    def from_state_dict(cls, state):
        keys = {"schema_version", "fine", "parent", "labels", "image_hashes", "meta", "shrinkage", "ridge", "fit_report", "statistics", "state_sha256"}
        if not isinstance(state, dict) or set(state) != keys or state["schema_version"] != SCHEMA_VERSION:
            raise ValueError("Invalid discovery geometry state schema")
        result = cls()
        result.shrinkage, result.ridge = _parameters(state["shrinkage"], state["ridge"])
        result.meta = _meta(state["meta"])
        # Validate and copy exact saved vectors/statistics. No call to fit,
        # _statistics, covariance calculation or mean estimation is permitted.
        # Renormalizing stored vectors could move threshold endpoints by an ulp.
        for name in ("fine", "parent"):
            if not torch.is_tensor(state[name]) or state[name].dtype != torch.float64:
                raise ValueError("Stored geometry support must be float64")
        result.fine = _features(state["fine"], "stored fine", normalized=True)
        if not len(result.fine):
            raise ValueError("Stored support cannot be empty")
        result.parent = _features(state["parent"], "stored parent", count=len(result.fine), normalized=True)
        result.labels = _ids(state["labels"], "stored labels", len(result.fine), len(result.meta["leaf_names"]))
        result.parent_labels = torch.tensor(result.meta["leaf_to_parent"])[result.labels]
        result.image_hashes = _hashes(state["image_hashes"], len(result.fine), unique=True)
        result._hashes = set(result.image_hashes)
        if not isinstance(state["statistics"], dict) or set(state["statistics"]) != {"leaf", "parent"}:
            raise ValueError("Stored geometry statistics lack a hierarchy level")
        result._stats = {}
        for level, features, labels in (("leaf", result.fine, result.labels), ("parent", result.parent, result.parent_labels)):
            count, dimension = len(result.meta["leaf_names" if level == "leaf" else "parent_names"]), features.shape[1]
            stat = state["statistics"][level]
            if not isinstance(stat, dict) or set(stat) != set(STATISTICS):
                raise ValueError("Stored geometry statistics schema differs")
            expected = {"counts": (count,), "means": (count, dimension), "global_mean": (dimension,),
                        "within_variance": (dimension,), "global_variance": (dimension,)}
            copied = {}
            for key, shape in expected.items():
                value = stat[key]
                dtype = torch.long if key == "counts" else torch.float64
                if (not torch.is_tensor(value) or value.dtype != dtype or tuple(value.shape) != shape
                        or not bool(torch.isfinite(value).all())):
                    raise ValueError("Invalid stored geometry statistic: " + key)
                copied[key] = value.detach().cpu().clone()
            if not torch.equal(copied["counts"], torch.bincount(labels, minlength=count)):
                raise ValueError("Stored class counts differ from support labels")
            if (bool((copied["within_variance"] < result.ridge).any())
                    or bool((copied["global_variance"] < result.ridge).any())
                    or bool((copied["means"].norm(dim=1) > 1. + 1e-12).any())
                    or float(copied["global_mean"].norm()) > 1. + 1e-12
                    or bool((copied["means"][copied["counts"] == 0] != 0).any())):
                raise ValueError("Stored covariance/mean statistics violate normalized support bounds")
            result._stats[level] = copied
        result.fit_report = result._report()
        if result.fit_report != state["fit_report"]:
            raise ValueError("Geometry state fit report differs from saved support/statistics metadata")
        if not isinstance(state["state_sha256"], str) or _state_digest(state) != state["state_sha256"]:
            raise ValueError("Geometry support/statistics state digest mismatch")
        return result


def evidence_features(scores, level, template_scores=None):
    """Per-candidate evidence; omitted text candidates never enter rival margins."""
    if level not in ("leaf", "parent"):
        raise ValueError("Invalid hierarchy level")
    values = [torch.as_tensor(scores[level + "_" + name], dtype=torch.float64) for name in FEATURE_NAMES]
    if any(value.ndim != 2 or value.shape != values[0].shape or not bool(torch.isfinite(value).all()) for value in values):
        raise ValueError("Geometry evidence must have matching finite [N,C] shapes")
    active = torch.as_tensor(scores[level + "_active"], dtype=torch.bool)
    if active.shape != (values[0].shape[1],) or not bool(active.any()):
        raise ValueError("Candidate support mask must contain at least one active class")
    if template_scores is not None:
        if not isinstance(template_scores, dict) or set(template_scores) != {"leaf", "parent"}:
            raise ValueError("template_scores must contain leaf and parent similarities")
        template = torch.as_tensor(template_scores[level], dtype=torch.float64).detach().cpu()
        if template.shape != values[0].shape or not bool(torch.isfinite(template).all()):
            raise ValueError("Template similarities must be finite and aligned with geometry")
        values += [template.masked_fill(~active[None, :], ABSENT_SCORE), _rival_margin(template, active)]
    return torch.stack(values, dim=2)
