"""TRAIN-only witness boundaries and matched-bank cross-query verification.

The Weibull fit is a two-parameter, zero-location MLE adaptation inspired by
EVM, not a reproduction of libMR. All witnesses are retained. Support holdouts
are pseudo unknowns: the source reference classifier/verifier used all known
TRAIN classes. Vanilla frozen CLIP is not fine-tuned here, and support-only
holdouts are not model-level unseen-class training.
"""
import copy
import hashlib
import json
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from taxosafe_discovery.geometry import GeometryBank, evidence_features
from taxosafe_discovery.verifier import SharedVerifier, KINDS
from taxosafe_routealign.proximity import _features, _hashes, _ids, _meta

BANK_SCHEMA = "boundary_bank_v1"
EPISODE_SCHEMA = "boundary_episodes_v1"
VERIFIER_SCHEMA = "boundary_verifier_v1"
LEVELS = ("leaf", "parent")
ABSENT = -1e6
EPS = 1e-12
MODES = ("bce8", "rank8", "bce9", "rank9")


def _stable_argsort(values, descending=False):
    """Stable CPU indices without requiring PyTorch's newer ``stable`` API.

    Only finite one-dimensional CPU float32/float64/int32/int64 tensors are
    accepted; the result is a CPU int64 tensor. Descending order reverses the
    order of equal-value groups, never the order of elements inside a tie.
    This also avoids negating int64's minimum value or rounding integer keys.
    """
    if (not torch.is_tensor(values) or values.device.type != "cpu" or values.ndim != 1
            or values.dtype not in (torch.float32, torch.float64, torch.int32, torch.int64)
            or type(descending) is not bool):
        raise ValueError("Stable sorting requires a CPU real/integer vector and boolean direction")
    array = values.detach().numpy()
    if not bool(np.isfinite(array).all()):
        raise ValueError("Stable sorting requires finite values")
    order = np.argsort(array, kind="mergesort")
    if descending and len(order):
        ordered = array[order]
        boundaries = np.flatnonzero(ordered[1:] != ordered[:-1]) + 1
        order = np.concatenate(np.split(order, boundaries)[::-1])
    return torch.from_numpy(np.asarray(order, dtype=np.int64).copy())


def _has_common_query(left, right):
    """Check query identity overlap using only the established CPU int64 API."""
    if any(not torch.is_tensor(value) or value.device.type != "cpu" or value.ndim != 1
           or value.dtype != torch.long for value in (left, right)):
        raise ValueError("Query identity overlap requires CPU int64 vectors")
    return bool(set(left.tolist()).intersection(right.tolist()))


def _digest(value):
    h = hashlib.sha256()
    def visit(v):
        if torch.is_tensor(v):
            t = v.detach().cpu().contiguous()
            h.update(json.dumps(["tensor", str(t.dtype), list(t.shape)]).encode())
            h.update(t.numpy().tobytes())
        elif isinstance(v, dict):
            h.update(b"{")
            for k in sorted(v):
                h.update(json.dumps(k).encode()); visit(v[k])
            h.update(b"}")
        elif isinstance(v, (list, tuple)):
            h.update(b"[")
            for item in v: visit(item)
            h.update(b"]")
        else:
            h.update(json.dumps(v, sort_keys=True, allow_nan=False).encode())
    visit(value)
    return h.hexdigest()


def _state_hash(state):
    return _digest({k: v for k, v in state.items() if k != "state_sha256"})


def _hash_digest(values):
    return hashlib.sha256("\n".join(sorted(values)).encode()).hexdigest()


def _distance(a, b):
    # No N x N x D intermediate. Distances are Euclidean on unit vectors.
    return (a.square().sum(1)[:, None] + b.square().sum(1)[None, :]
            - 2 * a @ b.T).clamp_min(0).sqrt()


@torch.no_grad()
def _fit_weibull(distances, labels, tail_size):
    n = len(labels)
    shape = torch.ones(n, dtype=torch.float64)
    scale = torch.full((n,), EPS, dtype=torch.float64)
    report = dict(witnesses=n, tail_size=tail_size, fallback_witnesses=0,
                  no_negative_witnesses=0, insufficient_tail_witnesses=0,
                  zero_tail_witnesses=0, constant_tail_witnesses=0,
                  shape_clamped_low=0, shape_clamped_high=0,
                  effective_tail_min=tail_size, effective_tail_max=0)
    # Class-wise groups have equally sized negative sets and can be fitted at once.
    for c in labels.unique().tolist():
        own = (labels == c).nonzero(as_tuple=True)[0]
        other = (labels != c).nonzero(as_tuple=True)[0]
        k = min(tail_size, len(other))
        report["effective_tail_min"] = min(report["effective_tail_min"], k)
        report["effective_tail_max"] = max(report["effective_tail_max"], k)
        if not k:
            report["no_negative_witnesses"] += len(own)
            report["fallback_witnesses"] += len(own)
            continue
        tail = (distances[own][:, other] * .5).topk(k, largest=False).values
        zero = (tail <= EPS).any(1)
        constant = (tail.amax(1) - tail.amin(1)) <= EPS
        insufficient = torch.full((len(own),), k < 2, dtype=torch.bool)
        fallback = zero | constant | insufficient
        report["zero_tail_witnesses"] += int(zero.sum())
        report["constant_tail_witnesses"] += int(constant.sum())
        report["insufficient_tail_witnesses"] += int(insufficient.sum())
        report["fallback_witnesses"] += int(fallback.sum())
        # Explicit exponential fallback; never silently claim successful MLE.
        scale[own] = tail.clamp_min(EPS).mean(1).clamp_min(EPS)
        valid = ~fallback
        if not bool(valid.any()):
            continue
        logx = tail[valid].log()
        meanlog = logx.mean(1)
        def equation(a):
            return a.reciprocal() + meanlog - (torch.softmax(a[:, None] * logx, 1) * logx).sum(1)
        lo = torch.full_like(meanlog, .2)
        hi = torch.full_like(meanlog, 10.)
        report["shape_clamped_low"] += int((equation(lo) <= 0).sum())
        report["shape_clamped_high"] += int((equation(hi) >= 0).sum())
        for _ in range(48):
            mid = (lo + hi) * .5
            positive = equation(mid) > 0
            lo = torch.where(positive, mid, lo)
            hi = torch.where(positive, hi, mid)
        fitted = (lo + hi) * .5
        fitted_scale = ((torch.logsumexp(fitted[:, None] * logx, 1) - math.log(k)) / fitted).exp()
        shape[own[valid]], scale[own[valid]] = fitted, fitted_scale.clamp_min(EPS)
    if not bool(torch.isfinite(shape).all() & torch.isfinite(scale).all()):
        raise ValueError("Weibull fit produced nonfinite parameters")
    return shape, scale, report


class BoundaryBank:
    """Immutable all-witness TRAIN boundary bank, independent of calibration."""
    @classmethod
    @torch.no_grad()
    def fit(cls, fine, parent, labels, image_hashes, meta, tail_size=32):
        meta = _meta(meta)
        fine = _features(fine, "fine")
        parent = _features(parent, "parent", count=len(fine))
        labels = _ids(labels, "labels", len(fine), len(meta["leaf_names"]))
        hashes = _hashes(image_hashes, len(fine), unique=True)
        return cls._fit_precomputed(fine, parent, labels, hashes, meta, tail_size,
                                    {"leaf": _distance(fine, fine), "parent": _distance(parent, parent)})

    @classmethod
    def _fit_precomputed(cls, fine, parent, labels, hashes, meta, tail_size, distances):
        if type(tail_size) is not int or tail_size < 2 or not len(fine):
            raise ValueError("Boundary fitting requires support and integer tail_size >= 2")
        result = cls()
        result.fine, result.parent = fine.clone(), parent.clone()
        result.labels, result.image_hashes = labels.clone(), list(hashes)
        result.meta, result.tail_size = copy.deepcopy(meta), tail_size
        result._hashes = set(hashes)
        result.parent_labels = torch.tensor(meta["leaf_to_parent"])[labels]
        result.parameters, reports = {}, {}
        for level, y in (("leaf", labels), ("parent", result.parent_labels)):
            shape, scale, report = _fit_weibull(distances[level], y, tail_size)
            result.parameters[level] = dict(shape=shape, scale=scale)
            reports[level] = report
        result.fit_report = dict(schema_version=BANK_SCHEMA, fit_split="known_train", support_count=len(fine),
            image_hash_digest=_hash_digest(hashes), tail_size=tail_size, shape_bounds=[.2, 10.],
            distance="euclidean_on_l2_normalized_features", negative_margin_factor=.5,
            fit_method="two_parameter_zero_location_Weibull_MLE_bisection_not_libMR",
            fallback="shape_1_scale_mean_clamped_tail_or_epsilon_if_no_negative", epsilon=EPS,
            set_cover=False, all_witnesses_retained=True, true_unknown_images_used=False,
            inference_recomputes_statistics=False, per_level=reports)
        return result

    @torch.no_grad()
    def score(self, fine, parent, query_hashes):
        fine = _features(fine, "fine", dimension=self.fine.shape[1])
        parent = _features(parent, "parent", count=len(fine), dimension=self.parent.shape[1])
        hashes = _hashes(query_hashes, len(fine), unique=False)
        if self._hashes.intersection(hashes):
            raise ValueError("Query/support hash overlap: exclude query before Weibull tail fitting")
        result = {}
        for level, q, s, y, count in (("leaf", fine, self.fine, self.labels, len(self.meta["leaf_names"])),
                ("parent", parent, self.parent, self.parent_labels, len(self.meta["parent_names"]))):
            active = torch.bincount(y, minlength=count) > 0
            values = torch.full((len(q), count), ABSENT, dtype=torch.float64)
            shape, logscale = self.parameters[level]["shape"], self.parameters[level]["scale"].log()
            for start in range(0, len(q), 128):
                order = -shape[None, :] * (_distance(q[start:start+128], s).clamp_min(EPS).log() - logscale[None, :])
                for c in active.nonzero(as_tuple=True)[0].tolist():
                    values[start:start+128, c] = order[:, y == c].amax(1)
            if not bool(torch.isfinite(values).all()):
                raise ValueError("Boundary score is not finite")
            result[level + "_scores"], result[level + "_active"] = values, active
        return result

    def state_dict(self):
        state = dict(schema_version=BANK_SCHEMA, fine=self.fine.clone(), parent=self.parent.clone(),
            labels=self.labels.clone(), image_hashes=list(self.image_hashes), meta=copy.deepcopy(self.meta),
            tail_size=self.tail_size, parameters=copy.deepcopy(self.parameters), fit_report=copy.deepcopy(self.fit_report))
        state["state_sha256"] = _state_hash(state)
        return state

    @classmethod
    def from_state_dict(cls, state):
        keys = {"schema_version", "fine", "parent", "labels", "image_hashes", "meta", "tail_size", "parameters", "fit_report", "state_sha256"}
        if not isinstance(state, dict) or set(state) != keys or state["schema_version"] != BANK_SCHEMA:
            raise ValueError("Invalid boundary bank schema")
        if state["state_sha256"] != _state_hash(state):
            raise ValueError("Boundary state digest mismatch")
        result = cls()
        result.meta = _meta(state["meta"])
        for name in ("fine", "parent"):
            if not torch.is_tensor(state[name]) or state[name].dtype != torch.float64:
                raise ValueError("Boundary support must be float64")
        result.fine = _features(state["fine"], "stored fine", normalized=True)
        result.parent = _features(state["parent"], "stored parent", count=len(result.fine), normalized=True)
        result.labels = _ids(state["labels"], "stored labels", len(result.fine), len(result.meta["leaf_names"]))
        result.image_hashes = _hashes(state["image_hashes"], len(result.fine), unique=True)
        result._hashes = set(result.image_hashes)
        result.parent_labels = torch.tensor(result.meta["leaf_to_parent"])[result.labels]
        result.tail_size = state["tail_size"]
        if type(result.tail_size) is not int or result.tail_size < 2 or not len(result.fine):
            raise ValueError("Invalid boundary bank support/tail size")
        if not isinstance(state["parameters"], dict) or set(state["parameters"]) != set(LEVELS):
            raise ValueError("Missing boundary parameters")
        result.parameters = {}
        for level in LEVELS:
            p = state["parameters"][level]
            if (not isinstance(p, dict) or set(p) != {"shape", "scale"}
                    or any(not torch.is_tensor(v) or v.dtype != torch.float64 or v.shape != (len(result.fine),)
                           or not bool(torch.isfinite(v).all()) for v in p.values())
                    or bool(((p["shape"] < .2) | (p["shape"] > 10)).any()) or bool((p["scale"] < EPS).any())):
                raise ValueError("Invalid saved Weibull parameters")
            result.parameters[level] = {k: v.detach().cpu().clone() for k, v in p.items()}
        report = state["fit_report"]
        if (report.get("schema_version") != BANK_SCHEMA or report.get("fit_split") != "known_train"
                or report.get("support_count") != len(result.fine) or report.get("tail_size") != result.tail_size
                or report.get("image_hash_digest") != _hash_digest(result.image_hashes)
                or report.get("true_unknown_images_used") is not False or report.get("inference_recomputes_statistics") is not False
                or report.get("negative_margin_factor") != .5 or report.get("shape_bounds") != [.2, 10.]
                or report.get("all_witnesses_retained") is not True):
            raise ValueError("Invalid boundary TRAIN audit")
        result.fit_report = copy.deepcopy(report)
        return result


def _balanced(y, source, kind):
    result = torch.zeros_like(y)
    multiplier = int(source.max()) + 1
    codes = kind * multiplier + source
    for outcome in (0., 1.):
        selected = y == outcome
        if not bool(selected.any()):
            raise ValueError("Episodes require both binary outcomes")
        groups, inverse, counts = codes[selected].unique(return_inverse=True, return_counts=True)
        result[selected] = .5 / (len(groups) * counts[inverse].float())
    return result


@torch.no_grad()
def build_episodes(fine, parent, labels, image_hashes, meta, template_scores=None,
                   folds=3, shrinkage=.1, seed=1, tail_size=32):
    """Every bank scores all fold queries: retained positives and held negatives.

    Absent class templates, geometry and boundaries are masked before assembly.
    A dropped leaf remains a positive parent query if a sibling is supported.
    """
    if type(folds) is not int or folds < 2 or type(seed) is not int or seed < 0:
        raise ValueError("Invalid episode folds or seed")
    meta = _meta(meta)
    fine = _features(fine, "fine")
    parent = _features(parent, "parent", count=len(fine))
    labels = _ids(labels, "labels", len(fine), len(meta["leaf_names"]))
    hashes = _hashes(image_hashes, len(fine), unique=True)
    c, p = len(meta["leaf_names"]), len(meta["parent_names"])
    if bool((torch.bincount(labels, minlength=c) < 2).any()) or p < 2:
        raise ValueError("Need >=2 images per known leaf and >=2 parents")
    if not isinstance(template_scores, dict) or set(template_scores) != set(LEVELS):
        raise ValueError("Boundary comparisons require the original eight-feature template inputs")
    template = {key: torch.as_tensor(value).detach().cpu().double() for key, value in template_scores.items()}
    if any(template[level].shape != (len(fine), width) or not bool(torch.isfinite(template[level]).all())
           for level, width in (("leaf", c), ("parent", p))):
        raise ValueError("Invalid TRAIN template scores")
    mapping = torch.tensor(meta["leaf_to_parent"], dtype=torch.long)
    parent_labels = mapping[labels]
    generator = torch.Generator().manual_seed(seed)
    assignment = torch.empty(len(fine), dtype=torch.long)
    for leaf in range(c):
        indices = (labels == leaf).nonzero(as_tuple=True)[0]
        indices = indices[torch.randperm(len(indices), generator=generator)]
        assignment[indices] = torch.arange(len(indices)) % min(folds, len(indices))
    distances = {"leaf": _distance(fine, fine), "parent": _distance(parent, parent)}
    fields = ("x", "y", "source_leaf", "kind", "query_index", "candidate", "bank_id", "withheld")
    storage = {level: {key: [] for key in fields} for level in LEVELS}
    reports = []
    for fold in range(folds):
        query = (assignment == fold).nonzero(as_tuple=True)[0]
        if not len(query):
            continue
        definitions = [("full", -1)] + [("drop_leaf", i) for i in range(c)] + [("drop_parent", i) for i in range(p)]
        for kind, identity in definitions:
            support_mask = assignment != fold
            if kind == "drop_leaf": support_mask &= labels != identity
            if kind == "drop_parent": support_mask &= parent_labels != identity
            support = support_mask.nonzero(as_tuple=True)[0]
            if not len(support):
                raise ValueError("An episode has no retained support")
            support_hashes = [hashes[i] for i in support.tolist()]
            query_hashes = [hashes[i] for i in query.tolist()]
            geo = GeometryBank.fit(fine[support], parent[support], labels[support], support_hashes, meta, shrinkage=shrinkage)
            boundary = BoundaryBank._fit_precomputed(fine[support], parent[support], labels[support], support_hashes, meta,
                        tail_size, {level: value[support][:, support] for level, value in distances.items()})
            scores = geo.score(fine[query], parent[query], query_hashes)
            new = boundary.score(fine[query], parent[query], query_hashes)
            bank_id = len(reports)
            for level, target in (("leaf", labels), ("parent", parent_labels)):
                active = scores[level + "_active"]
                candidates = active.nonzero(as_tuple=True)[0]
                x8 = evidence_features(scores, level, {k: v[query] for k, v in template.items()})
                x9 = torch.cat((x8, new[level + "_scores"][:, :, None]), 2)
                width = len(candidates)
                y = (target[query, None] == candidates[None, :]).float()
                absent_target = ~active[target[query]]
                values = dict(x=x9[:, candidates].reshape(-1, 9).float(), y=y.reshape(-1),
                    source_leaf=labels[query, None].expand(-1, width).reshape(-1),
                    kind=torch.full((len(query)*width,), KINDS[kind], dtype=torch.long),
                    query_index=query[:, None].expand(-1, width).reshape(-1),
                    candidate=candidates[None, :].expand(len(query), -1).reshape(-1),
                    bank_id=torch.full((len(query)*width,), bank_id, dtype=torch.long),
                    withheld=absent_target[:, None].expand(-1, width).reshape(-1))
                for key, value in values.items(): storage[level][key].append(value)
            reports.append(dict(bank_id=bank_id, fold=fold, kind=kind, identity=identity,
                query_count=len(query), support_count=len(support), support_indices=support.tolist(), query_indices=query.tolist(),
                support_hash_digest=_hash_digest(support_hashes), query_hash_digest=_hash_digest(query_hashes),
                query_support_overlap=0, active_leaf_ids=scores["leaf_active"].nonzero(as_tuple=True)[0].tolist(),
                active_parent_ids=scores["parent_active"].nonzero(as_tuple=True)[0].tolist(),
                withheld_leaf_ids=([identity] if kind == "drop_leaf" else
                    (mapping == identity).nonzero(as_tuple=True)[0].tolist() if kind == "drop_parent" else []),
                withheld_parent_ids=([identity] if kind == "drop_parent" else []),
                boundary_fit=copy.deepcopy(boundary.fit_report["per_level"])))
    result = {level: {k: torch.cat(v).contiguous() for k, v in values.items()} for level, values in storage.items()}
    for level in LEVELS:
        d = result[level]
        d["weight"] = _balanced(d["y"], d["source_leaf"], d["kind"])
        if not bool(torch.isfinite(d["x"]).all()): raise ValueError("Nonfinite episode features")
    result["report"] = dict(schema_version=EPISODE_SCHEMA, fit_split="known_train", train_count=len(fine),
        image_hash_digest=_hash_digest(hashes), input_feature_sha256={"leaf": _digest(fine), "parent": _digest(parent)},
        label_sha256=_digest(labels), template_sha256=_digest(template), folds=folds, seed=seed,
        covariance_shrinkage=shrinkage, tail_size=tail_size, true_unknown_images_used=False, dev_images_used=False,
        test_images_used=False, encoder_unseen_class_claim=False, candidate_ranking_changed=False,
        query_self_excluded_from="all_geometry_and_Weibull_tail_support_statistics",
        held_out_scope="support_statistics_and_candidate_templates_only",
        ranking_pair_semantics="same_bank_same_candidate_different_query_prefer_withheld_negative",
        episodes=reports, examples={level: dict(rows=len(d["y"]), positive=int(d["y"].sum()),
            negative=int((d["y"] == 0).sum()), withheld_negative=int(d["withheld"].sum()),
            feature_sha256=_digest(d["x"]), target_sha256=_digest(d["y"]),
            data_sha256=_digest(d), by_kind={k:int((d["kind"]==v).sum()) for k,v in KINDS.items()})
            for level,d in result.items()})
    return result


class _BoundaryHead(nn.Module):
    def __init__(self, dimension, hidden):
        super().__init__()
        self.dimension = dimension
        self.layers = nn.Sequential(nn.Linear(dimension, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, value):
        if self.dimension == 8:
            return self.layers(value).squeeze(-1)
        # Separate the zero-initialized column so changing GEMM width does not
        # introduce even a rounding difference from the original 8-D teacher.
        first = self.layers[0]
        hidden = F.linear(value[:, :8].contiguous(), first.weight[:, :8].contiguous(), first.bias)
        hidden = hidden + value[:, 8:9] * first.weight[:, 8][None, :]
        return self.layers[2](self.layers[1](hidden)).squeeze(-1)


def _load_source(state):
    # The historical loader seeds all CUDA generators while constructing its
    # CPU head. Preserve already initialized CUDA state without initializing it.
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        return SharedVerifier.from_state_dict(state)
    finally:
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _head_from_source(source_head, dimension, hidden):
    if dimension not in (8, 9):
        raise ValueError("Boundary heads need eight or nine inputs")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(0)
        result = _BoundaryHead(dimension, hidden)
    values = {k: v.detach().cpu().float().clone() for k, v in source_head.items()}
    if values["layers.0.weight"].shape != (hidden, 8):
        raise ValueError("Boundary warm start requires an eight-dimensional source")
    if dimension == 9:
        values["layers.0.weight"] = torch.cat((values["layers.0.weight"], torch.zeros(hidden, 1)), 1)
    result.load_state_dict(values, strict=True)
    return result


def _validate_episodes(episodes):
    if not isinstance(episodes, dict) or set(episodes) != {"leaf", "parent", "report"}:
        raise ValueError("Invalid boundary episodes")
    report = episodes["report"]
    if (not isinstance(report, dict) or report.get("schema_version") != EPISODE_SCHEMA
            or report.get("fit_split") != "known_train"
            or any(report.get(k) is not False for k in ("true_unknown_images_used", "dev_images_used", "test_images_used"))
            or report.get("encoder_unseen_class_claim") is not False):
        raise ValueError("Boundary training requires audited known TRAIN only")
    train_count = report.get("train_count")
    banks = report.get("episodes", [])
    if type(train_count) is not int or train_count < 1 or not banks:
        raise ValueError("Invalid TRAIN/bank counts")
    required = {"x", "y", "weight", "source_leaf", "kind", "query_index", "candidate", "bank_id", "withheld"}
    result = {}
    for level in LEVELS:
        d = episodes[level]
        if not isinstance(d, dict) or set(d) != required:
            raise ValueError("Invalid candidate example schema")
        if any(not torch.is_tensor(v) or v.device.type != "cpu" for v in d.values()):
            raise ValueError("Candidate examples must be CPU tensors")
        n = len(d["y"])
        if (not n or d["x"].shape != (n, 9) or d["x"].dtype != torch.float32
                or any(d[k].shape != (n,) for k in required - {"x"})
                or any(d[k].dtype != torch.long for k in ("source_leaf", "kind", "query_index", "candidate", "bank_id"))
                or d["withheld"].dtype != torch.bool or d["y"].dtype != torch.float32 or d["weight"].dtype != torch.float32
                or not bool(torch.isfinite(d["x"]).all() & torch.isfinite(d["y"]).all() & torch.isfinite(d["weight"]).all())
                or not bool(((d["y"] == 0) | (d["y"] == 1)).all()) or not bool((d["weight"] > 0).all())
                or not bool((d["y"] == 0).any() & (d["y"] == 1).any())
                or bool((d["query_index"] < 0).any() | (d["query_index"] >= train_count).any())
                or bool((d["candidate"] < 0).any() | (d["source_leaf"] < 0).any())
                or bool((d["bank_id"] < 0).any() | (d["bank_id"] >= len(banks)).any())
                or bool((d["kind"] < 0).any() | (d["kind"] > 2).any())
                or bool((d["withheld"] & (d["y"] != 0)).any())):
            raise ValueError("Invalid candidate example values")
        r = report.get("examples", {}).get(level, {})
        if (r.get("rows") != n or r.get("data_sha256") != _digest(d)
                or r.get("feature_sha256") != _digest(d["x"]) or r.get("target_sha256") != _digest(d["y"])):
            raise ValueError("Candidate examples differ from audited TRAIN digest")
        if not torch.allclose(d["weight"], _balanced(d["y"], d["source_leaf"], d["kind"]), atol=0, rtol=1e-6):
            raise ValueError("Candidate weights differ from outcome/kind/source balancing")
        result[level] = d
    # Check bank membership with small lookup matrices instead of scanning all
    # million examples once per bank.
    for level in LEVELS:
        d = result[level]
        maxcandidate = int(d["candidate"].max()) + 1
        allowed = torch.zeros(len(banks), maxcandidate, dtype=torch.bool)
        queries = torch.zeros(len(banks), train_count, dtype=torch.bool)
        bank_kinds = torch.empty(len(banks), dtype=torch.long)
        for i, bank in enumerate(banks):
            if bank.get("bank_id") != i or bank.get("query_support_overlap") != 0:
                raise ValueError("Invalid bank identity/exclusion report")
            support, query = bank["support_indices"], bank["query_indices"]
            if (not support or not query or set(support) & set(query)
                    or min(support + query) < 0 or max(support + query) >= train_count):
                raise ValueError("Invalid reported support/query split")
            active = bank["active_" + level + "_ids"]
            if not active or min(active) < 0 or max(active) >= maxcandidate:
                raise ValueError("Invalid active candidate report")
            allowed[i, active] = True; queries[i, query] = True
            bank_kinds[i] = KINDS[bank["kind"]]
        if (not bool(allowed[d["bank_id"], d["candidate"]].all())
                or not bool(queries[d["bank_id"], d["query_index"]].all())
                or not torch.equal(bank_kinds[d["bank_id"]], d["kind"])):
            raise ValueError("Example candidates/queries do not belong to their audited bank")
    return result


def _ranking_groups(data):
    """Group true positives and preferred negatives in exactly one support world."""
    candidate_count = int(data["candidate"].max()) + 1
    code = data["bank_id"] * candidate_count + data["candidate"]
    order = _stable_argsort(code)
    _, counts = code[order].unique_consecutive(return_counts=True)
    groups = []
    for indices in order.split(counts.tolist()):
        positive = indices[data["y"][indices] == 1]
        negative = indices[data["y"][indices] == 0]
        if not len(positive) or not len(negative):
            continue
        if _has_common_query(data["query_index"][positive], data["query_index"][negative]):
            raise ValueError("A query has conflicting targets for the same bank and candidate")
        preferred = negative[data["withheld"][negative]]
        has_preferred = bool(len(preferred))
        if has_preferred: negative = preferred
        groups.append(dict(bank_id=int(data["bank_id"][indices[0]]), candidate=int(data["candidate"][indices[0]]),
                           positive=positive, negative=negative, preferred_withheld=has_preferred))
    if not groups:
        raise ValueError("No valid same-bank cross-query ranking group")
    return groups


@torch.no_grad()
def _tail_pairs(head, x, groups, generator, pair_budget=256, tail_fraction=.2):
    """Sampled tails, not an exact global partial-AUC optimization.

    At most 32 candidate/bank groups and 32 positive/negative examples per group
    are examined. Mining uses a separate RNG and cannot change BCE minibatches.
    """
    if type(pair_budget) is not int or pair_budget < 1 or not 0 < tail_fraction <= 1:
        raise ValueError("Invalid ranking sampler budget")
    selected = torch.randperm(len(groups), generator=generator)[:min(32, pair_budget, len(groups))]
    pools, sizes = [], []
    for index in selected.tolist():
        g = groups[index]
        pos = g["positive"][torch.randperm(len(g["positive"]), generator=generator)[:32]]
        neg = g["negative"][torch.randperm(len(g["negative"]), generator=generator)[:32]]
        pools.append((pos, neg)); sizes.extend((len(pos), len(neg)))
    indices = torch.cat([a for pair in pools for a in pair])
    values = head(x[indices]).split(sizes)
    pairs = []
    for i, (positive, negative) in enumerate(pools):
        p = positive[_stable_argsort(values[2*i])[:max(1, math.ceil(len(positive)*tail_fraction))]]
        n = negative[_stable_argsort(values[2*i+1], descending=True)[:max(1, math.ceil(len(negative)*tail_fraction))]]
        count = pair_budget // len(pools) + int(i < pair_budget % len(pools))
        a = p[torch.randint(len(p), (count,), generator=generator)]
        b = n[torch.randint(len(n), (count,), generator=generator)]
        pairs.append(torch.stack((a, b), 1))
    return torch.cat(pairs)


@torch.no_grad()
def _logits(head, x):
    return torch.cat([head(part) for part in x.split(4096)])


def _options(mode, seed, steps, batch_size, lr, l2sp_weight, ranking_weight, ranking_margin):
    if mode not in MODES or type(seed) is not int or seed < 0:
        raise ValueError("Invalid boundary mode or seed")
    if any(type(v) is not int or v < 1 for v in (steps, batch_size)):
        raise ValueError("Boundary training requires positive step/batch budgets")
    for key, value in (("lr", lr), ("l2sp", l2sp_weight), ("rank", ranking_weight), ("margin", ranking_margin)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("Invalid boundary optimization " + key)
    if not 0 < lr <= .001:
        raise ValueError("Boundary learning rate must be in (0,.001]")


class BoundaryVerifier:
    @classmethod
    def fit(cls, source_verifier_state, episodes, mode="bce8", seed=1, steps=300, batch_size=1024,
            lr=.001, l2sp_weight=.01, ranking_weight=.2, ranking_margin=.2):
        _options(mode, seed, steps, batch_size, lr, l2sp_weight, ranking_weight, ranking_margin)
        source = _load_source(source_verifier_state)
        if source.dimension != 8:
            raise ValueError("The boundary study must start from the eight-feature D05 verifier")
        data = _validate_episodes(episodes)
        if source.fit_report["episode_report"]["image_hash_digest"] != episodes["report"]["image_hash_digest"]:
            raise ValueError("Boundary TRAIN image identities differ from source D05")
        result = cls()
        result.source_state = copy.deepcopy(source_verifier_state)
        dimension = int(mode[-1])
        result.dimensions, result.hidden = {level:dimension for level in LEVELS}, source.hidden
        result.heads, result.normalization = {}, {}
        histories, deltas, initial_hashes, batch_hashes, pair_hashes, changes, group_reports = {}, {}, {}, {}, {}, {}, {}
        teacher_difference = 0.
        for offset, level in enumerate(LEVELS):
            d = data[level]
            raw, y, weights = d["x"][:, :dimension], d["y"], d["weight"] / d["weight"].sum()
            norm = copy.deepcopy(source.normalization[level])
            if dimension == 9:
                mean = (weights * raw[:, 8]).sum()
                std = (weights * (raw[:, 8] - mean).square()).sum().sqrt().clamp_min(1e-4)
                norm = {"mean":torch.cat((norm["mean"], mean[None])), "scale":torch.cat((norm["scale"], std[None]))}
            result.normalization[level] = norm
            x = ((raw - norm["mean"]) / norm["scale"]).contiguous()
            if not bool(torch.isfinite(x).all()): raise ValueError("Invalid normalized TRAIN evidence")
            head = _head_from_source(source_verifier_state["heads"][level], dimension, source.hidden)
            initial = {k:v.detach().clone() for k,v in head.state_dict().items()}
            initial_hashes[level] = _digest(initial)
            initial_logits = _logits(head, x)
            teacher_x = ((d["x"][:, :8] - source.normalization[level]["mean"]) / source.normalization[level]["scale"]).contiguous()
            teacher_logits = _logits(source.heads[level], teacher_x)
            teacher_difference = max(teacher_difference, float((initial_logits-teacher_logits).abs().max()))
            if not torch.equal(initial_logits, teacher_logits):
                raise ValueError("Warm-start function is not exactly the original eight-feature D05")
            groups = _ranking_groups(d)
            group_reports[level] = dict(groups=len(groups), preferred_withheld_groups=sum(g["preferred_withheld"] for g in groups),
                groups_sha256=_digest(groups), max_groups_per_step=32, max_pool_per_outcome_per_group=32,
                tail_fraction=.2, pair_budget=256, mining="sampled_tail_not_global_partial_AUC")
            optimizer = torch.optim.Adam(head.parameters(), lr=lr)
            batch_rng = torch.Generator().manual_seed(seed + 101 + offset)
            rank_rng = torch.Generator().manual_seed(seed + 201 + offset)
            bh, ph = hashlib.sha256(), hashlib.sha256()
            order, position = torch.empty(0, dtype=torch.long), 0
            histories[level] = []
            for step in range(steps):
                if position >= len(order):
                    order, position = torch.randperm(len(x), generator=batch_rng), 0
                selected = order[position:position+batch_size]; position += len(selected)
                bh.update(selected.numpy().tobytes())
                optimizer.zero_grad(set_to_none=True)
                logits = head(x[selected])
                bce = (F.binary_cross_entropy_with_logits(logits, y[selected], reduction="none") * weights[selected]).sum() * len(x) / len(selected)
                rank = logits.sum() * 0.
                if mode.startswith("rank"):
                    pairs = _tail_pairs(head, x, groups, rank_rng)
                    ph.update(pairs.numpy().tobytes())
                    # Pair identities are sampled without autograd; both logits
                    # below retain their gradient to every trainable head weight.
                    rank = F.softplus(ranking_margin + head(x[pairs[:,1]]) - head(x[pairs[:,0]])).mean()
                l2sp = sum((value - initial[name]).square().sum() for name,value in head.named_parameters())
                total = bce + l2sp_weight*l2sp + (ranking_weight*rank if mode.startswith("rank") else 0.)
                if not bool(torch.isfinite(total)): raise ValueError("Nonfinite boundary training objective")
                total.backward()
                if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in head.parameters()):
                    raise ValueError("Missing/nonfinite boundary gradient")
                gradnorm = nn.utils.clip_grad_norm_(head.parameters(), 5.)
                if not bool(torch.isfinite(gradnorm)): raise ValueError("Nonfinite gradient norm")
                optimizer.step()
                if any(not bool(torch.isfinite(p).all()) for p in head.parameters()):
                    raise ValueError("Nonfinite boundary head update")
                histories[level].append(dict(step=step+1,bce=float(bce.detach()),rank=float(rank.detach()),
                    l2sp=float(l2sp.detach()),total=float(total.detach()),gradnorm=float(gradnorm),
                    bce_batch_count=len(selected)))
            final_logits = _logits(head, x)
            difference = final_logits-initial_logits
            checkpairs = _tail_pairs(_head_from_source(source_verifier_state["heads"][level], dimension, source.hidden),
                    x, groups, torch.Generator().manual_seed(seed+901+offset), pair_budget=2048)
            a,b = checkpairs[:,0],checkpairs[:,1]
            changes[level] = dict(mean=float(difference.mean()),std=float(difference.std(unbiased=False)),
                positive_mean=float(difference[y==1].mean()),negative_mean=float(difference[y==0].mean()),
                centered_change_l2=float((difference-difference.mean()).double().norm()),
                diagnostic_pairs=len(checkpairs), diagnostic_pair_sha256=_digest(checkpairs),
                pair_order_changed=int(((initial_logits[a]>initial_logits[b]) != (final_logits[a]>final_logits[b])).sum()),
                pair_correct_before=int((initial_logits[a]>initial_logits[b]).sum()),
                pair_correct_after=int((final_logits[a]>final_logits[b]).sum()),
                initial_bce=float((F.binary_cross_entropy_with_logits(initial_logits,y,reduction="none")*weights).sum()),
                final_bce=float((F.binary_cross_entropy_with_logits(final_logits,y,reduction="none")*weights).sum()))
            deltas[level] = math.sqrt(sum(float((value-initial[name]).double().square().sum()) for name,value in head.state_dict().items()))
            batch_hashes[level], pair_hashes[level] = bh.hexdigest(), ph.hexdigest()
            result.heads[level] = head.eval().requires_grad_(False)
        result.fit_report = dict(schema_version=VERIFIER_SCHEMA,mode=mode,seed=seed,steps=steps,batch_size=batch_size,
            lr=float(lr),l2sp_weight=float(l2sp_weight),ranking_weight=float(ranking_weight),ranking_margin=float(ranking_margin),
            optimizer="Adam",optimizer_steps=2*steps,optimizer_steps_by_level={level:steps for level in LEVELS},
            fit_split="known_train",true_unknown_images_used=False,dev_images_used=False,test_images_used=False,
            encoder_unseen_class_claim=False,frozen_encoder_updated=False,candidate_ranking_changed=False,
            checkpoint_selection="last_fixed_budget_step_no_DEV",warm_start_no_optimizer_resume=True,
            source_state_sha256=_digest(source_verifier_state),source_head_sha256=_digest(source_verifier_state["heads"]),
            source_normalization_sha256=_digest(source_verifier_state["normalization"]),
            normalization_first_eight_unchanged=True,new_column_normalization="TRAIN_episode_weighted_mean_std" if dimension==9 else "none",
            initial_teacher_max_abs_difference=teacher_difference,initial_parameter_sha256=initial_hashes,
            base_batch_sha256=batch_hashes,sampled_rank_pairs_sha256=pair_hashes,parameter_delta_l2=deltas,
            gradients_finite=True,history=histories,score_change=changes,ranking_groups=group_reports,
            ranking_pair_semantics="same_bank_same_candidate_different_query_prefer_withheld_negative",
            ranking_sampler="at_most_32_groups_32_examples_per_outcome_sampled_bottom_top_20_percent",
            episode_report=copy.deepcopy(episodes["report"]))
        return result

    @torch.no_grad()
    def score(self, geometry_scores, template_scores=None, boundary_scores=None):
        result = {}
        for level in LEVELS:
            x = evidence_features(geometry_scores,level,template_scores).float()
            if x.shape[-1] != 8:
                raise ValueError("Boundary inference requires the original eight source evidence features")
            if self.dimensions[level] == 9:
                if not isinstance(boundary_scores,dict) or level+"_scores" not in boundary_scores:
                    raise ValueError("Nine-feature boundary head requires frozen boundary scores")
                value = torch.as_tensor(boundary_scores[level+"_scores"]).detach().cpu().float()
                if value.shape != x.shape[:2] or not bool(torch.isfinite(value).all()):
                    raise ValueError("Boundary inference feature shape/finite mismatch")
                if level+"_active" in boundary_scores and not torch.equal(boundary_scores[level+"_active"],geometry_scores[level+"_active"]):
                    raise ValueError("Geometry/boundary candidate masks differ")
                x = torch.cat((x,value[:,:,None]),2)
            shape = x.shape[:2]
            x = x.reshape(-1,self.dimensions[level])
            x = ((x-self.normalization[level]["mean"])/self.normalization[level]["scale"]).contiguous()
            values = _logits(self.heads[level],x).reshape(shape).double()
            values[:,~geometry_scores[level+"_active"]] = ABSENT
            if not bool(torch.isfinite(values).all()): raise ValueError("Nonfinite boundary verifier inference")
            result[level+"_scores"] = values
        return result

    def state_dict(self):
        state = dict(schema_version=VERIFIER_SCHEMA,source_state=copy.deepcopy(self.source_state),
            dimensions=dict(self.dimensions),hidden=self.hidden,normalization=copy.deepcopy(self.normalization),
            heads={level:{k:v.detach().cpu().clone() for k,v in head.state_dict().items()} for level,head in self.heads.items()},
            fit_report=copy.deepcopy(self.fit_report))
        state["state_sha256"] = _state_hash(state)
        return state

    @classmethod
    def from_state_dict(cls,state):
        keys={"schema_version","source_state","dimensions","hidden","normalization","heads","fit_report","state_sha256"}
        if not isinstance(state,dict) or set(state)!=keys or state["schema_version"]!=VERIFIER_SCHEMA:
            raise ValueError("Invalid boundary verifier schema")
        if state["state_sha256"] != _state_hash(state): raise ValueError("Boundary verifier digest mismatch")
        source = _load_source(state["source_state"])
        report = state["fit_report"]
        if (source.dimension!=8 or state["hidden"]!=source.hidden or not isinstance(report,dict)
                or report.get("schema_version")!=VERIFIER_SCHEMA or report.get("source_state_sha256")!=_digest(state["source_state"])
                or report.get("source_head_sha256")!=_digest(state["source_state"]["heads"])
                or report.get("source_normalization_sha256")!=_digest(state["source_state"]["normalization"])
                or report.get("fit_split")!="known_train"
                or any(report.get(k) is not False for k in ("true_unknown_images_used","dev_images_used","test_images_used","frozen_encoder_updated","candidate_ranking_changed"))
                or report.get("normalization_first_eight_unchanged") is not True
                or report.get("initial_teacher_max_abs_difference")!=0.):
            raise ValueError("Invalid boundary source/provenance audit")
        guard = report.get("mode")=="leaf_guard"
        mode = report.get("inherited_mode") if guard else report.get("mode")
        _options(mode,report.get("seed"),report.get("steps"),report.get("batch_size"),report.get("lr"),
                 report.get("l2sp_weight"),report.get("ranking_weight"),report.get("ranking_margin"))
        expected = {level:int(mode[-1]) for level in LEVELS}
        if guard: expected["parent"] = 8
        if state["dimensions"] != expected or set(state["heads"])!=set(LEVELS) or set(state["normalization"])!=set(LEVELS):
            raise ValueError("Invalid boundary head dimensions")
        if guard:
            if (report.get("optimizer_steps")!=0 or report.get("optimizer_steps_by_level")!={"leaf":0,"parent":0}
                    or report.get("composition_source_state_sha256")!=_digest(state["source_state"])
                    or report.get("parent_restored_exactly") is not True
                    or _digest(state["heads"]["parent"])!=_digest(state["source_state"]["heads"]["parent"])):
                raise ValueError("Invalid zero-step leaf guard composition")
        else:
            if (report.get("optimizer_steps")!=2*report["steps"]
                    or report.get("optimizer_steps_by_level")!={level:report["steps"] for level in LEVELS}
                    or report.get("gradients_finite") is not True):
                raise ValueError("Invalid fixed training budget/gradient audit")
            for level in LEVELS:
                history=report.get("history",{}).get(level,[])
                if len(history)!=report["steps"] or any(r.get("step")!=i+1 or
                    any(not isinstance(r.get(k),(int,float)) or not math.isfinite(r[k]) for k in ("bce","rank","l2sp","total","gradnorm"))
                    for i,r in enumerate(history)):
                    raise ValueError("Invalid finite training history")
        ep=report.get("episode_report",{})
        if ep.get("schema_version")!=EPISODE_SCHEMA or ep.get("fit_split")!="known_train" or any(ep.get(k) is not False for k in ("true_unknown_images_used","dev_images_used","test_images_used")):
            raise ValueError("Invalid boundary episode provenance")
        if ep.get("image_hash_digest")!=source.fit_report["episode_report"]["image_hash_digest"]:
            raise ValueError("Boundary training image identities differ from source")
        result=cls(); result.source_state=copy.deepcopy(state["source_state"])
        result.dimensions,result.hidden=dict(expected),source.hidden
        result.normalization,result.heads={},{}
        for level in LEVELS:
            dimension=expected[level]
            norm=state["normalization"][level]
            if (not isinstance(norm,dict) or set(norm)!={"mean","scale"}
                    or any(not torch.is_tensor(v) or v.dtype!=torch.float32 or v.shape!=(dimension,)
                           or not bool(torch.isfinite(v).all()) for v in norm.values())
                    or not bool((norm["scale"]>0).all())
                    or any(not torch.equal(norm[k][:8],source.normalization[level][k]) for k in norm)):
                raise ValueError("Boundary source normalization was changed")
            result.normalization[level]={k:v.detach().cpu().clone() for k,v in norm.items()}
            head=_head_from_source(state["source_state"]["heads"][level],dimension,source.hidden)
            values=state["heads"][level]
            if not isinstance(values,dict) or set(values)!=set(head.state_dict()) or any(not torch.is_tensor(v) or v.dtype!=torch.float32 or not bool(torch.isfinite(v).all()) for v in values.values()):
                raise ValueError("Invalid boundary head tensor values")
            try: head.load_state_dict(values,strict=True)
            except RuntimeError as error: raise ValueError("Invalid boundary head tensor shapes") from error
            if not guard:
                initial=_head_from_source(state["source_state"]["heads"][level],dimension,source.hidden).state_dict()
                if report.get("initial_parameter_sha256",{}).get(level)!=_digest(initial):
                    raise ValueError("Invalid warm-start head binding")
                delta=math.sqrt(sum(float((values[k]-v).double().square().sum()) for k,v in initial.items()))
                if not math.isclose(delta,report.get("parameter_delta_l2",{}).get(level,-1),rel_tol=1e-7,abs_tol=1e-10):
                    raise ValueError("Boundary parameter delta audit mismatch")
            result.heads[level]=head.eval().requires_grad_(False)
        result.fit_report=copy.deepcopy(report)
        return result


def make_leaf_guard(state,source_verifier_state):
    """Compose a trained boundary leaf with the exact original D05 parent."""
    model=BoundaryVerifier.from_state_dict(state)
    if model.fit_report["mode"]!="rank9": raise ValueError("Leaf guard requires the rank9 source arm")
    source=_load_source(source_verifier_state)
    if _digest(source_verifier_state)!=model.fit_report["source_state_sha256"]:
        raise ValueError("Leaf guard source D05 differs from the trained model source")
    model.heads["parent"]=_head_from_source(source_verifier_state["heads"]["parent"],8,source.hidden).eval().requires_grad_(False)
    model.normalization["parent"]=copy.deepcopy(source.normalization["parent"])
    model.dimensions["parent"]=8
    inherited=copy.deepcopy(model.fit_report)
    model.fit_report.update(mode="leaf_guard",inherited_mode="rank9",optimizer_steps=0,
        optimizer_steps_by_level={"leaf":0,"parent":0},composition_source_state_sha256=_digest(source_verifier_state),
        composition_trained_state_sha256=state["state_sha256"],parent_restored_exactly=True,
        inherited_optimizer_steps=inherited["optimizer_steps"],inherited_training_report=inherited,
        history={level:[] for level in LEVELS},
        parameter_delta_l2={"leaf":inherited["parameter_delta_l2"]["leaf"],"parent":0.},
        parameter_delta_semantics="final_composed_head_vs_original_D05_with_zero_ninth_column_if_present",
        initial_parameter_sha256={"leaf":inherited["initial_parameter_sha256"]["leaf"],
                                  "parent":_digest(source_verifier_state["heads"]["parent"])},
        checkpoint_selection="zero_optimizer_step_leaf_parent_composition")
    result=model.state_dict()
    BoundaryVerifier.from_state_dict(result)
    return result,copy.deepcopy(model.fit_report)
