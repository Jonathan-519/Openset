"""Known-TRAIN support-intervention examples and the original D05 BCE heads.

Only the BCE execution path is retained. Candidate choice remains reference
owned; no ranking loss, prompt training, or projection experiment is exposed.
Historical BCE state dictionaries remain loadable without refitting.
"""
import copy
import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F

from .d05_geometry import GeometryBank, FEATURE_NAMES, evidence_features
from .tensor_utils import _features, _hashes, _ids, _meta


SCHEMA_VERSION = "discovery_verifier_v1"
EPISODE_SCHEMA = "discovery_episodes_v1"
KINDS = {"full": 0, "drop_leaf": 1, "drop_parent": 2}


def _digest(hashes):
    return hashlib.sha256("\n".join(sorted(hashes)).encode("utf-8")).hexdigest()


def _balanced_weights(y, source, kinds):
    # Each binary outcome receives half the total mass; within it, episode kind
    # and source species groups contribute equally despite TRAIN class imbalance.
    weights = torch.zeros_like(y)
    for outcome in (0., 1.):
        selected = y == outcome
        if not bool(selected.any()):
            raise ValueError("Verifier episodes need both positive and negative targets")
        groups = torch.stack((kinds[selected], source[selected]), 1).unique(dim=0)
        for kind, leaf in groups.tolist():
            mask = selected & (kinds == kind) & (source == leaf)
            weights[mask] = .5 / (len(groups) * int(mask.sum()))
    return weights


@torch.no_grad()
def build_episodes(fine, parent, labels, image_hashes, meta, template_scores=None,
                   folds=3, shrinkage=.1, seed=1):
    """Return deterministic full/drop-leaf/drop-parent candidate examples.

    Full episodes partition queries into stratified folds and remove every query
    from all support statistics. Drop episodes remove the entire query species or
    parent. Single-child parents contribute no drop-leaf parent-positive example.
    All absent geometry and text candidates are excluded before feature assembly.
    """
    if type(folds) is not int or folds < 2 or type(seed) is not int or seed < 0:
        raise ValueError("folds must be >= 2 and seed a nonnegative integer")
    meta = _meta(meta)
    fine = _features(fine, "fine")
    parent = _features(parent, "parent", count=len(fine))
    labels = _ids(labels, "labels", len(fine), len(meta["leaf_names"]))
    hashes = _hashes(image_hashes, len(fine), unique=True)
    counts = torch.bincount(labels, minlength=len(meta["leaf_names"]))
    if bool((counts < 2).any()):
        raise ValueError("Every known species needs at least two TRAIN images for disjoint support episodes")
    mapping = torch.tensor(meta["leaf_to_parent"], dtype=torch.long)
    parent_labels = mapping[labels]
    if len(meta["parent_names"]) < 2:
        raise ValueError("Parent holdout needs at least two TRAIN parents")
    if template_scores is not None:
        if not isinstance(template_scores, dict) or set(template_scores) != {"leaf", "parent"}:
            raise ValueError("template_scores keys must be leaf and parent")
        template_scores = {key: torch.as_tensor(value).detach().cpu().double() for key, value in template_scores.items()}
        for level, width in (("leaf", len(meta["leaf_names"])), ("parent", len(meta["parent_names"]))):
            if (template_scores[level].shape != (len(fine), width)
                    or not bool(torch.isfinite(template_scores[level]).all())):
                raise ValueError("TRAIN template scores shape/finite validation failed")
    generator = torch.Generator().manual_seed(seed)
    assignment = torch.empty(len(fine), dtype=torch.long)
    for leaf in range(len(meta["leaf_names"])):
        indices = (labels == leaf).nonzero(as_tuple=True)[0]
        indices = indices[torch.randperm(len(indices), generator=generator)]
        assignment[indices] = torch.arange(len(indices)) % min(folds, len(indices))
    storage = {level: {key: [] for key in ("x", "y", "source_leaf", "kind", "query_index", "candidate")}
               for level in ("leaf", "parent")}
    reports = []

    def add_episode(support_mask, query_mask, kind, identity):
        support_indices = support_mask.nonzero(as_tuple=True)[0]
        query_indices = query_mask.nonzero(as_tuple=True)[0]
        if not len(query_indices):
            return
        if not len(support_indices) or bool((support_mask & query_mask).any()):
            raise ValueError("Episode support must be nonempty and disjoint from queries")
        support_hashes = [hashes[i] for i in support_indices.tolist()]
        query_hashes = [hashes[i] for i in query_indices.tolist()]
        bank = GeometryBank.fit(fine[support_indices], parent[support_indices], labels[support_indices],
                                support_hashes, meta, shrinkage=shrinkage)
        scores = bank.score(fine[query_indices], parent[query_indices], query_hashes)
        template = None if template_scores is None else {k: v[query_indices] for k, v in template_scores.items()}
        detail = dict(kind=kind, identity=int(identity), query_count=len(query_indices),
            support_count=len(support_indices), support_hash_digest=_digest(support_hashes),
            query_hash_digest=_digest(query_hashes), query_support_overlap=0,
            active_leaf_ids=scores["leaf_active"].nonzero(as_tuple=True)[0].tolist(),
            active_parent_ids=scores["parent_active"].nonzero(as_tuple=True)[0].tolist(),
            withheld_leaf_ids=sorted(set(labels[query_indices].tolist())) if kind == "drop_leaf" else [],
            withheld_parent_ids=sorted(set(parent_labels[query_indices].tolist())) if kind == "drop_parent" else [],
            parent_near_examples_skipped=0)
        for level, target in (("leaf", labels), ("parent", parent_labels)):
            active = scores[level + "_active"]
            candidates = active.nonzero(as_tuple=True)[0]
            features = evidence_features(scores, level, template)
            if kind == "drop_leaf" and level == "parent" and not bool(active[target[query_indices][0]]):
                detail["parent_near_examples_skipped"] = len(query_indices)
                continue
            for position, query_index in enumerate(query_indices.tolist()):
                truths = (candidates == target[query_index]).float()
                width = len(candidates)
                storage[level]["x"].append(features[position, candidates].float())
                storage[level]["y"].append(truths)
                storage[level]["source_leaf"].append(torch.full((width,), int(labels[query_index]), dtype=torch.long))
                storage[level]["kind"].append(torch.full((width,), KINDS[kind], dtype=torch.long))
                storage[level]["query_index"].append(torch.full((width,), query_index, dtype=torch.long))
                storage[level]["candidate"].append(candidates.clone())
                if kind == "full":
                    positive = truths.nonzero(as_tuple=True)[0]
                    if len(positive) != 1:
                        raise ValueError("Full support episode lost its query class")
        reports.append(detail)

    for fold in range(folds):
        add_episode(assignment != fold, assignment == fold, "full", fold)
    for leaf in range(len(meta["leaf_names"])):
        add_episode(labels != leaf, labels == leaf, "drop_leaf", leaf)
    for parent_id in range(len(meta["parent_names"])):
        add_episode(parent_labels != parent_id, parent_labels == parent_id, "drop_parent", parent_id)
    result = {}
    for level, parts in storage.items():
        result[level] = {key: torch.cat(value) for key, value in parts.items()}
        data = result[level]
        data["weight"] = _balanced_weights(data["y"], data["source_leaf"], data["kind"])
        if not bool(torch.isfinite(data["x"]).all()):
            raise ValueError("Episode features must be finite")
    report = dict(schema_version=EPISODE_SCHEMA, fit_split="known_train", train_count=len(fine),
        image_hash_digest=_digest(hashes), seed=seed, folds=folds, covariance_shrinkage=shrinkage,
        true_unknown_images_used=False, dev_images_used=False, test_images_used=False,
        encoder_unseen_class_claim=False, held_out_scope="support_statistics_and_candidate_templates_only",
        query_self_excluded_from="means_covariances_neighbours_candidates", template_features=template_scores is not None,
        feature_names=list(FEATURE_NAMES) + (["template", "template_margin"] if template_scores is not None else []),
        single_child_parent_ids=[i for i in range(len(meta["parent_names"])) if int((mapping == i).sum()) == 1],
        single_child_parent_near_examples_skipped=sum(r["parent_near_examples_skipped"] for r in reports),
        episodes=reports,
        examples={level: dict(rows=len(data["y"]), positive=int(data["y"].sum()),
            negative=int((data["y"] == 0).sum()),
            feature_sha256=hashlib.sha256(data["x"].contiguous().numpy().tobytes()).hexdigest(),
            target_sha256=hashlib.sha256(data["y"].contiguous().numpy().tobytes()).hexdigest(),
            by_kind={kind: int((data["kind"] == code).sum()) for kind, code in KINDS.items()}) for level, data in result.items()})
    result["report"] = report
    return result


class _Head(nn.Module):
    def __init__(self, dimension, hidden):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(dimension, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, value):
        return self.layers(value).squeeze(-1)


def _validate_examples(data):
    if not isinstance(data, dict) or not {"x", "y", "weight"}.issubset(data):
        raise ValueError("Verifier examples are incomplete")
    x = torch.as_tensor(data["x"]).detach().cpu().float()
    y = torch.as_tensor(data["y"]).detach().cpu().float()
    weight = torch.as_tensor(data["weight"]).detach().cpu().float()
    if x.ndim != 2 or not len(x) or x.shape[1] not in (6, 8) or y.shape != (len(x),) or weight.shape != y.shape:
        raise ValueError("Invalid verifier example shapes")
    if (not bool(torch.isfinite(x).all() & torch.isfinite(y).all() & torch.isfinite(weight).all())
            or not bool(((y == 0) | (y == 1)).all()) or not bool((weight > 0).all())
            or not bool((y == 0).any() & (y == 1).any())):
        raise ValueError("Examples need finite features, positive weights and both binary outcomes")
    return x, y, weight / weight.sum()


class SharedVerifier:
    """Two small shared candidate heads. Candidate ranking remains caller-owned."""

    @classmethod
    def fit(cls, episodes, seed=1, epochs=100, batch_size=1024, lr=.001, hidden=32):
        if (not isinstance(episodes, dict) or set(episodes) != {"leaf", "parent", "report"}
                or episodes["report"].get("schema_version") != EPISODE_SCHEMA
                or episodes["report"].get("fit_split") != "known_train"
                or any(episodes["report"].get(k) is not False for k in ("true_unknown_images_used", "dev_images_used", "test_images_used"))):
            raise ValueError("Verifier fitting accepts only audited known TRAIN episodes")
        if any(type(v) is not int or v < 1 for v in (epochs, batch_size, hidden)) or type(seed) is not int or seed < 0:
            raise ValueError("Invalid positive training budget or nonnegative seed")
        if (not isinstance(lr, (float, int)) or isinstance(lr, bool)
                or not math.isfinite(lr) or not 0 < lr <= .1):
            raise ValueError("Invalid learning rate")
        data = {level: _validate_examples(episodes[level]) for level in ("leaf", "parent")}
        dimensions = {value[0].shape[1] for value in data.values()}
        if len(dimensions) != 1:
            raise ValueError("Leaf/parent evidence dimensions differ")
        result = cls()
        result.dimension, result.hidden = dimensions.pop(), hidden
        result.heads, result.normalization = {}, {}
        history, deltas, initial_digests, steps = {}, {}, {}, {}
        for offset, level in enumerate(("leaf", "parent")):
            x, y, weights = data[level]
            mean = (weights[:, None] * x).sum(0)
            scale = (weights[:, None] * (x - mean).square()).sum(0).sqrt().clamp_min(1e-4)
            x = (x - mean) / scale
            result.normalization[level] = {"mean": mean, "scale": scale}
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed + offset)
                head = _Head(result.dimension, hidden)
            initial = {k: v.detach().clone() for k, v in head.state_dict().items()}
            initial_digests[level] = hashlib.sha256(b"".join(v.numpy().tobytes() for v in initial.values())).hexdigest()
            optimizer = torch.optim.Adam(head.parameters(), lr=lr)
            batch_rng = torch.Generator().manual_seed(seed + 101 + offset)
            history[level], steps[level] = [], 0
            for epoch in range(epochs):
                order = torch.randperm(len(x), generator=batch_rng)
                bce_sum, batches = 0., 0
                for start in range(0, len(x), batch_size):
                    selected = order[start:start + batch_size]
                    optimizer.zero_grad()
                    logits = head(x[selected])
                    # Global normalization keeps the intended weighted objective;
                    # the scale len(x)/batch_count is its minibatch estimator.
                    bce = (F.binary_cross_entropy_with_logits(logits, y[selected], reduction="none")
                           * weights[selected]).sum() * len(x) / len(selected)
                    # Keep the exact D05 BCE arithmetic, including its scalar add.
                    value = bce + 0.
                    if not bool(torch.isfinite(value)):
                        raise ValueError("Nonfinite verifier training loss")
                    value.backward()
                    nn.utils.clip_grad_norm_(head.parameters(), 5.)
                    optimizer.step()
                    steps[level] += 1
                    bce_sum += float(bce.detach())
                    batches += 1
                history[level].append(dict(epoch=epoch + 1, bce=bce_sum / batches,
                    optimizer_steps=steps[level]))
            deltas[level] = math.sqrt(sum(float((head.state_dict()[k] - value).double().square().sum()) for k, value in initial.items()))
            head.eval().requires_grad_(False)
            result.heads[level] = head
        result.fit_report = dict(schema_version=SCHEMA_VERSION, fit_split="known_train", loss="bce",
            epochs=epochs, seed=seed, batch_size=batch_size, lr=float(lr),
            hidden=hidden, evidence_dimension=result.dimension, optimizer="Adam", optimizer_steps=sum(steps.values()),
            optimizer_steps_by_level=steps, parameter_delta_l2=deltas, initial_parameter_sha256=initial_digests,
            checkpoint_selection="last_fixed_budget_epoch_no_DEV", history=history,
            episode_report=copy.deepcopy(episodes["report"]), true_unknown_images_used=False,
            frozen_encoder_updated=False, candidate_ranking_changed=False,
            normalization_fit="TRAIN_episode_weighted_mean_std")
        return result

    @torch.no_grad()
    def score(self, geometry_scores, template_scores=None):
        output = {}
        for level in ("leaf", "parent"):
            x = evidence_features(geometry_scores, level, template_scores).float()
            if x.shape[2] != self.dimension:
                raise ValueError("Verifier template/evidence dimension differs from fitted state")
            shape = x.shape[:2]
            x = x.reshape(-1, self.dimension)
            x = (x - self.normalization[level]["mean"]) / self.normalization[level]["scale"]
            values = torch.cat([self.heads[level](x[start:start + 4096]) for start in range(0, len(x), 4096)])
            values = values.reshape(shape).double()
            values[:, ~geometry_scores[level + "_active"]] = -1e6
            if not bool(torch.isfinite(values).all()):
                raise ValueError("Verifier produced nonfinite evidence")
            output[level + "_scores"] = values
        return output

    def state_dict(self):
        return dict(schema_version=SCHEMA_VERSION, dimension=self.dimension, hidden=self.hidden,
            normalization={level: {k: v.detach().cpu().clone() for k, v in items.items()} for level, items in self.normalization.items()},
            heads={level: {k: v.detach().cpu().clone() for k, v in head.state_dict().items()} for level, head in self.heads.items()},
            fit_report=copy.deepcopy(self.fit_report))

    @classmethod
    def from_state_dict(cls, state):
        if (not isinstance(state, dict) or set(state) != {"schema_version", "dimension", "hidden", "normalization", "heads", "fit_report"}
                or state["schema_version"] != SCHEMA_VERSION or state["dimension"] not in (6, 8)
                or type(state["hidden"]) is not int or state["hidden"] < 1):
            raise ValueError("Invalid verifier state schema")
        report = state["fit_report"]
        if (not isinstance(report, dict) or report.get("schema_version") != SCHEMA_VERSION
                or report.get("evidence_dimension") != state["dimension"] or report.get("hidden") != state["hidden"]
                or report.get("fit_split") != "known_train" or report.get("true_unknown_images_used") is not False
                or report.get("loss") != "bce" or type(report.get("optimizer_steps")) is not int
                or report["optimizer_steps"] < 1):
            raise ValueError("Invalid verifier training report")
        episode_report = report.get("episode_report", {})
        if (episode_report.get("schema_version") != EPISODE_SCHEMA or episode_report.get("fit_split") != "known_train"
                or any(episode_report.get(k) is not False for k in ("true_unknown_images_used", "dev_images_used", "test_images_used"))
                or report.get("frozen_encoder_updated") is not False or report.get("candidate_ranking_changed") is not False):
            raise ValueError("Invalid verifier TRAIN provenance")
        for key in ("epochs", "batch_size"):
            if type(report.get(key)) is not int or report[key] < 1:
                raise ValueError("Invalid verifier training budget")
        steps_by_level = report.get("optimizer_steps_by_level", {})
        if set(steps_by_level) != {"leaf", "parent"} or sum(steps_by_level.values()) != report["optimizer_steps"]:
            raise ValueError("Inconsistent verifier optimizer steps")
        for level in ("leaf", "parent"):
            rows = episode_report.get("examples", {}).get(level, {}).get("rows")
            if (type(rows) is not int or rows < 1 or type(steps_by_level[level]) is not int
                    or steps_by_level[level] != report["epochs"] * math.ceil(rows / report["batch_size"])):
                raise ValueError("Verifier optimizer steps differ from fixed training budget")
        if set(state["heads"]) != {"leaf", "parent"} or set(state["normalization"]) != {"leaf", "parent"}:
            raise ValueError("Missing hierarchy head")
        result = cls()
        result.dimension, result.hidden = state["dimension"], state["hidden"]
        result.heads, result.normalization = {}, {}
        for level in ("leaf", "parent"):
            norm = state["normalization"][level]
            if not isinstance(norm, dict) or set(norm) != {"mean", "scale"}:
                raise ValueError("Invalid normalization schema")
            if any(not torch.is_tensor(value) or not value.is_floating_point() for value in norm.values()):
                raise ValueError("Normalization must contain real floating-point tensors")
            result.normalization[level] = {key: torch.as_tensor(value).detach().cpu().float().clone() for key, value in norm.items()}
            if (any(v.shape != (result.dimension,) or not bool(torch.isfinite(v).all()) for v in result.normalization[level].values())
                    or not bool((result.normalization[level]["scale"] > 0).all())):
                raise ValueError("Invalid normalization values")
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(0)
                head = _Head(result.dimension, result.hidden)
            values = state["heads"][level]
            if (not isinstance(values, dict) or set(values) != set(head.state_dict())
                    or any(not torch.is_tensor(v) or not v.is_floating_point() or not bool(torch.isfinite(v).all()) for v in values.values())):
                raise ValueError("Invalid verifier head tensors")
            try:
                head.load_state_dict(values, strict=True)
            except RuntimeError as error:
                raise ValueError("Invalid verifier head tensor shapes") from error
            result.heads[level] = head.eval().requires_grad_(False)
        result.fit_report = copy.deepcopy(report)
        return result


Verifier = SharedVerifier
