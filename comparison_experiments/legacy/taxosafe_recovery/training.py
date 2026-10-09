"""Fixed-budget, TRAIN-only continuation of the archived D05 verifier heads.

Every variant starts from identical D05 weights and normalization. This module
does not train an image encoder, fit a geometry bank, use unknown images, choose
thresholds, or select epochs on DEV. The original verifier state/report remain
intact inside a separate recovery schema; updated heads are never presented as
an original Discovery training artifact.
"""
import copy
import hashlib
import json
import math

import torch
from torch.nn import functional as F

from taxosafe_discovery.verifier import SharedVerifier, EPISODE_SCHEMA, _validate_examples


SCHEMA_VERSION = "recovery_verifier_v1"
AUDIT_SCHEMA = "recovery_training_v1"
DEFAULTS = dict(steps_per_head=200, batch_size=512, lr=1e-4, hard_weight=1.,
                negative_anchor_weight=1., l2sp_weight=.1, hardness_eta=2.)
MODES = ("bce", "hard", "anchor", "l2sp")


def state_hash(value):
    """Stable nested tensor/content hash, independent of pickle serialization."""
    digest = hashlib.sha256()

    def add(item):
        if torch.is_tensor(item):
            item = item.detach().cpu().contiguous()
            digest.update(json.dumps(["tensor", str(item.dtype), list(item.shape)]).encode("utf-8"))
            digest.update(item.numpy().tobytes())
        elif isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise ValueError("Artifact hash dictionaries require string keys")
            digest.update(b"{")
            for key in sorted(item):
                add(key)
                add(item[key])
            digest.update(b"}")
        elif isinstance(item, (list, tuple)):
            digest.update(b"[")
            for element in item:
                add(element)
            digest.update(b"]")
        elif item is None or isinstance(item, (str, int, float, bool)):
            digest.update(json.dumps([type(item).__name__, item], ensure_ascii=False,
                                     allow_nan=False, separators=(",", ":")).encode("utf-8"))
        else:
            raise ValueError("Unsupported recovery artifact type: " + type(item).__name__)

    add(value)
    return digest.hexdigest()


def _options(options):
    result = dict(DEFAULTS)
    if options is not None:
        if not isinstance(options, dict) or set(options) - set(DEFAULTS):
            raise ValueError("Unknown recovery training option")
        result.update(options)
    for key in ("steps_per_head", "batch_size"):
        if type(result[key]) is not int or result[key] < 1:
            raise ValueError("Recovery " + key + " must be a positive integer")
    for key in set(DEFAULTS) - {"steps_per_head", "batch_size"}:
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("Recovery " + key + " must be finite and positive")
    if result["lr"] > .001:
        raise ValueError("Recovery continuation requires lr <= .001")
    return result


def _tensor_digest(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _source(state):
    # The historical loader forks the CPU RNG, but its torch.manual_seed also
    # touches initialized CUDA generators. Preserve those without initializing
    # CUDA in this otherwise CPU-only continuation worker.
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        model = SharedVerifier.from_state_dict(state)
    finally:
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
    if model.fit_report["loss"] != "bce":
        raise ValueError("Recovery must start from the archived BCE-only D05 verifier")
    return model


def _episodes(episodes, source, candidate_ids):
    if not isinstance(episodes, dict) or set(episodes) != {"leaf", "parent", "report"}:
        raise ValueError("Recovery needs the complete audited TRAIN episode bundle")
    report = episodes["report"]
    original = source.fit_report["episode_report"]
    if (not isinstance(report, dict) or report.get("schema_version") != EPISODE_SCHEMA
            or report.get("fit_split") != "known_train"
            or any(report.get(key) is not False for key in ("true_unknown_images_used", "dev_images_used", "test_images_used"))
            or report.get("image_hash_digest") != original.get("image_hash_digest")
            or report.get("train_count") != original.get("train_count")
            or report.get("folds") != original.get("folds") or report != original):
        raise ValueError("Recovery TRAIN identity/folds/provenance differs from archived D05")
    if not isinstance(candidate_ids, dict) or set(candidate_ids) != {"leaf", "parent"}:
        raise ValueError("candidate_ids must contain unique-TRAIN leaf and parent vectors")
    count = report["train_count"]
    data = {}
    for level in ("leaf", "parent"):
        x, y, weights, pairs = _validate_examples(episodes[level])
        values = episodes[level]
        metadata = {}
        for key in ("source_leaf", "kind", "query_index", "candidate"):
            value = values.get(key)
            if (not torch.is_tensor(value) or value.dtype != torch.long or value.shape != y.shape
                    or bool((value < 0).any())):
                raise ValueError("Invalid TRAIN episode integer metadata: " + key)
            metadata[key] = value.detach().cpu().clone()
        if (bool((metadata["query_index"] >= count).any()) or bool((metadata["kind"] > 2).any())
                or x.shape[1] != source.dimension):
            raise ValueError("TRAIN episode query/kind/dimension differs")
        candidates = candidate_ids[level]
        width = int(metadata["candidate"].max()) + 1
        if (not torch.is_tensor(candidates) or candidates.dtype != torch.long or candidates.shape != (count,)
                or bool((candidates < 0).any() | (candidates >= width).any())):
            raise ValueError("Production candidate IDs must be legal LongTensor values, one per unique TRAIN image")
        candidates = candidates.detach().cpu().clone()
        description = report.get("examples", {}).get(level, {})
        if (description.get("rows") != len(y) or description.get("positive") != int(y.sum())
                or description.get("feature_sha256") != _tensor_digest(x)
                or description.get("target_sha256") != _tensor_digest(y)
                or description.get("ranking_pairs_sha256") != _tensor_digest(pairs)):
            raise ValueError("TRAIN episode tensor hashes/counts differ from audit report")
        if any(description.get(key) != original.get("examples", {}).get(level, {}).get(key)
               for key in ("feature_sha256", "target_sha256", "ranking_pairs_sha256")):
            raise ValueError("Recovery must reproduce the exact archived D05 episode tensors")
        focus = ((metadata["kind"] == 0) & (y == 1)
                 & (metadata["candidate"] == candidates[metadata["query_index"]]))
        if not bool(focus.any()):
            raise ValueError("No correct full-known production candidate available for " + level)
        data[level] = dict(x=x, y=y, weight=weights, pairs=pairs, focus=focus, candidates=candidates, **metadata)
    return data


def _distribution(values):
    values = values.detach().cpu().double().flatten()
    if not len(values) or not bool(torch.isfinite(values).all()):
        raise ValueError("Score-change diagnostic must be finite and nonempty")
    ordered = values.sort().values

    def quantile(q):
        index = (len(ordered) - 1) * q
        lo, hi = int(math.floor(index)), int(math.ceil(index))
        return float(ordered[lo] + (index - lo) * (ordered[hi] - ordered[lo]))

    return dict(count=len(values), mean=float(values.mean()), std=float(values.std(unbiased=False)),
                minimum=float(values.min()), maximum=float(values.max()),
                p05=quantile(.05), median=quantile(.5), p95=quantile(.95))


def _heads_cpu(model):
    return {level: {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
            for level, head in model.heads.items()}


def _make_state(source_state, heads, audit):
    result = dict(schema_version=SCHEMA_VERSION, source_state=copy.deepcopy(source_state),
                  updated_heads=copy.deepcopy(heads), audit=copy.deepcopy(audit))
    result["state_sha256"] = state_hash(result)
    return result


def finetune(source_verifier_state, episodes, candidate_ids, *, mode, options=None, seed=1):
    """Return ``(recovery_state, report)`` after fixed CPU head-only updates.

    The base BCE and its original weights are present in every arm. Additions
    are nested: hard positive; one-sided negative teacher anchor; L2-SP around
    the D05 weights. Sampling uses dedicated local generators; CPU RNG and any
    already initialized CUDA RNG states are preserved.
    """
    if mode not in MODES or type(seed) is not int or not 0 <= seed < 2 ** 31:
        raise ValueError("Invalid recovery mode or seed")
    settings = _options(options)
    before_source_hash = state_hash(source_verifier_state)
    teacher = _source(source_verifier_state)
    model = _source(source_verifier_state)
    data = _episodes(episodes, teacher, candidate_ids)
    normalization_hash = state_hash(source_verifier_state["normalization"])
    initial_heads = _heads_cpu(teacher)
    report = dict(schema_version=AUDIT_SCHEMA, mode=mode, seed=seed, options=settings,
        fit_split="known_train", true_unknown_images_used=False, dev_images_used=False, test_images_used=False,
        encoder_updated=False, geometry_updated=False, normalization_updated=False, candidate_ranking_changed=False,
        optimizer="Adam", optimizer_state_resumed=False, warm_start_no_optimizer_resume=True,
        checkpoint_selection="last_fixed_budget_step_no_DEV", source_state_sha256=before_source_hash,
        source_normalization_sha256=normalization_hash, normalization_sha256=normalization_hash,
        source_head_sha256={level: state_hash(value) for level, value in initial_heads.items()},
        candidate_ids_sha256={level: state_hash(data[level]["candidates"]) for level in data},
        train_image_hash_digest=episodes["report"]["image_hash_digest"],
        episode_report=copy.deepcopy(episodes["report"]), optimizer_steps=0, optimizer_steps_by_level={},
        gradients_all_finite=True, parameter_delta_l2={}, changed_tensor_count={}, history={}, levels={},
        objective="original_weighted_BCE + hard*lambda_h*full_known_correct_production_candidate_BCE"
                  " + anchor*lambda_n*weighted_negative_relu(student-teacher)^2"
                  " + l2sp*lambda_sp*sum(parameter-D05_parameter)^2",
        negative_anchor_direction="penalize_only_increases_of_TRAIN_negative_logits", gradient_clip_norm=5.)
    for offset, level in enumerate(("leaf", "parent")):
        values = data[level]
        norm = teacher.normalization[level]
        x = (values["x"] - norm["mean"]) / norm["scale"]
        y, weights, focus = values["y"], values["weight"], values["focus"]
        head = model.heads[level].train().requires_grad_(True)
        with torch.no_grad():
            teacher_scores = teacher.heads[level](x)
        if not bool(torch.isfinite(x).all() & torch.isfinite(teacher_scores).all()):
            raise ValueError("Nonfinite original-normalized TRAIN evidence")
        focus_indices = focus.nonzero(as_tuple=True)[0]
        focus_sources = values["source_leaf"][focus]
        source_ids = focus_sources.unique(sorted=True)
        hardness = 1. + settings["hardness_eta"] * torch.sigmoid(-teacher_scores[focus])
        focus_weights = torch.zeros_like(hardness)
        # Equal mass per source species; teacher hardness redistributes that
        # species' mass without allowing large/hard species to dominate.
        for source_id in source_ids:
            mask = focus_sources == source_id
            focus_weights[mask] = hardness[mask] / hardness[mask].sum() / len(source_ids)
        negative = y == 0
        negative_weights = weights * negative.float()
        negative_weights = negative_weights / negative_weights.sum()
        initial = {key: value.clone() for key, value in initial_heads[level].items()}
        optimizer = torch.optim.Adam(head.parameters(), lr=settings["lr"])
        generator = torch.Generator().manual_seed(seed + 101 + offset)
        batch_digest = hashlib.sha256()
        order, cursor = torch.empty(0, dtype=torch.long), 0
        report["history"][level] = []
        for step in range(1, settings["steps_per_head"] + 1):
            if cursor >= len(order):
                order = torch.randperm(len(x), generator=generator)
                cursor = 0
            selected = order[cursor:cursor + settings["batch_size"]]
            cursor += len(selected)
            batch_digest.update(selected.numpy().tobytes())
            optimizer.zero_grad()
            logits = head(x[selected])
            factor = len(x) / len(selected)
            base = (F.binary_cross_entropy_with_logits(logits, y[selected], reduction="none")
                    * weights[selected]).sum() * factor
            zero = logits.sum() * 0.
            hard, anchor, penalty = zero, zero, zero
            if mode != "bce":
                focus_scores = head(x[focus_indices])
                hard = (F.binary_cross_entropy_with_logits(focus_scores, torch.ones_like(focus_scores), reduction="none")
                        * focus_weights).sum()
            if mode in ("anchor", "l2sp"):
                anchor = (F.relu(logits - teacher_scores[selected]).square()
                          * negative_weights[selected]).sum() * factor
            if mode == "l2sp":
                penalty = sum((parameter - initial[name]).square().sum() for name, parameter in head.named_parameters())
            loss = base + settings["hard_weight"] * hard + settings["negative_anchor_weight"] * anchor + settings["l2sp_weight"] * penalty
            if not bool(torch.isfinite(loss)):
                raise ValueError("Nonfinite recovery loss")
            loss.backward()
            if any(parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()) for parameter in head.parameters()):
                raise ValueError("Missing or nonfinite recovery head gradient")
            grad_norm = math.sqrt(sum(float(parameter.grad.double().square().sum()) for parameter in head.parameters()))
            torch.nn.utils.clip_grad_norm_(head.parameters(), report["gradient_clip_norm"])
            optimizer.step()
            if any(not bool(torch.isfinite(parameter).all()) for parameter in head.parameters()):
                raise ValueError("Recovery optimizer produced nonfinite weights")
            report["history"][level].append(dict(step=step, base_bce=float(base.detach()),
                hard_positive_bce=float(hard.detach()), negative_anchor=float(anchor.detach()),
                l2sp=float(penalty.detach()), total_loss=float(loss.detach()),
                gradient_norm_before_clip=grad_norm, gradients_finite=True, base_batch_count=len(selected)))
        head.eval().requires_grad_(False)
        with torch.no_grad():
            final_scores = head(x)
        delta_scores = final_scores - teacher_scores
        if not bool(torch.isfinite(final_scores).all()):
            raise ValueError("Recovery final TRAIN scores are nonfinite")
        differences = [head.state_dict()[key] - value for key, value in initial.items()]
        delta_l2 = math.sqrt(sum(float(value.double().square().sum()) for value in differences))
        pairs = values["pairs"]
        before_order = teacher_scores[pairs[:, 0]] > teacher_scores[pairs[:, 1]]
        after_order = final_scores[pairs[:, 0]] > final_scores[pairs[:, 1]]
        score_change = _distribution(delta_scores)
        report["optimizer_steps_by_level"][level] = settings["steps_per_head"]
        report["optimizer_steps"] += settings["steps_per_head"]
        report["parameter_delta_l2"][level] = delta_l2
        report["changed_tensor_count"][level] = sum(int(bool(value.ne(0).any())) for value in differences)
        report["levels"][level] = dict(train_episode_rows=len(y), focus_count=len(focus_indices),
            focus_source_species_count=len(source_ids), focus_source_species_ids=source_ids.tolist(),
            negative_count=int(negative.sum()), positive_weight_mass=float(weights[y == 1].sum()),
            negative_weight_mass=float(weights[negative].sum()), base_weights_sha256=state_hash(weights),
            base_batch_order_sha256=batch_digest.hexdigest(), teacher_scores_sha256=state_hash(teacher_scores),
            focus_weights_sha256=state_hash(focus_weights), focus_indices_sha256=state_hash(focus_indices),
            episode_features_exactly_match_D05=(episodes["report"]["examples"][level]["feature_sha256"]
                == teacher.fit_report["episode_report"]["examples"][level]["feature_sha256"]),
            teacher_focus_bce=float((F.softplus(-teacher_scores[focus]) * focus_weights).sum()),
            final_focus_bce=float((F.softplus(-final_scores[focus]) * focus_weights).sum()),
            score_change=score_change, focus_score_change=_distribution(delta_scores[focus]),
            negative_score_change=_distribution(delta_scores[negative]),
            negative_weighted_mean_drift=float((delta_scores * negative_weights).sum()),
            negative_weighted_squared_increase=float((F.relu(delta_scores).square() * negative_weights).sum()),
            negative_increased_count=int((delta_scores[negative] > 1e-6).sum()),
            same_query_pair_order_changed=int((before_order != after_order).sum()),
            same_query_pair_count=len(pairs),
            degenerate_constant_score_change=score_change["std"] <= 1e-8)
    heads = _heads_cpu(model)
    report["updated_head_sha256"] = {level: state_hash(value) for level, value in heads.items()}
    if (state_hash(source_verifier_state) != before_source_hash
            or state_hash(model.normalization) != normalization_hash
            or any(parameter.requires_grad for head in teacher.heads.values() for parameter in head.parameters())):
        raise ValueError("Frozen D05 source, normalization or teacher changed during continuation")
    result = _make_state(source_verifier_state, heads, report)
    RecoveryVerifier.from_state_dict(result)
    return result, copy.deepcopy(report)


class RecoveryVerifier:
    """Inference-only wrapper preserving the original D05 training artifact."""

    @classmethod
    def from_state_dict(cls, state):
        keys = {"schema_version", "source_state", "updated_heads", "audit", "state_sha256"}
        if not isinstance(state, dict) or set(state) != keys or state["schema_version"] != SCHEMA_VERSION:
            raise ValueError("Invalid recovery verifier state schema")
        unsigned = {key: value for key, value in state.items() if key != "state_sha256"}
        if not isinstance(state["state_sha256"], str) or state_hash(unsigned) != state["state_sha256"]:
            raise ValueError("Recovery state digest mismatch")
        model = _source(state["source_state"])
        report = state["audit"]
        if (not isinstance(report, dict) or report.get("schema_version") != AUDIT_SCHEMA
                or report.get("mode") not in (*MODES, "leaf_guard") or report.get("fit_split") != "known_train"
                or any(report.get(key) is not False for key in ("true_unknown_images_used", "dev_images_used", "test_images_used",
                    "encoder_updated", "geometry_updated", "normalization_updated", "candidate_ranking_changed"))
                or report.get("source_state_sha256") != state_hash(state["source_state"])
                or report.get("source_normalization_sha256") != state_hash(model.normalization)
                or report.get("normalization_sha256") != report.get("source_normalization_sha256")
                or report.get("episode_report") != model.fit_report["episode_report"]
                or report.get("train_image_hash_digest") != model.fit_report["episode_report"]["image_hash_digest"]):
            raise ValueError("Recovery source/provenance/normalization audit differs")
        if not isinstance(report.get("options"), dict) or set(report["options"]) != set(DEFAULTS):
            raise ValueError("Recovery options audit is incomplete")
        settings = _options(report.get("options"))
        expected_steps = 0 if report["mode"] == "leaf_guard" else 2 * settings["steps_per_head"]
        if (type(report.get("optimizer_steps")) is not int or report["optimizer_steps"] != expected_steps
                or report.get("optimizer_steps_by_level") != {"leaf": expected_steps // 2, "parent": expected_steps // 2}
                or report.get("warm_start_no_optimizer_resume") is not True or report.get("optimizer_state_resumed") is not False
                or report.get("gradients_all_finite") is not True):
            raise ValueError("Recovery optimizer audit differs")
        if report["mode"] == "leaf_guard":
            composed = report.get("composition_source_training_audit", {})
            if (report.get("composition_source_mode") != "l2sp" or composed.get("mode") != "l2sp"
                    or composed.get("source_state_sha256") != report["source_state_sha256"]
                    or report.get("reused_optimizer_steps") != composed.get("optimizer_steps")
                    or report.get("parent_restored_to_source") is not True or report.get("history") != {}
                    or composed.get("updated_head_sha256", {}).get("leaf") != report.get("updated_head_sha256", {}).get("leaf")):
                raise ValueError("Invalid leaf-guard composition audit")
        else:
            if not isinstance(report.get("history"), dict) or set(report["history"]) != {"leaf", "parent"}:
                raise ValueError("Recovery loss/gradient history is incomplete")
            for level, history in report["history"].items():
                if not isinstance(history, list) or len(history) != settings["steps_per_head"]:
                    raise ValueError("Recovery history differs from the fixed step budget")
                for step, row in enumerate(history, 1):
                    if (row.get("step") != step or row.get("gradients_finite") is not True
                            or any(not isinstance(row.get(key), (int, float)) or isinstance(row[key], bool)
                                   or not math.isfinite(row[key]) or row[key] < 0 for key in
                                   ("base_bce", "hard_positive_bce", "negative_anchor", "l2sp", "total_loss", "gradient_norm_before_clip"))):
                        raise ValueError("Invalid finite recovery loss/gradient history")
        heads = state["updated_heads"]
        if not isinstance(heads, dict) or set(heads) != {"leaf", "parent"}:
            raise ValueError("Recovery heads are incomplete")
        for level in ("leaf", "parent"):
            values, expected = heads[level], model.heads[level].state_dict()
            if (not isinstance(values, dict) or set(values) != set(expected)
                    or any(not torch.is_tensor(v) or not v.is_floating_point() or v.is_complex()
                           or v.device.type != "cpu" or v.shape != expected[k].shape
                           or v.dtype != expected[k].dtype or not bool(torch.isfinite(v).all()) for k, v in values.items())
                    or report.get("source_head_sha256", {}).get(level) != state_hash(state["source_state"]["heads"][level])
                    or report.get("updated_head_sha256", {}).get(level) != state_hash(values)):
                raise ValueError("Invalid recovery head values/hash: " + level)
            if report["mode"] == "leaf_guard" and level == "parent" and state_hash(values) != report["source_head_sha256"][level]:
                raise ValueError("Leaf guard parent must exactly equal archived D05")
            actual_delta = math.sqrt(sum(float((values[key] - original).double().square().sum()) for key, original in expected.items()))
            actual_changed = sum(int(bool((values[key] != original).any())) for key, original in expected.items())
            reported_delta = report.get("parameter_delta_l2", {}).get(level)
            if (not isinstance(reported_delta, (int, float)) or not math.isfinite(reported_delta)
                    or not math.isclose(actual_delta, reported_delta, rel_tol=1e-12, abs_tol=1e-12)
                    or report.get("changed_tensor_count", {}).get(level) != actual_changed):
                raise ValueError("Recovery reported parameter changes differ from saved heads")
            model.heads[level].load_state_dict(values, strict=True)
            model.heads[level].eval().requires_grad_(False)
        result = cls()
        result._model = model
        result._state = copy.deepcopy(state)
        result.fit_report = copy.deepcopy(report)
        return result

    @torch.no_grad()
    def score(self, geometry_scores, template_scores=None):
        return self._model.score(geometry_scores, template_scores)

    def state_dict(self):
        return copy.deepcopy(self._state)


def make_leaf_guard(state, source_verifier_state):
    """Compose the F06 leaf head with the exact D05 parent; perform no updates."""
    RecoveryVerifier.from_state_dict(state)
    _source(source_verifier_state)
    if state["audit"]["mode"] != "l2sp" or state_hash(source_verifier_state) != state["audit"]["source_state_sha256"]:
        raise ValueError("Leaf guard requires the L2-SP arm and its identical archived D05 source")
    heads = copy.deepcopy(state["updated_heads"])
    heads["parent"] = copy.deepcopy(source_verifier_state["heads"]["parent"])
    report = copy.deepcopy(state["audit"])
    report.update(mode="leaf_guard", optimizer_steps=0, optimizer_steps_by_level={"leaf": 0, "parent": 0},
        training_execution="composed_no_optimizer", composition_source_mode="l2sp",
        composition_source_state_sha256=state["state_sha256"],
        reused_optimizer_steps=state["audit"]["optimizer_steps"],
        checkpoint_selection="F06_updated_leaf_plus_exact_D05_parent",
        composition_source_training_audit=copy.deepcopy(state["audit"]), history={}, levels={},
        parent_restored_to_source=True)
    report["parameter_delta_l2"]["parent"] = 0.
    report["changed_tensor_count"]["parent"] = 0
    report["updated_head_sha256"] = {level: state_hash(value) for level, value in heads.items()}
    result = _make_state(source_verifier_state, heads, report)
    RecoveryVerifier.from_state_dict(result)
    return result, copy.deepcopy(report)
