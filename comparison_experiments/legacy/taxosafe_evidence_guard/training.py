"""C00-preserving query fine tuning with genuine, audited TRAIN unknowns.

No class/parent support deletion, Gaussian synthesis, DEV gradients, or TEST
model fitting occurs here. Source-holdout folds remove that source's entire
TRAIN unknown population before sampling or fitting anything trainable.
"""
import copy
import hashlib
import json
import time
from collections import Counter, defaultdict

import numpy as np
import torch
from torch.nn import functional as F

from taxosafe_dcbs.protocol import normalized_name
from taxosafe_support import calibration as base
from .model import EvidenceGuard, RAW_FIELDS


TRAIN_SPLITS = {"train": "known", "train_intra": "intra", "oe_train": "extra"}


def training_inputs(cache, use_oe, exclude_sources=()):
    excluded = {normalized_name(name) for name in exclude_sources}
    parts, rows, hashes = defaultdict(list), [], []
    requested = ("train", "train_intra", "oe_train") if use_oe else ("train",)
    for split in requested:
        if split not in cache["groups"]:
            raise ValueError("Missing audited TRAIN split: " + split)
        group = cache["groups"][split]
        records = {row["image_sha256"]: row for row in base.unique_records(group["records"])}
        indices = []
        for index, image_hash in enumerate(group["image_sha256"]):
            row = records[image_hash]
            if row.get("split") != split or row.get("status") != TRAIN_SPLITS[split]:
                raise ValueError("Training split/status provenance differs")
            if split != "train" and normalized_name(row["source"]) in excluded:
                continue
            if image_hash in hashes:
                raise ValueError("TRAIN known/unknown content overlap")
            indices.append(index)
            rows.append(copy.deepcopy(row))
            hashes.append(image_hash)
        if not indices:
            continue
        take = torch.tensor(indices, dtype=torch.long)
        for key in ("parent", "fine", "parent_local", "fine_local"):
            value = group["encoded"].get(key)
            if value is not None:
                if not torch.isfinite(value).all():
                    raise ValueError("Non-finite cached TRAIN features")
                parts[key].append(value[take].detach().float().cpu())
    if not rows or not any(row["status"] == "known" for row in rows):
        raise ValueError("Known TRAIN queries are required")
    encoded = {key: torch.cat(value, 0) for key, value in parts.items()}
    if any(len(value) != len(rows) for value in encoded.values()):
        raise ValueError("Global/local TRAIN feature views are incomplete")
    return encoded, rows, hashes


def group_weights(rows, device):
    """Equal status mass, then equal known leaf or unknown source mass."""
    groups = defaultdict(lambda: defaultdict(list))
    for i, row in enumerate(rows):
        label = str(int(row["true_leaf"])) if row["status"] == "known" else normalized_name(row["source"])
        groups[row["status"]][label].append(i)
    weights = torch.zeros(len(rows), dtype=torch.float32, device=device)
    for members in groups.values():
        for indices in members.values():
            weights[indices] = 1. / (len(groups) * len(members) * len(indices))
    return weights


def sample_indices(rows, size, rng):
    pools = defaultdict(lambda: defaultdict(list))
    for i, row in enumerate(rows):
        group = str(int(row["true_leaf"])) if row["status"] == "known" else normalized_name(row["source"])
        pools[row["status"]][group].append(i)
    statuses = [status for status in ("known", "intra", "extra") if status in pools]
    if int(size) < len(statuses):
        raise ValueError("Batch size must cover every active TRAIN status")
    result = []
    for i in range(int(size)):
        status = statuses[i % len(statuses)]
        keys = sorted(pools[status])
        group = keys[int(rng.integers(len(keys)))]
        result.append(int(rng.choice(pools[status][group])))
    return torch.tensor(result, dtype=torch.long)


def _balanced_bce(logits, targets):
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    positive = targets > .5
    negative = ~positive
    pos = (loss * positive.to(loss.dtype)).sum(1) / positive.sum(1).clamp_min(1)
    neg = (loss * negative.to(loss.dtype)).sum(1) / negative.sum(1).clamp_min(1)
    count = positive.any(1).to(loss.dtype) + negative.any(1).to(loss.dtype)
    return (pos + neg) / count.clamp_min(1)


def four_head_anchor(student, teacher, router):
    """Per-query distillation of actual C00 rankings and gate margins."""
    terms = []
    for key in ("parent_logits", "leaf_logits"):
        terms.append(F.kl_div(F.log_softmax(student[key] / 2., -1),
                             F.softmax(teacher[key].detach() / 2., -1), reduction="none").sum(1) * 4.)
    for key, threshold in (("parent_membership_logits", "parent_threshold"),
                           ("leaf_membership_logits", "leaf_threshold")):
        target_logit = teacher[key].detach() - float(router[threshold])
        target = torch.sigmoid(target_logit)
        # BCE(target,target) subtraction gives Bernoulli KL, while the stable
        # logit loss retains sensitivity around the actual C00 gate margin.
        cross = F.binary_cross_entropy_with_logits(student[key] - float(router[threshold]), target, reduction="none")
        entropy = F.binary_cross_entropy_with_logits(target_logit, target, reduction="none")
        terms.append((cross - entropy).mean(1))
    return torch.stack(terms).mean(0)


def objective(model, output, teacher_known, rows, router, settings, anchor):
    device = output["parent_logits"].device
    weights = group_weights(rows, device)
    parents, leaves = output["parent_logits"].shape[1], output["leaf_logits"].shape[1]
    mapping = torch.as_tensor(model.meta["leaf_to_parent"], device=device)
    # Build labels on CPU once, rather than synchronizing CUDA for every row.
    target = torch.zeros(len(rows), dtype=torch.long)
    parent_target = torch.zeros(len(rows), parents)
    leaf_target = torch.zeros(len(rows), leaves)
    known_ids, parent_ids = [], []
    for i, row in enumerate(rows):
        if row["status"] != "extra":
            parent = int(row["true_parent"])
            if not 0 <= parent < parents:
                raise ValueError("Invalid supervised TRAIN parent")
            parent_target[i, parent] = 1.
            parent_ids.append(i)
            target[i] = 1 + parent
        if row["status"] == "known":
            leaf = int(row["true_leaf"])
            if not 0 <= leaf < leaves or int(model.meta["leaf_to_parent"][leaf]) != int(row["true_parent"]):
                raise ValueError("Invalid supervised TRAIN leaf/taxonomy")
            leaf_target[i, leaf] = 1.
            known_ids.append(i)
            target[i] = 1 + parents + leaf
    target, parent_target, leaf_target = target.to(device), parent_target.to(device), leaf_target.to(device)
    centered = model.centered(output, router)
    terminal = (F.nll_loss(centered["log_probs"], target, reduction="none") * weights).sum()
    membership = ((_balanced_bce(centered["parent_membership_logits"], parent_target)
                  + _balanced_bce(centered["leaf_membership_logits"], leaf_target)) * weights).sum() / 2.
    classification = terminal.new_zeros(())
    if parent_ids:
        local_rows = [rows[i] for i in parent_ids]
        labels = torch.tensor([int(row["true_parent"]) for row in local_rows], device=device)
        classification = (F.cross_entropy(output["parent_logits"][parent_ids], labels, reduction="none")
                          * group_weights(local_rows, device)).sum()
    if known_ids:
        local_rows = [rows[i] for i in known_ids]
        labels = torch.tensor([int(row["true_leaf"]) for row in local_rows], device=device)
        correct_parents = mapping[labels]
        logits = output["leaf_logits"][known_ids].masked_fill(mapping[None, :] != correct_parents[:, None], -torch.inf)
        classification = classification + (F.cross_entropy(logits, labels, reduction="none")
                                            * group_weights(local_rows, device)).sum()
    distillation = terminal.new_zeros(())
    if anchor:
        if teacher_known is None or not known_ids:
            raise ValueError("Four-head protection requires original known-query teacher outputs")
        student_known = {key: output[key][known_ids] for key in RAW_FIELDS}
        distillation = (four_head_anchor(student_known, teacher_known, router)
                        * group_weights([rows[i] for i in known_ids], device)).sum()
    residual = (output["guard_residual_norm"] * weights).sum()
    loss = (terminal + float(settings["classification"]) * classification
            + float(settings["membership"]) * membership
            + float(settings["distillation"]) * distillation
            + float(settings["residual"]) * residual)
    return loss, dict(terminal=terminal, classification=classification, membership=membership,
                      distillation=distillation, residual=residual)


def fit(cache, meta, arm, cfg, device="cpu", exclude_sources=(), reference_router=None):
    started = time.perf_counter()
    if arm.get("kind", "fit") != "fit":
        raise ValueError("Only learning arms may fit new weights")
    if meta != cache["meta"] or meta != cache["context"]["meta"]:
        raise ValueError("Training taxonomy differs from frozen C00 metadata")
    settings = cfg["training"]
    if int(settings["steps"]) < 1:
        raise ValueError("Learning arms require actual optimizer steps")
    torch.manual_seed(int(cfg["seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg["seed"]))
    rng = np.random.default_rng(int(cfg["seed"]))
    encoded, rows, hashes = training_inputs(cache, arm.get("use_oe", False), exclude_sources)
    geometry_state = None
    if arm.get("geometry", False):
        from .geometry import fit as fit_geometry
        geometry_state = fit_geometry(cache["groups"]["train"], meta, cfg["geometry"])
    context = copy.deepcopy(cache["context"])
    router = copy.deepcopy(reference_router if reference_router is not None else context["source_router"])
    model = EvidenceGuard(context, arm, settings, cfg.get("geometry"), geometry_state).to(device)
    initial = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    parameters = [value for value in model.parameters() if value.requires_grad]
    if not parameters:
        raise ValueError("Learning arm has no trainable residual parameters")
    optimizer = torch.optim.AdamW(parameters, lr=float(settings["learning_rate"]),
                                  weight_decay=float(settings["weight_decay"]))
    encoded = {key: value.to(device) for key, value in encoded.items()}
    fixed_geometry = None
    if geometry_state is not None:
        from .geometry import score as geometry_score
        pieces = defaultdict(list)
        with torch.no_grad():
            for start in range(0, len(rows), 128):
                values = geometry_score({key: value[start:start + 128] for key, value in encoded.items()},
                                        hashes[start:start + 128], geometry_state, device=device)
                for key, value in values.items():
                    pieces[key].append(value.detach())
        fixed_geometry = {key: torch.cat(value, 0) for key, value in pieces.items()}
    history, draws, used_indices = [], Counter(), set()
    sampled_digest = hashlib.sha256()
    model.train()
    for step in range(int(settings["steps"])):
        index = sample_indices(rows, int(settings["batch_size"]), rng)
        selected = index.tolist()
        used_indices.update(selected)
        batch_rows = [rows[i] for i in selected]
        batch_hashes = [hashes[i] for i in selected]
        batch = {key: value[index.to(device)] for key, value in encoded.items()}
        sampled_digest.update(json.dumps(batch_hashes, separators=(",", ":")).encode("utf-8"))
        draws.update(row["status"] for row in batch_rows)
        batch_geometry = None if fixed_geometry is None else {
            key: value[index.to(device)] for key, value in fixed_geometry.items()}
        output = model(batch, batch_hashes, geometry=batch_geometry)
        if any(not torch.isfinite(output[key]).all() for key in RAW_FIELDS):
            raise ValueError("Non-finite full-support training evidence; check per-leaf independent support")
        teacher_known = None
        if arm.get("anchor", True):
            known = [i for i, row in enumerate(batch_rows) if row["status"] == "known"]
            with torch.no_grad():
                teacher_known = model.teacher({key: value[known] for key, value in batch.items()},
                                              [batch_hashes[i] for i in known])
        loss, losses = objective(model, output, teacher_known, batch_rows, router, settings,
                                 arm.get("anchor", True))
        if not torch.isfinite(loss):
            raise ValueError("Non-finite C00 evidence-guard loss")
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 5.)
        optimizer.step()
        if step == 0 or (step + 1) % int(settings["log_every"]) == 0 or step + 1 == int(settings["steps"]):
            item = dict(step=step + 1, loss=float(loss.detach()),
                        **{key: float(value.detach()) for key, value in losses.items()})
            history.append(item)
            print(arm["id"], item, flush=True)
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    delta = {key: float((value.float() - initial[key].float()).square().sum().sqrt())
             for key, value in state.items() if value.is_floating_point()}
    if any(not torch.equal(initial[key], value) for key, value in state.items() if key.startswith("verifier.")):
        raise ValueError("Frozen C00 verifier unexpectedly changed")
    changed = sum(value ** 2 for key, value in delta.items() if not key.startswith("verifier.")) ** .5
    if changed <= 0:
        raise ValueError("No actual residual parameter update")
    by_status = Counter(row["status"] for row in rows)
    source_counts = Counter(row["source"] for row in rows if row["status"] != "known")
    used_rows = [rows[i] for i in sorted(used_indices)]
    state_digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        state_digest.update(key.encode("utf-8"))
        state_digest.update(str((str(value.dtype), tuple(value.shape))).encode("utf-8"))
        state_digest.update(value.contiguous().numpy().tobytes())
    report = dict(training_execution="frozen_C00_verifier_full_support_real_TRAIN_supervision",
        optimizer_steps=int(settings["steps"]), checkpoint_selection="final_predeclared_step",
        gradient_splits=[split for split in TRAIN_SPLITS if any(row["split"] == split for row in rows)],
        fit_image_sha256=hashes, fit_status_counts=dict(by_status), fit_unknown_source_counts=dict(source_counts),
        fit_image_sha256_scope="eligible_TRAIN_queries; actual sampled queries recorded separately",
        fit_queries=[dict(image_sha256=row["image_sha256"], split=row["split"], status=row["status"],
                          source=row["source"]) for row in rows],
        exclude_sources=sorted({normalized_name(name) for name in exclude_sources}),
        excluded_sources=sorted({normalized_name(name) for name in exclude_sources}),
        used_image_sha256=[row["image_sha256"] for row in used_rows],
        used_sources=sorted({normalized_name(row["source"]) for row in used_rows if row["status"] != "known"}),
        used_train_image_sha256={split: [row["image_sha256"] for row in used_rows if row["split"] == split]
                                  for split in TRAIN_SPLITS},
        gradient_used_queries=[dict(image_sha256=row["image_sha256"], split=row["split"], status=row["status"],
                                   source=row["source"]) for row in used_rows],
        training_seed=int(cfg["seed"]), state_sha256=state_digest.hexdigest(),
        unknown_images_used_for_gradients=any(row["status"] != "known" for row in rows),
        source_training_expanded=bool(arm.get("use_oe", False)),
        source_C00_training_unknown_images_used=False, old_C00_checkpoint_modified=False,
        frozen_encoder_updated=False, frozen_verifier_updated=False, frozen_support_updated=False,
        parent_query_adapter_updated=bool(arm.get("adapt_parent", False)),
        fine_query_adapter_updated=bool(arm.get("adapt_fine", False)),
        original_global_and_local_query_views_retained=True,
        teacher_four_head_preservation=bool(arm.get("anchor", True)),
        teacher_fields=list(RAW_FIELDS) if arm.get("anchor", True) else [],
        membership_distillation_center="frozen_reference_router_margins",
        loss_reference_parent_threshold=float(router["parent_threshold"]),
        loss_reference_leaf_threshold=float(router["leaf_threshold"]),
        all_training_query_content_excluded_from_support=True,
        class_or_parent_support_deletion=False, synthetic_feature_count=0,
        test_used_for_fitting=False, development_used_for_gradients=False,
        geometry_known_TRAIN_only=bool(arm.get("geometry", False)),
        geometry_and_new_membership_residual_are_combined_ablation=bool(arm.get("geometry", False)),
        parameter_delta_l2=changed, tensor_delta_l2=delta,
        sampled_status_counts=dict(draws), sampled_query_sha256=sampled_digest.hexdigest(),
        history=history, seconds=time.perf_counter() - started)
    payload = dict(schema_version="c00_evidence_guard_v1", context=context, arm=copy.deepcopy(arm),
                   settings=copy.deepcopy(settings), geometry_settings=copy.deepcopy(cfg.get("geometry", {})),
                   geometry_state=geometry_state, state=state, meta=copy.deepcopy(meta), report=report)
    return payload, report


@torch.no_grad()
def score(cache, payload, device="cpu"):
    if payload.get("schema_version") != "c00_evidence_guard_v1" or cache["meta"] != payload["meta"]:
        raise ValueError("Guard model schema/taxonomy mismatch")
    model = EvidenceGuard(payload["context"], payload["arm"], payload["settings"],
                          payload["geometry_settings"], payload.get("geometry_state")).to(device).eval()
    model.load_state_dict(payload["state"], strict=True)
    groups = {}
    batch_size = int(cache.get("reference_eval_batch_size", 16))
    for split, group in cache["groups"].items():
        scores = {}
        for start in range(0, len(group["image_sha256"]), batch_size):
            hashes = group["image_sha256"][start:start + batch_size]
            encoded = {key: value[start:start + batch_size].to(device)
                       for key, value in group["encoded"].items() if value is not None}
            output = model(encoded, hashes)
            if any(not torch.isfinite(output[key]).all() for key in RAW_FIELDS + ("log_probs",)):
                raise ValueError("Non-finite guard inference evidence")
            for i, image_hash in enumerate(hashes):
                scores[image_hash] = dict(log_probs=output["log_probs"][i].cpu().tolist(),
                    support_evidence={key: output[key][i].cpu().tolist() for key in RAW_FIELDS})
        records = []
        for row in group["records"]:
            result = copy.deepcopy(row)
            result["source_support_evidence"] = copy.deepcopy(row["support_evidence"])
            result["source_log_probs"] = copy.deepcopy(row["log_probs"])
            result.update(scores[row["image_sha256"]])
            records.append(result)
        groups[split] = records
    return groups
