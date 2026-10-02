"""Known-only support training, development calibration and immutable evaluation.

One differentiable query image pass serves every support intervention. References
are detached TRAIN-only features refreshed after each epoch and reused for the
next epoch. Real unknown images are never opened by :func:`train`.
"""
import argparse
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .calibration import apply_router, calibrate, evaluate_gates, evaluate_records, raw_records
from .encoders import SupportEncoder
from .evidence import HierarchicalEvidence
from .episodes import build_episodes
from .losses import hierarchical_losses, reference_supervision_counts, representation_losses
from .protocol import (PROJECT_ROOT, VARIANTS, claim_stage, effective_config, file_hash,
                       load_stage_rows, read_json, require_signature, resolve, run_lock,
                       signature, verify_artifact, write_json, write_records)
from .support import SupportBank

DEFAULT_CONFIG = "configs/Zooplankton_Taxonomic_Tree/TaxoSafe_support_new.yml"
SCHEMA_VERSION = 1


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def hierarchy(cfg):
    from loader.hierdata import _load_hierarchy
    from taxosafe_visual.runtime import EXPECTED_PARENTS
    data = dict(cfg["data"], hierarchy=str(resolve(cfg["data"]["hierarchy"])))
    tree = _load_hierarchy(data)
    names = tree["param_names"]
    meta = {"parent_names": [names[int(i)] for i in tree["intnl_nodes"][0]],
            "leaf_names": [names[int(i)] for i in tree["leaf_nodes"]],
            "leaf_to_parent": tree["sublabels"][:, 0].long().tolist()}
    if meta["parent_names"] != list(EXPECTED_PARENTS):
        raise ValueError("TaxoSafe parent order differs from the locked hierarchy")
    if len(meta["leaf_names"]) != int(data["num_known_leaves"]):
        raise ValueError("Known leaf count mismatch")
    return meta


class Images(Dataset):
    def __init__(self, rows, transform):
        self.rows, self.transform = rows, transform
        self.target = [r["true_leaf"] if r["status"] == "known" else
                       r["true_parent"] if r["status"] == "intra" else -1 for r in rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        from loader.utils import default_loader
        image = self.transform(default_loader(self.rows[index]["resolved_path"]))
        return image, self.target[index], index


def make_loader(rows, cfg, meta, training=False):
    from .preprocessing import get_transform
    from loader.hierarchical_episode_sampler import HierarchicalEpisodeBatchSampler
    data = cfg["data"]
    dataset = Images(rows, get_transform(data, training))
    kwargs = {"num_workers": int(data.get("n_workers", 4)), "pin_memory": True}
    sampler = dict(data.get("sampler", {}))
    name = sampler.pop("name", "random")
    if training and name == "hierarchical_episode":
        batch_sampler = HierarchicalEpisodeBatchSampler(
            dataset, meta["leaf_to_parent"], leaf_names=meta["leaf_names"], **sampler)
        if batch_sampler.batch_size != int(data["batch_size"]):
            raise ValueError("Episode size and data.batch_size differ")
        return DataLoader(dataset, batch_sampler=batch_sampler, **kwargs)
    return DataLoader(dataset, batch_size=int(data["batch_size"] if training else
                      data.get("eval_batch_size", 16)), shuffle=training, **kwargs)


def make_backbone(cfg, meta, device):
    from models import get_model
    names = meta["leaf_names"]
    active = cfg.get("strict_holdout", {}).get("active_leaf_mask")
    if active is not None:
        if len(active) != len(names) or not any(active):
            raise ValueError("Invalid strict holdout backbone class mask")
        names = [name for name, allowed in zip(names, active) if allowed]
    backbone = get_model(cfg["model"], names).to(device)
    for name, parameter in backbone.named_parameters():
        parameter.requires_grad_("prompt_learner" in name or "VPT" in name)
    if not any(p.requires_grad for p in backbone.parameters()):
        raise ValueError("No trainable MaPLe prompt parameters found")
    return backbone


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _cpu_state(module):
    return {key: value.detach().cpu() for key, value in module.state_dict().items()}


def _save_torch(path, value):
    temporary = Path(str(path) + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _load_torch(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # Repository also supports older PyTorch installations.
        return torch.load(path, map_location="cpu")


@torch.no_grad()
def reference_features(encoder, loader, device, collect_anchor=False):
    """Encode the complete TRAIN manifest once; support cannot contain val/test."""
    encoder.eval()
    values, labels, seen, anchor = {}, [], [], []
    text_features = encoder.text_features() if collect_anchor else None
    for images, target, indices in loader:
        encoded = encoder.encode(images.to(device), text_features=text_features, classify=collect_anchor)
        for key in ("parent", "fine", "parent_local", "fine_local"):
            if encoded.get(key) is not None:
                values.setdefault(key, []).append(encoded[key].detach())
        labels.append(target.to(device).long())
        seen.extend(indices.tolist())
        if collect_anchor:
            anchor.append(encoded["leaf_logits"].detach())
    if seen != list(range(len(loader.dataset))):
        raise ValueError("Reference pass must visit the complete TRAIN set once, in order")
    if not seen:
        raise ValueError("TRAIN reference set is empty")
    return ({key: torch.cat(tensors) for key, tensors in values.items()}, torch.cat(labels),
            torch.cat(anchor) if collect_anchor else None)


def reference_bank(encoder, loader, rows, cfg, meta, device, collect_anchor=False,
                   active_leaf_mask=None):
    _sync(device)
    started = time.perf_counter()
    encoded, labels, anchor = reference_features(encoder, loader, device, collect_anchor)
    bank = SupportBank(encoded["parent"], encoded["fine"], labels,
                       [r["image_sha256"] for r in rows], meta["leaf_to_parent"],
                       parent_local=encoded.get("parent_local"), fine_local=encoded.get("fine_local"),
                       max_per_leaf=int(cfg["support"].get("max_per_leaf", 8)),
                       required_leaf_mask=active_leaf_mask).to(device)
    _sync(device)
    return bank, anchor, time.perf_counter() - started


@torch.no_grad()
def known_validation(encoder, evidence, bank, loader, meta, device, selection_splits=None,
                     selection="text"):
    if selection not in ("text", "candidate"):
        raise ValueError("Known checkpoint selection must be text or candidate")
    encoder.eval()
    evidence.eval()
    correct, structured_correct, nll, count = 0, 0, 0., 0
    candidate_correct = 0
    leaf_offset = 1 + len(meta["parent_names"])
    text_features = encoder.text_features()
    for images, labels, _ in loader:
        labels = labels.to(device).long()
        encoded = encoder.encode(images.to(device), text_features=text_features)
        output = evidence(encoded, bank)
        correct += int((encoded["leaf_logits"].argmax(1) == labels).sum())
        structured_correct += int((output["log_probs"].argmax(1) == labels + leaf_offset).sum())
        if selection == "candidate":
            # Match the production membership decoder's identity rule without
            # using thresholds or any real unknown images for model selection.
            parent = output["parent_logits"].argmax(1)
            mapping = torch.as_tensor(meta["leaf_to_parent"], device=labels.device)
            permitted = mapping[None, :] == parent[:, None]
            candidate = output["leaf_logits"].masked_fill(~permitted, -torch.inf).argmax(1)
            candidate_correct += int((candidate == labels).sum())
        nll += float(F.nll_loss(output["log_probs"], labels + leaf_offset, reduction="sum"))
        count += len(labels)
    if not count:
        raise ValueError("Known validation split is empty")
    result = {"leaf_accuracy": correct / count, "uncalibrated_e2e": structured_correct / count,
            "structured_nll": nll / count, "count": count,
            "selection_splits": selection_splits or ["val_known"], "unknown_data_used": False}
    if selection == "candidate":
        result.update(candidate_leaf_accuracy=candidate_correct / count,
                      selection_rule="candidate_leaf_accuracy_then_structured_nll",
                      membership_thresholds_applied=False)
    return result


def _selection_key(validation, selection="text"):
    if selection not in ("text", "candidate"):
        raise ValueError("Known checkpoint selection must be text or candidate")
    metric = "candidate_leaf_accuracy" if selection == "candidate" else "leaf_accuracy"
    return validation[metric], -validation["structured_nll"]


def _save_checkpoint(path, encoder, evidence, cfg, meta, epoch, validation):
    _save_torch(path, {"schema_version": SCHEMA_VERSION, "method": "support_conditioned",
                      "config": cfg, "meta": meta, "epoch": epoch,
                      "validation": validation, "dimension": encoder.dimension,
                      "encoder": _cpu_state(encoder), "evidence": _cpu_state(evidence)})


def _make_evidence(encoder, cfg, meta, device):
    settings = cfg["support"]
    extra = ({"relation_dim": int(settings.get("relation_dim", 32))}
             if settings.get("membership", "prototype") == "relation" else {})
    return HierarchicalEvidence(encoder.dimension, meta["leaf_to_parent"],
                                hidden_dim=int(settings.get("hidden_dim", 32)),
                                temperature=float(settings.get("temperature", .1)),
                                local_enabled=bool(settings.get("local_enabled", True)),
                                decoupled=bool(settings.get("decoupled", False)),
                                membership_mode=settings.get("membership", "prototype"),
                                reference_topk=settings.get("reference_topk", 2), **extra).to(device)


def _optimizer(encoder, evidence, settings):
    prompts, adapters = [], []
    for name, parameter in encoder.named_parameters():
        if parameter.requires_grad:
            (prompts if name.startswith("backbone.") else adapters).append(parameter)
    heads = adapters + [p for p in evidence.parameters() if p.requires_grad]
    if not prompts or not heads:
        raise ValueError("Both prompt and hierarchical evidence parameters must be trainable")
    optimizer = torch.optim.SGD([
        {"params": prompts, "lr": float(settings["prompt_lr"])},
        {"params": heads, "lr": float(settings["head_lr"])}],
        momentum=.9, weight_decay=float(settings.get("weight_decay", .0005)))
    return optimizer, prompts + heads


def train(cfg, directory, device, debug=False):
    meta = hierarchy(cfg)
    groups, audit = load_stage_rows(cfg, "train", meta)
    sig = signature(cfg)
    return train_rows(cfg, directory, device, groups, meta, audit, sig, debug=debug)


def train_rows(cfg, directory, device, groups, meta, audit, sig, debug=False,
               active_leaf_mask=None, selection_splits=None):
    """Shared training engine for the main run and strict TRAIN-only folds.

    The public ``train`` stage owns normal manifest loading. Fold validation
    passes audited subsets of TRAIN without reopening development or test data.
    """
    if set(groups) != {"train", "val_known"} or any(not rows for rows in groups.values()):
        raise ValueError("Training requires nonempty train and known-validation rows")
    if any(row["status"] != "known" for rows in groups.values() for row in rows):
        raise ValueError("Only known rows may enter the training engine")
    if active_leaf_mask is not None:
        active_leaf_mask = torch.as_tensor(active_leaf_mask, dtype=torch.bool)
        if active_leaf_mask.shape != (len(meta["leaf_names"]),) or not bool(active_leaf_mask.any()):
            raise ValueError("Invalid TRAIN-only fold mask")
        fold = cfg.get("strict_holdout", {})
        if fold.get("active_leaf_mask") != active_leaf_mask.tolist():
            raise ValueError("Fold mask must be bound to the configuration signature")
        if any(not active_leaf_mask[row["true_leaf"]] for rows in groups.values() for row in rows):
            raise ValueError("Held-out images cannot enter training or known model selection")
    elif cfg.get("strict_holdout"):
        raise ValueError("A strict holdout fold requires its active-leaf mask")
    output = claim_stage(directory, "training")
    write_json(output / "config.json", cfg)
    write_json(output / "inputs.json", {"signature": sig, "audit": audit, "meta": meta})
    seed_all(cfg["seed"])
    train_loader = make_loader(groups["train"], cfg, meta, training=True)
    reference_loader = make_loader(groups["train"], cfg, meta)
    val_loader = make_loader(groups["val_known"], cfg, meta)
    encoder = SupportEncoder(make_backbone(cfg, meta, device), meta, cfg["support"],
                             active_leaf_mask=active_leaf_mask).to(device)
    evidence = _make_evidence(encoder, cfg, meta, device)
    settings = cfg["training"]
    optimizer, parameters = _optimizer(encoder, evidence, settings)
    epochs = 2 if debug else int(settings["epochs"])
    warmup = 0 if debug else int(settings["warmup_epochs"])
    use_episodes = bool(cfg["support"].get("episodes_enabled", True))
    minimum = (1 if debug else int(settings.get("min_episode_epochs", 5))) if use_episodes else 0
    if epochs < warmup + minimum:
        raise ValueError("Too few epochs for warmup and required support intervention exposure")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    bank, anchor, reference_seconds = reference_bank(
        encoder, reference_loader, groups["train"], cfg, meta, device, collect_anchor=warmup == 0,
        active_leaf_mask=active_leaf_mask)
    parameters_report = {"total": sum(p.numel() for p in encoder.parameters()) +
                         sum(p.numel() for p in evidence.parameters()),
                         "trainable": sum(p.numel() for p in parameters),
                         "query_backbone_passes_per_step": 1,
                         "references_per_leaf_max": int(cfg["support"].get("max_per_leaf", 8)),
                         "speed_parity_with_v11": "unmeasured; compare same hardware and manifest"}
    if cfg["support"].get("membership", "prototype") in ("reference", "relation"):
        parameters_report.update(membership=cfg["support"]["membership"], reference_topk=int(cfg["support"].get("reference_topk", 2)),
                                 pair_supervision="full_support_only; query content excluded",
                                 parent_positive_pairs="same parent, different leaf; singleton fallback logged")
    if cfg["support"].get("membership", "prototype") == "relation":
        parameters_report.update(relation_dim=int(cfg["support"].get("relation_dim", 32)),
                                 pair_negative_topk=cfg["support"].get("pair_negative_topk"),
                                 reference_pair_selection=("per_leaf_topk_negative/all_positive"
                                     if "pair_negative_topk" in cfg["support"] else "all_allowed_pairs"),
                                 checkpoint_selection=settings.get("selection", "text"))
    write_json(output / "model_cost.json", parameters_report)
    best_key, best_epoch, best_validation, bad_epochs = None, None, None, 0
    episode_epochs, exposure = 0, {}
    started = time.perf_counter()
    for epoch in range(epochs):
        if hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch)
        encoder.train()
        evidence.train()
        ramp = (min(1., max(0., (epoch - warmup + 1) /
                max(1, int(settings.get("episode_ramp_epochs", 5))))) if use_episodes else 0.)
        totals, diagnostics, counts, steps = {}, {}, {}, 0
        _sync(device)
        step_started = time.perf_counter()
        for step, (images, labels, indices) in enumerate(train_loader):
            labels = labels.to(device).long()
            encoded = encoder(images.to(device))
            query_hashes = [groups["train"][int(i)]["image_sha256"] for i in indices]
            # Query embeddings stay attached; only cached support and teacher logits detach.
            loss, terms, valid_counts = training_loss(
                encoded, labels, indices, query_hashes, bank, evidence, cfg, meta,
                seed=int(cfg["seed"]) * 1000003 + epoch * 10007 + step,
                episode_weight=ramp, anchor=anchor, active_leaf_mask=active_leaf_mask)
            if not bool(torch.isfinite(loss)):
                raise ValueError("Non-finite support training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, float(settings.get("gradient_clip", 5.)),
                                           error_if_nonfinite=True)
            optimizer.step()
            for key, value in dict(terms, total=loss).items():
                if key.startswith("diagnostic_"):
                    diagnostic = key[len("diagnostic_"):]
                    diagnostics[diagnostic] = diagnostics.get(diagnostic, 0.) + float(value.detach())
                else:
                    totals[key] = totals.get(key, 0.) + float(value.detach())
            for key, count in valid_counts.items():
                counts[key] = counts.get(key, 0) + int(count)
                exposure[key] = exposure.get(key, 0) + int(count)
            steps += 1
            if step % 40 == 0:
                print("epoch={} step={} loss={:.4f}".format(epoch + 1, step, float(loss.detach())), flush=True)
            if debug and step >= 1:
                break
        if not steps:
            raise ValueError("Training loader produced no batches")
        _sync(device)
        step_seconds = time.perf_counter() - step_started
        scheduler.step()
        if ramp > 0:
            episode_epochs += 1
        bank, warmup_anchor, next_reference_seconds = reference_bank(
            encoder, reference_loader, groups["train"], cfg, meta, device,
            collect_anchor=epoch + 1 == warmup, active_leaf_mask=active_leaf_mask)
        if warmup_anchor is not None:
            anchor = warmup_anchor
        _sync(device)
        validation_started = time.perf_counter()
        validation = known_validation(encoder, evidence, bank, val_loader, meta, device,
                                      selection_splits=selection_splits,
                                      selection=settings.get("selection", "text"))
        _sync(device)
        validation_seconds = time.perf_counter() - validation_started
        eligible = epoch + 1 >= warmup + minimum and episode_epochs >= minimum
        key = _selection_key(validation, settings.get("selection", "text"))
        if eligible and (best_key is None or key > best_key):
            best_key, best_epoch, best_validation, bad_epochs = key, epoch + 1, validation, 0
            _save_checkpoint(output / "best.pth", encoder, evidence, cfg, meta, best_epoch, validation)
        elif eligible:
            bad_epochs += 1
        record = {"epoch": epoch + 1, "loss": {k: v / steps for k, v in totals.items()},
                  "known_validation": validation, "valid_episode_queries": counts,
                  "episode_epochs": episode_epochs, "episode_weight": ramp,
                  "checkpoint_eligible": eligible, "best_epoch": best_epoch,
                  "timing": {"training_seconds": step_seconds, "step_seconds": step_seconds / steps,
                             "reference_seconds": next_reference_seconds,
                             "validation_seconds": validation_seconds,
                             "initial_reference_seconds": reference_seconds if epoch == 0 else 0.},
                  "elapsed_seconds": time.perf_counter() - started}
        if diagnostics:
            record["diagnostics"] = {key: value / steps for key, value in diagnostics.items()}
        with open(output / "train.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        print(json.dumps(record, allow_nan=False), flush=True)
        if int(settings.get("patience", 25)) > 0 and eligible and bad_epochs >= int(settings.get("patience", 25)):
            break
    if best_epoch is None:
        raise ValueError("No eligible known-only checkpoint; calibration has not run")
    checkpoint = _load_torch(output / "best.pth")
    encoder.load_state_dict(checkpoint["encoder"], strict=True)
    evidence.load_state_dict(checkpoint["evidence"], strict=True)
    bank, _, final_reference_seconds = reference_bank(
        encoder, reference_loader, groups["train"], cfg, meta, device,
        active_leaf_mask=active_leaf_mask)
    checkpoint_hash = file_hash(output / "best.pth")
    _save_torch(output / "support.pth", {"schema_version": SCHEMA_VERSION,
                "checkpoint_sha256": checkpoint_hash, "signature": sig, "meta": meta,
                "gradient_splits": ["train"], "bank": bank.state_dict()})
    require_signature(sig, signature(cfg))
    write_json(output / "completed.json", {"schema_version": SCHEMA_VERSION,
               "method": "support_conditioned", "debug": debug, "signature": sig,
               "config": cfg, "meta": meta, "audit": audit, "seed": cfg["seed"],
               "best_epoch": best_epoch, "known_validation": best_validation,
               "gradient_splits": ["train"], "support_splits": ["train"],
               "test_used_for_fitting": False, "unknown_images_used_for_gradients": False,
               "episode_epochs": episode_epochs, "valid_episode_queries": exposure,
               "anchor": {"source": "CLIP_initialized_encoder" if warmup == 0 else "known_only_warmup",
                          "epoch": warmup, "gradient_splits": ["train"], "detached": True},
               "model_cost": parameters_report, "final_reference_seconds": final_reference_seconds,
               "checkpoint": {"path": "best.pth", "sha256": checkpoint_hash},
               "support": {"path": "support.pth", "sha256": file_hash(output / "support.pth")}})
    print("Training finished: " + str(output / "completed.json"), flush=True)


def training_loss(encoded, labels, indices, query_hashes, bank, evidence, cfg, meta,
                  seed, episode_weight, anchor=None, active_leaf_mask=None):
    """Reuse one live query graph across all masked references and teacher KL."""
    settings = cfg["support"]
    weights = settings.get("loss", {})
    if active_leaf_mask is None:
        episodes = build_episodes(labels, meta["leaf_to_parent"], seed=seed)
    else:
        from .holdout import build_active_episodes
        episodes = build_active_episodes(labels, meta["leaf_to_parent"], active_leaf_mask, seed)
    names = ["full"]
    if episode_weight > 0 and settings.get("episodes_enabled", True):
        names.extend(["drop_leaf", "control_leaf"])
        if settings.get("parent_holdout_enabled", True):
            names.extend(["drop_parent", "control_parent"])
    kwargs = {}
    if settings.get("membership", "prototype") in ("reference", "relation"):
        # Explicit batch-local reuse keeps the pair/local graph attached while
        # each intervention independently masks and pools the same raw pairs.
        kwargs["reference_pair_logits"] = evidence.reference_pairs(encoded, bank)
    outputs = {name: evidence(encoded, bank, mask=episodes["masks"][name],
                              query_hashes=query_hashes, **kwargs) for name in names}
    support_weights = {"leaf": float(weights.get("support_leaf", .25)),
                       "parent": float(weights.get("support_parent", .1)),
                       "episode": episode_weight * float(weights.get("episode", 1.)),
                       "paired": episode_weight * float(weights.get("paired", .25)),
                       "control": episode_weight * float(weights.get("control", .25))}
    if settings.get("decoupled", False):
        for key in ("membership_parent", "membership_leaf"):
            support_weights[key] = episode_weight * float(weights.get(key, 1.))
    if settings.get("membership", "prototype") in ("reference", "relation"):
        support_weights.update(reference_parent=episode_weight * float(weights.get("pair_parent", .25)),
                               reference_leaf=episode_weight * float(weights.get("pair_leaf", .25)))
    loss_options = {}
    if settings.get("membership", "prototype") == "relation" and "pair_negative_topk" in settings:
        loss_options["reference_negative_topk"] = settings["pair_negative_topk"]
    components = hierarchical_losses(outputs, episodes, labels, meta["leaf_to_parent"],
                                    weights=support_weights, margins=settings.get("margins", {}),
                                    **loss_options)
    leaf = F.cross_entropy(encoded["leaf_logits"], labels)
    mapping = torch.tensor(meta["leaf_to_parent"], dtype=torch.long, device=labels.device)
    parent = F.cross_entropy(encoded["parent_logits"], mapping[labels])
    anchor_loss = leaf * 0.
    if anchor is not None and episode_weight > 0 and float(weights.get("anchor", .1)) > 0:
        temperature = float(cfg["training"].get("anchor_temperature", 2.))
        if temperature <= 0:
            raise ValueError("anchor_temperature must be positive")
        teacher = anchor[indices.to(anchor.device)].to(labels.device).detach()
        student = encoded["leaf_logits"]
        if active_leaf_mask is not None:
            active = torch.as_tensor(active_leaf_mask, dtype=torch.bool, device=student.device)
            student, teacher = student[:, active], teacher[:, active]
        anchor_loss = F.kl_div(F.log_softmax(student / temperature, dim=1),
                              F.softmax(teacher / temperature, dim=1), reduction="batchmean") * temperature ** 2
    total = (components["total"] + float(weights.get("leaf", 1.)) * leaf +
             float(weights.get("parent", .25)) * parent +
             episode_weight * float(weights.get("anchor", .1)) * anchor_loss)
    terms = {"encoder_leaf": leaf, "encoder_parent": parent, "anchor": anchor_loss}
    terms.update({"support_" + key: components[key] for key in
                  ("leaf", "parent", "episode", "paired", "control")})
    if settings.get("decoupled", False):
        for key in ("membership_parent", "membership_leaf"):
            terms[key] = components[key]
        representation = representation_losses(encoded, labels, meta["leaf_to_parent"],
                                                query_hashes=query_hashes,
                                                margin=float(settings.get("margins", {}).get("representation", .2)))
        for key in ("parent_cross_species", "leaf_sibling"):
            terms[key] = representation[key]
            total = total + episode_weight * float(weights.get(key, .1)) * representation[key]
    if settings.get("membership", "prototype") in ("reference", "relation"):
        for key in ("reference_parent", "reference_leaf"):
            terms[key] = components[key]
    rows = torch.arange(len(labels), device=labels.device)
    counts = {name: int((episodes["valid"][name] & torch.isfinite(
                       out["log_probs"][rows, episodes["targets"][name]])).sum())
              for name, out in outputs.items()}
    if settings.get("decoupled", False):
        counts.update({key: int(value) for key, value in representation.items() if key.startswith("valid_")})
    if settings.get("membership", "prototype") in ("reference", "relation"):
        count_options = {}
        if settings.get("membership", "prototype") == "relation" and "pair_negative_topk" in settings:
            count_options["negative_topk"] = settings["pair_negative_topk"]
        pair_counts = reference_supervision_counts(outputs["full"], labels, meta["leaf_to_parent"], **count_options)
        # Exposure means effective supervised pairs, not only available pairs:
        # warmup computes heads but pair losses carry zero weight.
        for key, value in pair_counts.items():
            component = "reference_parent" if key.startswith("parent_") else "reference_leaf"
            counts[key] = int(value) if support_weights[component] > 0 else 0
    if settings.get("membership", "prototype") == "relation":
        terms.update(_relation_diagnostics(encoded, outputs["full"], labels, mapping))
    return total, terms, counts


@torch.no_grad()
def _relation_diagnostics(encoded, output, labels, mapping):
    """Cheap TRAIN-only measurements; never an objective or selection score."""
    diagnostics = {}
    for branch in ("parent", "fine"):
        local = encoded.get(branch + "_local")
        if local is not None and local.shape[1] > 1:
            local = F.normalize(local.detach().float(), dim=-1)
            similarity = local @ local.transpose(1, 2)
            mask = ~torch.eye(local.shape[1], dtype=torch.bool, device=local.device)
            diagnostics["diagnostic_" + branch + "_local_offdiag_cosine"] = similarity[:, mask].mean()
    logits = output["leaf_membership_logits"].detach()
    rows = torch.arange(len(labels), device=labels.device)
    siblings = (mapping[None, :] == mapping[labels, None]) & output["active_leaves"]
    siblings[rows, labels] = False
    rival = logits.masked_fill(~siblings, -torch.inf).max(-1).values
    positive = logits[rows, labels]
    valid = torch.isfinite(positive) & torch.isfinite(rival)
    margin = (positive - rival)[valid]
    diagnostics["diagnostic_leaf_membership_sibling_margin"] = (
        margin.mean() if margin.numel() else logits.new_zeros(()))
    return diagnostics


def load_trained(cfg, directory, device):
    training = Path(directory) / "training"
    receipt, checkpoint_path = verify_artifact(training, "completed.json", "checkpoint")
    _, support_path = verify_artifact(training, "completed.json", "support")
    if receipt.get("debug"):
        raise ValueError("Debug runs cannot calibrate or evaluate locked test data")
    if receipt.get("schema_version") != SCHEMA_VERSION or receipt.get("method") != "support_conditioned":
        raise ValueError("Unsupported support training receipt")
    require_signature(receipt["signature"], signature(cfg))
    checkpoint, payload = _load_torch(checkpoint_path), _load_torch(support_path)
    meta = hierarchy(cfg)
    if (checkpoint.get("schema_version") != SCHEMA_VERSION or
            checkpoint.get("method") != "support_conditioned" or checkpoint["meta"] != meta or
            receipt["meta"] != meta or checkpoint["config"] != cfg or receipt["config"] != cfg):
        raise ValueError("Checkpoint configuration or taxonomy mismatch")
    if (payload.get("schema_version") != SCHEMA_VERSION or payload.get("meta") != meta or
            payload.get("checkpoint_sha256") != receipt["checkpoint"]["sha256"] or
            payload.get("gradient_splits") != ["train"]):
        raise ValueError("Cached support does not belong to this TRAIN-only checkpoint")
    require_signature(receipt["signature"], payload["signature"])
    bank = SupportBank.from_state_dict(payload["bank"]).to(device)
    if not set(bank.hashes).issubset(set(receipt["audit"]["train"]["image_hashes"])):
        raise ValueError("Support contains content outside the audited TRAIN split")
    active_leaf_mask = cfg.get("strict_holdout", {}).get("active_leaf_mask")
    encoder = SupportEncoder(make_backbone(cfg, meta, device), meta, cfg["support"],
                             active_leaf_mask=active_leaf_mask).to(device)
    if checkpoint["dimension"] != encoder.dimension:
        raise ValueError("Checkpoint encoder dimension mismatch")
    evidence = _make_evidence(encoder, cfg, meta, device)
    encoder.load_state_dict(checkpoint["encoder"], strict=True)
    evidence.load_state_dict(checkpoint["evidence"], strict=True)
    encoder.requires_grad_(False).eval()
    evidence.requires_grad_(False).eval()
    return encoder, evidence, receipt, bank


@torch.no_grad()
def collect(groups, cfg, meta, encoder, evidence, bank, device):
    encoder.eval()
    evidence.eval()
    text_features = encoder.text_features()
    result, timings = {}, {}
    for split, rows in groups.items():
        # Byte-identical aliases are encoded once, so GPU batch rounding cannot
        # create conflicting predictions for a single evaluation identity.
        canonical = {}
        for row in rows:
            canonical.setdefault(row["image_sha256"], row)
        unique_rows = list(canonical.values())
        scored, seen = [], []
        _sync(device)
        started = time.perf_counter()
        for images, _, indices in make_loader(unique_rows, cfg, meta):
            encoded = encoder.encode(images.to(device), text_features=text_features)
            output = evidence(encoded, bank)
            selected = [unique_rows[i] for i in indices.tolist()]
            seen.extend(indices.tolist())
            batch_records = raw_records(selected, {"log_probs": output["log_probs"].cpu().numpy()},
                                        encoded["leaf_logits"].cpu().numpy(), meta)
            if cfg["support"].get("decoupled", False):
                # Export the actual candidate evidence, rather than trying to
                # recover it from the mixed tree probabilities during analysis.
                diagnostic = {key: output[key].cpu().tolist() for key in (
                    "parent_membership_logits", "leaf_membership_logits", "parent_logits", "leaf_logits")}
                for i, record in enumerate(batch_records):
                    record["support_evidence"] = {key: value[i] for key, value in diagnostic.items()}
            scored.extend(batch_records)
        _sync(device)
        elapsed = time.perf_counter() - started
        if seen != list(range(len(unique_rows))):
            raise ValueError("Inference loader must visit every unique manifest image once in order")
        by_hash = {row["image_sha256"]: row for row in scored}
        records = [dict(by_hash[row["image_sha256"]], **row) for row in rows]
        result[split] = records
        timings[split] = {"manifest_rows": len(records), "unique_images": len(unique_rows),
                          "seconds": elapsed, "seconds_per_unique_image": elapsed / len(unique_rows)}
        print("Scored {}: {} unique images ({} rows) in {:.3f}s".format(
              split, len(unique_rows), len(records), elapsed), flush=True)
    return result, timings


def metrics_for(groups, router, meta):
    from metrics_open import evaluate_open_set
    routed = {split: apply_router(records, router, meta) for split, records in groups.items()}
    statuses = {kind: [r for rows in routed.values() for r in rows if r["status"] == kind]
                for kind in ("known", "intra", "extra")}
    return routed, evaluate_open_set(statuses["known"], statuses["intra"], statuses["extra"])


def calibrate_run(cfg, directory, device):
    if cfg.get("strict_holdout"):
        raise ValueError("TRAIN-only holdout folds cannot calibrate on real unknown development data")
    encoder, evidence, trained, bank = load_trained(cfg, directory, device)
    meta = trained["meta"]
    groups, audit = load_stage_rows(cfg, "calibrate", meta,
                                    forbidden_hashes=trained["audit"]["train"]["image_hashes"])
    if audit["val_known"] != trained["audit"]["val_known"]:
        raise ValueError("Validation known data changed since checkpoint selection")
    output = claim_stage(directory, "calibration")
    records, timings = collect(groups, cfg, meta, encoder, evidence, bank, device)
    write_records(output / "development_scores.jsonl", [r for rows in records.values() for r in rows])
    try:
        router = calibrate(records["val_known"], records["val_intra"], records["val_extra"],
                           meta, cfg["calibration"])
    except ValueError as exc:
        write_json(output / "failed.json", {"reason": str(exc), "test_used_for_fitting": False})
        raise
    router.update(checkpoint_sha256=trained["checkpoint"]["sha256"],
                  support_sha256=trained["support"]["sha256"], signature=trained["signature"])
    routed, metrics = metrics_for(records, router, meta)
    report = dict(router["validation_report"])
    checked = evaluate_records([r for rows in routed.values() for r in rows], meta)
    if any(report[key] != checked[key] for key in ("counts", "checks", "metrics", "targets_passed")):
        raise ValueError("Frozen decoder disagrees with the calibration report")
    write_json(output / "router.json", router)
    write_json(output / "development_metrics.json", metrics)
    write_json(output / "validation_report.json", report)
    write_json(output / "inference_timing.json", timings)
    write_records(output / "development_predictions.jsonl", [r for rows in routed.values() for r in rows])
    require_signature(trained["signature"], signature(cfg))
    write_json(output / "completed.json", {"schema_version": SCHEMA_VERSION,
               "method": "support_conditioned", "audit": audit, "signature": trained["signature"],
               "checkpoint_sha256": trained["checkpoint"]["sha256"],
               "support_sha256": trained["support"]["sha256"], "test_used_for_fitting": False,
               "fit_completed": True, "targets_passed": report["targets_passed"],
               "fit_splits": list(groups),
               "router": {"path": "router.json", "sha256": file_hash(output / "router.json")}})
    print("Development router frozen; targets_passed={}: {}".format(
          report["targets_passed"], output / "router.json"), flush=True)


def test_run(cfg, directory, device):
    if cfg.get("strict_holdout"):
        raise ValueError("TRAIN-only holdout folds cannot evaluate locked test data")
    from .calibration import unique_records
    calibrated, router_path = verify_artifact(Path(directory) / "calibration", "completed.json", "router")
    if calibrated.get("schema_version") != SCHEMA_VERSION or calibrated.get("method") != "support_conditioned":
        raise ValueError("Unsupported calibration receipt")
    router = read_json(router_path)
    encoder, evidence, trained, bank = load_trained(cfg, directory, device)
    require_signature(trained["signature"], calibrated["signature"])
    require_signature(trained["signature"], router["signature"])
    for artifact in ("checkpoint", "support"):
        digest = trained[artifact]["sha256"]
        if calibrated.get(artifact + "_sha256") != digest or router.get(artifact + "_sha256") != digest:
            raise ValueError("Frozen router and training artifacts differ")
    forbidden_hashes = set(trained["audit"]["train"]["image_hashes"])
    forbidden_sources = set()
    for split, audit in calibrated["audit"].items():
        forbidden_hashes.update(audit["image_hashes"])
        if split != "val_known":
            forbidden_sources.update(audit["sources"])
    groups, audit = load_stage_rows(cfg, "test", trained["meta"], forbidden_hashes, forbidden_sources)
    output = claim_stage(directory, "test")
    records, timings = collect(groups, cfg, trained["meta"], encoder, evidence, bank, device)
    routed, metrics_all = metrics_for(records, router, trained["meta"])
    unique = {split: unique_records(rows) for split, rows in routed.items()}
    _, metrics = metrics_for(unique, router, trained["meta"])
    report = evaluate_records([r for rows in routed.values() for r in rows], trained["meta"])
    gates = evaluate_gates(metrics, cfg.get("evaluation_gates", {}))
    for rows in routed.values():
        seen = set()
        for row in rows:
            row["evaluation_weight"] = int(row["image_sha256"] not in seen)
            seen.add(row["image_sha256"])
    write_records(output / "predictions.jsonl", [r for rows in routed.values() for r in rows])
    write_json(output / "metrics.json", metrics)
    write_json(output / "metrics_all_rows.json", metrics_all)
    write_json(output / "summary.json", report)
    write_json(output / "gates.json", gates)
    write_json(output / "inference_timing.json", timings)
    require_signature(trained["signature"], signature(cfg))
    write_json(output / "completed.json", {"schema_version": SCHEMA_VERSION,
               "method": "support_conditioned", "audit": audit, "signature": trained["signature"],
               "checkpoint_sha256": trained["checkpoint"]["sha256"],
               "router_sha256": calibrated["router"]["sha256"],
               "support_sha256": trained["support"]["sha256"],
               "test_used_for_fitting": False, "metric_unit": "unique_image_sha256",
               "gate_passed": gates["targets_passed"], "targets_passed": report["targets_passed"]})
    print(json.dumps(gates, ensure_ascii=False), flush=True)


def run(stage):
    if stage not in ("train", "calibrate", "test"):
        raise ValueError("Unknown stage: " + stage)
    parser = argparse.ArgumentParser(description="TaxoSafe support-conditioned hierarchy: " + stage)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--trial", type=int, default=1)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--variant", choices=VARIANTS, default="main")
    parser.add_argument("--run-dir")
    parser.add_argument("--preflight", action="store_true", help="Audit ONLY this stage's inputs, without CLIP/CUDA")
    if stage == "train":
        parser.add_argument("--debug", action="store_true", help="Short non-evaluable run with a separate directory")
    args = parser.parse_args()
    config_path = Path(args.config).resolve() if Path(args.config).exists() else resolve(args.config)
    os.chdir(PROJECT_ROOT)
    cfg = effective_config(config_path, args.variant, args.trial if args.seed is None else args.seed)
    debug = getattr(args, "debug", False)
    directory = resolve(args.run_dir) if args.run_dir else resolve(cfg["output_root"]) / args.variant / ("trial_" + str(args.trial))
    if debug:
        directory = directory.with_name(directory.name + "_debug")
    if args.preflight:
        meta = hierarchy(cfg)
        _, audit = load_stage_rows(cfg, stage, meta)
        print(json.dumps({"stage": stage, "inputs": {s: {"count": a["count"],
                          "unique_image_count": a["unique_image_count"], "source_count": len(a["sources"])}
                          for s, a in audit.items()}, "signature": signature(cfg)}, indent=2))
        return
    if not torch.cuda.is_available():
        raise SystemExit("Full MaPLe execution requires CUDA. CPU contract tests and --preflight remain available.")
    seed_all(cfg["seed"])
    print("Run directory: " + str(directory), flush=True)
    with run_lock(directory):
        if stage == "train":
            train(cfg, directory, torch.device("cuda"), debug)
        elif stage == "calibrate":
            calibrate_run(cfg, directory, torch.device("cuda"))
        else:
            test_run(cfg, directory, torch.device("cuda"))
