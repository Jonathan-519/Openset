"""Known-TRAIN fine-tuning aligned with the frozen reference membership router.

Every arm starts from the same source tensors. CLIP backbone weights stay
frozen; only the explicitly named prompts/adapters/evidence heads can update.
The primary checkpoint always comes from an epoch with optimizer updates.
"""
import copy
import json
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from taxosafe_support import pipeline as support
from taxosafe_support.protocol import file_hash, object_hash, read_json, write_json
from taxosafe_support.support import SupportBank
from taxosafe_refine.pipeline import _assert_source, _destination, _frozen
from . import protocol
from .losses import evidence_anchor_losses
from taxosafe_support.membership_calibration import _validate_state

SCHEMA_VERSION = "routealign_training_v1"
ARM = {"id": "A01_evidence_anchor", "kind": "finetune", "scope": "heads",
       "anchored": True, "evidence_anchor": True}
ARTIFACTS = {"checkpoint": "best.pth", "support": "support.pth", "log": "train.jsonl",
             "config_file": "config.json", "plan_file": "plan.json", "inputs": "inputs.json"}
DEFAULT_BUDGET = {"epochs": 20, "patience": 6, "min_epochs": 3,
                  "batches_per_epoch": 240, "prompt_lr": .0005, "head_lr": .001}


def _budget(value):
    result = dict(DEFAULT_BUDGET)
    if set(value) - set(result):
        raise ValueError("Unknown fine-tuning budget fields")
    result.update(value)
    for key in ("epochs", "patience", "min_epochs", "batches_per_epoch"):
        if type(result[key]) is not int or result[key] < 1:
            raise ValueError("Fine-tuning budget " + key + " must be a positive integer")
    if result["min_epochs"] > result["epochs"]:
        raise ValueError("min_epochs cannot exceed epochs")
    for key in ("prompt_lr", "head_lr"):
        if (isinstance(result[key], bool) or not isinstance(result[key], (int, float))
                or not 0 < result[key] <= .01):
            raise ValueError("Invalid fine-tuning " + key)
    return result


def _code():
    return protocol.code_signature()


def _signature(plan, binding):
    return {"method": SCHEMA_VERSION, "code": _code(), "plan": object_hash(plan),
            "source": object_hash(binding)}


def _claim(source, output_dir):
    output = _destination(source.directory, output_dir)
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise ValueError("Training output already exists; use a new experiment directory: " + str(output)) from error
    return output


def _artifact(directory, receipt, key):
    name = ARTIFACTS[key]
    descriptor = receipt.get(key)
    path = Path(directory) / name
    if (not isinstance(descriptor, dict) or descriptor.get("path") != name
            or path.is_symlink() or not path.is_file() or path.resolve().parent != Path(directory).resolve()
            or descriptor.get("sha256") != file_hash(path)):
        raise ValueError("Fine-tuning artifact hash/path mismatch: " + key)
    return path


def _inputs(source):
    groups, audit = support.load_stage_rows(source.config, "train", source.meta)
    if set(groups) != {"train", "val_known"} or audit != source.training["audit"]:
        raise ValueError("Fine-tuning known TRAIN/DEV differs from the frozen source audit")
    if any(not rows or any(r["status"] != "known" for r in rows) for rows in groups.values()):
        raise ValueError("Only nonempty known TRAIN/DEV may enter fine-tuning")
    return groups, audit


def build_student(source, device, scope="heads"):
    """Clone exact source tensors; explicitly restore only permitted gradients."""
    if scope not in ("heads", "prompts_heads"):
        raise ValueError("Fine-tuning scope must be heads or prompts_heads")
    _frozen(source)
    encoder = copy.deepcopy(source.encoder).to(device)
    evidence = copy.deepcopy(source.evidence).to(device)
    encoder.requires_grad_(False)
    evidence.requires_grad_(True)
    names = []
    for name, parameter in encoder.named_parameters():
        allowed = not name.startswith("backbone.") or (
            scope == "prompts_heads" and ("prompt_learner" in name or "VPT" in name))
        parameter.requires_grad_(allowed)
        if allowed:
            names.append("encoder." + name)
    names.extend("evidence." + name for name, parameter in evidence.named_parameters() if parameter.requires_grad)
    if not names or not any(name.startswith("encoder.") for name in names):
        raise ValueError("No trainable adapters/evidence parameters found")
    if scope == "prompts_heads" and not any(name.startswith("encoder.backbone.") for name in names):
        raise ValueError("Prompt arm has no recognized MaPLe prompt parameters")
    return encoder, evidence, sorted(names)


def _optimizer(encoder, evidence, budget, cfg):
    prompts = [p for name, p in encoder.named_parameters() if p.requires_grad and name.startswith("backbone.")]
    heads = [p for name, p in encoder.named_parameters() if p.requires_grad and not name.startswith("backbone.")]
    heads.extend(p for p in evidence.parameters() if p.requires_grad)
    groups = [{"params": heads, "lr": budget["head_lr"]}]
    if prompts:
        groups.insert(0, {"params": prompts, "lr": budget["prompt_lr"]})
    optimizer = torch.optim.SGD(groups, momentum=.9, weight_decay=float(cfg["training"].get("weight_decay", .0005)))
    return optimizer, prompts + heads


def anchor_losses(student, teacher, temperature=2.):
    """Teacher sees the exact same augmentation; no truth or unknown data used."""
    if temperature <= 0:
        raise ValueError("KD temperature must be positive")
    divergences = [F.kl_div(F.log_softmax(student[key] / temperature, dim=1),
                           F.softmax(teacher[key].detach() / temperature, dim=1),
                           reduction="batchmean") * temperature ** 2
                   for key in ("leaf_logits", "parent_logits")]
    feature = [(1. - F.cosine_similarity(student[key].float(), teacher[key].detach().float(), dim=1)).mean()
               for key in ("fine", "parent")]
    return sum(divergences) / len(divergences), sum(feature) / len(feature)


@torch.no_grad()
def known_validation(encoder, evidence, bank, loader, meta, device, source_router):
    """Select using only known DEV and the immutable source operating point.

    The existing membership decoder ranks one parent and then one child before
    comparing that candidate's raw scores to the source thresholds. No router
    is fitted here and no real unknown image enters checkpoint selection.
    """
    parent_threshold, leaf_threshold = _validate_state(source_router, meta)
    encoder.eval()
    evidence.eval()
    text = encoder.text_features()
    mapping = torch.as_tensor(meta["leaf_to_parent"], device=device)
    counts = dict(text=0, structured=0, candidate=0, routed=0, accepted=0, root=0, parent=0)
    nll, count = 0., 0
    offset = 1 + len(meta["parent_names"])
    for images, labels, _ in loader:
        labels = labels.to(device).long()
        encoded = encoder.encode(images.to(device), text_features=text)
        output = evidence(encoded, bank)
        parent = output["parent_logits"].argmax(1)
        permitted = mapping[None, :] == parent[:, None]
        candidate = output["leaf_logits"].masked_fill(~permitted, -torch.inf).argmax(1)
        rows = torch.arange(len(labels), device=device)
        # Float64 retains the exact serialized threshold for the same >= rule
        # used by the production numpy decoder at boundary values.
        pm = output["parent_membership_logits"][rows, parent].double()
        lm = output["leaf_membership_logits"][rows, candidate].double()
        if not bool(torch.isfinite(pm).all() & torch.isfinite(lm).all()):
            raise ValueError("Known validation requires finite candidate membership scores")
        parent_accept = pm >= parent_threshold
        accept = parent_accept & (lm >= leaf_threshold)
        correct = candidate == labels
        counts["text"] += int((encoded["leaf_logits"].argmax(1) == labels).sum())
        counts["structured"] += int((output["log_probs"].argmax(1) == labels + offset).sum())
        counts["candidate"] += int(correct.sum())
        counts["routed"] += int((correct & accept).sum())
        counts["accepted"] += int(accept.sum())
        counts["root"] += int((~parent_accept).sum())
        counts["parent"] += int((parent_accept & ~accept).sum())
        nll += float(F.nll_loss(output["log_probs"], labels + offset, reduction="sum"))
        count += len(labels)
    if not count or not math.isfinite(nll):
        raise ValueError("Known validation must be nonempty with finite NLL")
    return {"leaf_accuracy": counts["text"] / count, "uncalibrated_e2e": counts["structured"] / count,
            "candidate_leaf_accuracy": counts["candidate"] / count, "structured_nll": nll / count,
            "count": count, "selection_splits": ["val_known"], "unknown_data_used": False,
            "source_router_known_correct": counts["routed"], "source_router_known_accuracy": counts["routed"] / count,
            "source_router_leaf_outputs": counts["accepted"], "source_router_root_rejections": counts["root"],
            "source_router_parent_fallbacks": counts["parent"], "membership_thresholds_applied": True,
            "threshold_origin": "immutable_reference_router", "parent_threshold": parent_threshold,
            "leaf_threshold": leaf_threshold,
            "selection_rule": "frozen_source_router_known_correct_then_candidate_leaf_accuracy_then_structured_nll"}


def _selection_key(validation):
    return (validation["source_router_known_correct"], validation["candidate_leaf_accuracy"],
            -validation["structured_nll"])


def _delta(encoder, evidence, source, names):
    current = {"encoder." + k: v for k, v in encoder.named_parameters()}
    current.update({"evidence." + k: v for k, v in evidence.named_parameters()})
    original = {"encoder." + k: v for k, v in source.encoder.named_parameters()}
    original.update({"evidence." + k: v for k, v in source.evidence.named_parameters()})
    squared, maximum, changed = 0., 0., 0
    for name in names:
        difference = (current[name].detach().float() - original[name].detach().to(current[name].device).float())
        squared += float(difference.double().square().sum())
        maximum = max(maximum, float(difference.abs().max()))
        changed += int(bool((difference != 0).any()))
    return {"l2": squared ** .5, "maximum_absolute": maximum, "changed_parameter_tensors": changed}


def _verify_update_scope(encoder, evidence, source, names, expected_delta):
    """Check actual tensors, including frozen CLIP, independently of the receipt."""
    allowed = set(names)
    for prefix, model, baseline in (("encoder.", encoder, source.encoder),
                                     ("evidence.", evidence, source.evidence)):
        original = dict(baseline.named_parameters())
        for name, parameter in model.named_parameters():
            if prefix + name not in allowed and not torch.equal(
                    parameter.detach(), original[name].detach().to(parameter.device)):
                raise ValueError("Checkpoint changed a frozen parameter: " + prefix + name)
    actual = _delta(encoder, evidence, source, names)
    if (not actual["changed_parameter_tensors"] or not math.isfinite(actual["l2"])
            or actual["l2"] <= 0 or not isinstance(expected_delta, dict)
            or actual["changed_parameter_tensors"] != expected_delta.get("changed_parameter_tensors")):
        raise ValueError("Checkpoint must contain a finite, nonzero declared parameter update")
    for key in ("l2", "maximum_absolute"):
        claimed = expected_delta.get(key)
        if (isinstance(claimed, bool) or not isinstance(claimed, (int, float))
                or not math.isclose(actual[key], claimed, rel_tol=1e-5, abs_tol=1e-8)):
            raise ValueError("Checkpoint parameter delta disagrees with its training receipt")
    return actual


def _save_model(path, encoder, evidence, source, plan, signature, epoch, validation, delta):
    support._save_torch(path, {"schema_version": SCHEMA_VERSION, "signature": signature,
        "source_binding": source.binding, "plan": plan, "meta": source.meta,
        "dimension": encoder.dimension, "epoch": epoch, "validation": validation,
        "parameter_delta": delta, "encoder": support._cpu_state(encoder), "evidence": support._cpu_state(evidence)})


def _finish(output, source, plan, signature, audit, encoder, evidence, bank, details):
    checkpoint_hash = file_hash(output / "best.pth")
    support._save_torch(output / "support.pth", {"schema_version": SCHEMA_VERSION,
        "signature": signature, "meta": source.meta, "source_binding": source.binding,
        "checkpoint_sha256": checkpoint_hash, "support_splits": ["train"], "bank": bank.state_dict()})
    _frozen(source)
    _assert_source(source)
    if signature != _signature(plan, source.binding):
        raise ValueError("Fine-tuning source/code/plan changed during execution")
    receipt = {"schema_version": SCHEMA_VERSION, "signature": signature, "source_binding": source.binding,
        "arm": plan["arm"], "plan": plan, "effective_config": plan["config"], "meta": source.meta,
        "audit": audit, "training_completed": True, "test_used_for_fitting": False,
        "unknown_images_used_for_gradients": False, "support_splits": ["train"], **details}
    receipt.update({key: {"path": name, "sha256": file_hash(output / name)} for key, name in ARTIFACTS.items()})
    write_json(output / "completed.json", receipt)
    return receipt


def train_arm(source, arm, cfg, output_dir, device, seed, budget):
    """Train a real update arm; return a receipt bound to its own checkpoint/bank."""
    _frozen(source)
    _assert_source(source)
    if arm != ARM:
        raise ValueError("RouteAlign requires the fixed A01_evidence_anchor arm")
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError("seed must be an integer in [0,2**31)")
    budget = _budget(budget)
    cfg = copy.deepcopy(cfg)
    if cfg != source.config:
        raise ValueError("Pass the untouched source configuration; arm changes are applied explicitly")
    cfg["seed"] = seed
    cfg["training"].update(epochs=budget["epochs"], patience=budget["patience"], warmup_epochs=0,
                            prompt_lr=budget["prompt_lr"], head_lr=budget["head_lr"], selection="frozen_source_router_known_first",
                            anchor_temperature=2., episode_ramp_epochs=1, min_episode_epochs=0,
                            min_epochs=budget["min_epochs"])
    cfg["data"]["seed"] = seed
    cfg["data"].setdefault("sampler", {})["batches_per_epoch"] = budget["batches_per_epoch"]
    cfg["data"]["sampler"].update(seed=seed, holdout_seed=seed)
    cfg["support"]["loss"]["anchor"] = .5
    groups, audit = _inputs(source)
    output = _claim(source, output_dir)
    plan = {"arm": copy.deepcopy(arm), "config": cfg, "seed": seed, "budget": budget}
    signature = _signature(plan, source.binding)
    write_json(output / "plan.json", plan)
    write_json(output / "config.json", cfg)
    write_json(output / "inputs.json", {"signature": signature, "audit": audit, "meta": source.meta})
    support.seed_all(seed)
    encoder, evidence, names = build_student(source, device, arm.get("scope", "heads"))
    optimizer, parameters = _optimizer(encoder, evidence, budget, cfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, budget["epochs"])
    train_loader = support.make_loader(groups["train"], cfg, source.meta, training=True)
    reference_loader = support.make_loader(groups["train"], cfg, source.meta)
    val_loader = support.make_loader(groups["val_known"], cfg, source.meta)
    epoch0 = known_validation(source.encoder, source.evidence, source.bank, val_loader,
                              source.meta, device, source.router)
    bank, original_anchor, initial_bank_seconds = support.reference_bank(
        encoder, reference_loader, groups["train"], cfg, source.meta, device, collect_anchor=True)
    if original_anchor is None or len(original_anchor) != len(groups["train"]):
        raise ValueError("Initial frozen TRAIN logits must align with every TRAIN manifest row")
    teacher_text = source.encoder.text_features().detach() if arm.get("anchored", False) else None
    best_key, best_epoch, best_validation, best_delta = None, None, None, None
    best_overall_key = _selection_key(epoch0)
    best_overall_epoch, bad_epochs, steps_total = 0, 0, 0
    best_steps_total, stop_reason = None, "epoch_budget"
    exposure = {}
    started = time.perf_counter()
    for epoch in range(1, budget["epochs"] + 1):
        if hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch - 1)
        encoder.train()
        evidence.train()
        totals, counts, steps = {}, {}, 0
        epoch_started = time.perf_counter()
        for batch_index, (images, labels, indices) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device).long()
            hashes = [groups["train"][int(i)]["image_sha256"] for i in indices]
            encoded = encoder(images)
            loss, terms, valid = support.training_loss(encoded, labels, indices, hashes, bank, evidence,
                cfg, source.meta, seed=seed * 1000003 + (epoch - 1) * 10007 + batch_index,
                episode_weight=1., anchor=original_anchor)
            if arm.get("anchored", False):
                with torch.no_grad():
                    teacher = source.encoder.encode(images, text_features=teacher_text)
                kd, feature = anchor_losses(encoded, teacher, temperature=2.)
                with torch.no_grad():
                    teacher_output = source.evidence(teacher, source.bank, query_hashes=hashes)
                student_output = evidence(encoded, bank, query_hashes=hashes)
                membership, ranking, anchor_audit = evidence_anchor_losses(
                    student_output, teacher_output, labels, source.meta["leaf_to_parent"], temperature=2.)
                loss = loss + .5 * kd + .5 * feature + membership + .5 * ranking
                terms = dict(terms, teacher_kd=kd, teacher_feature=feature,
                             evidence_membership_anchor=membership, evidence_rank_anchor=ranking)
                valid = dict(valid, **{"evidence_anchor_" + k: v for k, v in anchor_audit.items()})
            if not bool(torch.isfinite(loss)):
                raise ValueError("Nonfinite fine-tuning loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, float(cfg["training"].get("gradient_clip", 5.)), error_if_nonfinite=True)
            optimizer.step()
            steps += 1
            steps_total += 1
            for key, value in dict(terms, total=loss).items():
                totals[key] = totals.get(key, 0.) + float(value.detach())
            for key, value in valid.items():
                counts[key] = counts.get(key, 0) + int(value)
                exposure[key] = exposure.get(key, 0) + int(value)
            if batch_index % 40 == 0:
                print("{} epoch={} step={} loss={:.5f}".format(arm["id"], epoch, batch_index, float(loss.detach())), flush=True)
        if not steps:
            raise ValueError("Fine-tuning loader returned no batches")
        scheduler.step()
        bank, _, refresh_seconds = support.reference_bank(encoder, reference_loader, groups["train"], cfg, source.meta, device)
        validation = known_validation(encoder, evidence, bank, val_loader, source.meta, device, source.router)
        key = _selection_key(validation)
        delta = _delta(encoder, evidence, source, names)
        if not delta["changed_parameter_tensors"] or not delta["l2"] > 0:
            raise ValueError("Fine-tuning produced no parameter update; cannot label the arm trained")
        improved = best_key is None or key > best_key
        if improved:
            best_key, best_epoch, best_validation, best_delta, bad_epochs = key, epoch, validation, delta, 0
            best_steps_total = steps_total
            _save_model(output / "best.pth", encoder, evidence, source, plan, signature, epoch, validation, delta)
        else:
            bad_epochs += 1
        if key > best_overall_key:
            best_overall_key, best_overall_epoch = key, epoch
        record = {"epoch": epoch, "steps": steps, "optimizer_steps_total": steps_total,
            "loss": {k: v / steps for k, v in totals.items()}, "known_validation": validation,
            "valid_episode_queries": counts, "parameter_delta": delta,
            "best_updated_epoch": best_epoch, "best_including_source_epoch": best_overall_epoch,
            "checkpoint_selection": "frozen_source_router_known_correct_then_candidate_then_nll;updated_epochs_only",
            "elapsed_seconds": time.perf_counter() - started,
            "epoch_seconds": time.perf_counter() - epoch_started, "support_refresh_seconds": refresh_seconds}
        with (output / "train.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        print(json.dumps(record, allow_nan=False), flush=True)
        if epoch >= budget["min_epochs"] and bad_epochs >= budget["patience"]:
            stop_reason = "patience"
            break
    if best_epoch is None:
        raise ValueError("No actual updated checkpoint available")
    selected = support._load_torch(output / "best.pth")
    encoder.load_state_dict(selected["encoder"], strict=True)
    evidence.load_state_dict(selected["evidence"], strict=True)
    bank, _, final_refresh_seconds = support.reference_bank(encoder, reference_loader, groups["train"], cfg, source.meta, device)
    return _finish(output, source, plan, signature, audit, encoder, evidence, bank, {
        "kind": "finetune", "gradient_splits": ["train"], "optimizer_steps": steps_total,
        "epochs_completed": epoch, "best_epoch": best_epoch, "known_validation": best_validation,
        "selected_checkpoint_optimizer_steps": best_steps_total, "stop_reason": stop_reason,
        "primary_checkpoint_semantics": "best_updated_epoch_ge_1_even_if_source_is_better",
        "parameter_delta": best_delta, "trainable_parameter_names": names,
        "trainable_parameter_count": sum(p.numel() for p in parameters),
        "query_visual_passes_per_step": 2 if arm.get("anchored", False) else 1,
        "baseline_selection": {"epoch0_validation": epoch0, "best_including_source_epoch": best_overall_epoch,
            "source_would_be_retained": best_overall_epoch == 0,
            "source_checkpoint_sha256": source.binding["checkpoint_sha256"], "used_as_primary": False},
        "anchor": {"initial_train_leaf_logits_weight": .5, "temperature": 2.,
            "teacher_same_augmentation": bool(arm.get("anchored", False)),
            "teacher_kd_weight": .5 if arm.get("anchored", False) else 0.,
            "teacher_feature_weight": .5, "membership_raw_smooth_l1_weight": 1.,
            "evidence_rank_kl_weight": .5, "membership_beta": 1.,
            "evidence_views": ["full_with_query_hash_excluded"],
            "membership_reduction": "positive_negative_per_query_then_equal_query_leaf_then_equal_parent_leaf",
            "ranking_reduction": "parent_KL_and_mean_within_parent_leaf_KL_then_equal_query_leaf",
            "teacher_support": "immutable_source_TRAIN", "student_support": "own_refreshed_TRAIN"},
        "valid_episode_queries": exposure, "initial_support_seconds": initial_bank_seconds,
        "final_support_seconds": final_refresh_seconds, "elapsed_seconds": time.perf_counter() - started})


def load_arm_model(source, training_dir, device):
    """Load only a matched updated checkpoint and its own TRAIN bank."""
    _frozen(source)
    _assert_source(source)
    directory = Path(training_dir)
    receipt = read_json(directory / "completed.json")
    if (receipt.get("schema_version") != SCHEMA_VERSION or receipt.get("source_binding") != source.binding
            or receipt.get("training_completed") is not True or receipt.get("test_used_for_fitting") is not False
            or receipt.get("unknown_images_used_for_gradients") is not False or receipt.get("support_splits") != ["train"]
            or receipt.get("meta") != source.meta or receipt.get("audit") != source.training["audit"]):
        raise ValueError("Arm training source/schema/data-role binding mismatch")
    if receipt.get("kind") == "finetune":
        if (receipt.get("gradient_splits") != ["train"] or type(receipt.get("optimizer_steps")) is not int
                or receipt["optimizer_steps"] < 1 or type(receipt.get("best_epoch")) is not int
                or receipt["best_epoch"] < 1):
            raise ValueError("Fine-tuned primary checkpoint must represent actual optimizer updates")
    else:
        raise ValueError("Unsupported fine-tuning artifact kind")
    for key in ARTIFACTS:
        _artifact(directory, receipt, key)
    plan = read_json(directory / "plan.json")
    if (receipt.get("plan") != plan or receipt.get("arm") != plan.get("arm")
            or receipt.get("effective_config") != plan.get("config")
            or read_json(directory / "config.json") != plan.get("config")
            or receipt.get("signature") != _signature(plan, source.binding)
            or read_json(directory / "inputs.json") != {
                "signature": receipt["signature"], "audit": receipt["audit"], "meta": source.meta}):
        raise ValueError("Arm training configuration/plan/code signature mismatch")
    checkpoint = support._load_torch(directory / "best.pth")
    payload = support._load_torch(directory / "support.pth")
    for value in (checkpoint, payload):
        if (value.get("schema_version") != SCHEMA_VERSION or value.get("signature") != receipt["signature"]
                or value.get("source_binding") != source.binding or value.get("meta") != source.meta):
            raise ValueError("Arm checkpoint/support binding mismatch")
    if (checkpoint.get("plan") != plan or checkpoint.get("epoch") != receipt.get("best_epoch")
            or checkpoint.get("validation") != receipt.get("known_validation")
            or checkpoint.get("parameter_delta") != receipt.get("parameter_delta")
            or payload.get("checkpoint_sha256") != receipt["checkpoint"]["sha256"]
            or payload.get("support_splits") != ["train"]):
        raise ValueError("Arm selected checkpoint and support disagree")
    if plan.get("arm") != ARM:
        raise ValueError("RouteAlign checkpoint belongs to an unsupported arm")
    scope = "heads"
    encoder, evidence, names = build_student(source, device, scope)
    if receipt.get("trainable_parameter_names") != names:
        raise ValueError("Checkpoint trainable parameter names disagree with the declared scope")
    if checkpoint.get("dimension") != encoder.dimension:
        raise ValueError("Arm encoder dimensions differ from the source")
    encoder.load_state_dict(checkpoint["encoder"], strict=True)
    evidence.load_state_dict(checkpoint["evidence"], strict=True)
    _verify_update_scope(encoder, evidence, source, names, receipt["parameter_delta"])
    encoder.requires_grad_(False).eval()
    evidence.requires_grad_(False).eval()
    bank = SupportBank.from_state_dict(payload["bank"]).to(device)
    if (not set(bank.hashes) <= set(source.training["audit"]["train"]["image_hashes"])
            or bank.leaf_to_parent.tolist() != source.meta["leaf_to_parent"]
            or bank.max_per_leaf != int(source.config["support"].get("max_per_leaf", 8))
            or hasattr(bank, "required_leaf_mask")):
        raise ValueError("Arm support must contain only complete known TRAIN taxonomy")
    return encoder, evidence, bank, receipt
