"""Separate train, development calibration and frozen-test entry points."""
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

from .calibration import apply_router, calibrate, fit_score_support, raw_records
from .model import DCBSHeads, dcbs_loss
from .protocol import (PROJECT_ROOT, VARIANTS, claim_stage, effective_config, file_hash,
                       load_stage_rows, read_json, require_signature, resolve, run_lock,
                       signature, verify_artifact, write_json, write_records)
from .synthesis import empty_synthetic, fit_support, synthesize

DEFAULT_CONFIG = "configs/Zooplankton_Taxonomic_Tree/TaxoSafe_dcbs_v11.yml"


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
        self.target = [r["true_leaf"] if r["status"] == "known" else r["true_parent"] if r["status"] == "intra" else -1 for r in rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        from loader.utils import default_loader
        image = self.transform(default_loader(self.rows[index]["resolved_path"]))
        return image, self.target[index], index


def make_loader(rows, cfg, meta, training=False):
    from loader.transforms import get_transform
    from loader.hierarchical_episode_sampler import HierarchicalEpisodeBatchSampler
    data = cfg["data"]
    dataset = Images(rows, get_transform(data.get("transform", "clip"), training))
    kwargs = {"num_workers": int(data.get("n_workers", 4)), "pin_memory": True}
    sampler = dict(data.get("sampler", {}))
    name = sampler.pop("name", "random")
    if training and name == "hierarchical_episode":
        batch_sampler = HierarchicalEpisodeBatchSampler(dataset, meta["leaf_to_parent"], leaf_names=meta["leaf_names"], **sampler)
        if batch_sampler.batch_size != int(data["batch_size"]):
            raise ValueError("Episode size and data.batch_size differ")
        return DataLoader(dataset, batch_sampler=batch_sampler, **kwargs)
    return DataLoader(dataset, batch_size=int(data["batch_size"] if training else data.get("eval_batch_size", 16)),
                      shuffle=training, **kwargs)


def make_backbone(cfg, meta, device):
    from models import get_model
    backbone = get_model(cfg["model"], meta["leaf_names"]).to(device)
    for name, parameter in backbone.named_parameters():
        parameter.requires_grad_("prompt_learner" in name or "VPT" in name)
    if not any(p.requires_grad for p in backbone.parameters()):
        raise ValueError("No trainable MaPLe prompt parameters found")
    return backbone


def scale_of(backbone):
    return backbone.model.logit_scale.exp().float()


def encode_texts(backbone, meta):
    return (backbone.encode_text(meta["leaf_names"], normalize=True).float(),
            backbone.encode_text(meta["parent_names"], normalize=True).float())


def encode_training_batch(backbone, images, meta):
    h = backbone.encode_image(images, normalize=True).float()
    leaves, parents = encode_texts(backbone, meta)
    return h, scale_of(backbone) * h @ leaves.T, scale_of(backbone) * h @ parents.T


@torch.no_grad()
def reference_features(backbone, loader, device):
    backbone.eval()
    features, labels, seen = [], [], []
    for images, target, index in loader:
        features.append(backbone.encode_image(images.to(device), normalize=True).float())
        labels.append(target.to(device).long())
        seen.extend(index.tolist())
    if seen != list(range(len(loader.dataset))):
        raise ValueError("Reference pass must visit the complete TRAIN set once, in order")
    return torch.cat(features), torch.cat(labels)


@torch.no_grad()
def known_validation(backbone, heads, loader, meta, device):
    backbone.eval()
    heads.eval()
    leaves, _ = encode_texts(backbone, meta)
    correct, root_loss, count = 0, 0., 0
    mapping = heads.leaf_to_parent
    for images, labels, _ in loader:
        labels = labels.to(device).long()
        h = backbone.encode_image(images.to(device), normalize=True).float()
        logits = scale_of(backbone) * h @ leaves.T
        correct += int((logits.argmax(1) == labels).sum())
        root_loss += float(F.cross_entropy(heads(h)["root"], mapping[labels], reduction="sum"))
        count += len(labels)
    return {"leaf_accuracy": correct / count, "root_ce": root_loss / count, "count": count}


def _cpu_state(module):
    return {k: v.detach().cpu() for k, v in module.state_dict().items()}


def _save_checkpoint(path, backbone, heads, cfg, meta, epoch, validation, input_dim):
    temporary = Path(str(path) + ".tmp")
    torch.save({"schema_version": 11, "config": cfg, "meta": meta, "epoch": epoch,
                "validation": validation, "input_dim": input_dim,
                "backbone": _cpu_state(backbone), "heads": _cpu_state(heads)}, temporary)
    temporary.replace(path)


def _load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # Older repository environments predate weights_only.
        return torch.load(path, map_location="cpu")


def _sample_pool(pool, count):
    result = dict(pool)
    for kind in ("near", "extra"):
        points = pool[kind]
        index = torch.randperm(len(points), device=points.device)[:count]
        result[kind] = points[index]
        if kind == "near":
            result["near_parent"] = pool["near_parent"][index]
    return result


def train(cfg, directory, device, debug=False):
    meta = hierarchy(cfg)
    groups, audit = load_stage_rows(cfg, "train", meta)
    sig = signature(cfg)
    output = claim_stage(directory, "training")
    write_json(output / "config.json", cfg)
    write_json(output / "inputs.json", {"signature": sig, "audit": audit, "meta": meta})
    seed_all(cfg["seed"])
    train_loader = make_loader(groups["train"], cfg, meta, training=True)
    reference_loader = make_loader(groups["train"], cfg, meta)
    val_loader = make_loader(groups["val_known"], cfg, meta)
    backbone = make_backbone(cfg, meta, device)
    with torch.no_grad():
        input_dim = encode_texts(backbone, meta)[0].shape[1]
    heads = DCBSHeads(input_dim, meta["leaf_to_parent"], len(meta["parent_names"]), cfg["dcbs"]).to(device)
    parameters = [p for p in backbone.parameters() if p.requires_grad]
    settings, synthesis_cfg = cfg["training"], cfg["dcbs"]["synthesis"]
    optimizer = torch.optim.SGD([{"params": parameters, "lr": float(settings["prompt_lr"])},
                                 {"params": heads.parameters(), "lr": float(settings["head_lr"])}],
                                momentum=.9, weight_decay=float(settings.get("weight_decay", .0005)))
    epochs = 2 if debug else int(settings["epochs"])
    warmup = 0 if debug else int(settings["warmup_epochs"])
    min_synthesis = 1 if debug else int(settings.get("min_synthesis_epochs", 5))
    if epochs < warmup + min_synthesis:
        raise ValueError("Too few epochs for warmup and required synthesis exposure")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    best_key, best_epoch, best_validation, bad_epochs = None, None, None, 0
    synthesis_epochs = {"near": 0, "extra": 0}
    coverage = {str(p): 0 for p in range(len(meta["parent_names"]))}
    start = time.monotonic()
    for epoch in range(epochs):
        if hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch)
        h_reference, reference_labels = reference_features(backbone, reference_loader, device)
        bank = fit_support(h_reference, reference_labels, meta["leaf_to_parent"], synthesis_cfg)
        with torch.no_grad():
            leaves, _ = encode_texts(backbone, meta)
            pool = synthesize(bank, leaves, scale_of(backbone), synthesis_cfg) if epoch >= warmup else empty_synthetic(bank)
        for kind in synthesis_epochs:
            synthesis_epochs[kind] += int(len(pool[kind]) > 0)
        for p, value in pool["stats"]["near_by_parent"].items():
            coverage[p] += int(value)
        ramp = min(1., max(0., (epoch - warmup + 1) / max(1, int(settings.get("synthesis_ramp_epochs", 5)))))
        backbone.train()
        heads.train()
        totals, steps = {}, 0
        for step, (images, labels, _) in enumerate(train_loader):
            h, leaf_logits, parent_logits = encode_training_batch(backbone, images.to(device), meta)
            synthetic = _sample_pool(pool, int(synthesis_cfg.get("batch_samples", 32)))
            loss, terms = dcbs_loss(heads, h, labels.to(device).long(), leaf_logits, parent_logits,
                                    synthetic, cfg["dcbs"], novelty_weight=ramp)
            if not bool(torch.isfinite(loss)):
                raise ValueError("Non-finite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters + list(heads.parameters()), float(settings.get("gradient_clip", 5.)), error_if_nonfinite=True)
            optimizer.step()
            for key, value in dict(terms, total=loss).items():
                totals[key] = totals.get(key, 0.) + float(value.detach())
            steps += 1
            if step % 40 == 0:
                print("epoch={} step={} loss={:.4f}".format(epoch + 1, step, float(loss.detach())), flush=True)
            if debug and step >= 1:
                break
        scheduler.step()
        validation = known_validation(backbone, heads, val_loader, meta, device)
        eligible = epoch + 1 >= warmup + min_synthesis
        for kind in synthesis_epochs:
            if synthesis_cfg.get(kind + "_enabled", True) and synthesis_epochs[kind] < min_synthesis:
                eligible = False
        if debug:
            eligible = True
        key = (validation["leaf_accuracy"], -validation["root_ce"])
        if eligible and (best_key is None or key > best_key):
            best_key, best_epoch, best_validation, bad_epochs = key, epoch + 1, validation, 0
            _save_checkpoint(output / "best.pth", backbone, heads, cfg, meta, best_epoch, validation, input_dim)
        elif eligible:
            bad_epochs += 1
        record = {"epoch": epoch + 1, "loss": {k: v / steps for k, v in totals.items()},
                  "known_validation": validation, "synthesis": pool["stats"], "synthesis_epochs": dict(synthesis_epochs),
                  "novelty_weight": ramp, "checkpoint_eligible": eligible, "best_epoch": best_epoch,
                  "elapsed_seconds": time.monotonic() - start}
        with open(output / "train.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        print(json.dumps(record, allow_nan=False), flush=True)
        patience = int(settings.get("patience", 25))
        if patience > 0 and eligible and bad_epochs >= patience:
            break
    if best_epoch is None:
        raise ValueError("No eligible checkpoint: inspect synthetic support yields; thresholds were not fitted")
    checkpoint = _load_checkpoint(output / "best.pth")
    backbone.load_state_dict(checkpoint["backbone"], strict=True)
    heads.load_state_dict(checkpoint["heads"], strict=True)
    heads.eval()
    h_reference, reference_labels = reference_features(backbone, reference_loader, device)
    with torch.no_grad():
        taxonomy, fine = heads.embeddings(h_reference)
    support = fit_score_support(taxonomy.cpu().numpy(), fine.cpu().numpy(), reference_labels.cpu().numpy(), meta)
    write_json(output / "score_support.json", support)
    require_signature(sig, signature(cfg))
    write_json(output / "completed.json", {"schema_version": 11, "debug": debug, "signature": sig,
               "config": cfg, "meta": meta, "audit": audit, "seed": cfg["seed"], "best_epoch": best_epoch,
               "known_validation": best_validation, "gradient_splits": ["train"], "test_used_for_fitting": False,
               "synthesis_epochs": synthesis_epochs, "near_candidates_kept_by_parent":
               {meta["parent_names"][int(p)]: n for p, n in coverage.items()},
               "checkpoint": {"path": "best.pth", "sha256": file_hash(output / "best.pth")},
               "support": {"path": "score_support.json", "sha256": file_hash(output / "score_support.json")}})
    print("Training finished: " + str(output / "completed.json"), flush=True)


def load_trained(cfg, directory, device):
    training = Path(directory) / "training"
    receipt, path = verify_artifact(training, "completed.json", "checkpoint")
    _, support_path = verify_artifact(training, "completed.json", "support")
    if receipt.get("debug"):
        raise ValueError("Debug runs cannot calibrate or evaluate locked test data")
    require_signature(receipt["signature"], signature(cfg))
    checkpoint = _load_checkpoint(path)
    meta = hierarchy(cfg)
    if checkpoint.get("schema_version") != 11 or checkpoint["meta"] != meta or receipt["meta"] != meta or checkpoint["config"] != cfg:
        raise ValueError("Checkpoint configuration or taxonomy mismatch")
    backbone = make_backbone(cfg, meta, device)
    heads = DCBSHeads(checkpoint["input_dim"], meta["leaf_to_parent"], len(meta["parent_names"]), cfg["dcbs"]).to(device)
    backbone.load_state_dict(checkpoint["backbone"], strict=True)
    heads.load_state_dict(checkpoint["heads"], strict=True)
    backbone.requires_grad_(False).eval()
    heads.requires_grad_(False).eval()
    return backbone, heads, receipt, read_json(support_path)


@torch.no_grad()
def collect(groups, cfg, meta, backbone, heads, support, device):
    backbone.eval()
    heads.eval()
    leaves, _ = encode_texts(backbone, meta)
    result = {}
    for split, rows in groups.items():
        records, seen = [], []
        for images, _, indices in make_loader(rows, cfg, meta):
            h = backbone.encode_image(images.to(device), normalize=True).float()
            output = {k: v.cpu().numpy() for k, v in heads(h).items()}
            logits = (scale_of(backbone) * h @ leaves.T).cpu().numpy()
            seen.extend(indices.tolist())
            records.extend(raw_records([rows[i] for i in indices.tolist()], output, logits, support, meta, cfg["dcbs"]))
        if seen != list(range(len(rows))):
            raise ValueError("Inference loader must visit every manifest row once in order")
        result[split] = records
        print("Scored {}: {} images".format(split, len(records)), flush=True)
    return result


def metrics_for(groups, router, meta):
    from metrics_open import evaluate_open_set
    routed = {split: apply_router(records, router, meta) for split, records in groups.items()}
    statuses = {kind: [r for rows in routed.values() for r in rows if r["status"] == kind] for kind in ("known", "intra", "extra")}
    return routed, evaluate_open_set(statuses["known"], statuses["intra"], statuses["extra"])


def calibrate_run(cfg, directory, device):
    backbone, heads, trained, support = load_trained(cfg, directory, device)
    meta = trained["meta"]
    groups, audit = load_stage_rows(cfg, "calibrate", meta, forbidden_hashes=trained["audit"]["train"]["image_hashes"])
    if audit["val_known"] != trained["audit"]["val_known"]:
        raise ValueError("Validation known data changed since checkpoint selection")
    output = claim_stage(directory, "calibration")
    records = collect(groups, cfg, meta, backbone, heads, support, device)
    write_records(output / "development_scores.jsonl", [r for rows in records.values() for r in rows])
    try:
        router = calibrate(records["val_known"], records["val_intra"], records["val_extra"], meta, cfg["calibration"])
    except ValueError as exc:
        write_json(output / "failed.json", {"reason": str(exc), "test_used_for_fitting": False})
        raise
    router.update(checkpoint_sha256=trained["checkpoint"]["sha256"], support_sha256=trained["support"]["sha256"], signature=trained["signature"])
    routed, metrics = metrics_for(records, router, meta)
    write_json(output / "router.json", router)
    write_json(output / "development_metrics.json", metrics)
    write_records(output / "development_predictions.jsonl", [r for rows in routed.values() for r in rows])
    write_json(output / "completed.json", {"schema_version": 11, "audit": audit, "signature": trained["signature"],
               "checkpoint_sha256": trained["checkpoint"]["sha256"], "test_used_for_fitting": False,
               "fit_splits": list(groups), "router": {"path": "router.json", "sha256": file_hash(output / "router.json")}})
    print("Development router frozen: " + str(output / "router.json"), flush=True)


def evaluate_gates(metrics, settings):
    definitions = {
        "near_correct_fallback_min": (metrics["intra"]["correct_fallback_rate"], "min"),
        "extra_global_rejection_min": (metrics["extra"]["global_unknown_recall"], "min"),
        "extra_false_known_max": (metrics["extra"]["false_known_leaf_rate"], "max"),
        "accepted_leaf_precision_min": (metrics["overall"]["open_world_accepted_leaf_precision"], "min"),
        "known_closed_accuracy_min": (metrics["known"]["global_leaf_accuracy"], "min"),
        "known_e2e_min": (metrics["known"]["end_to_end_leaf_accuracy"], "min"),
    }
    checks = {}
    for key, target in settings.items():
        value, direction = definitions[key]
        passed = value is not None and (value >= target if direction == "min" else value <= target)
        checks[key] = {"value": value, "target": target, "passed": bool(passed)}
    return {"passed": bool(checks) and all(c["passed"] for c in checks.values()), "checks": checks,
            "note": "Predeclared test reporting gates; never used to fit or adjust the router"}


def test_run(cfg, directory, device):
    calibrated, router_path = verify_artifact(Path(directory) / "calibration", "completed.json", "router")
    router = read_json(router_path)
    backbone, heads, trained, support = load_trained(cfg, directory, device)
    require_signature(trained["signature"], calibrated["signature"])
    require_signature(trained["signature"], router["signature"])
    if calibrated["checkpoint_sha256"] != trained["checkpoint"]["sha256"] or router["checkpoint_sha256"] != trained["checkpoint"]["sha256"] or router["support_sha256"] != trained["support"]["sha256"]:
        raise ValueError("Frozen router and training artifacts differ")
    forbidden_hashes = set(trained["audit"]["train"]["image_hashes"])
    forbidden_sources = set()
    for split, audit in calibrated["audit"].items():
        forbidden_hashes.update(audit["image_hashes"])
        if split != "val_known":
            forbidden_sources.update(audit["sources"])
    groups, audit = load_stage_rows(cfg, "test", trained["meta"], forbidden_hashes, forbidden_sources)
    output = claim_stage(directory, "test")
    records = collect(groups, cfg, trained["meta"], backbone, heads, support, device)
    routed, metrics_all = metrics_for(records, router, trained["meta"])
    unique = {}
    for split, rows in routed.items():
        seen = set()
        unique[split] = []
        for row in rows:
            row["evaluation_weight"] = int(row["image_sha256"] not in seen)
            if row["evaluation_weight"]:
                unique[split].append(row)
            seen.add(row["image_sha256"])
    _, metrics = metrics_for(unique, router, trained["meta"])
    gates = evaluate_gates(metrics, cfg.get("evaluation_gates", {}))
    write_records(output / "predictions.jsonl", [r for rows in routed.values() for r in rows])
    write_json(output / "metrics.json", metrics)
    write_json(output / "metrics_all_rows.json", metrics_all)
    write_json(output / "gates.json", gates)
    write_json(output / "completed.json", {"schema_version": 11, "audit": audit, "signature": trained["signature"],
               "checkpoint_sha256": trained["checkpoint"]["sha256"], "router_sha256": calibrated["router"]["sha256"],
               "test_used_for_fitting": False, "metric_unit": "unique_image_sha256", "gate_passed": gates["passed"]})
    print(json.dumps(gates, ensure_ascii=False), flush=True)


def run(stage):
    parser = argparse.ArgumentParser(description="TaxoSafe v11 DCBS: " + stage)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--trial", type=int, default=1)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--variant", choices=VARIANTS, default="main")
    parser.add_argument("--run-dir")
    parser.add_argument("--preflight", action="store_true", help="Audit ONLY this stage's allowed inputs, without CLIP/CUDA")
    if stage == "train":
        parser.add_argument("--debug", action="store_true", help="Two short epochs in a separate, non-evaluable run")
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
        print(json.dumps({"stage": stage, "inputs": {s: {"count": a["count"], "unique_image_count": a["unique_image_count"], "source_count": len(a["sources"])} for s, a in audit.items()}, "signature": signature(cfg)}, indent=2))
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
        elif stage == "test":
            test_run(cfg, directory, torch.device("cuda"))
        else:
            raise ValueError("Unknown stage: " + stage)
