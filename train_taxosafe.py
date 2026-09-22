"""Train TaxoSafe on the known-leaf training split.

This entry point integrates:
    * novelty-aware dynamic tree-cut (NDTL);
    * node consistency learning (NCL);
    * parent/leaf probability consistency;
    * parent-child novelty ranking;
    * optional outlier exposure (OE).

Place this file in the ProTeCt-main project root together with
``engine_taxosafe.py``.
"""

import argparse
import os
import random
import shutil
import sys

import numpy as np
import torch
import yaml
from tensorboardX import SummaryWriter
from tqdm import tqdm


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engine_taxosafe import (
    eval_one_epoch,
    train_one_epoch,
)
from loader import get_dataloader
from models import get_model
from models.open_treecut_generator import OpenTreecutGenerator
from optim.lr_scheduler import build_lr_scheduler
from optim.optimizer import build_optimizer
from utils import get_logger, print_args


EXPECTED_PARENT_NAMES = [
    "Amphipoda",
    "Appendiculata",
    "Cladocera",
    "Copepoda",
    "Euphausiacea",
    "Medusae",
    "Sagittoidea",
]


def build_hier_meta(
    param_names,
    leaf_nodes,
    intnl_nodes,
    sublabels,
    device,
    expected_num_leaves=23,
):
    """Build the two-level metadata consumed by the TaxoSafe engine."""
    if len(intnl_nodes) == 0:
        raise ValueError("intnl_nodes is empty; check tree.npy")

    parent_param_indices = [int(index) for index in intnl_nodes[0]]
    leaf_param_indices = [int(index) for index in leaf_nodes]

    parent_names = [param_names[index] for index in parent_param_indices]
    leaf_names = [param_names[index] for index in leaf_param_indices]
    leaf_to_parent = sublabels[:, 0].to(
        device=device,
        dtype=torch.long,
    )

    if parent_names != EXPECTED_PARENT_NAMES:
        raise ValueError(
            "\nParent names or order are incorrect.\n"
            "Expected: {}\n"
            "Actual:   {}\n"
            "Check that loader/treelibs.py uses sorted(os.listdir(...)), "
            "then rebuild tree.npy.".format(
                EXPECTED_PARENT_NAMES,
                parent_names,
            )
        )

    if expected_num_leaves is not None and len(leaf_names) != int(
        expected_num_leaves
    ):
        raise ValueError(
            "Fold 1 should contain {} known leaves, but tree.npy contains "
            "{}.".format(expected_num_leaves, len(leaf_names))
        )

    if leaf_to_parent.numel() != len(leaf_names):
        raise ValueError(
            "sublabels has {} leaf rows, but tree.npy exposes {} leaves".format(
                leaf_to_parent.numel(), len(leaf_names)
            )
        )
    if torch.any(leaf_to_parent < 0) or torch.any(
        leaf_to_parent >= len(parent_names)
    ):
        raise ValueError(
            "At least one known leaf cannot be mapped to a root child"
        )

    children_by_parent = []
    for parent_id, parent_name in enumerate(parent_names):
        child_ids = torch.where(leaf_to_parent == parent_id)[0]
        if child_ids.numel() == 0:
            raise ValueError(
                "Parent '{}' has no known leaves in tree.npy".format(
                    parent_name
                )
            )
        children_by_parent.append(child_ids)

    return {
        "parent_names": parent_names,
        "leaf_names": leaf_names,
        "parent_param_indices": torch.tensor(
            parent_param_indices,
            dtype=torch.long,
            device=device,
        ),
        "leaf_param_indices": torch.tensor(
            leaf_param_indices,
            dtype=torch.long,
            device=device,
        ),
        "leaf_to_parent": leaf_to_parent,
        "children_by_parent": children_by_parent,
        "num_parents": len(parent_names),
        "num_leaves": len(leaf_names),
    }


def print_hier_meta(hier_meta):
    """Print the exact label-ID mapping used during training."""
    print("\n" + "=" * 70)
    print("TaxoSafe hierarchy metadata")
    print("=" * 70)

    parent_names = hier_meta["parent_names"]
    leaf_names = hier_meta["leaf_names"]
    for parent_id, parent_name in enumerate(parent_names):
        child_ids = (
            hier_meta["children_by_parent"][parent_id]
            .detach()
            .cpu()
            .tolist()
        )
        child_names = [leaf_names[child_id] for child_id in child_ids]
        print("[{}] {}: {}".format(parent_id, parent_name, child_names))

    print("-" * 70)
    print("Number of parents: {}".format(hier_meta["num_parents"]))
    print("Number of known leaves: {}".format(hier_meta["num_leaves"]))
    print("=" * 70 + "\n")


def set_random_seed(seed):
    """Seed Python, NumPy and PyTorch for reproducible fold experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_prompt_model(cfg, hier_meta, device):
    """Create the prompt model with the known leaves as initial labels."""
    if cfg.get("init_label_set") != "leaf":
        raise ValueError(
            "TaxoSafe currently requires init_label_set: leaf"
        )

    model = get_model(
        cfg["model"],
        hier_meta["leaf_names"],
    ).to(device)

    # The original MaPLe wrapper registers token_embedding outside
    # ``self.model`` after freezing the CLIP encoders. Freeze everything here
    # once more and then explicitly enable prompt parameters only. This avoids
    # accidentally optimising token_embedding.weight.
    for name, parameter in model.named_parameters():
        is_prompt = "prompt_learner" in name or "VPT" in name
        parameter.requires_grad_(is_prompt)

    trainable_params = []
    trainable_names = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            trainable_params.append(parameter)
            trainable_names.append(name)

    if not trainable_params:
        raise RuntimeError("The model has no trainable prompt parameters")

    print("Parameters to be updated: {}".format(set(trainable_names)))
    return model, trainable_params


def run_known_evaluation(
    model,
    loader,
    param_names,
    device,
    epoch,
    cfg,
    args,
    leaf_nodes,
    intnl_nodes,
    sublabels,
    hier_meta,
):
    """Call the known-set evaluator with complete hierarchy metadata."""
    return eval_one_epoch(
        model=model,
        data_loader=loader,
        param_names=param_names,
        device=device,
        epoch=epoch,
        cfg=cfg,
        args=args,
        leaf_nodes=leaf_nodes,
        intnl_nodes=intnl_nodes,
        sublabels=sublabels,
        hier_meta=hier_meta,
    )


def save_model_state(model, path):
    """Save a model-only state dict compatible with the existing test code."""
    model_core = model.module if hasattr(model, "module") else model
    torch.save(model_core.state_dict(), path)


def log_meters(split, meters, epoch, writer, logger):
    """Write all scalar metrics to TensorBoard and the text log."""
    logger.info("=== %s epoch %d ===", split, epoch)
    for name, value in meters.items():
        writer.add_scalar(
            "{}/{}".format(split, name),
            value,
            epoch + 1,
        )
        message = "Average {} {}: {}".format(split, name, value)
        print(message)
        logger.info(message)
    logger.info("======")


def main(cfg, args, writer, logger, logdir):
    if not torch.cuda.is_available():
        raise SystemExit("TaxoSafe training requires a CUDA GPU")

    print("CUDA_VISIBLE_DEVICES={}".format(
        os.environ.get("CUDA_VISIBLE_DEVICES", "not set")
    ))
    print_args(args, cfg)
    cfg.setdefault("eval_freq", 10)

    seed = int(cfg.get("seed", 1))
    set_random_seed(seed)
    device = torch.device("cuda")

    # The episodic sampler is built from cfg["data"], so propagate the global
    # seed when no sampler-specific seed was supplied.
    cfg["data"].setdefault("seed", seed)
    sampler_cfg = cfg["data"].setdefault("sampler", {"name": "random"})
    if sampler_cfg.get("name") in ("hierarchical_episode", "nshot"):
        sampler_cfg.setdefault("seed", seed)

    open_cfg = cfg.get("open_treecut", {})
    holdout_mode = str(open_cfg.get("holdout_mode", "epoch")).lower()
    holdout_seed = int(open_cfg.get("holdout_seed", seed))
    if sampler_cfg.get("name") == "hierarchical_episode":
        sampler_cfg.setdefault("pseudo_holdout_mode", holdout_mode)
        sampler_cfg.setdefault("holdout_seed", holdout_seed)
        if str(sampler_cfg["pseudo_holdout_mode"]).lower() != holdout_mode:
            raise ValueError(
                "data.sampler.pseudo_holdout_mode and "
                "open_treecut.holdout_mode must be identical"
            )
        if int(sampler_cfg["holdout_seed"]) != holdout_seed:
            raise ValueError(
                "data.sampler.holdout_seed and open_treecut.holdout_seed "
                "must be identical"
            )


    lambda_oe = float(
        cfg.get("loss", {}).get(
            "lambda_oe",
            0.0,
        )
    )
    
    # Training/checkpoint selection must not consume any unknown split.
    # Near-Unknown Dev and Global-OOD Dev are reserved for post-training
    # calibration only.
    splits = [
        "train",
        "val_known",
    ]
    
    if lambda_oe > 0.0:
        splits.append("oe_train")
    
    data_loader = get_dataloader(
        cfg["data"],
        splits,
        cfg["data"]["batch_size"],
    )
    
    required_loader_keys = {
        "train",
        "val_known",
        "param_names",
        "leaf_nodes",
        "intnl_nodes",
        "sublabels",
    }
    
    if lambda_oe > 0.0:
        required_loader_keys.add("oe_train")
        
    
    
    missing_loader_keys = required_loader_keys.difference(data_loader)
    if missing_loader_keys:
        raise KeyError(
            "Data loader is missing keys: {}".format(
                sorted(missing_loader_keys)
            )
        )

    if lambda_oe > 0.0 and "oe_train" not in data_loader:
        raise ValueError(
            "loss.lambda_oe is positive, but oe_train was not loaded. "
            "Check data.oe_train/data.ood_root, or temporarily set "
            "lambda_oe: 0.0."
        )

    if lambda_oe > 0.0:
        oe_count = len(data_loader["oe_train"].dataset)
        minimum_oe = int(
            cfg.get("loss", {}).get("min_oe_train_samples", 100)
        )
        allow_small_oe = bool(
            cfg.get("loss", {}).get("allow_small_oe", False)
        )
        if oe_count < minimum_oe and not allow_small_oe:
            raise ValueError(
                "OE training is enabled, but oe_train contains only {} "
                "images (minimum {}). Add diverse hard-OOD training data, "
                "or explicitly set loss.allow_small_oe: true for an "
                "ablation only.".format(oe_count, minimum_oe)
            )

    param_names = data_loader["param_names"]
    leaf_nodes = data_loader["leaf_nodes"]
    intnl_nodes = data_loader["intnl_nodes"]
    sublabels = data_loader["sublabels"].to(device)

    expected_num_leaves = cfg["data"].get("num_known_leaves", 17)
    hier_meta = build_hier_meta(
        param_names=param_names,
        leaf_nodes=leaf_nodes,
        intnl_nodes=intnl_nodes,
        sublabels=sublabels,
        device=device,
        expected_num_leaves=expected_num_leaves,
    )
    print_hier_meta(hier_meta)

    open_treecut = OpenTreecutGenerator(
        parent_names=hier_meta["parent_names"],
        leaf_names=hier_meta["leaf_names"],
        children_by_parent=hier_meta["children_by_parent"],
        collapse_prob=open_cfg.get("collapse_prob", 0.30),
        hide_prob=open_cfg.get("hide_prob", 0.50),
        unknown_template=open_cfg.get(
            "unknown_template",
            "novel member of {}",
        ),
        force_one_pseudo_per_batch=open_cfg.get(
            "force_one_pseudo_per_batch",
            False,
        ),
        holdout_mode=holdout_mode,
        holdout_seed=holdout_seed,
    )

    model, trainable_params = build_prompt_model(cfg, hier_meta, device)
    optimizer = build_optimizer(trainable_params, cfg["optim"])
    scheduler = build_lr_scheduler(optimizer, cfg["optim"])
    use_scheduler = bool(cfg["optim"].get("use_scheduler", False))

    best_result = -float("inf")
    best_epoch = None
    evaluations_without_improvement = 0
    best_path = os.path.join(logdir, "ckpt", "best.pth")
    last_path = os.path.join(logdir, "ckpt", "last.pth")

    max_epoch = int(cfg["optim"]["max_epoch"])
    eval_freq = int(cfg["eval_freq"])
    early_stopping_patience = int(
        cfg["optim"].get("early_stopping_patience", 0)
    )
    early_stopping_min_delta = float(
        cfg["optim"].get("early_stopping_min_delta", 0.0)
    )
    if early_stopping_patience < 0:
        raise ValueError("optim.early_stopping_patience must be >= 0")
    if eval_freq <= 0:
        raise ValueError(
            "TaxoSafe requires eval_freq > 0 because best.pth must be "
            "selected only on val_known"
        )

    for epoch in tqdm(range(max_epoch)):
        # The sampler and tree-cut generator share the same deterministic
        # epoch-level holdout schedule.  Setting both explicitly prevents a
        # leaf from switching between known and pseudo-unseen within an epoch.
        train_batch_sampler = data_loader.get("train_batch_sampler")
        if train_batch_sampler is not None and hasattr(
            train_batch_sampler, "set_epoch"
        ):
            train_batch_sampler.set_epoch(epoch)
        open_treecut.set_epoch(epoch)

        # Pass sched=None so engine_taxosafe.py does not advance the scheduler
        # per batch. Its T_max/max_epoch configuration is epoch-based.
        train_meters = train_one_epoch(
            model=model,
            optimizer=optimizer,
            sched=None,
            data_loader=data_loader["train"],
            param_names=param_names,
            device=device,
            epoch=epoch,
            cfg=cfg,
            args=args,
            treecut_generator=None,
            leaf_nodes=leaf_nodes,
            intnl_nodes=intnl_nodes,
            sublabels=sublabels,
            hier_meta=hier_meta,
            open_treecut=open_treecut,
            oe_data_loader=data_loader.get("oe_train"),
        )

        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("train/lr", current_lr, epoch + 1)
        log_meters("train", train_meters, epoch, writer, logger)

        # ProTeCt's scheduler is parameterised in epochs. Advance exactly once
        # after completing all episodic batches of this epoch.
        if use_scheduler and scheduler is not None:
            scheduler.step()

        should_evaluate = (
            (epoch + 1) % eval_freq == 0
            or (epoch + 1) == max_epoch
            or args.debug
        )
        if should_evaluate:
            val_meters = run_known_evaluation(
                model=model,
                loader=data_loader["val_known"],
                param_names=param_names,
                device=device,
                epoch=epoch,
                cfg=cfg,
                args=args,
                leaf_nodes=leaf_nodes,
                intnl_nodes=intnl_nodes,
                sublabels=sublabels,
                hier_meta=hier_meta,
            )
            log_meters("val", val_meters, epoch, writer, logger)

            # Leakage-free checkpoint selection: use Known validation only.
            # Near-Unknown Dev and Global-OOD Dev are intentionally not loaded
            # during training; they are used only after best.pth is frozen.
            result = float(val_meters["leaf_acc"])
            message = "Best result: {}; epoch result: {}".format(
                best_result,
                result,
            )
            print(message)
            logger.info(message)

            improved = result > best_result + early_stopping_min_delta
            if improved:
                best_result = result
                best_epoch = epoch
                evaluations_without_improvement = 0
                print("Saving the best model")
                save_model_state(model, best_path)
            else:
                evaluations_without_improvement += 1

        save_model_state(model, last_path)
        logger.info("======")

        if args.debug:
            break

        if (
            early_stopping_patience > 0
            and should_evaluate
            and evaluations_without_improvement >= early_stopping_patience
        ):
            message = (
                "Early stopping at epoch {}: no val_known improvement for "
                "{} evaluations; best epoch was {} with score {:.4f}."
                .format(
                    epoch + 1,
                    evaluations_without_improvement,
                    None if best_epoch is None else best_epoch + 1,
                    best_result,
                )
            )
            print(message)
            logger.info(message)
            break

    if not os.path.isfile(best_path):
        raise RuntimeError(
            "best.pth was not created; check val_known and eval_freq"
        )

    logger.info("Loading best checkpoint: %s", best_path)
    state_dict = torch.load(best_path, map_location=device)
    model_core = model.module if hasattr(model, "module") else model
    model_core.load_state_dict(state_dict)

    message = (
        "Training finished; best epoch={} and val_known leaf accuracy={:.4f}. "
        "best.pth was reloaded. No Near-Unknown, Global-OOD, or locked test "
        "split was used for checkpoint selection. Run post-training "
        "calibration only after this checkpoint is frozen."
    ).format(None if best_epoch is None else best_epoch + 1, best_result)
    print(message)
    logger.info(message)


def parse_args():
    parser = argparse.ArgumentParser(description="Train TaxoSafe")
    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="YAML configuration file",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run one training/evaluation batch",
    )
    parser.add_argument(
        "--trial",
        type=str,
        default="1",
        help="Experiment trial identifier",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override model, sampler and pseudo-holdout seeds together",
    )
    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = parse_args()

    # Resolve the config before changing cwd. Afterwards, all relative dataset
    # paths in the YAML are consistently resolved from ProTeCt-main.
    config_path = os.path.abspath(parsed_args.config)
    with open(config_path, "r", encoding="utf-8") as file_pointer:
        config = yaml.load(file_pointer, Loader=yaml.SafeLoader)
    if not isinstance(config, dict):
        raise ValueError("The YAML root must be a mapping")
    if parsed_args.seed is not None:
        seed_override = int(parsed_args.seed)
        config["seed"] = seed_override
        config["data"]["seed"] = seed_override
        config["data"].setdefault("sampler", {})["seed"] = seed_override
        config["data"]["sampler"]["holdout_seed"] = seed_override
        config.setdefault("open_treecut", {})[
            "holdout_seed"
        ] = seed_override

    os.chdir(PROJECT_ROOT)
    parsed_args.config = config_path

    output_root = "runs/debug" if parsed_args.debug else "runs"
    run_directory = os.path.join(
        output_root,
        config["data"]["name"],
        config["model"]["arch"],
        config["exp"],
        "trial_{}".format(parsed_args.trial),
    )
    os.makedirs(os.path.join(run_directory, "ckpt"), exist_ok=True)

    summary_writer = SummaryWriter(log_dir=run_directory)
    print("RUNDIR: {}".format(run_directory))
    shutil.copy2(config_path, run_directory)

    run_logger = get_logger(run_directory)
    run_logger.info("Start logging")
    try:
        main(
            cfg=config,
            args=parsed_args,
            writer=summary_writer,
            logger=run_logger,
            logdir=run_directory,
        )
    finally:
        summary_writer.close()
