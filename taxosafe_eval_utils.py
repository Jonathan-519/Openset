"""Shared, read-only evaluation utilities for TaxoSafe calibration/testing."""

import hashlib
import json
import os

import torch
import yaml

from engine_taxosafe import score_label_sets
from loader import get_dataloader
from models import get_model
from train_taxosafe import build_hier_meta, set_random_seed


def load_yaml(path):
    path = os.path.abspath(path)
    with open(path, "r", encoding="utf-8") as stream:
        cfg = yaml.load(stream, Loader=yaml.SafeLoader)
    if not isinstance(cfg, dict):
        raise ValueError("The YAML root must be a mapping")
    return cfg, path


def resolve_run_dir(cfg, trial, project_root, explicit_run_dir=None):
    if explicit_run_dir:
        return os.path.abspath(explicit_run_dir)
    return os.path.join(
        project_root,
        "runs",
        cfg["data"]["name"],
        cfg["model"]["arch"],
        cfg["exp"],
        "trial_{}".format(trial),
    )


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def write_jsonl(path, records, drop_vector_fields=False):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        for record in records:
            output = dict(record)
            if drop_vector_fields:
                output.pop("parent_cosine", None)
                output.pop("leaf_cosine", None)
                output.pop("image_feature", None)
            stream.write(json.dumps(output, ensure_ascii=False) + "\n")


def load_model_and_data(cfg, splits, checkpoint_path, device):
    """Load only the caller-provided splits and the frozen best checkpoint."""
    checkpoint_path = os.path.abspath(checkpoint_path)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            "Checkpoint does not exist: {}".format(checkpoint_path)
        )

    set_random_seed(int(cfg.get("seed", 1)))
    eval_batch_size = int(
        cfg["data"].get("eval_batch_size", cfg["data"]["batch_size"])
    )
    loaders = get_dataloader(
        cfg["data"],
        list(splits),
        eval_batch_size,
    )
    missing = set(splits).difference(loaders)
    if missing:
        raise KeyError(
            "Data loader is missing splits: {}".format(sorted(missing))
        )

    param_names = loaders["param_names"]
    leaf_nodes = loaders["leaf_nodes"]
    intnl_nodes = loaders["intnl_nodes"]
    sublabels = loaders["sublabels"].to(device)
    hier_meta = build_hier_meta(
        param_names=param_names,
        leaf_nodes=leaf_nodes,
        intnl_nodes=intnl_nodes,
        sublabels=sublabels,
        device=device,
        expected_num_leaves=cfg["data"].get("num_known_leaves", 17),
    )

    model = get_model(cfg["model"], hier_meta["leaf_names"]).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint must contain a model state dictionary")
    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {
            key[len("module."):]: value
            for key, value in state_dict.items()
        }
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    return model, loaders, hier_meta


def _batch_indices(value, batch_size):
    raw = value[2]
    if torch.is_tensor(raw):
        indices = raw.detach().cpu().reshape(-1).tolist()
    else:
        indices = list(raw)
    if len(indices) != batch_size:
        raise ValueError("Dataset index count does not match batch size")
    return [int(index) for index in indices]


@torch.no_grad()
def collect_score_records(
    model,
    data_loader,
    status,
    hier_meta,
    device,
    include_vectors=False,
    include_image_features=False,
    debug=False,
    unknown_template="novel member of {}",
):
    """Collect parent/branch-restricted child cosine evidence per sample."""
    if status not in {"known", "intra", "extra"}:
        raise ValueError("status must be known, intra or extra")

    records = []
    num_parents = len(hier_meta["parent_names"])
    num_leaves = len(hier_meta["leaf_names"])
    leaf_to_parent = hier_meta["leaf_to_parent"].to(
        device=device,
        dtype=torch.long,
    )
    dataset_paths = data_loader.dataset.data

    for step, value in enumerate(data_loader):
        images = value[0].to(device)
        targets = value[1].to(device=device, dtype=torch.long)
        batch_size = int(targets.numel())
        dataset_indices = _batch_indices(value, batch_size)

        unknown_names = [
            str(unknown_template).format(name)
            for name in hier_meta["parent_names"]
        ]
        scored = score_label_sets(
            model,
            images,
            {
                "parent": hier_meta["parent_names"],
                "leaf": hier_meta["leaf_names"],
                "local_unknown": unknown_names,
            },
            return_representations=include_image_features,
        )
        if include_image_features:
            scores, scale, representations = scored
            image_features = representations["image"]
        else:
            scores, scale = scored
            image_features = None
        safe_scale = scale.detach().clamp_min(1e-8)
        parent_cosine = scores["parent"] / safe_scale
        leaf_cosine = scores["leaf"] / safe_scale
        parent_logits = scores["parent"].float()
        leaf_logits = scores["leaf"].float()
        parent_probabilities = torch.softmax(parent_logits, dim=-1)
        parent_entropy = -(
            parent_probabilities
            * parent_probabilities.clamp_min(1e-12).log()
        ).sum(dim=-1)
        parent_entropy = parent_entropy / torch.log(torch.tensor(
            float(num_parents), device=device
        ))
        parent_top_values = torch.topk(
            parent_cosine, k=min(2, num_parents), dim=-1
        ).values
        if num_parents >= 2:
            parent_margin = parent_top_values[:, 0] - parent_top_values[:, 1]
        else:
            parent_margin = torch.zeros_like(parent_top_values[:, 0])
        pred_parents = parent_cosine.argmax(dim=-1)
        global_pred_leaves = leaf_cosine.argmax(dim=-1)

        for row in range(batch_size):
            raw_target = int(targets[row].item())
            if status == "known":
                if raw_target < 0 or raw_target >= num_leaves:
                    raise ValueError(
                        "val/test_known labels must be known leaf IDs in "
                        "[0, {}], got {}".format(num_leaves - 1, raw_target)
                    )
                true_leaf = raw_target
                true_parent = int(leaf_to_parent[true_leaf].item())
            elif status == "intra":
                if raw_target < 0 or raw_target >= num_parents:
                    raise ValueError(
                        "val/test_intra labels must be parent IDs in "
                        "[0, {}], got {}".format(num_parents - 1, raw_target)
                    )
                true_parent = raw_target
                true_leaf = None
            else:
                if raw_target != -1:
                    raise ValueError(
                        "val/test_extra labels must be -1, got {}".format(
                            raw_target
                        )
                    )
                true_parent = None
                true_leaf = None

            pred_parent = int(pred_parents[row].item())
            children = torch.as_tensor(
                hier_meta["children_by_parent"][pred_parent],
                dtype=torch.long,
                device=device,
            )
            local_scores = leaf_cosine[row, children]
            local_logits = leaf_logits[row, children]
            local_index = int(local_scores.argmax().item())
            pred_leaf = int(children[local_index].item())
            local_probabilities = torch.softmax(local_logits, dim=-1)
            local_unknown_logit = scores["local_unknown"][row, pred_parent]
            local_joint_logits = torch.cat(
                [local_logits, local_unknown_logit.reshape(1)]
            )
            local_joint_probabilities = torch.softmax(
                local_joint_logits.float(), dim=-1
            )
            if int(children.numel()) >= 2:
                child_top_values = torch.topk(
                    local_scores, k=2, dim=-1
                ).values
                child_margin = float(
                    (child_top_values[0] - child_top_values[1]).item()
                )
                child_entropy = -(
                    local_probabilities
                    * local_probabilities.clamp_min(1e-12).log()
                ).sum() / torch.log(torch.tensor(
                    float(children.numel()), device=device
                ))
                child_neg_entropy = float((-child_entropy).item())
            else:
                child_margin = 0.0
                child_neg_entropy = 0.0

            dataset_index = dataset_indices[row]
            record = {
                "status": status,
                "dataset_index": dataset_index,
                "path": str(dataset_paths[dataset_index]),
                "true_parent": true_parent,
                "true_leaf": true_leaf,
                "pred_parent": pred_parent,
                "pred_leaf": pred_leaf,
                "global_pred_leaf": int(global_pred_leaves[row].item()),
                "parent_score": float(
                    parent_cosine[row, pred_parent].item()
                ),
                "parent_margin": float(parent_margin[row].item()),
                "parent_msp": float(
                    parent_probabilities[row, pred_parent].item()
                ),
                "parent_neg_entropy": float((-parent_entropy[row]).item()),
                "parent_logsumexp": float(
                    torch.logsumexp(parent_logits[row], dim=-1).item()
                ),
                "child_score": float(local_scores[local_index].item()),
                "child_margin": child_margin,
                "child_msp": float(local_probabilities[local_index].item()),
                "child_neg_entropy": child_neg_entropy,
                "local_child_logit": float(local_logits[local_index].item()),
                "local_unknown_logit": float(local_unknown_logit.item()),
                "local_known_margin": float(
                    (local_logits[local_index] - local_unknown_logit).item()
                ),
                "local_unknown_probability": float(
                    local_joint_probabilities[-1].item()
                ),
                "logit_scale": float(safe_scale.item()),
            }
            if include_vectors:
                record["parent_cosine"] = [
                    float(value)
                    for value in parent_cosine[row].detach().cpu().tolist()
                ]
                record["leaf_cosine"] = [
                    float(value)
                    for value in leaf_cosine[row].detach().cpu().tolist()
                ]
            if include_image_features:
                record["image_feature"] = [
                    float(value)
                    for value in image_features[row].detach().cpu().tolist()
                ]
            records.append(record)

        if debug:
            break

    if not records:
        raise ValueError("The {} split is empty".format(status))
    return records
