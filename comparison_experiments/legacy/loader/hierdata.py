"""Hierarchical data loader with TaxoSafe episodic-batch support."""

import logging
import os
from collections import Counter

import numpy as np
import torch
import torch.utils.data as data

from .collate import get_collate_fn
from .hierarchical_episode_sampler import HierarchicalEpisodeBatchSampler
from .img_flist import ImageFilelist
from .sampler import get_sampler
from .transforms import get_transform
from .utils import prepro_node_name


logger = logging.getLogger("mylogger")


def _load_hierarchy(cfg):
    tree = np.load(cfg["hierarchy"], allow_pickle=True).tolist()
    tree.gen_param_lists()

    n_intnl_node = len(tree.intnl_nodes)
    intnl_nodes = [
        tree.nodes.get(tree.intnl_nodes[i]).param_list
        for i in range(n_intnl_node)
    ]
    n_leaf_node = len(tree.leaf_nodes)
    leaf_nodes = np.asarray(
        [
            tree.get_nodeId(tree.leaf_nodes[i]) - 1
            for i in range(n_leaf_node)
        ]
    )
    nodes = sorted(
        [node for node in tree.nodes.values()],
        key=lambda node: node.node_id,
    )[1:]
    param_names = [prepro_node_name(node.name) for node in nodes]

    tree.gen_codewords("class")
    codewords = np.asarray([node.codeword for node in nodes])
    tree.gen_dependence()

    masks = -np.ones((n_intnl_node, len(param_names)))
    intnl_params = np.asarray(
        [
            tree.get_nodeId(tree.intnl_nodes[i]) - 1
            for i in range(n_intnl_node)
        ]
    )
    for i in range(n_intnl_node):
        leaf_idx = tree.sublabels[:, i] >= 0
        masks[i, leaf_nodes[leaf_idx]] = 1
        intnl_idx = tree.dependence[:, i] == 1
        masks[i, intnl_params[intnl_idx]] = 1
        intnl_idx = tree.dependence[i, 1:] == 1
        masks[i, intnl_params[1:][intnl_idx]] = 0

    return {
        "tree_info": {
            "dependence": torch.from_numpy(tree.dependence).float(),
            "masks": torch.from_numpy(masks).int(),
            "codewords": torch.from_numpy(codewords).int(),
        },
        "sublabels": torch.from_numpy(tree.sublabels).float(),
        "intnl_nodes": intnl_nodes,
        "leaf_nodes": leaf_nodes,
        "param_names": param_names,
        "leaf_to_parent": torch.from_numpy(
            tree.sublabels[:, 0]
        ).long(),
    }


def HierDataLoader(cfg, splits, batch_size):
    """Build split loaders and expose semantic hierarchy information."""
    data_root = cfg.get("data_root", "/path/to/dataset")
    if not os.path.isdir(data_root):
        raise FileNotFoundError("{} does not exist".format(data_root))

    hierarchy = _load_hierarchy(cfg)
    num_workers = int(cfg.get("n_workers", 4))
    sampler_cfg = cfg.get("sampler", {"name": "random"})
    sampler_name = sampler_cfg.get("name", "random")
    output = {}

    for split in splits:
        split_batch_size = (
            int(batch_size)
            if "train" in split
            else int(cfg.get("eval_batch_size", 128))
        )
        is_train = "train" in split
        transform = get_transform(
            cfg.get("transform", "imagenet"), is_train
        )
        data_list = cfg.get(split)
        if not data_list or not os.path.isfile(data_list):
            raise FileNotFoundError(
                "Data list for split '{}' is not available: {}".format(
                    split, data_list
                )
            )

        dataset = ImageFilelist(
            root_dir=data_root,
            flist=data_list,
            transform=transform,
        )
        if is_train:
            counter = Counter(int(x) for x in dataset.target)
            n_class = max(counter.keys()) + 1 if counter else 0
            output["cls_num_list"] = np.asarray(
                [counter.get(i, 1e-7) for i in range(n_class)]
            )

        collate_fn = get_collate_fn(
            cfg.get("rot", False) if is_train else False
        )

        if is_train and sampler_name == "hierarchical_episode":
            params = dict(sampler_cfg)
            params.pop("name", None)
            params.setdefault("seed", cfg.get("seed", 1))
            params.setdefault(
                "leaf_names",
                [
                    hierarchy["param_names"][int(index)]
                    for index in hierarchy["leaf_nodes"]
                ],
            )
            batch_sampler = HierarchicalEpisodeBatchSampler(
                dataset=dataset,
                leaf_to_parent=hierarchy["leaf_to_parent"],
                **params
            )
            if split_batch_size != batch_sampler.batch_size:
                raise ValueError(
                    "data.batch_size={} but hierarchical episode produces "
                    "{} samples (parents_per_batch * species_per_parent * "
                    "images_per_species)".format(
                        split_batch_size, batch_sampler.batch_size
                    )
                )
            output[split] = data.DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                collate_fn=collate_fn,
                pin_memory=True,
                num_workers=num_workers,
            )
            output["{}_batch_sampler".format(split)] = batch_sampler
        elif is_train and sampler_name != "random":
            sampler = get_sampler(dataset, sampler_cfg)
            output[split] = data.DataLoader(
                dataset,
                batch_size=split_batch_size,
                sampler=sampler,
                shuffle=False,
                collate_fn=collate_fn,
                drop_last=bool(cfg.get("drop_last", False)),
                pin_memory=True,
                num_workers=num_workers,
            )
        else:
            output[split] = data.DataLoader(
                dataset,
                batch_size=split_batch_size,
                sampler=None,
                shuffle=is_train,
                collate_fn=collate_fn,
                drop_last=(bool(cfg.get("drop_last", False)) if is_train else False),
                pin_memory=True,
                num_workers=num_workers,
            )

        logger.info("%s: %d", split, len(dataset))

    output.update(
        {
            "tree_info": hierarchy["tree_info"],
            "sublabels": hierarchy["sublabels"],
            "intnl_nodes": hierarchy["intnl_nodes"],
            "leaf_nodes": hierarchy["leaf_nodes"],
            "param_names": hierarchy["param_names"],
        }
    )
    logger.info("Building data loader with %d workers", num_workers)
    return output
