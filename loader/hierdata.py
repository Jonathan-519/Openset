"""Load the locked taxonomy without importing unrelated dataset runners."""
import numpy as np
import torch

from .utils import prepro_node_name


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
