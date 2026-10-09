"""Training and evaluation loops for TaxoSafe.

This module keeps the public call style of the original ProTeCt engine, but
computes parent, leaf and novelty-aware tree-cut logits from one image encoding.
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm

from metrics import averageMeter
from losses.taxosafe_loss import compute_taxosafe_loss, morphology_pool


def _model_core(model):
    """Return the actual prompt model when DataParallel is used."""
    return model.module if hasattr(model, "module") else model


def score_label_sets(
    model, image, label_sets, return_representations=False
):
    """Encode images once and score one or more text-label sets.

    Gradients must remain enabled here because MaPLe's visual and textual
    prompts are trainable, even though the CLIP backbone is frozen.
    """
    model_core = _model_core(model)
    spatial_features = None
    if return_representations:
        image_features, spatial_features = (
            model_core.encode_image_with_spatial(image, normalize=True)
        )
    else:
        image_features = model_core.encode_image(image, normalize=True)
    scale = model_core.model.logit_scale.exp().float()

    scores = {}
    text_by_set = {}
    for key, label_names in label_sets.items():
        if not isinstance(label_names, (list, tuple)):
            raise TypeError(
                "{} label names must be a list or tuple, got {}".format(
                    key, type(label_names)
                )
            )
        if len(label_names) == 0:
            raise ValueError("{} label set is empty".format(key))

        text_features = model_core.encode_text(
            list(label_names), normalize=True
        )
        text_by_set[key] = text_features
        # Carry out the small similarity products in fp32 for stable KL/margins.
        scores[key] = (
            scale
            * image_features.float()
            @ text_features.float().t()
        )

    if return_representations:
        return scores, scale, {
            "image": image_features,
            "spatial": spatial_features,
            "text": text_by_set,
        }
    return scores, scale


def _resolve_hier_meta(
    hier_meta,
    param_names,
    leaf_nodes,
    intnl_nodes,
    sublabels,
    device,
):
    """Use supplied metadata or reconstruct it from HierDataLoader outputs."""
    if hier_meta is not None:
        required = {
            "parent_names",
            "leaf_names",
            "leaf_to_parent",
            "children_by_parent",
        }
        missing = required.difference(hier_meta.keys())
        if missing:
            raise KeyError(
                "hier_meta is missing keys: {}".format(sorted(missing))
            )
        hier_meta["leaf_to_parent"] = hier_meta["leaf_to_parent"].to(
            device=device, dtype=torch.long
        )
        hier_meta["children_by_parent"] = [
            torch.as_tensor(x, dtype=torch.long, device=device)
            for x in hier_meta["children_by_parent"]
        ]
        hier_meta.setdefault("num_parents", len(hier_meta["parent_names"]))
        hier_meta.setdefault("num_leaves", len(hier_meta["leaf_names"]))
        return hier_meta

    if any(
        x is None
        for x in (param_names, leaf_nodes, intnl_nodes, sublabels)
    ):
        raise ValueError(
            "TaxoSafe needs hier_meta, or all of param_names/leaf_nodes/"
            "intnl_nodes/sublabels"
        )
    if len(intnl_nodes) == 0:
        raise ValueError("intnl_nodes is empty")

    parent_names = [param_names[int(i)] for i in intnl_nodes[0]]
    leaf_names = [param_names[int(i)] for i in leaf_nodes]
    leaf_to_parent = sublabels[:, 0].to(device=device, dtype=torch.long)
    children_by_parent = [
        torch.where(leaf_to_parent == parent_id)[0]
        for parent_id in range(len(parent_names))
    ]

    return {
        "parent_names": parent_names,
        "leaf_names": leaf_names,
        "leaf_to_parent": leaf_to_parent,
        "children_by_parent": children_by_parent,
        "num_parents": len(parent_names),
        "num_leaves": len(leaf_names),
    }


@torch.no_grad()
def _sample_builtin_open_treecut(
    leaf_target,
    parent_target,
    hier_meta,
    cfg,
):
    """Sample a two-level novelty-aware tree cut for the current batch.

    For each parent branch, the sampler either collapses the branch to its
    parent label or expands it to known leaves. On an expanded branch, one leaf
    that occurs in the current batch may be hidden and mapped to a semantic
    ``novel member of <parent>`` label.
    """
    open_cfg = cfg.get("open_treecut", {})
    collapse_prob = float(open_cfg.get("collapse_prob", 0.3))
    hide_prob = float(open_cfg.get("hide_prob", 0.5))
    unknown_template = open_cfg.get(
        "unknown_template", "novel member of {}"
    )
    force_one_pseudo = bool(
        open_cfg.get("force_one_pseudo_per_batch", False)
    )

    if not 0.0 <= collapse_prob <= 1.0:
        raise ValueError("collapse_prob must be in [0, 1]")
    if not 0.0 <= hide_prob <= 1.0:
        raise ValueError("hide_prob must be in [0, 1]")

    device = leaf_target.device
    batch_size = leaf_target.numel()
    label_names = []
    open_target = torch.full(
        (batch_size,), -1, dtype=torch.long, device=device
    )
    pseudo_mask = torch.zeros(
        batch_size, dtype=torch.bool, device=device
    )
    hidden_leaf_by_parent = {}
    forced_parent = None
    if force_one_pseudo:
        eligible = []
        for parent_id, children in enumerate(
            hier_meta["children_by_parent"]
        ):
            present = torch.unique(
                leaf_target[parent_target == parent_id]
            )
            if torch.as_tensor(children).numel() >= 2 and present.numel() >= 2:
                eligible.append(parent_id)
        if not eligible:
            raise RuntimeError(
                "No rankable parent exists although "
                "force_one_pseudo_per_batch is enabled"
            )
        forced_parent = eligible[int(torch.randint(
            0, len(eligible), (1,), device=device
        ).item())]

    for parent_id, parent_name in enumerate(hier_meta["parent_names"]):
        sample_mask = parent_target == parent_id
        collapse = False if parent_id == forced_parent else (
            torch.rand((), device=device).item() < collapse_prob
        )

        if collapse:
            open_label = len(label_names)
            label_names.append(parent_name)
            open_target[sample_mask] = open_label
            hidden_leaf_by_parent[parent_id] = None
            continue

        child_ids = torch.as_tensor(
            hier_meta["children_by_parent"][parent_id],
            dtype=torch.long,
            device=device,
        )
        if child_ids.numel() == 0:
            raise ValueError(
                "Parent {} has no known children".format(parent_name)
            )

        hidden_leaf = None
        present_leaf_ids = torch.unique(leaf_target[sample_mask])
        can_hide = (
            child_ids.numel() >= 2
            and present_leaf_ids.numel() > 0
            and (
                parent_id == forced_parent
                or torch.rand((), device=device).item() < hide_prob
            )
        )
        if can_hide:
            selected = torch.randint(
                low=0,
                high=present_leaf_ids.numel(),
                size=(1,),
                device=device,
            )
            hidden_leaf = int(present_leaf_ids[selected].item())

        leaf_to_open_label = {}
        for child_id_tensor in child_ids:
            child_id = int(child_id_tensor.item())
            if hidden_leaf is not None and child_id == hidden_leaf:
                continue
            leaf_to_open_label[child_id] = len(label_names)
            label_names.append(hier_meta["leaf_names"][child_id])

        if hidden_leaf is not None:
            unknown_label = len(label_names)
            label_names.append(unknown_template.format(parent_name))
            hidden_mask = sample_mask & (leaf_target == hidden_leaf)
            open_target[hidden_mask] = unknown_label
            pseudo_mask[hidden_mask] = True

        for child_id, open_label in leaf_to_open_label.items():
            child_mask = sample_mask & (leaf_target == child_id)
            open_target[child_mask] = open_label

        hidden_leaf_by_parent[parent_id] = hidden_leaf

    if torch.any(open_target < 0):
        bad = torch.where(open_target < 0)[0].detach().cpu().tolist()
        raise RuntimeError(
            "Open treecut did not assign targets for batch indices {}".format(
                bad
            )
        )
    if force_one_pseudo and not torch.any(pseudo_mask):
        raise RuntimeError("Forced pseudo-unseen sampling failed")

    return {
        "label_names": label_names,
        "target": open_target,
        "pseudo_mask": pseudo_mask,
        "hidden_leaf_by_parent": hidden_leaf_by_parent,
        # Canonical name consumed by losses/taxosafe_loss.py.
        "hidden_by_parent": hidden_leaf_by_parent,
    }


def _validate_open_cut(open_cut, batch_size, device):
    required = {"label_names", "target", "pseudo_mask"}
    missing = required.difference(open_cut.keys())
    if missing:
        raise KeyError("open_cut is missing keys: {}".format(sorted(missing)))

    label_names = list(open_cut["label_names"])
    if len(label_names) == 0:
        raise ValueError("open_cut label_names is empty")

    open_target = torch.as_tensor(
        open_cut["target"], dtype=torch.long, device=device
    )
    pseudo_mask = torch.as_tensor(
        open_cut["pseudo_mask"], dtype=torch.bool, device=device
    )
    if open_target.shape != (batch_size,):
        raise ValueError(
            "open target shape {} != ({},)".format(
                tuple(open_target.shape), batch_size
            )
        )
    if pseudo_mask.shape != (batch_size,):
        raise ValueError(
            "pseudo_mask shape {} != ({},)".format(
                tuple(pseudo_mask.shape), batch_size
            )
        )
    if torch.any(open_target < 0) or torch.any(open_target >= len(label_names)):
        raise ValueError("open target contains an invalid label index")
    return label_names, open_target, pseudo_mask



def _real_intra_unknown_loss(model, value, device, cfg, hier_meta):
    """Supervise development near-unknowns without exposing locked test taxa.

    Each sample carries its true parent ID. The loss keeps the sample inside
    that parent while teaching the explicit local-unknown prompt to outrank
    every known child in the branch.
    """
    image = value[0].to(device)
    parent_target = value[1].to(device=device, dtype=torch.long)
    if torch.any(parent_target < 0) or torch.any(
        parent_target >= hier_meta["num_parents"]
    ):
        raise ValueError(
            "train_intra labels must be parent IDs in [0, {}]".format(
                hier_meta["num_parents"] - 1
            )
        )

    unknown_names = [
        cfg.get("open_treecut", {}).get(
            "unknown_template", "novel member of {}"
        ).format(parent_name)
        for parent_name in hier_meta["parent_names"]
    ]
    scores, _ = score_label_sets(
        model,
        image,
        {
            "parent": hier_meta["parent_names"],
            "leaf": hier_meta["leaf_names"],
            "local_unknown": unknown_names,
        },
    )
    parent_loss = F.cross_entropy(scores["parent"], parent_target)
    local_terms, margin_terms = [], []
    margin = float(cfg.get("loss", {}).get("real_intra_margin", 0.15))
    for parent_id, children in enumerate(hier_meta["children_by_parent"]):
        mask = parent_target == parent_id
        if not torch.any(mask):
            continue
        active = torch.as_tensor(
            children, dtype=torch.long, device=device
        )
        child_logits = scores["leaf"][mask][:, active]
        unknown_logit = scores["local_unknown"][
            mask, parent_id : parent_id + 1
        ]
        joint = torch.cat([child_logits, unknown_logit], dim=1)
        target = torch.full(
            (joint.shape[0],),
            child_logits.shape[1],
            dtype=torch.long,
            device=device,
        )
        local_terms.append(F.cross_entropy(joint, target))
        strongest_child = child_logits.max(dim=1).values
        margin_terms.append(
            F.relu(margin + strongest_child - unknown_logit[:, 0]).mean()
        )

    zero = scores["parent"].sum() * 0.0
    local_loss = torch.stack(local_terms).mean() if local_terms else zero
    margin_loss = torch.stack(margin_terms).mean() if margin_terms else zero
    total = (
        float(cfg.get("loss", {}).get("lambda_real_intra_parent", 0.25))
        * parent_loss
        + float(cfg.get("loss", {}).get("lambda_real_intra_local", 1.0))
        * local_loss
        + float(cfg.get("loss", {}).get("lambda_real_intra_margin", 0.5))
        * margin_loss
    )
    with torch.no_grad():
        parent_acc = (
            scores["parent"].argmax(dim=1).eq(parent_target).float().mean()
            * 100.0
        )
    return {
        "loss": total,
        "parent_loss": parent_loss,
        "local_loss": local_loss,
        "margin_loss": margin_loss,
        "parent_acc": parent_acc,
        "batch_size": int(parent_target.numel()),
    }


def train_one_epoch(
    model,
    optimizer,
    sched,
    data_loader,
    param_names,
    device,
    epoch,
    cfg,
    args,
    treecut_generator=None,
    leaf_nodes=None,
    intnl_nodes=None,
    sublabels=None,
    hier_meta=None,
    open_treecut=None,
    oe_data_loader=None,
    intra_data_loader=None,
):
    """Train one TaxoSafe epoch on known leaf images."""
    del treecut_generator  # retained only for call compatibility

    hier_meta = _resolve_hier_meta(
        hier_meta,
        param_names,
        leaf_nodes,
        intnl_nodes,
        sublabels,
        device,
    )

    meter = {
        "loss": averageMeter(),
        "acc": averageMeter(),
        "loss_ndtl": averageMeter(),
        "loss_ncl": averageMeter(),
        "loss_cons": averageMeter(),
        "loss_rank": averageMeter(),
        "loss_child_known": averageMeter(),
        "loss_child_novel": averageMeter(),
        "loss_pceg": averageMeter(),
        "rank_branch_count": averageMeter(),
        "loss_root": averageMeter(),
        "loss_oe": averageMeter(),
        "loss_local_unknown": averageMeter(),
        "loss_tax_contrast": averageMeter(),
        "loss_sibling_boundary": averageMeter(),
        "loss_real_intra": averageMeter(),
        "loss_real_intra_parent": averageMeter(),
        "loss_real_intra_local": averageMeter(),
        "loss_real_intra_margin": averageMeter(),
        "real_intra_parent_acc": averageMeter(),
        "sibling_boundary_pair_count": averageMeter(),
        "sibling_boundary_margin": averageMeter(),
        "root_known_score": averageMeter(),
        "known_child_score": averageMeter(),
        "pseudo_child_score": averageMeter(),
        "pseudo_pceg": averageMeter(),
        "local_known_margin": averageMeter(),
        "local_pseudo_margin": averageMeter(),
        "novelty_weight_scale": averageMeter(),
        "parent_acc": averageMeter(),
        "leaf_acc": averageMeter(),
        "open_acc": averageMeter(),
        "consistency": averageMeter(),
        "pseudo_rate": averageMeter(),
    }

    print_freq = cfg.get("print_freq", 20)
    model.train()
    lambda_oe = float(cfg.get("loss", {}).get("lambda_oe", 0.0))
    lambda_real_intra = float(
        cfg.get("loss", {}).get("lambda_real_intra", 0.0)
    )
    if lambda_oe > 0.0 and oe_data_loader is None:
        raise ValueError(
            "loss.lambda_oe is positive, but train_one_epoch did not receive "
            "oe_data_loader. Pass data_loader.get('oe_train'), or set "
            "lambda_oe: 0 until OE data is ready."
        )
    # Do not encode OE batches during ablations with lambda_oe == 0.
    oe_iterator = (
        iter(oe_data_loader)
        if lambda_oe > 0.0 and oe_data_loader is not None
        else None
    )
    intra_iterator = (
        iter(intra_data_loader)
        if lambda_real_intra > 0.0 and intra_data_loader is not None
        else None
    )
    if lambda_real_intra > 0.0 and intra_iterator is None:
        raise ValueError(
            "loss.lambda_real_intra is positive, but train_intra was not loaded"
        )
    if oe_iterator is not None:
        meter["root_oe_score"] = averageMeter()
        meter["leaf_oe_score"] = averageMeter()
        meter["root_oe_confidence"] = averageMeter()
        meter["oe_weight_scale"] = averageMeter()

    for step, value in tqdm(enumerate(data_loader), total=len(data_loader)):
        image = value[0].to(device)
        leaf_target = value[1].to(device=device, dtype=torch.long)
        batch_size = leaf_target.shape[0]

        if torch.any(leaf_target < 0) or torch.any(
            leaf_target >= hier_meta["num_leaves"]
        ):
            raise ValueError(
                "The TaxoSafe training loader must contain known leaf labels "
                "in [0, {}]".format(hier_meta["num_leaves"] - 1)
            )

        parent_target = hier_meta["leaf_to_parent"][leaf_target]

        if open_treecut is None:
            open_cut = _sample_builtin_open_treecut(
                leaf_target,
                parent_target,
                hier_meta,
                cfg,
            )
        else:
            open_cut = open_treecut.sample(
                leaf_target=leaf_target,
                parent_target=parent_target,
            )

        open_names, open_target, pseudo_mask = _validate_open_cut(
            open_cut, batch_size, device
        )

        unknown_names = [
            cfg.get("open_treecut", {}).get(
                "unknown_template", "novel member of {}"
            ).format(parent_name)
            for parent_name in hier_meta["parent_names"]
        ]
        scores, scale, representations = score_label_sets(
            model,
            image,
            {
                "parent": hier_meta["parent_names"],
                "leaf": hier_meta["leaf_names"],
                "open": open_names,
                "local_unknown": unknown_names,
            },
            return_representations=True,
        )
        parent_logits = scores["parent"]
        leaf_logits = scores["leaf"]
        open_logits = scores["open"]
        unknown_logits = scores["local_unknown"]
        morphology_features = morphology_pool(
            spatial_features=representations["spatial"],
            leaf_text_features=representations["text"]["leaf"],
            parent_target=parent_target,
            hier_meta=hier_meta,
            attention_temperature=cfg.get("loss", {}).get(
                "morphology_attention_temperature", 0.10
            ),
        )

        expected_parent_shape = (batch_size, hier_meta["num_parents"])
        expected_leaf_shape = (batch_size, hier_meta["num_leaves"])
        if tuple(parent_logits.shape) != expected_parent_shape:
            raise ValueError(
                "parent logits shape {} != {}".format(
                    tuple(parent_logits.shape), expected_parent_shape
                )
            )
        if tuple(leaf_logits.shape) != expected_leaf_shape:
            raise ValueError(
                "leaf logits shape {} != {}".format(
                    tuple(leaf_logits.shape), expected_leaf_shape
                )
            )
        if tuple(open_logits.shape) != (batch_size, len(open_names)):
            raise ValueError(
                "open logits shape {} is inconsistent with {} labels".format(
                    tuple(open_logits.shape), len(open_names)
                )
            )

        oe_parent_logits = None
        oe_leaf_logits = None
        oe_scale = None
        if oe_iterator is not None:
            try:
                oe_value = next(oe_iterator)
            except StopIteration:
                oe_iterator = iter(oe_data_loader)
                oe_value = next(oe_iterator)
            oe_image = oe_value[0].to(device)
            oe_scores, oe_scale = score_label_sets(
                model,
                oe_image,
                {
                    "parent": hier_meta["parent_names"],
                    "leaf": hier_meta["leaf_names"],
                },
            )
            oe_parent_logits = oe_scores["parent"]
            oe_leaf_logits = oe_scores["leaf"]

        loss_output = compute_taxosafe_loss(
            parent_logits=parent_logits,
            leaf_logits=leaf_logits,
            open_logits=open_logits,
            scale=scale,
            leaf_target=leaf_target,
            open_cut=open_cut,
            hier_meta=hier_meta,
            loss_cfg=cfg.get("loss", {}),
            oe_parent_logits=oe_parent_logits,
            oe_leaf_logits=oe_leaf_logits,
            oe_scale=oe_scale,
            epoch=epoch,
            unknown_logits=unknown_logits,
            morphology_features=morphology_features,
            image_features=representations["image"],
            leaf_text_features=representations["text"]["leaf"],
            unknown_text_features=representations["text"]["local_unknown"],
        )
        loss = loss_output["loss"]
        real_intra = None
        if intra_iterator is not None:
            try:
                intra_value = next(intra_iterator)
            except StopIteration:
                intra_iterator = iter(intra_data_loader)
                intra_value = next(intra_iterator)
            real_intra = _real_intra_unknown_loss(
                model=model,
                value=intra_value,
                device=device,
                cfg=cfg,
                hier_meta=hier_meta,
            )
            loss = loss + lambda_real_intra * real_intra["loss"]

        loss_ndtl = loss_output["loss_ndtl"]
        loss_ncl = loss_output["loss_ncl"]
        loss_cons = loss_output["loss_cons"]
        loss_rank = loss_output["loss_rank"]
        loss_child_known = loss_output["loss_child_known"]
        loss_child_novel = loss_output["loss_child_novel"]
        loss_pceg = loss_output["loss_pceg"]
        rank_branch_count = loss_output["rank_branch_count"]
        loss_root = loss_output["loss_root"]
        loss_oe = loss_output["loss_oe"]
        loss_local_unknown = loss_output["loss_local_unknown"]
        loss_tax_contrast = loss_output["loss_tax_contrast"]
        loss_sibling_boundary = loss_output["loss_sibling_boundary"]
        root_known_score = loss_output["root_known_score_mean"]
        root_oe_score = loss_output["root_oe_score_mean"]
        leaf_oe_score = loss_output["leaf_oe_score_mean"]
        root_oe_confidence = loss_output["root_oe_confidence_mean"]
        known_child_score = loss_output["known_child_score_mean"]
        pseudo_child_score = loss_output["pseudo_child_score_mean"]
        pseudo_pceg = loss_output["pseudo_pceg_mean"]
        local_known_margin = loss_output["local_known_margin_mean"]
        local_pseudo_margin = loss_output["local_pseudo_margin_mean"]
        sibling_boundary_margin = loss_output[
            "sibling_boundary_margin_mean"
        ]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Kept compatible with the current repository. If the scheduler was
        # changed to epoch-wise stepping in train_taxosafe.py, remove this block.
        if cfg.get("optim", {}).get("use_scheduler", False) and sched is not None:
            sched.step()

        with torch.no_grad():
            parent_pred = parent_logits.argmax(dim=1)
            leaf_pred = leaf_logits.argmax(dim=1)
            open_pred = open_logits.argmax(dim=1)

            parent_correct = parent_pred.eq(parent_target)
            leaf_correct = leaf_pred.eq(leaf_target)
            parent_acc = parent_correct.float().mean() * 100.0
            leaf_acc = leaf_correct.float().mean() * 100.0
            open_acc = open_pred.eq(open_target).float().mean() * 100.0
            hca = (parent_correct & leaf_correct).float().mean() * 100.0
            pseudo_rate = pseudo_mask.float().mean() * 100.0

        meter["loss"].update(loss.item(), batch_size)
        if real_intra is not None:
            intra_bs = real_intra["batch_size"]
            meter["loss_real_intra"].update(
                real_intra["loss"].item(), intra_bs
            )
            meter["loss_real_intra_parent"].update(
                real_intra["parent_loss"].item(), intra_bs
            )
            meter["loss_real_intra_local"].update(
                real_intra["local_loss"].item(), intra_bs
            )
            meter["loss_real_intra_margin"].update(
                real_intra["margin_loss"].item(), intra_bs
            )
            meter["real_intra_parent_acc"].update(
                real_intra["parent_acc"].item(), intra_bs
            )
        meter["acc"].update(leaf_acc.item(), batch_size)
        meter["loss_ndtl"].update(loss_ndtl.item(), batch_size)
        meter["loss_ncl"].update(loss_ncl.item(), batch_size)
        meter["loss_cons"].update(loss_cons.item(), batch_size)
        meter["loss_rank"].update(loss_rank.item(), batch_size)
        meter["loss_child_known"].update(
            loss_child_known.item(), batch_size
        )
        meter["loss_child_novel"].update(
            loss_child_novel.item(), batch_size
        )
        meter["loss_pceg"].update(loss_pceg.item(), batch_size)
        meter["rank_branch_count"].update(rank_branch_count, 1)
        meter["loss_root"].update(loss_root.item(), batch_size)
        meter["loss_oe"].update(loss_oe.item(), batch_size)
        meter["loss_local_unknown"].update(
            loss_local_unknown.item(), batch_size
        )
        meter["loss_tax_contrast"].update(
            loss_tax_contrast.item(), batch_size
        )
        meter["loss_sibling_boundary"].update(
            loss_sibling_boundary.item(), batch_size
        )
        meter["sibling_boundary_pair_count"].update(
            loss_output["sibling_boundary_pair_count"], 1
        )
        if sibling_boundary_margin is not None:
            meter["sibling_boundary_margin"].update(
                sibling_boundary_margin.item(),
                max(1, loss_output["sibling_boundary_pair_count"]),
            )
        meter["root_known_score"].update(
            root_known_score.item(), batch_size
        )
        if known_child_score is not None:
            meter["known_child_score"].update(
                known_child_score.item(), batch_size
            )
        if pseudo_child_score is not None:
            meter["pseudo_child_score"].update(
                pseudo_child_score.item(), batch_size
            )
        if pseudo_pceg is not None:
            meter["pseudo_pceg"].update(
                pseudo_pceg.item(), batch_size
            )
        if local_known_margin is not None:
            meter["local_known_margin"].update(
                local_known_margin.item(), batch_size
            )
        if local_pseudo_margin is not None:
            meter["local_pseudo_margin"].update(
                local_pseudo_margin.item(), batch_size
            )
        meter["novelty_weight_scale"].update(
            loss_output["novelty_weight_scale"], 1
        )
        if root_oe_score is not None:
            meter["root_oe_score"].update(
                root_oe_score.item(), int(oe_image.shape[0])
            )
        if leaf_oe_score is not None:
            meter["leaf_oe_score"].update(
                leaf_oe_score.item(), int(oe_image.shape[0])
            )
        if root_oe_confidence is not None:
            meter["root_oe_confidence"].update(
                root_oe_confidence.item(), int(oe_image.shape[0])
            )
        if oe_iterator is not None:
            meter["oe_weight_scale"].update(
                loss_output["oe_weight_scale"], 1
            )
        meter["parent_acc"].update(parent_acc.item(), batch_size)
        meter["leaf_acc"].update(leaf_acc.item(), batch_size)
        meter["open_acc"].update(open_acc.item(), batch_size)
        meter["consistency"].update(hca.item(), batch_size)
        meter["pseudo_rate"].update(pseudo_rate.item(), batch_size)

        if (step + 1) % print_freq == 0 or args.debug:
            print(
                "[Train] Epoch={}, Step={}, Loss={:.4f}, "
                "NDTL={:.4f}, NCL={:.4f}, Cons={:.4f}, "
                "Rank={:.4f}, ChildK={:.4f}, ChildU={:.4f}, "
                "PCEG={:.4f}, RankBranches={}, RootMargin={:.4f}, "
                "OE={:.4f}, KnownRoot={:.4f}, "
                "ParentAcc={:.2f}, LeafAcc={:.2f}, OpenAcc={:.2f}, "
                "HCA={:.2f}, PseudoRate={:.2f}".format(
                    epoch,
                    step,
                    loss.item(),
                    loss_ndtl.item(),
                    loss_ncl.item(),
                    loss_cons.item(),
                    loss_rank.item(),
                    loss_child_known.item(),
                    loss_child_novel.item(),
                    loss_pceg.item(),
                    rank_branch_count,
                    loss_root.item(),
                    loss_oe.item(),
                    root_known_score.item(),
                    parent_acc.item(),
                    leaf_acc.item(),
                    open_acc.item(),
                    hca.item(),
                    pseudo_rate.item(),
                )
            )

        if args.debug:
            break

    return {key: value.avg for key, value in meter.items()}


@torch.no_grad()
def eval_one_epoch(
    model,
    data_loader,
    param_names,
    device,
    epoch,
    cfg,
    args,
    leaf_nodes=None,
    intnl_nodes=None,
    sublabels=None,
    hier_meta=None,
):
    """Evaluate known samples at parent and leaf levels with one image pass."""
    hier_meta = _resolve_hier_meta(
        hier_meta,
        param_names,
        leaf_nodes,
        intnl_nodes,
        sublabels,
        device,
    )

    meter = {
        "loss": averageMeter(),
        "acc": averageMeter(),
        "loss_parent": averageMeter(),
        "loss_leaf": averageMeter(),
        "parent_acc": averageMeter(),
        "leaf_acc": averageMeter(),
        "consistency": averageMeter(),
        "local_known_acceptance": averageMeter(),
    }

    print_freq = cfg.get("print_freq", 20)
    model.eval()

    for step, value in tqdm(enumerate(data_loader), total=len(data_loader)):
        image = value[0].to(device)
        leaf_target = value[1].to(device=device, dtype=torch.long)
        batch_size = leaf_target.shape[0]
        parent_target = hier_meta["leaf_to_parent"][leaf_target]

        unknown_names = [
            cfg.get("open_treecut", {}).get(
                "unknown_template", "novel member of {}"
            ).format(name)
            for name in hier_meta["parent_names"]
        ]
        scores, _ = score_label_sets(
            model,
            image,
            {
                "parent": hier_meta["parent_names"],
                "leaf": hier_meta["leaf_names"],
                "local_unknown": unknown_names,
            },
        )
        parent_logits = scores["parent"]
        leaf_logits = scores["leaf"]

        loss_parent = F.cross_entropy(parent_logits, parent_target)
        loss_leaf = F.cross_entropy(leaf_logits, leaf_target)
        lambda_ncl = float(cfg.get("loss", {}).get("lambda_ncl", 0.5))
        loss = loss_leaf + lambda_ncl * loss_parent

        parent_pred = parent_logits.argmax(dim=1)
        leaf_pred = leaf_logits.argmax(dim=1)
        parent_correct = parent_pred.eq(parent_target)
        leaf_correct = leaf_pred.eq(leaf_target)
        parent_acc = parent_correct.float().mean() * 100.0
        leaf_acc = leaf_correct.float().mean() * 100.0
        hca = (parent_correct & leaf_correct).float().mean() * 100.0
        local_accept = []
        for row in range(batch_size):
            parent_id = int(parent_target[row].item())
            children = torch.as_tensor(
                hier_meta["children_by_parent"][parent_id],
                dtype=torch.long,
                device=device,
            )
            child_max = leaf_logits[row, children].max()
            local_accept.append(
                child_max >= scores["local_unknown"][row, parent_id]
            )
        local_acceptance = torch.stack(local_accept).float().mean() * 100.0

        meter["loss"].update(loss.item(), batch_size)
        meter["acc"].update(leaf_acc.item(), batch_size)
        meter["loss_parent"].update(loss_parent.item(), batch_size)
        meter["loss_leaf"].update(loss_leaf.item(), batch_size)
        meter["parent_acc"].update(parent_acc.item(), batch_size)
        meter["leaf_acc"].update(leaf_acc.item(), batch_size)
        meter["consistency"].update(hca.item(), batch_size)
        meter["local_known_acceptance"].update(
            local_acceptance.item(), batch_size
        )

        if (step + 1) % print_freq == 0 or args.debug:
            print(
                "[Eval] Epoch={}, Step={}, Loss={:.4f}, ParentAcc={:.2f}, "
                "LeafAcc={:.2f}, HCA={:.2f}".format(
                    epoch,
                    step,
                    loss.item(),
                    parent_acc.item(),
                    leaf_acc.item(),
                    hca.item(),
                )
            )

        if args.debug:
            break

    return {key: value.avg for key, value in meter.items()}


@torch.no_grad()
def eval_local_unknown_epoch(model, data_loader, device, cfg, hier_meta):
    """Evaluate unseen leaves while retaining their supplied parent label."""
    model.eval()
    unknown_names = [
        cfg.get("open_treecut", {}).get(
            "unknown_template", "novel member of {}"
        ).format(name)
        for name in hier_meta["parent_names"]
    ]
    total = 0
    parent_correct = 0
    local_rejected = 0
    retained_and_rejected = 0
    margins = []
    for value in data_loader:
        image = value[0].to(device)
        true_parent = value[1].to(device=device, dtype=torch.long)
        scores, _ = score_label_sets(
            model,
            image,
            {
                "parent": hier_meta["parent_names"],
                "leaf": hier_meta["leaf_names"],
                "local_unknown": unknown_names,
            },
        )
        predicted_parent = scores["parent"].argmax(dim=1)
        for row in range(len(true_parent)):
            route = int(predicted_parent[row].item())
            children = torch.as_tensor(
                hier_meta["children_by_parent"][route],
                dtype=torch.long,
                device=device,
            )
            child_max = scores["leaf"][row, children].max()
            unknown_logit = scores["local_unknown"][row, route]
            rejected = bool((unknown_logit >= child_max).item())
            correct_parent = route == int(true_parent[row].item())
            total += 1
            parent_correct += int(correct_parent)
            local_rejected += int(rejected)
            retained_and_rejected += int(correct_parent and rejected)
            margins.append(float((unknown_logit - child_max).item()))
    if total == 0:
        raise ValueError("development unknown split is empty")
    return {
        "sample_count": total,
        "parent_accuracy": 100.0 * parent_correct / total,
        "local_unknown_recall": 100.0 * local_rejected / total,
        "ancestor_retained_unknown_accuracy": (
            100.0 * retained_and_rejected / total
        ),
        "unknown_minus_child_margin": sum(margins) / len(margins),
    }
