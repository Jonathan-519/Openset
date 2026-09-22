"""Losses for TaxoSafe hierarchical open-set prompt learning."""

import math

import torch
import torch.nn.functional as F


def js_divergence(p, q, eps=1e-8):
    """Jensen-Shannon divergence between two batches of distributions."""
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    middle = 0.5 * (p + q)
    return 0.5 * (
        F.kl_div(middle.log(), p, reduction="batchmean")
        + F.kl_div(middle.log(), q, reduction="batchmean")
    )


def _zero_like(logits):
    return logits.sum() * 0.0


def _children_list(children):
    return [
        int(value)
        for value in torch.as_tensor(children)
        .detach()
        .cpu()
        .reshape(-1)
        .tolist()
    ]


def _normalise_hidden(open_cut, num_parents):
    """Convert supported hidden-leaf representations to sets of IDs."""
    raw = open_cut.get(
        "hidden_by_parent",
        open_cut.get("hidden_leaf_by_parent", {}),
    )
    hidden = []
    for parent_id in range(num_parents):
        if isinstance(raw, dict):
            value = raw.get(parent_id, raw.get(str(parent_id), None))
        elif isinstance(raw, (list, tuple)) and parent_id < len(raw):
            value = raw[parent_id]
        else:
            value = None

        if value is None:
            hidden.append(set())
        elif torch.is_tensor(value):
            hidden.append(set(
                int(item)
                for item in value.detach().cpu().reshape(-1).tolist()
            ))
        elif isinstance(value, (list, tuple, set)):
            hidden.append(set(int(item) for item in value))
        else:
            hidden.append({int(value)})
    return hidden


def _branch_local_ncl(
    parent_logits,
    leaf_logits,
    leaf_target,
    parent_target,
    pseudo_mask,
    hier_meta,
    hidden,
):
    """Root CE plus branch-local leaf CE over active known children."""
    ncl_terms = [F.cross_entropy(parent_logits, parent_target)]
    num_leaves = len(hier_meta["leaf_names"])
    device = leaf_target.device

    for parent_id, children_tensor in enumerate(
        hier_meta["children_by_parent"]
    ):
        children = _children_list(children_tensor)
        active_children = [
            leaf_id
            for leaf_id in children
            if leaf_id not in hidden[parent_id]
        ]
        known_mask = (parent_target == parent_id) & (~pseudo_mask)
        if not torch.any(known_mask) or not active_children:
            continue

        active = torch.tensor(
            active_children, dtype=torch.long, device=device
        )
        lookup = torch.full(
            (num_leaves,), -1, dtype=torch.long, device=device
        )
        lookup[active] = torch.arange(
            len(active_children), dtype=torch.long, device=device
        )
        local_target = lookup[leaf_target[known_mask]]
        if torch.any(local_target < 0):
            raise RuntimeError(
                "A non-pseudo target was removed from its active branch"
            )
        local_logits = leaf_logits[known_mask][:, active]
        ncl_terms.append(F.cross_entropy(local_logits, local_target))

    return torch.stack(ncl_terms).mean()


def _consistency_loss(
    parent_logits,
    leaf_logits,
    pseudo_mask,
    hier_meta,
    hidden=None,
):
    """Match direct parent probabilities to leaf-aggregated probabilities."""
    consistency_mask = ~pseudo_mask
    if not torch.any(consistency_mask):
        return _zero_like(parent_logits)

    num_leaves = len(hier_meta["leaf_names"])
    num_parents = len(hier_meta["parent_names"])
    device = leaf_logits.device
    leaf_to_parent = hier_meta["leaf_to_parent"].to(
        device=device, dtype=torch.long
    )

    aggregation = torch.zeros(
        num_leaves,
        num_parents,
        dtype=leaf_logits.dtype,
        device=device,
    )
    aggregation[
        torch.arange(num_leaves, device=device), leaf_to_parent
    ] = 1.0

    p_parent = F.softmax(parent_logits, dim=-1)
    if hidden is not None:
        active = torch.ones(num_leaves, dtype=torch.bool, device=device)
        for leaves in hidden:
            if leaves:
                active[torch.tensor(sorted(leaves), dtype=torch.long, device=device)] = False
        for parent in range(num_parents):
            if not torch.any(active & (leaf_to_parent == parent)):
                raise ValueError("Consistency needs at least one active leaf per parent")
        # Removing pseudo QUERY rows alone does not remove held LABEL columns.
        # masked_fill also makes gradients to held logits exactly zero.
        leaf_logits = leaf_logits.masked_fill(~active[None, :], float("-inf"))
    p_leaf = F.softmax(leaf_logits, dim=-1)
    return js_divergence(
        p_parent[consistency_mask],
        (p_leaf @ aggregation)[consistency_mask],
    )


def _mean_or_none(values):
    if not values:
        return None
    return torch.cat([value.reshape(-1) for value in values]).mean()


def morphology_pool(
    spatial_features,
    leaf_text_features,
    parent_target,
    hier_meta,
    attention_temperature=0.10,
):
    """Pool patches that discriminate among siblings in the routed branch.

    The attention descriptor is centred within each parent, so generic
    plankton/background similarity cannot dominate the morphology loss.
    """
    if spatial_features.ndim != 3:
        raise ValueError("spatial_features must have shape [B, patches, D]")
    if leaf_text_features.ndim != 2:
        raise ValueError("leaf_text_features must have shape [leaves, D]")
    if spatial_features.shape[-1] != leaf_text_features.shape[-1]:
        raise ValueError("Spatial and text feature dimensions differ")
    temperature = float(attention_temperature)
    if temperature <= 0.0:
        raise ValueError("morphology attention temperature must be positive")

    pooled = torch.empty(
        spatial_features.shape[0],
        spatial_features.shape[-1],
        dtype=spatial_features.dtype,
        device=spatial_features.device,
    )
    for parent_id, children_tensor in enumerate(
        hier_meta["children_by_parent"]
    ):
        mask = parent_target == parent_id
        if not torch.any(mask):
            continue
        children = torch.as_tensor(
            children_tensor,
            dtype=torch.long,
            device=spatial_features.device,
        )
        descriptors = leaf_text_features[children]
        descriptors = descriptors - descriptors.mean(dim=0, keepdim=True)
        tokens = F.normalize(spatial_features[mask].float(), dim=-1)
        if len(children) == 1:
            weights = torch.full(
                tokens.shape[:2],
                1.0 / float(tokens.shape[1]),
                dtype=tokens.dtype,
                device=tokens.device,
            )
        else:
            salience = (tokens @ descriptors.float().t()).square().mean(-1)
            weights = F.softmax(salience / temperature, dim=1)
        pooled[mask] = F.normalize(
            (weights.unsqueeze(-1) * tokens).sum(dim=1), dim=-1
        ).to(dtype=spatial_features.dtype)
    return pooled


def taxonomy_weighted_contrastive_loss(
    features,
    leaf_target,
    parent_target,
    temperature=0.10,
    sibling_weight=2.0,
    distant_weight=0.25,
):
    """Supervised contrastive loss with emphasis on sibling negatives."""
    if features.ndim != 2:
        raise ValueError("features must have shape [B, D]")
    temperature = float(temperature)
    sibling_weight = float(sibling_weight)
    distant_weight = float(distant_weight)
    if temperature <= 0.0:
        raise ValueError("contrastive temperature must be positive")
    if sibling_weight <= 0.0 or distant_weight <= 0.0:
        raise ValueError("contrastive negative weights must be positive")

    features = F.normalize(features.float(), dim=-1)
    count = int(features.shape[0])
    eye = torch.eye(count, dtype=torch.bool, device=features.device)
    positive = leaf_target[:, None].eq(leaf_target[None, :]) & ~eye
    valid = positive.any(dim=1)
    if not torch.any(valid):
        return _zero_like(features)

    same_parent = parent_target[:, None].eq(parent_target[None, :])
    weights = torch.full(
        (count, count),
        distant_weight,
        dtype=features.dtype,
        device=features.device,
    )
    weights = torch.where(
        same_parent & ~positive,
        torch.full_like(weights, sibling_weight),
        weights,
    )
    weights = torch.where(positive, torch.ones_like(weights), weights)
    logits = (features @ features.t()) / temperature
    logits = logits.masked_fill(eye, float("-inf"))
    denominator = torch.logsumexp(logits + weights.log(), dim=1)
    positive_logits = logits.masked_fill(~positive, float("-inf"))
    numerator = torch.logsumexp(positive_logits, dim=1) - positive.sum(
        dim=1
    ).clamp_min(1).float().log()
    return (denominator[valid] - numerator[valid]).mean()


def sibling_boundary_unknown_loss(
    image_features,
    leaf_text_features,
    unknown_text_features,
    leaf_target,
    parent_target,
    hier_meta,
    scale,
    mix_min=0.35,
    mix_max=0.65,
    pairs_per_parent=8,
):
    """Reserve open space between sibling species for the local unknown.

    Feature-space mixtures are constructed only from different leaves under
    the same parent.  Each mixture is classified against that parent's known
    children plus its explicit ``novel member of <parent>`` prompt, with the
    local-unknown prompt as the target.  This is deliberately separate from
    the epoch-wise whole-species holdout: the latter models a complete unseen
    species, while this term keeps the gaps between known sibling regions from
    being filled by an arbitrary known leaf.
    """
    if image_features is None:
        return {
            "loss": _zero_like(leaf_text_features),
            "pair_count": 0,
            "unknown_margin_mean": None,
        }
    if image_features.ndim != 2:
        raise ValueError("image_features must have shape [B, D]")
    if leaf_text_features.ndim != 2 or unknown_text_features.ndim != 2:
        raise ValueError("text features must have shape [labels, D]")
    if image_features.shape[-1] != leaf_text_features.shape[-1]:
        raise ValueError("Image and leaf text dimensions differ")
    if image_features.shape[-1] != unknown_text_features.shape[-1]:
        raise ValueError("Image and unknown text dimensions differ")
    if unknown_text_features.shape[0] != len(hier_meta["parent_names"]):
        raise ValueError("Expected one local-unknown text feature per parent")

    mix_min, mix_max = float(mix_min), float(mix_max)
    pairs_per_parent = int(pairs_per_parent)
    if not 0.0 <= mix_min <= mix_max <= 1.0:
        raise ValueError("boundary mix range must lie in [0, 1]")
    if pairs_per_parent < 1:
        raise ValueError("boundary pairs_per_parent must be positive")

    terms, margins = [], []
    pair_count = 0
    for parent_id, children_tensor in enumerate(
        hier_meta["children_by_parent"]
    ):
        branch_ids = torch.where(parent_target == parent_id)[0]
        if branch_ids.numel() < 2:
            continue
        branch_leaves = leaf_target[branch_ids]
        if torch.unique(branch_leaves).numel() < 2:
            continue

        first, second = [], []
        order = branch_ids[torch.randperm(
            branch_ids.numel(), device=branch_ids.device
        )]
        for index in order.tolist():
            alternatives = branch_ids[leaf_target[branch_ids] != leaf_target[index]]
            if alternatives.numel() == 0:
                continue
            choice = alternatives[torch.randint(
                alternatives.numel(), (1,), device=alternatives.device
            )]
            first.append(index)
            second.append(int(choice.item()))
            if len(first) >= pairs_per_parent:
                break
        if not first:
            continue

        first = torch.tensor(first, dtype=torch.long, device=image_features.device)
        second = torch.tensor(second, dtype=torch.long, device=image_features.device)
        mix = torch.empty(
            len(first), 1, dtype=image_features.dtype,
            device=image_features.device,
        ).uniform_(mix_min, mix_max)
        mixed = F.normalize(
            mix * image_features[first]
            + (1.0 - mix) * image_features[second],
            dim=-1,
        )
        children = torch.as_tensor(
            children_tensor, dtype=torch.long, device=image_features.device
        )
        labels = torch.cat([
            leaf_text_features[children],
            unknown_text_features[parent_id : parent_id + 1],
        ])
        logits = scale.float() * mixed.float() @ labels.float().t()
        target = torch.full(
            (len(first),), len(children), dtype=torch.long,
            device=image_features.device,
        )
        terms.append(F.cross_entropy(logits, target))
        margins.append(
            logits[:, -1] - logits[:, :-1].max(dim=-1).values
        )
        pair_count += len(first)

    zero = _zero_like(leaf_text_features)
    return {
        "loss": zero if not terms else torch.stack(terms).mean(),
        "pair_count": pair_count,
        "unknown_margin_mean": _mean_or_none(margins),
    }


def _local_unknown_loss(
    leaf_logits,
    unknown_logits,
    leaf_target,
    parent_target,
    pseudo_mask,
    hier_meta,
    hidden,
):
    """Train one explicit unknown-descendant class at every parent node."""
    if unknown_logits is None:
        return {
            "loss": _zero_like(leaf_logits),
            "known_margin_mean": None,
            "pseudo_margin_mean": None,
        }
    expected = (leaf_logits.shape[0], len(hier_meta["parent_names"]))
    if tuple(unknown_logits.shape) != expected:
        raise ValueError(
            "unknown_logits shape {} != {}".format(
                tuple(unknown_logits.shape), expected
            )
        )

    terms = []
    known_margins = []
    pseudo_margins = []
    for parent_id, children_tensor in enumerate(
        hier_meta["children_by_parent"]
    ):
        branch_mask = parent_target == parent_id
        if not torch.any(branch_mask):
            continue
        children = [
            leaf_id
            for leaf_id in _children_list(children_tensor)
            if leaf_id not in hidden[parent_id]
        ]
        if not children:
            raise RuntimeError(
                "Local unknown branch {} has no active known child".format(
                    parent_id
                )
            )
        active = torch.tensor(
            children, dtype=torch.long, device=leaf_logits.device
        )
        branch_leaf = leaf_logits[branch_mask][:, active]
        branch_unknown = unknown_logits[branch_mask, parent_id : parent_id + 1]
        local_logits = torch.cat([branch_leaf, branch_unknown], dim=1)
        branch_leaf_target = leaf_target[branch_mask]
        branch_pseudo = pseudo_mask[branch_mask]
        lookup = torch.full(
            (len(hier_meta["leaf_names"]),),
            -1,
            dtype=torch.long,
            device=leaf_logits.device,
        )
        lookup[active] = torch.arange(len(active), device=leaf_logits.device)
        target = lookup[branch_leaf_target]
        target = torch.where(
            branch_pseudo,
            torch.full_like(target, len(active)),
            target,
        )
        if torch.any(target < 0):
            raise RuntimeError("An active local target was removed")
        terms.append(F.cross_entropy(local_logits, target))

        known = ~branch_pseudo
        if torch.any(known):
            true_known = branch_leaf[known].gather(
                1, target[known][:, None]
            ).squeeze(1)
            known_margins.append(true_known - branch_unknown[known, 0])
        if torch.any(branch_pseudo):
            strongest_sibling = branch_leaf[branch_pseudo].max(dim=1).values
            pseudo_margins.append(
                branch_unknown[branch_pseudo, 0] - strongest_sibling
            )

    return {
        "loss": torch.stack(terms).mean() if terms else _zero_like(leaf_logits),
        "known_margin_mean": _mean_or_none(known_margins),
        "pseudo_margin_mean": _mean_or_none(pseudo_margins),
    }


def _child_novelty_losses(
    parent_logits,
    leaf_logits,
    scale,
    leaf_target,
    parent_target,
    pseudo_mask,
    hier_meta,
    hidden,
    rank_margin,
    known_child_margin,
    pseudo_child_margin,
    pceg_margin,
):
    """Learn nested parent/child decision regions.

    Unlike the v2 ranking implementation, the known score is the score of the
    *true* active leaf, not merely the largest score in the branch.  For a
    pseudo-unseen query, its temporarily hidden leaf is absent from the active
    label set and the score is the strongest remaining sibling.
    """
    safe_scale = scale.detach().clamp_min(1e-8)
    leaf_cosine = leaf_logits / safe_scale
    parent_cosine = parent_logits / safe_scale

    rank_terms = []
    known_boundary_terms = []
    novel_boundary_terms = []
    pceg_terms = []
    known_values = []
    novel_values = []
    pceg_values = []

    for parent_id, children_tensor in enumerate(
        hier_meta["children_by_parent"]
    ):
        children = _children_list(children_tensor)
        active_children = [
            leaf_id
            for leaf_id in children
            if leaf_id not in hidden[parent_id]
        ]
        known_mask = (parent_target == parent_id) & (~pseudo_mask)
        novel_mask = (parent_target == parent_id) & pseudo_mask

        known_score = None
        if torch.any(known_mask):
            known_score = leaf_cosine[known_mask].gather(
                1, leaf_target[known_mask].unsqueeze(1)
            ).squeeze(1)
            known_values.append(known_score.detach())
            known_boundary_terms.append(
                F.relu(float(known_child_margin) - known_score).mean()
            )

        novel_score = None
        if torch.any(novel_mask):
            if not active_children:
                raise RuntimeError(
                    "Pseudo-unseen branch {} has no active sibling".format(
                        parent_id
                    )
                )
            active = torch.tensor(
                active_children,
                dtype=torch.long,
                device=leaf_logits.device,
            )
            novel_score = leaf_cosine[novel_mask][:, active].max(
                dim=-1
            ).values
            novel_values.append(novel_score.detach())
            novel_boundary_terms.append(
                F.relu(novel_score - float(pseudo_child_margin)).mean()
            )

            true_parent_score = parent_cosine[novel_mask, parent_id]
            novelty_gap = true_parent_score - novel_score
            pceg_values.append(novelty_gap.detach())
            pceg_terms.append(
                F.relu(float(pceg_margin) - novelty_gap).mean()
            )

        if known_score is not None and novel_score is not None:
            pairwise = (
                float(rank_margin)
                + novel_score[:, None]
                - known_score[None, :]
            )
            rank_terms.append(F.relu(pairwise).mean())

    zero = _zero_like(leaf_logits)
    output = {
        "loss_rank": (
            zero if not rank_terms else torch.stack(rank_terms).mean()
        ),
        "loss_child_known": (
            zero
            if not known_boundary_terms
            else torch.stack(known_boundary_terms).mean()
        ),
        "loss_child_novel": (
            zero
            if not novel_boundary_terms
            else torch.stack(novel_boundary_terms).mean()
        ),
        "loss_pceg": (
            zero if not pceg_terms else torch.stack(pceg_terms).mean()
        ),
        "rank_branch_count": len(rank_terms),
        "known_child_score_mean": _mean_or_none(known_values),
        "pseudo_child_score_mean": _mean_or_none(novel_values),
        "pseudo_pceg_mean": _mean_or_none(pceg_values),
    }
    return output


def _root_known_margin_loss(parent_logits, scale, parent_target, margin):
    """Keep in-taxonomy images above an absolute parent cosine margin."""
    parent_cosine = parent_logits / scale.detach().clamp_min(1e-8)
    true_score = parent_cosine.gather(
        1, parent_target[:, None]
    ).squeeze(1)
    return (
        F.relu(float(margin) - true_score).mean(),
        true_score.detach().mean(),
    )


def _oe_loss(
    oe_parent_logits,
    oe_leaf_logits,
    oe_scale,
    margin,
    leaf_margin,
    entropy_weight,
    leaf_weight,
    reference_logits,
):
    """Suppress absolute OE similarity and parent over-confidence."""
    if oe_parent_logits is None or oe_scale is None:
        return {
            "loss": _zero_like(reference_logits),
            "parent_score_mean": None,
            "leaf_score_mean": None,
            "parent_confidence_mean": None,
        }

    safe_scale = oe_scale.detach().clamp_min(1e-8)
    parent_cosine = oe_parent_logits / safe_scale
    parent_knownness = parent_cosine.max(dim=-1).values
    root_hinge = F.relu(parent_knownness - float(margin)).mean()

    probabilities = F.softmax(oe_parent_logits.float(), dim=-1)
    num_parents = int(probabilities.shape[-1])
    if num_parents > 1:
        entropy = -(
            probabilities * probabilities.clamp_min(1e-12).log()
        ).sum(dim=-1) / math.log(float(num_parents))
        parent_confidence = 1.0 - entropy
        entropy_loss = parent_confidence.mean()
    else:
        parent_confidence = torch.zeros_like(parent_knownness)
        entropy_loss = _zero_like(oe_parent_logits)

    leaf_hinge = _zero_like(oe_parent_logits)
    leaf_score_mean = None
    if oe_leaf_logits is not None:
        leaf_knownness = (oe_leaf_logits / safe_scale).max(dim=-1).values
        leaf_hinge = F.relu(
            leaf_knownness - float(leaf_margin)
        ).mean()
        leaf_score_mean = leaf_knownness.detach().mean()

    return {
        "loss": (
            root_hinge
            + float(entropy_weight) * entropy_loss
            + float(leaf_weight) * leaf_hinge
        ),
        "parent_score_mean": parent_knownness.detach().mean(),
        "leaf_score_mean": leaf_score_mean,
        "parent_confidence_mean": parent_confidence.detach().mean(),
    }


def _warmup_scale(epoch, warmup_epochs):
    warmup_epochs = int(warmup_epochs)
    if warmup_epochs <= 0:
        return 1.0
    return min(1.0, float(int(epoch) + 1) / float(warmup_epochs))


def compute_taxosafe_loss(
    parent_logits,
    leaf_logits,
    open_logits,
    scale,
    leaf_target,
    open_cut,
    hier_meta,
    loss_cfg,
    oe_parent_logits=None,
    oe_leaf_logits=None,
    oe_scale=None,
    epoch=0,
    unknown_logits=None,
    morphology_features=None,
    image_features=None,
    leaf_text_features=None,
    unknown_text_features=None,
):
    """Compute the TaxoLocal training objective and diagnostics."""
    device = leaf_logits.device
    leaf_target = leaf_target.to(device=device, dtype=torch.long)
    parent_target = hier_meta["leaf_to_parent"][leaf_target].long()
    open_target = torch.as_tensor(
        open_cut["target"], dtype=torch.long, device=device
    )
    pseudo_mask = torch.as_tensor(
        open_cut["pseudo_mask"], dtype=torch.bool, device=device
    )
    hidden = _normalise_hidden(
        open_cut, len(hier_meta["parent_names"])
    )

    known_child_margin = float(
        loss_cfg.get("known_child_margin", 0.30)
    )
    pseudo_child_margin = float(
        loss_cfg.get("pseudo_child_margin", 0.28)
    )
    if known_child_margin <= pseudo_child_margin:
        raise ValueError(
            "known_child_margin must be greater than pseudo_child_margin"
        )

    loss_ndtl = F.cross_entropy(open_logits, open_target)
    loss_ncl = _branch_local_ncl(
        parent_logits,
        leaf_logits,
        leaf_target,
        parent_target,
        pseudo_mask,
        hier_meta,
        hidden,
    )
    loss_cons = _consistency_loss(
        parent_logits, leaf_logits, pseudo_mask, hier_meta,
        hidden=hidden if loss_cfg.get("mask_hidden_in_consistency", False) else None,
    )
    novelty = _child_novelty_losses(
        parent_logits=parent_logits,
        leaf_logits=leaf_logits,
        scale=scale,
        leaf_target=leaf_target,
        parent_target=parent_target,
        pseudo_mask=pseudo_mask,
        hier_meta=hier_meta,
        hidden=hidden,
        rank_margin=loss_cfg.get("rank_margin", 0.08),
        known_child_margin=known_child_margin,
        pseudo_child_margin=pseudo_child_margin,
        pceg_margin=loss_cfg.get("pceg_margin", 0.04),
    )
    loss_root, root_known_score_mean = _root_known_margin_loss(
        parent_logits,
        scale,
        parent_target,
        loss_cfg.get("known_parent_margin", 0.28),
    )
    oe = _oe_loss(
        oe_parent_logits=oe_parent_logits,
        oe_leaf_logits=oe_leaf_logits,
        oe_scale=oe_scale,
        margin=loss_cfg.get("oe_margin", 0.15),
        leaf_margin=loss_cfg.get("oe_leaf_margin", 0.15),
        entropy_weight=loss_cfg.get("oe_entropy_weight", 0.10),
        leaf_weight=loss_cfg.get("oe_leaf_weight", 0.25),
        reference_logits=parent_logits,
    )
    local_unknown = _local_unknown_loss(
        leaf_logits=leaf_logits,
        unknown_logits=unknown_logits,
        leaf_target=leaf_target,
        parent_target=parent_target,
        pseudo_mask=pseudo_mask,
        hier_meta=hier_meta,
        hidden=hidden,
    )
    loss_tax_contrast = (
        _zero_like(leaf_logits)
        if morphology_features is None
        else taxonomy_weighted_contrastive_loss(
            morphology_features,
            leaf_target,
            parent_target,
            temperature=loss_cfg.get("contrastive_temperature", 0.10),
            sibling_weight=loss_cfg.get("sibling_negative_weight", 2.0),
            distant_weight=loss_cfg.get("distant_negative_weight", 0.25),
        )
    )
    boundary_enabled = float(
        loss_cfg.get("lambda_sibling_boundary", 0.0)
    ) > 0.0
    if boundary_enabled and any(
        value is None
        for value in (
            image_features, leaf_text_features, unknown_text_features
        )
    ):
        raise ValueError(
            "Sibling-boundary loss requires image, leaf-text and "
            "unknown-text features"
        )
    boundary = (
        {
            "loss": _zero_like(leaf_logits),
            "pair_count": 0,
            "unknown_margin_mean": None,
        }
        if not boundary_enabled
        else sibling_boundary_unknown_loss(
            image_features=image_features,
            leaf_text_features=leaf_text_features,
            unknown_text_features=unknown_text_features,
            leaf_target=leaf_target,
            parent_target=parent_target,
            hier_meta=hier_meta,
            scale=scale,
            mix_min=loss_cfg.get("boundary_mix_min", 0.35),
            mix_max=loss_cfg.get("boundary_mix_max", 0.65),
            pairs_per_parent=loss_cfg.get(
                "boundary_pairs_per_parent", 8
            ),
        )
    )

    novelty_scale = _warmup_scale(
        epoch, loss_cfg.get("novelty_warmup_epochs", 5)
    )
    oe_scale_weight = _warmup_scale(
        epoch, loss_cfg.get("oe_warmup_epochs", 5)
    )
    total = (
        loss_ndtl
        + float(loss_cfg.get("lambda_ncl", 0.5)) * loss_ncl
        + float(loss_cfg.get("lambda_cons", 0.1)) * loss_cons
        + novelty_scale
        * float(loss_cfg.get("lambda_rank", 0.25))
        * novelty["loss_rank"]
        + float(loss_cfg.get("lambda_child_known", 0.10))
        * novelty["loss_child_known"]
        + novelty_scale
        * float(loss_cfg.get("lambda_child_novel", 0.50))
        * novelty["loss_child_novel"]
        + novelty_scale
        * float(loss_cfg.get("lambda_pceg", 0.20))
        * novelty["loss_pceg"]
        + float(loss_cfg.get("lambda_root", 0.20)) * loss_root
        + oe_scale_weight
        * float(loss_cfg.get("lambda_oe", 0.0))
        * oe["loss"]
        + novelty_scale
        * float(loss_cfg.get("lambda_local_unknown", 1.0))
        * local_unknown["loss"]
        + novelty_scale
        * float(loss_cfg.get("lambda_tax_contrast", 0.25))
        * loss_tax_contrast
        + novelty_scale
        * float(loss_cfg.get("lambda_sibling_boundary", 0.0))
        * boundary["loss"]
    )

    return {
        "loss": total,
        "loss_ndtl": loss_ndtl,
        "loss_ncl": loss_ncl,
        "loss_cons": loss_cons,
        "loss_rank": novelty["loss_rank"],
        "loss_child_known": novelty["loss_child_known"],
        "loss_child_novel": novelty["loss_child_novel"],
        "loss_pceg": novelty["loss_pceg"],
        "rank_branch_count": novelty["rank_branch_count"],
        "loss_root": loss_root,
        "loss_oe": oe["loss"],
        "loss_local_unknown": local_unknown["loss"],
        "loss_tax_contrast": loss_tax_contrast,
        "loss_sibling_boundary": boundary["loss"],
        "sibling_boundary_pair_count": boundary["pair_count"],
        "sibling_boundary_margin_mean": boundary["unknown_margin_mean"],
        "local_known_margin_mean": local_unknown["known_margin_mean"],
        "local_pseudo_margin_mean": local_unknown["pseudo_margin_mean"],
        "root_known_score_mean": root_known_score_mean,
        "root_oe_score_mean": oe["parent_score_mean"],
        "leaf_oe_score_mean": oe["leaf_score_mean"],
        "root_oe_confidence_mean": oe["parent_confidence_mean"],
        "known_child_score_mean": novelty["known_child_score_mean"],
        "pseudo_child_score_mean": novelty["pseudo_child_score_mean"],
        "pseudo_pceg_mean": novelty["pseudo_pceg_mean"],
        "novelty_weight_scale": float(novelty_scale),
        "oe_weight_scale": float(oe_scale_weight),
        "parent_target": parent_target,
        "open_target": open_target,
        "pseudo_mask": pseudo_mask,
    }
