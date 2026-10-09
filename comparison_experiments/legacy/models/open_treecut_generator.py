"""Novelty-aware dynamic tree-cut sampler used by TaxoSafe."""

import torch

from taxosafe_episode import epoch_holdout_leaf


class OpenTreecutGenerator:
    """Create parent/leaf tree cuts with species-level pseudo unknowns.

    ``holdout_mode='epoch'`` is the TaxoSafe v3 policy. One leaf of every
    multi-leaf parent is fixed as pseudo-unseen for the whole epoch and rotates
    between epochs. ``holdout_mode='batch'`` preserves the earlier stochastic
    behaviour for ablation experiments.
    """

    def __init__(
        self,
        parent_names,
        leaf_names,
        children_by_parent,
        collapse_prob=0.30,
        hide_prob=0.50,
        unknown_template="novel member of {}",
        force_one_pseudo_per_batch=False,
        holdout_mode="epoch",
        holdout_seed=1,
    ):
        self.parent_names = list(parent_names)
        self.leaf_names = list(leaf_names)
        self.children_by_parent = [
            [
                int(leaf_id)
                for leaf_id in torch.as_tensor(children)
                .detach()
                .cpu()
                .reshape(-1)
                .tolist()
            ]
            for children in children_by_parent
        ]
        self.collapse_prob = float(collapse_prob)
        self.hide_prob = float(hide_prob)
        self.unknown_template = str(unknown_template)
        self.force_one_pseudo_per_batch = bool(force_one_pseudo_per_batch)
        self.holdout_mode = str(holdout_mode).lower()
        self.holdout_seed = int(holdout_seed)
        self.epoch = 0

        if len(self.parent_names) != len(self.children_by_parent):
            raise ValueError(
                "children_by_parent must contain one entry per parent"
            )
        if not 0.0 <= self.collapse_prob <= 1.0:
            raise ValueError("collapse_prob must be in [0, 1]")
        if not 0.0 <= self.hide_prob <= 1.0:
            raise ValueError("hide_prob must be in [0, 1]")
        if self.holdout_mode not in {"epoch", "batch"}:
            raise ValueError("holdout_mode must be 'epoch' or 'batch'")

        all_children = []
        for parent_id, children in enumerate(self.children_by_parent):
            if not children:
                raise ValueError(
                    "Parent '{}' has no known leaves".format(
                        self.parent_names[parent_id]
                    )
                )
            all_children.extend(children)
        expected_children = list(range(len(self.leaf_names)))
        if sorted(all_children) != expected_children:
            raise ValueError(
                "children_by_parent must partition every known leaf exactly "
                "once; expected {}, got {}".format(
                    expected_children, sorted(all_children)
                )
            )

        try:
            self.unknown_template.format(self.parent_names[0])
        except (IndexError, KeyError, ValueError) as error:
            raise ValueError(
                "unknown_template must accept one parent name"
            ) from error

    def set_epoch(self, epoch):
        """Fix the pseudo-unseen schedule for one complete epoch."""
        self.epoch = int(epoch)

    def _epoch_hidden(self, parent_id):
        return epoch_holdout_leaf(
            self.children_by_parent[parent_id],
            parent_id,
            self.epoch,
            self.holdout_seed,
        )

    @staticmethod
    def _eligible_for_forcing(
        parent_id, leaf_target, parent_target, hidden_leaf
    ):
        if hidden_leaf is None:
            return False
        present = torch.unique(leaf_target[parent_target == parent_id])
        if present.numel() < 2:
            return False
        contains_hidden = bool(torch.any(present == hidden_leaf).item())
        contains_active = bool(torch.any(present != hidden_leaf).item())
        return contains_hidden and contains_active

    @torch.no_grad()
    def sample(self, leaf_target, parent_target):
        """Return remapped labels plus pseudo-unseen masks and metadata."""
        if leaf_target.ndim != 1 or parent_target.ndim != 1:
            raise ValueError("leaf_target and parent_target must be 1-D")
        if leaf_target.shape != parent_target.shape:
            raise ValueError(
                "leaf_target and parent_target must have identical shapes"
            )

        device = leaf_target.device
        leaf_target = leaf_target.to(device=device, dtype=torch.long)
        parent_target = parent_target.to(device=device, dtype=torch.long)
        batch_size = leaf_target.numel()
        if torch.any(leaf_target < 0) or torch.any(
            leaf_target >= len(self.leaf_names)
        ):
            raise ValueError("leaf_target contains an invalid known-leaf ID")
        if torch.any(parent_target < 0) or torch.any(
            parent_target >= len(self.parent_names)
        ):
            raise ValueError("parent_target contains an invalid parent ID")

        hidden_by_parent = {}
        if self.holdout_mode == "epoch":
            for parent_id in range(len(self.parent_names)):
                hidden_by_parent[parent_id] = self._epoch_hidden(parent_id)

        forced_parent = None
        if self.force_one_pseudo_per_batch:
            eligible = []
            if self.holdout_mode == "epoch":
                for parent_id in range(len(self.parent_names)):
                    if self._eligible_for_forcing(
                        parent_id,
                        leaf_target,
                        parent_target,
                        hidden_by_parent[parent_id],
                    ):
                        eligible.append(parent_id)
            else:
                for parent_id, children in enumerate(self.children_by_parent):
                    present = torch.unique(
                        leaf_target[parent_target == parent_id]
                    )
                    if len(children) >= 2 and present.numel() >= 2:
                        eligible.append(parent_id)
            if not eligible:
                raise RuntimeError(
                    "force_one_pseudo_per_batch=True, but the batch has no "
                    "held leaf together with an active sibling. Ensure that "
                    "the episode sampler uses the same holdout mode/seed."
                )
            selected = torch.randint(
                0, len(eligible), (1,), device=device
            )
            forced_parent = eligible[int(selected.item())]

        label_names = []
        target = torch.full(
            (batch_size,), -1, dtype=torch.long, device=device
        )
        pseudo_mask = torch.zeros(
            batch_size, dtype=torch.bool, device=device
        )

        for parent_id, parent_name in enumerate(self.parent_names):
            sample_mask = parent_target == parent_id
            children = self.children_by_parent[parent_id]

            if torch.any(sample_mask):
                valid_children = torch.tensor(
                    children, dtype=torch.long, device=device
                )
                branch_target = leaf_target[sample_mask]
                valid = (
                    branch_target[:, None] == valid_children[None, :]
                ).any(dim=1)
                if not torch.all(valid):
                    raise ValueError(
                        "A sample assigned to '{}' has a leaf outside that "
                        "branch".format(parent_name)
                    )

            if self.holdout_mode == "batch":
                hidden_leaf = None
                present = torch.unique(leaf_target[sample_mask])
                can_hide = (
                    len(children) >= 2
                    and present.numel() >= 2
                    and (
                        parent_id == forced_parent
                        or torch.rand((), device=device).item()
                        < self.hide_prob
                    )
                )
                if can_hide:
                    selected = torch.randint(
                        0, present.numel(), (1,), device=device
                    )
                    hidden_leaf = int(present[int(selected.item())].item())
                hidden_by_parent[parent_id] = hidden_leaf
            else:
                hidden_leaf = hidden_by_parent[parent_id]

            hidden_mask = torch.zeros_like(sample_mask)
            if hidden_leaf is not None:
                hidden_mask = sample_mask & (leaf_target == hidden_leaf)
                pseudo_mask |= hidden_mask

            collapse = False if parent_id == forced_parent else (
                torch.rand((), device=device).item() < self.collapse_prob
            )
            if collapse:
                parent_label = len(label_names)
                label_names.append(parent_name)
                target[sample_mask] = parent_label
                continue

            active_leaf_to_label = {}
            for leaf_id in children:
                if leaf_id == hidden_leaf:
                    continue
                active_leaf_to_label[leaf_id] = len(label_names)
                label_names.append(self.leaf_names[leaf_id])

            if hidden_leaf is not None:
                unknown_label = len(label_names)
                label_names.append(self.unknown_template.format(parent_name))
                target[hidden_mask] = unknown_label

            for leaf_id, label_id in active_leaf_to_label.items():
                active_mask = sample_mask & (leaf_target == leaf_id)
                target[active_mask] = label_id

        if torch.any(target < 0):
            bad_indices = (
                torch.where(target < 0)[0].detach().cpu().tolist()
            )
            raise RuntimeError(
                "Open treecut did not assign targets to batch indices {}"
                .format(bad_indices)
            )
        if self.force_one_pseudo_per_batch and not torch.any(pseudo_mask):
            raise RuntimeError(
                "Open treecut failed to create the forced pseudo-unseen "
                "samples"
            )

        return {
            "label_names": label_names,
            "target": target,
            "pseudo_mask": pseudo_mask,
            "hidden_by_parent": hidden_by_parent,
            "hidden_leaf_by_parent": hidden_by_parent,
            "forced_parent": forced_parent,
            "holdout_mode": self.holdout_mode,
            "episode_epoch": self.epoch,
        }
