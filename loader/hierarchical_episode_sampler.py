"""Hierarchical episodic batch sampler for TaxoSafe.

Version 3 coordinates its pseudo-unseen leaf schedule with
``OpenTreecutGenerator``.  In epoch-holdout mode, every selected multi-leaf
parent contributes its fixed held-out leaf plus at least one active sibling.
"""

import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Sampler

from taxosafe_episode import epoch_holdout_leaf


class HierarchicalEpisodeBatchSampler(Sampler):
    """Yield parent/species-balanced batches for novelty episodes.

    Singleton parents are retained by sampling their only leaf with
    replacement.  They supervise parent and closed-set objectives but cannot
    contribute a sibling-novelty loss.
    """

    def __init__(
        self,
        dataset,
        leaf_to_parent,
        parents_per_batch=2,
        species_per_parent=2,
        images_per_species=2,
        batches_per_epoch=200,
        seed=1,
        ensure_rankable_parent=True,
        pseudo_holdout_mode="epoch",
        holdout_seed=None,
        leaf_names=None,
        hard_sibling_sampling=True,
    ):
        self.dataset = dataset
        self.parents_per_batch = int(parents_per_batch)
        self.species_per_parent = int(species_per_parent)
        self.images_per_species = int(images_per_species)
        self.batches_per_epoch = int(batches_per_epoch)
        self.seed = int(seed)
        self.holdout_seed = (
            self.seed if holdout_seed is None else int(holdout_seed)
        )
        self.ensure_rankable_parent = bool(ensure_rankable_parent)
        self.pseudo_holdout_mode = str(pseudo_holdout_mode).lower()
        self.hard_sibling_sampling = bool(hard_sibling_sampling)
        self.epoch = 0

        if self.pseudo_holdout_mode not in {"epoch", "batch"}:
            raise ValueError(
                "pseudo_holdout_mode must be 'epoch' or 'batch'"
            )
        for name, value in (
            ("parents_per_batch", self.parents_per_batch),
            ("species_per_parent", self.species_per_parent),
            ("images_per_species", self.images_per_species),
            ("batches_per_epoch", self.batches_per_epoch),
        ):
            if value <= 0:
                raise ValueError("{} must be positive".format(name))
        if self.species_per_parent < 2 and self.ensure_rankable_parent:
            raise ValueError(
                "species_per_parent must be at least 2 when "
                "ensure_rankable_parent=True"
            )

        self.leaf_to_parent = np.asarray(
            torch.as_tensor(leaf_to_parent).detach().cpu(), dtype=np.int64
        )
        if leaf_names is None:
            self.leaf_names = [str(index) for index in range(len(self.leaf_to_parent))]
        else:
            self.leaf_names = [str(name) for name in leaf_names]
            if len(self.leaf_names) != len(self.leaf_to_parent):
                raise ValueError("leaf_names and leaf_to_parent size differ")
        targets = np.asarray(dataset.target, dtype=np.int64)
        if targets.ndim != 1 or len(targets) != len(dataset):
            raise ValueError("dataset.target must contain one label per image")
        if targets.size == 0:
            raise ValueError("training dataset is empty")
        if targets.min() < 0 or targets.max() >= len(self.leaf_to_parent):
            raise ValueError("dataset contains an invalid known-leaf label")

        self.indices_by_leaf = defaultdict(list)
        for index, leaf_id in enumerate(targets.tolist()):
            self.indices_by_leaf[int(leaf_id)].append(index)

        self.leaves_by_parent = defaultdict(list)
        for leaf_id in sorted(self.indices_by_leaf):
            parent_id = int(self.leaf_to_parent[leaf_id])
            self.leaves_by_parent[parent_id].append(leaf_id)

        self.available_parents = sorted(self.leaves_by_parent)
        self.rankable_parents = [
            parent_id
            for parent_id in self.available_parents
            if len(self.leaves_by_parent[parent_id]) >= 2
        ]
        if len(self.available_parents) < self.parents_per_batch:
            raise ValueError(
                "Need at least {} represented parents, found {}".format(
                    self.parents_per_batch, len(self.available_parents)
                )
            )
        if self.ensure_rankable_parent and not self.rankable_parents:
            raise ValueError(
                "No parent contains at least two species for novelty episodes"
            )

    @property
    def batch_size(self):
        return (
            self.parents_per_batch
            * self.species_per_parent
            * self.images_per_species
        )

    def set_epoch(self, epoch):
        """Synchronise this sampler with the open-treecut generator."""
        self.epoch = int(epoch)

    def _select_parents(self, rng):
        selected = []
        if self.ensure_rankable_parent:
            selected.append(rng.choice(self.rankable_parents))
        candidates = [
            parent_id
            for parent_id in self.available_parents
            if parent_id not in selected
        ]
        selected.extend(
            rng.sample(candidates, self.parents_per_batch - len(selected))
        )
        rng.shuffle(selected)
        return selected

    def _taxonomic_distance(self, first, second):
        """Use genus when available, then the known parent neighbourhood."""
        if first == second:
            return 0
        first_genus = self.leaf_names[first].split("_", 1)[0].lower()
        second_genus = self.leaf_names[second].split("_", 1)[0].lower()
        if first_genus == second_genus:
            return 1
        if self.leaf_to_parent[first] == self.leaf_to_parent[second]:
            return 2
        return 3

    def _sample_other_leaves(self, leaves, count, rng, anchor=None):
        if count <= 0:
            return []
        if self.hard_sibling_sampling and anchor is not None:
            shuffled = list(leaves)
            rng.shuffle(shuffled)
            shuffled.sort(
                key=lambda leaf: self._taxonomic_distance(anchor, leaf)
            )
            if len(shuffled) >= count:
                return shuffled[:count]
        if len(leaves) >= count:
            return rng.sample(leaves, count)
        return rng.choices(leaves, k=count)

    def _select_leaves(self, parent_id, rng, episode_epoch):
        leaves = list(self.leaves_by_parent[parent_id])
        if len(leaves) == 1:
            return rng.choices(leaves, k=self.species_per_parent)

        if self.pseudo_holdout_mode == "epoch":
            held_leaf = epoch_holdout_leaf(
                leaves,
                parent_id,
                episode_epoch,
                self.holdout_seed,
            )
            active = [leaf for leaf in leaves if leaf != held_leaf]
            selected = [held_leaf]
            selected.extend(self._sample_other_leaves(
                active,
                self.species_per_parent - 1,
                rng,
                anchor=held_leaf,
            ))
            rng.shuffle(selected)
            return selected

        if len(leaves) >= self.species_per_parent:
            return rng.sample(leaves, self.species_per_parent)
        return rng.choices(leaves, k=self.species_per_parent)

    def __iter__(self):
        episode_epoch = int(self.epoch)
        rng = random.Random(self.seed + episode_epoch)
        for _ in range(self.batches_per_epoch):
            batch = []
            for parent_id in self._select_parents(rng):
                selected_leaves = self._select_leaves(
                    parent_id, rng, episode_epoch
                )
                for leaf_id in selected_leaves:
                    indices = self.indices_by_leaf[leaf_id]
                    if len(indices) >= self.images_per_species:
                        chosen = rng.sample(indices, self.images_per_species)
                    else:
                        chosen = rng.choices(
                            indices, k=self.images_per_species
                        )
                    batch.extend(chosen)
            if len(batch) != self.batch_size:
                raise RuntimeError(
                    "Generated batch has {} indices, expected {}".format(
                        len(batch), self.batch_size
                    )
                )
            yield batch

        # Preserve sensible behaviour for callers that do not explicitly call
        # set_epoch. train_taxosafe.py does call it, so this increment is reset
        # to the exact requested value at the next epoch.
        self.epoch = episode_epoch + 1

    def __len__(self):
        return self.batches_per_epoch
