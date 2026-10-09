"""TRAIN-only support interventions with recomputed D05 and spatial evidence.

The query fold/leaf/parent is removed BEFORE geometry fitting, reference
selection, rival margins, templates and normalization inputs. The pretrained
D05 normalization is restored unchanged, never fitted with DEV/TEST.
"""
import hashlib

import torch

from taxosafe_discovery.geometry import GeometryBank
from taxosafe_discovery.verifier import SharedVerifier, _balanced_weights
from taxosafe_discovery.backend import _templates
from .support_bank import select_references

KINDS = {"full": 0, "drop_leaf": 1, "drop_parent": 2, "unrelated_leaf_removal": 3, "unrelated_parent_removal": 4}


@torch.no_grad()
def build_episodes(view, source_payload, meta, settings, seed):
    features, labels, hashes = view["raw_clip"], view["labels"], view["image_sha256"]
    rows = view["records"]
    if (not rows or any(r["split"] != "train" or r["status"] != "known" for r in rows)
            or len(hashes) != len(set(hashes))):
        raise ValueError("Spatial episodes accept only unique known TRAIN")
    leaves, parents = len(meta["leaf_names"]), len(meta["parent_names"])
    if bool((torch.bincount(labels, minlength=leaves) < 2).any()) or parents < 2:
        raise ValueError("Disjoint episodes require at least 2 images/species and 2 parents")
    folds = settings["folds"]
    generator = torch.Generator().manual_seed(seed)
    assignment = torch.empty(len(labels), dtype=torch.long)
    for leaf in range(leaves):
        ix = (labels == leaf).nonzero(as_tuple=True)[0]
        ix = ix[torch.randperm(len(ix), generator=generator)]
        assignment[ix] = torch.arange(len(ix)) % min(folds, len(ix))
    mapping = torch.tensor(meta["leaf_to_parent"])
    parent_labels = mapping[labels]
    verifier = SharedVerifier.from_state_dict(source_payload["verifier"])
    templates = _templates(features, source_payload["text"])
    shrinkage = source_payload["geometry"]["shrinkage"]
    table = {level: [] for level in ("leaf", "parent")}
    episodes = []

    def add(support_mask, query_mask, kind, identity):
        si, qi = support_mask.nonzero(as_tuple=True)[0], query_mask.nonzero(as_tuple=True)[0]
        if not len(qi):
            return
        if not len(si) or bool((support_mask & query_mask).any()):
            raise ValueError("Query must be excluded before ANY support statistics")
        sh, qh = [hashes[i] for i in si.tolist()], [hashes[i] for i in qi.tolist()]
        bank = GeometryBank.fit(features[si], features[si], labels[si], sh, meta, shrinkage=shrinkage)
        evidence = bank.score(features[qi], features[qi], qh)
        scores = verifier.score(evidence, {key: value[qi] for key, value in templates.items()})
        references = select_references(features, labels, hashes, support_mask, settings["references_per_leaf"])
        episode = dict(id=len(episodes), kind=kind, identity=identity, support_indices=si.tolist(),
                       query_indices=qi.tolist(), reference_indices=references.tolist(),
                       support_image_sha256=sh, query_image_sha256=qh,
                       active_leaves=evidence["leaf_active"].nonzero(as_tuple=True)[0].tolist(),
                       parent_near_examples_skipped=0)
        for level, targets in (("leaf", labels), ("parent", parent_labels)):
            active = evidence[level + "_active"]
            candidates = active.nonzero(as_tuple=True)[0].tolist()
            for j, query in enumerate(qi.tolist()):
                if kind == "drop_leaf" and level == "parent" and not bool(active[targets[query]]):
                    episode["parent_near_examples_skipped"] += 1
                    continue
                for candidate in candidates:
                    table[level].append(dict(episode=episode["id"], query=query, candidate=candidate,
                        target=float(candidate == int(targets[query])), source_leaf=int(labels[query]),
                        kind=KINDS[kind], base=float(scores[level + "_scores"][j, candidate]),
                        ce_allowed=kind in ("full", "unrelated_leaf_removal", "unrelated_parent_removal")))
        episodes.append(episode)

    for fold in range(folds):
        add(assignment != fold, assignment == fold, "full", fold)
        # Removal choices depend on fold and seed, NOT the query's class or
        # outcome. These known controls deny a deterministic count->unknown cue.
        removed_leaf = (fold + seed) % leaves
        removed_parent = (fold + seed) % parents
        add((assignment != fold) & (labels != removed_leaf),
            (assignment == fold) & (labels != removed_leaf), "unrelated_leaf_removal", removed_leaf)
        add((assignment != fold) & (parent_labels != removed_parent),
            (assignment == fold) & (parent_labels != removed_parent), "unrelated_parent_removal", removed_parent)
    for leaf in range(leaves):
        add(labels != leaf, labels == leaf, "drop_leaf", leaf)
    for parent in range(parents):
        add(parent_labels != parent, parent_labels == parent, "drop_parent", parent)
    for level, values in table.items():
        weights = _balanced_weights(torch.tensor([r["target"] for r in values]),
                                    torch.tensor([r["source_leaf"] for r in values]),
                                    torch.tensor([r["kind"] for r in values]))
        for row, weight in zip(values, weights.tolist()):
            row["weight"] = weight
    report = dict(schema_version="morphology_support_episodes_v1", seed=seed, folds=folds,
                  fit_split="known_train", train_count=len(hashes),
                  image_hash_digest=hashlib.sha256("\n".join(sorted(hashes)).encode()).hexdigest(),
                  true_unknown_images_used=False, dev_images_used=False, test_images_used=False,
                  encoder_unseen_class_claim=False, old_scores_recomputed_after_support_exclusion=True,
                  support_count_or_episode_id_provided_to_network=False,
                  references_balanced_per_active_leaf=settings["references_per_leaf"],
                  singleton_reference_fallback="use its only real individual, no synthetic animal",
                  unrelated_removal_controls=True, support_size_shortcuts_proven_absent=False,
                  single_child_parent_ids=[p for p in range(parents) if int((mapping == p).sum()) == 1],
                  single_child_parent_near_examples_skipped=sum(e["parent_near_examples_skipped"] for e in episodes),
                  examples={level: len(values) for level, values in table.items()}, episodes=episodes)
    return dict(examples=table, report=report)
