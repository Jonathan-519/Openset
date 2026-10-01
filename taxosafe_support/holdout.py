"""Strict TRAIN-only class holdouts using the shared main training pipeline.

A held-out species/parent is absent from that fold's gradient samples, candidate
classes and reference bank. This is distinct from removing references after a
model has already been trained on the held-out species. Original manifests and
taxonomy indices are never edited. Scores are diagnostic, uncalibrated scores;
these folds do not prove generalization to arbitrary real unknown classes.
"""
import copy
import hashlib
import time

import torch

from . import protocol
from .episodes import build_episodes

SCHEMA_VERSION = "strict_train_holdout_v1"


def _rank(seed, value):
    return hashlib.sha256((str(seed) + ":" + str(value)).encode("utf-8")).hexdigest()


def _validate_train(rows, meta):
    if not rows:
        raise ValueError("The known TRAIN manifest is empty")
    protocol.audit_rows({"train": rows})
    count = len(meta["leaf_to_parent"])
    by_leaf = {c: [] for c in range(count)}
    for i, row in enumerate(rows):
        if row.get("status") != "known" or row.get("split") != "train":
            raise ValueError("Strict holdouts accept known TRAIN rows only")
        leaf = row.get("true_leaf")
        if not isinstance(leaf, int) or leaf not in by_leaf:
            raise ValueError("TRAIN contains an invalid leaf label")
        if row.get("true_parent") != meta["leaf_to_parent"][leaf]:
            raise ValueError("TRAIN parent label differs from the locked taxonomy")
        by_leaf[leaf].append(i)
    if any(not indices for indices in by_leaf.values()):
        raise ValueError("Every original known leaf must occur in TRAIN before holdout")
    return by_leaf


def build_folds(rows, meta, seed=1, kind="both", known_val_fraction=.2,
                min_train_per_leaf=2):
    """Create all eligible folds deterministically using only TRAIN identities.

    Internal known validation reserves images only when at least
    ``min_train_per_leaf`` remain for gradients. Sparse leaves stay in training
    and are explicitly reported as lacking internal known validation coverage.
    Species folds with a single-known-leaf parent are deliberately not invented.
    """
    if kind not in ("species", "parent", "both"):
        raise ValueError("kind must be species, parent or both")
    if not 0 < float(known_val_fraction) < 1 or int(min_train_per_leaf) < 1:
        raise ValueError("Invalid inner validation fraction or training minimum")
    by_leaf = _validate_train(rows, meta)
    mapping = list(meta["leaf_to_parent"])
    parent_ids = sorted(set(mapping))
    if parent_ids != list(range(len(meta["parent_names"]))):
        raise ValueError("Taxonomy parent indices must be contiguous")
    leaves_by_parent = {p: [c for c, value in enumerate(mapping) if value == p] for p in parent_ids}
    candidates = []
    if kind in ("species", "both"):
        candidates.extend(("species", c, [c]) for c in by_leaf if len(leaves_by_parent[mapping[c]]) > 1)
    if kind in ("parent", "both") and len(parent_ids) > 1:
        candidates.extend(("parent", p, leaves_by_parent[p]) for p in parent_ids)
    folds = []
    for fold_kind, node, held_leaves in candidates:
        active = [c not in held_leaves for c in by_leaf]
        train_indices, validation_indices, held_indices, missing_validation = [], [], [], []
        for c, indices in by_leaf.items():
            if not active[c]:
                held_indices.extend(indices)
                continue
            ordered = sorted(indices, key=lambda i: _rank(seed, rows[i]["image_sha256"]))
            capacity = max(0, len(ordered) - int(min_train_per_leaf))
            reserve = min(capacity, max(1, int(len(ordered) * float(known_val_fraction)))) if capacity else 0
            validation_indices.extend(ordered[:reserve])
            train_indices.extend(ordered[reserve:])
            if not reserve:
                missing_validation.append(c)
        # Fold checkpoint selection cannot silently reuse gradient images.
        if not validation_indices:
            continue
        folds.append({"schema_version": SCHEMA_VERSION,
                      "id": ("species_%03d" if fold_kind == "species" else "parent_%03d") % node,
                      "kind": fold_kind, "node": node, "seed": int(seed),
                      "heldout_leaf_ids": sorted(held_leaves), "active_leaf_mask": active,
                      "target_parent": mapping[node] if fold_kind == "species" else None,
                      "train_indices": sorted(train_indices), "val_known_indices": sorted(validation_indices),
                      "heldout_indices": sorted(held_indices),
                      "inner_validation_missing_leaf_ids": missing_validation,
                      "known_val_fraction": float(known_val_fraction),
                      "min_train_per_leaf": int(min_train_per_leaf)})
    return folds


def select_folds(folds, seed=1, fold_ids=(), max_folds=0):
    """Select declared folds reproducibly, without looking at image scores."""
    fold_ids = tuple(fold_ids)
    requested = set(fold_ids)
    if len(requested) != len(fold_ids):
        raise ValueError("Duplicate fold IDs")
    by_id = {fold["id"]: fold for fold in folds}
    if requested - set(by_id):
        raise ValueError("Unknown or ineligible fold IDs: " + ", ".join(sorted(requested - set(by_id))))
    selected = [by_id[name] for name in sorted(requested)] if requested else list(folds)
    if int(max_folds) < 0:
        raise ValueError("max_folds must be nonnegative; zero means all")
    if max_folds and len(selected) > int(max_folds):
        selected = sorted(selected, key=lambda fold: _rank(seed, fold["id"]))[:int(max_folds)]
    return sorted(selected, key=lambda fold: fold["id"])


def build_active_episodes(labels, leaf_to_parent, active_leaf_mask, seed=0):
    """Create interventions within the remaining tree, retaining global IDs.

    Building on the full tree and AND-ing its masks afterwards would permit
    controls to 'remove' already absent leaves and fabricate singleton siblings.
    The temporary compressed numbering here is internal; returned masks and
    targets use exactly the original root/parent/leaf output indices.
    """
    labels = torch.as_tensor(labels, dtype=torch.long)
    mapping = torch.as_tensor(leaf_to_parent, dtype=torch.long, device=labels.device)
    active = torch.as_tensor(active_leaf_mask, dtype=torch.bool, device=labels.device)
    if active.shape != mapping.shape or not bool(active.any()):
        raise ValueError("The fold must retain at least one known leaf")
    if labels.ndim != 1 or bool(((labels < 0) | (labels >= len(mapping))).any()) or not bool(active[labels].all()):
        raise ValueError("A held-out label reached the gradient episode")
    leaves = torch.where(active)[0]
    parents = torch.unique(mapping[leaves], sorted=True)
    inverse_leaf = torch.full_like(mapping, -1)
    inverse_leaf[leaves] = torch.arange(len(leaves), device=labels.device)
    inverse_parent = torch.full((int(mapping.max()) + 1,), -1, dtype=torch.long, device=labels.device)
    inverse_parent[parents] = torch.arange(len(parents), device=labels.device)
    compressed = build_episodes(inverse_leaf[labels], inverse_parent[mapping[leaves]], seed=seed)
    masks, targets = {}, {}
    parent_count = int(mapping.max()) + 1
    for name, mask in compressed["masks"].items():
        restored = torch.zeros(len(labels), len(mapping), dtype=torch.bool, device=labels.device)
        restored[:, leaves] = mask
        masks[name] = restored
        target = compressed["targets"][name]
        result = torch.zeros_like(target)
        is_parent = (target > 0) & (target <= len(parents))
        is_leaf = target > len(parents)
        result[is_parent] = 1 + parents[target[is_parent] - 1]
        result[is_leaf] = 1 + parent_count + leaves[target[is_leaf] - 1 - len(parents)]
        targets[name] = result
    return {"masks": masks, "valid": compressed["valid"], "targets": targets}


def fold_rows(rows, fold, meta):
    """Validate a declared fold and return independent train/inner-val/query lists."""
    _validate_train(rows, meta)
    names = ("train_indices", "val_known_indices", "heldout_indices")
    sets = [set(fold[name]) for name in names]
    if any(len(indices) != len(fold[name]) for name, indices in zip(names, sets)):
        raise ValueError("A fold contains repeated manifest indices")
    if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Fold gradient, inner validation and heldout queries overlap")
    if set.union(*sets) != set(range(len(rows))):
        raise ValueError("Fold must partition the entire original TRAIN manifest")
    active = fold["active_leaf_mask"]
    if len(active) != len(meta["leaf_to_parent"]) or not any(active):
        raise ValueError("Invalid active leaf mask")
    expected_held = {c for c, allowed in enumerate(active) if not allowed}
    if expected_held != set(fold["heldout_leaf_ids"]):
        raise ValueError("Heldout leaf IDs and active mask differ")
    groups = {"train": [copy.deepcopy(rows[i]) for i in fold["train_indices"]],
              "val_known": [copy.deepcopy(rows[i]) for i in fold["val_known_indices"]]}
    queries = [copy.deepcopy(rows[i]) for i in fold["heldout_indices"]]
    if not groups["train"] or not groups["val_known"] or not queries:
        raise ValueError("Every strict fold needs gradients, independent inner validation and heldout queries")
    if any(not active[r["true_leaf"]] for part in groups.values() for r in part):
        raise ValueError("Heldout class leaked into gradient/selection samples")
    if any(active[r["true_leaf"]] for r in queries):
        raise ValueError("An active class appears among heldout queries")
    if fold["kind"] == "species":
        if len(expected_held) != 1:
            raise ValueError("Species folds remove exactly one leaf")
        parent = meta["leaf_to_parent"][next(iter(expected_held))]
        if parent != fold["target_parent"] or not any(active[c] and p == parent for c, p in enumerate(meta["leaf_to_parent"])):
            raise ValueError("A heldout species must retain a sibling in its parent")
    elif fold["kind"] == "parent":
        removed = {c for c, parent in enumerate(meta["leaf_to_parent"]) if parent == fold["node"]}
        if expected_held != removed:
            raise ValueError("A parent fold must remove every descendant leaf")
    else:
        raise ValueError("Unsupported fold kind")
    audit = protocol.audit_rows(dict(groups, heldout=queries))
    for split in audit:
        audit[split]["origin"] = "known_train_manifest"
    return groups, queries, audit


def _json_scores(tensor):
    if bool(torch.isnan(tensor).any() | torch.isposinf(tensor).any()):
        raise ValueError("Non-finite holdout evidence; only inactive negative infinity is permitted")
    return [float(value) if torch.isfinite(value) else None for value in tensor.detach().cpu()]


@torch.no_grad()
def score_fold(encoder, evidence, bank, cfg, meta, groups, fold, device):
    """Frozen uncalibrated evaluation; no score fitting on heldout images."""
    from . import pipeline
    encoder.eval()
    evidence.eval()
    active = torch.tensor(fold["active_leaf_mask"], dtype=torch.bool, device=device)
    offset = 1 + len(meta["parent_names"])
    text_features = encoder.text_features()
    records = []
    for role, rows in groups.items():
        seen = []
        for images, _, indices in pipeline.make_loader(rows, cfg, meta):
            encoded = encoder.encode(images.to(device), text_features=text_features)
            mask = active[None].expand(len(images), -1)
            output = evidence(encoded, bank, mask=mask)
            logits = encoded["leaf_logits"].masked_fill(~mask, -torch.inf)
            predictions = output["log_probs"].argmax(1)
            for i, index in enumerate(indices.tolist()):
                row = rows[index]
                node = int(predictions[i])
                kind = "root" if node == 0 else "parent" if node < offset else "leaf"
                expected = offset + row["true_leaf"] if role == "val_known" else (
                    1 + fold["target_parent"] if fold["kind"] == "species" else 0)
                record = {key: row.get(key) for key in
                          ("path", "source", "image_sha256", "true_leaf", "true_parent")}
                record.update(evaluation_role=role, source_split="train", fold_id=fold["id"],
                              expected_node=int(expected), predicted_node=node, prediction_type=kind,
                              correct=bool(node == expected), closed_pred_leaf=int(logits[i].argmax()),
                              log_probs=_json_scores(output["log_probs"][i]),
                              root_knownness=float(torch.sigmoid(output["root_logit"][i])))
                if "parent_membership_logits" in output:
                    candidate_leaf = record["closed_pred_leaf"]
                    candidate_parent = meta["leaf_to_parent"][candidate_leaf]
                    record.update(candidate_leaf=candidate_leaf, candidate_parent=candidate_parent,
                                  parent_membership_logits=_json_scores(output["parent_membership_logits"][i]),
                                  leaf_membership_logits=_json_scores(output["leaf_membership_logits"][i]),
                                  candidate_parent_membership=float(torch.sigmoid(output["parent_membership_logits"][i, candidate_parent])),
                                  candidate_leaf_membership=float(torch.sigmoid(output["leaf_membership_logits"][i, candidate_leaf])))
                records.append(record)
            seen.extend(indices.tolist())
        if seen != list(range(len(rows))):
            raise ValueError("Holdout inference did not visit the declared rows exactly once")
    known = [r for r in records if r["evaluation_role"] == "val_known"]
    held = [r for r in records if r["evaluation_role"] == "heldout"]
    report = {"fold_id": fold["id"], "kind": fold["kind"], "schema_version": SCHEMA_VERSION,
              "evaluation_unit": "unique_train_image_sha256", "decoder": "uncalibrated_joint_argmax",
              "calibration_used": False, "diagnostic_only_decoder": True,
              "heldout_used_for_checkpoint_selection": False,
              "known_count": len(known), "heldout_count": len(held),
              "known_closed_accuracy": sum(r["closed_pred_leaf"] == r["true_leaf"] for r in known) / len(known),
              "known_e2e": sum(r["correct"] for r in known) / len(known),
              "heldout_correct_count": sum(r["correct"] for r in held),
              "heldout_correct_rate": sum(r["correct"] for r in held) / len(held),
              "heldout_target": "correct_parent" if fold["kind"] == "species" else "root",
              "heldout_prediction_counts": {kind: sum(r["prediction_type"] == kind for r in held)
                                             for kind in ("root", "parent", "leaf")},
              "inactive_log_probability_encoding": "null denotes impossible inactive taxonomy node"}
    return records, report


def run_fold(cfg, directory, device, fold, rows, meta):
    """Train one fresh model through pipeline.train_rows, then score heldout once."""
    from . import pipeline
    groups, queries, audit = fold_rows(rows, fold, meta)
    fold_cfg = copy.deepcopy(cfg)
    fold_cfg["strict_holdout"] = {key: copy.deepcopy(value) for key, value in fold.items()}
    fold_cfg["strict_holdout"]["input_hashes"] = {key: value["image_hashes"] for key, value in audit.items()}
    sig = protocol.signature(fold_cfg)
    fold_dir = protocol.claim_stage(directory, fold["id"])
    protocol.write_json(fold_dir / "fold.json", {"fold": fold, "audit": audit, "signature": sig})
    started = time.perf_counter()
    pipeline.train_rows(fold_cfg, fold_dir, device, groups, meta,
                        {key: audit[key] for key in groups}, sig,
                        active_leaf_mask=fold["active_leaf_mask"],
                        selection_splits=["known_train_inner_validation"])
    encoder, evidence, trained, bank = pipeline.load_trained(fold_cfg, fold_dir, device)
    expected_hashes = set(audit["train"]["image_hashes"])
    if not set(bank.hashes).issubset(expected_hashes) or set(bank.hashes) & set(audit["heldout"]["image_hashes"]):
        raise ValueError("The strict-fold reference cache contains forbidden content")
    records, report = score_fold(encoder, evidence, bank, fold_cfg, meta,
                                 {"val_known": groups["val_known"], "heldout": queries}, fold, device)
    protocol.require_signature(sig, protocol.signature(fold_cfg))
    protocol.write_records(fold_dir / "predictions.jsonl", records)
    protocol.write_json(fold_dir / "metrics.json", report)
    protocol.write_json(fold_dir / "completed.json", {
        "schema_version": SCHEMA_VERSION, "fold_id": fold["id"], "signature": sig,
        "training_checkpoint_sha256": trained["checkpoint"]["sha256"],
        "support_sha256": trained["support"]["sha256"],
        "gradient_origin": "known_train_only_excluding_heldout_classes",
        "heldout_used_for_checkpoint_selection": False, "real_unknown_or_test_inputs_used": False,
        "elapsed_seconds": time.perf_counter() - started,
        "predictions_sha256": protocol.file_hash(fold_dir / "predictions.jsonl"),
        "metrics_sha256": protocol.file_hash(fold_dir / "metrics.json")})
    return report


def run_validation(cfg, directory, device, folds, rows, meta):
    """Execute declared folds sequentially in an immutable experiment directory."""
    if not folds:
        raise ValueError("No eligible strict folds; inspect TRAIN per-class image counts")
    output = protocol.claim_stage(directory, "holdout")
    protocol.write_json(output / "plan.json", {"schema_version": SCHEMA_VERSION, "folds": folds,
                        "signature": protocol.signature(cfg), "meta": meta,
                        "source_manifest": str(cfg["data"]["train"]),
                        "source_manifest_sha256": protocol.file_hash(protocol.resolve(cfg["data"]["train"])),
                        "real_unknown_or_test_inputs_used": False})
    reports = []
    for fold in folds:
        print("Strict TRAIN holdout: " + fold["id"], flush=True)
        reports.append(run_fold(cfg, output, device, fold, rows, meta))
    by_kind = {}
    for kind in ("species", "parent"):
        selected = [report for report in reports if report["kind"] == kind]
        count = sum(report["heldout_count"] for report in selected)
        by_kind[kind] = {"fold_count": len(selected), "heldout_evaluations": count,
                         "micro_correct_rate": (sum(report["heldout_correct_count"] for report in selected) / count
                                                if count else None),
                         "macro_fold_correct_rate": (sum(report["heldout_correct_rate"] for report in selected) / len(selected)
                                                     if selected else None)}
    summary = {"schema_version": SCHEMA_VERSION, "reports": reports, "by_kind": by_kind,
               "formal_test_used": False, "real_unknown_data_used": False,
               "interpretation": "Independent known-TRAIN held-class transfer diagnostics; not final-test performance or an OOD guarantee"}
    protocol.write_json(output / "summary.json", summary)
    protocol.write_json(output / "completed.json", {"schema_version": SCHEMA_VERSION,
                        "fold_ids": [fold["id"] for fold in folds],
                        "summary_sha256": protocol.file_hash(output / "summary.json")})
    return summary
