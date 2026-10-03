"""TRAIN-only normalization and candidate-specific frozen parent evidence.

Parent text scores are learned-prompt encoder logits, not zero-shot CLIP
scores. They remain separate from the support-bank parent identity logits.
All score families have one shared TRAIN median/MAD transform across parents;
the parent transforms are fitted on each known TRAIN image's true parent.
No DEV/TEST labels enter collection or transformation.
"""
import json
import time

import torch

from taxosafe_geometry.core import (
    HierarchicalGeometry, RobustScoreStandardizer, _distance, _normalized,
)
from taxosafe_refine.pipeline import _frozen, encode_baseline
from taxosafe_support import pipeline as support_pipeline
from taxosafe_support.calibration import raw_records, unique_records
from taxosafe_support.membership_calibration import RAW_FIELDS, candidate_scores


BASE_SCORE_NAMES = ("baseline_parent", "baseline_leaf", "geometry_parent", "geometry_leaf")
PARENT_SCORE_NAMES = ("parent_text", "parent_membership", "parent_geometry")
SCORE_NAMES = BASE_SCORE_NAMES + PARENT_SCORE_NAMES


class ParentEvidenceGeometry(HierarchicalGeometry):
    """Historical geometry with an explicit all-parent RMD interface.

The original state schema and candidate score arithmetic are retained.
Changing the candidate parent never reuses another parent's distance.
"""

    @torch.no_grad()
    def score_parents(self, parent):
        """Return finite detached [B,P] RMD, in the taxonomy's parent order."""
        raw = torch.as_tensor(parent)
        dtype = torch.float64 if raw.dtype == torch.float64 else torch.float32
        features = _normalized(raw, "parent", raw.device, dtype,
                               dimension=self.parent_dimension)
        state = self._on_device(raw.device, dtype)
        background = _distance(features, state["parent_global_mean"],
                               state["parent_global_cholesky"])
        values = torch.stack([
            background - _distance(features, state["parent_means"][p],
                                   state["parent_within_cholesky"])
            for p in range(self.num_parents)
        ], dim=1)
        if values.shape != (len(features), self.num_parents) or not bool(torch.isfinite(values).all()):
            raise ValueError("All-parent geometry must be finite [B,P] evidence")
        return values


def _parent_scores(records, parent, meta, geometry):
    count = len(meta["parent_names"])
    values = {}
    for name, container, field in (
            ("parent_text", "encoder_evidence", "parent_text_logits"),
            ("parent_membership", "support_evidence", "parent_membership_logits")):
        try:
            values[name] = torch.tensor([r[container][field] for r in records], dtype=torch.float64)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Missing finite parent evidence: " + container + "." + field) from error
        if not records:
            values[name] = torch.empty((0, count), dtype=torch.float64)
    values["parent_geometry"] = geometry.score_parents(parent).detach().cpu()
    for name, value in values.items():
        if value.shape != (len(records), count) or not bool(torch.isfinite(value).all()):
            raise ValueError("Invalid finite all-parent evidence: " + name)
    return values


def _candidate_scores(records, fine, parent, meta, geometry):
    candidates = candidate_scores(records, meta)
    selected = geometry.score(
        fine, parent,
        torch.as_tensor(candidates["parent"], dtype=torch.long, device=parent.device),
        torch.as_tensor(candidates["leaf"], dtype=torch.long, device=fine.device))
    scores = {
        "baseline_parent": torch.as_tensor(candidates["parent_score"], dtype=torch.float64),
        "baseline_leaf": torch.as_tensor(candidates["leaf_score"], dtype=torch.float64),
        "geometry_parent": selected["parent_score"].detach().cpu(),
        "geometry_leaf": selected["leaf_score"].detach().cpu(),
    }
    for name, value in scores.items():
        if value.shape != (len(records),) or not bool(torch.isfinite(value).all()):
            raise ValueError("Invalid finite candidate evidence: " + name)
    return candidates, scores


@torch.no_grad()
def add_evidence(records, fine, parent, meta, geometry, scales):
    """Attach existing geometry channels and all-parent channels in place.

This inference function only consumes frozen features and scores. Truth,
status, source, and split fields do not affect any score or candidate.
"""
    if set(scales) != set(SCORE_NAMES):
        raise ValueError("Parent evidence requires all seven TRAIN-fitted score scales")
    _, scores = _candidate_scores(records, fine, parent, meta, geometry)
    normalized = {key: scales[key].transform(value) for key, value in scores.items()}
    parents = _parent_scores(records, parent, meta, geometry)
    parent_z = {key: scales[key].transform(value) for key, value in parents.items()}
    for i, row in enumerate(records):
        row.update(
            baseline_parent_z=float(normalized["baseline_parent"][i]),
            baseline_leaf_z=float(normalized["baseline_leaf"][i]),
            geometry_parent_score=float(normalized["geometry_parent"][i]),
            geometry_leaf_score=float(normalized["geometry_leaf"][i]),
            geometry_parent_raw=float(scores["geometry_parent"][i]),
            geometry_leaf_raw=float(scores["geometry_leaf"][i]),
            baseline_parent_raw=float(scores["baseline_parent"][i]),
            baseline_leaf_raw=float(scores["baseline_leaf"][i]))
        row["parent_evidence"] = {
            "text_z": parent_z["parent_text"][i].tolist(),
            "membership_z": parent_z["parent_membership"][i].tolist(),
            "geometry_z": parent_z["parent_geometry"][i].tolist(),
            "geometry_raw": parents["parent_geometry"][i].tolist(),
        }
    return records


@torch.no_grad()
def fit_evidence(train_records, cached_features, meta, cfg_geometry):
    """Fit known-TRAIN statistics, attach training scores, return model/scales/audit.

``cached_features`` contains aligned unique ``fine``, ``parent`` tensors and
``image_sha256``. Returns ``(geometry, scales, audit)``. The audit has one
entry per scale with exact TRAIN content hashes and the selection policy.
    """
    train_records = list(train_records)
    # The historical deduplicator validates support heads but does not know
    # this namespace. Check every alias before it can hide conflicting text.
    seen_text = {}
    for row in train_records:
        try:
            text = json.dumps(row.get("encoder_evidence"), sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError("Invalid finite encoder_evidence on known TRAIN") from error
        digest = row["image_sha256"]
        if digest in seen_text and seen_text[digest] != text:
            raise ValueError("Same TRAIN content has inconsistent encoder_evidence: " + digest)
        seen_text[digest] = text
    records = unique_records(train_records)
    if not records or any(r.get("split") != "train" or r.get("status") != "known" for r in records):
        raise ValueError("Evidence fitting requires nonempty known TRAIN only")
    hashes = [r["image_sha256"] for r in records]
    if hashes != list(cached_features["image_sha256"]):
        raise ValueError("Deduplicated TRAIN records and feature hashes differ")
    mapping = meta["leaf_to_parent"]
    for row in records:
        label = row.get("true_leaf")
        if (isinstance(label, bool) or not isinstance(label, int) or not 0 <= label < len(mapping)
                or row.get("true_parent") != mapping[label]):
            raise ValueError("Invalid known TRAIN leaf/parent annotations")
    labels = torch.tensor([r["true_leaf"] for r in records], dtype=torch.long)
    fine, parent = cached_features["fine"], cached_features["parent"]
    geometry = ParentEvidenceGeometry.fit(fine, parent, labels, mapping, **cfg_geometry)
    candidates, scores = _candidate_scores(records, fine, parent, meta, geometry)
    truth_parent = torch.as_tensor(mapping, dtype=torch.long)[labels]
    correct_parent = torch.as_tensor(candidates["parent"], dtype=torch.long) == truth_parent
    correct_leaf = correct_parent & (torch.as_tensor(candidates["leaf"], dtype=torch.long) == labels)
    if not bool(correct_parent.any()) or not bool(correct_leaf.any()):
        raise ValueError("TRAIN needs correct parent and leaf candidates for existing geometry normalization")
    scales, audit = {}, {}
    for key in BASE_SCORE_NAMES:
        mask = correct_parent if key.endswith("parent") else correct_leaf
        scales[key] = RobustScoreStandardizer.fit(scores[key][mask])
        audit[key] = {
            "count": int(mask.sum()), "source_split": "train",
            "selection": "correct_parent_candidate" if key.endswith("parent") else "correct_leaf_candidate",
            "image_sha256": [digest for digest, selected in zip(hashes, mask.tolist()) if selected],
        }
    parents = _parent_scores(records, parent, meta, geometry)
    indices = torch.arange(len(records))
    for key in PARENT_SCORE_NAMES:
        scales[key] = RobustScoreStandardizer.fit(parents[key][indices, truth_parent])
        audit[key] = {
            "count": len(records), "source_split": "train",
            "selection": "true_parent_per_known_train_image",
            "parameter_sharing": "one_transform_per_score_family_shared_across_parents",
            "image_sha256": list(hashes),
        }
    add_evidence(records, fine, parent, meta, geometry, scales)
    return geometry, scales, audit


def _support_audits(bank, output, hashes, meta):
    """Measure effective references after exclusions, using CUDA-safe sums."""
    allowed = bank.permitted(len(hashes), query_hashes=hashes)
    if "reference_allowed" in output and not torch.equal(allowed, output["reference_allowed"]):
        raise ValueError("Reference evidence permissions differ from query-content exclusions")
    # Avoid integer CUDA matrix multiplication; bool reductions support the
    # older ProTeCt environment and preserve exact integer counts.
    leaf_counts = torch.stack([
        (allowed & (bank.labels == c)[None, :]).sum(1)
        for c in range(len(meta["leaf_names"]))
    ], dim=1)
    mapping = torch.as_tensor(meta["leaf_to_parent"], dtype=torch.long, device=allowed.device)
    parent_counts = torch.stack([
        leaf_counts[:, mapping == p].sum(1) for p in range(len(meta["parent_names"]))
    ], dim=1)
    if bool((leaf_counts == 0).any()):
        affected = torch.nonzero(leaf_counts == 0, as_tuple=False).cpu().tolist()
        raise ValueError(
            "Query-content exclusion leaves a known leaf without references; "
            "full-taxonomy decoding requires active support for every leaf. "
            "Do not pad inactive scores or disable self exclusion. "
            "Use a separately trained reference with more than one support image per leaf. "
            "Affected (batch row, leaf) IDs: " + str(affected))
    expected_leaves = leaf_counts > 0
    expected_parents = parent_counts > 0
    for name, expected in (("active_leaves", expected_leaves), ("active_parents", expected_parents)):
        if name in output and not torch.equal(output[name], expected):
            raise ValueError("Reference active taxonomy differs from effective support: " + name)
    original_leaf_counts = torch.bincount(mapping, minlength=len(meta["parent_names"])).cpu().tolist()
    records = []
    for i, digest in enumerate(hashes):
        excluded = sum(digest == h for h in bank.hashes)
        records.append({
            "query_in_support": bool(excluded), "excluded_self_references": excluded,
            "bank_references": len(bank.hashes), "effective_references": int(allowed[i].sum()),
            "effective_references_per_leaf": leaf_counts[i].cpu().tolist(),
            "effective_references_per_parent": parent_counts[i].cpu().tolist(),
            "known_leaves_per_parent": original_leaf_counts,
            "query_hash_exclusion_applied": True,
        })
    return records


@torch.no_grad()
def collect_features(groups, reference, device, geometry=None, scales=None, keep_features=True):
    """One frozen visual pass per unique image; always exclude its support hash.

Returns ``(records_by_split, cached_features_by_split, timing_and_audit)``.
Inactive full-taxonomy outputs fail closed. Strict class-holdout masked
taxonomies need an explicit adapter and are not silently accepted here.
    """
    _frozen(reference)
    if (geometry is None) != (scales is None):
        raise ValueError("Geometry and TRAIN score scales must be supplied together")
    text = reference.encoder.text_features()
    records, cached, timings = {}, {}, {}
    for split, rows in groups.items():
        rows = list(rows)
        unique = unique_records(rows)
        if not unique:
            raise ValueError("Empty parent-evidence split: " + split)
        seen, scored, fine_values, parent_values = [], [], [], []
        support_pipeline._sync(device)
        started = time.perf_counter()
        for images, _, indices in support_pipeline.make_loader(unique, reference.config, reference.meta):
            encoded = encode_baseline(reference.encoder, images.to(device), text_features=text)
            selected = [unique[i] for i in indices.tolist()]
            hashes = [row["image_sha256"] for row in selected]
            seen.extend(indices.tolist())
            fine = encoded["fine"].detach().float()
            parent = encoded["parent"].detach().float()
            if not bool(torch.isfinite(fine).all() and torch.isfinite(parent).all()):
                raise ValueError("Frozen parent evidence features must be finite")
            output = reference.evidence(encoded, reference.bank, query_hashes=hashes)
            support_audit = _support_audits(reference.bank, output, hashes, reference.meta)
            batch = raw_records(selected, {"log_probs": output["log_probs"].cpu().numpy()},
                                encoded["leaf_logits"].cpu().numpy(), reference.meta)
            raw = {key: output[key].detach().float().cpu().tolist() for key in RAW_FIELDS}
            parent_text = encoded["parent_logits"].detach().float().cpu()
            if (parent_text.shape != (len(batch), len(reference.meta["parent_names"]))
                    or not bool(torch.isfinite(parent_text).all())):
                raise ValueError("Encoder parent text scores must be finite and full-taxonomy")
            for i, record in enumerate(batch):
                record["support_evidence"] = {key: value[i] for key, value in raw.items()}
                record["encoder_evidence"] = {"parent_text_logits": parent_text[i].tolist()}
                record["support_audit"] = support_audit[i]
            candidate_scores(batch, reference.meta)  # Validate all heads before serializing.
            if geometry is not None:
                add_evidence(batch, fine, parent, reference.meta, geometry, scales)
            scored.extend(batch)
            if keep_features:
                fine_values.append(fine.cpu())
                parent_values.append(parent.cpu())
        support_pipeline._sync(device)
        elapsed = time.perf_counter() - started
        if seen != list(range(len(unique))):
            raise ValueError("Inference must visit unique images once in manifest order")
        by_hash = {row["image_sha256"]: row for row in scored}
        records[split] = [dict(by_hash[row["image_sha256"]], **row) for row in rows]
        if keep_features:
            cached[split] = {"fine": torch.cat(fine_values), "parent": torch.cat(parent_values),
                             "image_sha256": [row["image_sha256"] for row in unique]}
        timings[split] = {
            "manifest_rows": len(rows), "unique_images": len(unique), "seconds": elapsed,
            "seconds_per_unique_image": elapsed / len(unique), "visual_passes_per_unique_image": 1,
            "query_hash_exclusion_applied": True,
            "support_overlap_unique_images": sum(int(r["support_audit"]["query_in_support"]) for r in scored),
            "excluded_self_references": sum(r["support_audit"]["excluded_self_references"] for r in scored),
            "effective_references_min": min(r["support_audit"]["effective_references"] for r in scored),
            "effective_references_max": max(r["support_audit"]["effective_references"] for r in scored),
        }
    _frozen(reference)
    return records, cached, timings
