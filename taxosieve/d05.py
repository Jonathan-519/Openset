"""The fixed D05 evidence component used internally by TaxoSieve.

This module has no experiment matrix or command-line entry point. It fits only
the original known-TRAIN geometry and BCE verifier, then scores immutable
reference candidates. The caller owns source receipts and the DEV/TEST gate.
"""
import copy
import hashlib

import torch

from taxosafe_support.io import object_hash
from taxosafe_support import calibration as base
from taxosafe_support import membership_calibration as membership
from .d05_geometry import GeometryBank
from .d05_verifier import SharedVerifier, build_episodes
from .tensor_utils import _features, _meta


ARM_ID = "D05_episode_bce"
MODEL_SPEC = dict(id=ARM_ID, kind="verifier", representation="clip",
                  method="bce", router="global", candidate="reference")
DEFAULT_SETTINGS = dict(seed=1, folds=3, epochs=100, batch_size=1024,
                        lr=.001, shrinkage=.1, hidden=32)
TEXT_KEYS = {"single_leaf", "single_parent", "ensemble_leaf", "ensemble_parent"}
FEATURE_KEYS = {"source_fine", "source_parent", "clip"}


def tensor_hash(value):
    """Historical Discovery tensor digest, kept byte-for-byte compatible."""
    value = value.detach().cpu().contiguous()
    return hashlib.sha256((str(value.dtype) + str(tuple(value.shape))).encode()
                          + value.numpy().tobytes()).hexdigest()


def _text_contract(cache):
    provenance = cache["provenance"]
    keys = ("clip_core_sha256", "clip_initialization", "templates_sha256", "preprocessing")
    return object_hash(dict(meta=cache["meta"],
        text={k: tensor_hash(v) for k, v in cache["text"].items()},
        core_and_prompts={k: provenance[k] for k in keys}))


def collect_cache(source, groups, device):
    """Collect the original four-text/three-feature cache for supplied splits."""
    from .d05_features import collect_cache as collect
    return collect(source, groups, device)


def _aligned_rows(group):
    rows = {row["image_sha256"]: row for row in group["records"]}
    return [rows[key] for key in group["image_sha256"]]


def _templates(feature, text):
    return {level: feature.float() @ text["ensemble_" + level].float().T
            for level in ("leaf", "parent")}


def _validate_text(text, meta, dimension=None):
    if not isinstance(text, dict) or set(text) != TEXT_KEYS:
        raise ValueError("D05 requires the complete frozen four-variant text cache")
    sizes = set()
    for key, value in text.items():
        count = len(meta["leaf_names"] if key.endswith("leaf") else meta["parent_names"])
        if (not torch.is_tensor(value) or not value.is_floating_point()
                or value.ndim != 2 or value.shape[0] != count or value.shape[1] < 1
                or not bool(torch.isfinite(value).all())):
            raise ValueError("Invalid frozen text features: " + key)
        sizes.add(value.shape[1])
    if len(sizes) != 1 or (dimension is not None and sizes != {dimension}):
        raise ValueError("Frozen text dimensions differ from D05 CLIP features")


def validate_cache(cache, meta=None, stage=None):
    """Validate tensor/record alignment; file and reference bindings are external.

    Keep the original cache layout so a legacy importer can verify all source
    fields before reusing them. This function never fits or reads a file.
    """
    if not isinstance(cache, dict):
        raise ValueError("Expected a D05 cache mapping")
    checked_meta = _meta(cache.get("meta"))
    if meta is not None and checked_meta != _meta(meta):
        raise ValueError("D05 cache taxonomy differs from the locked hierarchy")
    meta = checked_meta
    _validate_text(cache.get("text"), meta)
    groups = cache.get("groups")
    if not isinstance(groups, dict) or not groups:
        raise ValueError("D05 cache requires nonempty split groups")
    if stage is not None:
        if stage not in ("train", "development", "test"):
            raise ValueError("Unknown D05 cache stage")
        expected = ({"train"} if stage == "train" else
                    {("val_" if stage == "development" else "test_") + s
                     for s in base.STATUSES})
        if set(groups) != expected:
            raise ValueError("D05 cache splits differ from the requested stage")
    seen = set()
    for split, group in groups.items():
        if split == "train":
            status = "known"
        elif split in {prefix + s for prefix in ("val_", "test_") for s in base.STATUSES}:
            status = split.split("_", 1)[1]
        else:
            raise ValueError("Unknown cache split: " + str(split))
        hashes, rows = group["image_sha256"], group["records"]
        if (not isinstance(hashes, list) or not hashes or len(hashes) != len(set(hashes))
                or seen.intersection(hashes)):
            raise ValueError("D05 cached unique images overlap or are empty")
        unique = base.unique_records(rows)
        if set(hashes) != {base._digest(r) for r in unique}:
            raise ValueError("D05 image hashes and record aliases disagree")
        seen.update(hashes)
        if any(r.get("split") != split or r.get("status") != status for r in rows):
            raise ValueError("D05 cached rows were moved between splits")
        if not isinstance(group.get("features"), dict) or set(group["features"]) != FEATURE_KEYS:
            raise ValueError("D05 cache requires the original three feature variants")
        for key, value in group["features"].items():
            if (not torch.is_tensor(value) or not value.is_floating_point() or value.ndim != 2
                    or len(value) != len(hashes) or value.shape[1] < 1
                    or not bool(torch.isfinite(value).all())):
                raise ValueError("Invalid D05 cached feature: " + key)
        _validate_text(cache["text"], meta, group["features"]["clip"].shape[1])
        indices = group.get("record_feature_indices")
        if not isinstance(indices, list) or len(indices) != len(rows):
            raise ValueError("D05 alias-to-feature mapping length differs")
        for row, index in zip(rows, indices):
            if type(index) is not int or not 0 <= index < len(hashes) or base._digest(row) != hashes[index]:
                raise ValueError("D05 alias-to-feature index mismatch")
        # Candidate identities remain controlled by the original reference heads.
        membership.candidate_scores(rows, meta)
    if not isinstance(cache.get("provenance"), dict):
        raise ValueError("D05 cache provenance is missing")
    _text_contract(cache)
    return cache


def fit_payload(train_cache, *, seed=1, folds=3, epochs=100, batch_size=1024,
                lr=.001, shrinkage=.1, hidden=32):
    """Fit original D05 evidence from known TRAIN only, with the fixed BCE path.

    The default 100 epochs and batch size 1024 yield 12,800 Adam updates for the
    archived 100,459 leaf and 28,910 parent examples. Actual steps are derived
    from the supplied episode counts; no dataset-dependent step count is forged.
    """
    validate_cache(train_cache, stage="train")
    meta = train_cache["meta"]
    group = train_cache["groups"]["train"]
    rows = _aligned_rows(group)
    labels = torch.tensor([r["true_leaf"] for r in rows], dtype=torch.long)
    if any(r["true_parent"] != meta["leaf_to_parent"][int(r["true_leaf"])] for r in rows):
        raise ValueError("Known TRAIN labels disagree with the locked taxonomy")
    fine = parent = group["features"]["clip"]
    # The historical D05 backend limited CPU math threads before both fits.
    torch.set_num_threads(min(4, torch.get_num_threads()))
    geometry = GeometryBank.fit(fine, parent, labels, group["image_sha256"],
                                meta, shrinkage=shrinkage)
    episodes = build_episodes(fine, parent, labels, group["image_sha256"], meta,
        template_scores=_templates(fine, train_cache["text"]),
        folds=folds, seed=seed, shrinkage=shrinkage)
    verifier = SharedVerifier.fit(episodes, seed=seed, epochs=epochs,
        batch_size=batch_size, lr=lr, hidden=hidden)
    if verifier.dimension != 8:
        raise ValueError("The original D05 verifier requires all eight evidence features")
    report = dict(training_execution="completed",
                  optimizer_steps=verifier.fit_report["optimizer_steps"],
                  geometry=copy.deepcopy(geometry.fit_report),
                  verifier=copy.deepcopy(verifier.fit_report))
    payload = dict(meta=copy.deepcopy(meta), model_spec=copy.deepcopy(MODEL_SPEC),
        text={k: v.detach().cpu().clone() for k, v in train_cache["text"].items()},
        provenance=copy.deepcopy(train_cache["provenance"]), projection=None,
        inference_spec_sha256=_text_contract(train_cache),
        geometry=geometry.state_dict(), verifier=verifier.state_dict(),
        fit_report=copy.deepcopy(report))
    validate_payload(payload, meta, train_cache=train_cache)
    return payload, report


def validate_payload(payload, meta, train_cache=None):
    """Restore and validate original BCE tensors without fitting any statistics.

    Accepts both a native internal payload and the original complete D05 payload.
    Returns ``(geometry, verifier)``. Legacy file signatures remain the caller's
    responsibility and are never rewritten here.
    """
    meta = _meta(meta)
    if (not isinstance(payload, dict) or not {"geometry", "verifier", "text", "fit_report"} <= set(payload)
            or payload.get("projection") is not None or "prompt" in payload
            or ("meta" in payload and payload["meta"] != meta)):
        raise ValueError("Expected the original D05 geometry/BCE payload")
    if "model_spec" in payload and payload["model_spec"] != MODEL_SPEC:
        raise ValueError("D05 payload names a different experiment or candidate policy")
    geometry = GeometryBank.from_state_dict(payload["geometry"])
    verifier = SharedVerifier.from_state_dict(payload["verifier"])
    _validate_text(payload["text"], meta, geometry.fine.shape[1])
    report = payload["fit_report"]
    if (not isinstance(report, dict) or geometry.meta != meta
            or report.get("geometry") != geometry.fit_report
            or report.get("verifier") != verifier.fit_report
            or verifier.fit_report.get("loss") != "bce" or verifier.dimension != 8
            or verifier.fit_report.get("normalization_fit") != "TRAIN_episode_weighted_mean_std"
            or report.get("optimizer_steps") != verifier.fit_report["optimizer_steps"]):
        raise ValueError("D05 tensor and training-report provenance differ")
    if train_cache is not None:
        validate_cache(train_cache, meta, stage="train")
        group = train_cache["groups"]["train"]
        rows = _aligned_rows(group)
        labels = torch.tensor([r["true_leaf"] for r in rows], dtype=torch.long)
        expected = _features(group["features"]["clip"], "cached TRAIN CLIP")
        if (list(geometry.image_hashes) != group["image_sha256"]
                or not torch.equal(geometry.labels, labels)
                or not torch.equal(geometry.fine, expected)
                or not torch.equal(geometry.parent, expected)):
            raise ValueError("D05 geometry support differs from the exact TRAIN cache")
        if (any(not torch.equal(payload["text"][key].detach().cpu(), train_cache["text"][key].detach().cpu())
                for key in TEXT_KEYS)
                or payload.get("inference_spec_sha256") != _text_contract(train_cache)
                or payload.get("provenance") != train_cache["provenance"]):
            raise ValueError("D05 text/core preprocessing differs from the frozen TRAIN cache")
        episode = verifier.fit_report["episode_report"]
        digest = hashlib.sha256("\n".join(sorted(group["image_sha256"])).encode("utf-8")).hexdigest()
        if episode.get("image_hash_digest") != digest or episode.get("train_count") != len(labels):
            raise ValueError("D05 episode normalization is bound to another TRAIN cache")
    return geometry, verifier


def score_groups(cache, payload, meta):
    """Pure D05 inference on original reference candidates, without refitting."""
    validate_cache(cache, meta)
    if _text_contract(cache) != payload.get("inference_spec_sha256"):
        raise ValueError("D05 inference cache changed its frozen core or templates")
    geometry, verifier = validate_payload(payload, meta)
    result = {}
    for split, group in cache["groups"].items():
        rows = copy.deepcopy(group["records"])
        fine = parent = group["features"]["clip"]
        evidence = geometry.score(fine, parent, group["image_sha256"])
        scored = verifier.score(evidence, _templates(fine, payload["text"]))
        leaf, parents = (scored[key].detach().cpu().double()
                         for key in ("leaf_scores", "parent_scores"))
        if not bool(torch.isfinite(leaf).all() and torch.isfinite(parents).all()):
            raise ValueError("Nonfinite D05 inference scores")
        identities = membership.candidate_scores(rows, meta)
        for i, row in enumerate(rows):
            index = group["record_feature_indices"][i]
            p, c = int(identities["parent"][i]), int(identities["leaf"][i])
            row["discovery"] = dict(leaf_scores=leaf[index].tolist(),
                parent_scores=parents[index].tolist(), candidate_leaf=c, candidate_parent=p)
            row["discovery_method"] = "bce"
            row["support_evidence_origin"] = "immutable_reference_diagnostic"
            row["log_probs_origin"] = "immutable_reference_diagnostic"
        result[split] = rows
    return result
