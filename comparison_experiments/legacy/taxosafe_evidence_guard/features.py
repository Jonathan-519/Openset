"""Audited real-unknown TRAIN and complete, immutable C00 feature caches.

The added unknown TRAIN images are image-disjoint from DEV/TEST and source-
disjoint from TEST. They deliberately share sources with DEV; this expanded
training regime must not be described as the original known-only C00 regime.
No support class is removed and no TEST feature is opened during fitting.
"""
import copy
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

from taxosafe_dcbs.protocol import normalized_name
from taxosafe_support import pipeline as support
from taxosafe_support import protocol as base_protocol
from taxosafe_support import calibration as base
from taxosafe_support.membership_calibration import RAW_FIELDS
from taxosafe_refine.importer import load_reference
from taxosafe_refine.pipeline import load_training_rows, _check_baseline_development
from taxosafe_discovery.features import tensor_hash


DEFAULT_DATA = {
    "train_intra": "prepro/data/Zooplankton_TT_v9_rebuild/gt_train_intra_v10.txt",
    "oe_train": "prepro/data/Zooplankton_TT_v9_rebuild/gt_oe_train_v10.txt",
}
ENCODED_FIELDS = ("parent", "fine", "parent_local", "fine_local", "leaf_logits", "parent_logits")


def _unknown_rows(info, split, manifest):
    """Parse only explicit added TRAIN manifests, retaining global taxonomy ids."""
    if split not in DEFAULT_DATA:
        raise ValueError("Unsupported added TRAIN split: " + str(split))
    near = split == "train_intra"
    root_key = "near_dev_root" if near else "ood_dev_root"
    root = base_protocol.resolve(info["config"]["data"][root_key]).resolve()
    path = base_protocol.resolve(manifest)
    meta = info["meta"]
    known = {normalized_name(name) for name in meta["leaf_names"]}
    rows, source_labels = [], {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            relative, label, manifest_index = line.rsplit(",", 2)
            label, manifest_index = int(label), int(manifest_index)
        except ValueError as error:
            raise ValueError("Malformed added TRAIN manifest {} line {}".format(path, line_number)) from error
        parts = Path(relative).parts
        if (not relative or "\\" in relative or Path(relative).is_absolute() or ".." in parts
                or len(parts) != (3 if near else 2) or manifest_index < 0):
            raise ValueError("Unsafe or unlabelled added TRAIN image path: " + relative)
        source = parts[-2]
        normalized = normalized_name(source)
        if not normalized or normalized in known:
            raise ValueError("Known/invalid source appears in unknown TRAIN: " + relative)
        resolved = (root / relative).resolve()
        try:
            resolved.relative_to(root)
        except ValueError as error:
            raise ValueError("Added TRAIN image escapes its declared root: " + relative) from error
        if near:
            if (not 0 <= label < len(meta["parent_names"])
                    or normalized_name(parts[-3]) != normalized_name(meta["parent_names"][label])):
                raise ValueError("Near TRAIN parent label/name mismatch: " + relative)
            parent = label
        else:
            if label != -1:
                raise ValueError("Extra TRAIN label must be -1: " + relative)
            parent = None
        if normalized in source_labels and source_labels[normalized] != parent:
            raise ValueError("Unknown TRAIN source has conflicting parent labels: " + source)
        source_labels[normalized] = parent
        rows.append({"path": relative, "resolved_path": str(resolved), "source": source,
                     "status": "intra" if near else "extra", "split": split,
                     "dataset_index": len(rows), "manifest_index": manifest_index,
                     "true_leaf": None, "true_parent": parent,
                     "image_sha256": base_protocol.file_hash(resolved)})
    if not rows:
        raise ValueError("Empty added TRAIN split: " + split)
    return rows


def _test_isolation_audit(info):
    """Require all TEST identities, reading image bytes only when no receipt exists."""
    names = ("test_known", "test_intra", "test_extra")
    available = {name: info["audit"][name] for name in names if name in info["audit"]}
    if len(available) == len(names):
        return available, {"source": "validated_C00_TEST_receipt", "test_image_hashes_read": False,
                           "test_features_extracted": False}
    # No source TEST receipt does not authorize skipping TEST exclusion. Read
    # manifests and SHA256 only; do not construct a loader or an encoder here.
    forbidden = {h for name, item in info["audit"].items() if not name.startswith("test_")
                 for h in item["image_hashes"]}
    forbidden_sources = {source for name in ("val_intra", "val_extra")
                         for source in info["audit"][name]["sources"]}
    _, audited = support.load_stage_rows(info["config"], "test", info["meta"],
                                        forbidden_hashes=forbidden, forbidden_sources=forbidden_sources)
    for name, previous in available.items():
        if audited[name] != previous:
            raise ValueError("TEST identities differ from the available C00 audit: " + name)
    return audited, {"source": "TEST_manifests_and_image_SHA256_only", "test_image_hashes_read": True,
                     "test_features_extracted": False,
                     "manifest_sha256": {name: audited[name]["manifest_sha256"] for name in names}}


def _development_source_roles(info):
    """Read locked manifest taxonomy only, not DEV features or model responses."""
    roles = {}
    for split in ("val_intra", "val_extra"):
        path = base_protocol.resolve(info["config"]["data"][split])
        if base_protocol.file_hash(path) != info["audit"][split]["manifest_sha256"]:
            raise ValueError("DEV manifest changed before TRAIN source-role validation: " + split)
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            relative, label, _ = line.rsplit(",", 2)
            source = normalized_name(Path(relative).parts[-2])
            role = ("intra", int(label)) if split == "val_intra" else ("extra", None)
            if source in roles and roles[source] != role:
                raise ValueError("DEV unknown source has conflicting taxonomy roles: " + source)
            roles[source] = role
    return roles


def source_rows(info, stage, cfg=None):
    if stage == "train":
        data = dict(DEFAULT_DATA)
        if cfg is not None:
            supplied = cfg.get("data", {})
            if not isinstance(supplied, dict) or set(supplied) - set(DEFAULT_DATA):
                raise ValueError("Added TRAIN data accepts only train_intra and oe_train manifests")
            data.update(supplied)
        rows, known_audit = load_training_rows(SimpleNamespace(**info))
        test_audit, test_read = _test_isolation_audit(info)
        all_audit = dict(info["audit"], **test_audit)
        forbidden = {h for item in all_audit.values() for h in item["image_hashes"]}
        test_sources = {source for split in ("test_intra", "test_extra")
                        for source in test_audit[split]["sources"]}
        unknown = {split: _unknown_rows(info, split, data[split]) for split in DEFAULT_DATA}
        # One audit rejects both within-split duplicates and cross-added-split
        # collisions before any image loader or gradient computation is opened.
        extra_audit = base_protocol.audit_rows(unknown, forbidden_hashes=forbidden,
                                               forbidden_sources=test_sources)
        near_names = {normalized_name(row["source"]) for row in unknown["train_intra"]}
        extra_names = {normalized_name(row["source"]) for row in unknown["oe_train"]}
        if near_names & extra_names:
            raise ValueError("One added TRAIN source cannot be both near and extra")
        roles = _development_source_roles(info)
        for split_rows in unknown.values():
            for row in split_rows:
                source = normalized_name(row["source"])
                if source in roles and roles[source] != (row["status"], row["true_parent"]):
                    raise ValueError("TRAIN/DEV unknown source has conflicting taxonomy roles: " + row["source"])
        dev_names = {normalized_name(source) for split in ("val_intra", "val_extra")
                     for source in info["audit"][split]["sources"]}
        for split, item in extra_audit.items():
            item.update(manifest_sha256=base_protocol.file_hash(base_protocol.resolve(data[split])),
                        expanded_training_data=True, real_unknown_images=True,
                        full_C00_support_retained=True,
                        development_source_overlap=sorted(source for source in item["sources"]
                                                          if normalized_name(source) in dev_names),
                        test_source_overlap=[], test_isolation=copy.deepcopy(test_read))
        return dict(train=rows, **unknown), dict(train=known_audit, **extra_audit)
    if stage not in ("development", "test"):
        raise ValueError("Unknown feature stage: " + str(stage))
    forbidden = set(info["training"]["audit"]["train"]["image_hashes"])
    names = set()
    if stage == "test":
        for split, audit in info["calibration"]["audit"].items():
            forbidden.update(audit["image_hashes"])
            if split in ("val_intra", "val_extra"):
                names.update(audit["sources"])
    groups, audit = support.load_stage_rows(info["config"], "calibrate" if stage == "development" else "test",
                                           info["meta"], forbidden_hashes=forbidden, forbidden_sources=names)
    if any(name in info["audit"] and item != info["audit"][name] for name, item in audit.items()):
        raise ValueError("Input images or manifests differ from immutable C00 audit")
    return groups, audit


def audit_images(groups):
    problems = []
    for split, rows in groups.items():
        for row in base.unique_records(rows):
            path = Path(row["resolved_path"])
            try:
                if not path.is_file() or base_protocol.file_hash(path) != row["image_sha256"]:
                    raise ValueError("Missing/changed image bytes")
                with Image.open(path) as image:
                    image.verify()
            except (OSError, ValueError) as error:
                problems.append(dict(split=split, path=str(path), error=str(error)))
    return dict(valid=not problems, problems=problems,
                image_count=sum(len(base.unique_records(rows)) for rows in groups.values()))


def _cpu_copy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    return copy.deepcopy(value)


@torch.no_grad()
def collect(info, groups, device):
    """Cache original four query streams and verifiers; preserve original DEV arithmetic."""
    device = torch.device(device)
    source = load_reference(info["directory"], device)
    if source.binding != info["binding"]:
        raise ValueError("C00 binding changed before extraction")
    text = source.encoder.text_features()
    output = {}
    for split, raw in groups.items():
        rows = base.unique_records(raw)
        values = {name: [] for name in ENCODED_FIELDS}
        scored, seen = [], []
        for images, _, indices in support.make_loader(rows, source.config, source.meta, training=False):
            batch_rows = [rows[index] for index in indices.tolist()]
            encoded = source.encoder.encode(images.to(device), text_features=text)
            hashes = [row["image_sha256"] for row in batch_rows]
            # Known TRAIN self-exclusion matches the original training rule.
            # Every unknown/DEV/TEST query keeps all 23 supported leaves.
            evidence = source.evidence(encoded, source.bank,
                                       query_hashes=hashes if split == "train" else None)
            batch = base.raw_records(batch_rows, {"log_probs": evidence["log_probs"].cpu().numpy()},
                                     encoded["leaf_logits"].cpu().numpy(), source.meta)
            diagnostic = {key: evidence[key].cpu().tolist() for key in RAW_FIELDS}
            for index, row in enumerate(batch):
                row["support_evidence"] = {key: value[index] for key, value in diagnostic.items()}
            scored.extend(batch)
            seen.extend(indices.tolist())
            for name in ENCODED_FIELDS:
                value = encoded.get(name)
                if value is not None:
                    if not bool(torch.isfinite(value).all()):
                        raise ValueError("Nonfinite C00 feature stream: " + name)
                    values[name].append(value.detach().float().cpu())
        if seen != list(range(len(rows))):
            raise ValueError("Reference image order changed")
        by_hash = {row["image_sha256"]: row for row in scored}
        output[split] = dict(records=[dict(by_hash[row["image_sha256"]], **row) for row in raw],
                             image_sha256=[row["image_sha256"] for row in rows],
                             encoded={name: torch.cat(items) if items else None for name, items in values.items()},
                             known_self_hash_exclusion=split == "train", full_support_classes_retained=True)
        print("C00 complete cache {}: {} images; original eval batch {}".format(
            split, len(rows), source.config["data"].get("eval_batch_size", 16)), flush=True)
    settings = source.config["support"]
    kwargs = dict(hidden_dim=int(settings.get("hidden_dim", 32)),
                  temperature=float(settings.get("temperature", .1)),
                  local_enabled=bool(settings.get("local_enabled", True)),
                  decoupled=bool(settings.get("decoupled", False)),
                  membership_mode=settings.get("membership", "prototype"),
                  reference_topk=int(settings.get("reference_topk", 2)))
    context = dict(dimension=int(source.encoder.dimension), evidence_kwargs=kwargs,
                   evidence_state=_cpu_copy(source.evidence.state_dict()), bank_state=_cpu_copy(source.bank.state_dict()),
                   source_router=copy.deepcopy(source.router), meta=copy.deepcopy(source.meta))
    result = dict(groups=output, context=context, meta=copy.deepcopy(source.meta),
                  source_binding=copy.deepcopy(source.binding), text=_cpu_copy(text),
                  logit_scale=float(source.encoder.backbone.model.logit_scale.exp().float()),
                  reference_eval_batch_size=source.config["data"].get("eval_batch_size", 16),
                  frozen_encoder_updated=False, frozen_evidence_updated=False, frozen_support_updated=False,
                  feature_origin="C00 complete parent/fine global and local TokenBranch outputs; original verifiers/support",
                  support_intervention="none; only exact known TRAIN query hash excluded")
    if groups and all(name.startswith("val_") for name in groups):
        result["source_reproduction"] = _check_baseline_development(source,
            {name: [dict(row, reconstruction_score=0.) for row in group["records"]]
             for name, group in output.items()})
    source.encoder.cpu()
    source.evidence.cpu()
    source.bank.to("cpu")
    return result


def summary(cache):
    return dict(feature_origin=cache["feature_origin"],
                reference_eval_batch_size=cache["reference_eval_batch_size"], frozen_encoder_updated=False,
                frozen_evidence_updated=False, frozen_support_updated=False,
                support_intervention=cache["support_intervention"], source_reproduction=cache.get("source_reproduction"),
                dimension=cache["context"]["dimension"],
                groups={name: dict(count=len(group["image_sha256"]),
                    known_self_hash_exclusion=group["known_self_hash_exclusion"],
                    streams={key: None if value is None else dict(shape=list(value.shape), sha256=tensor_hash(value))
                             for key, value in group["encoded"].items()}) for name, group in cache["groups"].items()})
