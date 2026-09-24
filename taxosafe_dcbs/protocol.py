"""Explicit stage allowlists, byte-identity audits and immutable run receipts."""
import copy
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("main", "lite", "heads_only", "near_only", "no_ha", "no_pooling")
STAGE_SPLITS = {
    "train": ("train", "val_known"),
    "calibrate": ("val_known", "val_intra", "val_extra"),
    "test": ("test_known", "test_intra", "test_extra"),
}


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_records(path, records):
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def effective_config(path, variant="main", seed=1):
    cfg = copy.deepcopy(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
    if not isinstance(cfg, dict) or "dcbs" not in cfg or variant not in VARIANTS:
        raise ValueError("A v11 DCBS config and supported variant are required")
    cfg["seed"], cfg["variant"] = int(seed), variant
    cfg["data"]["seed"] = int(seed)
    cfg["data"].setdefault("sampler", {})["seed"] = int(seed)
    cfg["data"]["sampler"]["holdout_seed"] = int(seed)
    settings = cfg["dcbs"]
    weights = settings.setdefault("loss", {})
    if variant == "lite":
        settings["shared_projection"] = True
        weights.update(ha=0., margin=0.)
        cfg["calibration"]["use_margin"] = False
    if variant == "no_ha":
        weights["ha"] = 0.
    if variant == "heads_only":
        settings["synthesis"].update(near_enabled=False, extra_enabled=False)
    if variant == "near_only":
        settings["synthesis"]["extra_enabled"] = False
    if variant == "no_pooling":
        cfg["calibration"]["partial_pooling"] = False
    if cfg.get("checkpoint") or cfg.get("init_checkpoint") or cfg["model"].get("pretrained"):
        raise ValueError("v11 requires fresh prompts from the original CLIP initialization, not a v10/OE checkpoint")
    if any(float(cfg.get("loss", {}).get(k, 0)) for k in ("lambda_oe", "lambda_real_intra")):
        raise ValueError("Real unknown/OE gradient losses are not permitted in v11")
    if settings.get("shared_projection", False) and float(weights.get("ha", 0)):
        raise ValueError("HA requires separate projections")
    if cfg["model"].get("arch") != "maple":
        raise ValueError("This implementation expects the repository's MaPLe encoder")
    if int(cfg["training"]["epochs"]) <= int(cfg["training"]["warmup_epochs"]):
        raise ValueError("epochs must exceed warmup_epochs")
    if any(float(v) < 0 for v in weights.values()):
        raise ValueError("Loss weights must be non-negative")
    return cfg


def signature(cfg):
    dependencies = ["models/maple.py", "models/maple_model.py", "models/maple_clip.py", "models/__init__.py",
                    "loader/hierdata.py", "loader/transforms.py", "loader/hierarchical_episode_sampler.py",
                    "loader/treelibs.py", "loader/utils.py", "taxosafe_episode.py", "metrics_open.py"]
    files = sorted(PROJECT_ROOT.joinpath("taxosafe_dcbs").glob("*.py"))
    files += [PROJECT_ROOT / p for p in dependencies]
    result = {"config": object_hash(cfg), "hierarchy": file_hash(resolve(cfg["data"]["hierarchy"])),
              "code": object_hash({str(p.relative_to(PROJECT_ROOT)): file_hash(p) for p in files})}
    if cfg["data"].get("known_preparation_audit"):
        result["known_preparation_audit"] = file_hash(resolve(cfg["data"]["known_preparation_audit"]))
    # Test never needs to reopen fitting manifests; their hashes live in receipts.
    return result


def require_signature(expected, actual):
    if expected != actual:
        raise ValueError("Configuration, hierarchy, code or preparation audit changed since training; use a new run")


def normalized_name(value):
    return "".join(c for c in str(value).casefold() if c.isalnum())


def split_root(cfg, split):
    key = "near_dev_root" if split == "val_intra" else "near_test_root" if split == "test_intra" else (
        "ood_dev_root" if split == "val_extra" else "ood_test_root" if split == "test_extra" else "data_root")
    return resolve(cfg["data"][key])


def read_split(cfg, split, meta):
    if split not in {s for splits in STAGE_SPLITS.values() for s in splits}:
        raise ValueError("Unsupported split: " + split)
    status = "intra" if split.endswith("intra") else "extra" if split.endswith("extra") else "known"
    root, rows = split_root(cfg, split), []
    known_names = {normalized_name(v) for v in meta["leaf_names"]}
    for line_no, line in enumerate(resolve(cfg["data"][split]).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            relative, label, manifest_index = line.rsplit(",", 2)
            label, manifest_index = int(label), int(manifest_index)
        except ValueError as exc:
            raise ValueError("Malformed manifest {} line {}".format(split, line_no)) from exc
        parts = Path(relative).parts
        if Path(relative).is_absolute() or ".." in parts or len(parts) < 2:
            raise ValueError("Unsafe or unlabelled image path: " + relative)
        source = parts[-2]
        path = (root / relative).resolve()
        path.relative_to(root.resolve())
        if status == "known":
            if not 0 <= label < len(meta["leaf_names"]) or normalized_name(source) != normalized_name(meta["leaf_names"][label]):
                raise ValueError("Known label/name mismatch: " + relative)
            parent = meta["leaf_to_parent"][label]
            if len(parts) < 3 or normalized_name(parts[-3]) != normalized_name(meta["parent_names"][parent]):
                raise ValueError("Known parent/name mismatch: " + relative)
            leaf = label
        elif status == "intra":
            if not 0 <= label < len(meta["parent_names"]) or len(parts) < 3 or normalized_name(parts[-3]) != normalized_name(meta["parent_names"][label]):
                raise ValueError("Near parent/name mismatch: " + relative)
            if normalized_name(source) in known_names:
                raise ValueError("Known species appears as unknown: " + relative)
            parent, leaf = label, None
        else:
            if label != -1 or normalized_name(source) in known_names:
                raise ValueError("Invalid extra label/source: " + relative)
            parent, leaf = None, None
        rows.append({"path": relative, "resolved_path": str(path), "source": source, "status": status,
                     "split": split, "dataset_index": len(rows), "manifest_index": manifest_index,
                     "true_leaf": leaf, "true_parent": parent, "image_sha256": file_hash(path)})
    if not rows:
        raise ValueError("Empty split: " + split)
    return rows


def audit_rows(groups, forbidden_hashes=(), forbidden_sources=(), allow_within_split=False):
    forbidden = set(forbidden_hashes)
    blocked_sources = {normalized_name(s) for s in forbidden_sources}
    hashes, paths, audit = {}, {}, {}
    for split, rows in groups.items():
        local = {}
        for row in rows:
            digest, path = row["image_sha256"], row["resolved_path"]
            identity = (row["status"], row.get("true_leaf"), row.get("true_parent"), normalized_name(row["source"]))
            if digest in forbidden or (row["status"] != "known" and normalized_name(row["source"]) in blocked_sources):
                raise ValueError("Fitting/evaluation identity or unknown-source overlap: " + path)
            if digest in local:
                if not allow_within_split or local[digest] != identity:
                    raise ValueError("Duplicate image or conflicting labels within " + split + ": " + path)
            elif digest in hashes or path in paths:
                raise ValueError("Duplicate image across splits: " + path)
            local[digest], hashes[digest], paths[path] = identity, split, split
        audit[split] = {"count": len(rows), "unique_image_count": len(local),
                        "image_hashes": sorted(local), "sources": sorted({r["source"] for r in rows})}
    return audit


def load_stage_rows(cfg, stage, meta, forbidden_hashes=(), forbidden_sources=()):
    if stage not in STAGE_SPLITS:
        raise ValueError("Unknown stage: " + stage)
    groups = {split: read_split(cfg, split, meta) for split in STAGE_SPLITS[stage]}
    audit = audit_rows(groups, forbidden_hashes, forbidden_sources, allow_within_split=stage == "test")
    for split in groups:
        audit[split]["manifest_sha256"] = file_hash(resolve(cfg["data"][split]))
    return groups, audit


@contextmanager
def run_lock(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / ".dcbs.lock", "a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another DCBS process owns this run directory") from exc
        try:
            handle.seek(0)
            handle.truncate()
            handle.write(str(os.getpid()))
            handle.flush()
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
    # Keep the lock inode: unlinking it introduces a race for waiting processes.


def claim_stage(directory, stage):
    path = Path(directory) / stage
    try:
        path.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise ValueError("Stage already exists; choose a fresh --run-dir: " + str(path)) from exc
    return path


def verify_artifact(directory, receipt_name, artifact_key):
    receipt = read_json(Path(directory) / receipt_name)
    descriptor = receipt[artifact_key]
    path = Path(directory) / descriptor["path"]
    if file_hash(path) != descriptor["sha256"]:
        raise ValueError("Artifact hash mismatch: " + str(path))
    return receipt, path
