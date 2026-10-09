"""Shared content-audited I/O, manifests and immutable stage ownership.

Hash serialization deliberately retains the original default JSON separators.
These are storage/data contracts; no historical training family is imported.
"""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
