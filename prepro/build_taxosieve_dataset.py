#!/usr/bin/env python3
"""Reconcile the current image inventory into an immutable TaxoSieve dataset.

This is dataset preparation, never model fitting: image decoding checks file
integrity only. Frozen manifests preserve existing image identities, labels and
row order, and reject missing or unapproved new identities. Audited, restored
historical TEST identities stay in separate inactive lists. Only seeds without
a frozen order use content-hash splits for new identities. Unknown development
reserves must never supply training gradients.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEED = "prepro/protocols/taxosieve_seed"
DEFAULT_IMAGES = "prepro/data/image"
DEFAULT_OUTPUT = "prepro/data"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "val_known", "test_known", "val_intra", "test_intra",
          "val_extra", "test_extra", "reserved_near", "reserved_extra")
FILENAMES = {s: "gt_" + s + ".txt" for s in SPLITS}
ROOT_ROLES = {"train": "known", "val_known": "known", "test_known": "known",
              "val_intra": "near_dev", "reserved_near": "near_dev", "test_intra": "near_test",
              "val_extra": "extra_dev", "reserved_extra": "extra_dev", "test_extra": "extra_test"}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def normalized(value):
    return "".join(c for c in value.casefold() if c.isalnum())


def relative(value):
    if not isinstance(value, str) or not value or any(c in value for c in "\\\r\n\0,"):
        raise ValueError("Unsafe manifest path: " + repr(value))
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ValueError("Expected a canonical relative path: " + value)
    return path


def locate(root, value):
    path = Path(value)
    path = path if path.is_absolute() else root / path
    # Do not follow an image/output/seed symlink outside the audited project.
    parts = path.absolute().relative_to(root).parts
    cursor = root
    for part in parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError("Symlinks are not permitted: " + str(cursor))
    path = path.resolve()
    path.relative_to(root)
    return path


def priority(split):
    return 0 if split.startswith("test_") else 1 if split.startswith("val_") else 2 if split == "train" else 3


def load_seed(seed_dir):
    raw = (seed_dir / "seed.json").read_bytes()
    seed = json.loads(raw)
    if seed.get("schema_version") != "taxosieve_seed_v1":
        raise ValueError("Unsupported TaxoSieve seed schema")
    expected_roles = {"known", "near_dev", "near_test", "extra_dev", "extra_test"}
    if set(seed["roots"]) != expected_roles or len(set(seed["roots"].values())) != 5:
        raise ValueError("Seed must define five distinct image roots")
    for dirname in seed["roots"].values():
        if len(relative(dirname).parts) != 1 or dirname not in seed["classes"]:
            raise ValueError("Invalid seed image root: " + dirname)
    if set(seed["classes"]) != set(seed["roots"].values()):
        raise ValueError("Unexpected seed class root")
    source_roles = {}
    for role, dirname in seed["roots"].items():
        for class_path, label in seed["classes"][dirname].items():
            path = relative(class_path)
            if len(path.parts) != (1 if role.startswith("extra") else 2) or type(label) is not int:
                raise ValueError("Invalid class definition: " + class_path)
            source = normalized(path.name)
            if source in source_roles and source_roles[source] != role:
                raise ValueError("Known/unknown or TEST/DEV source overlap: " + class_path)
            source_roles[source] = role
            if role.startswith("extra") and label != -1:
                raise ValueError("OOD classes require label -1")
    taxonomy = {}
    if set(seed["taxonomy_sha256"]) != {"tree.npy", "leaf_nodes.npy", "known_leaf_order.txt"}:
        raise ValueError("Incomplete taxonomy seed")
    for name, expected in seed["taxonomy_sha256"].items():
        data = (seed_dir / name).read_bytes()
        if digest(data) != expected:
            raise ValueError("Taxonomy seed hash differs: " + name)
        taxonomy[name] = data
    order = {}
    for line in taxonomy["known_leaf_order.txt"].decode("utf-8").splitlines():
        label, name = line.split("\t")
        if int(label) in order:
            raise ValueError("Duplicate known leaf label")
        order[int(label)] = name
    known = seed["classes"][seed["roots"]["known"]]
    if {label: PurePosixPath(path).name for path, label in known.items()} != order:
        raise ValueError("Known class labels differ from frozen leaf order")
    seen = set()
    for row in seed["images"]:
        parts = relative(row["path"]).parts
        if row["path"] in seen or row["split"] not in SPLITS:
            raise ValueError("Duplicate seed path or unsupported split")
        seen.add(row["path"])
        role = ROOT_ROLES[row["split"]]
        if parts[0] != seed["roots"][role]:
            raise ValueError("Seed split/root mismatch: " + row["path"])
        if seed["classes"][parts[0]].get("/".join(parts[1:-1])) != row["label"]:
            raise ValueError("Seed label mismatch: " + row["path"])
        if len(row["blob_sha1"]) != 40 or any(c not in "0123456789abcdef" for c in row["blob_sha1"]):
            raise ValueError("Invalid seed Git blob identity")
        if type(row["size"]) is not int or row["size"] < 0:
            raise ValueError("Invalid seed image size")
    return seed, taxonomy, digest(raw)


def scan_images(image_root, seed):
    if not image_root.is_dir():
        raise ValueError("Missing image root: " + str(image_root))
    allowed_dirs = {""}
    for dirname, classes in seed["classes"].items():
        allowed_dirs.add(dirname)
        for class_path in classes:
            path = PurePosixPath(dirname) / class_path
            while path.as_posix() != ".":
                allowed_dirs.add(path.as_posix())
                path = path.parent
    current = []
    root_to_role = {v: k for k, v in seed["roots"].items()}
    for directory, dirs, files in os.walk(image_root, followlinks=False):
        dirs.sort()
        for name in dirs:
            child = Path(directory) / name
            rel = child.relative_to(image_root).as_posix()
            relative(rel)
            if child.is_symlink() or rel not in allowed_dirs:
                raise ValueError("Unsupported image directory/class or symlink: " + rel)
        for name in sorted(files):
            path = Path(directory) / name
            rel = path.relative_to(image_root).as_posix()
            parts = relative(rel).parts
            if path.is_symlink() or not path.is_file() or path.suffix.lower() not in EXTENSIONS:
                raise ValueError("Unsupported file in image inventory: " + rel)
            dirname, class_path = parts[0], "/".join(parts[1:-1])
            if dirname not in seed["classes"] or class_path not in seed["classes"][dirname]:
                raise ValueError("Unsupported image class/path: " + rel)
            before = path.stat()
            data = path.read_bytes()
            after = path.stat()
            if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                raise ValueError("Image changed during preparation: " + rel)
            try:
                with Image.open(io.BytesIO(data)) as image:
                    image.verify()
                with Image.open(io.BytesIO(data)) as image:
                    image.load()
                    width, height = image.size
            except Exception as exc:
                raise ValueError("Invalid or undecodable image: " + rel) from exc
            current.append({"path": rel, "root": dirname, "class": class_path, "source": parts[-2],
                            "role": root_to_role[dirname], "label": seed["classes"][dirname][class_path],
                            "sha256": digest(data), "blob_sha1": hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest(),
                            "size": len(data), "width": width, "height": height})
    missing_roots = [name for name in seed["roots"].values() if not (image_root / name).is_dir()]
    if missing_roots:
        raise ValueError("Missing role directories: " + ", ".join(missing_roots))
    if not current:
        raise ValueError("Image inventory is empty")
    return sorted(current, key=lambda row: row["path"])


def new_split(row):
    # The literal seed is part of the protocol; neither filenames nor ordering
    # can affect a newly observed image's split.
    fraction = int(hashlib.sha256(("taxosieve-v1:2026:" + row["sha256"]).encode("ascii")).hexdigest(), 16) / 2**256
    role = row["role"]
    if role == "known":
        return "train" if fraction < .7 else "val_known" if fraction < .8 else "test_known"
    if role == "near_test":
        return "test_intra"
    if role == "extra_test":
        return "test_extra"
    return ("val_intra" if fraction < .4 else "reserved_near") if role == "near_dev" else ("val_extra" if fraction < .4 else "reserved_extra")


def reconcile(seed, current):
    old_groups, new_groups = defaultdict(list), defaultdict(list)
    historical_labels = defaultdict(set)
    for row in seed["images"]:
        path = PurePosixPath(row["path"])
        key = (path.parent.as_posix(), row["blob_sha1"], row["size"])
        old_groups[key].append(row)
        historical_labels[(row["blob_sha1"], row["size"])].add(("/".join(path.parts[1:-1]), row["label"]))
    for row in current:
        prior_labels = historical_labels.get((row["blob_sha1"], row["size"]), set())
        if prior_labels and prior_labels != {(row["class"], row["label"])}:
            raise ValueError("Historical image content carries conflicting labels/sources: " + row["path"])
        key = (PurePosixPath(row["path"]).parent.as_posix(), row["blob_sha1"], row["size"])
        new_groups[key].append(row)
    matched, removed, added, renamed = {}, [], [], []
    for key in sorted(set(old_groups) | set(new_groups)):
        old = {row["path"]: row for row in old_groups[key]}
        new = {row["path"]: row for row in new_groups[key]}
        for path in sorted(set(old) & set(new)):
            matched[path] = old.pop(path)
            new.pop(path)
        old_left = sorted(old.values(), key=lambda row: (priority(row["split"]), row["path"]))
        new_left = sorted(new.values(), key=lambda row: row["path"])
        common = min(len(old_left), len(new_left))
        for prior, now in zip(old_left[:common], new_left[:common]):
            matched[now["path"]] = prior
            renamed.append({"old_path": prior["path"], "new_path": now["path"], "split": prior["split"], "sha256": now["sha256"]})
        removed.extend(old_left[common:])
        added.extend({"path": row["path"], "sha256": row["sha256"]} for row in new_left[common:])
    for row in current:
        prior = matched.get(row["path"])
        historical_alias = None
        if prior is None:
            key = (PurePosixPath(row["path"]).parent.as_posix(), row["blob_sha1"], row["size"])
            candidates = old_groups.get(key, [])
            if candidates:
                # A newly added filename is not necessarily new content. Do
                # not let hashing an alias promote TRAIN/VAL bytes into TEST.
                historical_alias = min(candidates, key=lambda item: (priority(item["split"]), item["path"]))
                prior = historical_alias
        row["assigned_split"] = prior["split"] if prior else new_split(row)
        row["assignment"] = "historical_content_alias" if historical_alias else "historical_identity" if prior else "new_content_hash"
        row["baseline_path"] = prior["path"] if prior else None
    # Exact bytes must never cross label/source identities, including unknowns.
    identities, by_hash = {}, defaultdict(list)
    for row in current:
        identity = ("known" if row["role"] == "known" else "near" if row["role"].startswith("near") else "extra", row["class"], row["label"])
        if row["sha256"] in identities and identities[row["sha256"]] != identity:
            raise ValueError("Exact image duplicate carries conflicting labels/sources: " + row["path"])
        identities[row["sha256"]] = identity
        by_hash[row["sha256"]].append(row)
    excluded = []
    for sha in sorted(by_hash):
        group = sorted(by_hash[sha], key=lambda row: (priority(row["assigned_split"]), row["assigned_split"], row["path"]))
        owner = group[0]
        for i, row in enumerate(group):
            # Preserve all TEST aliases (historical evaluation weighting),
            # while fitting/calibration/reserve lists use one identity each.
            keep = i == 0 or (row["assigned_split"].startswith("test_") and row["assigned_split"] == owner["assigned_split"])
            row["disposition"] = "selected" if keep else "excluded_exact_duplicate"
            row["owner_path"] = owner["path"]
            row["owner_split"] = owner["assigned_split"]
            if not keep:
                excluded.append({"path": row["path"], "split": row["assigned_split"], "label": row["label"],
                                 "sha256": sha, "same_as_path": owner["path"], "same_as_split": owner["assigned_split"]})
    selected = {split: [row for row in current if row["assigned_split"] == split and row["disposition"] == "selected"] for split in SPLITS}
    train_classes = {row["class"] for row in selected["train"]}
    missing_train = set(seed["classes"][seed["roots"]["known"]]) - train_classes
    if missing_train:
        raise ValueError("Known species have no independent training images: " + ", ".join(sorted(missing_train)))
    for split in SPLITS[:7]:
        if not selected[split]:
            raise ValueError("Required model split is empty: " + split)
    report = {"schema_version": "taxosieve_reconciliation_v1", "is_new_dataset_version": True,
              "old_test_results_comparable_without_re_evaluation": False,
              "baseline_count": len(seed["images"]), "current_count": len(current),
              "matched_count": len(matched), "added_count": len(added), "removed_count": len(removed),
              "renamed_count": len(renamed), "added": sorted(added, key=lambda row: row["path"]),
              "removed": sorted(removed, key=lambda row: row["path"]), "renamed": sorted(renamed, key=lambda row: row["new_path"]),
              "removed_by_split": dict(sorted(Counter(row["split"] for row in removed).items())),
              "excluded_exact_duplicates": sorted(excluded, key=lambda row: row["path"])}
    return selected, report


def preserve_manifest_order(seed_dir, seed, selected, report):
    """Keep the established main experiment's identities AND row order.

    Renaming files must not change a seeded sampler's index-to-image mapping.
    Historical TEST files restored after the previous dataset was frozen stay
    in separately documented, inactive manifests; they cannot enlarge TEST.
    """
    expected = seed.get("manifest_order_sha256")
    if expected is None:
        return {}, None
    raw = (seed_dir / "manifest_order.json").read_bytes()
    if digest(raw) != expected:
        raise ValueError("Frozen manifest order hash differs")
    order = json.loads(raw)
    if order.get("schema_version") != "taxosieve_manifest_order_v1" or set(order["splits"]) != set(SPLITS):
        raise ValueError("Unsupported or incomplete frozen manifest order")
    if set(order["restored_test"]) != {"test_intra", "test_extra"}:
        raise ValueError("Restored identities must be inactive historical TEST")
    restored = {}
    content_hashes = {}
    for split in SPLITS:
        candidates = defaultdict(list)
        for row in selected[split]:
            candidates[(row["sha256"], row["label"])].append(row)
        retained = []
        for identity in order["splits"][split]:
            key = (identity["sha256"], identity["label"])
            if not candidates[key]:
                raise ValueError("Missing frozen main-experiment identity in " + split + ": " + key[0])
            row = candidates[key].pop(0)
            row["manifest_index"] = len(retained)
            retained.append(row)
        extra = sorted((row for rows in candidates.values() for row in rows), key=lambda row: (row["sha256"], row["path"]))
        permitted = Counter(order["restored_test"].get(split, []))
        actual = Counter(row["sha256"] for row in extra)
        if actual != permitted or any(row["assignment"] != "historical_identity" for row in extra):
            raise ValueError("Inventory differs from frozen experiment/restored TEST; create an explicit new protocol: " + split)
        for row in extra:
            row["disposition"] = "inactive_restored_test"
        if split in order["restored_test"]:
            restored[split] = extra
        selected[split] = retained
        content_hashes[split] = digest(json_bytes(order["splits"][split]))
    report["main_experiment_preserved"] = {
        "baseline_repository_commit": order["source_repository_commit"],
        "baseline_inventory_sha256": order["baseline_inventory_sha256"],
        "baseline_manifest_sha256": order["baseline_manifest_sha256"],
        "content_labels_and_row_order_unchanged": True,
        "ordered_content_sha256": content_hashes,
        "restored_test_active": False,
        "restored_test_counts": {split: len(rows) for split, rows in restored.items()},
        "relocated_source_roots": order["relocated_source_roots"],
    }
    return restored, expected


def build(project_root=ROOT, image_root=DEFAULT_IMAGES, output=DEFAULT_OUTPUT,
          seed_dir=DEFAULT_SEED, dry_run=False, dataset_version=None):
    root = Path(project_root).resolve()
    image_root, output, seed_dir = (locate(root, value) for value in (image_root, output, seed_dir))
    if output == image_root or image_root in output.parents:
        raise ValueError("Output must not be the image root or a directory inside it")
    if output in image_root.parents and image_root.parent != output:
        raise ValueError("Images nested in the dataset must be its direct child directory")
    seed, taxonomy, seed_hash = load_seed(seed_dir)
    current = scan_images(image_root, seed)
    selected, reconciliation = reconcile(seed, current)
    restored, order_hash = preserve_manifest_order(seed_dir, seed, selected, reconciliation)
    image_path, output_path = image_root.relative_to(root).as_posix(), output.relative_to(root).as_posix()
    roots = {role: image_path + "/" + name for role, name in seed["roots"].items()}
    files = dict(taxonomy)
    def manifest(rows):
        lines = ["{},{},{}".format(row["path"].split("/", 1)[1], row["label"], index) for index, row in enumerate(rows)]
        return (("\n".join(lines) + "\n") if lines else "").encode("utf-8")
    for split in SPLITS:
        files[FILENAMES[split]] = manifest(selected[split])
    for split, rows in restored.items():
        files["gt_restored_" + split + ".txt"] = manifest(rows)
    files["gt_train_reference.txt"] = files[FILENAMES["train"]]
    count = {split: len(rows) for split, rows in selected.items()}
    unique = {split: len({row["sha256"] for row in rows}) for split, rows in selected.items()}
    per_source = {}
    for row in current:
        key = row["root"] + "/" + row["class"]
        entry = per_source.setdefault(key, {"label": row["label"], "role": row["role"], "inventory_count": 0,
                                             "selected": {s: 0 for s in SPLITS}, "excluded_exact_duplicates": 0,
                                             "inactive_restored_test": 0})
        entry["inventory_count"] += 1
        if row["disposition"] == "selected":
            entry["selected"][row["assigned_split"]] += 1
        elif row["disposition"] == "inactive_restored_test":
            entry["inactive_restored_test"] += 1
        else:
            entry["excluded_exact_duplicates"] += 1
    inventory = b"".join((json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8") for row in current)
    inventory_hash = digest(inventory)
    protocol = {"protocol_version": dataset_version or "taxosieve_v1_renamed", "experiment": "TaxoSieve",
                "schema_version": "taxosieve_dataset_v1", "seed_sha256": seed_hash,
                "historical_source_commit": seed["baseline_source_commit"], "inventory_sha256": inventory_hash,
                "new_dataset_after_inventory_change": True, "roots": roots, "output": output_path,
                "known_identity_policy": "preserve historical assignments; SHA256 priority TEST > VAL > TRAIN; keep TEST aliases",
                "new_known_split": None if order_hash else {"train": .7, "val_known": .1, "test_known": .2},
                "new_dev_unknown_split": None if order_hash else {"calibration": .4, "inactive_reserve": .6},
                "content_hash_seed": None if order_hash else "taxosieve-v1:2026",
                "split_policy": ("frozen main-experiment identities, labels and row order; reject missing/new identities; audited restored TEST inactive"
                                 if order_hash else "hash probabilities for new identities, no forced reshuffle"),
                "unknown_gradient_training": False, "unknown_reserves_active": False,
                "unknown_test_sources_disjoint_from_dev": True, "image_decode_scope": "integrity audit only; no model features or fitting",
                "test_alias_policy": "all assigned TEST rows retained, unique counts reported separately",
                "taxonomy_sha256": seed["taxonomy_sha256"],
                "manifest_order_sha256": order_hash,
                "restored_test_manifests": {split: {"path": output_path + "/gt_restored_" + split + ".txt",
                                                    "root": roots[ROOT_ROLES[split]], "count": len(rows),
                                                    "sha256": digest(files["gt_restored_" + split + ".txt"]),
                                                    "active": False, "gradient_training": False}
                                             for split, rows in restored.items()},
                "manifests": {split: {"path": output_path + "/" + FILENAMES[split], "root": roots[ROOT_ROLES[split]],
                                        "sha256": digest(files[FILENAMES[split]]), "count": count[split], "unique_count": unique[split],
                                        "gradient_training": split == "train", "active": not split.startswith("reserved_")} for split in SPLITS}}
    statistics = {"schema_version": "taxosieve_statistics_v1", "inventory_count": len(current),
                  "inventory_unique_sha256": len({row["sha256"] for row in current}), "selected_counts": count,
                  "selected_unique_counts": unique, "excluded_exact_duplicate_count": len(reconciliation["excluded_exact_duplicates"]),
                  "inactive_restored_test_count": sum(len(rows) for rows in restored.values()),
                  "all_images_accounted_for": sum(count.values()) + len(reconciliation["excluded_exact_duplicates"]) + sum(len(rows) for rows in restored.values()) == len(current),
                  "known_species_without_independent_validation": sorted(set(seed["classes"][seed["roots"]["known"]]) - {row["class"] for row in selected["val_known"]}),
                  "per_source": per_source}
    dedup = {"schema_version": "taxosieve_known_identity_v1", "operation": "image_identity_only",
             "inventory_sha256": inventory_hash, "is_new_dataset_version": True,
             "test_known_used_only_for_identity_audit": True, "locked_test_manifest_changed": True,
             "test_known_identity_set_changed": ({row["blob_sha1"] for row in seed["images"] if row["split"] == "test_known"}
                                                 != {row["blob_sha1"] for row in selected["test_known"]}),
             "original_counts": {s: sum(row["assigned_split"] == s for row in current) for s in ("train", "val_known", "test_known")},
             "retained_counts": {s: count[s] for s in ("train", "val_known", "test_known")},
             "unique_test_known_images": unique["test_known"],
             "validation_labels_without_independent_images": sorted(seed["classes"][seed["roots"]["known"]][name] for name in statistics["known_species_without_independent_validation"]),
             "removed_fitting_rows": [row for row in reconciliation["excluded_exact_duplicates"] if row["split"] in ("train", "val_known")]}
    species_roles = {"schema_version": "taxosieve_species_roles_v1", "roots": roots, "classes": seed["classes"],
                     "unknown_gradient_training": False, "reserved_unknown_policy": "inactive; not consumed by training"}
    files.update({"inventory.jsonl": inventory, "protocol.json": json_bytes(protocol), "species_roles.json": json_bytes(species_roles),
                  "split_statistics.json": json_bytes(statistics), "reconciliation.json": json_bytes(reconciliation),
                  "known_deduplication.json": json_bytes(dedup)})
    files["dataset.sha256"] = "".join(digest(data) + "  " + name + "\n" for name, data in sorted(files.items())).encode("ascii")
    if not dry_run:
        if output.exists():
            if not output.is_dir() or any(p.is_symlink() for p in output.iterdir()):
                raise ValueError("Output must be a plain dataset directory")
            actual = {p.name for p in output.iterdir() if p != image_root}
            if actual and (actual != set(files) or any((output / name).read_bytes() != data for name, data in files.items())):
                raise ValueError("Existing dataset differs; choose a NEW --output directory/version: " + str(output))
        else:
            actual = set()
        if not actual:
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = Path(tempfile.mkdtemp(prefix=".taxosieve-build-", dir=str(output.parent)))
            try:
                for name, data in files.items():
                    (temporary / name).write_bytes(data)
                if output.exists():
                    # Images already live in output/image. Validate every
                    # generated byte before installing only the new files.
                    for name in files:
                        (temporary / name).rename(output / name)
                else:
                    temporary.rename(output)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
    return {"output": output_path, "dry_run": dry_run, "statistics": statistics, "reconciliation": reconciliation,
            "inventory_sha256": inventory_hash, "dataset_manifest_sha256": digest(files["dataset.sha256"])}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--image-root", default=DEFAULT_IMAGES)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--seed-dir", default=DEFAULT_SEED)
    parser.add_argument("--dataset-version", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Read/hash/decode and report only; do not create the dataset")
    parser.add_argument("--report", type=Path, help="Optional standalone JSON report; never overwrite a different report")
    args = parser.parse_args(argv)
    result = build(args.project_root, args.image_root, args.output, args.seed_dir, args.dry_run, args.dataset_version)
    if args.report:
        data = json_bytes(result)
        if args.report.is_symlink() or (args.report.exists() and args.report.read_bytes() != data):
            raise ValueError("Existing report differs; choose a new report path")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_bytes(data)
    print(json.dumps({"output": result["output"], "dry_run": result["dry_run"],
                      "inventory_count": result["statistics"]["inventory_count"],
                      "selected_counts": result["statistics"]["selected_counts"],
                      "excluded_exact_duplicate_count": result["statistics"]["excluded_exact_duplicate_count"],
                      "added_count": result["reconciliation"]["added_count"], "removed_count": result["reconciliation"]["removed_count"],
                      "renamed_count": result["reconciliation"]["renamed_count"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
