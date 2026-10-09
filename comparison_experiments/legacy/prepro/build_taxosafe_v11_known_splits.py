"""Prepare disjoint KNOWN fitting manifests; retain the locked test manifest.

This identity-only audit reads known image bytes, including test-known bytes,
without decoding images or fitting features. No unknown images are accessed.
"""
import argparse
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/Zooplankton_Taxonomic_Tree/Zooplankton_Taxonomic_Tree_v10_perf.yml"
DEFAULT_OUTPUT = ROOT / "prepro/data/Zooplankton_TT_v11_dcbs"


def digest(path):
    value = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def prepare(config=DEFAULT_CONFIG, output=DEFAULT_OUTPUT, root=ROOT):
    root, output = Path(root), Path(output)
    cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))["data"]
    locate = lambda p: Path(p) if Path(p).is_absolute() else root / p
    image_root = locate(cfg["data_root"])
    rows, inputs, identities = {}, {}, {}
    for split in ("train", "val_known", "test_known"):
        manifest = locate(cfg[split])
        rows[split] = []
        inputs[split] = {"manifest": str(cfg[split]), "sha256": digest(manifest)}
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            path, label, index = line.rsplit(",", 2)
            label, index = int(label), int(index)
            resolved = (image_root / path).resolve()
            resolved.relative_to(image_root.resolve())
            sha = digest(resolved)
            if sha in identities and identities[sha] != label:
                raise ValueError("Same image bytes carry conflicting known labels: " + path)
            identities[sha] = label
            rows[split].append({"path": path, "label": label, "sha256": sha, "line": line})
    owners = {r["sha256"]: ("test_known", r["path"]) for r in rows["test_known"]}
    kept, removed = {"test_known": rows["test_known"]}, []
    for split in ("val_known", "train"):
        kept[split] = []
        for row in rows[split]:
            sha = row["sha256"]
            if sha in owners:
                other_split, other_path = owners[sha]
                removed.append({"split": split, "path": row["path"], "sha256": sha,
                                "same_as_split": other_split, "same_as_path": other_path})
            else:
                kept[split].append(row)
                owners[sha] = split, row["path"]
    if any(not kept[s] for s in kept):
        raise ValueError("Identity preparation would empty a split")
    if {r["label"] for r in kept["train"]} != {r["label"] for r in rows["train"]}:
        raise ValueError("Identity preparation would remove all training support for a known leaf")
    report = {"schema_version": 11, "operation": "known_image_identity_only",
              "unknown_images_read": False, "test_known_used_only_for_identity_audit": True,
              "locked_test_manifest_changed": False, "inputs": inputs,
              "original_counts": {s: len(r) for s, r in rows.items()},
              "retained_counts": {s: len(kept[s]) for s in rows},
              "unique_test_known_images": len({r["sha256"] for r in rows["test_known"]}),
              "validation_labels_without_independent_images": sorted({r["label"] for r in rows["val_known"]} - {r["label"] for r in kept["val_known"]}),
              "removed_fitting_rows": removed}
    files = {"gt_train_known.txt": "\n".join(r["line"] for r in kept["train"]) + "\n",
             "gt_val_known.txt": "\n".join(r["line"] for r in kept["val_known"]) + "\n",
             "known_deduplication.json": json.dumps(report, ensure_ascii=False, indent=2) + "\n"}
    for name, content in files.items():
        destination = output / name
        if destination.exists() and destination.read_text(encoding="utf-8") != content:
            raise ValueError("Existing preparation differs; use a fresh output directory: " + str(destination))
    output.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (output / name).write_text(content, encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = prepare(args.config, args.output)
    print(json.dumps({k: result[k] for k in ("original_counts", "retained_counts", "unique_test_known_images", "validation_labels_without_independent_images")}, indent=2))
