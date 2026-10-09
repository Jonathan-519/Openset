#!/usr/bin/env python3
"""Audit exact-image leakage and duplicate records in TaxoSafe manifests.

This command is read-only: it never edits manifests or images.  Run it from
the ProTeCt repository root and send the generated JSON for review.
Compatible with Python 3.8.
"""

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml


SPLIT_ORDER = (
    "train",
    "val_known",
    "test_known",
    "oe_train",
    "val_intra",
    "test_intra",
    "val_extra",
    "test_extra",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_yaml(path):
    with path.open("r", encoding="utf-8-sig") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError("YAML root must be a mapping: {}".format(path))
    return value


def resolve(project_root, value):
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def load_configuration(project_root, config_path):
    extension_path = resolve(project_root, config_path)
    extension = load_yaml(extension_path)
    if "base_config" in extension:
        base_path = resolve(project_root, extension["base_config"])
        cfg = load_yaml(base_path)
    else:
        base_path = extension_path
        cfg = extension
    if not isinstance(cfg.get("data"), dict):
        raise ValueError("Configuration has no data mapping")
    return cfg, extension_path.resolve(), base_path.resolve()


def split_kind(split):
    if split in ("train", "val_known", "test_known"):
        return "known"
    if split in ("val_intra", "test_intra"):
        return "intra"
    return "extra"


def split_root(project_root, data, split):
    kind = split_kind(split)
    if kind == "known":
        key = "data_root"
    elif kind == "intra":
        key = "full_data_root"
    else:
        key = "ood_root"
    if key not in data:
        raise ValueError("data.{} is required for {}".format(key, split))
    return resolve(project_root, data[key]).resolve()


def read_manifest(project_root, data, split):
    manifest_value = data.get(split)
    if not manifest_value:
        return [], {"status": "not_configured"}
    manifest = resolve(project_root, manifest_value).resolve()
    if not manifest.is_file():
        return [], {"status": "missing", "manifest": str(manifest)}
    root = split_root(project_root, data, split)
    records = []
    with manifest.open("r", encoding="utf-8-sig") as stream:
        for line_number, raw in enumerate(stream, 1):
            stripped = raw.strip()
            if not stripped:
                continue
            fields = stripped.rsplit(",", 2)
            if len(fields) != 3:
                raise ValueError(
                    "{}:{} must contain path,label,index".format(
                        manifest, line_number
                    )
                )
            relative_path, label_text, index_text = fields
            try:
                label = int(label_text)
                index = int(index_text)
            except ValueError:
                raise ValueError(
                    "{}:{} label/index is not an integer".format(
                        manifest, line_number
                    )
                )
            image = Path(relative_path)
            if not image.is_absolute():
                image = root / image
            image = image.resolve()
            if not image.is_file():
                raise FileNotFoundError(
                    "{}:{} image not found: {}".format(
                        manifest, line_number, image
                    )
                )
            records.append(
                {
                    "split": split,
                    "kind": split_kind(split),
                    "manifest": str(manifest),
                    "line_number": line_number,
                    "relative_path": relative_path,
                    "resolved_path": str(image),
                    "label": label,
                    "index": index,
                    "sha256": sha256_file(image),
                }
            )
    return records, {
        "status": "ok",
        "manifest": str(manifest),
        "root": str(root),
        "record_count": len(records),
        "unique_content_count": len({record["sha256"] for record in records}),
    }


def group_details(records):
    return {
        "sha256": records[0]["sha256"],
        "record_count": len(records),
        "splits": sorted({record["split"] for record in records}),
        "kinds": sorted({record["kind"] for record in records}),
        "labels": sorted({record["label"] for record in records}),
        "has_status_or_label_conflict": len(
            {(record["kind"], record["label"]) for record in records}
        ) > 1,
        "records": records,
    }


def audit(project_root, config_path):
    cfg, extension_path, base_path = load_configuration(project_root, config_path)
    data = cfg["data"]
    all_records = []
    summaries = {}
    for split in SPLIT_ORDER:
        records, summary = read_manifest(project_root, data, split)
        all_records.extend(records)
        summaries[split] = summary

    by_hash = defaultdict(list)
    for record in all_records:
        by_hash[record["sha256"]].append(record)

    within = []
    cross = []
    conflicts = []
    for records in by_hash.values():
        if len(records) < 2:
            continue
        details = group_details(records)
        counts = defaultdict(int)
        for record in records:
            counts[record["split"]] += 1
        if any(count > 1 for count in counts.values()):
            within.append(details)
        if len(details["splits"]) > 1:
            cross.append(details)
        if details["has_status_or_label_conflict"]:
            conflicts.append(details)

    sort_key = lambda group: (group["splits"], group["sha256"])
    return {
        "schema_version": 1,
        "read_only": True,
        "project_root": str(project_root.resolve()),
        "requested_config": str(extension_path),
        "resolved_base_config": str(base_path),
        "split_summaries": summaries,
        "totals": {
            "records": len(all_records),
            "unique_image_contents": len(by_hash),
            "within_split_duplicate_groups": len(within),
            "cross_split_duplicate_groups": len(cross),
            "status_or_label_conflict_groups": len(conflicts),
        },
        "cross_split_duplicates": sorted(cross, key=sort_key),
        "within_split_duplicates": sorted(within, key=sort_key),
        "status_or_label_conflicts": sorted(conflicts, key=sort_key),
        "interpretation": {
            "cross_split_duplicates": (
                "Exact image bytes occur in more than one split. Do not use "
                "these manifests for a leakage-free experiment."
            ),
            "within_split_duplicates": (
                "Repeated bytes inside one split alter sample weighting and "
                "effective sample size."
            ),
            "status_or_label_conflicts": (
                "The same bytes have different split status or label; inspect "
                "manually before generating any replacement manifest."
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Read-only exact-duplicate audit for TaxoSafe manifests"
    )
    parser.add_argument(
        "--config",
        default="configs/Zooplankton_Taxonomic_Tree/TaxoSafe_visual.yml",
    )
    parser.add_argument(
        "--output", default="taxosafe_split_audit.json"
    )
    args = parser.parse_args()
    project_root = Path.cwd()
    report = audit(project_root, args.config)
    output = resolve(project_root, args.output)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")

    totals = report["totals"]
    print("TaxoSafe split audit complete (read-only)")
    print("records: {}".format(totals["records"]))
    print("unique image contents: {}".format(totals["unique_image_contents"]))
    print("within-split duplicate groups: {}".format(
        totals["within_split_duplicate_groups"]
    ))
    print("cross-split duplicate groups: {}".format(
        totals["cross_split_duplicate_groups"]
    ))
    print("status/label conflict groups: {}".format(
        totals["status_or_label_conflict_groups"]
    ))
    print("report: {}".format(output.resolve()))
    if totals["cross_split_duplicate_groups"]:
        print("STOP: inspect the JSON before calibration or testing.")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
