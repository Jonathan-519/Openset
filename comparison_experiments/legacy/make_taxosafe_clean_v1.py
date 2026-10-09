#!/usr/bin/env python3
"""Create leakage-free TaxoSafe manifests without modifying original files.

Policy for an exact-content duplicate group:
  test > validation > training/OE, then the earliest line within that split.

The command refuses label/status conflicts, rewrites the third CSV field to a
contiguous split-local index, generates clean training/visual YAML files, and
then re-audits its outputs. Compatible with Python 3.8.
"""

import argparse
import copy
import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml

from audit_taxosafe_splits import SPLIT_ORDER, audit, load_configuration, resolve


PRIORITY = {
    "test_known": 30,
    "test_intra": 30,
    "test_extra": 30,
    "val_known": 20,
    "val_intra": 20,
    "val_extra": 20,
    "train": 10,
    "oe_train": 10,
}


def parse_manifest_line(raw, manifest, line_number):
    fields = raw.strip().rsplit(",", 2)
    if len(fields) != 3:
        raise ValueError(
            "{}:{} must contain path,label,index".format(manifest, line_number)
        )
    relative, label_text, index_text = fields
    try:
        label = int(label_text)
        int(index_text)
    except ValueError:
        raise ValueError(
            "{}:{} label/index is not an integer".format(manifest, line_number)
        )
    return relative, label


def duplicate_groups(report):
    groups = {}
    for key in ("cross_split_duplicates", "within_split_duplicates"):
        for group in report[key]:
            groups[group["sha256"]] = group
    return groups


def select_records_to_drop(report):
    if report["status_or_label_conflicts"]:
        raise ValueError(
            "Refusing automatic cleanup: status/label conflicts require review"
        )
    order = {split: index for index, split in enumerate(SPLIT_ORDER)}
    dropped = []
    for sha256, group in sorted(duplicate_groups(report).items()):
        records = group["records"]
        chosen = min(
            records,
            key=lambda record: (
                -PRIORITY[record["split"]],
                order[record["split"]],
                record["line_number"],
            ),
        )
        for record in records:
            if record is not chosen:
                dropped.append(
                    {
                        "sha256": sha256,
                        "dropped": record,
                        "kept": chosen,
                        "reason": (
                            "exact-content duplicate; retain evaluation over "
                            "training, then earliest line"
                        ),
                    }
                )
    return dropped


def protect_outputs(paths, overwrite):
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Outputs already exist; do not mix revisions: {}. Use a new "
            "revision name or inspect before --overwrite.".format(existing)
        )


def write_yaml(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(
            value,
            stream,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Generate exact-content-deduplicated TaxoSafe v1 manifests"
    )
    parser.add_argument(
        "--config",
        default="configs/Zooplankton_Taxonomic_Tree/TaxoSafe_visual.yml",
        help="Existing visual YAML used for the read-only audit",
    )
    parser.add_argument(
        "--output-data-dir",
        default="prepro/data/Zooplankton_Taxonomic_Tree_clean_v1",
    )
    parser.add_argument(
        "--training-config-output",
        default="configs/Zooplankton_Taxonomic_Tree/TaxoSafe_clean_v1.yml",
    )
    parser.add_argument(
        "--visual-config-output",
        default="configs/Zooplankton_Taxonomic_Tree/TaxoSafe_visual_clean_v1.yml",
    )
    parser.add_argument(
        "--cleanup-report",
        default="taxosafe_clean_v1_report.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project_root = Path.cwd().resolve()
    cfg, extension_path, base_path = load_configuration(project_root, args.config)
    with extension_path.open("r", encoding="utf-8-sig") as stream:
        extension = yaml.safe_load(stream)
    if not isinstance(extension, dict) or "visual_support" not in extension:
        raise ValueError("Input must be the TaxoSafe visual extension YAML")

    original_report = audit(project_root, args.config)
    dropped = select_records_to_drop(original_report)
    drop_lines = defaultdict(set)
    for item in dropped:
        record = item["dropped"]
        drop_lines[record["split"]].add(int(record["line_number"]))

    output_dir = resolve(project_root, args.output_data_dir)
    training_config_path = resolve(project_root, args.training_config_output)
    visual_config_path = resolve(project_root, args.visual_config_output)
    cleanup_report_path = resolve(project_root, args.cleanup_report)
    manifest_outputs = {
        split: output_dir / ("gt_{}.txt".format(split))
        for split in SPLIT_ORDER
        if cfg["data"].get(split)
    }
    protect_outputs(
        list(manifest_outputs.values())
        + [training_config_path, visual_config_path, cleanup_report_path],
        args.overwrite,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    counts = {}
    for split, destination in manifest_outputs.items():
        source = resolve(project_root, cfg["data"][split]).resolve()
        output_rows = []
        with source.open("r", encoding="utf-8-sig") as stream:
            for line_number, raw in enumerate(stream, 1):
                if not raw.strip():
                    continue
                relative, label = parse_manifest_line(raw, source, line_number)
                if line_number in drop_lines[split]:
                    continue
                output_rows.append((relative, label))
        with destination.open("w", encoding="utf-8") as stream:
            for new_index, (relative, label) in enumerate(output_rows):
                stream.write("{},{},{}\n".format(relative, label, new_index))
        counts[split] = {
            "before": int(original_report["split_summaries"][split]["record_count"]),
            "after": len(output_rows),
            "removed": int(original_report["split_summaries"][split]["record_count"])
            - len(output_rows),
        }

    clean_cfg = copy.deepcopy(cfg)
    for split, destination in manifest_outputs.items():
        clean_cfg["data"][split] = str(destination.relative_to(project_root))
    clean_cfg["data"]["name"] = "Zooplankton_TaxoSafe_fold1_clean_v1"
    clean_cfg["data"]["split_revision"] = "exact-content-deduplicated-v1"
    clean_cfg["exp"] = "ViT-B_16/full/TaxoSafe-v3-clean-v1/fold1"
    write_yaml(training_config_path, clean_cfg)

    clean_extension = {
        "base_config": str(training_config_path.relative_to(project_root)),
        "visual_support": copy.deepcopy(extension["visual_support"]),
    }
    write_yaml(visual_config_path, clean_extension)

    clean_report = audit(project_root, str(visual_config_path))
    if clean_report["totals"]["cross_split_duplicate_groups"] != 0:
        raise RuntimeError("Generated manifests still contain cross-split duplicates")
    if clean_report["totals"]["within_split_duplicate_groups"] != 0:
        raise RuntimeError("Generated manifests still contain within-split duplicates")

    report = {
        "schema_version": 1,
        "policy": "test > validation > training/OE; earliest line within split",
        "original_config": str(base_path),
        "generated_training_config": str(training_config_path),
        "generated_visual_config": str(visual_config_path),
        "counts": counts,
        "removed_record_count": len(dropped),
        "removed_records": dropped,
        "post_cleanup_audit": clean_report,
        "requires_retraining": True,
        "warning": (
            "This produces leakage-free exact-byte manifests, but does not "
            "detect near-duplicates or restore an untouched final test set."
        ),
    }
    cleanup_report_path.parent.mkdir(parents=True, exist_ok=True)
    with cleanup_report_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")

    print("Clean TaxoSafe manifests generated; original files were NOT modified.")
    for split in SPLIT_ORDER:
        if split in counts:
            value = counts[split]
            print(
                "{}: {} -> {} (removed {})".format(
                    split, value["before"], value["after"], value["removed"]
                )
            )
    print("training config: {}".format(training_config_path))
    print("visual config: {}".format(visual_config_path))
    print("cleanup report: {}".format(cleanup_report_path))
    print("Post-cleanup exact-duplicate audit: PASS")
    print("STOP: retrain a new experiment; do not reuse Trial 4 best.pth.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
