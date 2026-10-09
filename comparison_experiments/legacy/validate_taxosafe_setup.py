"""Preflight validation for TaxoSafe hierarchy, split files and image paths."""

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
EXPECTED_PARENTS = [
    "Amphipoda",
    "Appendiculata",
    "Cladocera",
    "Copepoda",
    "Euphausiacea",
    "Medusae",
    "Sagittoidea",
]
EXPECTED_KNOWN_SPECIES_PARENT = {
    _normalised: parent
    for parent, names in {
        "Amphipoda": ["Themisto_gracilipes"],
        "Appendiculata": ["Oikopleura"],
        "Cladocera": ["Evadne_tergestina", "Penilia_avirostris"],
        "Copepoda": [
            "Acartia_hongi",
            "Calanus_sinicus",
            "Centropages_dorsispinatus",
            "Eurytemora_pacifica",
            "Oithona_plumifera",
            "Paracalanus_parvus",
        ],
        "Euphausiacea": ["Euphausia_pacifica"],
        "Medusae": [
            "Clytia_folleata",
            "Muggiaea_atlantica",
            "Obelia_dichotoma",
            "Proboscidactyla_flavicirrata",
            "Sugiura_chengshanense",
        ],
        "Sagittoidea": ["Sagitta"],
    }.items()
    for name in names
    for _normalised in [re.sub(r"[^a-z0-9]+", "", name.lower())]
}
SPLIT_KINDS = {
    "train": "known",
    "val_known": "known",
    "test_known": "known",
    "val_intra": "intra",
    "test_intra": "intra",
    "dev_unknown": "intra",
    "test_unknown": "intra",
    "oe_train": "extra",
    "val_extra": "extra",
    "test_extra": "extra",
}


def _normalise(value):
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_split(path):
    records = []
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, raw in enumerate(stream, 1):
            line = raw.strip()
            if not line:
                continue
            fields = [field.strip() for field in line.rsplit(",", 2)]
            if len(fields) != 3:
                raise ValueError(
                    "{}:{} must have path,label,index".format(
                        path, line_number
                    )
                )
            try:
                label = int(fields[1])
                index = int(fields[2])
            except ValueError as error:
                raise ValueError(
                    "{}:{} label/index must be integers".format(
                        path, line_number
                    )
                ) from error
            records.append({
                "path": fields[0],
                "label": label,
                "index": index,
                "line": line_number,
            })
    if not records:
        raise ValueError("{} is empty".format(path))
    return records


def _coerce_leaf_mapping(value):
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if isinstance(value, dict):
        output = {}
        for key, item in value.items():
            if isinstance(key, (int, np.integer)):
                output[int(key)] = str(item)
            elif isinstance(item, (int, np.integer)):
                output[int(item)] = str(key)
        return output
    if isinstance(value, (list, tuple, np.ndarray)):
        return {index: str(item) for index, item in enumerate(value)}
    return {}


def _load_leaf_mapping(hierarchy_path):
    # Prefer the explicit local label map produced together with tree.npy.
    fallback = hierarchy_path.parent / "leaf_nodes.npy"
    if fallback.is_file():
        mapping = _coerce_leaf_mapping(
            np.load(str(fallback), allow_pickle=True)
        )
        if mapping:
            return mapping

    # Loading tree.npy requires the repository's loader package to be present,
    # which is true when this script is copied to the ProTeCt project root.
    tree = np.load(str(hierarchy_path), allow_pickle=True)
    if isinstance(tree, np.ndarray) and tree.shape == ():
        tree = tree.item()
    mapping = _coerce_leaf_mapping(getattr(tree, "leaf_nodes", None))
    if not mapping:
        raise ValueError(
            "Cannot read leaf IDs from {} or {}".format(
                hierarchy_path, fallback
            )
        )
    return mapping


def _candidate_image_paths(cfg, split_name, relative_path):
    value = Path(relative_path)
    if value.is_absolute():
        return [value]
    data = cfg["data"]
    if split_name in {
        "val_intra", "test_intra", "dev_unknown", "test_unknown"
    }:
        roots = [data.get("full_data_root"), data.get("data_root")]
    elif split_name in {"oe_train", "val_extra", "test_extra"}:
        roots = [data.get("ood_root"), data.get("data_root")]
    else:
        roots = [data.get("data_root")]
    paths = [PROJECT_ROOT / value]
    for root in roots:
        if root is not None:
            paths.append(_resolve(root) / value)
    # Preserve order while removing duplicates.
    return list(dict.fromkeys(path.resolve(strict=False) for path in paths))


def validate(cfg, skip_image_check=False, max_missing_report=20):
    errors = []
    warnings = []
    data = cfg.get("data", {})
    sampler = data.get("sampler", {})
    if sampler.get("name") == "hierarchical_episode":
        episode_size = (
            int(sampler.get("parents_per_batch", 0))
            * int(sampler.get("species_per_parent", 0))
            * int(sampler.get("images_per_species", 0))
        )
        if episode_size != int(data.get("batch_size", -1)):
            errors.append(
                "Hierarchical episode size {} differs from data.batch_size {}"
                .format(episode_size, data.get("batch_size"))
            )
    if cfg.get("open_treecut", {}).get("force_one_pseudo_per_batch", False):
        if sampler.get("name") != "hierarchical_episode":
            errors.append(
                "force_one_pseudo_per_batch requires hierarchical_episode"
            )
        if not sampler.get("ensure_rankable_parent", False):
            errors.append(
                "force_one_pseudo_per_batch requires "
                "sampler.ensure_rankable_parent: true"
            )
    open_cfg = cfg.get("open_treecut", {})
    if sampler.get("name") == "hierarchical_episode":
        open_mode = str(open_cfg.get("holdout_mode", "epoch")).lower()
        sampler_mode = str(
            sampler.get("pseudo_holdout_mode", open_mode)
        ).lower()
        if open_mode != sampler_mode:
            errors.append(
                "open_treecut.holdout_mode and "
                "data.sampler.pseudo_holdout_mode must match"
            )
        open_seed = int(open_cfg.get("holdout_seed", cfg.get("seed", 1)))
        sampler_seed = int(sampler.get("holdout_seed", open_seed))
        if open_seed != sampler_seed:
            errors.append(
                "open_treecut.holdout_seed and "
                "data.sampler.holdout_seed must match"
            )
    loss_cfg = cfg.get("loss", {})
    if float(loss_cfg.get("known_child_margin", 0.30)) <= float(
        loss_cfg.get("pseudo_child_margin", 0.28)
    ):
        errors.append(
            "loss.known_child_margin must be greater than "
            "loss.pseudo_child_margin"
        )
    if cfg.get("optim", {}).get("sgd_dampening") is None:
        errors.append(
            "optim.sgd_dampening must be numeric (use 0.0 for this project)"
        )
    expected_leaves = int(data.get("num_known_leaves", 17))
    hierarchy_path = _resolve(data["hierarchy"])
    if not hierarchy_path.is_file():
        return ["Missing hierarchy: {}".format(hierarchy_path)], warnings

    try:
        leaf_mapping = _load_leaf_mapping(hierarchy_path)
    except Exception as error:
        return ["Cannot load hierarchy leaf mapping: {}".format(error)], warnings
    expected_ids = list(range(expected_leaves))
    if sorted(leaf_mapping) != expected_ids:
        errors.append(
            "tree.npy leaf IDs must be {}, got {}".format(
                expected_ids, sorted(leaf_mapping)
            )
        )
    tree_species = {_normalise(name) for name in leaf_mapping.values()}
    expected_species = set(EXPECTED_KNOWN_SPECIES_PARENT)
    if tree_species != expected_species:
        errors.append(
            "tree/leaf_nodes species differ from Fold 1 definition; "
            "missing={}, unexpected={}".format(
                sorted(expected_species - tree_species),
                sorted(tree_species - expected_species),
            )
        )

    all_records = {}
    missing_images = []
    is_taxolocal = "dev_unknown" in data or "test_unknown" in data
    required_splits = (
        {
            "train", "val_known", "test_known", "dev_unknown",
            "test_unknown", "val_extra", "test_extra",
        }
        if is_taxolocal
        else set(SPLIT_KINDS).difference({"dev_unknown", "test_unknown"})
    )
    for split_name, kind in SPLIT_KINDS.items():
        configured = data.get(split_name)
        if not configured:
            if split_name in required_splits:
                errors.append(
                    "data.{} is missing from YAML".format(split_name)
                )
            continue
        split_path = _resolve(configured)
        if not split_path.is_file():
            errors.append("Missing {} list: {}".format(split_name, split_path))
            continue
        try:
            records = _read_split(split_path)
        except Exception as error:
            errors.append(str(error))
            continue
        all_records[split_name] = records

        seen_indices = set()
        for record in records:
            label = record["label"]
            location = "{}:{}".format(split_path, record["line"])
            if record["index"] in seen_indices:
                errors.append(
                    "{} duplicates index {}".format(location, record["index"])
                )
            seen_indices.add(record["index"])
            if kind == "known" and not 0 <= label < expected_leaves:
                errors.append("{} known label {} is out of range".format(
                    location, label
                ))
            elif kind == "intra" and not 0 <= label < len(EXPECTED_PARENTS):
                errors.append("{} intra parent label {} is invalid".format(
                    location, label
                ))
            elif kind == "extra" and label != -1:
                errors.append("{} extra/OE label must be -1".format(location))

            parts = Path(record["path"]).parts
            if kind == "known" and len(parts) >= 2 and label in leaf_mapping:
                species_folder = parts[-2]
                species_key = _normalise(species_folder)
                if species_key != _normalise(
                    leaf_mapping[label]
                ):
                    errors.append(
                        "{} path species '{}' disagrees with tree leaf {} "
                        "('{}')".format(
                            location,
                            species_folder,
                            label,
                            leaf_mapping[label],
                        )
                    )
                if len(parts) >= 3 and species_key in EXPECTED_KNOWN_SPECIES_PARENT:
                    parent_folder = parts[-3]
                    expected_parent = EXPECTED_KNOWN_SPECIES_PARENT[species_key]
                    if _normalise(parent_folder) != _normalise(expected_parent):
                        errors.append(
                            "{} known path parent '{}' disagrees with '{}'"
                            .format(location, parent_folder, expected_parent)
                        )
            if kind == "intra" and len(parts) >= 3:
                parent_folder = parts[-3]
                if _normalise(parent_folder) != _normalise(
                    EXPECTED_PARENTS[label]
                ):
                    errors.append(
                        "{} path parent '{}' disagrees with parent ID {} "
                        "('{}')".format(
                            location,
                            parent_folder,
                            label,
                            EXPECTED_PARENTS[label],
                        )
                    )

            if not skip_image_check:
                candidates = _candidate_image_paths(
                    cfg, split_name, record["path"]
                )
                existing = next(
                    (path for path in candidates if path.is_file()), None
                )
                if existing is None:
                    missing_images.append((location, record["path"], candidates))
                else:
                    record["canonical_path"] = str(existing.resolve())

    if missing_images:
        for location, relative, candidates in missing_images[:max_missing_report]:
            errors.append(
                "{} image not found for '{}'; tried {}".format(
                    location, relative, [str(path) for path in candidates]
                )
            )
        if len(missing_images) > max_missing_report:
            errors.append(
                "... plus {} more missing images".format(
                    len(missing_images) - max_missing_report
                )
            )

    # No image may cross train/validation/test boundaries within a protocol.
    groups = (
        ("train", "val_known", "test_known"),
        ("val_intra", "test_intra", "dev_unknown", "test_unknown"),
        ("oe_train", "val_extra", "test_extra"),
    )
    for group in groups:
        for left_index, left in enumerate(group):
            if left not in all_records:
                continue
            left_paths = {
                record.get("canonical_path", record["path"])
                for record in all_records[left]
            }
            for right in group[left_index + 1:]:
                if right not in all_records:
                    continue
                overlap = left_paths.intersection(
                    record.get("canonical_path", record["path"])
                    for record in all_records[right]
                )
                if overlap:
                    errors.append(
                        "{} and {} overlap on {} image paths (example: {})"
                        .format(left, right, len(overlap), sorted(overlap)[0])
                    )

    known_species = {
        _normalise(Path(record["path"]).parts[-2])
        for split_name in ("train", "val_known", "test_known")
        for record in all_records.get(split_name, [])
        if len(Path(record["path"]).parts) >= 2
    }
    intra_species = {
        _normalise(Path(record["path"]).parts[-2])
        for split_name in (
            "val_intra", "test_intra", "dev_unknown", "test_unknown"
        )
        for record in all_records.get(split_name, [])
        if len(Path(record["path"]).parts) >= 2
    }
    species_overlap = known_species.intersection(intra_species)
    if species_overlap:
        errors.append(
            "Known and intra-unknown splits share {} species folder(s): {}"
            .format(len(species_overlap), sorted(species_overlap))
        )

    if is_taxolocal:
        development_species = {
            _normalise(Path(record["path"]).parts[-2])
            for record in all_records.get("dev_unknown", [])
            if len(Path(record["path"]).parts) >= 2
        }
        locked_species = {
            _normalise(Path(record["path"]).parts[-2])
            for record in all_records.get("test_unknown", [])
            if len(Path(record["path"]).parts) >= 2
        }
        overlap = development_species.intersection(locked_species)
        if overlap:
            errors.append(
                "Development and locked unknown splits share species: {}"
                .format(sorted(overlap))
            )

    if float(loss_cfg.get("lambda_oe", 0.0)) > 0.0:
        oe_count = len(all_records.get("oe_train", []))
        minimum_oe = int(loss_cfg.get("min_oe_train_samples", 100))
        if oe_count < minimum_oe:
            message = (
                "lambda_oe is enabled with only {} OE records (minimum {}); "
                "expand and deduplicate hard-OOD data."
                .format(oe_count, minimum_oe)
            )
            if loss_cfg.get("allow_small_oe", False):
                warnings.append(message)
            else:
                errors.append(message)

    if not errors:
        warnings.append(
            "Preflight passed. This validates IDs/paths/split separation, "
            "not image quality or biological labels."
        )
    return errors, warnings


def parse_args():
    parser = argparse.ArgumentParser(description="Validate TaxoSafe inputs")
    parser.add_argument("--config", required=True)
    parser.add_argument("--skip-image-check", action="store_true")
    parser.add_argument("--max-missing-report", type=int, default=20)
    parser.add_argument("--max-error-report", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = _resolve(args.config)
    with config_path.open("r", encoding="utf-8-sig") as stream:
        cfg = yaml.load(stream, Loader=yaml.SafeLoader)
    if not isinstance(cfg, dict):
        raise ValueError("YAML root must be a mapping")
    errors, warnings = validate(
        cfg,
        skip_image_check=args.skip_image_check,
        max_missing_report=args.max_missing_report,
    )
    for warning in warnings:
        print("WARNING: {}".format(warning))
    if errors:
        print("TaxoSafe preflight FAILED with {} error(s):".format(len(errors)))
        for error in errors[:args.max_error_report]:
            print("  - {}".format(error))
        if len(errors) > args.max_error_report:
            print(
                "  ... {} additional errors omitted".format(
                    len(errors) - args.max_error_report
                )
            )
        return 1
    print("TaxoSafe preflight PASSED")
    print("Known leaf mapping:")
    leaf_mapping = _load_leaf_mapping(_resolve(cfg["data"]["hierarchy"]))
    for leaf_id in sorted(leaf_mapping):
        print("  {:2d}  {}".format(leaf_id, leaf_mapping[leaf_id]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
