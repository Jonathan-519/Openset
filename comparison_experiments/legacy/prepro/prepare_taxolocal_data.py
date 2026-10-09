#!/usr/bin/env python3
"""Build species-disjoint development and locked novel-species manifests.

The final test assignment is declared in source and never sampled from image
rows. Exact duplicates and conservative within-species pHash duplicates are
removed before manifests are written. Images are never copied or modified.
"""

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
PARENT_IDS = {
    "Amphipoda": 0,
    "Appendiculata": 1,
    "Cladocera": 2,
    "Copepoda": 3,
    "Euphausiacea": 4,
    "Medusae": 5,
    "Sagittoidea": 6,
}

# Balanced by parent and image count while keeping every species intact.
SPECIES_ASSIGNMENT = {
    "development": {
        "Copepoda": (
            "Calanopia_thompsoni",
            "Pseudodiaptomus_marinus",
        ),
        "Medusae": (
            "Aurelia_aurita",
            "Bougainvillia_muscus",
            "Zanclea_costata",
        ),
    },
    "locked_test": {
        "Copepoda": (
            "Corycaeus_affinis",
            "Pontellopsis_tenuicauda",
            "Tortanus_derjugini",
        ),
        "Medusae": (
            "Dipurena_ophiogaeter",
            "Gonionemus_vertens",
        ),
    },
}


def natural_key(path):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def phash(path, image_size=32, low_frequency_size=8):
    """Dependency-light 63-bit DCT perceptual hash."""
    with Image.open(path) as image:
        image.load()
        pixels = np.asarray(
            image.convert("L").resize(
                (image_size, image_size), Image.Resampling.LANCZOS
            ),
            dtype=np.float64,
        )
    x = np.arange(image_size)
    u = np.arange(low_frequency_size)[:, None]
    cosine = np.cos(
        np.pi * (2 * x + 1) * u / (2 * image_size)
    )
    cosine[0] /= math.sqrt(2.0)
    coefficients = cosine @ pixels @ cosine.T
    values = coefficients[:low_frequency_size, :low_frequency_size]
    values = values.reshape(-1)[1:]
    median = float(np.median(values))
    output = 0
    for value in values:
        output = (output << 1) | int(value > median)
    return output


def image_paths(folder):
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    paths = [
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if not paths:
        raise ValueError("No images in {}".format(folder))
    return sorted(paths, key=natural_key)


def components(paths, hashes, threshold):
    parent = list(range(len(paths)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first, second):
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    for first in range(len(paths)):
        for second in range(first + 1, len(paths)):
            distance = bin(hashes[first] ^ hashes[second]).count("1")
            if distance <= threshold:
                union(first, second)
    groups = defaultdict(list)
    for index, path in enumerate(paths):
        groups[find(index)].append(path)
    return [group for group in groups.values() if len(group) > 1]


def deduplicate_species(paths, phash_threshold, display_root=None):
    def display(path):
        if display_root is None:
            return str(path)
        return path.relative_to(display_root).as_posix()

    exact_seen = {}
    exact_removed = []
    exact_unique = []
    for path in paths:
        digest = sha256_file(path)
        if digest in exact_seen:
            exact_removed.append({
                "dropped": display(path),
                "kept": display(exact_seen[digest]),
                "reason": "exact_sha256",
            })
        else:
            exact_seen[digest] = path
            exact_unique.append(path)

    perceptual = [phash(path) for path in exact_unique]
    near_groups = components(exact_unique, perceptual, phash_threshold)
    near_removed = []
    removed_paths = set()
    for group in near_groups:
        kept = min(group, key=natural_key)
        for path in group:
            if path == kept:
                continue
            removed_paths.add(path)
            near_removed.append({
                "dropped": display(path),
                "kept": display(kept),
                "reason": "within_species_phash_le_{}".format(
                    phash_threshold
                ),
            })
    retained = [path for path in exact_unique if path not in removed_paths]
    return retained, exact_removed, near_removed


def discovered_species(new_root):
    return {
        (parent.name, species.name)
        for parent in new_root.iterdir()
        if parent.is_dir()
        for species in parent.iterdir()
        if species.is_dir()
    }


def declared_species():
    entries = []
    for split, parents in SPECIES_ASSIGNMENT.items():
        for parent, species_names in parents.items():
            for species in species_names:
                entries.append((split, parent, species))
    return entries


def write_manifest(path, rows, overwrite):
    if path.exists() and not overwrite:
        raise FileExistsError(
            "Output exists; use --overwrite after review: {}".format(path)
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for index, (relative, label) in enumerate(rows):
            stream.write("{},{},{}\n".format(relative, label, index))


def remap_known_manifest(project_root, source):
    """Replace the untracked fold1_known symlink view with raw image paths."""
    rows = []
    with source.open("r", encoding="utf-8-sig") as stream:
        for line_number, raw in enumerate(stream, 1):
            if not raw.strip():
                continue
            fields = raw.strip().rsplit(",", 2)
            if len(fields) != 3:
                raise ValueError(
                    "{}:{} must contain path,label,index".format(
                        source, line_number
                    )
                )
            old_path, label_text, _ = fields
            parts = Path(old_path).parts
            try:
                view_index = parts.index("fold1_known")
            except ValueError as error:
                raise ValueError(
                    "Unexpected known-view path: {}".format(old_path)
                ) from error
            suffix = parts[view_index + 1 :]
            relative = Path(
                "raw/Zooplankton_Taxonomic_Tree", *suffix
            ).as_posix()
            if not (project_root / "prepro" / relative).is_file():
                raise FileNotFoundError(project_root / "prepro" / relative)
            rows.append((relative, int(label_text)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-root", default="prepro/raw/new")
    parser.add_argument("--reference-root", default="prepro/raw")
    parser.add_argument(
        "--output-dir",
        default="prepro/data/Zooplankton_Taxonomic_Tree_taxolocal_v1",
    )
    parser.add_argument(
        "--known-manifest-dir",
        default="prepro/data/Zooplankton_Taxonomic_Tree_clean_v1",
    )
    parser.add_argument("--phash-threshold", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project_root = Path.cwd().resolve()
    new_root = (project_root / args.new_root).resolve()
    reference_root = (project_root / args.reference_root).resolve()
    output_dir = (project_root / args.output_dir).resolve()
    known_manifest_dir = (project_root / args.known_manifest_dir).resolve()
    if not 0 <= args.phash_threshold <= 8:
        raise ValueError("phash-threshold must be between 0 and 8")

    declared = declared_species()
    declared_pairs = {(parent, species) for _, parent, species in declared}
    actual_pairs = discovered_species(new_root)
    if declared_pairs != actual_pairs:
        raise ValueError(
            "Assignment does not exactly cover new species; missing={}, "
            "unexpected={}".format(
                sorted(actual_pairs - declared_pairs),
                sorted(declared_pairs - actual_pairs),
            )
        )
    if len(declared_pairs) != len(declared):
        raise ValueError("A species appears in more than one partition")

    rows_by_split = {"development": [], "locked_test": []}
    species_report = []
    all_retained_hashes = {}
    reference_hashes = {}
    reference_image_count = 0
    for path in reference_root.rglob("*"):
        if (
            not path.is_file()
            or path.suffix.lower() not in IMAGE_SUFFIXES
            or new_root == path
            or new_root in path.parents
        ):
            continue
        reference_image_count += 1
        reference_hashes.setdefault(sha256_file(path), path)
    for split, parent, species in declared:
        folder = new_root / parent / species
        original = image_paths(folder)
        retained, exact_removed, near_removed = deduplicate_species(
            original, args.phash_threshold, display_root=project_root
        )
        for path in retained:
            digest = sha256_file(path)
            if digest in all_retained_hashes:
                raise ValueError(
                    "Cross-species exact duplicate: {} and {}".format(
                        all_retained_hashes[digest], path
                    )
                )
            if digest in reference_hashes:
                raise ValueError(
                    "New data duplicates an existing protocol image: {} and "
                    "{}".format(reference_hashes[digest], path)
                )
            all_retained_hashes[digest] = path
            relative = path.relative_to(project_root / "prepro").as_posix()
            rows_by_split[split].append((relative, PARENT_IDS[parent]))
        species_report.append({
            "partition": split,
            "parent": parent,
            "parent_id": PARENT_IDS[parent],
            "species": species,
            "original_count": len(original),
            "retained_count": len(retained),
            "exact_removed": exact_removed,
            "perceptual_removed": near_removed,
        })

    outputs = {
        "development": output_dir / "gt_dev_unknown.txt",
        "locked_test": output_dir / "gt_test_unknown.txt",
    }
    known_outputs = {
        name: output_dir / "gt_{}.txt".format(name)
        for name in ("train", "val_known", "test_known")
    }
    report_path = output_dir / "split_protocol.json"
    for path in list(outputs.values()) + list(known_outputs.values()) + [report_path]:
        if path.exists() and not args.overwrite:
            raise FileExistsError(
                "Output exists; use --overwrite after review: {}".format(path)
            )
    for split, path in outputs.items():
        write_manifest(path, rows_by_split[split], args.overwrite)
    known_counts = {}
    for name, path in known_outputs.items():
        rows = remap_known_manifest(
            project_root,
            known_manifest_dir / "gt_{}.txt".format(name),
        )
        write_manifest(path, rows, args.overwrite)
        known_counts[name] = len(rows)

    development_species = {
        item["species"]
        for item in species_report
        if item["partition"] == "development"
    }
    test_species = {
        item["species"]
        for item in species_report
        if item["partition"] == "locked_test"
    }
    if development_species & test_species:
        raise RuntimeError("Development and locked test species overlap")
    report = {
        "schema_version": 1,
        "split_unit": "species",
        "new_data_used_for_training": False,
        "phash_algorithm": "63-bit DCT pHash",
        "phash_threshold": args.phash_threshold,
        "reference_image_count_exact_hash_checked": reference_image_count,
        "cross_reference_exact_duplicate_count": 0,
        "known_manifests_use_raw_paths": True,
        "known_counts": known_counts,
        "development_manifest": str(outputs["development"].relative_to(project_root)),
        "locked_test_manifest": str(outputs["locked_test"].relative_to(project_root)),
        "development_count": len(rows_by_split["development"]),
        "locked_test_count": len(rows_by_split["locked_test"]),
        "development_species": sorted(development_species),
        "locked_test_species": sorted(test_species),
        "species": species_report,
    }
    with report_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
