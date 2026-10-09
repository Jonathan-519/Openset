from pathlib import Path
import json
import math
import os
import random
import shutil
import sys

import numpy as np

# 让脚本优先导入 prepro/tree.py
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from loader.treelibs import Tree

RAW_ROOT = PROJECT_ROOT / "prepro" / "raw"
DATA_OUT = PROJECT_ROOT / "prepro" / "data" / "Zooplankton_TT_v9_rebuild"

KNOWN_ROOT = RAW_ROOT / "Zooplankton_TT"
NEAR_DEV_ROOT = RAW_ROOT / "Zooplankton_NearUnknown_Dev"
NEAR_TEST_ROOT = RAW_ROOT / "Zooplankton_NearUnknown_Test"

OOD_ALL_ROOT = RAW_ROOT / "Zooplankton_OOD"
OOD_DEV_ROOT = RAW_ROOT / "Zooplankton_OOD_Dev"
OOD_TEST_ROOT = RAW_ROOT / "Zooplankton_OOD_Test"

PROTOCOL_SEED = 2026

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

PARENT_ID = {
    "Amphipoda": 0,
    "Appendiculata": 1,
    "Cladocera": 2,
    "Copepoda": 3,
    "Euphausiacea": 4,
    "Medusae": 5,
    "Sagittoidea": 6,
}

LOCKED_NEAR_TEST = {
    "Copepoda": [
        "Corycaeus_affinis（测试）",
        "Oithona_plumifera（测试）",
        "Tortanus_derjugini（测试）",
    ],
    "Medusae": [
        "Gonionemus_vertens（测试）",
        "Turritopsis_nutricula（测试）",
    ],
}


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def sorted_dirs(path: Path):
    return sorted([p for p in path.iterdir() if p.is_dir()], key=lambda x: x.name)


def list_images(path: Path):
    return sorted(
        [
            p for p in path.rglob("*")
            if p.is_file() and p.suffix.lower() in IMG_EXTS
        ],
        key=lambda x: x.as_posix()
    )


def normalize_species_name(name: str):
    return name.replace("（测试）", "").replace("(测试)", "").strip()


def move_dir(src: Path, dst: Path):
    if not src.exists():
        raise FileNotFoundError(f"Missing source directory: {src}")
    if dst.exists():
        raise FileExistsError(f"Destination already exists: {dst}")
    ensure_dir(dst.parent)
    shutil.move(str(src), str(dst))


def write_manifest(manifest_path: Path, rows):
    ensure_dir(manifest_path.parent)
    with manifest_path.open("w", encoding="utf-8") as f:
        for idx, (rel_path, label) in enumerate(rows):
            f.write(f"{rel_path},{label},{idx}\n")


def write_json(path: Path, obj):
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def species_count(path: Path):
    if not path.exists():
        return 0
    total = 0
    for parent in sorted_dirs(path):
        total += len(sorted_dirs(parent))
    return total


def category_count(path: Path):
    if not path.exists():
        return 0
    return len(sorted_dirs(path))


def move_locked_near_test():
    """
    把 5 个 locked test species 从 KNOWN_ROOT 移到 NEAR_TEST_ROOT，
    并去掉目录名中的 （测试） 后缀。
    """
    print("=" * 80)
    print("[1/6] Moving locked near-unknown test species ...")
    print("=" * 80)

    for parent, species_list in LOCKED_NEAR_TEST.items():
        for species_name in species_list:
            src = KNOWN_ROOT / parent / species_name
            dst = NEAR_TEST_ROOT / parent / normalize_species_name(species_name)
            print(f"MOVE: {src} -> {dst}")
            move_dir(src, dst)


def choose_and_move_dev_near_unknown(seed=PROTOCOL_SEED):
    """
    从剩余的 Copepoda/Medusae 中各随机抽 2 个 species，
    移到 NEAR_DEV_ROOT。
    """
    print("=" * 80)
    print("[2/6] Selecting and moving near-unknown dev species ...")
    print("=" * 80)

    rng = random.Random(seed)

    result = {"Copepoda": [], "Medusae": []}

    for parent, n_pick in [("Copepoda", 2), ("Medusae", 2)]:
        parent_dir = KNOWN_ROOT / parent
        candidates = [d.name for d in sorted_dirs(parent_dir)]
        if len(candidates) < n_pick:
            raise RuntimeError(
                f"Not enough candidates under {parent}: have {len(candidates)}, need {n_pick}"
            )

        picked = sorted(rng.sample(candidates, n_pick))
        result[parent] = picked

        for species_name in picked:
            src = parent_dir / species_name
            dst = NEAR_DEV_ROOT / parent / species_name
            print(f"MOVE: {src} -> {dst}")
            move_dir(src, dst)

    return result


def split_ood(seed=PROTOCOL_SEED + 1):
    """
    从 Zooplankton_OOD 的 10 个类别中随机抽 4 类作为 OOD_Dev，
    剩余 6 类作为 OOD_Test。
    """
    print("=" * 80)
    print("[3/6] Splitting OOD categories into Dev/Test ...")
    print("=" * 80)

    rng = random.Random(seed)

    all_categories = [d.name for d in sorted_dirs(OOD_ALL_ROOT)]
    if len(all_categories) != 10:
        print(f"[WARN] Expected 10 OOD categories, but found {len(all_categories)}")

    dev_classes = sorted(rng.sample(all_categories, 4))
    test_classes = sorted([x for x in all_categories if x not in dev_classes])

    for cls_name in dev_classes:
        src = OOD_ALL_ROOT / cls_name
        dst = OOD_DEV_ROOT / cls_name
        print(f"MOVE: {src} -> {dst}")
        move_dir(src, dst)

    for cls_name in test_classes:
        src = OOD_ALL_ROOT / cls_name
        dst = OOD_TEST_ROOT / cls_name
        print(f"MOVE: {src} -> {dst}")
        move_dir(src, dst)

    # 原目录如果空了，可以删除
    try:
        if OOD_ALL_ROOT.exists() and not any(OOD_ALL_ROOT.iterdir()):
            OOD_ALL_ROOT.rmdir()
    except Exception:
        pass

    return dev_classes, test_classes


def build_tree_and_leaf_nodes():
    """
    用当前 KNOWN_ROOT（只包含 Known species）生成
    tree.npy 和 leaf_nodes.npy。
    """
    print("=" * 80)
    print("[4/6] Building tree.npy and leaf_nodes.npy ...")
    print("=" * 80)

    ensure_dir(DATA_OUT)

    tree = Tree(str(KNOWN_ROOT))

    tree_path = DATA_OUT / "tree.npy"
    np.save(tree_path, tree, allow_pickle=True)

    # 你的 Tree.leaf_nodes 本身就是：
    # {leaf_id: species_name}
    #
    # manifest 生成时需要反过来的：
    # {species_name: leaf_id}
    leaf_map = {
        species_name: int(leaf_id)
        for leaf_id, species_name in tree.leaf_nodes.items()
    }

    leaf_nodes_path = DATA_OUT / "leaf_nodes.npy"
    np.save(leaf_nodes_path, leaf_map, allow_pickle=True)

    # 额外生成可读版本，便于人工检查
    leaf_order_path = DATA_OUT / "known_leaf_order.txt"
    with leaf_order_path.open("w", encoding="utf-8") as f:
        for leaf_id, species_name in tree.leaf_nodes.items():
            f.write(f"{leaf_id}\t{species_name}\n")

    print(f"Number of known leaves: {len(leaf_map)}")
    print(f"Saved: {tree_path}")
    print(f"Saved: {leaf_nodes_path}")
    print(f"Saved: {leaf_order_path}")

    return leaf_map


def split_known_species_images(leaf_map, seed=PROTOCOL_SEED):
    """
    对 23 个 Known species 按 7:1:2 生成:
      gt_train_known.txt
      gt_train_reference.txt
      gt_val_known.txt
      gt_test_known.txt
    """
    print("=" * 80)
    print("[5/6] Building known train/val/test manifests ...")
    print("=" * 80)

    rng = random.Random(seed + 100)

    train_rows = []
    train_ref_rows = []
    val_rows = []
    test_rows = []

    split_stats = {
        "known_species": {},
        "near_unknown_dev": {},
        "near_unknown_test": {},
        "ood_dev": {},
        "ood_test": {},
    }

    for parent_dir in sorted_dirs(KNOWN_ROOT):
        for species_dir in sorted_dirs(parent_dir):
            species_name = species_dir.name
            label = leaf_map[species_name]

            imgs = list_images(species_dir)
            n = len(imgs)
            if n < 4:
                raise RuntimeError(
                    f"Known species {parent_dir.name}/{species_name} has too few images: {n}"
                )

            imgs = imgs[:]  # copy
            rng.shuffle(imgs)

            n_test = max(2, int(math.floor(n * 0.2)))
            n_val = max(1, int(math.floor(n * 0.1)))
            n_train = n - n_test - n_val

            if n_train < 1:
                raise RuntimeError(
                    f"Known species {parent_dir.name}/{species_name} has invalid split: "
                    f"n={n}, train={n_train}, val={n_val}, test={n_test}"
                )

            train_imgs = imgs[:n_train]
            val_imgs = imgs[n_train:n_train + n_val]
            test_imgs = imgs[n_train + n_val:]

            for p in train_imgs:
                rel = p.relative_to(KNOWN_ROOT).as_posix()
                train_rows.append((rel, label))
                train_ref_rows.append((rel, label))

            for p in val_imgs:
                rel = p.relative_to(KNOWN_ROOT).as_posix()
                val_rows.append((rel, label))

            for p in test_imgs:
                rel = p.relative_to(KNOWN_ROOT).as_posix()
                test_rows.append((rel, label))

            split_stats["known_species"][f"{parent_dir.name}/{species_name}"] = {
                "label": label,
                "total": n,
                "train": len(train_imgs),
                "val": len(val_imgs),
                "test": len(test_imgs),
            }

    write_manifest(DATA_OUT / "gt_train_known.txt", train_rows)
    write_manifest(DATA_OUT / "gt_train_reference.txt", train_ref_rows)
    write_manifest(DATA_OUT / "gt_val_known.txt", val_rows)
    write_manifest(DATA_OUT / "gt_test_known.txt", test_rows)

    return split_stats


def build_intra_manifest(root: Path, out_name: str, split_stats):
    """
    Near-Unknown:
      label 不是 leaf ID，而是 parent ID
      Copepoda -> 3
      Medusae  -> 5
    """
    rows = []
    for parent_dir in sorted_dirs(root):
        parent_name = parent_dir.name
        pid = PARENT_ID[parent_name]
        for species_dir in sorted_dirs(parent_dir):
            imgs = list_images(species_dir)
            key = f"{parent_name}/{species_dir.name}"
            split_stats[key] = {
                "parent_id": pid,
                "images": len(imgs),
            }
            for img in imgs:
                rel = img.relative_to(root).as_posix()
                rows.append((rel, pid))

    write_manifest(DATA_OUT / out_name, rows)


def build_extra_manifest(root: Path, out_name: str, split_stats):
    """
    OOD:
      label 固定为 -1
    """
    rows = []
    for cls_dir in sorted_dirs(root):
        imgs = list_images(cls_dir)
        split_stats[cls_dir.name] = {"images": len(imgs)}
        for img in imgs:
            rel = img.relative_to(root).as_posix()
            rows.append((rel, -1))

    write_manifest(DATA_OUT / out_name, rows)


def build_unknown_manifests(split_stats):
    """
    生成:
      gt_val_intra.txt
      gt_test_intra.txt
      gt_val_extra.txt
      gt_test_extra.txt
    """
    print("=" * 80)
    print("[6/6] Building unknown manifests ...")
    print("=" * 80)

    build_intra_manifest(
        NEAR_DEV_ROOT,
        "gt_val_intra.txt",
        split_stats["near_unknown_dev"],
    )
    build_intra_manifest(
        NEAR_TEST_ROOT,
        "gt_test_intra.txt",
        split_stats["near_unknown_test"],
    )
    build_extra_manifest(
        OOD_DEV_ROOT,
        "gt_val_extra.txt",
        split_stats["ood_dev"],
    )
    build_extra_manifest(
        OOD_TEST_ROOT,
        "gt_test_extra.txt",
        split_stats["ood_test"],
    )


def build_role_json(dev_near, dev_ood, test_ood):
    roles = {
        "known_root": str(KNOWN_ROOT),
        "near_dev_root": str(NEAR_DEV_ROOT),
        "near_test_root": str(NEAR_TEST_ROOT),
        "ood_dev_root": str(OOD_DEV_ROOT),
        "ood_test_root": str(OOD_TEST_ROOT),

        "locked_near_test": {
            "Copepoda": [
                normalize_species_name(x) for x in LOCKED_NEAR_TEST["Copepoda"]
            ],
            "Medusae": [
                normalize_species_name(x) for x in LOCKED_NEAR_TEST["Medusae"]
            ],
        },
        "dev_near_unknown": dev_near,
        "dev_ood_classes": dev_ood,
        "test_ood_classes": test_ood,
    }
    write_json(DATA_OUT / "species_roles.json", roles)
    return roles


def build_protocol_json(dev_near, dev_ood, test_ood):
    protocol = {
        "protocol_version": "taxosafe_v9_rebuild_v1",
        "protocol_seed": PROTOCOL_SEED,
        "known_split": {
            "train": 0.7,
            "val": 0.1,
            "test": 0.2,
            "min_val": 1,
            "min_test": 2,
        },
        "near_unknown": {
            "locked_test_species": {
                "Copepoda": [
                    normalize_species_name(x) for x in LOCKED_NEAR_TEST["Copepoda"]
                ],
                "Medusae": [
                    normalize_species_name(x) for x in LOCKED_NEAR_TEST["Medusae"]
                ],
            },
            "dev_species": dev_near,
            "policy": "copepoda_2 + medusae_2, random once then frozen",
        },
        "global_ood": {
            "dev_classes": dev_ood,
            "test_classes": test_ood,
            "policy": "4 dev + 6 test, category-disjoint, random once then frozen",
            "oe_training": False,
        },
        "roots": {
            "known_root": str(KNOWN_ROOT),
            "near_dev_root": str(NEAR_DEV_ROOT),
            "near_test_root": str(NEAR_TEST_ROOT),
            "ood_dev_root": str(OOD_DEV_ROOT),
            "ood_test_root": str(OOD_TEST_ROOT),
            "data_out": str(DATA_OUT),
        },
    }
    write_json(DATA_OUT / "protocol.json", protocol)
    return protocol


def save_split_statistics(split_stats):
    write_json(DATA_OUT / "split_statistics.json", split_stats)


def sanity_check():
    known_n = species_count(KNOWN_ROOT)
    near_dev_n = species_count(NEAR_DEV_ROOT)
    near_test_n = species_count(NEAR_TEST_ROOT)

    ood_dev_n = category_count(OOD_DEV_ROOT)
    ood_test_n = category_count(OOD_TEST_ROOT)

    print("=" * 80)
    print("SANITY CHECK")
    print("=" * 80)
    print(f"Known species count          : {known_n}")
    print(f"NearUnknown Dev species count: {near_dev_n}")
    print(f"NearUnknown Test species count: {near_test_n}")
    print(f"OOD Dev class count          : {ood_dev_n}")
    print(f"OOD Test class count         : {ood_test_n}")

    assert known_n == 23, f"Expected 23 known species, got {known_n}"
    assert near_dev_n == 4, f"Expected 4 dev near-unknown species, got {near_dev_n}"
    assert near_test_n == 5, f"Expected 5 test near-unknown species, got {near_test_n}"
    assert ood_dev_n == 4, f"Expected 4 dev OOD classes, got {ood_dev_n}"
    assert ood_test_n == 6, f"Expected 6 test OOD classes, got {ood_test_n}"


def main():
    # 防止重复运行覆盖
    if DATA_OUT.exists():
        raise RuntimeError(
            f"{DATA_OUT} already exists. Please move/delete it first if you really want to rebuild."
        )
    if NEAR_DEV_ROOT.exists() or NEAR_TEST_ROOT.exists() or OOD_DEV_ROOT.exists() or OOD_TEST_ROOT.exists():
        raise RuntimeError(
            "Near/OOD split roots already exist. "
            "This script is intended for a clean one-shot rebuild. "
            "Restore from backup first if needed."
        )

    ensure_dir(DATA_OUT)
    ensure_dir(NEAR_DEV_ROOT)
    ensure_dir(NEAR_TEST_ROOT)
    ensure_dir(OOD_DEV_ROOT)
    ensure_dir(OOD_TEST_ROOT)

    move_locked_near_test()
    dev_near = choose_and_move_dev_near_unknown(seed=PROTOCOL_SEED)
    dev_ood, test_ood = split_ood(seed=PROTOCOL_SEED + 1)

    sanity_check()

    leaf_map = build_tree_and_leaf_nodes()
    split_stats = split_known_species_images(leaf_map, seed=PROTOCOL_SEED)
    build_unknown_manifests(split_stats)

    build_role_json(dev_near, dev_ood, test_ood)
    build_protocol_json(dev_near, dev_ood, test_ood)
    save_split_statistics(split_stats)

    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Generated files under: {DATA_OUT}")


if __name__ == "__main__":
    main()