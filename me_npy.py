from pathlib import Path

import numpy as np

from loader.treelibs import Tree


# ============================================================
# 1. 路径设置
# ============================================================

# me_npy.py 位于 ProTeCt-main 根目录
project_root = Path(__file__).resolve().parent

# 只包含17个已知物种的分类树目录
known_root = (
    project_root
    / "prepro"
    / "views"
    / "Zooplankton"
    / "fold1_known"
)

# tree.npy等文件保存位置
out_dir = (
    project_root
    / "prepro"
    / "data"
    / "Zooplankton_Taxonomic_Tree"
)

tree_file = out_dir / "tree.npy"
leaf_file = out_dir / "leaf_nodes.npy"


# ============================================================
# 2. 检查输入目录
# ============================================================

if not known_root.is_dir():
    raise FileNotFoundError(
        f"已知物种目录不存在：\n{known_root}\n"
        "请先检查 fold1_known 是否创建成功。"
    )

# np.save不会自动创建父目录，因此这里必须创建
out_dir.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print(f"项目根目录：{project_root}")
print(f"已知树目录：{known_root}")
print(f"输出目录：  {out_dir}")
print("=" * 70)


# ============================================================
# 3. 构建只包含已知物种的分类树
# ============================================================

tree = Tree(str(known_root))

# ============================================================
# 让 tree.npy 的叶节点编号与现有 known TXT 完全一致
# ============================================================

txt_leaf_order = [
    "Paracalanus_parvus",             # 0
    "Centropages_dorsispinatus",      # 1
    "Calanus_sinicus",                # 2
    "Acartia_hongi",                  # 3
    "Oithona_plumifera",              # 4
    "Eurytemora_pacifica",            # 5
    "Obelia_dichotoma",               # 6
    "Sugiura_chengshanense",          # 7
    "Clytia_folleata",                # 8
    "Muggiaea_atlantica",             # 9
    "Proboscidactyla_flavicirrata",   # 10
    "Evadne_tergestina",              # 11
    "Penilia_avirostris",             # 12
    "Sagitta",                         # 13
    "Euphausia_pacifica",             # 14
    "Oikopleura",                      # 15
    "Themisto_gracilipes",            # 16
]

actual_leaf_names = set(tree.leaf_nodes.values())
expected_leaf_names = set(txt_leaf_order)

if actual_leaf_names != expected_leaf_names:
    raise ValueError(
        "known tree 中的物种与 TXT 编号表不一致。\n"
        "缺少：{}\n"
        "多余：{}".format(
            sorted(expected_leaf_names - actual_leaf_names),
            sorted(actual_leaf_names - expected_leaf_names),
        )
    )

# 重新定义 leaf ID
tree.leaf_nodes = {
    leaf_id: leaf_name
    for leaf_id, leaf_name in enumerate(txt_leaf_order)
}

# leaf 顺序改变后，必须重新计算 leaf -> parent 映射
tree._gen_sublabels()


# ============================================================
# 4. 检查七个父类
# ============================================================

expected_parents = [
    "Amphipoda",
    "Appendiculata",
    "Cladocera",
    "Copepoda",
    "Euphausiacea",
    "Medusae",
    "Sagittoidea",
]

actual_parents = list(tree.root.children.values())

if actual_parents != expected_parents:
    raise ValueError(
        "父类编号顺序不正确。\n"
        "期望：{}\n"
        "实际：{}".format(
            expected_parents,
            actual_parents,
        )
    )
    
    
# ============================================================
# 5. 检查未知物种没有进入训练树
# ============================================================

held_out_species = {
    # validation intra-unknown
    "Centropages_tenuiremis",
    "Eirene",

    # test intra-unknown
    "Acartia_pacifica",
    "Oithona_similis",
    "Turritopsis_nutricula",
}

leaked_species = held_out_species.intersection(tree.nodes.keys())

if leaked_species:
    raise ValueError(
        "发现未知物种泄漏到训练分类树中：\n"
        f"{sorted(leaked_species)}\n"
        "请从 fold1_known 中删除这些目录或符号链接。"
    )


# ============================================================
# 6. 检查叶节点数量
# ============================================================

if len(tree.leaf_nodes) != 17:
    leaf_names = [
        tree.leaf_nodes[i]
        for i in range(len(tree.leaf_nodes))
    ]

    raise ValueError(
        f"叶节点数量应该是17，但实际是{len(tree.leaf_nodes)}。\n"
        f"当前叶节点：{leaf_names}"
    )


# ============================================================
# 7. 保存tree.npy
# ============================================================

np.save(str(tree_file), tree)


# ============================================================
# 8. 保存物种名称到叶标签的映射
# ============================================================

leaf_name_to_id = {
    name: leaf_id
    for leaf_id, name in tree.leaf_nodes.items()
}

np.save(str(leaf_file), leaf_name_to_id)


# ============================================================
# 9. 显示结果
# ============================================================

print("\n生成的分类树：")
tree.show()

print("\n父类编号：")
for parent_id, parent_name in enumerate(tree.root.children.values()):
    print(f"{parent_id:2d} -> {parent_name}")

print("\n叶节点编号：")
for leaf_name, leaf_id in sorted(
    leaf_name_to_id.items(),
    key=lambda item: item[1]
):
    parent_name = tree.nodes[leaf_name].parent

    print(
        f"{leaf_id:2d} -> "
        f"{parent_name}/{leaf_name}"
    )

print("\n统计信息：")
print(f"父类数量：{len(tree.root.children)}")
print(f"叶节点数量：{len(tree.leaf_nodes)}")
print(f"内部节点数量：{len(tree.intnl_nodes)}")
print(f"全部节点数量：{len(tree.nodes)}")

print("\n文件生成成功：")
print(tree_file)
print(leaf_file)