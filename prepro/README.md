# TaxoSieve 数据准备

当前主实验的数据集中放在 `prepro/data/`。图像目录为单数 `image/`；标签清单、分类树和审计文件直接位于 `data/`，不再使用 `images/` 或 `taxosieve_v1/` 子目录。

```text
prepro/
  build_taxosieve_dataset.py
  protocols/taxosieve_seed/     # 原始身份、类别和现役清单内容顺序的冻结依据
  data/
    image/
      Zooplankton_TT/
      Zooplankton_NearUnknown_Dev/
      Zooplankton_NearUnknown_Test/
      Zooplankton_OOD_Dev/
      Zooplankton_OOD_Test/
    gt_train.txt
    gt_train_reference.txt
    gt_val_known.txt
    gt_test_known.txt
    gt_val_intra.txt
    gt_test_intra.txt
    gt_val_extra.txt
    gt_test_extra.txt
    gt_reserved_near.txt
    gt_reserved_extra.txt
    gt_restored_test_intra.txt
    gt_restored_test_extra.txt
    tree.npy
    leaf_nodes.npy
    known_leaf_order.txt
    protocol.json
    species_roles.json
    inventory.jsonl
    split_statistics.json
    reconciliation.json
    known_deduplication.json
    dataset.sha256
```

## 主实验保持不变的范围

重命名后按图像内容继承原角色和标签，并保留现役各清单的**逐行图像内容顺序**。这既避免 TRAIN/DEV/TEST 重新分配，也避免固定 seed 的采样索引、episode 构造因文件名字典序改变而对应到不同图像。Known 的23个叶类、7个父类、分类树与原 ID 顺序保持原样。

| 清单 | 行数 | 独立内容数 | 用途 |
|---|---:|---:|---|
| `gt_train.txt` | 1610 | 1610 | 梯度训练及支持库 |
| `gt_train_reference.txt` | 1610 | 1610 | reference 使用，与 `gt_train.txt` 字节一致 |
| `gt_val_known.txt` | 219 | 219 | 模型选择与校准 |
| `gt_test_known.txt` | 451 | 450 | 冻结后测试，保留1个重复别名 |
| `gt_val_intra.txt` | 69 | 69 | 近域未知校准 |
| `gt_test_intra.txt` | 182 | 182 | 现役近域未知测试 |
| `gt_val_extra.txt` | 88 | 88 | 域外未知校准 |
| `gt_test_extra.txt` | 210 | 210 | 现役域外未知测试 |
| `gt_reserved_near.txt` | 98 | 98 | 未启用，不参与训练/校准 |
| `gt_reserved_extra.txt` | 127 | 127 | 未启用，不参与训练/校准 |

`gt_train_reference.txt` 是同一训练清单的 reference 输入别名，不是额外数据或对比实验。清单格式为 `相对图像路径,类别ID,从0起的行索引`；根目录由 `protocol.json` 与主配置指定。Known/Near 保留“父类/物种/图像”层级，Extra 保留“来源/图像”层级。

Known 的精确重复内容继续按 **TEST > DEV > TRAIN** 处理，19条重复拟合记录排除；TEST 的1个重复别名保留，评价按唯一图像身份处理。`Calanopia_thompsoni` 仍无独立验证图像，不能声称已消除这一历史限制。

## 当前库存中的84张恢复测试图像

当前仓库的3157个文件、3137种唯一图像内容与原始3157路径 seed 的 Git blob身份及字节长度多重集完全相同。相对上一轮实际使用的3073文件，恢复了历史缺失的84张 TEST 图像：Near 22张、Extra 62张；原现役图像没有缺失或内容替换。

为满足主实验不受影响的约束，这84张保留在 `image/`，并另外生成：

| 清单 | 数量 | 默认状态 |
|---|---:|---|
| `gt_restored_test_intra.txt` | 22 | inactive，未接入主配置 |
| `gt_restored_test_extra.txt` | 62 | inactive，未接入主配置 |

它们不进入训练、开发校准或默认主测试；因此主测试的唯一内容分母仍为 **Known450 / Near182 / Extra210**。把它们加入 TEST 会得到另一份评估协议，必须显式创建新版本、单独重评，不能与当前主实验结果混用。

本次还纠正了新目录中3个来源的 DEV/TEST 位置：`Alima_larva` 39张归回 OOD_Dev；`Enteromorphaprolifra` 17张与 `Fish_larva` 41张归回 OOD_Test。逐图内容身份已与原始冻结角色核对；图像字节和当前新文件名均未修改。具体记录见 `reconciliation.json` 的 `main_experiment_preserved.relocated_source_roots`。

## 构建与验证

从仓库根执行：

```bash
python prepro/build_taxosieve_dataset.py --dry-run
python prepro/build_taxosieve_dataset.py
python run_taxosieve.py preflight
(cd prepro/data && sha256sum -c dataset.sha256)
```

构建器会读取、计算哈希并完整解码每张图像，只做数据完整性审计，不提取模型特征或拟合阈值。已有相同输出可幂等核验；不同输出拒绝覆盖。默认 `data/` 允许已有 `image/`，只生成配套文件。

`protocols/taxosieve_seed/seed.json` 固定原始角色、标签和身份，`manifest_order.json` 固定现役每个 split 的 SHA256 与标签顺序，并列出允许保留的84张历史恢复测试图像。顺序文件本身由 seed 中的 SHA256 校验。缺少任何现役图像、未经批准的新增内容、类别冲突、损坏图像或越界符号链接都会报错，不能静默缩小测试集或重新分配。

`--project-root`、`--image-root`、`--seed-dir`、`--output` 可指定项目内路径，`--report <新文件.json>` 可输出独立报告。未来确需改变数据协议，应提供显式的新冻结 seed 与新输出版本，并同步主配置中的全部清单、树和 preparation audit，再用新 run 完整验证。只改变输出目录不会解除现役身份与顺序限制。通用构建器仅在 seed 不包含冻结顺序时使用内容哈希划分新图像，本仓库默认协议不采用该模式。

`inventory.jsonl` 记录每张图的 SHA256、原身份、当前角色、行索引和处理状态；恢复图像标为 `inactive_restored_test`。`split_statistics.json` 中现役/保留清单3054行、重复排除19行、恢复测试84行之和为3157，全部图像都有明确去向。`dataset.sha256` 校验全部生成资产，不包含体积较大的图像字节；图像字节由库存记录和完整预检核验。
