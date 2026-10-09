# TaxoSieve · 浮游动物层级开集识别

**TaxoSieve: Support-Intervention and Staged Evidence Verification for Hierarchical Open-Set Zooplankton Recognition**

本版本以 `master@638853b47b0e7b8d2d7fe2ad387e030b255512c8` 为基点，适配新图像文件名、统一数据目录并删除对比实验。主实验仍是 MaPLe/CLIP reference → Known TRAIN 支持集干预 → D05 父/叶验证器 → DEV 先根后叶校准及条件 OOF → 冻结 TEST。模型、损失、训练预算、候选路径、阈值选择和评价定义没有修改。

## 数据和目录

所有图像和配套标签位于 `prepro/data/`，图像目录为单数 **`image`**。标签格式仍为 `相对图像路径,标签ID,行索引`，路径相对于配置中对应的 Known/Near/Extra 图像根目录。

| 路径 | 作用 |
|---|---|
| `prepro/data/image/` | Known、Near DEV/TEST、Extra DEV/TEST 五个图像池 |
| `prepro/data/gt_train.txt` | Known TRAIN |
| `prepro/data/gt_val_known.txt`、`gt_val_intra.txt`、`gt_val_extra.txt` | 开发集 |
| `prepro/data/gt_test_known.txt`、`gt_test_intra.txt`、`gt_test_extra.txt` | 默认主实验测试集 |
| `prepro/data/tree.npy`、`leaf_nodes.npy`、`known_leaf_order.txt` | 原 7 父类、23 已知叶类分类树 |
| `prepro/data/protocol.json`、`inventory.jsonl`、`split_statistics.json` | 内容身份、角色、划分和统计 |
| `prepro/build_taxosieve_dataset.py` | 唯一数据构建器 |
| `prepro/protocols/taxosieve_seed/` | 原始内容身份和现役清单顺序的冻结种子 |
| `run_taxosieve.py`、`taxosieve/` | 主实验入口、D05、校准、测试及严格收据校验 |
| `taxosafe_support/`、`models/`、`loader/` | 主实验实际使用的 reference 实现 |
| `tools/run_taxosieve_seeds.py` | 多 seed 与 batch 设置的唯一实现 |
| `reproducibility/` | 精简的历史回放证据及本轮验证报告 |

已删除对比实验实现、旧数据制作工具、v9/v11 配套数据和冗余文件名备份。主方法内用于保持 Known 表现的 reference/D05 比较与 OOF 审计仍属于算法本身，予以保留。

### 为什么补回的 84 张图没有直接加入默认 TEST

当前仓库有 **3,157 个图像文件、3,137 个唯一内容**。相对上一轮实际运行的 3,073 文件库存，补回了历史缺失的 Near TEST 22 张和 Extra TEST 62 张；原有内容没有缺失或修改。若直接加入 TEST，会改变原实验分母。为保持主实验，本版本保留这些图像并生成独立的 `gt_restored_test_intra.txt`、`gt_restored_test_extra.txt`，默认训练、校准、测试均不读取它们。

另有三个 OOD 来源在上传时跨 DEV/TEST 放置。按逐图内容身份核对后，`Alima_larva` 恢复到 OOD DEV，`Enteromorphaprolifra` 和 `Fish_larva` 恢复到 OOD TEST；保留用户的新文件名和全部原图字节。

| 默认划分 | 标签行数 | 唯一内容数 |
|---|---:|---:|
| Known TRAIN | 1610 | 1610 |
| Known DEV | 219 | 219 |
| Known TEST | 451 | 450 |
| Near DEV | 69 | 69 |
| Near TEST | 182 | 182 |
| Extra DEV | 88 | 88 |
| Extra TEST | 210 | 210 |
| Near 保留池（未启用） | 98 | 98 |
| Extra 保留池（未启用） | 127 | 127 |

默认 TEST 仍是 **843 条记录、842 张唯一图像**。主清单不仅保留相同划分和标签，还保留逐行图像内容顺序，避免新文件名排序改变 sampler、D05 折分或训练轨迹。Known 重复内容仍按 TEST > DEV > TRAIN 去重，TEST alias 仍按唯一内容计量。未启用的保留池及补回测试图不参与梯度或阈值拟合。

仓库已生成全部标签，正常运行无需重建。构建器及不可覆盖规则见 [prepro/README.md](prepro/README.md)。

## 运行主实验

保留原 Linux `ProTeCt` 环境（已知可运行组合为 Python 3.8、torch 1.12.1、torchvision 0.13.1、CUDA 构建 11.3）。`requirements.txt` 是依赖范围，不是升级要求。图像训练/提取需要可用 GPU 和原 CLIP ViT-B/16 权重缓存。

**将新分支解压到独立目录，不覆盖已完成或正在运行的旧 suite。** 数据路径和配置身份改变后，应使用新的 run 目录；旧 seed5 的复核继续使用原 suite 的 `runtime/run_taxosieve.py` 和原数据资产。

在新版本项目根目录执行：

```bash
conda activate ProTeCt
python run_taxosieve.py preflight
python -u run_taxosieve.py all --run-dir runs/taxosieve/main_clean_trial_1 --device cuda --save-scores
```

根入口仍固定 seed1 配方。`all` 已包含训练、校准、测试；不要再对同一目录重复启动。也可选择以下分步方式：

```bash
python -u run_taxosieve.py train --run-dir runs/taxosieve/main_clean_trial_1 --device cuda
python -u run_taxosieve.py calibrate --run-dir runs/taxosieve/main_clean_trial_1 --save-scores
python -u run_taxosieve.py test --run-dir runs/taxosieve/main_clean_trial_1 --save-scores
```

`--device` 不用于 calibrate/test。`--resume` 只验证并复用已完成阶段，不是任意优化器断点恢复。

多 seed 仍使用独立代码快照，按“全部 TRAIN/DEV → DEV-OOF 冻结选择 → 全部 TEST”顺序执行：

```bash
CUDA_VISIBLE_DEVICES=0 python -u run_taxosieve_seeds.py --seeds 2 3 4 5
CUDA_VISIBLE_DEVICES=0 python -u run_taxosieve_seeds_batch.py --seeds 8 18 28 38 48 --train-batch-size 24
```

两个旧入口均调用同一实现。batch12 为 240 批/轮，batch24 为 120 批/轮，均为每轮抽样 2880 次；优化器更新次数不同，不能把不同 batch 的变化全部归因于 seed。只准备可加 `--prepare-only`。详见 [多 seed 运行说明](docs/taxosieve_seeds.md)。

## 输出、评价及验证边界

`test/completed.json` 含总体和逐物种/来源结果，`test/predictions.jsonl` 含最终输出和 `evaluation_weight`。Near 正确必须回退到正确父类，ROOT 不算 Near 正确。原四项门槛保持：Known >90%、Near ≥85%、Extra ROOT >90%、叶接收精确率 >90%。本次重构没有声称提升这些指标。

```bash
python -m unittest discover -s tests -v
python run_taxosieve.py preflight
```

验证覆盖真实图像身份、划分/顺序、来源签名、D05 优化器、分阶段校准、OOF、TEST、缓存与收据防篡改。CPU 和数据验证不能代替目标 GPU 上的全量训练结果；本轮没有重新训练并证明 seed5 指标逐位复现。

`inspect`、同 run 的 `replay --scores ... --router ...` 保留。历史 `import-d05` 的桥接接口也保留，但已不内置整套旧对比实现：只有显式设置 `H02_ORIGINAL_SOURCE` 指向完整原始快照，且代码、配置、数据和所有收据均通过原有严格校验时才能导入。新数据配置与旧模型不符时仍拒绝导入。默认主实验完全不依赖这个外部归档。

历史数值等价测试可用 `TAXOSIEVE_ORIGINAL_SOURCE` 指定外部原始快照；默认测试明确跳过这些可选归档测试，其余主流程回归继续执行。历史黄金回放说明见 [reproducibility/README.md](reproducibility/README.md)。
