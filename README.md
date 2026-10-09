# TaxoSieve · 浮游动物层级开集识别

**TaxoSieve: Support-Intervention and Staged Evidence Verification for Hierarchical Open-Set Zooplankton Recognition**

中文名称：**基于支持集干预与分阶段证据验证的浮游动物层级开集识别**。

TaxoSieve 将原 H02_d05_staged 主线统一为可维护的实验名称：MaPLe/CLIP reference 训练 → Known TRAIN 支持集干预 → D05 父/叶验证器 → DEV 先根后叶校准 → 冻结 TEST。名称突出方法机制；本次是数据与工程重构，没有新增算法或证明性能提升。原 reference 候选路径、八维证据、损失、训练预算、阈值选择与评价门槛保持不变。

## 1. 当前数据与目录

图像统一存放在 `prepro/data/images/`，当前数据版本为 `prepro/data/taxosieve_v1/`。`prepro/raw` 只是供旧工具读取的相对符号链接，没有第二份图像。原图字节未修改、没有删除现存图像。

本次从仓库基点 `d7e2f472e1564873fd0a93bd31bf8428c052ab3b` 的实际图像库存重建。它有 3,073 个图像文件，比历史协议少 84 个 TEST 图像。新数据版本明确记录这一变化，不能声称恢复了原 TEST，也不能将原 450/204/272 分母下的性能数字套用到新版本。新数据统计、逐物种数量和缺失清单见数据目录中的 `split_statistics.json`、`inventory.jsonl`、`reconciliation.json`。

| 路径 | 用途 |
|---|---|
| `run_taxosieve.py`、`taxosieve/` | 主入口、验证器、分阶段校准和阶段收据 |
| `configs/taxosieve.yml`、`configs/taxosieve_reference.yml` | 固定实验配置、reference 模型与新数据路径 |
| `prepro/build_taxosieve_dataset.py` | 当前唯一推荐的数据集构建器 |
| `prepro/data/images/` | Known、Near DEV/TEST、Extra DEV/TEST 五个图像池 |
| `prepro/data/taxosieve_v1/` | 当前清单、分类树、内容身份、去重与版本审计 |
| `prepro/protocols/taxosieve_seed/` | 不可变的历史分组种子，用于保留图像归属和角色 |
| `taxosafe_support/`、`models/`、`loader/` | 主线实际依赖的 reference 内核 |
| `prepro/legacy_tools/` | 历史数据制作工具；不用于覆盖当前数据版本 |
| `comparison_experiments/` | 其他方法及其历史依赖，不进入主训练流程 |
| `reproducibility/` | 历史证明与本次工程验证，明确区分新旧数据 |

`prepro/data/Zooplankton_TT_v9_rebuild`、`Zooplankton_TT_v11_dcbs` 是历史清单快照，仅供溯源及旧工具使用；当前模型不读取它们。签名绑定的归档模型、tokenizer 和来源证明不能因名称相似而删除。

## 2. 数据重建规则

- 分类树保留原 **7 父类、23 已知叶类**的 ID、顺序与 NPY 字节。
- 已有图像以内容身份继承原划分；改名不导致重新随机划分。新增图像采用固定种子与内容哈希规则，规则写入 `protocol.json`。
- Known 重复内容采用 TEST → DEV → TRAIN 优先级；训练和校准只保留独立内容，TEST 保留 alias 并由模型按唯一 SHA256 计量。
- 未知来源的 DEV/TEST 物种角色隔离；既有 Near/OE 保留池继续保留，**不参与梯度训练**。
- 所有当前图像都必须被清单或审计解释。损坏图片、未知目录、类别冲突和跨角色内容冲突会失败，不静默跳过。
- 已存在且内容不同的数据输出不能覆盖。数据变化后生成新版本，并更新配置中的数据路径；已完成 run 的哈希或收据不得手工修改。

构建器用法见 [prepro/README.md](prepro/README.md)。仓库已经包含本次生成的清单；正常训练无需重复构建。

## 3. 环境与运行

用户已有服务器环境：Linux，`ProTeCt`，项目目录 `/home/ubuntu/hdd/data/qz/Openset`。保留可用的 torch/torchvision/CUDA 组合；`requirements.txt` 是源码依赖清单，不是升级指令或历史精确锁文件。图像训练与提取需要 CUDA，且首次初始化需要原 CLIP ViT-B/16 预训练权重或可用的下载缓存。

```bash
conda activate ProTeCt
cd /home/ubuntu/hdd/data/qz/Openset
python run_taxosieve.py preflight
python -u run_taxosieve.py all --run-dir runs/taxosieve/dataset_v1_trial_1 --device cuda --save-scores
```

使用一个**不存在的新 run 目录**。旧 H02 run 属于旧代码与数据协议，不能直接续跑。`preflight --metadata-only` 仅检查元数据，不能替代完整图像检查。

也可以逐阶段执行，不能和上面的 `all` 在同一个目录重复启动：

```bash
python -u run_taxosieve.py train --run-dir runs/taxosieve/dataset_v1_trial_1 --device cuda
python -u run_taxosieve.py calibrate --run-dir runs/taxosieve/dataset_v1_trial_1 --save-scores
python -u run_taxosieve.py test --run-dir runs/taxosieve/dataset_v1_trial_1 --save-scores
```

`--device` 用于 train/all/import-d05；calibrate/test 从 run 读取。`--resume` 只验证并复用已完成阶段，不代表任意优化器断点恢复。首次没有保存完整 scores 的已完成阶段不会因重复命令自动补写它们。

## 4. 输出与评价

正常输出依次写入 `reference/`、`training/`、`cache/`、`calibration/`、`test/`。`test/completed.json` 含总体及逐已知物种、近域未知物种、域外来源指标；`test/predictions.jsonl` 含逐图终端和计数权重。原始分数是连续证据，不等于概率。

| 类型 | 正确输出 | 门槛 |
|---|---|---|
| Known | 正确物种叶节点 | >90% |
| Near | 正确父节点 | ≥85% |
| Extra | ROOT / global_unknown | >90% |
| 全部接纳叶输出 | 正确 Known 叶数 / 全部叶输出数 | >90% |

父类不统一称为“属”。Near 回退到错误父类仍算错；TEST 中同内容 alias 不重复计数。条件 OOF 审计与模型是否通过四项指标是不同结论。

历史 H02 在旧 TEST 的 K/N/E/Leaf PPV 为 90.22%/74.02%/60.66%/86.02%，未联合达标。这些是历史结果，**本次未在新数据上训练或报告新性能**。

## 5. 验证与历史功能

```bash
python -m unittest discover -s tests -v
python run_taxosieve.py --help
python run_taxosieve.py preflight
```

本次验证覆盖真实图像完整性、内容与角色隔离、数据构建异常、来源签名、reference/D05 数值等价性、阶段顺序和旧工具迁移。完整 GPU 训练仍需在用户服务器运行。

`inspect`、`replay` 及来源导入的严格校验代码保留；来源导入要求配置、数据、代码身份全部一致。当前 TaxoSieve 新数据配置与历史 H02/D05 配置不同，因此不能直接使用 `import-d05` 或 `--reference-run-dir` 复用旧数据模型。需要原协议回放时使用归档环境及原配套资产。旧模型不能靠修改收据移植到新数据。`replay` 必须显式提供与分数属于同一 run 的 `--router runs/taxosieve/<run>/calibration/router.json`，不会默认使用历史阈值。历史黄金回放入口为 `tools/verify_taxosieve_replay.py`，输入必须是经过哈希验证的原始黄金文件。其历史下载提交目前不可访问，不能把下载失败当作新模型训练失败，也不能替换成当前数据冒充黄金输入。
