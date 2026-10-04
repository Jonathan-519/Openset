# 冻结 reference 基线的细粒度拒识精修

> 2026-10-03 实测更新：本入口 TEST known 由 94.00% 降为 93.11%，near 由 60.78% 升为 62.75%，extra 保持 80.88%，未通过全部验收。代码保留用于复现与消融；后续独立实验使用 [层级相对距离验证](TAXOSAFE_GEOMETRY.md)，直接复用 reference 源实验。

本方案直接使用已完成的 **reference v3** 实验，保留其编码器、类别排序、支持库、父级成员性分数以及父级阈值，只训练新的叶级重构验证器。它不需要重新训练 MaPLe，不接受 relation v4 或旧 DCBS/OE 检查点。原 `train_taxosafe_new.py`、`calibrate_taxosafe_new.py`、`test_taxosafe_new.py` 的行为与配置保持不变。

算法依据、适配细节和局限见 [研究说明](TAXOSAFE_REFERENCE_REFINE_RESEARCH.md)。默认使用冻结的全局 fine 特征，不使用本次实验中高度重复的四个 learned local token。每类采用无 bias 的 `512 → 16 → 512` 重构模块和 tanh；默认 23 类共新增 376,833 个可训练参数（含一个正分类尺度），旧模型参数全部冻结。

## 适用范围与验收含义

上一版 TEST 为 known 423/450、near 124/204、extra 220/272、开放世界叶精度 423/509。这些是**已有基线结果，不是新方法结果**。

固定父级路由后，域外根拒识结果保持基线，不能由本轮叶级精修提高到 90%。当前 reference DEV 中能通过父门且父候选正确的 near 只有 52/69，故本阶段的近域回退上限为 75.36%，低于既定 85% 门槛。报告会按实际数据重算此上限，四项原验收标准不会被放宽。

本轮用于验证一个独立问题：在保留较好 reference 分类和父路由的条件下，语义重构是否能减少近域未知的叶级误接纳。校准保护的是 **DEV 聚合计数**：known 正确数不低于基线且通过 >90%，near 正确数不低于基线，开放世界叶精度不低于基线。它不能保证 TEST 已知准确率仍为 94%，也不保证每个物种均改善。

如果没有满足保护条件且严格改善 near 或叶精度的阈值组合，`status` 明确写为 `baseline_fallback`，禁用新重构门并恢复原叶阈值。这种回退对任意后续样本都使用原路由，不只是对 DEV 中的最小分数有效。

## 更新服务器代码

服务器目录 `/home/ubuntu/hdd/data/qz/Openset` 没有 `.git` 时，下载 [new 分支完整源码 ZIP](https://github.com/Jonathan-519/Openset/archive/refs/heads/new.zip)，合并更新代码即可。保留 `prepro/raw/`、`prepro/data/`、`runs/`、权重与缓存；不要删除已有 reference 实验，也不要覆盖已有运行目录。

先停止使用项目源码的训练进程，并保留旧源码备份。可沿用 [reference 文档的代码合并脚本](TAXOSAFE_REFERENCE.md#2-更新服务器代码与检查)：该脚本跳过数据、运行结果和权重，备份有变动的代码文件，不要求项目是 Git 仓库。

更新后检查本轮清单和专项测试：

```bash
conda activate ProTeCt
cd /home/ubuntu/hdd/data/qz/Openset
sha256sum -c SUPPORT_REFERENCE_REFINE_FILES.sha256
python tools/run_unit_tests.py --pattern 'test_taxosafe_refine*.py' --quiet
```

服务器还可直接验证新模块的 CUDA 前向与反向；这项测试不读取图片或下载 CLIP：

```bash
python -m unittest tests.test_taxosafe_refine_reconstruction.ReconstructionTest.test_cuda_cached_fit_and_scores -v
```

应显示 `OK`；`skipped` 代表该环境没有实际运行 CUDA 测试。

不再用旧版本的整包 SHA 清单检验新版本工具文件；它们描述的是历史快照。当前没有在助手环境完成真实图像 GPU 实验，软件测试通过不代表识别门禁通过。

## 检查源实验

以下命令假设较优 reference 实验仍位于：

```bash
base=runs/taxosafe_new/reference/trial_1
```

需要完整的训练和校准产物，包括 `training/best.pth`、`training/support.pth`、配置/输入/完成记录，以及 `calibration/router.json`、完成记录与 `development_scores.jsonl`。文本 review 压缩包不包含权重，不能代替源实验。新流程不改写或复制旧模型权重，三个阶段都会验证其哈希，因此运行期间必须保留源目录不变。

允许的源代码快照仅为已审查的 `54dff0e` 和 `4394be5` 中的 **reference 模式**；其他配置、层级、数据准备审计与产物哈希仍须一致。这是独立入口中的显式历史版本导入，不会关闭原训练流水线的签名检查。若验证失败，应检查源实验或代码快照，不要删除校验代码。

## 执行拟合、校准和测试

使用新的独立输出目录。`fit` 只提取一次 TRAIN 特征并训练轻量模块，**不会重训 backbone/prompt**。TRAIN 内部按内容哈希分层留出图片，只用于新模块早停；这不等同于编码器未见物种的严格留类实验。

```bash
conda activate ProTeCt
cd /home/ubuntu/hdd/data/qz/Openset

cfg=configs/Zooplankton_Taxonomic_Tree/TaxoSafe_reference_refine.yml
base=runs/taxosafe_new/reference/trial_1
run_dir=runs/taxosafe_new/reference_refine/trial_1

python refine_taxosafe_reference.py fit --config "$cfg" --reference-run-dir "$base" --run-dir "$run_dir" --preflight &&
python -u refine_taxosafe_reference.py fit --config "$cfg" --reference-run-dir "$base" --run-dir "$run_dir" &&
python -u refine_taxosafe_reference.py calibrate --config "$cfg" --reference-run-dir "$base" --run-dir "$run_dir" &&
python -u refine_taxosafe_reference.py test --config "$cfg" --reference-run-dir "$base" --run-dir "$run_dir"
```

默认 seed=1、CUDA。真实 CLIP 权重与图像使用服务器原有环境。每个阶段完成后都写入独立收据；已完成目录不允许覆盖。若改变配置，请使用另一个 `run_dir`。

校准阶段会重新计算基线 DEV 证据并与原记录对照，验证类别/父层行为一致后才拟合叶阈值。TEST 只读取已经冻结的重构器和路由，不重新拟合。

## 打包结果

原打包工具现在也支持新目录：

```bash
python tools/pack_taxosafe_new_review.py --run-dir runs/taxosafe_new/reference_refine/trial_1
```

它只打包选定运行目录内的文本诊断，不跟随源目录引用，不包含图片、特征缓存或权重。把输出的 `.tar.gz` 发回即可比较基线与精修结果。

分析时首先查看 `calibration/validation_report.json` 的 `status`、`preservation_audit`、`parent_routing_near_upper_bound` 和来源留出结果，再查看 TEST 四项门禁、逐物种结果以及与基线对照。新组合分数是两个阈值余量的最小值，单位不同；其 AUROC 只描述该固定组合分数，不能当成单独重构分数的 AUROC。

## 后续实验的边界

本轮首先运行默认全局 fine 版本。`features: raw_spatial` 与 `score_mode: cssr` 是显式可选消融，不应同时启用后把变化归因于单个因素；空间模式的缓存和显存成本明显更高，应降低新模块 batch size。当前默认相对 L1 误差是对归一化 CLIP 特征的适配，不是完整 CSSR 复现。

若要进一步突破父门导致的近域上限并提高域外拒识，需要独立的父级证据实验。严格留物种验证必须连 reference 编码器也在该折排除被留类别，不能只删除支持样本或重构器类别后声称未知泛化。新阶段不加入真实 DEV/TEST 未知图片的梯度，不改变固定数据划分，不使用 TEST 选择方案。
