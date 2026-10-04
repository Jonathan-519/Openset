# 从 reference 基线出发的六组顺序实验

本版本在 `new` 分支新增独立 `taxosafe_sweep`，不会把已有冻结统计拟合称为网络训练，也不会把校准失败自动替换成 reference 后再报告为新方法。E01–E04 做真实梯度更新，E05 做固定权重插值，E00 是原始已训练 reference 对照。

## 基线与分支选择

核对时远端有 `master`、`taxosafe-v10-perf`、`taxosafe-v11-dcbs` 和 `new`。继续使用 `new` 的数据审计、CUDA 整数计数修复、reference 支持库及校准实现；回退的是初始化权重，不是将整个仓库换成旧分支。

这里“原始模型”明确指已经恢复训练并完成校准的 `support_reference_v3`：

```text
runs/taxosafe_new/reference/trial_1_retrain_20261003_185220
```

最近服务器结果包记录的 checkpoint SHA256 是 `e4c39c378cb3685533ebc76658338b8aed09cc52662d8967cca113d14a279775`，支持库 SHA256 是 `bdb8ee6213d89260d0443d23ae48afc0a679049f5905c7eff174a8c759f1d89f`。实际入口仍使用原 importer 完整验证源配置、权重、支持库、router、收据、数据签名和来源；不会仅凭目录名信任文件，也不会伪造已删除的权重。

这不是从 CLIP 预训练权重重新训练 reference，也不是载入另一个分支的未知 checkpoint。若该目录里的 `training/best.pth` 或 `training/support.pth` 确实又被删除，必须先恢复完整匹配的源运行；文本 review 包和源码包无法还原模型权重。

## 实验矩阵

| 组 | 更新内容 | 用途 |
|---|---|---|
| E00_reference | 完全冻结源权重、支持库和原阈值 | 同环境重新测量对照 |
| E01_heads | 仅父/叶适配层与证据头；原训练目标 | 测量小学习率继续训练的效果 |
| E02_heads_anchor | E01 加源模型输出蒸馏与表征约束 | 比较基线约束是否减少遗忘 |
| E03_prompt_anchor | E02 再开放 MaPLe 提示参数 | 比较提示参数参与微调的效果 |
| E04_hierarchy_anchor | E03 增强已有层级损失权重 | 比较父级泛化与兄弟叶区分监督 |
| E05_hierarchy_blend | E00 与 E04 权重各 50%，重建支持库 | 比较固定插值能否缓和表示漂移 |

E01–E04 分别从同一个 E00 开始，互不串接；E05 唯一依赖 E04。CLIP 主干始终冻结，开放提示参数不等于全量微调 ViT。E02–E04 的教师是当前任务已训练 reference，不是原始零样本 CLIP。

四个训练组使用共同预算：最多 20 epoch、每 epoch 240 batch、batch 12，SGD 提示学习率 0.0005、适配层/证据头学习率 0.001、至少 3 epoch 后允许按 patience 6 提前停止。沿用原始数据视图、支持干预、查询内容排除与 TRAIN-only 监督。最佳更新 checkpoint 由 known DEV 的候选叶准确率、再按结构 NLL 选取；不使用真实 near/extra 梯度或 TEST 选 epoch。

原训练目标中的缓存叶 logits anchor 保留；新增蒸馏使用同一增强图像上的冻结教师父/叶文本 logits，温度 2；新增特征约束约束父/叶表征。E04 将既有 `parent_cross_species` 和 `leaf_sibling` 从 0.1 调至 0.3，`pair_parent` 和 `pair_leaf` 从 0.25 调至 0.5。这是预先固定的损失权重消融，不是声称发明了这些损失。

**主结果使用 epoch≥1 的最佳更新权重。** 即使更新后比 epoch 0 差，也保留并测试更新结果，同时记录 epoch 0 基线指标。部署推荐与实验结果分开，避免把回退隐藏在新方法名下。E05 不额外训练、不扫描插值系数，固定 alpha=0.5，插值后重新从 TRAIN 提取支持库并单独校准。

## 安装与运行

将本次 `Openset_sweep_full_20261003.tar.gz` 放在 `/home/ubuntu/hdd/data/qz/`，执行：

```bash
cd /home/ubuntu/hdd/data/qz/
tar -xzf Openset_sweep_full_20261003.tar.gz
conda activate ProTeCt
cd /home/ubuntu/hdd/data/qz/Openset/
sha256sum -c SUPPORT_SWEEP_FILES.sha256

SOURCE_RUN="runs/taxosafe_new/reference/trial_1_retrain_20261003_185220"
SWEEP_RUN="runs/taxosafe_new/reference_sweep/trial_1_$(date +%Y%m%d_%H%M%S)"
bash tools/run_taxosafe_sweep.sh \
  --config configs/Zooplankton_Taxonomic_Tree/TaxoSafe_reference_sweep.yml \
  --reference-run-dir "$SOURCE_RUN" \
  --run-dir "$SWEEP_RUN" \
  --device cuda
```

压缩包顶层是 `Openset/`。覆盖源码不会删除原有 `runs/`、图像、数据清单、CLIP 缓存和模型文件。包内不含训练权重、真实图像或旧运行。不要在有运行任务时覆盖源码；收据会检查运行中代码是否变化。

入口默认执行完整实验与 TEST，无需追加 `--evaluate-test`。先依次完成各组训练/校准，再冻结仅由 DEV 决定的推荐，最后依次执行各组 TEST。这样既输出所有失败实验，又避免利用先测出的 TEST 修改后续组规则。

`targets_passed=false` 是研究指标未达标，**不会阻断测试**，也不会自动换成基线 router。有效的 best-effort 阈值照常冻结并评估。文件缺失、哈希不符、NaN、程序异常则属于执行失败，不能伪造一个有效模型或阈值；该组记录错误及无法测试的原因，其他独立组继续，依赖组明确标为阻塞。

## 读取结果

每组在 `arms/<组名>/` 下保存自己的训练、校准和测试产物；训练组保存 `training/best.pth` 与配套 `training/support.pth`。`calibration/` 和 `test/` 中保留逐图分数/预测、四项指标、是否通过、逐物种表及产物绑定。

套件汇总包含 CSV/JSON 对比、DEV 推荐与 TEST 探索性排名、相对 E00 的 known 正确样本损失/恢复、near/extra 各物种的变化。优先看校准和测试是否实际完成，再看指标，不能把执行错误当成零准确率，也不能把 `targets_passed=false` 当成未运行。

| 文件 | 阅读用途 |
|---|---|
| `comparison_all.csv` | 六组运行状态、梯度步数、DEV/TEST 指标及失败原因 |
| `comparison_development.csv` / `comparison_test.csv` | 完成评估的组按固定规则排序 |
| `dev_selection.json` | 第一组 TEST 之前冻结的 DEV 推荐 |
| `summary.json` | 完整对比、探索性最优组、未通过项和运行失败 |
| `per_species_comparison.csv` | 每个已知/未知物种相对基线的正确数损失、恢复与净变化 |
| `paired_development.json` / `paired_test.json` | 正确性变化的图片路径、内容哈希及前后输出 |
| `logs/<组名>/` | 各阶段训练日志和错误信息 |

训练过程中的详细输出写入日志；主终端显示各阶段开始和完成。如需实时观察当前训练，可在另一终端运行 `tail -f "$SWEEP_RUN/logs/E01_heads/training.stdout.log"`（先将变量设置为本次实际路径）。

四项目标保持为 known >90%、near 正确父类 ≥85%、extra 根拒识 >90%、开放叶精确率 >90%。保守推荐还要求 known 正确数不低于 E00。如果没有合格候选，推荐保留 E00，同时列出探索性最优候选及其未通过项。TEST 排名用于本次实验比较，不反过来更改已经冻结的 DEV 推荐；已有 TEST 多次被研究者查看，不能称新的独立确认集。

每个结果按图片内容哈希计量；正式 TEST 为 926 张唯一图像，清单 927 行。近域输出根仍为错误，远域输出父仍为错误，叶精确率分母包含所有未知误接受。逐物种数据和配对变化应一起看，不以总体平均掩盖少数类损伤。

运行结束自动输出文本 review 压缩包，包含所有组的诊断与错误记录，便于回传分析。该包不含 checkpoint，不是权重备份。请保留源 reference 和各组完整运行目录。

## 复现与验证边界

源运行和每组输出必须分离且不能互相嵌套。重新实验使用新目录；`--resume` 只复用代码、配置、源绑定及产物都匹配的完整阶段，不覆盖或删除部分阶段。仍沿用已经成功的 `ProTeCt` 环境，不为本次实验盲目升级 PyTorch/CUDA。

本次开发环境有图像和历史文本结果，但没有源 baseline 二进制权重，也没有 CUDA。因此发布记录会分别列出 CPU 软件测试和真实微调执行状态；不会把模拟测试或历史 reference 数据填成六组新实验结果。真实准确率及最优组须由上述服务器流程生成。先运行固定六组的单种子筛选；多种子复核与独立新测试集仍是后续验证工作。

研究依据和问题分析见 [TAXOSAFE_SWEEP_RESEARCH.md](TAXOSAFE_SWEEP_RESEARCH.md)。
