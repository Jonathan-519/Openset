# TaxoSafe v11：Depth-Conditioned Boundary Synthesis（DCBS）

基于 `taxosafe-v10-perf` 的 `4b9b44704e02c3b50f0f34808903094fab8f3c7a`，实现 known → 叶节点、near → 正确父节点、extra → 根节点的联合训练与分阶段评估。全部代码通过新的 v11 入口运行，原 v10 脚本、配置、图片、清单和历史结果保持原样。

这是待真实 GPU 实验验证的方法实现。CPU 测试验证软件契约，不代表真实浮游动物识别性能，也不证明优于 v10 或达到 SOTA。没有把官方 ProHOC 的复现结果或公开 benchmark 成绩作为本实现的结果。

## 1. 先处理已发现的图片重复

原清单的路径不同不等于图片内容不同。对三个 known 集合进行 SHA256 身份审计后发现：

| 重复位置 | 重复组数 | 处理方式 |
| --- | ---: | --- |
| train 内部 | 7 | 仅新训练清单保留首次出现 |
| train 与 val_known | 3 | 从新训练清单移除 |
| train 与 test_known | 8 | 从新训练清单移除 |
| val_known 与 test_known | 1 | 从新验证清单移除 |
| test_known 内部 | 1 | 保留锁定清单全部行，主指标按唯一图片计算 |

新目录 `prepro/data/Zooplankton_TT_v11_dcbs/` 的训练清单为 **1610** 行，验证清单为 **219** 行。原测试清单仍为 **451** 行，对应 **450** 张唯一图片。逐项移除原因见 `known_deduplication.json`。没有删除或覆盖原图片、原清单。

`Calanopia_thompsoni`（leaf 5）唯一的原验证图片与测试图片同内容，因此新验证集覆盖 22 个 known 物种；训练集仍覆盖全部 23 个。这种验证缺口会记录在审计里，校准的稀疏分支使用父级/全局统计，不能把它当成该物种已通过独立验证。

**边界说明：**准备清单时读取了 known 测试图片的文件字节，仅用于身份去重；没有解码、提取特征或调参，也没有读取 unknown 测试图片。训练/校准阶段不加载测试清单或测试特征。`test_used_for_fitting=false` 指模型及阈值拟合；身份准备另有 `test_known_used_only_for_identity_audit=true` 明确披露。

历史 v10 成绩使用了原清单，不能直接当成无泄漏、同协议的公平基线。正式比较需要按同一干净 known 协议重训基线，并明确区分 v10 的真实未知梯度监督与 v11 的合成监督设置。

## 2. 在现有 Linux 项目里安装

如果 `/home/ubuntu/hdd/data/qz/Openset` 没有 `.git`，使用下面的增量安装器，无需 `git pull`：

```bash
cd /home/ubuntu/hdd/data/qz/Openset
curl -fL https://raw.githubusercontent.com/Jonathan-519/Openset/taxosafe-v11-dcbs/tools/install_taxosafe_dcbs_v11.py -o /tmp/install_taxosafe_dcbs_v11.py
python /tmp/install_taxosafe_dcbs_v11.py --project "$PWD" --ref taxosafe-v11-dcbs
sha256sum -c DCBS_V11_FILES.sha256
```

安装器把分支解析成一个固定 commit，下载清单中的所有文件、检查 SHA256 后再写入；同名旧文件保存到 `dcbs_v11_backups/时间戳/`。不会下载图片、历史训练结果或权重。需要精确复现时，把 `--ref` 替换成完整 commit SHA。

如果是正常 Git checkout，可直接切换到 `taxosafe-v11-dcbs`。沿用能够运行原项目的 ProTeCt/PyTorch CUDA 环境；没有新增机器学习框架依赖。MaPLe 仍使用原 CLIP 初始化，v11 的 prompt 和新增头重新训练；禁止加载已经用真实未知/OE 训练过的 v10 checkpoint 并将其称为纯合成训练。

## 3. 校验、训练、冻结校准、最后测试

先在项目根目录执行：

```bash
python -m unittest tests.test_taxosafe_dcbs tests.test_taxosafe_dcbs_pipeline tests.test_taxosafe_dcbs_delivery tests.test_taxosafe_v10_protocol tests.test_taxosafe_suite -q
python prepro/build_taxosafe_v11_known_splits.py
python train_taxosafe_v11.py --preflight
python calibrate_taxosafe_v11.py --preflight
```

准备脚本可重复运行：内容相同则通过；数据改变造成结果不同时拒绝覆盖，需要先检查原因。训练预检只读取 train/val_known；校准预检只读取三个 development 验证集合。预检不需要 CUDA 或加载 CLIP。正常训练与评估需要 CUDA。

可先跑独立调试目录：

```bash
python -u train_taxosafe_v11.py --trial 1 --seed 1 --variant main --debug
```

调试只跑 2 个 epoch、每个 epoch 2 个训练 batch；完整训练参考遍历和 known 验证仍执行，所以不是即时返回。输出放在 `trial_1_debug`，其 checkpoint 禁止用于校准和锁定测试。

正式顺序如下，每条成功结束后再执行下一条：

```bash
python -u train_taxosafe_v11.py --trial 1 --seed 1 --variant main
python -u calibrate_taxosafe_v11.py --trial 1 --seed 1 --variant main
python -u test_taxosafe_v11.py --trial 1 --seed 1 --variant main
python tools/pack_taxosafe_dcbs_review.py --run-dir runs/taxosafe_v11_dcbs/main/trial_1
```

默认配置：`configs/Zooplankton_Taxonomic_Tree/TaxoSafe_dcbs_v11.yml`。训练 120 个 epoch 上限，warmup 5，合成损失渐增 5，known 验证早停 patience 25。只有 warmup 后至少经历 5 个有效合成 epoch 的 checkpoint 才可入选；启用 near/extra 时分别检查实际有合格样本，防止选中尚未学习拒识的模型。checkpoint 按 known 验证叶分类准确率优先、root CE 次优选择，不用 unknown/test 成绩选 checkpoint。

看到 `training/completed.json` 和 `Training finished` 才表示训练阶段完成。每个 epoch 的合成候选数、保留数、逐父节点覆盖、loss、checkpoint eligibility 都写入 `training/train.jsonl`。空候选不会被强制伪装成 near 或 extra；若始终没有有效合成监督，训练结束会报错，不生成可校准的完成记录。

每个 run 有单写者锁；已有阶段目录拒绝覆盖。中断后本版不支持无损续训，应为重跑选择新目录，三个入口使用一致参数：

```bash
python -u train_taxosafe_v11.py --trial 1 --seed 1 --variant main --run-dir runs/taxosafe_v11_dcbs/main/trial_1_retry
python -u calibrate_taxosafe_v11.py --trial 1 --seed 1 --variant main --run-dir runs/taxosafe_v11_dcbs/main/trial_1_retry
python -u test_taxosafe_v11.py --trial 1 --seed 1 --variant main --run-dir runs/taxosafe_v11_dcbs/main/trial_1_retry
```

不要在同一目录同时启动多个进程。查看正在运行的入口可用 `pgrep -af 'train_taxosafe_v11|calibrate_taxosafe_v11|test_taxosafe_v11'`。多种子实验使用独立 trial/seed，例如 `--trial 2 --seed 2`、`--trial 3 --seed 3`。报告均值和离散程度，不凭某次锁定测试成绩反复调参。

## 4. 模型与数据协议

| 组成 | v11 行为 |
| --- | --- |
| 真实梯度训练 | 只使用干净的 known TRAIN，冻结 CLIP 主干，仅优化 MaPLe prompts/VPT 与新增头 |
| 深度监督 | known=(parent, child)，near=(parent, STOP)，extra=(root STOP) |
| 双投影 | taxonomy 投影保留父分类信息；fine 投影学习父内区别与局部 STOP |
| 层级对抗 HA | 对 fine 投影反向梯度；输入 detach，禁止反向擦除共享主干及 taxonomy 投影的信息 |
| sibling margin | 仅比较同父兄弟的余弦分数；单子节点分支跳过 margin |
| 训练支持域 | 每个 epoch 对完整 TRAIN 做确定性遍历；叶均值、等叶权重父均值及收缩余弦半径 |
| 合成 near | 同父异叶混合，必须位于全部叶支持域外、目标父支持域内 |
| 合成 extra | 跨父混合/外推，必须位于全部父与叶支持域外 |
| 合成空间 | 与真实输入一致的归一化 backbone 特征空间，进入投影之前；合成样本 detach |
| 难样本筛选 | 合格候选中优先取旧叶分类器置信度高的样本，不改变其深度标签条件 |
| 推理 | 每张图片编码一次；保留原 MaPLe 叶分类器，头提供 root/local STOP 证据 |
| 路径一致性 | 全局叶预测的父节点与 root 预测冲突时禁止接受叶，回退父节点 |

合成池每个 epoch 生成一次、每个 batch 随机取子集，候选采样按类别批处理。仍有每个 epoch 一次完整参考遍历的代价；真实 GPU 时间、显存和精度未在当前 CPU 环境测量。

原 `train_intra`、`oe_train` 不进入 v11 的任何梯度更新，也不偷偷并入校准。保留原 v10 的 `gt_val_intra_v10.txt`（69 张）、`gt_val_extra_v10.txt`（88 张）作为 development 校准，test 仍使用原锁定清单。

## 5. 校准和冻结机制

root 使用已知父与 root STOP 的 log-mean-exp 比值及 taxonomy 支持相似度；local 使用子节点与 local STOP 的 log-mean-exp 比值、fine 支持相似度及兄弟 margin。log-mean-exp 消除分支子节点数对 log-sum-exp 的固定偏置。单子节点不伪造 margin。常量证据通道被丢弃，全部通道退化则拒绝校准。

分数由 **val_known** 的平滑经验 CDF 归一化，再融合有效证据。local CDF 和阈值按分支样本数向全局收缩；缺 near 的分支使用 known 分位数。root 在 known 保留率 ≥0.97、near 保留率 ≥0.95 的约束下优先优化最差 development extra 来源的拒识率。

source leave-one-out 把被留出的 extra 来源排除于该折阈值候选及目标函数，输出其拒识率。评分函数固定且由 known 拟合；这是一项来源转移诊断。标量阈值的单调性和保留率约束可能使多折阈值相同，不能把 LOO 包装成必然改善泛化的机制。

local known 保留率目标 ≥0.90；**最终收缩后的** known 端到端准确率必须 ≥`max(0.80, closed_accuracy - 0.08)`。先检查 root/路径决定是否使约束可行，再选择和收缩 local 阈值并重新检查。不能满足则保存 `calibration/failed.json` 并停止，不静默降低 floor，也不运行测试来补调阈值。这些都是有限 development 样本上的经验约束，不是统计泛化保证。

完成后 `router.json` 记录阈值、CDF、逐分支样本量/收缩权重、LOO、known guard。配置、实现代码、taxonomy、准备审计、checkpoint、支持原型和 router 都由 SHA256 绑定；改动后必须用新的 run。测试只加载冻结产物和三个测试集合；即使删除 train/development 清单仍可运行（由端到端测试覆盖）。

## 6. 消融与报告

| `--variant` | 改动 |
| --- | --- |
| `main` | 双投影、near/extra 合成、HA、margin、分支收缩 |
| `lite` | 共享投影、关闭 HA 和 margin，保留深度合成与冻结校准 |
| `heads_only` | 关闭两类合成；用于量化合成监督贡献 |
| `near_only` | 关闭 extra 合成 |
| `no_ha` | 关闭层级对抗 |
| `no_pooling` | 关闭分支阈值部分收缩，使用统一 local 阈值；CDF 归一化保持一致 |

各消融必须从头训练，不能交换其他 variant 的 checkpoint/router。主版本和 lite 可先各跑一个种子排查流程，再按预先确定的协议做多种子比较。

默认 run 为 `runs/taxosafe_v11_dcbs/main/trial_1/`：

- `training/`：配置、输入审计、逐 epoch 记录、best.pth、TRAIN 支持原型、完成记录。
- `calibration/`：development 原始证据、逐行预测、指标、冻结 router、完成记录；不可行时有失败原因。
- `test/predictions.jsonl`：原清单全部行，包含 `evaluation_weight`；重复字节首次为 1，其余为 0。
- `test/metrics.json`：按唯一图片计算的主指标，包括逐 known 物种、逐 near 物种、逐 extra 来源及区间。
- `test/metrics_all_rows.json`：按原清单全部行计算的辅助指标，方便定位历史统计口径差异。
- `test/gates.json`：预设验收目标及通过情况；不会据此更新模型/阈值。
- `tools/pack_taxosafe_dcbs_review.py`：打包清单、审计、日志、router 和结果，不包含大权重或图片，拒绝覆盖已有包。

验收目标：near 正确父回退 ≥0.70、extra 根拒识 ≥0.75、extra 错收为 known ≤0.075、接受叶精确率 ≥0.85、known 闭集准确率 ≥0.90、known 端到端准确率 ≥0.80。这些是目标，不是已经取得的结果。注意 `Oithona_similis` 是 known，`Oithona_plumifera` 是 unknown，报告不能混用名称。

## 7. 本次验证范围

39 项 CPU 测试：21 项模型/支持域/校准/协议测试、2 项端到端/去重测试、3 项安装/打包测试、13 项原有 v10/TaxoSafe 回归测试。端到端测试替换编码器与图像加载，真实执行 v11 训练、选择、保存、校准和冻结测试，不下载 CLIP。

已对仓库真实文件执行 known 去重、训练与 development 校准预检。没有在当前环境完成真实 CUDA 训练、未知测试推理或性能对比；这些必须通过上述训练流程取得，之后再判断方法是否改善 near/extra 同时保持 known。
